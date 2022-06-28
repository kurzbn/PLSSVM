/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/csvm.hpp"

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::cuda_p::detail::device_ptr
#include "plssvm/backends/CUDA/detail/utility.cuh"     // plssvm::cuda_p::detail::device_synchronize, plssvm::detail::cuda_p::get_device_count, plssvm::detail::cuda_p::set_device, plssvm::detail::cuda_p::peek_at_last_error
#include "plssvm/backends/CUDA/exceptions.hpp"         // plssvm::cuda_p::backend_exception
#include "plssvm/backends/CUDA/predict_kernel.cuh"     // plssvm::cuda_p::kernel_w, plssvm::cuda_p::predict_points_poly, plssvm::cuda_p::predict_points_rbf
#include "plssvm/backends/CUDA/q_kernel.cuh"           // plssvm::cuda_p::device_kernel_q_linear, plssvm::cuda_p::device_kernel_q_poly, plssvm::cuda_p::device_kernel_q_radial
#include "plssvm/backends/CUDA/svm_kernel.cuh"         // plssvm::cuda_p::device_kernel_linear, plssvm::cuda_p::device_kernel_poly, plssvm::cuda_p::device_kernel_radial
#include "plssvm/backends/gpu_csvm.hpp"                // plssvm::detail::gpu_csvm
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"           // plssvm::detail::execution_range
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <exception>  // std::terminate
#include <numeric>    // std::iota
#include <utility>    // std::pair, std::make_pair
#include <vector>     // std::vector

namespace plssvm::cuda_p {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    base_type{ params } {
    // check if supported target platform has been selected
    if (target_ != target_platform::automatic && target_ != target_platform::gpu_nvidia) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the CUDA backend!", target_) };
    } else {
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
        throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
    }

    if (print_info_) {
        fmt::print("Using CUDA as backend.\n");
    }

    // get all available devices wrt the requested target platform
    devices_.resize(std::min<std::size_t>(detail::get_device_count(), num_features_));
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no CUDA devices could be found
    if (devices_.empty()) {
        throw backend_exception{ "CUDA backend selected but no CUDA devices were found!" };
    }

    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel_ == kernel_type::polynomial || kernel_ == kernel_type::rbf) {
        devices_.resize(1);
    }

    // GPU 4 ist belegt ...
    devices_.resize(1);

    // resize vectors accordingly
    data_d_.resize(devices_.size());
    data_last_d_.resize(devices_.size());

    if (print_info_) {
        // print found CUDA devices
        fmt::print("Found {} CUDA device(s):\n", devices_.size());
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, devices_[device]);
            fmt::print("  [{}, {}, {}.{}]\n", devices_[device], prop.name, prop.major, prop.minor);
        }
        fmt::print("\n");
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the CUDA devices have finished before destruction
        for (const queue_type &device : devices_) {
            detail::device_synchronize(device);
        }
    } catch (const plssvm::exception &e) {
        fmt::print("{}\n", e.what_with_loc());
        std::terminate();
    }
}

template <typename T>
void csvm<T>::device_synchronize(queue_type &queue) {
    detail::device_synchronize(queue);
}

std::pair<dim3, dim3> execution_range_to_native(const ::plssvm::detail::execution_range &range) {
    dim3 grid(range.grid[0], range.grid[1], range.grid[2]);
    dim3 block(range.block[0], range.block[1], range.block[2]);
    return std::make_pair(grid, block);
}

template <typename T>
void csvm<T>::run_q_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &q_d, const std::size_t num_features) {
    auto [grid, block] = execution_range_to_native(range);

    detail::set_device(device);
    switch (kernel_) {
        case kernel_type::linear:
            cuda_p::device_kernel_q_linear<<<grid, block>>>(q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_features);
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            cuda_p::device_kernel_q_poly<<<grid, block>>>(q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            cuda_p::device_kernel_q_radial<<<grid, block>>>(q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, gamma_);
            break;
    }
    detail::peek_at_last_error();
}

template <typename T>
void csvm<T>::run_svm_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const std::size_t num_features) {
    auto [grid, block] = execution_range_to_native(range);

    detail::set_device(device);
    switch (kernel_) {
        case kernel_type::linear:
            // PLSSVM_CUDA_ERROR_CHECK(cudaFuncSetAttribute(cuda_p::device_kernel_linear<double>, cudaFuncAttributeMaxDynamicSharedMemorySize, INTERNAL_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * (256 + 2) * 8 ));
            cuda_p::device_kernel_linear<<<grid, block, INTERNAL_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * (256 + 2)>>>(q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_features, add, device);
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            cuda_p::device_kernel_poly<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            cuda_p::device_kernel_radial<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, gamma_);
            break;
    }
    detail::peek_at_last_error();
}

template <typename T>
void csvm<T>::run_w_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, const std::size_t num_features) {
    auto [grid, block] = execution_range_to_native(range);

    detail::set_device(device);
    cuda_p::device_kernel_w_linear<<<grid, block>>>(w_d.get(), data_d_[device].get(), data_last_d_[device].get(), alpha_d.get(), num_data_points_, num_features);
    detail::peek_at_last_error();
}

template <typename T>
void csvm<T>::run_predict_kernel(const ::plssvm::detail::execution_range &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const std::size_t num_predict_points) {
    auto [grid, block] = execution_range_to_native(range);

    detail::set_device(0);
    switch (kernel_) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            cuda_p::device_kernel_predict_poly<<<grid, block>>>(out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            cuda_p::device_kernel_predict_radial<<<grid, block>>>(out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, gamma_);
            break;
    }
    detail::peek_at_last_error();
}

template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::cuda
