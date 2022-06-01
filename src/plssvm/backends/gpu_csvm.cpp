#include "plssvm/backends/gpu_csvm.hpp"

#include "plssvm/constants.hpp"               // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm.hpp"                    // plssvm::csvm
#include "plssvm/detail/execution_range.hpp"  // plssvm::detail::execution_range
#include "plssvm/detail/operators.hpp"        // various operator overloads for std::vector and scalars
#include "plssvm/exceptions/exceptions.hpp"   // plssvm::exception
#include "plssvm/parameter.hpp"               // plssvm::parameter

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    // used for explicitly instantiating the CUDA backend
    #include "plssvm/backends/CUDA/detail/device_ptr.cuh"
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    // used for explicitly instantiating the HIP backend
    #include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    // used for explicitly instantiating the OpenCL backend
    #include "plssvm/backends/OpenCL/detail/command_queue.hpp"
    #include "plssvm/backends/OpenCL/detail/device_ptr.hpp"
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // used for explicitly instantiating the SYCL backend
    #include "sycl/sycl.hpp"
#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    #include "plssvm/backends/autogenerated/DPCPP/detail/constants.hpp"
    #include "plssvm/backends/autogenerated/DPCPP/detail/device_ptr.hpp"
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    #include "plssvm/backends/autogenerated/hipSYCL/detail/constants.hpp"
    #include "plssvm/backends/autogenerated/hipSYCL/detail/device_ptr.hpp"
#endif
#endif

#include "fmt/chrono.h"  // directly print std::chrono literals with fmt
#include "fmt/core.h"    // fmt::print

#include <algorithm>  // std::all_of, std::min, std::max
#include <chrono>     // std::chrono
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

namespace plssvm::detail {


gpu_csvm::gpu_csvm(const parameter &params) :
    csvm::csvm{ params } {}

auto gpu_csvm::predict(const std::vector<std::vector<real_type>> &points) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor

    // return empty vector if there are no points to predict
    if (points.empty()) {
        return std::vector<real_type>{};
    }

    // sanity checks
    if (!std::all_of(points.begin(), points.end(), [&](const std::vector<real_type> &point) { return point.size() == points.front().size(); })) {
        throw exception{ "All points in the prediction point vector must have the same number of features!" };
    } else if (points.front().size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per predict point ({})!", data_ptr_->front().size(), points.front().size()) };
    } else if (alpha_ptr_ == nullptr) {
        throw exception{ "No alphas provided for prediction!" };
    }

    PLSSVM_ASSERT(data_ptr_->size() == alpha_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());  // exception in constructor

    // check if data already resides on the first device
    if (data_d_[0].empty()) {
        setup_data_on_device();
    }

    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> out(points.size());

    if (kernel_ == kernel_type::linear) {
        // use faster methode in case of the linear kernel function
        if (w_.empty()) {
            update_w();
        }
        #pragma omp parallel for
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < points.size(); ++i) {
            out[i] = transposed<real_type>{ w_ } * points[i] + bias_;
        }
    } else {
        // create result vector on the device
        device_ptr_type out_d{ points.size() + boundary_size_, devices_[0] };
        out_d.memset(0);

        // transform prediction data
        const std::vector<real_type> transformed_data = csvm::transform_data(points, boundary_size_, points.size());
        device_ptr_type point_d{ points[0].size() * (points.size() + boundary_size_), devices_[0] };
        point_d.memset(0);
        point_d.memcpy_to_device(transformed_data, 0, transformed_data.size());

        // create the weight vector on the device and copy data
        device_ptr_type alpha_d{ num_data_points_ + THREAD_BLOCK_SIZE, devices_[0] };
        alpha_d.memset(0);
        alpha_d.memcpy_to_device(*alpha_ptr_.get(), 0, num_data_points_);

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_data_points_) / static_cast<real_type>(THREAD_BLOCK_SIZE))),
                                              static_cast<std::size_t>(std::ceil(static_cast<real_type>(points.size()) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_data_points_), std::min<std::size_t>(THREAD_BLOCK_SIZE, points.size()) });

        // perform prediction on the first device
        run_predict_kernel(range, out_d, alpha_d, point_d, points.size());

        out_d.memcpy_to_host(out, 0, points.size());

        // add bias_ to all predictions
        out += bias_;
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Predicted {} data points in {}.\n", points.size(), std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    return out;
}


void gpu_csvm::setup_data_on_device() {
    // set values of member variables
    dept_ = num_data_points_ - 1;
    boundary_size_ = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    num_rows_ = dept_ + boundary_size_;
    num_cols_ = num_features_;
    feature_ranges_.reserve(devices_.size() + 1);
    for (typename std::vector<queue_type>::size_type device = 0; device <= devices_.size(); ++device) {
        feature_ranges_.push_back(device * num_cols_ / devices_.size());
    }

    // transform 2D to 1D data
    const std::vector<real_type> transformed_data = csvm::transform_data(*data_ptr_, boundary_size_, dept_);

    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::size_t num_features = feature_ranges_[device + 1] - feature_ranges_[device];

        const detail::execution_range range_r ({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features + boundary_size_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_features + boundary_size_) });

        // initialize data_last on device
        data_last_d_[device] = device_ptr_type{ num_features + boundary_size_, devices_[device] };
        data_last_d_[device].memset(0);
        data_last_d_[device].memcpy_to_device(data_ptr_->back().data() + feature_ranges_[device], 0, num_features);
        
        data_last_d_f_[device] = device_ptr_type_float{ num_features + boundary_size_, devices_[device] };
        data_last_d_f_[device].memset(0);
        run_transformation_kernel_df(0, range_r, data_last_d_f_[0], data_last_d_[0]);


        const std::size_t device_data_size = num_features * (dept_ + boundary_size_);
        data_d_[device] = device_ptr_type{ device_data_size, devices_[device] };
        data_d_[device].memcpy_to_device(transformed_data.data() + feature_ranges_[device] * (dept_ + boundary_size_), 0, device_data_size);

        const detail::execution_range range_full ({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features * (dept_ + boundary_size_)) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, dept_) });

        data_d_f_[device] = device_ptr_type_float{ device_data_size, devices_[device] };
        run_transformation_kernel_df(device, range_full, data_d_f_[0], data_d_[0]);
    }
}

auto gpu_csvm::generate_q() -> std::vector<real_type> {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<device_ptr_type> q_d(devices_.size());

    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        q_d[device].memset(0);

        // feature splitting on multiple devices
        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, dept_) });

        run_q_kernel(device, range, q_d[device], feature_ranges_[device + 1] - feature_ranges_[device]);
    }

    std::vector<real_type> q(dept_);
    device_reduction(q_d, q);
    return q;
}

auto gpu_csvm::solver_CG(const std::vector<real_type> &b, const std::size_t imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<real_type> x(dept_, 1.0);
    std::vector<float> x_f(dept_, 1.0);
    std::vector<device_ptr_type> x_d(devices_.size());
    std::vector<device_ptr_type_float> x_d_f(devices_.size());

    std::vector<real_type> r(dept_, 0.0);
    std::vector<float> r_f(dept_, 0.0); //Debugging only
    std::vector<device_ptr_type> r_d(devices_.size());
    std::vector<device_ptr_type_float> r_d_f(devices_.size());

    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        x_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept_);

        x_d_f[device] = device_ptr_type_float{ dept_ + boundary_size_, devices_[device] };
        x_d_f[device].memset(0);

        r_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        r_d[device].memset(0);

        r_d_f[device] = device_ptr_type_float{ dept_ + boundary_size_, devices_[device] };
        r_d_f[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept_);

    std::vector<device_ptr_type> q_d(devices_.size());
    std::vector<device_ptr_type_float> q_d_f(devices_.size());
    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept_);

        q_d_f[device] = device_ptr_type_float{ dept_ + boundary_size_, devices_[device] };
        q_d_f[device].memset(0);
        
        // r = Ax (r = b - Ax)
        run_device_kernel(device, q_d[device], r_d[device], x_d[device], -1);
    }
    device_reduction(r_d, r);

    // delta = r.T * r
    real_type delta = transposed<double>{ r } * r;
    const real_type delta0 = delta;

    std::vector<real_type> Ad(dept_);
    std::vector<real_type> Ad_test(dept_);
    std::vector<float> Ad_f(dept_);

    std::vector<device_ptr_type> Ad_d(devices_.size());
    std::vector<device_ptr_type> Ad_d_test(devices_.size());
    std::vector<device_ptr_type_float> Ad_d_f(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        Ad_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        Ad_d_test[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        Ad_d_f[device] = device_ptr_type_float{ dept_ + boundary_size_, devices_[device] };
    }

    std::vector<real_type> d(r);

    // conversions
    const detail::execution_range range_r ({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, dept_) });

    // Test section:
    /*
    real_type r_orig = transposed<double>{ r } * r;
    run_transformation_kernel_df(0, range_r, r_d_f[0], r_d[0]);
    device_reduction_f(r_d_f, r_f);
    real_type r_float = transposed<float>{ r_f } *r_f;
    run_transformation_kernel_fd(0, range_r, r_d[0], r_d_f[0]);
    device_reduction(r_d, r);
    real_type r_transformed = transposed<double>{ r } *r;
    fmt::print("orig: {}  float: {} transformed: {} \n", r_orig, r_float, r_transformed);
    */

    run_transformation_kernel_df(0, range_r, r_d_f[0], r_d[0]); // TODO: Add for loop for each device
    run_transformation_kernel_df(0, range_r, q_d_f[0], q_d[0]);
    run_transformation_kernel_df(0, range_r, x_d_f[0], x_d[0]);

    // std::vector<float> d_f(dept_);
    
    for(typename std::vector<queue_type>::size_type cast_i = 0; cast_i < dept_; ++cast_i)
    {
        // d_f[cast_i] = static_cast<float>(d[cast_i]);
        // Ad_f[cast_i] = static_cast<float>(Ad[cast_i]);
        // x_f[cast_i] = static_cast<float>(x[cast_i]);
    }

    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        auto iteration_end_time = std::chrono::steady_clock::now();
        auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        fmt::print("Done in {}.\n", iteration_duration);
        average_iteration_time += iteration_duration;
    };

    std::size_t run = 0;
    for (; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}). ", run + 1, imax, delta, eps * eps * delta0);
        }
        iteration_start_time = std::chrono::steady_clock::now();

        // Ad = A * r (q = A * d)
        #pragma omp parallel for
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            Ad_d[device].memset(0);
            Ad_d_test[device].memset(0);
            Ad_d_f[device].memset(0);
            r_d[device].memset(0, dept_);
            r_d_f[device].memset(0, dept_);

            run_device_kernel_f(device, q_d_f[device], Ad_d_f[device], r_d_f[device], 1);
            run_device_kernel(device, q_d[device], Ad_d[device], r_d[device], 1);
            // run_transformation_kernel_fd(device, range_r, r_d[device], r_d_f[device]);
        }
        // update Ad (q)
        device_reduction_f(Ad_d_f, Ad_f);
        device_reduction(Ad_d, Ad);

        
        std::vector<real_type> testvec_d (dept_, 1);
        real_type r_orig = transposed<double>{ Ad } * testvec_d;
        //std::vector<float> testvec_f (dept_, 1);
        // real_type r_float = transposed<float>{ Ad_f } *testvec_f;
        for(typename std::vector<queue_type>::size_type cast_i = 0; cast_i < dept_; ++cast_i)
        {
            // d[cast_i] = static_cast<double>(d_f[cast_i]);
            // Ad[cast_i] = static_cast<double>(Ad_f[cast_i]);
            // x[cast_i] = static_cast<double>(x_f[cast_i]);
            Ad[cast_i] = Ad_test[cast_i];
        }        
        real_type r_transformed = transposed<double>{ Ad } * testvec_d;
        real_type error_sum = (r_orig > r_transformed) ? r_orig-r_transformed : r_transformed-r_orig;
        real_type rel_error = error_sum / r_orig;
        fmt::print("\n orig: {}  float: x transformed: {} RundungsfehlerSummeMatMul: {} relativerFehler {} \n", r_orig, r_transformed, error_sum, rel_error);
        



        /* for(typename std::vector<queue_type>::size_type cast_i = 0; cast_i < dept_; ++cast_i)
        {
            // d[cast_i] = static_cast<double>(d_f[cast_i]);
            Ad[cast_i] = static_cast<double>(Ad_f[cast_i]);
            // x[cast_i] = static_cast<double>(x_f[cast_i]);
        }   */     

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed<double>{ d } * Ad);

        // (x = x + alpha * d)
        x += alpha_cd * d;

        if (run % 4 == 3) {  //run % 50 == 49

            for(typename std::vector<queue_type>::size_type cast_i = 0; cast_i < dept_; ++cast_i) {
                x_f[cast_i] = static_cast<float>(x[cast_i]);
            }

            #pragma omp parallel for
            for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
                x_d_f[device].memcpy_to_device(x_f, 0, dept_);
                // x_d[device].memcpy_to_device(x, 0, dept_);
            }

            #pragma omp parallel for
            for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
                if(device == 0) {
                    // r = b
                    // r_d[0].memcpy_to_device(b, 0, dept_);
                    r_d[0].memcpy_to_device(b, 0, dept_);
                    run_transformation_kernel_df(0, range_r, r_d_f[0], r_d[0]);
                } else {
                    // set r to 0
                    r_d[device].memset(0);
                    // r_d_f[device].memset(0);
                }

                // r -= A * x
                // run_device_kernel(device, q_d[device], r_d[device], x_d[device], -1);
                run_device_kernel_f(device, q_d_f[device], r_d_f[device], x_d_f[device], -1);
            }
            run_transformation_kernel_fd(0, range_r, r_d[0], r_d_f[0]);
            device_reduction(r_d, r);
        } else {
            // r -= alpha_cd * Ad (r = r - alpha * q)
            r -= alpha_cd * Ad;
        }

        // (delta = r^T * r)
        const real_type delta_old = delta;
        delta = transposed<double>{ r } * r;
        // if we are exact enough stop CG iterations
        if (delta <= eps * eps * delta0) {
            if (print_info_) {
                output_iteration_duration();
            }
            break;
        }

        // (beta = delta_new / delta_old)
        const real_type beta = delta / delta_old;
        // d = beta * d + r
        d = beta * d + r;

        // r_d = d
        #pragma omp parallel for
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            r_d[device].memcpy_to_device(d, 0, dept_);
            run_transformation_kernel_df(device, range_r, r_d_f[0], r_d[0]);
        }

        if (print_info_) {
            output_iteration_duration();
        }
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}) and an average iteration time of {}.\n",
                   std::min(run + 1, imax),
                   delta,
                   eps * eps * delta0,
                   average_iteration_time / std::min(run + 1, imax));
    }

    return std::vector<real_type>(x.begin(), x.begin() + dept_);
}

void gpu_csvm::update_w() {
    w_.resize(num_features_);
    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        // feature splitting on multiple devices
        const std::size_t num_features = feature_ranges_[device + 1] - feature_ranges_[device];

        // create the w vector on the device
        device_ptr_type w_d = device_ptr_type{ num_features, devices_[device] };
        // create the weight vector on the device and copy data
        device_ptr_type alpha_d{ num_data_points_ + THREAD_BLOCK_SIZE, devices_[device] };
        alpha_d.memcpy_to_device(*alpha_ptr_.get(), 0, num_data_points_);

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_features) });

        // calculate the w vector on the device
        run_w_kernel(device, range, w_d, alpha_d, num_features);
        device_synchronize(devices_[device]);

        // copy back to host memory
        w_d.memcpy_to_host(w_.data() + feature_ranges_[device], 0, num_features);
    }
}

void gpu_csvm::run_device_kernel(const std::size_t device, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add) {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(boundary_size_)));
    const detail::execution_range range({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    run_svm_kernel(device, range, q_d, r_d, x_d, add, feature_ranges_[device + 1] - feature_ranges_[device]);
}

void gpu_csvm::run_device_kernel_f(const std::size_t device, const device_ptr_type_float &q_d, device_ptr_type_float &r_d, const device_ptr_type_float &x_d, const float add) {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(boundary_size_)));
    const detail::execution_range range({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    run_svm_kernel_f(device, range, q_d, r_d, x_d, add, feature_ranges_[device + 1] - feature_ranges_[device]);
}

void gpu_csvm::run_device_kernel_m(const std::size_t device, const device_ptr_type_float &q_d, device_ptr_type &r_d, const device_ptr_type_float &x_d, const float add) {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(boundary_size_)));
    const detail::execution_range range({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    run_svm_kernel_m(device, range, q_d, r_d, x_d, add, feature_ranges_[device + 1] - feature_ranges_[device]);
}

void gpu_csvm::device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer) {
    using namespace plssvm::operators;

    device_synchronize(devices_[0]);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (devices_.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (typename std::vector<queue_type>::size_type device = 1; device < devices_.size(); ++device) {
            device_synchronize(devices_[device]);
            buffer_d[device].memcpy_to_host(ret, 0, ret.size());

            buffer += ret;
        }

        #pragma omp parallel for
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
        }
    }
}

void gpu_csvm::device_reduction_f(std::vector<device_ptr_type_float> &buffer_d, std::vector<float> &buffer) {
    using namespace plssvm::operators;

    device_synchronize(devices_[0]);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (devices_.size() > 1) {
        std::vector<float> ret(buffer.size());
        for (typename std::vector<queue_type>::size_type device = 1; device < devices_.size(); ++device) {
            device_synchronize(devices_[device]);
            buffer_d[device].memcpy_to_host(ret, 0, ret.size());

            buffer += ret;
        }

        #pragma omp parallel for
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
        }
    }
}

/*
// explicitly instantiate template class depending on available backends
#if defined(PLSSVM_HAS_CUDA_BACKEND)
// template class gpu_csvm<float, ::plssvm::cuda::detail::device_ptr<float>, int>;
template class gpu_csvm<double, ::plssvm::cuda::detail::device_ptr<double>, int>;
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
// template class gpu_csvm<float, ::plssvm::hip::detail::device_ptr<float>, int>;
template class gpu_csvm<double, ::plssvm::hip::detail::device_ptr<double>, int>;
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
// template class gpu_csvm<float, ::plssvm::opencl::detail::device_ptr<float>, ::plssvm::opencl::detail::command_queue>;
template class gpu_csvm<double, ::plssvm::opencl::detail::device_ptr<double>, ::plssvm::opencl::detail::command_queue>;
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
// template class gpu_csvm<float, ::plssvm::dpcpp::detail::device_ptr<float>, std::unique_ptr<::plssvm::dpcpp::detail::sycl::queue>>;
template class gpu_csvm<double, ::plssvm::dpcpp::detail::device_ptr<double>, std::unique_ptr<::plssvm::dpcpp::detail::sycl::queue>>;
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
// template class gpu_csvm<float, ::plssvm::hipsycl::detail::device_ptr<float>, std::unique_ptr<::plssvm::hipsycl::detail::sycl::queue>>;
template class gpu_csvm<double, ::plssvm::hipsycl::detail::device_ptr<double>, std::unique_ptr<::plssvm::hipsycl::detail::sycl::queue>>;
#endif
#endif */

}  // namespace plssvm::detail
