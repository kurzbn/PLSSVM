/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/q_kernel.cuh"

#include "plssvm/constants.hpp"  // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type gamma) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (kernel_index_type i = 0; i < feature_range; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = temp * gamma;
}
// template __global__ void device_kernel_q_linear(float *, const float *, const float *, const kernel_index_type, const kernel_index_type);
template __global__ void device_kernel_q_linear(double *, const double *, const double *, const kernel_index_type, const kernel_index_type, const real_type gamma);

__global__ void device_kernel_q_linear_f(float *q_f, const float *data_d_f, const float *data_last_f, const kernel_index_type num_rows, const kernel_index_type feature_range, const float gamma_f) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp{ 0.0 };
    for (kernel_index_type i = 0; i < feature_range; ++i) {
        temp += data_d_f[i * num_rows + index] * data_last_f[i];
    }
    q_f[index] = temp * gamma_f;
}

template <typename real_type>
__global__ void device_kernel_q_poly(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = pow(gamma * temp + coef0, degree);
}
// template __global__ void device_kernel_q_poly(float *, const float *, const float *, const kernel_index_type, const kernel_index_type, const int, const float, const float);
template __global__ void device_kernel_q_poly(double *, const double *, const double *, const kernel_index_type, const kernel_index_type, const int, const double, const double);

__global__ void device_kernel_q_poly_f(float *q_f, const float *data_d_f, const float *data_last_f, const kernel_index_type num_rows, const kernel_index_type num_cols, const int degree, const float gamma_f, const float coef0_f){
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp{ 0.0 };
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += data_d_f[i * num_rows + index] * data_last_f[i];
    }
    q_f[index] = pow(gamma_f * temp + coef0_f, degree);
}

template <typename real_type>
__global__ void device_kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type gamma) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp{ 0.0 };
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += (data_d[i * num_rows + index] - data_last[i]) * (data_d[i * num_rows + index] - data_last[i]);
    }
    q[index] = exp(-gamma * temp);
}
// template __global__ void device_kernel_q_radial(float *, const float *, const float *, const kernel_index_type, const kernel_index_type, const float);
template __global__ void device_kernel_q_radial(double *, const double *, const double *, const kernel_index_type, const kernel_index_type, const double);

__global__ void device_kernel_q_radial_f(float *q_f, const float *data_d_f, const float *data_last_f, const kernel_index_type num_rows, const kernel_index_type num_cols, const float gamma_f){
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp{ 0.0 };
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += (data_d_f[i * num_rows + index] - data_last_f[i]) * (data_d_f[i * num_rows + index] - data_last_f[i]);
    }
    q_f[index] = exp(-gamma_f * temp);
}

}  // namespace plssvm::cuda
