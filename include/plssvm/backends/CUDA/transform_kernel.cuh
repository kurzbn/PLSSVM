#pragma once

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda {
/**
 * @brief Transforms a given vector from double to float.
 * @details Supports multi-GPU execution.
 * @param[in] in_d to be transformed vector in double
 * @param[out] out_d transformed vector in float
 * @param[in] size_t size of both vectors
 */
__global__ void device_kernel_cast_double_to_float(const double *in_d, float *out_d, int size_d);
/**
 * @brief Transforms a given vector from float to double.
 * @details Supports multi-GPU execution.
 * @param[in] in_d to be transformed vector in float
 * @param[out] out_d transformed vector in double
 * @param[in] size_t size of both vectors
 */
__global__ void device_kernel_cast_float_to_double(const float *in_d, double *out_d, int size_d);
}  // namespace plssvm::cuda