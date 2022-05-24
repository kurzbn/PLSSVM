/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief File containing a single std::string which is propagated with all OpenCL kernels during CMake configuration.
 */

#pragma once

#include <string>  // std::string

namespace plssvm::opencl::detail {

/// An std::string containing all OpenCL kernel sources. Created and configured during CMake configuration.
constexpr const char* raw_kernel_src_string = R"(
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines atomic functions for floating point types.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/**
 * @brief Implementation of an atomic add function for double-precision floating point types.
 * @param[in,out] addr the source value to add @p val to
 * @param[in] val the value to add to @p addr
 */
inline void __attribute__((overloadable)) atomicAdd(__global const double *addr, const double val) {
    union {
        ulong u64;
        double f64;
    } next, expected, current;
    current.f64 = *addr;
    do {
        expected.f64 = current.f64;
        next.f64 = expected.f64 + val;
        current.u64 = atom_cmpxchg((volatile __global ulong *) addr,
                                   expected.u64,
                                   next.u64);
    } while (current.u64 != expected.u64);
}

/**
 * @brief Implementation of an atomic add function for single-precision floating point types.
 * @param[in,out] addr the source value to add @p val to
 * @param[in] val the value to add to @p addr
 */
inline void __attribute__((overloadable)) atomicAdd(__global const float *addr, const float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *) addr,
                                     expected.u32,
                                     next.u32);
    } while (current.u32 != expected.u32);
}/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines CUDA functions for generating the `q` vector.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Calculates the `q` vector using the linear C-SVM kernel.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] num_rows the number of rows in the data matrix
 * @param[in] feature_range number of features used for the calculation
 */
__kernel void device_kernel_q_linear(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const kernel_index_type num_rows, const kernel_index_type feature_range) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    for (kernel_index_type i = 0; i < feature_range; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = temp;
}

/**
 * @brief Calculates the `q` vector using the polynomial C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] num_rows the number of rows in the data matrix
 * @param[in] num_cols the number of columns in the data matrix
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
__kernel void device_kernel_q_poly(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    for (int i = 0; i < num_cols; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = pown(gamma * temp + coef0, degree);
}

/**
 * @brief Calculates the `q` vector using the radial basis functions C-SVM kernel.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[out] q the calculated `q` vector
 * @param[in] data_d the one-dimensional data matrix
 * @param[in] data_last the last row in the data matrix
 * @param[in] num_rows the number of rows in the data matrix
 * @param[in] num_cols the number of columns in the data matrix
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
__kernel void device_kernel_q_radial(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type gamma) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    for (kernel_index_type i = 0; i < num_cols; ++i) {
        temp += (data_d[i * num_rows + index] - data_last[i]) * (data_d[i * num_rows + index] - data_last[i]);
    }
    q[index] = exp(-gamma * temp);
}/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the kernel functions for the C-SVM using the CUDA backend.
 */

//#include "detail/atomics.cl"  // atomicAdd -> included via string concatenation when building the device kernels

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] num_rows the number of columns in the data matrix
 * @param[in] feature_range  number of features used for the calculation on the device @p id
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] id the id of the current device
 */
__kernel void device_kernel_linear(__global const real_type *q, __global real_type *ret, __global const real_type *d, __global const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type add, const kernel_index_type id) {
    kernel_index_type i = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE;

    __local real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __local real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += get_local_id(0) * INTERNAL_BLOCK_SIZE;
        j += get_local_id(1) * INTERNAL_BLOCK_SIZE;
        // cache data
        for (kernel_index_type vec_index = 0; vec_index < feature_range * num_rows; vec_index += num_rows) {
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (get_local_id(1) == idx) {
                    data_intern_i[get_local_id(0)][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id % THREAD_BLOCK_SIZE;
                if (get_local_id(0) == idx_2) {
                    data_intern_j[get_local_id(1)][block_id] = data_d[block_id + vec_index + j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[get_local_id(1)][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[get_local_id(0)][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                real_type temp;
                if (id == 0) {
                    temp = (matr[x][y] + QA_cost - q[i + y] - q[j + x]) * add;
                } else {
                    temp = matr[x][y] * add;
                }
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    ret_jx += temp * d[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    if (id == 0) {
                        ret_jx += (temp + cost * add) * d[i + y];
                    } else {
                        ret_jx += temp * d[i + y];
                    }
                }
            }
            atomicAdd(&ret[j + x], ret_jx);
        }
    }
}

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] num_rows the number of columns in the data matrix
 * @param[in] num_cols the number of rows in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
__kernel void device_kernel_poly(__global const real_type *q, __global real_type *ret, __global const real_type *d, __global const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    kernel_index_type i = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE;

    __local real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __local real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += get_local_id(0) * INTERNAL_BLOCK_SIZE;
        j += get_local_id(1) * INTERNAL_BLOCK_SIZE;
        // cache data
        for (kernel_index_type vec_index = 0; vec_index < num_cols * num_rows; vec_index += num_rows) {
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (get_local_id(1) == idx) {
                    data_intern_i[get_local_id(0)][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id % THREAD_BLOCK_SIZE;
                if (get_local_id(0) == idx_2) {
                    data_intern_j[get_local_id(1)][block_id] = data_d[block_id + vec_index + j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[get_local_id(1)][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[get_local_id(0)][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                const real_type temp = (pown(gamma * matr[x][y] + coef0, degree) + QA_cost - q[i + y] - q[j + x]) * add;
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    ret_jx += temp * d[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    ret_jx += (temp + cost * add) * d[i + y];
                }
            }
            atomicAdd(&ret[j + x], ret_jx);
        }
    }
}

/**
 * @brief Calculates the C-SVM kernel using the radial basis function kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[out] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data_d the one-dimension data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] num_rows the number of columns in the data matrix
 * @param[in] num_cols the number of rows in the data matrix
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
__kernel void device_kernel_radial(__global const real_type *q, __global real_type *ret, __global const real_type *d, __global const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const real_type gamma) {
    kernel_index_type i = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE;

    __local real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __local real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += get_local_id(0) * INTERNAL_BLOCK_SIZE;
        j += get_local_id(1) * INTERNAL_BLOCK_SIZE;
        // cache data
        for (kernel_index_type vec_index = 0; vec_index < num_cols * num_rows; vec_index += num_rows) {
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (get_local_id(1) == idx) {
                    data_intern_i[get_local_id(0)][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id % THREAD_BLOCK_SIZE;
                if (get_local_id(0) == idx_2) {
                    data_intern_j[get_local_id(1)][block_id] = data_d[block_id + vec_index + j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[get_local_id(1)][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[get_local_id(0)][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += (data_i - data_j[k]) * (data_i - data_j[k]);
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                const real_type temp = (exp(-gamma * matr[x][y]) + QA_cost - q[i + y] - q[j + x]) * add;
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    ret_jx += temp * d[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    ret_jx += (temp + cost * add) * d[i + y];
                }
            }
            atomicAdd(&ret[j + x], ret_jx);
        }
    }
}/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the OpenCL backend.
 */

//#include "detail/atomics.cl"  // atomicAdd -> included via string concatenation when building the device kernels

/**
 * @brief Calculate the `w` vector to speed up the prediction of the labels for data points using the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam real_type the type of the data
 * @param[out] w_d the `w` vector to assemble
 * @param[in] data_d the one-dimension support vector matrix
 * @param[in] data_last_d the last row of the support vector matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] num_data_points the total number of support vectors
 * @param[in] num_features the number of features per support vector
 */
__kernel void device_kernel_w_linear(__global real_type *w_d, __global real_type *data_d, __global real_type *data_last_d, __global real_type *alpha_d, const kernel_index_type num_data_points, const kernel_index_type num_features) {
    const kernel_index_type index = get_global_id(0);
    real_type temp = 0.0;
    if (index < num_features) {
        for (kernel_index_type dat = 0; dat < num_data_points - 1; ++dat) {
            temp += alpha_d[dat] * data_d[dat + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * index];
        }
        temp += alpha_d[num_data_points - 1] * data_last_d[index];
        w_d[index] = temp;
    }
}

/**
 * @brief Predicts the labels for data points using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] out_d the calculated predictions
 * @param[in] data_d the one-dimension support vector matrix
 * @param[in] data_last_d the last row of the support vector matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] num_data_points the total number of support vectors
 * @param[in] points the data points to predict
 * @param[in] num_predict_points the total number of data points to predict
 * @param[in] num_features the number of features per support vector and point to predict
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 */
__kernel void device_kernel_predict_poly(__global real_type *out_d, __global const real_type *data_d, __global const real_type *data_last_d, __global const real_type *alpha_d, const kernel_index_type num_data_points, __global const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type data_point_index = get_global_id(0);
    const kernel_index_type predict_point_index = get_global_id(1);

    real_type temp = 0.0;
    if (predict_point_index < num_predict_points) {
        for (kernel_index_type feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points) {
                temp += data_last_d[feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            } else {
                temp += data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            }
        }

        temp = alpha_d[data_point_index] * pow(gamma * temp + coef0, degree);
        atomicAdd(&out_d[predict_point_index], temp);
    }
}

/**
 * @brief Predicts the labels for data points using the radial basis functions kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam real_type the type of the data
 * @param[in] out_d the calculated predictions
 * @param[in] data_d the one-dimension support vector matrix
 * @param[in] data_last_d the last row of the support vector matrix
 * @param[in] alpha_d the previously calculated weight for each data point
 * @param[in] num_data_points the total number of support vectors
 * @param[in] points the data points to predict
 * @param[in] num_predict_points the total number of data points to predict
 * @param[in] num_features the number of features per support vector and point to predict
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 */
__kernel void device_kernel_predict_radial(__global real_type *out_d, __global const real_type *data_d, __global const real_type *data_last_d, __global const real_type *alpha_d, const kernel_index_type num_data_points, __global const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const real_type gamma) {
    const kernel_index_type data_point_index = get_global_id(0);
    const kernel_index_type predict_point_index = get_global_id(1);

    real_type temp = 0.0;
    if (predict_point_index < num_predict_points) {
        for (kernel_index_type feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points) {
                temp += (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            } else {
                temp += (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            }
        }

        temp = alpha_d[data_point_index] * exp(-gamma * temp);
        atomicAdd(&out_d[predict_point_index], temp);
    }
}// /usr/local.nfs/sw/cuda/cuda-11.4.3/lib64/libOpenCL.so;
)";

}
