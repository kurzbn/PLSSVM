/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/svm_kernel.cuh"

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/constants.hpp"                     // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::cuda {

__global__ void device_kernel_linear_mixed(const float *q, real_type *ret, const float *d, const float *data_d, const float QA_cost, const float cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const float add, const kernel_index_type id) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    __shared__ real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNAL_BLOCK_SIZE;
        const kernel_index_type ji = j + threadIdx.x * INTERNAL_BLOCK_SIZE;
        j += threadIdx.y * INTERNAL_BLOCK_SIZE;
        // cache data
        for (kernel_index_type vec_index = 0; vec_index < feature_range * num_rows; vec_index += num_rows) {
            __syncthreads();
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id + INTERNAL_BLOCK_SIZE % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx_2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
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
                    const double double_tmp = static_cast<double>(temp * d[j + x]);
                    atomicAdd(&ret[i + y], double_tmp);
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
            const double ret_double = static_cast<double>(ret_jx);
            atomicAdd(&ret[j + x], ret_double);
        }
    }
}

__global__ void device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type add, const kernel_index_type id) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    const int warp_id = threadIdx.y / 2;
    const int smemPlusStride = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE + 2;

    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];
    // __shared__ alignas(alignof(double2)) real_type Is[4][THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE];
    // __shared__ alignas(alignof(double2)) real_type Js[4][THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE];
    // __shared__ alignas(alignof(double2)) real_type Mats[8][64];
    // stride privious: 8 * 64
    __shared__ alignas(alignof(double2)) real_type Is[4 * smemPlusStride];
    __shared__ alignas(alignof(double2)) real_type Js[4 * smemPlusStride];
    __shared__ alignas(alignof(double2)) real_type Is_2[4 * smemPlusStride];
    __shared__ alignas(alignof(double2)) real_type Js_2[4 * smemPlusStride];
    // extern __shared__ alignas(alignof(double2)) real_type Mats[INTERNAL_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * (256 + 2)]; 
    extern __shared__ alignas(alignof(double2)) unsigned char Mats_d[];
    real_type *Mats = reinterpret_cast<real_type *>(Mats_d);
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    // real_type data_j[INTERNAL_BLOCK_SIZE];
    if (threadIdx.x == 0) {
        init(&bar[0], blockDim.x * blockDim.y);
        init(&bar[1], blockDim.x * blockDim.y);
    }
    __syncthreads();



    if (i >= j) {
        // cache data
        if(threadIdx.y < 12){
            double2 *const I2s = reinterpret_cast<double2 *>(&Is[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
            double2 *const J2s = reinterpret_cast<double2 *>(&Js[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
            const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[i + threadIdx.x / 4 * feature_range + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
            const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[j + threadIdx.x / 4 * feature_range + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // j =  Block, feature_range --> welche der 4 Zeilen, vec Index iter, Rest wo im Block

            cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
            cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]);
        }

        for (kernel_index_type vec_index = 0; vec_index < feature_range * num_rows; vec_index += 8*num_rows) {

            // __syncthreads();


            bar[0].arrive_and_wait();

            if(threadIdx.y < 12){
                double2 *const I2s_2 = reinterpret_cast<double2 *>(&Is_2[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                double2 *const J2s_2 = reinterpret_cast<double2 *>(&Js_2[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                const double2 *const D_i2_2 = reinterpret_cast<const double2 *>(&data_d[i + (threadIdx.x / 4 + 4) * feature_range + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                const double2 *const D_j2_2 = reinterpret_cast<const double2 *>(&data_d[j + (threadIdx.x / 4 + 4) * feature_range + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // j =  Block, feature_range --> welche der 4 Zeilen, vec Index iter, Rest wo im Block

                cuda::memcpy_async(I2s_2, D_i2_2, sizeof(double2), bar[1]);
                cuda::memcpy_async(J2s_2, D_j2_2, sizeof(double2), bar[1]);
            }

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::col_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> c_frag; 

            // #pragma unroll INTERNAL_BLOCK_SIZE
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE / 2; ++l) {
                // double *shmem_j_ptr = (double*)&Js[0][0] + 8*l + (warp_id % 4) * 24; 
                double *shmem_j_ptr = (double*)&Js[0] + 8*l + (warp_id % 4) * 24;
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, smemPlusStride);
                
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    double *shmem_i_ptr = (double*)&Is[0] + 8*k + warp_id * 12;
                    nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, smemPlusStride);
                    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    // double *shmem_m_ptr = (double*)&Mats[0][8 * warp_id];
                    double *shmem_m_ptr = (double*)&Mats[8 * warp_id];
                    nvcuda::wmma::store_matrix_sync(shmem_m_ptr , c_frag, 8*8+2, nvcuda::wmma::mem_row_major);
                    // matr[k][2 * l] += Mats[threadIdx.x/4 + threadIdx.y % 2 * 4][threadIdx.x % 4 + warp_id*8];
                    // matr[k][2 * l + 1] += Mats[threadIdx.x/4 + (threadIdx.y % 2) * 4][threadIdx.x % 4 + 4 + warp_id*8];     
                    matr[k][2 * l] += Mats[(threadIdx.x/4 + threadIdx.y % 2 * 4) * 66 + threadIdx.x % 4 + warp_id*8];
                    matr[k][2 * l + 1] += Mats[(threadIdx.x/4 + (threadIdx.y % 2) * 4) * 66 + threadIdx.x % 4 + 4 + warp_id*8];                  
                }
            }

            bar[1].arrive_and_wait();

            if(threadIdx.y < 12 && vec_index + 8 < feature_range * num_rows){
                double2 *const I2s = reinterpret_cast<double2 *>(&Is[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                double2 *const J2s = reinterpret_cast<double2 *>(&Js[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[i + (threadIdx.x / 4 + 8) * feature_range + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[j + (threadIdx.x / 4 + 8) * feature_range + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // j =  Block, feature_range --> welche der 4 Zeilen, vec Index iter, Rest wo im Block

                cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
                cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]);
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE / 2; ++l) {
                // double *shmem_j_ptr = (double*)&Js[0][0] + 8*l + (warp_id % 4) * 24; 
                double *shmem_j_ptr = (double*)&Js_2[0] + 8*l + (warp_id % 4) * 24;
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, smemPlusStride);
                
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    double *shmem_i_ptr = (double*)&Is_2[0] + 8*k + warp_id * 12;
                    nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, smemPlusStride);
                    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    // double *shmem_m_ptr = (double*)&Mats[0][8 * warp_id];
                    double *shmem_m_ptr = (double*)&Mats[8 * warp_id];
                    nvcuda::wmma::store_matrix_sync(shmem_m_ptr , c_frag, 8*8+2, nvcuda::wmma::mem_row_major);
                    // matr[k][2 * l] += Mats[threadIdx.x/4 + threadIdx.y % 2 * 4][threadIdx.x % 4 + warp_id*8];
                    // matr[k][2 * l + 1] += Mats[threadIdx.x/4 + (threadIdx.y % 2) * 4][threadIdx.x % 4 + 4 + warp_id*8];     
                    matr[k][2 * l] += Mats[(threadIdx.x/4 + threadIdx.y % 2 * 4) * 66 + threadIdx.x % 4 + warp_id*8];
                    matr[k][2 * l + 1] += Mats[(threadIdx.x/4 + (threadIdx.y % 2) * 4) * 66 + threadIdx.x % 4 + 4 + warp_id*8];                  
                }
            }
        }
        j += (threadIdx.x % 4) + (warp_id % 4) * 24;
        i += (threadIdx.x / 4) + 4 * threadIdx.y % 2 + 48 * (threadIdx.y / 8);
        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type x = 0; x < 8 * INTERNAL_BLOCK_SIZE; x = x + 8) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type y = 0; y < 4 * INTERNAL_BLOCK_SIZE; y = y + 4) {
                real_type temp;
                if (id == 0) {
                    temp = (matr[x/8][y/4] + QA_cost - q[i + x] - q[j + y]) * add;
                } else {
                    temp = matr[x/8][y/4] * add;
                }
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[j + y], temp * d[i + x]);
                    ret_jx += temp * d[j + y];
                } else if (i + x == j + y) {
                    // diagonal
                    if (id == 0) {
                        ret_jx += (temp + cost * add) * d[j + y];
                    } else {
                        ret_jx += temp * d[j + y];
                    }
                }
            }
            atomicAdd(&ret[i + x], ret_jx);
        }
    }
}

template <typename real_type>
__global__ void device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type add, const real_type gamma, const kernel_index_type id) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    __shared__ real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNAL_BLOCK_SIZE;
        const kernel_index_type ji = j + threadIdx.x * INTERNAL_BLOCK_SIZE;
        j += threadIdx.y * INTERNAL_BLOCK_SIZE;
        // cache data
        for (kernel_index_type vec_index = 0; vec_index < feature_range * num_rows; vec_index += num_rows) {
            __syncthreads();
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id + INTERNAL_BLOCK_SIZE % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx_2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
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
                    temp = (matr[x][y] * gamma + QA_cost - q[i + y] - q[j + x]) * add;
                } else {
                    temp = matr[x][y] * gamma * add;
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

template __global__ void device_kernel_linear(const float *, float *, const float *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const float, const float, const kernel_index_type);
template __global__ void device_kernel_linear(const double *, double *, const double *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const double, const real_type, const kernel_index_type);

template <typename real_type>
__global__ void device_kernel_poly(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    __shared__ real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNAL_BLOCK_SIZE;
        const kernel_index_type ji = j + threadIdx.x * INTERNAL_BLOCK_SIZE;
        j += threadIdx.y * INTERNAL_BLOCK_SIZE;
        for (kernel_index_type vec_index = 0; vec_index < num_cols * num_rows; vec_index += num_rows) {
            __syncthreads();
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx_2 = block_id + INTERNAL_BLOCK_SIZE % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx_2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
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
                const real_type temp = (pow(gamma * matr[x][y] + coef0, degree) + QA_cost - q[i + y] - q[j + x]) * add;
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

template __global__ void device_kernel_poly(const float *, float *, const float *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const float, const int, const float, const float);
template __global__ void device_kernel_poly(const double *, double *, const double *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const double, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_radial(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_cols, const real_type add, const real_type gamma) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    __shared__ real_type data_intern_i[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type data_intern_j[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNAL_BLOCK_SIZE;
        const kernel_index_type ji = j + threadIdx.x * INTERNAL_BLOCK_SIZE;
        j += threadIdx.y * INTERNAL_BLOCK_SIZE;
        for (kernel_index_type vec_index = 0; vec_index < num_cols * num_rows; vec_index += num_rows) {
            __syncthreads();
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const kernel_index_type idx = block_id % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const kernel_index_type idx2 = block_id + INTERNAL_BLOCK_SIZE % THREAD_BLOCK_SIZE;
                if (threadIdx.y == idx2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
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
}
template __global__ void device_kernel_radial(const float *, float *, const float *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const float, const float);
template __global__ void device_kernel_radial(const double *, double *, const double *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const double, const double);

}  // namespace plssvm::cuda