/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/svm_kernel.cuh"

#include <cuda/barrier>                             // cuda::barrier
#include <mma.h>                                    // wmma related functions to use tensor cores

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/constants.hpp"                     // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

// Disables 'pipeline_shared_state' initialization warning.
#pragma diag_suppress static_var_with_dynamic_init

namespace plssvm::cuda {

__global__ void device_kernel_linear_td(const double *q, double *ret, const double *d, const double *data_d, const double QA_cost, const double cost, const kernel_index_type points, const kernel_index_type feature_range, const double add, const double gamma, const kernel_index_type id) {
    const kernel_index_type i = blockIdx.x * BLOCK_SIZE;
    const kernel_index_type j = blockIdx.y * BLOCK_SIZE;
    // TODO Check here or in setup for integer overflow - for all Kernels!
    // idea:
    // if(cast (points * feature_range) > int_max)  
    // split *data_d, adjust feature_range
    // const double *in[split] = ...
    // feature_range[split] = ...
    // loop BUILD_MATRIX (excluded fragment init and store phase) over *data_d[] and feature_range[]

    if (i >= j) { // if lower triangular matrix
        // helper variables for flow and copy
        const kernel_index_type id_1d = threadIdx.y * 32 + threadIdx.x;
        const kernel_index_type line = id_1d / (BLOCK_SIZE / 2);
        const kernel_index_type mem_num = id_1d % (BLOCK_SIZE / 2);
        const kernel_index_type transfer_line = id_1d / (BLOCK_SIZE / 4);
        const kernel_index_type tranfser_offset = id_1d % (BLOCK_SIZE / 4); 

        __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar;

        extern __shared__ double solution[];
        double *Is = (double *) &solution[0];
        double *Js = (double *) &solution[0] + 8 * BLOCK_OFF;
        double *Vjs = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF;
        double *Vis = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF + 1 * BLOCK_OFF;
        double *Qis = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF + 2 * BLOCK_OFF;
        double *Qjs = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF + 3 * BLOCK_OFF;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            init(&bar, THREADS_PER_BLOCK);
        }
        __syncthreads();

        // Build Matrix
        // ---------------------

        // load Data for first iteration
        if (threadIdx.y < 6) { // warp 0-5
            // 6 warps per 8 feature lines for I
            double2 *const I2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4]);
            const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[transfer_line * points + tranfser_offset * 4 + i]);
            ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(double2), bar);
        } else {  // warp 6-11
            // 6 warps per 8 feature lines for J
            double2 *const J2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4]);
            const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[(transfer_line - 8) * points + tranfser_offset * 4 + j]);
            ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(double2), bar);
        }

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> c_frag[ROLL_SIZE];
        for (kernel_index_type frags = 0; frags < ROLL_SIZE; ++frags) {
            // fill c_frags with zeros
            nvcuda::wmma::fill_fragment(c_frag[frags], 0.0f);
        }

        // loop over features
        for (kernel_index_type feature_it = 8; feature_it < feature_range; feature_it = feature_it + 8) {
            const kernel_index_type off_plus = (feature_it & 8) * BLOCK_OFF * 2;
            const kernel_index_type off_minus = abs(off_plus - 16 * BLOCK_OFF);

            // wait for new I- and J-data to be loaded into shared memory
            bar.arrive_and_wait();

            if (threadIdx.y < 6) {  // warp 0-5
                // 6 warps per 8 feature lines for I
                double2 *const I2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4 + off_plus]);
                const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[transfer_line * points + tranfser_offset * 4 + feature_it * points + i]);
                ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(double2), bar);
            } else{  // warp 6-11
                // 6 warps per 8 feature lines for J
                double2 *const J2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4 + off_plus]);
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[(transfer_line - 8) * points + tranfser_offset * 4 + feature_it * points + j]);
                ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(double2), bar);
            }

            // Do 2 iterations
            for (kernel_index_type  mem_roll = 0; mem_roll < 2; ++mem_roll)
            {
                // load matrix_frag A
                double *const shmem_i_ptr = (double *) &Is[0] + 8 * threadIdx.y + 4 * BLOCK_OFF * mem_roll + off_minus;
                nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);
#pragma unroll
                for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE; ++j_roll) {
                    // load matrix_frag B
                    double *const shmem_j_ptr = (double *) &Js[0] + 8 * j_roll + 4 * BLOCK_OFF * mem_roll + off_minus;
                    nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

                    // tensorcore operation, C += A*B
                    nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
                }
            }
            

        }

        // wait for last I- and J-data to be loaded into shared memory
        bar.arrive_and_wait();
        const kernel_index_type off_last = ((feature_range - 8) & 8) * BLOCK_OFF * 2;
        
        // Do 2 iterations
        for (kernel_index_type mem_roll = 0; mem_roll < 2 ; ++mem_roll)
        {   
            // load matrix_frag A
            const double *shmem_i_ptr = (double *) &Is[0] + 8 * threadIdx.y + 4 * BLOCK_OFF * mem_roll + off_last;
            nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);
    #pragma unroll
            for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE; ++j_roll) {
                // load matrix_frag B
                double *const shmem_j_ptr = (double *) &Js[0] + 8 * j_roll + 4 * BLOCK_OFF * mem_roll + off_last;
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

                // tensorcore operation, C += A*B
                nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
            }
        }

        // store solution part acc_frag C back in shared
        for (kernel_index_type frags = 0; frags < ROLL_SIZE; ++frags) {
            double *const shmem_m_ptr = (double *) &solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * frags;
            nvcuda::wmma::store_matrix_sync(shmem_m_ptr, c_frag[frags], BLOCK_OFF, nvcuda::wmma::mem_row_major);
        }

        // Building Matrix done
        // ---------------------
        
        // get d in shared memory
        if (line == 0) {
            ::cuda::memcpy_async(&Vjs[2 * (mem_num)], &d[j + 2 * mem_num], sizeof(double2), bar);
        }
        if (i > j && line == 6) {
            ::cuda::memcpy_async(&Vis[2 * (mem_num)], &d[i + 2 * mem_num], sizeof(double2), bar);
        }

        if (id == 0) {
            if (line == 2) {
                ::cuda::memcpy_async(&Qis[2 * (mem_num)], &q[i + 2 * (mem_num)], sizeof(double2), bar);
            }
            if (line == 4) {
                ::cuda::memcpy_async(&Qjs[2 * (mem_num)], &q[j + 2 * (mem_num)], sizeof(double2), bar);
            }
            bar.arrive_and_wait();

            // offset - gemv
            if (threadIdx.y < BLOCK_SIZE / WARP_SIZE) {  // 96/32=3 --> Warp 0-2
                double sol_tmp = 0.0;
                // each of the 96 Threads iterate over 1 row in the 96x96 Block.
                for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE; ++store_it) {
                    const kernel_index_type index = (id_1d + store_it) % BLOCK_SIZE;
                    const double Qi = Qis[id_1d];
                    sol_tmp += (solution[id_1d * BLOCK_OFF + index] * gamma - Qi - Qjs[index] + QA_cost) * Vjs[index];
                }
                if (i == j) {
                    // add cost on diagonal
                    sol_tmp += cost * Vjs[id_1d];
                }
                atomicAdd(&ret[i + id_1d], sol_tmp * add);
            } else if(i > j && threadIdx.y < 2 * BLOCK_SIZE / WARP_SIZE) { // 196/32=6 + first if --> Warp 3-5
            // upper triangular  
            // no offset - gemv
            double sol_tmp = 0.0;
            const double Qj = Qjs[id_1d - BLOCK_SIZE];
            for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE; ++store_it) {
                sol_tmp += (solution[id_1d - BLOCK_SIZE + store_it * BLOCK_OFF] * gamma - Qj - Qis[store_it] + QA_cost) * Vis[store_it];
            }
            atomicAdd(&ret[j + id_1d - BLOCK_SIZE], sol_tmp * add);
            }
        } else {
            bar.arrive_and_wait();
            // offset - gemv
            if (threadIdx.y < BLOCK_SIZE / WARP_SIZE) {
                double sol_tmp = 0.0;
                for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE; ++store_it) {
                    const kernel_index_type index = (id_1d + store_it) % BLOCK_SIZE;
                    sol_tmp += solution[id_1d * BLOCK_OFF + index] * gamma * Vjs[index];
                }
                atomicAdd(&ret[i + id_1d], sol_tmp * add);
            } else if(i > j && threadIdx.y < 2 * BLOCK_SIZE / WARP_SIZE) {
                // upper triangular
                // no offset - gemv
                double sol_tmp = 0.0;
                for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE; ++store_it) {
                    sol_tmp += solution[id_1d - BLOCK_SIZE + store_it * BLOCK_OFF] * gamma * Vis[store_it];
                }
                atomicAdd(&ret[j + id_1d - BLOCK_SIZE], sol_tmp * add);
            }
        }
    }
}

__global__ void device_kernel_linear_tf(const float *q, float *ret, const float *d, const float *data_d, const float QA_cost, const float cost, const kernel_index_type points, const kernel_index_type feature_range, const float add, const float gamma, const kernel_index_type id) {
    const kernel_index_type i = blockIdx.x * BLOCK_SIZE_F;
    const kernel_index_type j = blockIdx.y * BLOCK_SIZE_F;
    if (i >= j) {
        const kernel_index_type id_1d = threadIdx.y * 32 + threadIdx.x;
        const kernel_index_type line = id_1d / (BLOCK_SIZE_F / 8);
        const kernel_index_type mem_num = id_1d & 15;
        __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar[1];

        extern __shared__ float solution_f[];
        float *Is = (float *) &solution_f[0];
        float *Js = (float *) &solution_f[0] + 8 * BLOCK_OFF_F;
        float *Vis = (float *) &solution_f[0] + (BLOCK_SIZE_F + 1) * BLOCK_OFF_F;
        float *Vjs = (float *) &solution_f[0] + (BLOCK_SIZE_F + 2) * BLOCK_OFF_F;
        float *Qis = (float *) &solution_f[0] + (BLOCK_SIZE_F + 3) * BLOCK_OFF_F;
        float *Qjs = (float *) &solution_f[0] + (BLOCK_SIZE_F + 4) * BLOCK_OFF_F;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            init(&bar[0], THREADS_PER_BLOCK_F);
        }

        __syncthreads();

        // Build Matrix
        // ---------------------

        // load Data for first iteration
        if(threadIdx.y < 4) {
            float4 *const I2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8]);
            const float4 *const D_i2 = reinterpret_cast<const float4 *>(&data_d[line * points + mem_num * 8 + i]);
            ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(float4), bar[0]);
        } else {
            // Each warp one line for J
            float4 *const J2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8]);
            const float4 *const D_j2 = reinterpret_cast<const float4 *>(&data_d[(line - 8) * points + mem_num * 8 + j]);
            ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(float4), bar[0]);
        }

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> c_frag[ROLL_SIZE_F];
        for (kernel_index_type frags = 0; frags < ROLL_SIZE_F; ++frags) {
            // fill c_frags with zeros
            nvcuda::wmma::fill_fragment(c_frag[frags], 0.0f);
        }

        // loop over features
        for (kernel_index_type feature_it = 8; feature_it < feature_range; feature_it = feature_it + 8) {
            const kernel_index_type off_plus = (feature_it & 8) * BLOCK_OFF_F * 2;
            const kernel_index_type off_minus = abs(off_plus - 16 * BLOCK_OFF_F);

            // wait for new I-data to be loaded into shared memory
            bar[0].arrive_and_wait();

            if(threadIdx.y < 4) {
                float4 *const I2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8 + off_plus]);
                const float4 *const D_i2 = reinterpret_cast<const float4 *>(&data_d[line * points + mem_num * 8 + feature_it * points + i]);
                ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(float4), bar[0]);
            } else {
                // Each warp one line for J
                float4 *const J2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8 + off_plus]);
                const float4 *const D_j2 = reinterpret_cast<const float4 *>(&data_d[(line - 8) * points + mem_num * 8 + feature_it * points + j]);
                ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(float4), bar[0]);
            }

            // load matrix_frag A
            float *const shmem_i_ptr = (float *) &Is[0] + 16 * threadIdx.y + off_minus;
            nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF_F);

            // Transform a_frag to tf32
#pragma unroll
            for (kernel_index_type t = 0; t < a_frag.num_elements; ++t) {
                a_frag.x[t] = nvcuda::wmma::__float_to_tf32(a_frag.x[t]);
            }

#pragma unroll
            for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE_F; ++j_roll) {
                float *const shmem_j_ptr = (float *) &Js[0] + 16 * j_roll + off_minus;
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF_F);

                // Transform b_frag to tf32
#pragma unroll
                for (kernel_index_type t = 0; t < b_frag.num_elements; ++t) {
                    b_frag.x[t] = nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
                }

                // tensorcore operation, C += A*B
                nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
            }
        }

        // wait for new I-data to be loaded into shared memory
        bar[0].arrive_and_wait();
        const kernel_index_type off_last = ((feature_range - 8) & 8) * BLOCK_OFF_F * 2;

        // load matrix_frag A
        const float *shmem_i_ptr = (float *) &Is[0] + 16 * threadIdx.y + off_last;
        nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF_F);

        // Transform a_frag to tf32
#pragma unroll
        for (kernel_index_type t = 0; t < a_frag.num_elements; ++t) {
            a_frag.x[t] = nvcuda::wmma::__float_to_tf32(a_frag.x[t]);
        }

#pragma unroll
        for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE_F; ++j_roll) {
            float *const shmem_j_ptr = (float *) &Js[0] + 16 * j_roll + off_last;
            nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF_F);

            // Transform b_frag to tf32
#pragma unroll
            for (kernel_index_type t = 0; t < b_frag.num_elements; ++t) {
                b_frag.x[t] = nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
            }

            // tensorcore operation, C += A*B
            nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
        }

        // store solution part acc_frag C back in shared
        for (kernel_index_type frags = 0; frags < ROLL_SIZE_F; ++frags) {
            float *const shmem_m_ptr = (float *) &solution_f[0] + 16 * threadIdx.y * BLOCK_OFF_F + 16 * frags;
            nvcuda::wmma::store_matrix_sync(shmem_m_ptr, c_frag[frags], BLOCK_OFF_F, nvcuda::wmma::mem_row_major);
        }

        // Building Matrix done
        // ---------------------

        // get d in shared memory
        if (threadIdx.y < 1) {
            ::cuda::memcpy_async(&Vjs[4 * id_1d], &d[j + 4 * id_1d], sizeof(float4), bar[0]);
        }

        if (id == 0) {
            if (threadIdx.y > 0 && threadIdx.y < 2) {
                ::cuda::memcpy_async(&Qis[4 * (id_1d - 32)], &q[i + 4 * (id_1d - 32)], sizeof(float4), bar[0]);
            }
            if (threadIdx.y > 1 && threadIdx.y < 3) {
                ::cuda::memcpy_async(&Qjs[4 * (id_1d - 64)], &q[j + 4 * (id_1d - 64)], sizeof(float4), bar[0]);
            }

            bar[0].arrive_and_wait();

            // offset - gemv
            if (threadIdx.y < BLOCK_SIZE_F / WARP_SIZE) {  // 128/32=4 --> Warp 0-3
                float sol_tmp = 0.0;
                // each of the 128 Threads iterate over 1 row in the 128x128 Block.
                for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE_F; ++store_it) {
                    const kernel_index_type index = (id_1d + store_it) & (BLOCK_SIZE_F - 1);  // modulo 128
                    const float Qi = Qis[id_1d];
                    sol_tmp += (solution_f[id_1d * BLOCK_OFF_F + index] * gamma - Qi - Qjs[index] + QA_cost) * Vjs[index];
                }
                if (i == j) {
                    // int index = id_1d * 97; // offset + 1
                    sol_tmp += cost * Vjs[id_1d];
                }
                atomicAdd(&ret[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if (i > j) {
                if (threadIdx.y < 1) {
                    ::cuda::memcpy_async(&Vis[4 * id_1d], &d[i + 4 * id_1d], sizeof(float4), bar[0]);
                }
                bar[0].arrive_and_wait();

                // no offset - gemv
                if (threadIdx.y < BLOCK_SIZE_F / WARP_SIZE) {  // 128/32=4 --> Warp 0-3
                    float sol_tmp = 0.0;
                    const float Qj = Qjs[id_1d];
                    for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE_F; ++store_it) {
                        sol_tmp += (solution_f[id_1d + store_it * BLOCK_OFF_F] * gamma - Qj - Qis[store_it] + QA_cost) * Vis[store_it];
                    }
                    atomicAdd(&ret[j + id_1d], sol_tmp * add);
                }
            }
        } else {
            bar[0].arrive_and_wait();
            // offset - gemv
            if (threadIdx.y < BLOCK_SIZE_F / WARP_SIZE) {
                float sol_tmp = 0.0;
                for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE_F; ++store_it) {
                    const kernel_index_type index = (id_1d + store_it) & (BLOCK_SIZE_F - 1);
                    sol_tmp += solution_f[id_1d * BLOCK_OFF_F + index] * gamma * Vjs[index];
                }
                atomicAdd(&ret[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if (i > j) {
                if (threadIdx.y < 1) {
                    ::cuda::memcpy_async(&Vis[4 * id_1d], &d[i + 4 * id_1d], sizeof(float4), bar[0]);
                }
                bar[0].arrive_and_wait();

                // no offset - gemv
                if (threadIdx.y < BLOCK_SIZE_F / WARP_SIZE) {
                    float sol_tmp = 0.0;
                    for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE_F; ++store_it) {
                        sol_tmp += solution_f[id_1d + store_it * BLOCK_OFF_F] * gamma * Vis[store_it];
                    }
                    atomicAdd(&ret[j + id_1d], sol_tmp * add);
                }
            }
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

__global__ void device_kernel_poly_td(const double *q, double *ret, const double *d, const double *data_d, const double QA_cost, const double cost, const kernel_index_type points, const kernel_index_type feature_range, const double add, const kernel_index_type degree, const double gamma, const double coef0) {
    const kernel_index_type i = blockIdx.x * BLOCK_SIZE;
    const kernel_index_type j = blockIdx.y * BLOCK_SIZE;
    // TODO Check here or in setup for integer overflow - for all Kernels!
    // idea:
    // if(cast (points * feature_range) > int_max)
    // split *data_d, adjust feature_range
    // const double *in[split] = ...
    // feature_range[split] = ...
    // loop BUILD_MATRIX (excluded fragment init and store phase) over *data_d[] and feature_range[]

    if (i >= j) {  // if lower triangular matrix
        // helper variables for flow and copy
        const kernel_index_type id_1d = threadIdx.y * 32 + threadIdx.x;
        const kernel_index_type line = id_1d / (BLOCK_SIZE / 2);
        const kernel_index_type mem_num = id_1d % (BLOCK_SIZE / 2);
        const kernel_index_type transfer_line = id_1d / (BLOCK_SIZE / 4);
        const kernel_index_type tranfser_offset = id_1d % (BLOCK_SIZE / 4);

        __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar;

        extern __shared__ double solution[];
        double *Is = (double *) &solution[0];
        double *Js = (double *) &solution[0] + 8 * BLOCK_OFF;
        double *Vjs = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF;
        double *Vis = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF + 1 * BLOCK_OFF;
        double *Qis = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF + 2 * BLOCK_OFF;
        double *Qjs = (double *) &solution[0] + BLOCK_SIZE * BLOCK_OFF + 3 * BLOCK_OFF;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            init(&bar, THREADS_PER_BLOCK);
        }
        __syncthreads();

        // Build Matrix
        // ---------------------

        // load Data for first iteration
        if (threadIdx.y < 6) {  // warp 0-5
            // 6 warps per 8 feature lines for I
            double2 *const I2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4]);
            const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[transfer_line * points + tranfser_offset * 4 + i]);
            ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(double2), bar);
        } else {  // warp 6-11
            // 6 warps per 8 feature lines for J
            double2 *const J2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4]);
            const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[(transfer_line - 8) * points + tranfser_offset * 4 + j]);
            ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(double2), bar);
        }

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> c_frag[ROLL_SIZE];
        for (kernel_index_type frags = 0; frags < ROLL_SIZE; ++frags) {
            // fill c_frags with zeros
            nvcuda::wmma::fill_fragment(c_frag[frags], 0.0f);
        }

        // loop over features
        for (kernel_index_type feature_it = 8; feature_it < feature_range; feature_it = feature_it + 8) {
            const kernel_index_type off_plus = (feature_it & 8) * BLOCK_OFF * 2;
            const kernel_index_type off_minus = abs(off_plus - 16 * BLOCK_OFF);

            // wait for new I- and J-data to be loaded into shared memory
            bar.arrive_and_wait();

            if (threadIdx.y < 6) {  // warp 0-5
                // 6 warps per 8 feature lines for I
                double2 *const I2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4 + off_plus]);
                const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[transfer_line * points + tranfser_offset * 4 + feature_it * points + i]);
                ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(double2), bar);
            } else {  // warp 6-11
                // 6 warps per 8 feature lines for J
                double2 *const J2s = reinterpret_cast<double2 *>(&Is[transfer_line * BLOCK_OFF + tranfser_offset * 4 + off_plus]);
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[(transfer_line - 8) * points + tranfser_offset * 4 + feature_it * points + j]);
                ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(double2), bar);
            }

            // Do 2 iterations
            for (kernel_index_type mem_roll = 0; mem_roll < 2; ++mem_roll) {
                // load matrix_frag A
                double *const shmem_i_ptr = (double *) &Is[0] + 8 * threadIdx.y + 4 * BLOCK_OFF * mem_roll + off_minus;
                nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);
#pragma unroll
                for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE; ++j_roll) {
                    // load matrix_frag B
                    double *const shmem_j_ptr = (double *) &Js[0] + 8 * j_roll + 4 * BLOCK_OFF * mem_roll + off_minus;
                    nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

                    // tensorcore operation, C += A*B
                    nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
                }
            }
        }

        // wait for last I- and J-data to be loaded into shared memory
        bar.arrive_and_wait();
        const kernel_index_type off_last = ((feature_range - 8) & 8) * BLOCK_OFF * 2;

        // Do 2 iterations
        for (kernel_index_type mem_roll = 0; mem_roll < 2; ++mem_roll) {
            // load matrix_frag A
            const double *shmem_i_ptr = (double *) &Is[0] + 8 * threadIdx.y + 4 * BLOCK_OFF * mem_roll + off_last;
            nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);
#pragma unroll
            for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE; ++j_roll) {
                // load matrix_frag B
                double *const shmem_j_ptr = (double *) &Js[0] + 8 * j_roll + 4 * BLOCK_OFF * mem_roll + off_last;
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

                // tensorcore operation, C += A*B
                nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
            }
        }

        // store solution part acc_frag C back in shared
        for (kernel_index_type frags = 0; frags < ROLL_SIZE; ++frags) {
            double *const shmem_m_ptr = (double *) &solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * frags;
            nvcuda::wmma::store_matrix_sync(shmem_m_ptr, c_frag[frags], BLOCK_OFF, nvcuda::wmma::mem_row_major);
        }

        // Building Matrix done
        // ---------------------

        // get d in shared memory
        if (line == 0) {
            ::cuda::memcpy_async(&Vjs[2 * (mem_num)], &d[j + 2 * mem_num], sizeof(double2), bar);
        }
        if (i > j && line == 6) {
            ::cuda::memcpy_async(&Vis[2 * (mem_num)], &d[i + 2 * mem_num], sizeof(double2), bar);
        }

        if (line == 2) {
            ::cuda::memcpy_async(&Qis[2 * (mem_num)], &q[i + 2 * (mem_num)], sizeof(double2), bar);
        }
        if (line == 4) {
            ::cuda::memcpy_async(&Qjs[2 * (mem_num)], &q[j + 2 * (mem_num)], sizeof(double2), bar);
        }
        bar.arrive_and_wait();

        // offset - gemv
        if (threadIdx.y < BLOCK_SIZE / WARP_SIZE) {  // 96/32=3 --> Warp 0-2
            double sol_tmp = 0.0;
            // each of the 96 Threads iterate over 1 row in the 96x96 Block.
            for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE; ++store_it) {
                const kernel_index_type index = (id_1d + store_it) % BLOCK_SIZE;
                const double Qi = Qis[id_1d];
                sol_tmp += (pow(solution[id_1d * BLOCK_OFF + index] * gamma + coef0, degree) - Qi - Qjs[index] + QA_cost) * Vjs[index];
            }
            if (i == j) {
                // add cost on diagonal
                sol_tmp += cost * Vjs[id_1d];
            }
            atomicAdd(&ret[i + id_1d], sol_tmp * add);
        } else if (i > j && threadIdx.y < 2 * BLOCK_SIZE / WARP_SIZE) {  // 196/32=6 + first if --> Warp 3-5
            // upper triangular
            // no offset - gemv
            double sol_tmp = 0.0;
            const double Qj = Qjs[id_1d - BLOCK_SIZE];
            for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE; ++store_it) {
                sol_tmp += (pow(solution[id_1d - BLOCK_SIZE + store_it * BLOCK_OFF] * gamma + coef0, degree) - Qj - Qis[store_it] + QA_cost) * Vis[store_it];
            }
            atomicAdd(&ret[j + id_1d - BLOCK_SIZE], sol_tmp * add);
        }
    }
}

__global__ void device_kernel_poly_tf(const float *q, float *ret, const float *d, const float *data_d, const float QA_cost, const float cost, const kernel_index_type points, const kernel_index_type feature_range, const float add, const kernel_index_type degree, const float gamma, const real_type coef0) {
    const kernel_index_type i = blockIdx.x * BLOCK_SIZE_F;
    const kernel_index_type j = blockIdx.y * BLOCK_SIZE_F;
    if (i >= j) {
        const kernel_index_type id_1d = threadIdx.y * 32 + threadIdx.x;
        const kernel_index_type line = id_1d / (BLOCK_SIZE_F / 8);
        const kernel_index_type mem_num = id_1d & 15;
        __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar[1];

        extern __shared__ float solution_f[];
        float *Is = (float *) &solution_f[0];
        float *Js = (float *) &solution_f[0] + 8 * BLOCK_OFF_F;
        float *Vis = (float *) &solution_f[0] + (BLOCK_SIZE_F + 1) * BLOCK_OFF_F;
        float *Vjs = (float *) &solution_f[0] + (BLOCK_SIZE_F + 2) * BLOCK_OFF_F;
        float *Qis = (float *) &solution_f[0] + (BLOCK_SIZE_F + 3) * BLOCK_OFF_F;
        float *Qjs = (float *) &solution_f[0] + (BLOCK_SIZE_F + 4) * BLOCK_OFF_F;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            init(&bar[0], THREADS_PER_BLOCK_F);
        }

        __syncthreads();

        // Build Matrix
        // ---------------------

        // load Data for first iteration
        if(threadIdx.y < 4) {
            float4 *const I2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8]);
            const float4 *const D_i2 = reinterpret_cast<const float4 *>(&data_d[line * points + mem_num * 8 + i]);
            ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(float4), bar[0]);
        } else {
            // Each warp one line for J
            float4 *const J2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8]);
            const float4 *const D_j2 = reinterpret_cast<const float4 *>(&data_d[(line - 8) * points + mem_num * 8 + j]);
            ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(float4), bar[0]);
        }

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> c_frag[ROLL_SIZE_F];
        for (kernel_index_type frags = 0; frags < ROLL_SIZE_F; ++frags) {
            // fill c_frags with zeros
            nvcuda::wmma::fill_fragment(c_frag[frags], 0.0f);
        }

        // loop over features
        for (kernel_index_type feature_it = 8; feature_it < feature_range; feature_it = feature_it + 8) {
            const kernel_index_type off_plus = (feature_it & 8) * BLOCK_OFF_F * 2;
            const kernel_index_type off_minus = abs(off_plus - 16 * BLOCK_OFF_F);

            // wait for new I-data to be loaded into shared memory
            bar[0].arrive_and_wait();

            if(threadIdx.y < 4) {
                float4 *const I2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8 + off_plus]);
                const float4 *const D_i2 = reinterpret_cast<const float4 *>(&data_d[line * points + mem_num * 8 + feature_it * points + i]);
                ::cuda::memcpy_async(I2s, D_i2, 2 * sizeof(float4), bar[0]);
            } else {
                // Each warp one line for J
                float4 *const J2s = reinterpret_cast<float4 *>(&Is[line * BLOCK_OFF_F + mem_num * 8 + off_plus]);
                const float4 *const D_j2 = reinterpret_cast<const float4 *>(&data_d[(line - 8) * points + mem_num * 8 + feature_it * points + j]);
                ::cuda::memcpy_async(J2s, D_j2, 2 * sizeof(float4), bar[0]);
            }

            // load matrix_frag A
            float *const shmem_i_ptr = (float *) &Is[0] + 16 * threadIdx.y + off_minus;
            nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF_F);

            // Transform a_frag to tf32
#pragma unroll
            for (kernel_index_type t = 0; t < a_frag.num_elements; ++t) {
                a_frag.x[t] = nvcuda::wmma::__float_to_tf32(a_frag.x[t]);
            }

#pragma unroll
            for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE_F; ++j_roll) {
                float *const shmem_j_ptr = (float *) &Js[0] + 16 * j_roll + off_minus;
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF_F);

                // Transform b_frag to tf32
#pragma unroll
                for (kernel_index_type t = 0; t < b_frag.num_elements; ++t) {
                    b_frag.x[t] = nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
                }

                // tensorcore operation, C += A*B
                nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
            }
        }

        // wait for new I-data to be loaded into shared memory
        bar[0].arrive_and_wait();
        const kernel_index_type off_last = ((feature_range - 8) & 8) * BLOCK_OFF_F * 2;

        // load matrix_frag A
        const float *shmem_i_ptr = (float *) &Is[0] + 16 * threadIdx.y + off_last;
        nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF_F);

        // Transform a_frag to tf32
#pragma unroll
        for (kernel_index_type t = 0; t < a_frag.num_elements; ++t) {
            a_frag.x[t] = nvcuda::wmma::__float_to_tf32(a_frag.x[t]);
        }

#pragma unroll
        for (kernel_index_type j_roll = 0; j_roll < ROLL_SIZE_F; ++j_roll) {
            float *const shmem_j_ptr = (float *) &Js[0] + 16 * j_roll + off_last;
            nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF_F);

            // Transform b_frag to tf32
#pragma unroll
            for (kernel_index_type t = 0; t < b_frag.num_elements; ++t) {
                b_frag.x[t] = nvcuda::wmma::__float_to_tf32(b_frag.x[t]);
            }

            // tensorcore operation, C += A*B
            nvcuda::wmma::mma_sync(c_frag[j_roll], a_frag, b_frag, c_frag[j_roll]);
        }

        // store solution part acc_frag C back in shared
        for (kernel_index_type frags = 0; frags < ROLL_SIZE_F; ++frags) {
            float *const shmem_m_ptr = (float *) &solution_f[0] + 16 * threadIdx.y * BLOCK_OFF_F + 16 * frags;
            nvcuda::wmma::store_matrix_sync(shmem_m_ptr, c_frag[frags], BLOCK_OFF_F, nvcuda::wmma::mem_row_major);
        }

        // Building Matrix done
        // ---------------------

        // get d in shared memory
        if (threadIdx.y < 1) {
            ::cuda::memcpy_async(&Vjs[4 * id_1d], &d[j + 4 * id_1d], sizeof(float4), bar[0]);
        }

        if (threadIdx.y > 0 && threadIdx.y < 2) {
            ::cuda::memcpy_async(&Qis[4 * (id_1d - 32)], &q[i + 4 * (id_1d - 32)], sizeof(float4), bar[0]);
        }
        if (threadIdx.y > 1 && threadIdx.y < 3) {
            ::cuda::memcpy_async(&Qjs[4 * (id_1d - 64)], &q[j + 4 * (id_1d - 64)], sizeof(float4), bar[0]);
        }

        bar[0].arrive_and_wait();

        // offset - gemv
        if (threadIdx.y < BLOCK_SIZE_F / WARP_SIZE) {  // 128/32=4 --> Warp 0-3
            float sol_tmp = 0.0;
            // each of the 128 Threads iterate over 1 row in the 128x128 Block.
            for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE_F; ++store_it) {
                const kernel_index_type index = (id_1d + store_it) & (BLOCK_SIZE_F - 1);  // modulo 128
                const float Qi = Qis[id_1d];
                sol_tmp += (pow(solution_f[id_1d * BLOCK_OFF + index] * gamma + coef0, degree) - Qi - Qjs[index] + QA_cost) * Vjs[index];
           }
            if (i == j) {
                sol_tmp += cost * Vjs[id_1d];
            }
            atomicAdd(&ret[i + id_1d], sol_tmp * add);
        }

        __syncthreads();

        // upper triangular
        if (i > j) {
            if (threadIdx.y < 1) {
                ::cuda::memcpy_async(&Vis[4 * id_1d], &d[i + 4 * id_1d], sizeof(float4), bar[0]);
            }
            bar[0].arrive_and_wait();

            // no offset - gemv
            if (threadIdx.y < BLOCK_SIZE_F / WARP_SIZE) {  // 128/32=4 --> Warp 0-3
                float sol_tmp = 0.0;
                const float Qj = Qjs[id_1d];
                for (kernel_index_type store_it = 0; store_it < BLOCK_SIZE_F; ++store_it) {
                    sol_tmp += (pow(solution_f[id_1d + store_it * BLOCK_OFF_F] * gamma + coef0, degree) - Qj - Qis[store_it] + QA_cost) * Vis[store_it];
                }
                atomicAdd(&ret[j + id_1d], sol_tmp * add);
            }
        }
    }
}

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