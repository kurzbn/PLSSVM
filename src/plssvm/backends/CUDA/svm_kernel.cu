/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/svm_kernel.cuh"

#include <cuda/barrier>
#include <mma.h>

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd for double precision floating point numbers on older CUDA hardware
#include "plssvm/constants.hpp"                     // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::cuda {

__global__ void device_kernel_linear_mixed(const float *q, real_type *ret, const float *d, const float *data_d, const float QA_cost, const float cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const float add, const kernel_index_type id) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    // __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar[2];

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

/*__global__ void device_kernel_linear_t(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type feature_range, const real_type add, const real_type gamma, const kernel_index_type id) {
    kernel_index_type i = blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
    kernel_index_type j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;

    const int warp_id = threadIdx.y / 2;
    const int smemPlusStride = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE + 2;

    __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar[2];
    // __shared__ alignas(alignof(double2)) real_type Is[4][THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE];
    // __shared__ alignas(alignof(double2)) real_type Js[4][THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE];
    // __shared__ alignas(alignof(double2)) real_type Mats[8][64];
    // stride privious: 8 * 64
    __shared__ alignas(alignof(double2)) real_type Is[4 * smemPlusStride];
    __shared__ alignas(alignof(double2)) real_type Js[4 * smemPlusStride];
    __shared__ alignas(alignof(double2)) real_type Is_2[4 * smemPlusStride];
    __shared__ alignas(alignof(double2)) real_type Js_2[4 * smemPlusStride];
    __shared__ alignas(alignof(double2)) real_type Mats[8 * 66]; // INTERNAL_BLOCK_SIZE * INTERNAL_BLOCK_SIZE * (256 + 2)
    //extern __shared__ double2 Mats_[];
    //double * Mats = Mats_;
    // extern __shared__ alignas(alignof(double2)) real_type Mats_test[2]; 
    // extern __shared__ alignas(alignof(double2)) unsigned char Mats_d[];
    // real_type *Mats = reinterpret_cast<real_type *>(Mats_d);
    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    // real_type data_j[INTERNAL_BLOCK_SIZE];
    if (threadIdx.x == 0) {
        init(&bar[0], blockDim.x * blockDim.y);
        init(&bar[1], blockDim.x * blockDim.y);
    }
    __syncthreads();



    if (i >= j) {
        // cache data - 4 times "THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE" for i and j.
        if(threadIdx.y < 12){
            double2 *const I2s = reinterpret_cast<double2 *>(&Is[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
            double2 *const J2s = reinterpret_cast<double2 *>(&Js[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
            const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[i + threadIdx.x / 4 * num_rows + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // feature range = num rows?
            const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[j + threadIdx.x / 4 * num_rows + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // j =  Block, feature_range --> welche der 4 Zeilen, vec Index iter, Rest wo im Block

            ::cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
            ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]);
        }

        for (kernel_index_type vec_index = 0; vec_index < feature_range * num_rows; vec_index += 8*num_rows) {

            // __syncthreads();


            bar[0].arrive_and_wait();

            if(threadIdx.y < 12){
                double2 *const I2s_2 = reinterpret_cast<double2 *>(&Is_2[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                double2 *const J2s_2 = reinterpret_cast<double2 *>(&Js_2[(threadIdx.x / 4) * smemPlusStride + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                const double2 *const D_i2_2 = reinterpret_cast<const double2 *>(&data_d[i + (threadIdx.x / 4 + 4) * num_rows + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // feature range = num rows?
                const double2 *const D_j2_2 = reinterpret_cast<const double2 *>(&data_d[j + (threadIdx.x / 4 + 4) * num_rows + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // j =  Block, feature_range --> welche der 4 Zeilen, vec Index iter, Rest wo im Block

                ::cuda::memcpy_async(I2s_2, D_i2_2, sizeof(double2), bar[1]);
                ::cuda::memcpy_async(J2s_2, D_j2_2, sizeof(double2), bar[1]);
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
                    double *shmem_i_ptr = (double*)&Is[0] + 8*k + (warp_id/4) * 48; // (double*)&Is[0] + 8*k + warp_id * 12; wrong? --> (double*)&Is[0] + 8*k + warp_id/4 * 48;
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
                const double2 *const D_i2 = reinterpret_cast<const double2 *>(&data_d[i + (threadIdx.x / 4 + 8) * num_rows + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]);
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&data_d[j + (threadIdx.x / 4 + 8) * num_rows + vec_index + (threadIdx.x % 4)*2 + 8 * threadIdx.y]); // j =  Block, feature_range --> welche der 4 Zeilen, vec Index iter, Rest wo im Block

                ::cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
                ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]);
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (kernel_index_type l = 0; l < INTERNAL_BLOCK_SIZE / 2; ++l) {
                // double *shmem_j_ptr = (double*)&Js[0][0] + 8*l + (warp_id % 4) * 24; 
                double *shmem_j_ptr = (double*)&Js_2[0] + 8*l + (warp_id % 4) * 24;
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, smemPlusStride);
                
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (kernel_index_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    double *shmem_i_ptr = (double*)&Is_2[0] + 8*k + (warp_id/4) * 48;
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
}*/

/* __global__ void device_kernel_linear_t(const double *q, double *out, const double* vec, const double *in, const double QA_cost, const double cost, const int points, const int feature_range, const double add, const double gamma, const int id) {
    const int i = blockIdx.x * blockDim.x * 3 * 2; // X_Roll
    const int j = blockIdx.y * blockDim.y * 6; // Y_Roll
    if(i >= j){
        const int id_1d = threadIdx.y * 16 + threadIdx.x;
        const int warp_id = threadIdx.y / 2;
        // const int offset = 96 + 0;
        const int offset = 96 + 8;

        __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar[3];

        extern __shared__ double solution[];
        double *Is= (double*)&solution[0] + offset*96;
        double *Js= (double*)&solution[0] + offset*96 + offset*4;
        double *Vs= (double*)&solution[0] + offset*96 + offset*8;
        double *Qis= (double*)&solution[0] + offset*96 + offset*8 + 96;
        double *Qjs= (double*)&solution[0] + offset*96 + offset*8 + 2*96;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            init(&bar[0], 256);
            init(&bar[1], 256);
            init(&bar[2], 256);
        }

        for(int set_zero = 0; set_zero < 37; ++set_zero){ // Test - 36!
            int index = id_1d + set_zero*256;
            if(index < 10272) solution[index] = 0.0;
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> c_frag; 

        for(int feature_it = 0; feature_it < feature_range; feature_it = feature_it +4){
            if(threadIdx.y < 12){
                double2 *const I2s = reinterpret_cast<double2 *>(&Is[(threadIdx.y / 3)*offset + ((threadIdx.y % 3)*16 + threadIdx.x) * 2]);
                double2 *const J2s = reinterpret_cast<double2 *>(&Js[(threadIdx.y / 3)*offset + ((threadIdx.y % 3)*16 + threadIdx.x) * 2]);
                const double2 *const D_i2 = reinterpret_cast<const double2 *>(&in[(threadIdx.y / 3)*points + ((threadIdx.y % 3)*16 + threadIdx.x) * 2 + feature_it*points + i]); // feature range = num rows?
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&in[(threadIdx.y / 3)*points + ((threadIdx.y % 3)*16 + threadIdx.x) * 2 + feature_it*points + j]); // j =  Block, feature_range --> welche der 4 Zeilen, vec Index iter, Rest wo im Block
                
                ::cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
                ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[1]);
            }

            bar[0].arrive_and_wait();
            bar[1].arrive_and_wait();
            
            for(int i_roll = 0; i_roll < 3; ++i_roll){ //X_Roll = 3
                double *shmem_i_ptr = (double*)&Is[0] + i_roll*8 + (warp_id / 2) * 24; // 24 lanes
                nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, offset);

                for(int j_roll = 0; j_roll < 6; ++j_roll){ //Y_Roll = 6
                    double *shmem_j_ptr = (double*)&Js[0] + j_roll*8 + (warp_id % 2) * 48; // 48=96/2, halber offset
                    nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, offset);

                    double *shmem_m_ptr = (double*)&solution[0] + j_roll* 8 + (warp_id % 2) * 48 + (i_roll* 8 + (warp_id / 2) * 24) * offset;
                    nvcuda::wmma::load_matrix_sync(c_frag, shmem_m_ptr, offset, nvcuda::wmma::mem_row_major);

                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

                    
                    nvcuda::wmma::store_matrix_sync(shmem_m_ptr , c_frag, offset, nvcuda::wmma::mem_row_major);
                }
            }

            __syncthreads();
        }

        // get vec in shared memory
        if(threadIdx.y < 3){    
            ::cuda::memcpy_async(&Vs[2*id_1d], &vec[j + 2*id_1d], sizeof(double2), bar[2]);   
        }
        
        if(id==0){
            if(threadIdx.y > 3 && threadIdx.y < 7){    
                ::cuda::memcpy_async(&Qis[2*(id_1d - 64)], &q[i + 2*(id_1d - 64)], sizeof(double2), bar[2]);   
            }
            if(threadIdx.y > 7 && threadIdx.y < 11){    
                ::cuda::memcpy_async(&Qjs[2*(id_1d - 128)], &q[j + 2*(id_1d - 128)], sizeof(double2), bar[2]);   
            }

            bar[2].arrive_and_wait();

            // no offset - gemv
            /
            if(threadIdx.y < 6){
                double sol_tmp = 0.0;
                for(int store_it = 0; store_it < 96; ++store_it){

                    sol_tmp += solution[id_1d*offset + store_it] * Vs[store_it];
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                }
                atomicAdd(&out[i + id_1d], sol_tmp);
            }
            /

            // offset - gemv
            if(threadIdx.y < 6){
                double sol_tmp = 0.0;
                for(int store_it = 0; store_it < 96; ++store_it){
                    int index = (id_1d + store_it)%96;
                    double Qi = Qis[id_1d];
                    sol_tmp += (solution[id_1d*offset + index] * gamma - Qi - Qjs[index] + QA_cost)* Vs[index];
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0) printf("Index: %i - Qi: %f - Qj: %f\n", index, Qi, Qjs[index]);
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0) printf("Index: %i - sol_tmp: %f \n", index, sol_tmp);
                }
                if(i==j){
                    // int index = id_1d * 97; // offset + 1
                    sol_tmp += cost * Vs[id_1d];
                }
                atomicAdd(&out[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if(i>j){
                if(threadIdx.y < 3){    
                    ::cuda::memcpy_async(&Vs[2*id_1d], &vec[i + 2*id_1d], sizeof(double2), bar[2]);   
                }
                bar[2].arrive_and_wait();
                
                // no offset - gemv
                if(threadIdx.y < 6){
                    double sol_tmp = 0.0;
                    double Qj = Qjs[id_1d];
                    for(int store_it = 0; store_it < 96; ++store_it){
                        sol_tmp += (solution[id_1d + store_it * offset] * gamma - Qj - Qis[store_it] + QA_cost)* Vs[store_it];
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 1 && blockIdx.y == 0) printf("Index: %i - Qj: %f - Qis: %f\n", store_it, Qj, Qis[store_it]);
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                    }
                    atomicAdd(&out[j + id_1d], sol_tmp *add);
                }
            }
        } else {
            bar[2].arrive_and_wait();
           // offset - gemv
            if(threadIdx.y < 6){
                double sol_tmp = 0.0;
                for(int store_it = 0; store_it < 96; ++store_it){
                    int index = (id_1d + store_it)%96;
                    sol_tmp += solution[id_1d*offset + index] * gamma * Vs[index];
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                }
                atomicAdd(&out[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if(i>j){
                if(threadIdx.y < 3){    
                    ::cuda::memcpy_async(&Vs[2*id_1d], &vec[i + 2*id_1d], sizeof(double2), bar[2]);   
                }
                bar[2].arrive_and_wait();
                
                // no offset - gemv
                if(threadIdx.y < 6){
                    double sol_tmp = 0.0;
                    for(int store_it = 0; store_it < 96; ++store_it){
                        sol_tmp += solution[id_1d + store_it * offset] * gamma * Vs[store_it];
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                    }
                    atomicAdd(&out[j + id_1d], sol_tmp *add);
                }
            }
        }
    }
} */

/* __global__ void device_kernel_linear_t(const double *q, double *out, const double* vec, const double *in, const double QA_cost, const double cost, const int points, const int feature_range, const double add, const double gamma, const int id) {
    const int i = blockIdx.x * BLOCK_SIZE;
    const int j = blockIdx.y * BLOCK_SIZE;
    if(i >= j){
        const int id_1d = threadIdx.y * 32 + threadIdx.x;
        __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar[3];

        extern __shared__ double solution[];
        double *Is= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF;
        double *Js= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 4 *BLOCK_OFF;
        double *Vs= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 8 *BLOCK_OFF;
        double *Qis= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 9*BLOCK_OFF;
        double *Qjs= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 10*BLOCK_OFF;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            init(&bar[0], THREADS_PER_BLOCK);
            init(&bar[1], THREADS_PER_BLOCK);
            init(&bar[2], THREADS_PER_BLOCK);
        }

        // reset shared memory
        for(int set_zero = 0; set_zero < 36; ++set_zero){
            const int index = id_1d + set_zero * THREADS_PER_BLOCK;
            if(index < 18880) solution[index] = 0.0;
        }
        __syncthreads();

        // Build Matrix
        // ---------------------

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> c_frag; 

        // load Data for first iteration
        if(threadIdx.y >> 3 == 0){ //threadIdx.y/8 --> warp 0-7
            // two warps per feature line for I
            double2 *const I2s = reinterpret_cast<double2 *>(&Is[(threadIdx.y >> 1)*BLOCK_OFF + (id_1d & 63) * 2]);
            const double2 *const D_i2 = reinterpret_cast<const double2 *>(&in[(threadIdx.y >> 1) * points + (id_1d & 63) * 2 + i]);
            ::cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
            // printf("Indizes - in %i from %i value: %f \n", (threadIdx.y >> 1)*BLOCK_OFF + (32 *(threadIdx.y & 1) + threadIdx.x) * 2, (threadIdx.y >> 1) * points + (32 * (threadIdx.y & 1) + threadIdx.x) * 2 + i, in[(threadIdx.y >> 1) * points + (32 * (threadIdx.y >> 1) + threadIdx.x) * 2 + i]);     
        } else{ // warp 8-15
            // two warps per feature line for J
            double2 *const J2s = reinterpret_cast<double2 *>(&Js[((threadIdx.y - 8) >> 1)*BLOCK_OFF + (id_1d & 63) * 2]);
            const double2 *const D_j2 = reinterpret_cast<const double2 *>(&in[((threadIdx.y - 8) >> 1) * points + (id_1d & 63) * 2 + j]);
            ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]);
        }

        // loop over features
        for(int feature_it = 4; feature_it < feature_range; feature_it = feature_it +4){
            // wait for new I-data to be loaded into shared memory 
            bar[0].arrive_and_wait();
            // printf("Hi, %i %i \n", threadIdx.y, threadIdx.x);

            // load matrix_frag A
            double *const shmem_i_ptr = (double*)&Is[0] + 8 * threadIdx.y;
            nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);

            // bar[1].arrive_and_wait();

            for(int j_roll = 0; j_roll < ROLL_SIZE_HALF; ++j_roll){ // first half of tensor operations - full i times half j
                // load matrix_frag B
                double *const shmem_j_ptr = (double*)&Js[0] + 8 * j_roll; // shuffle for better access?
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

                // load solution part acc_frag C from shared
                double *const shmem_m_ptr = (double*)&solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * j_roll;
                nvcuda::wmma::load_matrix_sync(c_frag, shmem_m_ptr, BLOCK_OFF, nvcuda::wmma::mem_row_major);
                // tensorcore operation, C += A*B
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                // store solution part acc_frag C back in shared             
                nvcuda::wmma::store_matrix_sync(shmem_m_ptr , c_frag, BLOCK_OFF, nvcuda::wmma::mem_row_major);    
            }

            bar[1].arrive_and_wait();

            // printf("Hi after \n");

            // load I and first 8 J parts for next iteration
            if(threadIdx.y >> 3 == 1){ // warp 8-15
                // two warps per feature line for I
                double2 *const I2s = reinterpret_cast<double2 *>(&Is[((threadIdx.y - 8 ) >> 1) * BLOCK_OFF + (id_1d & 63) * 2]);
                const double2 *const D_i2 = reinterpret_cast<const double2 *>(&in[((threadIdx.y - 8 ) >> 1) * points + (id_1d & 63) * 2 + feature_it * points + i]);
                ::cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]); 
            } else if(threadIdx.y >> 2 == 0){ // warp 0-3
                // one warp per half feature line for J
                double2 *const J2s = reinterpret_cast<double2 *>(&Js[threadIdx.y * BLOCK_OFF + 2 * threadIdx.x]);
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&in[threadIdx.y * points + 2 * threadIdx.x + feature_it * points + j]);
                ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]); 
            }
            
            for(int j_roll = 8; j_roll < ROLL_SIZE_HALF + 8; ++j_roll){ // second half of tensor operations - full i times half j
                // load matrix_frag B
                double *shmem_j_ptr = (double*)&Js[0] + 8 * j_roll; // shuffle for better access?
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

                // load solution part acc_frag C from shared
                double *shmem_m_ptr = (double*)&solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * j_roll;
                nvcuda::wmma::load_matrix_sync(c_frag, shmem_m_ptr, BLOCK_OFF, nvcuda::wmma::mem_row_major);
                // tensorcore operation, C += A*B
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                // store solution part acc_frag C back in shared             
                nvcuda::wmma::store_matrix_sync(shmem_m_ptr , c_frag, BLOCK_OFF, nvcuda::wmma::mem_row_major);    
            }

            if(threadIdx.y < 4){ // warp 0-3
                // one warp per half feature line for J
                double2 *const J2s = reinterpret_cast<double2 *>(&Js[threadIdx.y * BLOCK_OFF + 2 * (threadIdx.x + 32)]);
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&in[threadIdx.y * points + 2 * (threadIdx.x + 32) + feature_it * points + j]);
                ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[1]); 
            }
        }

        // wait for new I-data to be loaded into shared memory
        bar[0].arrive_and_wait();

        // load matrix_frag A
        const double *shmem_i_ptr = (double *)&Is[0] + 8 * threadIdx.y;
        nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);

        // bar[1].arrive_and_wait();

        for (int j_roll = 0; j_roll < ROLL_SIZE_HALF; ++j_roll)
        { // first half of tensor operations - full i times half j
            // load matrix_frag B
            double *shmem_j_ptr = (double *)&Js[0] + 8 * j_roll; // shuffle for better access?
            nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

            // load solution part acc_frag C from shared
            double *shmem_m_ptr = (double *)&solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * j_roll;
            nvcuda::wmma::load_matrix_sync(c_frag, shmem_m_ptr, BLOCK_OFF, nvcuda::wmma::mem_row_major);
            // tensorcore operation, C += A*B
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // stor solution part acc_frag C back in shared
            nvcuda::wmma::store_matrix_sync(shmem_m_ptr, c_frag, BLOCK_OFF, nvcuda::wmma::mem_row_major);
        }

        bar[1].arrive_and_wait();

        for (int j_roll = 8; j_roll < ROLL_SIZE_HALF + 8; ++j_roll)
        { // second half of tensor operations - full i times half j
            // load matrix_frag B
            double *shmem_j_ptr = (double *)&Js[0] + 8 * j_roll; // shuffle for better access?
            nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

            // load solution part acc_frag C from shared
            double *shmem_m_ptr = (double *)&solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * j_roll;
            nvcuda::wmma::load_matrix_sync(c_frag, shmem_m_ptr, BLOCK_OFF, nvcuda::wmma::mem_row_major);
            // tensorcore operation, C += A*B
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // stor solution part acc_frag C back in shared
            nvcuda::wmma::store_matrix_sync(shmem_m_ptr, c_frag, BLOCK_OFF, nvcuda::wmma::mem_row_major);
        }

        // Building Matrix done 
        // ---------------------


        // get vec in shared memory
        if(threadIdx.y < 2){    
            ::cuda::memcpy_async(&Vs[2*(id_1d)], &vec[j + 2*id_1d], sizeof(double2), bar[0]);   
        }
        
        if(id==0){
            if(threadIdx.y > 1 && threadIdx.y < 4){    
                ::cuda::memcpy_async(&Qis[2*(id_1d - 64)], &q[i + 2*(id_1d - 64)], sizeof(double2), bar[0]);   
            }
            if(threadIdx.y > 3 && threadIdx.y < 6){    
                ::cuda::memcpy_async(&Qjs[2*(id_1d - 128)], &q[j + 2*(id_1d - 128)], sizeof(double2), bar[0]);   
            }

            bar[0].arrive_and_wait();

            // offset - gemv
            if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){
                double sol_tmp = 0.0;
                for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                    const int index = (id_1d + store_it) & (BLOCK_SIZE - 1);
                    const double Qi = Qis[id_1d];
                    sol_tmp += (solution[id_1d*BLOCK_OFF + index] * gamma - Qi - Qjs[index] + QA_cost)* Vs[index];
                    // sol_tmp += solution[id_1d*offset + index] * Vs[index]; // Test!
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0) printf("Index: %i - Qi: %f - Qj: %f\n", index, Qi, Qjs[index]);
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0) printf("Index: %i - sol_tmp: %f \n", index, sol_tmp);
                }
                if(i==j){
                    // int index = id_1d * 97; // offset + 1
                    sol_tmp += cost * Vs[id_1d];
                }
                atomicAdd(&out[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if(i>j){
                if(threadIdx.y < 2){    
                    ::cuda::memcpy_async(&Vs[2*id_1d], &vec[i + 2*id_1d], sizeof(double2), bar[0]);   
                }
                bar[0].arrive_and_wait();
                
                // no offset - gemv
                if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){
                    double sol_tmp = 0.0;
                    const double Qj = Qjs[id_1d];
                    for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                        sol_tmp += (solution[id_1d + store_it * BLOCK_OFF] * gamma - Qj - Qis[store_it] + QA_cost)* Vs[store_it];
                        // sol_tmp += solution[id_1d + store_it * offset] * Vs[store_it]; // Test!
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 1 && blockIdx.y == 0) printf("Index: %i - Qj: %f - Qis: %f\n", store_it, Qj, Qis[store_it]);
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                    }
                    atomicAdd(&out[j + id_1d], sol_tmp *add);
                }
            }
        } else {
            bar[0].arrive_and_wait();
           // offset - gemv
            if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){
                double sol_tmp = 0.0;
                for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                    const int index = (id_1d + store_it) & (BLOCK_SIZE - 1);
                    sol_tmp += solution[id_1d*BLOCK_OFF + index] * gamma * Vs[index];
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                }
                atomicAdd(&out[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if(i>j){
                if(threadIdx.y < 2){    
                    ::cuda::memcpy_async(&Vs[2*id_1d], &vec[i + 2*id_1d], sizeof(double2), bar[0]);   
                }
                bar[0].arrive_and_wait();
                
                // no offset - gemv
                if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){
                    double sol_tmp = 0.0;
                    for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                        sol_tmp += solution[id_1d + store_it * BLOCK_OFF] * gamma * Vs[store_it];
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                    }
                    atomicAdd(&out[j + id_1d], sol_tmp *add);
                }
            }
        }
    }
} */

__global__ void device_kernel_linear_t(const double *q, double *out, const double* vec, const double *in, const double QA_cost, const double cost, const int points, const int feature_range, const double add, const double gamma, const int id) {
    const int i = blockIdx.x * BLOCK_SIZE;
    const int j = blockIdx.y * BLOCK_SIZE;
    if(i >= j){
        const int id_1d = threadIdx.y * 32 + threadIdx.x;
        __shared__ ::cuda::barrier<::cuda::thread_scope_block> bar[1];

        extern __shared__ double solution[];
        double *Is= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF;
        double *Js= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 8 *BLOCK_OFF;
        double *Vs= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 16 *BLOCK_OFF;
        double *Qis= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 17*BLOCK_OFF;
        double *Qjs= (double*)&solution[0] + BLOCK_SIZE*BLOCK_OFF + 18*BLOCK_OFF;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            init(&bar[0], THREADS_PER_BLOCK);
            // init(&bar[1], THREADS_PER_BLOCK);
            // init(&bar[2], THREADS_PER_BLOCK);
        }

        // reset shared memory
        for(int set_zero = 0; set_zero < 36; ++set_zero){
            const int index = id_1d + set_zero * THREADS_PER_BLOCK;
            if(index < 18880) solution[index] = 0.0;
        }
        __syncthreads();

        // Build Matrix
        // ---------------------

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> c_frag; 

        // load Data for first iteration
        if(threadIdx.y >> 3 == 0){ //threadIdx.y/8 --> warp 0-7
            // two warps per feature line for I
            double2 *const I2s = reinterpret_cast<double2 *>(&Is[(threadIdx.y >> 1)*BLOCK_OFF + (id_1d & 63) * 2]);
            const double2 *const D_i2 = reinterpret_cast<const double2 *>(&in[(threadIdx.y >> 1) * points + (id_1d & 63) * 2 + i]);
            ::cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
            // printf("Indizes - in %i from %i value: %f \n", (threadIdx.y >> 1)*BLOCK_OFF + (32 *(threadIdx.y & 1) + threadIdx.x) * 2, (threadIdx.y >> 1) * points + (32 * (threadIdx.y & 1) + threadIdx.x) * 2 + i, in[(threadIdx.y >> 1) * points + (32 * (threadIdx.y >> 1) + threadIdx.x) * 2 + i]);     
        } else{ // warp 8-15
            // two warps per feature line for J
            double2 *const J2s = reinterpret_cast<double2 *>(&Js[((threadIdx.y - 8) >> 1)*BLOCK_OFF + (id_1d & 63) * 2]);
            const double2 *const D_j2 = reinterpret_cast<const double2 *>(&in[((threadIdx.y - 8) >> 1) * points + (id_1d & 63) * 2 + j]);
            ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]);
        }

        // loop over features
        for(int feature_it = 4; feature_it < feature_range; feature_it = feature_it +4){
            const int off_plus = (feature_it & 4)*BLOCK_OFF;
            const int off_minus = abs(off_plus - 4 * BLOCK_OFF);

            // wait for new I-data to be loaded into shared memory 
            bar[0].arrive_and_wait();

            if(threadIdx.y >> 3 == 0){ //threadIdx.y/8 --> warp 0-7
                // two warps per feature line for I
                double2 *const I2s = reinterpret_cast<double2 *>(&Is[(threadIdx.y >> 1)*BLOCK_OFF + (id_1d & 63) * 2 + off_plus]);
                const double2 *const D_i2 = reinterpret_cast<const double2 *>(&in[(threadIdx.y >> 1) * points + (id_1d & 63) * 2 + feature_it * points + i]);
                ::cuda::memcpy_async(I2s, D_i2, sizeof(double2), bar[0]);
                // printf("Indizes - in %i from %i value: %f \n", (threadIdx.y >> 1)*BLOCK_OFF + (32 *(threadIdx.y & 1) + threadIdx.x) * 2, (threadIdx.y >> 1) * points + (32 * (threadIdx.y & 1) + threadIdx.x) * 2 + i, in[(threadIdx.y >> 1) * points + (32 * (threadIdx.y >> 1) + threadIdx.x) * 2 + i]);     
            } else{ // warp 8-15
                // two warps per feature line for J
                double2 *const J2s = reinterpret_cast<double2 *>(&Js[((threadIdx.y - 8) >> 1)*BLOCK_OFF + (id_1d & 63) * 2 + off_plus]);
                const double2 *const D_j2 = reinterpret_cast<const double2 *>(&in[((threadIdx.y - 8) >> 1) * points + (id_1d & 63) * 2 + feature_it * points + j]);
                ::cuda::memcpy_async(J2s, D_j2, sizeof(double2), bar[0]);
            }

            // load matrix_frag A
            double *const shmem_i_ptr = (double*)&Is[0] + 8 * threadIdx.y + off_minus;
            nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);

            for(int j_roll = 0; j_roll < ROLL_SIZE; ++j_roll){ // first half of tensor operations - full i times half j
                // load matrix_frag B
                double *const shmem_j_ptr = (double*)&Js[0] + 8 * j_roll + off_minus; // shuffle for better access?
                nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

                // load solution part acc_frag C from shared
                double *const shmem_m_ptr = (double*)&solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * j_roll;
                nvcuda::wmma::load_matrix_sync(c_frag, shmem_m_ptr, BLOCK_OFF, nvcuda::wmma::mem_row_major);
                // tensorcore operation, C += A*B
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                // store solution part acc_frag C back in shared             
                nvcuda::wmma::store_matrix_sync(shmem_m_ptr , c_frag, BLOCK_OFF, nvcuda::wmma::mem_row_major);    
            }
        }

        // wait for new I-data to be loaded into shared memory
        bar[0].arrive_and_wait();
        const int off_last = ((feature_range - 4) & 4)*BLOCK_OFF;

        // load matrix_frag A
        const double *shmem_i_ptr = (double *)&Is[0] + 8 * threadIdx.y + off_last;
        nvcuda::wmma::load_matrix_sync(a_frag, shmem_i_ptr, BLOCK_OFF);

        for (int j_roll = 0; j_roll < ROLL_SIZE; ++j_roll)
        { // first half of tensor operations - full i times half j
            // load matrix_frag B
            double *shmem_j_ptr = (double *)&Js[0] + 8 * j_roll + off_last; // shuffle for better access?
            nvcuda::wmma::load_matrix_sync(b_frag, shmem_j_ptr, BLOCK_OFF);

            // load solution part acc_frag C from shared
            double *shmem_m_ptr = (double *)&solution[0] + 8 * threadIdx.y * BLOCK_OFF + 8 * j_roll;
            nvcuda::wmma::load_matrix_sync(c_frag, shmem_m_ptr, BLOCK_OFF, nvcuda::wmma::mem_row_major);
            // tensorcore operation, C += A*B
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            // stor solution part acc_frag C back in shared
            nvcuda::wmma::store_matrix_sync(shmem_m_ptr, c_frag, BLOCK_OFF, nvcuda::wmma::mem_row_major);
        }

        // Building Matrix done 
        // ---------------------


        // get vec in shared memory
        if(threadIdx.y < 2){    
            ::cuda::memcpy_async(&Vs[2*(id_1d)], &vec[j + 2*id_1d], sizeof(double2), bar[0]);   
        }
        
        if(id==0){
            if(threadIdx.y > 1 && threadIdx.y < 4){    
                ::cuda::memcpy_async(&Qis[2*(id_1d - 64)], &q[i + 2*(id_1d - 64)], sizeof(double2), bar[0]);   
            }
            if(threadIdx.y > 3 && threadIdx.y < 6){    
                ::cuda::memcpy_async(&Qjs[2*(id_1d - 128)], &q[j + 2*(id_1d - 128)], sizeof(double2), bar[0]);   
            }

            bar[0].arrive_and_wait();

            // offset - gemv
            if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){ // 128/32=4 --> Warp 0-3
                double sol_tmp = 0.0;
                // each of the 128 Threads iterate over 1 row in the 128x128 Block. 
                for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                    const int index = (id_1d + store_it) & (BLOCK_SIZE - 1); // modulo 128
                    const double Qi = Qis[id_1d];
                    sol_tmp += (solution[id_1d*BLOCK_OFF + index] * gamma - Qi - Qjs[index] + QA_cost)* Vs[index];
                    // sol_tmp += solution[id_1d*offset + index] * Vs[index]; // Test!
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0) printf("Index: %i - Qi: %f - Qj: %f\n", index, Qi, Qjs[index]);
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0) printf("Index: %i - sol_tmp: %f \n", index, sol_tmp);
                }
                if(i==j){
                    // int index = id_1d * 97; // offset + 1
                    sol_tmp += cost * Vs[id_1d];
                }
                atomicAdd(&out[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if(i>j){
                if(threadIdx.y < 2){    
                    ::cuda::memcpy_async(&Vs[2*id_1d], &vec[i + 2*id_1d], sizeof(double2), bar[0]);   
                }
                bar[0].arrive_and_wait();
                
                // no offset - gemv
                if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){ // 128/32=4 --> Warp 0-3
                    double sol_tmp = 0.0;
                    const double Qj = Qjs[id_1d];
                    for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                        sol_tmp += (solution[id_1d + store_it * BLOCK_OFF] * gamma - Qj - Qis[store_it] + QA_cost)* Vs[store_it];
                        // sol_tmp += solution[id_1d + store_it * offset] * Vs[store_it]; // Test!
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 1 && blockIdx.y == 0) printf("Index: %i - Qj: %f - Qis: %f\n", store_it, Qj, Qis[store_it]);
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                    }
                    atomicAdd(&out[j + id_1d], sol_tmp *add);
                }
            }
        } else {
            bar[0].arrive_and_wait();
           // offset - gemv
            if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){
                double sol_tmp = 0.0;
                for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                    const int index = (id_1d + store_it) & (BLOCK_SIZE - 1);
                    sol_tmp += solution[id_1d*BLOCK_OFF + index] * gamma * Vs[index];
                    // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                }
                atomicAdd(&out[i + id_1d], sol_tmp * add);
            }

            __syncthreads();

            // upper triangular
            if(i>j){
                if(threadIdx.y < 2){    
                    ::cuda::memcpy_async(&Vs[2*id_1d], &vec[i + 2*id_1d], sizeof(double2), bar[0]);   
                }
                bar[0].arrive_and_wait();
                
                // no offset - gemv
                if(threadIdx.y < BLOCK_SIZE/WARP_SIZE){
                    double sol_tmp = 0.0;
                    for(int store_it = 0; store_it < BLOCK_SIZE; ++store_it){
                        sol_tmp += solution[id_1d + store_it * BLOCK_OFF] * gamma * Vs[store_it];
                        // if(threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 1) printf("sol_tmp: %f - sol: %f - Vs %f \n", sol_tmp, solution[id_1d*offset + store_it], Vs[store_it]);
                    }
                    atomicAdd(&out[j + id_1d], sol_tmp *add);
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