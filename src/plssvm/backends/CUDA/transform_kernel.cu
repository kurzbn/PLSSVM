#include "plssvm/backends/CUDA/transform_kernel.cuh"

#include "plssvm/constants.hpp"                     // plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

namespace plssvm::cuda {

__global__ void device_kernel_cast_double_to_float(const double *in_d, float *out_d, int size_d) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size_d) {
        out_d[index] = __double2float_rn(in_d[index]);
    }
}

__global__ void device_kernel_cast_float_to_double(const float *in_d, double *out_d, int size_d) {
    const kernel_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size_d) {
        out_d[index] = static_cast<double>(in_d[index]);
    }
}

}  // namespace plssvm::cuda