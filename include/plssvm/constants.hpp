/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Global type definitions and compile-time constants.
 */

#pragma once

namespace plssvm {

/// Integer type used inside kernels.
using kernel_index_type = int;
using real_type = double;

/// Global compile-time constant used for internal caching.
#if defined(PLSSVM_THREAD_BLOCK_SIZE)
constexpr kernel_index_type THREAD_BLOCK_SIZE = PLSSVM_THREAD_BLOCK_SIZE;
#else
constexpr kernel_index_type THREAD_BLOCK_SIZE = 16;
#endif

/// Global compile-time constant used for internal caching.
#if defined(PLSSVM_INTERNAL_BLOCK_SIZE)
constexpr kernel_index_type INTERNAL_BLOCK_SIZE = PLSSVM_INTERNAL_BLOCK_SIZE;
#else
constexpr kernel_index_type INTERNAL_BLOCK_SIZE = 6;
#endif

/// Global compile-time constant used for internal caching in the OpenMP kernel.
#if defined(PLSSVM_OPENMP_BLOCK_SIZE)
constexpr kernel_index_type OPENMP_BLOCK_SIZE = PLSSVM_OPENMP_BLOCK_SIZE;
#else
constexpr kernel_index_type OPENMP_BLOCK_SIZE = 64;
#endif

// Test CUDA Defines
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8 //16
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define BLOCK_SIZE 64 // 128
#define BLOCK_OFF 72 //136
#define ROLL_SIZE 8 // 16
#define ROLL_SIZE_HALF 8

#define MIXED 0
#define POLAK_RIBIERE 0

// perform sanity checks
static_assert(THREAD_BLOCK_SIZE > 0, "THREAD_BLOCK_SIZE must be greater than 0!");
static_assert(INTERNAL_BLOCK_SIZE > 0, "INTERNAL_BLOCK_SIZE must be greater than 0!");
static_assert(OPENMP_BLOCK_SIZE > 0, "OPENMP_BLOCK_SIZE must be greater than 0!");

}  // namespace plssvm