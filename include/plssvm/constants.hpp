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

#define WARPS_PER_BLOCK 12
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define BLOCK_SIZE 96
#define BLOCK_OFF 100
#define ROLL_SIZE 12

#define WARPS_PER_BLOCK_F 8
#define THREADS_PER_BLOCK_F (WARP_SIZE * WARPS_PER_BLOCK_F)
#define BLOCK_SIZE_F 128
#define BLOCK_OFF_F 136
#define ROLL_SIZE_F 8

// Recommended settings: 
// define TENSOR for Ampere GPUs or newer
// define MIXED only for Desktop GPUs
// don't use POLAK-RIBIERE without reliable_updates

// if defined, tensor-Kernels are getting used
#define TENSOR
// if defined, mixed precision is getting used
// #define MIXED
// if defined, beta will be calclulated after polak-ribiere
// #define POLAK_RIBIERE


// For testing only
// if defined always run single gpu 
// #define SINGLE_TEST
// define the number of runs for your test
// #define RUNTIME_TEST 0

enum class correction_scheme {
    /** The default corection_scheme: none */
    zero,
    /** The one used in PLSSVM current main build. */
    NewRScheme,
    /** ReliableUpdate. */
    ReliableUpdate,
};

// Choose one correction_scheme from correction_scheme, defualt is zero.
constexpr correction_scheme CORRECTION_SCHEME = correction_scheme::zero;

// perform sanity checks
static_assert(THREAD_BLOCK_SIZE > 0, "THREAD_BLOCK_SIZE must be greater than 0!");
static_assert(INTERNAL_BLOCK_SIZE > 0, "INTERNAL_BLOCK_SIZE must be greater than 0!");
static_assert(OPENMP_BLOCK_SIZE > 0, "OPENMP_BLOCK_SIZE must be greater than 0!");

}  // namespace plssvm