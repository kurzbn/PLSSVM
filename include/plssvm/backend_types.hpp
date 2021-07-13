/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines all possible backends. Can also include backends not available on the target platform.
 */

#pragma once

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <algorithm>    // std::transform
#include <cctype>       // std::tolower
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string_view>  // std::string_view

namespace plssvm {

/**
 * @brief Enum class for the different backend types.
 */
enum class backend_type {
    /** [OpenMP](https://www.openmp.org/) */
    openmp = 0,
    /** [CUDA](https://developer.nvidia.com/cuda-zone) */
    cuda = 1,
    /** [OpenCL](https://www.khronos.org/opencl/) */
    opencl = 2
};

/**
 * @brief Stream-insertion-operator overload for convenient printing of the backend type @p backend.
 * @param[inout] out the output-stream to write the backend type to
 * @param[in] backend the backend type
 * @return the output-stream
 */
inline std::ostream &operator<<(std::ostream &out, const backend_type backend) {
    switch (backend) {
        case backend_type::openmp:
            return out << "OpenMP";
        case backend_type::cuda:
            return out << "CUDA";
        case backend_type::opencl:
            return out << "OpenCL";
        default:
            return out << "unknown";
    }
}

/**
 * @brief Stream-extraction-operator overload for convenient converting a string to a backend type.
 * @param[inout] in input-stream to extract the backend type from
 * @param[in] backend the backend type
 * @return the input-stream
 */
inline std::istream &operator>>(std::istream &in, backend_type &backend) {
    std::string str;
    in >> str;
    std::transform(str.begin(), str.end(), str.begin(), [](const char c) { return std::tolower(c); });

    if (str == "openmp") {
        backend = backend_type::openmp;
    } else if (str == "cuda") {
        backend = backend_type::cuda;
    } else if (str == "opencl") {
        backend = backend_type::opencl;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm
