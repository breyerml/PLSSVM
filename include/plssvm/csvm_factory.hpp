/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Factory function for constructing a new C-SVM using one of the available backends based on the provided command line arguments.
 */

#pragma once

#include "plssvm/parameter.hpp"
#include "plssvm/backend_types.hpp"          // plssvm::backend
#include "plssvm/csvm.hpp"                   // plssvm::csvm
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

// only include requested/available backends
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include "plssvm/backends/OpenMP/csvm.hpp"  // plssvm::openmp::csvm
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include "plssvm/backends/CUDA/csvm.hpp"  // plssvm::cuda::csvm
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    #include "plssvm/backends/HIP/csvm.hpp"  // plssvm::hip::csvm
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "plssvm/backends/OpenCL/csvm.hpp"  // plssvm::opencl::csvm
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    #include "plssvm/backends/SYCL/implementation_type.hpp"
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
        #include "plssvm/backends/autogenerated/DPCPP/csvm.hpp"  // plssvm::dpcpp::csvm
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
        #include "plssvm/backends/autogenerated/hipSYCL/csvm.hpp"  // plssvm::hipsycl::csvm
    #endif
#endif

#include "igor/igor.hpp"

#include <memory>   // std::unique_ptr, std::make_unique
#include <utility>  // std::forward

namespace plssvm {

/**
 * @brief Construct a new C-SVM with the parameters given through @p params using the requested backend.
 * @tparam T the type of the data
 * @param[in] params class encapsulating all possible parameters
 * @throws plssvm::unsupported_backend_exception if the requested backend isn't available
 * @return [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr) to the constructed C-SVM (`[[nodiscard]]`)
 */

// make_csvm using parameter and sycl_parameter struct
template <typename T, typename... Args>
[[nodiscard]] std::unique_ptr<csvm<T>> make_csvm(const backend_type backend, const target_platform target, const parameter<T> params, Args&&... args) {
    return make_csvm<T>(backend, target, params.kernel,
                        plssvm::degree = params.degree,
                        plssvm::gamma = params.gamma,
                        plssvm::coef0 = params.coef0,
                        plssvm::cost = params.cost,
                        std::forward<Args>(args)...);
}

// make_csvm using igor flags --X
template <typename T, typename... Args>
[[nodiscard]] std::unique_ptr<csvm<T>> make_csvm(const backend_type backend, const target_platform target, const kernel_type kernel, Args&&... args) {
    // check igor parameter
    igor::parser p{ args... };
    // compile time check: only named parameter are permitted
    static_assert(!p.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!p.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!p.has_other_than(gamma, degree, coef0, cost, sycl_implementation_type, sycl_kernel_invocation_type), "An illegal named parameter has been passed!");


    switch (backend) {
        case backend_type::automatic:
            return make_csvm<T>(determine_default_backend(), target, kernel, std::forward<Args>(args)...);
        case backend_type::openmp:
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
            return std::make_unique<openmp::csvm<T>>(target, kernel, std::forward<Args>(args)...);
#else
            throw unsupported_backend_exception{ "No OpenMP backend available!" };
#endif

        case backend_type::cuda:
#if defined(PLSSVM_HAS_CUDA_BACKEND)
            return std::make_unique<cuda::csvm<T>>(target, kernel, std::forward<Args>(args)...);
#else
            throw unsupported_backend_exception{ "No CUDA backend available!" };
#endif

        case backend_type::hip:
#if defined(PLSSVM_HAS_HIP_BACKEND)
            return std::make_unique<hip::csvm<T>>(target, kernel, std::forward<Args>(args)...);
#else
            throw unsupported_backend_exception{ "No HIP backend available!" };
#endif

        case backend_type::opencl:
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
            return std::make_unique<opencl::csvm<T>>(target, kernel, std::forward<Args>(args)...);
#else
            throw unsupported_backend_exception{ "No OpenCL backend available!" };
#endif

        case backend_type::sycl:
#if defined(PLSSVM_HAS_SYCL_BACKEND)
            {
                sycl_generic::implementation_type impl_type = sycl_generic::implementation_type::automatic;
                // check whether a specific SYCL implementation type has been requested
                if constexpr (p.has(sycl_implementation_type)) {
                    // compile time check: the value must have the correct type
                    static_assert(std::is_same_v<detail::remove_cvref_t<decltype(p(sycl_implementation_type))>, sycl_generic::implementation_type>, "sycl_implementation_type must be convertible to a plssvm::sycl::implementation_type!");
                    impl_type = static_cast<sycl_generic::implementation_type>(p(sycl_implementation_type));
                }
                switch (impl_type) {
                    case sycl_generic::implementation_type::automatic:
                        return std::make_unique<PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION::csvm<T>>(target, kernel, std::forward<Args>(args)...);
                    case sycl_generic::implementation_type::dpcpp:
#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
                        return std::make_unique<dpcpp::csvm<T>>(target, kernel, std::forward<Args>(args)...);
#else
                        throw unsupported_backend_exception{ "No SYCL backend using DPC++ available!" };
#endif
                    case sycl_generic::implementation_type::hipsycl:
#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
                        return std::make_unique<hipsycl::csvm<T>>(target, kernel, std::forward<Args>(args)...);
#else
                        throw unsupported_backend_exception{ "No SYCL backend using hipSYCL available!" };
#endif
                }
            }
#else
            throw unsupported_backend_exception{ "No SYCL backend available!" };
#endif
    }
    throw unsupported_backend_exception{ "Can't recognize backend !" };
}

}  // namespace plssvm
