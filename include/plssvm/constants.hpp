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

#ifndef PLSSVM_CONSTANTS_HPP_
#define PLSSVM_CONSTANTS_HPP_
#pragma once

#include "plssvm/detail/type_list.hpp"  // plssvm::detail::type_list_contains_v

#include <type_traits>  // std::is_same_v

namespace plssvm {

/// The used floating point type. May be changed during the CMake configuration step.
#if defined(PLSSVM_FLOAT_AS_REAL_TYPE)
using real_type = float;
#else
using real_type = double;
#endif

/// Global compile-time constant used for internal thread block caching. May be changed during the CMake configuration step.
#if defined(PLSSVM_THREAD_BLOCK_SIZE)
constexpr unsigned long long THREAD_BLOCK_SIZE = PLSSVM_THREAD_BLOCK_SIZE;
#else
constexpr unsigned THREAD_BLOCK_SIZE = 8;
#endif

/// Global compile time constant used for internal feature caching. May be changed during the CMake configuration step.
#if defined(PLSSVM_FEATURE_BLOCK_SIZE)
constexpr unsigned long long FEATURE_BLOCK_SIZE = PLSSVM_FEATURE_BLOCK_SIZE;
#else
constexpr unsigned FEATURE_BLOCK_SIZE = 16;
#endif

/// Global compile-time constant used for internal caching. May be changed during the CMake configuration step.
#if defined(PLSSVM_INTERNAL_BLOCK_SIZE)
constexpr unsigned long long INTERNAL_BLOCK_SIZE = PLSSVM_INTERNAL_BLOCK_SIZE;
#else
constexpr unsigned INTERNAL_BLOCK_SIZE = 4;
#endif

/// Global compile-time constant used for internal caching in the OpenMP kernel. May be changed during the CMake configuration step.
#if defined(PLSSVM_OPENMP_BLOCK_SIZE)
constexpr unsigned long long OPENMP_BLOCK_SIZE = PLSSVM_OPENMP_BLOCK_SIZE;
#else
constexpr unsigned long long OPENMP_BLOCK_SIZE = 64;  // TODO: use again?!
#endif

// perform sanity checks
static_assert(detail::type_list_contains_v<real_type, detail::real_type_list>, "Illegal real type provided! See the 'real_type_list' in the type_list.hpp header for a list of the allowed types.");
static_assert(THREAD_BLOCK_SIZE > 0, "THREAD_BLOCK_SIZE must be greater than 0!");
static_assert(FEATURE_BLOCK_SIZE > 0, "FEATURE_BLOCK_SIZE must be greater than 0!");
static_assert(FEATURE_BLOCK_SIZE == 2 * THREAD_BLOCK_SIZE, "Invalid FEATURE_BLOCK_SIZE or THREAD_BLOCK_SIZE!");
static_assert(INTERNAL_BLOCK_SIZE > 0, "INTERNAL_BLOCK_SIZE must be greater than 0!");
static_assert(OPENMP_BLOCK_SIZE > 0, "OPENMP_BLOCK_SIZE must be greater than 0!");

// TODO: restrictions/assumptions for BLOCK sizes!

}  // namespace plssvm

#endif  // PLSSVM_CONSTANTS_HPP_