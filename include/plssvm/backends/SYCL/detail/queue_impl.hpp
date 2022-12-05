/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief PImpl implementation encapsulating a single ::sycl::queue.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_IMPL_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_IMPL_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/queue.hpp"  // plssvm::sycl::detail::queue

#include "sycl/sycl.hpp"  // sycl::queue

#include <utility>  // std::forward

namespace plssvm::sycl::detail {

/**
 * @brief The PImpl implementation struct encapsulating a single ::sycl::queue.
 */
struct queue::queue_impl {
    /**
     * @brief Construct a ::sycl::queue by forwarding all parameters in @p args.
     * @tparam Args the type of the parameters
     * @param[in] args the parameters to construct a ::sycl::queue
     */
    template <typename... Args>
    queue_impl(Args... args) :
        sycl_queue{ std::forward<Args>(args)... } {}

    /// The wrapped ::sycl::queue.
    ::sycl::queue sycl_queue;
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_IMPL_HPP_
