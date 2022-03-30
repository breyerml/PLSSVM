/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Defines a very small RAII wrapper around a cl_context including all associated devices and one command queue per device.
*/

#pragma once

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id, clReleaseCommandQueue

#include <iostream>
#include <vector>  // std::vector

namespace plssvm::opencl::detail {

class context {
  public:
    /**
     * @brief Empty default construct context.
     */
    context() = default;
    /**
     * @brief Construct a new OpenCL context.
     * @param[in] device_context the associated OpenCL cl_context
     */
    context(cl_context device_context, std::vector<cl_device_id> devices);

    /**
     * @brief Delete copy-constructor to make context a move only type.
     */
    context(const context &) = delete;
    /**
     * @brief Move-constructor as context is a move-only type.
     * @param[in,out] other the context to move the resources from
     */
    context(context &&other) noexcept;
    /**
     * @brief Delete copy-assignment-operator to make context a move only type.
     */
    context &operator=(const context &) = delete;
    /**
     * @brief Move-assignment-operator as context is a move-only type.
     * @param[in,out] other the context to move the resources from
     * @return `*this`
     */
    context &operator=(context &&other);

    /**
     * @brief Release the context resources on destruction.
     */
    ~context();

    /// The OpenCL context associated with the platform containing the respective devices.
    cl_context device_context{};
    /// All devices associated with this context.
    std::vector<cl_device_id> devices{};
    /// One OpenCL command queue for each device associated with this context.
    std::vector<cl_command_queue> queues{};
};

inline std::ostream &operator<<(std::ostream &out, const context &cont) {
    out << (void *) cont.device_context << " " << cont.devices.size() << ": ";
    for (const cl_device_id dev : cont.devices) {
        out << (void *) dev << " ";
    }
    return out;
}

}  // namespace plssvm::opencl::detail