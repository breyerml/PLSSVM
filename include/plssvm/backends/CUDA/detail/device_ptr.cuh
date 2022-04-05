/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Small wrapper around a CUDA device pointer.
 */

#pragma once

#include "plssvm/backends/gpu_device_ptr.hpp"  // plssvm::detail::gpu_device_ptr

namespace plssvm::cuda::detail {

/**
 * @brief Small wrapper class around a CUDA device pointer together with commonly used device functions.
 * @tparam T the type of the kernel pointer to wrap
 */
template <typename T>
class device_ptr : public ::plssvm::detail::gpu_device_ptr<T, int> {
    /// The template base type of the OpenCL device_ptr class.
    using base_type = ::plssvm::detail::gpu_device_ptr<T, int>;

    using base_type::data_;
    using base_type::queue_;
    using base_type::size_;

  public:
    // Be able to use overloaded base class functions.
    using base_type::memcpy_to_device;
    using base_type::memcpy_to_host;
    using base_type::memset;

    using typename base_type::const_host_pointer_type;
    using typename base_type::device_pointer_type;
    using typename base_type::host_pointer_type;
    using typename base_type::queue_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

    /**
     * @brief Default construct a CUDA device_ptr with a size of 0.
     * @details Always associated with device 0.
     */
    device_ptr() = default;
    /**
     * @brief Allocates `size * sizeof(T)` bytes on the device with ID @p device.
     * @param[in] size the number of elements represented by the device_ptr
     * @param[in] device the associated CUDA device
     * @throws plssvm::cuda::backend_exception if the given device ID is smaller than 0 or greater or equal than the available number of devices
     */
    explicit device_ptr(size_type size, int device = 0);

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::gpu_device_ptr(const gpu_device_ptr&)
     */
    device_ptr(const device_ptr &) = delete;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::gpu_device_ptr(gpu_device_ptr&&)
     */
    device_ptr(device_ptr &&other) noexcept = default;

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::operator=(const gpu_device_ptr&)
     */
    device_ptr &operator=(const device_ptr &) = delete;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::operator=(gpu_device_ptr&&)
     */
    device_ptr &operator=(device_ptr &&other) noexcept = default;

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::~gpu_device_ptr()
     */
    ~device_ptr();

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::memset(int, size_type, size_type)
     */
    void memset(int value, size_type pos, size_type count) override;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::memcpy_to_device(const_host_pointer_type, size_type, size_type)
     */
    void memcpy_to_device(const_host_pointer_type data_to_copy, size_type pos, size_type count) override;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::memcpy_to_host(host_pointer_type, size_type, size_type) const
     */
    void memcpy_to_host(host_pointer_type buffer, size_type pos, size_type count) const override;
};

extern template class device_ptr<float>;
extern template class device_ptr<double>;

}  // namespace plssvm::cuda::detail