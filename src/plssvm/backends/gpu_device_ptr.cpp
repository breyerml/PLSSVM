#include "plssvm/backends/gpu_device_ptr.hpp"

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    // used for explicitly instantiating the OpenCL backend
    #include "CL/cl.h"                                          // cl_mem
    #include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // used for explicitly instantiating the SYCL backend
    #include "sycl/sycl.hpp"
#endif

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::gpu_device_ptr_exception

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <utility>    // std::exchange, std::move, std::swap
#include <vector>     // std::vector

namespace plssvm::detail {

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(queue_type queue, size_type size) :
    queue_{ queue }, size_{ size } {}

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(gpu_device_ptr &&other) noexcept :
    queue_{ std::exchange(other.queue_, queue_type{}) },
    data_{ std::exchange(other.data_, device_pointer_type{}) },
    size_{ std::exchange(other.size_, size_type{}) } {}

template <typename T, typename queue_t, typename device_pointer_t>
auto gpu_device_ptr<T, queue_t, device_pointer_t>::operator=(gpu_device_ptr &&other) noexcept -> gpu_device_ptr & {
    // guard against self-assignment
    if (this != std::addressof(other)) {
        queue_ = std::exchange(other.queue_, queue_type{});
        data_ = std::exchange(other.data_, device_pointer_type{});
        size_ = std::exchange(other.size_, size_type{});
    }
    return *this;
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::swap(gpu_device_ptr &other) noexcept {
    std::swap(queue_, other.queue_);
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memset(const int value, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memset(value, pos, size_);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memcpy_to_device(const std::vector<value_type> &data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_device(data_to_copy, 0, size_);
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memcpy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    if (data_to_copy.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Too few data to perform memcpy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
    }
    this->memcpy_to_device(data_to_copy.data(), pos, rcount);
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memcpy_to_device(const_host_pointer_type data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_device(data_to_copy, 0, size_);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memcpy_to_host(std::vector<value_type> &buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memcpy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    if (buffer.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform memcpy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->memcpy_to_host(buffer.data(), pos, rcount);
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memcpy_to_host(host_pointer_type buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}

// explicitly instantiate template class depending on available backends
#if defined(PLSSVM_HAS_CUDA_BACKEND) || defined(PLSSVM_HAS_HIP_BACKEND)
template class gpu_device_ptr<float, int>;
template class gpu_device_ptr<double, int>;
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
template class gpu_device_ptr<float, ::plssvm::opencl::detail::command_queue *, cl_mem>;
template class gpu_device_ptr<double, ::plssvm::opencl::detail::command_queue *, cl_mem>;
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
template class gpu_device_ptr<float, ::sycl::queue *>;
template class gpu_device_ptr<double, ::sycl::queue *>;
#endif

}  // namespace plssvm::detail