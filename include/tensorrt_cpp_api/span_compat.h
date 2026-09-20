#pragma once

#include <cstddef>
#include <type_traits>

// CUDA 11.8 nvcc parses host headers in C++17 mode and MSVC does not expose
// std::span until C++20. The CUDA preprocessing target only needs the small
// non-owning subset below; host C++20 builds use the standard implementation.
#if defined(__CUDACC__) && __cplusplus < 202002L
namespace std {

template <class T> class span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using pointer = T *;
    using iterator = pointer;

    constexpr span() noexcept = default;
    constexpr span(pointer data, std::size_t size) noexcept : data_(data), size_(size) {}

    constexpr pointer data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr iterator begin() const noexcept { return data_; }
    constexpr iterator end() const noexcept { return data_ + size_; }
    constexpr T &operator[](std::size_t index) const noexcept { return data_[index]; }

private:
    pointer data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace std
#endif

