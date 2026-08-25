#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace reco::core {

using CudaDevicePtr = std::uint64_t;

struct CudaDeviceInfo {
  int ordinal = 0;
  std::string name;
  std::array<std::uint8_t, 16> uuid{};
};

struct CudaMemoryInfo {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

struct Cuda2DCopy {
  CudaDevicePtr src = 0;
  std::size_t src_pitch = 0;
  CudaDevicePtr dst = 0;
  std::size_t dst_pitch = 0;
  std::size_t width_bytes = 0;
  std::size_t height = 0;
};

struct CudaHostToDevice2DCopy {
  const void* src = nullptr;
  std::size_t src_pitch = 0;
  CudaDevicePtr dst = 0;
  std::size_t dst_pitch = 0;
  std::size_t width_bytes = 0;
  std::size_t height = 0;
};

struct CudaDeviceToHost2DCopy {
  void* dst = nullptr;
  std::size_t dst_pitch = 0;
  CudaDevicePtr src = 0;
  std::size_t src_pitch = 0;
  std::size_t width_bytes = 0;
  std::size_t height = 0;
};

class CudaDeviceBuffer {
public:
  CudaDeviceBuffer() = default;
  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
  CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;
  ~CudaDeviceBuffer();

  [[nodiscard]] CudaDevicePtr ptr() const { return ptr_; }
  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] explicit operator bool() const { return ptr_ != 0; }
  void reset();

private:
  friend class CudaBackend;

  CudaDeviceBuffer(CudaDevicePtr ptr, std::size_t size, std::function<void(CudaDevicePtr)> free);

  CudaDevicePtr ptr_ = 0;
  std::size_t size_ = 0;
  std::function<void(CudaDevicePtr)> free_;
};

class CudaBackend {
public:
  [[nodiscard]] static bool is_available();
  [[nodiscard]] static std::string availability_error();
  [[nodiscard]] static CudaBackend create();

  [[nodiscard]] int device_count() const;
  [[nodiscard]] CudaDeviceInfo device_info(int ordinal = 0) const;
  void ensure_primary_context(int ordinal = 0) const;
  [[nodiscard]] CudaMemoryInfo memory_info() const;
  [[nodiscard]] CudaDeviceBuffer allocate(std::size_t bytes) const;
  void memset_d8(const CudaDeviceBuffer& buffer, std::uint8_t value) const;
  [[nodiscard]] std::vector<std::uint8_t> copy_to_host(const CudaDeviceBuffer& buffer) const;
  void copy_host_to_device_2d(const CudaHostToDevice2DCopy& copy) const;
  void copy_device_to_device_2d(const Cuda2DCopy& copy) const;
  void copy_device_to_host_2d(const CudaDeviceToHost2DCopy& copy) const;
  void synchronize() const;

private:
  friend class CudaDeviceBuffer;

  struct Impl;

  explicit CudaBackend(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

} // namespace reco::core
