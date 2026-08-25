#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
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

struct CudaDim3 {
  std::uint32_t x = 1;
  std::uint32_t y = 1;
  std::uint32_t z = 1;
};

struct CudaLaunchConfig {
  CudaDim3 grid;
  CudaDim3 block;
  std::uint32_t shared_memory_bytes = 0;
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

class CudaKernel;

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
  [[nodiscard]] CudaKernel load_kernel_from_ptx(std::string_view ptx,
                                                std::string_view function_name) const;
  void synchronize() const;

private:
  friend class CudaDeviceBuffer;
  friend class CudaKernel;

  struct Impl;

  explicit CudaBackend(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

class CudaKernel {
public:
  CudaKernel() = default;
  CudaKernel(const CudaKernel&) = delete;
  CudaKernel& operator=(const CudaKernel&) = delete;
  CudaKernel(CudaKernel&& other) noexcept;
  CudaKernel& operator=(CudaKernel&& other) noexcept;
  ~CudaKernel();

  [[nodiscard]] explicit operator bool() const {
    return context_ != nullptr && module_ != nullptr && function_ != nullptr;
  }
  void launch(const CudaLaunchConfig& config, std::span<void*> args) const;
  void synchronize() const;
  void reset();

private:
  friend class CudaBackend;

  CudaKernel(std::shared_ptr<CudaBackend::Impl> backend, void* context, void* module, void* function);

  std::shared_ptr<CudaBackend::Impl> backend_;
  void* context_ = nullptr;
  void* module_ = nullptr;
  void* function_ = nullptr;
};

} // namespace reco::core
