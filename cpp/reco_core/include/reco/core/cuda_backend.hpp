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

#if defined(_WIN32)
using CudaShareableHandle = void*;
#else
using CudaShareableHandle = int;
#endif

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

/// CUDA device allocation whose row pitch is valid for two-dimensional copies.
struct CudaPitchedAllocation {
  /// Device storage spanning `pitch * height` bytes.
  CudaDeviceBuffer buffer;
  /// Driver-selected distance between adjacent rows, in bytes.
  std::size_t pitch = 0;
};

class CudaSharedMemory;
class CudaKernel;

/// Optional non-throwing trace sink for explicit CUDA backend operations.
class CudaBackendTraceSink {
public:
  virtual ~CudaBackendTraceSink() = default;
  /// Called after a device-to-device copy has been accepted by the CUDA driver.
  virtual void device_to_device_copy_submitted() noexcept {}
  /// Called after an explicit CUDA context synchronization succeeds.
  virtual void context_synchronized() noexcept {}
};

class CudaBackend {
public:
  [[nodiscard]] static bool is_available();
  [[nodiscard]] static std::string availability_error();
  [[nodiscard]] static CudaBackend create();
  /// Returns a backend view that reports explicit operations to `trace_sink`.
  [[nodiscard]] CudaBackend with_trace_sink(std::shared_ptr<CudaBackendTraceSink> trace_sink) const;

  [[nodiscard]] int device_count() const;
  [[nodiscard]] CudaDeviceInfo device_info(int ordinal = 0) const;
  void ensure_primary_context(int ordinal = 0) const;
  [[nodiscard]] CudaMemoryInfo memory_info() const;
  [[nodiscard]] CudaDeviceBuffer allocate(std::size_t bytes) const;
  /// Allocates 2D storage with a driver-selected pitch valid for `cuMemcpy2D`.
  [[nodiscard]] CudaPitchedAllocation allocate_pitched(std::size_t width_bytes, std::size_t height,
                                                       unsigned int element_size_bytes) const;
  [[nodiscard]] CudaSharedMemory allocate_shared_memory(std::size_t bytes) const;
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
  friend class CudaSharedMemory;
  friend class CudaKernel;

  struct Impl;

  explicit CudaBackend(std::shared_ptr<Impl> impl,
                       std::shared_ptr<CudaBackendTraceSink> trace_sink = {});
  std::shared_ptr<Impl> impl_;
  std::shared_ptr<CudaBackendTraceSink> trace_sink_;
};

class CudaSharedMemory {
public:
  CudaSharedMemory() = default;
  CudaSharedMemory(const CudaSharedMemory&) = delete;
  CudaSharedMemory& operator=(const CudaSharedMemory&) = delete;
  CudaSharedMemory(CudaSharedMemory&& other) noexcept;
  CudaSharedMemory& operator=(CudaSharedMemory&& other) noexcept;
  ~CudaSharedMemory();

  [[nodiscard]] CudaDevicePtr ptr() const { return ptr_; }
  [[nodiscard]] std::size_t size() const { return size_; }
  // Borrow the exported OS handle without transferring ownership.
  [[nodiscard]] CudaShareableHandle shareable_handle() const { return shareable_handle_; }
  [[nodiscard]] explicit operator bool() const { return ptr_ != 0 && size_ != 0; }
  // Transfer the exported OS handle to a graphics API import path such as Vulkan.
  // After release, this object still owns the CUDA VMM mapping but will not close the handle.
  [[nodiscard]] CudaShareableHandle release_shareable_handle();
  void reset();

private:
  friend class CudaBackend;

  CudaSharedMemory(std::shared_ptr<CudaBackend::Impl> backend, void* context, CudaDevicePtr ptr,
                   std::size_t size, CudaShareableHandle shareable_handle);

  std::shared_ptr<CudaBackend::Impl> backend_;
  void* context_ = nullptr;
  CudaDevicePtr ptr_ = 0;
  std::size_t size_ = 0;
#if defined(_WIN32)
  CudaShareableHandle shareable_handle_ = nullptr;
#else
  CudaShareableHandle shareable_handle_ = -1;
#endif
  bool owns_shareable_handle_ = false;
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
