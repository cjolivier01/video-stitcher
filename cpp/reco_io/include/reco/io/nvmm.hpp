#pragma once

#include "reco/core/cuda_backend.hpp"
#include "reco/core/video_format.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace reco::io {

constexpr std::size_t kNvbufMaxPlanes = 4;
constexpr std::size_t kNvbufStructurePadding = 4;
constexpr std::uint32_t kNvbufMemSurfaceArray = 4;

struct NvBufSurfacePlaneParams {
  std::uint32_t num_planes = 0;
  std::array<std::uint32_t, kNvbufMaxPlanes> width{};
  std::array<std::uint32_t, kNvbufMaxPlanes> height{};
  std::array<std::uint32_t, kNvbufMaxPlanes> pitch{};
  std::array<std::uint32_t, kNvbufMaxPlanes> offset{};
  std::array<std::uint32_t, kNvbufMaxPlanes> psize{};
  std::array<std::uint32_t, kNvbufMaxPlanes> bytes_per_pix{};
  std::array<void*, kNvbufStructurePadding * kNvbufMaxPlanes> reserved{};
};

struct NvBufSurfaceMappedAddr {
  std::array<void*, kNvbufMaxPlanes> addr{};
  void* egl_image = nullptr;
  std::array<void*, kNvbufStructurePadding> reserved{};
};

struct NvBufSurfaceParams {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t pitch = 0;
  std::uint32_t color_format = 0;
  std::uint32_t layout = 0;
  std::uint32_t pad0 = 0;
  std::int64_t buffer_desc = -1;
  std::uint32_t data_size = 0;
  std::uint32_t pad1 = 0;
  void* data_ptr = nullptr;
  NvBufSurfacePlaneParams plane_params;
  NvBufSurfaceMappedAddr mapped_addr;
  void* paramex = nullptr;
  std::array<void*, kNvbufStructurePadding - 1> reserved{};
};

struct NvBufSurface {
  std::uint32_t gpu_id = 0;
  std::uint32_t batch_size = 0;
  std::uint32_t num_filled = 0;
  std::uint8_t is_contiguous = 0;
  std::array<std::uint8_t, 3> pad0{};
  std::uint32_t mem_type = 0;
  std::uint32_t pad1 = 0;
  NvBufSurfaceParams* surface_list = nullptr;
  std::array<void*, kNvbufStructurePadding> reserved{};
};

class NvmmError : public std::runtime_error {
public:
  explicit NvmmError(std::string message) : std::runtime_error(std::move(message)) {}
};

enum class NvbufSurfaceAbi : std::uint32_t {
  DeepStream7_1 = 701,
  DeepStream9_1 = 901,
};

struct NvmmFrameInfo;
struct NvmmCudaFrame;

/// Retains the exact NvBufSurface and DeepStream-version libraries used to
/// select an ABI and to map frames produced by a decoder.
class NvbufSurfaceRuntime final {
public:
  ~NvbufSurfaceRuntime();

  NvbufSurfaceRuntime(const NvbufSurfaceRuntime&) = delete;
  NvbufSurfaceRuntime& operator=(const NvbufSurfaceRuntime&) = delete;

  /// Returns the ABI proven by this runtime binding.
  [[nodiscard]] NvbufSurfaceAbi abi() const noexcept;
  /// Returns the loaded object that provides NvBufSurface mapping.
  [[nodiscard]] std::string_view library() const noexcept;

private:
  struct State;
  explicit NvbufSurfaceRuntime(std::unique_ptr<State> state);

  std::unique_ptr<State> state_;

  friend std::shared_ptr<const NvbufSurfaceRuntime> discover_nvbufsurface_runtime();
  friend std::optional<std::string> validate_nvbufsurface_runtime_provenance(
      const std::shared_ptr<const NvbufSurfaceRuntime>& runtime);
  friend NvmmCudaFrame map_nvmm_frame_to_cuda(const NvmmFrameInfo& info,
                                              std::shared_ptr<void> owner);
};

/// Discovers the installed DeepStream NvBufSurface ABI from the runtime
/// `nvds_version` API and verifies the symbols required by that ABI.
///
/// Only the explicitly adapted DeepStream 7.1 and 9.1 layouts are accepted.
/// Missing, mixed, or unsupported runtimes throw `NvmmError` rather than
/// selecting a compatibility layout implicitly.
[[nodiscard]] NvbufSurfaceAbi discover_nvbufsurface_abi();
/// Discovers and retains the exact runtime objects that establish the ABI.
[[nodiscard]] std::shared_ptr<const NvbufSurfaceRuntime> discover_nvbufsurface_runtime();
/// Rejects a second loaded NvBufSurface provider that could own decoded frames.
[[nodiscard]] std::optional<std::string>
validate_nvbufsurface_runtime_provenance(const std::shared_ptr<const NvbufSurfaceRuntime>& runtime);

enum class NvmmMemoryType : std::uint32_t {
  CudaDevice = 2,
  SurfaceArray = 4,
};

using Nv12ColorMatrix = core::YuvColorMatrix;
using Nv12ColorRange = core::YuvColorRange;

struct NvmmFrameInfo {
  std::int32_t dmabuf_fd = -1;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t y_offset = 0;
  std::uint32_t y_pitch = 0;
  std::uint32_t y_size = 0;
  std::uint32_t uv_offset = 0;
  std::uint32_t uv_pitch = 0;
  std::uint32_t uv_size = 0;
  std::uint32_t total_size = 0;
  void* surface_ptr = nullptr;
  NvbufSurfaceAbi abi = NvbufSurfaceAbi::DeepStream7_1;
  NvmmMemoryType memory_type = NvmmMemoryType::SurfaceArray;
  std::uint32_t gpu_id = 0;
  core::CudaDevicePtr cuda_base_ptr = 0;
  Nv12ColorMatrix color_matrix = Nv12ColorMatrix::Bt601;
  Nv12ColorRange color_range = Nv12ColorRange::Limited;
  std::shared_ptr<const NvbufSurfaceRuntime> runtime;
};

struct NvmmCudaFrame {
  core::CudaDevicePtr y_ptr = 0;
  core::CudaDevicePtr uv_ptr = 0;
  std::size_t y_pitch = 0;
  std::size_t uv_pitch = 0;
  /// Bytes accessible from the Y pointer within its driver-reported mapping and plane allocation.
  std::size_t y_accessible_bytes = 0;
  /// Bytes accessible from the UV pointer within its driver-reported mapping and plane allocation.
  std::size_t uv_accessible_bytes = 0;
  /// Base address of the CUDA mapping containing the Y plane.
  core::CudaDevicePtr y_mapping_base = 0;
  /// Base address of the CUDA mapping containing the UV plane.
  core::CudaDevicePtr uv_mapping_base = 0;
  /// Driver-reported byte size of the CUDA mapping containing the Y plane.
  std::size_t y_mapping_bytes = 0;
  /// Driver-reported byte size of the CUDA mapping containing the UV plane.
  std::size_t uv_mapping_bytes = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t gpu_id = 0;
  /// Process-local identity of the CUDA context proven by the driver for both planes.
  std::uintptr_t context_id = 0;
  /// CUDA device ordinal proven by the driver for both planes.
  int device_ordinal = -1;
  Nv12ColorMatrix color_matrix = Nv12ColorMatrix::Bt601;
  Nv12ColorRange color_range = Nv12ColorRange::Limited;
  /// Exact NvBufSurface provider retained through mapping cleanup.
  std::shared_ptr<const NvbufSurfaceRuntime> runtime;
  std::shared_ptr<void> owner;
};

// Compatibility overload for the DeepStream 7.1 Jetson ABI used by the
// original C++ NVMM contract.
[[nodiscard]] NvmmFrameInfo extract_nvmm_frame_info(const void* mapped_data);
[[nodiscard]] NvmmFrameInfo extract_nvmm_frame_info(const void* mapped_data, NvbufSurfaceAbi abi);
[[nodiscard]] std::optional<std::string> validate_nvmm_frame_info(const NvmmFrameInfo& info);
[[nodiscard]] bool is_nvmm_cuda_interop_available();
[[nodiscard]] std::string nvmm_cuda_interop_availability_error();
[[nodiscard]] NvmmCudaFrame map_nvmm_frame_to_cuda(const NvmmFrameInfo& info,
                                                   std::shared_ptr<void> owner);

} // namespace reco::io
