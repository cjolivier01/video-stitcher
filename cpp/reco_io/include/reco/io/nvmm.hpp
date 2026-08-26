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
  std::uint32_t uv_offset = 0;
  std::uint32_t uv_pitch = 0;
  std::uint32_t total_size = 0;
  void* surface_ptr = nullptr;
  NvbufSurfaceAbi abi = NvbufSurfaceAbi::DeepStream7_1;
  NvmmMemoryType memory_type = NvmmMemoryType::SurfaceArray;
  std::uint32_t gpu_id = 0;
  core::CudaDevicePtr cuda_base_ptr = 0;
  Nv12ColorMatrix color_matrix = Nv12ColorMatrix::Bt601;
  Nv12ColorRange color_range = Nv12ColorRange::Limited;
};

struct NvmmCudaFrame {
  core::CudaDevicePtr y_ptr = 0;
  core::CudaDevicePtr uv_ptr = 0;
  std::size_t y_pitch = 0;
  std::size_t uv_pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t gpu_id = 0;
  Nv12ColorMatrix color_matrix = Nv12ColorMatrix::Bt601;
  Nv12ColorRange color_range = Nv12ColorRange::Limited;
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
