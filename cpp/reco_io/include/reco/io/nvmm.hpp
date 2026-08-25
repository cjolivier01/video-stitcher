#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace reco::io {

class NvmmError : public std::runtime_error {
public:
  explicit NvmmError(std::string message) : std::runtime_error(std::move(message)) {}
};

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
};

[[nodiscard]] NvmmFrameInfo extract_nvmm_frame_info(const void* mapped_data);

} // namespace reco::io
