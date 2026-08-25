#include "reco/io/nvmm.hpp"

#include <cstddef>
#include <limits>

namespace reco::io {

static_assert(offsetof(NvBufSurfaceParams, buffer_desc) == 24);
static_assert(offsetof(NvBufSurfaceParams, data_ptr) == 40);
static_assert(offsetof(NvBufSurfaceParams, plane_params) == 48);
static_assert(offsetof(NvBufSurface, surface_list) == 24);
static_assert(sizeof(NvBufSurfaceMappedAddr) == 72);
static_assert(sizeof(NvBufSurfacePlaneParams) == 232);
static_assert(sizeof(NvBufSurfaceParams) == 384);
static_assert(sizeof(NvBufSurface) == 64);

NvmmFrameInfo extract_nvmm_frame_info(const void* mapped_data) {
  if (mapped_data == nullptr) {
    throw NvmmError("NvBufSurface mapped data is null");
  }
  const auto* surface = static_cast<const NvBufSurface*>(mapped_data);
  if (surface->mem_type != kNvbufMemSurfaceArray) {
    throw NvmmError("expected NVBUF_MEM_SURFACE_ARRAY");
  }
  if (surface->batch_size == 0 || surface->num_filled == 0) {
    throw NvmmError("NvBufSurface has 0 filled buffers");
  }
  if (surface->surface_list == nullptr) {
    throw NvmmError("NvBufSurface surface_list is null");
  }
  const auto& params = surface->surface_list[0];
  if (params.plane_params.num_planes < 2) {
    throw NvmmError("expected at least 2 NV12 planes");
  }
  if (params.buffer_desc < 0 ||
      params.buffer_desc > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
    throw NvmmError("NvBufSurface DMA-buf descriptor is out of range");
  }
  const auto uv_offset = params.plane_params.offset[1];
  const auto uv_pitch = params.plane_params.pitch[1];
  const auto uv_height = params.plane_params.height[1];
  if (uv_pitch != 0 &&
      uv_height > (std::numeric_limits<std::uint32_t>::max() - uv_offset) / uv_pitch) {
    throw NvmmError("NvBufSurface total size overflows");
  }
  const auto total_size = uv_offset + uv_pitch * uv_height;

  return NvmmFrameInfo{
      .dmabuf_fd = static_cast<std::int32_t>(params.buffer_desc),
      .width = params.width,
      .height = params.height,
      .y_offset = params.plane_params.offset[0],
      .y_pitch = params.plane_params.pitch[0],
      .uv_offset = uv_offset,
      .uv_pitch = uv_pitch,
      .total_size = total_size,
      .surface_ptr = const_cast<void*>(mapped_data),
  };
}

} // namespace reco::io
