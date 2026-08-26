#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace reco::io::detail::nvbufsurface_7_1 {

// DeepStream 7.1 NvBufSurface ABI used by the Jetson target sysroot.
constexpr std::size_t kMaxPlanes = 4;
constexpr std::size_t kStructurePadding = 4;
constexpr std::uint32_t kMemCudaDevice = 2;
constexpr std::uint32_t kMemSurfaceArray = 4;

struct PlaneParams {
  std::uint32_t num_planes = 0;
  std::array<std::uint32_t, kMaxPlanes> width{};
  std::array<std::uint32_t, kMaxPlanes> height{};
  std::array<std::uint32_t, kMaxPlanes> pitch{};
  std::array<std::uint32_t, kMaxPlanes> offset{};
  std::array<std::uint32_t, kMaxPlanes> psize{};
  std::array<std::uint32_t, kMaxPlanes> bytes_per_pix{};
  std::array<void*, kStructurePadding * kMaxPlanes> reserved{};
};

struct MappedAddr {
  std::array<void*, kMaxPlanes> addr{};
  void* egl_image = nullptr;
  std::array<void*, kStructurePadding> reserved{};
};

struct SurfaceParams {
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
  PlaneParams plane_params;
  MappedAddr mapped_addr;
  void* paramex = nullptr;
  std::array<void*, kStructurePadding - 1> reserved{};
};

struct Surface {
  std::uint32_t gpu_id = 0;
  std::uint32_t batch_size = 0;
  std::uint32_t num_filled = 0;
  std::uint8_t is_contiguous = 0;
  std::array<std::uint8_t, 3> pad0{};
  std::uint32_t mem_type = 0;
  std::uint32_t pad1 = 0;
  SurfaceParams* surface_list = nullptr;
  std::array<void*, kStructurePadding> reserved{};
};

static_assert(sizeof(PlaneParams) == 232);
static_assert(sizeof(MappedAddr) == 72);
static_assert(sizeof(SurfaceParams) == 384);
static_assert(sizeof(Surface) == 64);
static_assert(offsetof(SurfaceParams, buffer_desc) == 24);
static_assert(offsetof(SurfaceParams, data_ptr) == 40);
static_assert(offsetof(SurfaceParams, plane_params) == 48);
static_assert(offsetof(SurfaceParams, mapped_addr) == 280);
static_assert(offsetof(SurfaceParams, paramex) == 352);
static_assert(offsetof(Surface, mem_type) == 16);
static_assert(offsetof(Surface, surface_list) == 24);

} // namespace reco::io::detail::nvbufsurface_7_1
