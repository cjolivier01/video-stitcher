#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace reco::io::detail::nvbufsurface_9_1 {

// DeepStream 9.1 NvBufSurface ABI. Keep this adapter versioned: these objects
// are supplied by a dynamically loaded proprietary runtime and are not a
// stable, cross-version C ABI.
constexpr std::size_t kMaxPlanes = 4;
constexpr std::size_t kStructurePadding = 4;
constexpr std::uint32_t kMemCudaDevice = 2;
constexpr std::uint32_t kMemSurfaceArray = 4;
constexpr std::uint32_t kLayoutPitch = 0;
constexpr std::uint32_t kColorNv12 = 6;
constexpr std::uint32_t kColorNv12Er = 7;
constexpr std::uint32_t kColorNv12_709 = 33;
constexpr std::uint32_t kColorNv12_709Er = 34;
constexpr std::uint32_t kColorNv12_2020 = 36;

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
  void* nvmm_ptr = nullptr;
  void* cuda_ptr = nullptr;
  std::array<void*, kStructurePadding> reserved{};
};

struct CudaBuffer {
  void* base_ptr = nullptr;
  void* data_ptr = nullptr;
  void* external_memory = nullptr;
  void* mipmap = nullptr;
  std::array<std::uint8_t, 64> reserved{};
};

struct SurfaceParams {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t pitch = 0;
  std::uint32_t color_format = 0;
  std::uint32_t layout = 0;
  std::uint32_t pad0 = 0;
  std::uint64_t buffer_desc = 0;
  std::uint32_t data_size = 0;
  std::uint32_t pad1 = 0;
  void* data_ptr = nullptr;
  PlaneParams plane_params;
  MappedAddr mapped_addr;
  void* params_ex = nullptr;
  CudaBuffer* cuda_buffer = nullptr;
  std::array<void*, kStructurePadding> reserved{};
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
  std::uint8_t is_imported_buffer = 0;
  std::array<std::uint8_t, 7> pad2{};
  std::array<void*, kStructurePadding> reserved{};
};

static_assert(sizeof(PlaneParams) == 232);
static_assert(sizeof(MappedAddr) == 88);
static_assert(sizeof(CudaBuffer) == 96);
static_assert(sizeof(SurfaceParams) == 416);
static_assert(sizeof(Surface) == 72);
static_assert(offsetof(SurfaceParams, buffer_desc) == 24);
static_assert(offsetof(SurfaceParams, data_ptr) == 40);
static_assert(offsetof(SurfaceParams, plane_params) == 48);
static_assert(offsetof(SurfaceParams, mapped_addr) == 280);
static_assert(offsetof(SurfaceParams, params_ex) == 368);
static_assert(offsetof(SurfaceParams, cuda_buffer) == 376);
static_assert(offsetof(Surface, mem_type) == 16);
static_assert(offsetof(Surface, surface_list) == 24);
static_assert(offsetof(Surface, is_imported_buffer) == 32);

} // namespace reco::io::detail::nvbufsurface_9_1
