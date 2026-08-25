#include "reco/detect/cuda_preprocess.hpp"

#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace reco::detect {
namespace {

using reco::core::CudaBackend;
using reco::core::CudaDevicePtr;

constexpr char kP010ToNv12Ptx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry p010_to_nv12(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 n
)
{
    .reg .u32 %i, %n_val, %tmp, %tmp2;
    .reg .u64 %src_ptr, %dst_ptr, %addr;
    .reg .u16 %val16;
    .reg .u32 %val32;
    .reg .pred %p;

    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %i, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %i, %i, %tmp;

    ld.param.u32 %n_val, [n];
    setp.ge.u32 %p, %i, %n_val;
    @%p bra done;

    ld.param.u64 %src_ptr, [src];
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 1;
    add.u64 %addr, %src_ptr, %addr;
    ld.global.u16 %val16, [%addr];

    cvt.u32.u16 %val32, %val16;
    shr.u32 %val32, %val32, 8;

    ld.param.u64 %dst_ptr, [dst];
    cvt.u64.u32 %addr, %i;
    add.u64 %addr, %dst_ptr, %addr;
    cvt.u16.u32 %val16, %val32;
    st.global.u8 [%addr], %val16;

done:
    ret;
}
)ptx";

constexpr char kNormalizeHwcToChwPtx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry normalize_hwc_to_chw(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 width,
    .param .u32 height
)
{
    .reg .u32 %x, %y, %w, %h, %hw, %src_idx, %dst_idx;
    .reg .u64 %src_ptr, %dst_ptr, %addr;
    .reg .u32 %tmp, %tmp2;
    .reg .u16 %pixel;
    .reg .f32 %val, %scale;
    .reg .pred %p;

    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %x, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %x, %x, %tmp;
    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %y, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %y, %y, %tmp;

    ld.param.u32 %w, [width];
    ld.param.u32 %h, [height];
    setp.ge.u32 %p, %x, %w;
    @%p bra done;
    setp.ge.u32 %p, %y, %h;
    @%p bra done;

    mul.lo.u32 %hw, %w, %h;
    mul.lo.u32 %src_idx, %y, %w;
    add.u32 %src_idx, %src_idx, %x;
    mul.lo.u32 %src_idx, %src_idx, 3;
    mul.lo.u32 %dst_idx, %y, %w;
    add.u32 %dst_idx, %dst_idx, %x;

    ld.param.u64 %src_ptr, [src];
    ld.param.u64 %dst_ptr, [dst];
    mov.f32 %scale, 0f3B808081;

    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    cvt.u64.u32 %addr, %dst_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 1;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    add.u32 %tmp, %hw, %dst_idx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 2;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    add.u32 %tmp, %hw, %hw;
    add.u32 %tmp, %tmp, %dst_idx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

done:
    ret;
}
)ptx";

constexpr char kNormalizeRgbaToChwPtx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry normalize_rgba_to_chw(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 width,
    .param .u32 height
)
{
    .reg .u32 %x, %y, %w, %h, %hw, %src_idx, %dst_idx;
    .reg .u64 %src_ptr, %dst_ptr, %addr;
    .reg .u32 %tmp, %tmp2;
    .reg .u16 %pixel;
    .reg .f32 %val, %scale;
    .reg .pred %p;

    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %x, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %x, %x, %tmp;
    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %y, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %y, %y, %tmp;

    ld.param.u32 %w, [width];
    ld.param.u32 %h, [height];
    setp.ge.u32 %p, %x, %w;
    @%p bra done;
    setp.ge.u32 %p, %y, %h;
    @%p bra done;

    mul.lo.u32 %hw, %w, %h;
    mul.lo.u32 %src_idx, %y, %w;
    add.u32 %src_idx, %src_idx, %x;
    shl.b32 %src_idx, %src_idx, 2;
    mul.lo.u32 %dst_idx, %y, %w;
    add.u32 %dst_idx, %dst_idx, %x;

    ld.param.u64 %src_ptr, [src];
    ld.param.u64 %dst_ptr, [dst];
    mov.f32 %scale, 0f3B808081;

    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    cvt.u64.u32 %addr, %dst_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 1;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    add.u32 %tmp, %hw, %dst_idx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 2;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    add.u32 %tmp, %hw, %hw;
    add.u32 %tmp, %tmp, %dst_idx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

done:
    ret;
}
)ptx";

void validate_device_ptrs(CudaDevicePtr src, CudaDevicePtr dst) {
  if (src == 0 || dst == 0) {
    throw std::invalid_argument("CUDA preprocess requires live source and destination pointers");
  }
}

void validate_extent(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("CUDA preprocess dimensions must be non-zero");
  }
}

std::uint32_t ceil_div_u32(std::uint32_t value, std::uint32_t divisor) {
  return value / divisor + (value % divisor == 0 ? 0 : 1);
}

std::uint32_t checked_sample_count(std::uint32_t width, std::uint32_t height) {
  const auto samples = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
  if (samples > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("CUDA P010 plane sample count exceeds u32 range");
  }
  return static_cast<std::uint32_t>(samples);
}

const reco::core::CudaKernel& cached_kernel(CudaBackend& backend, std::string_view ptx,
                                            std::string_view name) {
  struct KernelCache {
    std::once_flag once;
    std::unique_ptr<reco::core::CudaKernel> kernel;
    std::exception_ptr error;
  };

  static KernelCache p010;
  static KernelCache rgb;
  static KernelCache rgba;
  KernelCache* cache = nullptr;
  if (name == "p010_to_nv12") {
    cache = &p010;
  } else if (name == "normalize_hwc_to_chw") {
    cache = &rgb;
  } else if (name == "normalize_rgba_to_chw") {
    cache = &rgba;
  } else {
    throw std::invalid_argument("unknown CUDA preprocess kernel");
  }

  std::call_once(cache->once, [&] {
    try {
      cache->kernel =
          std::make_unique<reco::core::CudaKernel>(backend.load_kernel_from_ptx(ptx, name));
    } catch (...) {
      cache->error = std::current_exception();
    }
  });
  if (cache->error) {
    std::rethrow_exception(cache->error);
  }
  return *cache->kernel;
}

void launch_p010(CudaBackend& backend, CudaDevicePtr src, CudaDevicePtr dst,
                 std::uint32_t samples) {
  validate_device_ptrs(src, dst);
  if (samples == 0) {
    throw std::invalid_argument("CUDA P010 conversion sample count must be non-zero");
  }
  const auto& kernel = cached_kernel(backend, kP010ToNv12Ptx, "p010_to_nv12");
  constexpr std::uint32_t block = 256;
  const std::uint32_t grid = ceil_div_u32(samples, block);
  CudaDevicePtr src_arg = src;
  CudaDevicePtr dst_arg = dst;
  std::uint32_t samples_arg = samples;
  void* args[] = {&src_arg, &dst_arg, &samples_arg};
  kernel.launch({.grid = {.x = grid, .y = 1, .z = 1}, .block = {.x = block, .y = 1, .z = 1}},
                args);
  kernel.synchronize();
}

} // namespace

void p010_to_nv12(CudaBackend& backend, CudaDevicePtr src, CudaDevicePtr dst,
                  std::uint32_t samples) {
  launch_p010(backend, src, dst, samples);
}

void p010_plane_to_nv12(CudaBackend& backend, CudaDevicePtr src, std::size_t src_pitch,
                        CudaDevicePtr dst, std::uint32_t width, std::uint32_t height) {
  validate_device_ptrs(src, dst);
  validate_extent(width, height);
  const std::size_t min_src_pitch = static_cast<std::size_t>(width) * 2;
  if (src_pitch < min_src_pitch) {
    throw std::invalid_argument("CUDA P010 source pitch is smaller than width");
  }
  if (src_pitch == min_src_pitch) {
    launch_p010(backend, src, dst, checked_sample_count(width, height));
    return;
  }
  for (std::uint32_t row = 0; row < height; ++row) {
    const auto row_offset = static_cast<CudaDevicePtr>(static_cast<std::size_t>(row) * src_pitch);
    const auto dst_offset =
        static_cast<CudaDevicePtr>(static_cast<std::size_t>(row) * static_cast<std::size_t>(width));
    launch_p010(backend, src + row_offset, dst + dst_offset, width);
  }
}

void normalize_hwc_to_chw(CudaBackend& backend, CudaDevicePtr src, CudaDevicePtr dst,
                          std::uint32_t width, std::uint32_t height) {
  validate_device_ptrs(src, dst);
  validate_extent(width, height);
  const auto& kernel = cached_kernel(backend, kNormalizeHwcToChwPtx, "normalize_hwc_to_chw");
  constexpr reco::core::CudaDim3 block{.x = 16, .y = 16, .z = 1};
  const reco::core::CudaDim3 grid{.x = ceil_div_u32(width, block.x),
                                  .y = ceil_div_u32(height, block.y),
                                  .z = 1};
  CudaDevicePtr src_arg = src;
  CudaDevicePtr dst_arg = dst;
  std::uint32_t width_arg = width;
  std::uint32_t height_arg = height;
  void* args[] = {&src_arg, &dst_arg, &width_arg, &height_arg};
  kernel.launch({.grid = grid, .block = block}, args);
  kernel.synchronize();
}

void normalize_rgba_to_chw(CudaBackend& backend, CudaDevicePtr src, CudaDevicePtr dst,
                           std::uint32_t width, std::uint32_t height) {
  validate_device_ptrs(src, dst);
  validate_extent(width, height);
  const auto& kernel = cached_kernel(backend, kNormalizeRgbaToChwPtx, "normalize_rgba_to_chw");
  constexpr reco::core::CudaDim3 block{.x = 16, .y = 16, .z = 1};
  const reco::core::CudaDim3 grid{.x = ceil_div_u32(width, block.x),
                                  .y = ceil_div_u32(height, block.y),
                                  .z = 1};
  CudaDevicePtr src_arg = src;
  CudaDevicePtr dst_arg = dst;
  std::uint32_t width_arg = width;
  std::uint32_t height_arg = height;
  void* args[] = {&src_arg, &dst_arg, &width_arg, &height_arg};
  kernel.launch({.grid = grid, .block = block}, args);
  kernel.synchronize();
}

} // namespace reco::detect
