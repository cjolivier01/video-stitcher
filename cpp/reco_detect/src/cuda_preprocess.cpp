#include "reco/detect/cuda_preprocess.hpp"

#include <cmath>
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

constexpr char kNv12ToRgbChwFullrangePtx[] = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry nv12_to_rgb_chw_fullrange(
    .param .u64 y_ptr,
    .param .u64 uv_ptr,
    .param .u64 dst,
    .param .u32 y_pitch,
    .param .u32 src_w,
    .param .u32 src_h,
    .param .u32 dst_w,
    .param .u32 dst_h,
    .param .u32 pad_x,
    .param .u32 pad_y,
    .param .f32 scale,
    .param .u32 flip180
)
{
    .reg .u32 %ox, %oy, %dw, %dh, %px, %py, %sw, %sh, %ypitch;
    .reg .u32 %sx0, %sy0;
    .reg .u32 %tmp, %tmp2, %plane, %didx;
    .reg .u64 %yp, %uvp, %dp, %addr;
    .reg .f32 %srcx, %srcy, %inv_scale;
    .reg .f32 %zero, %f255, %c128;
    .reg .f32 %y00, %u_val, %v_val, %r, %g, %b;
    .reg .f32 %t1, %t2;
    .reg .u16 %pix;
    .reg .pred %p, %q;

    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %ox, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %ox, %ox, %tmp;

    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %oy, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %oy, %oy, %tmp;

    ld.param.u32 %dw, [dst_w];
    ld.param.u32 %dh, [dst_h];
    setp.ge.u32 %p, %ox, %dw;
    @%p bra done;
    setp.ge.u32 %p, %oy, %dh;
    @%p bra done;

    ld.param.u32 %px, [pad_x];
    ld.param.u32 %py, [pad_y];
    ld.param.u32 %sw, [src_w];
    ld.param.u32 %sh, [src_h];
    ld.param.f32 %inv_scale, [scale];

    setp.lt.u32 %p, %ox, %px;
    @%p bra done;
    setp.lt.u32 %p, %oy, %py;
    @%p bra done;
    sub.u32 %tmp, %dw, %px;
    setp.ge.u32 %p, %ox, %tmp;
    @%p bra done;
    sub.u32 %tmp, %dh, %py;
    setp.ge.u32 %p, %oy, %tmp;
    @%p bra done;

    sub.u32 %tmp, %ox, %px;
    cvt.rn.f32.u32 %srcx, %tmp;
    div.rn.f32 %srcx, %srcx, %inv_scale;
    cvt.rzi.u32.f32 %sx0, %srcx;
    sub.u32 %tmp2, %sw, 1;
    min.u32 %sx0, %sx0, %tmp2;

    sub.u32 %tmp, %oy, %py;
    cvt.rn.f32.u32 %srcy, %tmp;
    div.rn.f32 %srcy, %srcy, %inv_scale;
    cvt.rzi.u32.f32 %sy0, %srcy;
    sub.u32 %tmp2, %sh, 1;
    min.u32 %sy0, %sy0, %tmp2;

    ld.param.u32 %tmp, [flip180];
    setp.ne.u32 %q, %tmp, 0;
    sub.u32 %tmp, %sw, 1;
    @%q sub.u32 %sx0, %tmp, %sx0;
    sub.u32 %tmp, %sh, 1;
    @%q sub.u32 %sy0, %tmp, %sy0;

    ld.param.u64 %yp, [y_ptr];
    ld.param.u32 %ypitch, [y_pitch];
    mul.lo.u32 %tmp, %sy0, %ypitch;
    add.u32 %tmp, %tmp, %sx0;
    cvt.u64.u32 %addr, %tmp;
    add.u64 %addr, %yp, %addr;
    ld.global.u8 %pix, [%addr];
    cvt.rn.f32.u16 %y00, %pix;

    ld.param.u64 %uvp, [uv_ptr];
    shr.u32 %tmp, %sy0, 1;
    mul.lo.u32 %tmp, %tmp, %ypitch;
    shr.u32 %tmp2, %sx0, 1;
    shl.b32 %tmp2, %tmp2, 1;
    add.u32 %tmp, %tmp, %tmp2;
    cvt.u64.u32 %addr, %tmp;
    add.u64 %addr, %uvp, %addr;
    ld.global.u8 %pix, [%addr];
    cvt.rn.f32.u16 %u_val, %pix;
    ld.global.u8 %pix, [%addr+1];
    cvt.rn.f32.u16 %v_val, %pix;

    mov.f32 %c128, 0f43000000;
    sub.f32 %u_val, %u_val, %c128;
    sub.f32 %v_val, %v_val, %c128;

    mov.f32 %t1, 0f3FC9930C;
    fma.rn.f32 %r, %t1, %v_val, %y00;
    mov.f32 %t1, 0fBE3FCB92;
    fma.rn.f32 %g, %t1, %u_val, %y00;
    mov.f32 %t2, 0fBEEFAACE;
    fma.rn.f32 %g, %t2, %v_val, %g;
    mov.f32 %t1, 0f3FED844D;
    fma.rn.f32 %b, %t1, %u_val, %y00;

    mov.f32 %zero, 0f00000000;
    mov.f32 %f255, 0f437F0000;
    max.f32 %r, %r, %zero;
    min.f32 %r, %r, %f255;
    max.f32 %g, %g, %zero;
    min.f32 %g, %g, %f255;
    max.f32 %b, %b, %zero;
    min.f32 %b, %b, %f255;

    mov.f32 %t1, 0f3B808081;
    mul.f32 %r, %r, %t1;
    mul.f32 %g, %g, %t1;
    mul.f32 %b, %b, %t1;

    mul.lo.u32 %plane, %dw, %dh;
    mul.lo.u32 %didx, %oy, %dw;
    add.u32 %didx, %didx, %ox;
    ld.param.u64 %dp, [dst];

    cvt.u64.u32 %addr, %didx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dp, %addr;
    st.global.f32 [%addr], %r;

    add.u32 %tmp, %plane, %didx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dp, %addr;
    st.global.f32 [%addr], %g;

    add.u32 %tmp, %plane, %plane;
    add.u32 %tmp, %tmp, %didx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dp, %addr;
    st.global.f32 [%addr], %b;

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
  } else if (name == "nv12_to_rgb_chw_fullrange") {
    static KernelCache nv12;
    cache = &nv12;
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

void nv12_to_rgb_chw_fullrange(CudaBackend& backend, CudaDevicePtr y, CudaDevicePtr uv,
                               CudaDevicePtr dst, std::uint32_t y_pitch,
                               std::uint32_t src_width, std::uint32_t src_height,
                               std::uint32_t dst_width, std::uint32_t dst_height,
                               std::uint32_t pad_x, std::uint32_t pad_y, float scale,
                               int rotation_degrees) {
  if (y == 0 || uv == 0 || dst == 0) {
    throw std::invalid_argument("CUDA NV12 preprocess requires live source and destination pointers");
  }
  validate_extent(src_width, src_height);
  validate_extent(dst_width, dst_height);
  if (src_width % 2 != 0 || src_height % 2 != 0) {
    throw std::invalid_argument("CUDA NV12 source dimensions must be even");
  }
  if (y_pitch < src_width) {
    throw std::invalid_argument("CUDA NV12 pitch is smaller than source width");
  }
  if (!std::isfinite(scale) || scale <= 0.0F) {
    throw std::invalid_argument("CUDA NV12 scale must be finite and positive");
  }
  if (pad_x > dst_width || pad_y > dst_height || pad_x >= dst_width - pad_x ||
      pad_y >= dst_height - pad_y) {
    throw std::invalid_argument("CUDA NV12 padding leaves no content region");
  }

  const auto& kernel =
      cached_kernel(backend, kNv12ToRgbChwFullrangePtx, "nv12_to_rgb_chw_fullrange");
  constexpr reco::core::CudaDim3 block{.x = 16, .y = 16, .z = 1};
  const reco::core::CudaDim3 grid{.x = ceil_div_u32(dst_width, block.x),
                                  .y = ceil_div_u32(dst_height, block.y),
                                  .z = 1};
  CudaDevicePtr y_arg = y;
  CudaDevicePtr uv_arg = uv;
  CudaDevicePtr dst_arg = dst;
  std::uint32_t pitch_arg = y_pitch;
  std::uint32_t src_w_arg = src_width;
  std::uint32_t src_h_arg = src_height;
  std::uint32_t dst_w_arg = dst_width;
  std::uint32_t dst_h_arg = dst_height;
  std::uint32_t pad_x_arg = pad_x;
  std::uint32_t pad_y_arg = pad_y;
  float scale_arg = scale;
  std::uint32_t flip_arg = rotation_degrees == 180 ? 1U : 0U;
  void* args[] = {&y_arg,     &uv_arg,    &dst_arg,   &pitch_arg, &src_w_arg, &src_h_arg,
                  &dst_w_arg, &dst_h_arg, &pad_x_arg, &pad_y_arg, &scale_arg, &flip_arg};
  kernel.launch({.grid = grid, .block = block}, args);
  kernel.synchronize();
}

} // namespace reco::detect
