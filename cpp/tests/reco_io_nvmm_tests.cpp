namespace reco::io {
struct NvBufSurfacePlaneParams;
struct NvBufSurfaceMappedAddr;
struct NvBufSurfaceParams;
struct NvBufSurface;
} // namespace reco::io

#include "reco/io/detail/nvbufsurface_7_1.hpp"
#include "reco/io/detail/nvbufsurface_9_1.hpp"
#include "reco/io/nvmm.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>

using namespace reco::io;
namespace abi = reco::io::detail::nvbufsurface_9_1;
namespace abi7 = reco::io::detail::nvbufsurface_7_1;

namespace {

int failures = 0;

NvmmFrameInfo extract_info(const void* surface) {
  return extract_nvmm_frame_info(surface, NvbufSurfaceAbi::DeepStream9_1);
}

void expect_true(bool value, std::string_view message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_nvmm_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const NvmmError&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Params> Params make_params_as() {
  Params params;
  params.width = 1920;
  params.height = 1080;
  params.pitch = 2048;
  params.color_format = abi::kColorNv12;
  params.layout = abi::kLayoutPitch;
  params.buffer_desc = 42;
  params.data_size = 2048 * 1620;
  params.plane_params.num_planes = 2;
  params.plane_params.width[0] = 1920;
  params.plane_params.height[0] = 1080;
  params.plane_params.pitch[0] = 2048;
  params.plane_params.offset[0] = 0;
  params.plane_params.psize[0] = 2048 * 1080;
  params.plane_params.bytes_per_pix[0] = 1;
  params.plane_params.width[1] = 960;
  params.plane_params.height[1] = 540;
  params.plane_params.pitch[1] = 2048;
  params.plane_params.offset[1] = 2048 * 1080;
  params.plane_params.psize[1] = 2048 * 540;
  params.plane_params.bytes_per_pix[1] = 2;
  return params;
}

abi::SurfaceParams make_params() { return make_params_as<abi::SurfaceParams>(); }

template <typename Surface, typename Params> Surface make_surface_as(Params& params) {
  Surface surface;
  surface.gpu_id = 0;
  surface.batch_size = 1;
  surface.num_filled = 1;
  surface.mem_type = abi::kMemSurfaceArray;
  surface.surface_list = &params;
  return surface;
}

abi::Surface make_surface(abi::SurfaceParams& params) {
  return make_surface_as<abi::Surface>(params);
}

void nvmm_extraction_matches_rust_layout() {
  auto params = make_params();
  auto surface = make_surface(params);
  const auto info = extract_info(&surface);
  expect_true(info.abi == NvbufSurfaceAbi::DeepStream9_1, "versioned ABI");
  expect_true(info.memory_type == NvmmMemoryType::SurfaceArray, "surface-array memory type");
  expect_eq(info.gpu_id, 0U, "GPU id");
  expect_eq(info.dmabuf_fd, 42, "DMA-buf fd");
  expect_eq(info.width, 1920U, "width");
  expect_eq(info.height, 1080U, "height");
  expect_eq(info.y_offset, 0U, "Y offset");
  expect_eq(info.y_pitch, 2048U, "Y pitch");
  expect_eq(info.uv_offset, 2048U * 1080U, "UV offset");
  expect_eq(info.uv_pitch, 2048U, "UV pitch");
  expect_eq(info.total_size, 2048U * 1620U, "total size");
  expect_true(info.color_matrix == Nv12ColorMatrix::Bt601, "BT.601 color matrix preserved");
  expect_true(info.color_range == Nv12ColorRange::Limited, "limited color range preserved");
  expect_eq(info.surface_ptr, static_cast<void*>(&surface), "surface pointer preserved");
}

void deepstream_7_1_compatibility_overload_uses_original_abi() {
  static_assert(kNvbufMaxPlanes == 4);
  static_assert(kNvbufStructurePadding == 4);
  static_assert(kNvbufMemSurfaceArray == 4);
  static_assert(sizeof(NvBufSurfacePlaneParams) == sizeof(abi7::PlaneParams));
  static_assert(sizeof(NvBufSurfaceMappedAddr) == sizeof(abi7::MappedAddr));
  static_assert(sizeof(NvBufSurfaceParams) == sizeof(abi7::SurfaceParams));
  static_assert(sizeof(NvBufSurface) == sizeof(abi7::Surface));

  NvBufSurfaceParams public_params;
  public_params.paramex = nullptr;
  expect_eq(public_params.buffer_desc, std::int64_t{-1}, "7.1 public defaults preserved");

  NvmmFrameInfo positional{42, 1920, 1080, 0, 2048, 2048 * 1080, 2048, 2048 * 1620, nullptr};
  expect_eq(positional.dmabuf_fd, 42, "legacy positional frame-info fd preserved");
  expect_eq(positional.width, 1920U, "legacy positional frame-info width preserved");

  auto params = make_params_as<abi7::SurfaceParams>();
  auto surface = make_surface_as<abi7::Surface>(params);
  const auto info = extract_nvmm_frame_info(&surface);
  expect_true(info.abi == NvbufSurfaceAbi::DeepStream7_1, "compatibility overload uses 7.1 ABI");
  expect_eq(info.width, 1920U, "7.1 width");
  expect_eq(info.dmabuf_fd, 42, "7.1 DMA-BUF fd");
}

void nvmm_validation_rejects_bad_metadata() {
  expect_nvmm_error([] { (void)extract_info(nullptr); }, "null mapped data");
  alignas(abi::Surface) std::array<std::byte, sizeof(abi::Surface) + 1> misaligned{};
  expect_nvmm_error([&] { (void)extract_info(misaligned.data() + 1); },
                    "misaligned surface pointer");

  auto params = make_params();
  auto surface = make_surface(params);
  surface.gpu_id = 1;
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "unsupported nonzero GPU id");

  surface = make_surface(params);
  surface.batch_size = 0;
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "empty batch size");

  surface = make_surface(params);
  surface.mem_type = 0;
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "wrong memory type");

  surface = make_surface(params);
  surface.num_filled = 0;
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "empty batch");

  surface = make_surface(params);
  surface.surface_list = nullptr;
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "null surface list");

  params = make_params();
  params.plane_params.num_planes = 1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "missing UV plane");

  params = make_params();
  params.buffer_desc = std::numeric_limits<std::uint64_t>::max();
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "out-of-range dmabuf fd");

  params = make_params();
  params.color_format = 0;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "non-NV12 format");

  params = make_params();
  params.layout = 1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "block-linear layout");

  params = make_params();
  params.plane_params.bytes_per_pix[1] = 1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "bad UV pixel size");

  params = make_params();
  params.plane_params.pitch[1] += 64;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "unequal plane pitch");

  params = make_params();
  params.plane_params.psize[1] -= 1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "short UV allocation");

  params = make_params();
  params.data_size -= 1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "short surface allocation");

  params = make_params();
  params.plane_params.offset[1] -= 1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "overlapping allocations");

  params = make_params();
  params.width = 1919;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_info(&surface); }, "odd NV12 width");
}

void cuda_device_surface_uses_direct_gpu_pointer() {
  auto params = make_params();
  params.data_ptr = reinterpret_cast<void*>(0x10000000);
  auto surface = make_surface(params);
  surface.mem_type = abi::kMemCudaDevice;

  const auto info = extract_info(&surface);
  expect_true(info.memory_type == NvmmMemoryType::CudaDevice, "CUDA-device memory type");
  expect_eq(info.dmabuf_fd, -1, "CUDA-device frame does not invent DMA-BUF fd");
  expect_eq(info.cuda_base_ptr, 0x10000000U, "CUDA-device base pointer");
}

} // namespace

int main() {
  nvmm_extraction_matches_rust_layout();
  deepstream_7_1_compatibility_overload_uses_original_abi();
  nvmm_validation_rejects_bad_metadata();
  cuda_device_surface_uses_direct_gpu_pointer();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
