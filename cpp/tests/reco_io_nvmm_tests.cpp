#include "reco/io/nvmm.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

using namespace reco::io;

namespace {

int failures = 0;

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

NvBufSurfaceParams make_params() {
  NvBufSurfaceParams params;
  params.width = 1920;
  params.height = 1080;
  params.pitch = 2048;
  params.buffer_desc = 42;
  params.plane_params.num_planes = 2;
  params.plane_params.width[0] = 1920;
  params.plane_params.height[0] = 1080;
  params.plane_params.pitch[0] = 2048;
  params.plane_params.offset[0] = 0;
  params.plane_params.width[1] = 1920;
  params.plane_params.height[1] = 540;
  params.plane_params.pitch[1] = 2048;
  params.plane_params.offset[1] = 2048 * 1080;
  return params;
}

NvBufSurface make_surface(NvBufSurfaceParams& params) {
  NvBufSurface surface;
  surface.batch_size = 1;
  surface.num_filled = 1;
  surface.mem_type = kNvbufMemSurfaceArray;
  surface.surface_list = &params;
  return surface;
}

void nvmm_extraction_matches_rust_layout() {
  auto params = make_params();
  auto surface = make_surface(params);
  const auto info = extract_nvmm_frame_info(&surface);
  expect_eq(info.dmabuf_fd, 42, "DMA-buf fd");
  expect_eq(info.width, 1920U, "width");
  expect_eq(info.height, 1080U, "height");
  expect_eq(info.y_offset, 0U, "Y offset");
  expect_eq(info.y_pitch, 2048U, "Y pitch");
  expect_eq(info.uv_offset, 2048U * 1080U, "UV offset");
  expect_eq(info.uv_pitch, 2048U, "UV pitch");
  expect_eq(info.total_size, 2048U * 1620U, "total size");
  expect_eq(info.surface_ptr, static_cast<void*>(&surface), "surface pointer preserved");
}

void nvmm_validation_rejects_bad_metadata() {
  expect_nvmm_error([] { (void)extract_nvmm_frame_info(nullptr); }, "null mapped data");

  auto params = make_params();
  auto surface = make_surface(params);
  surface.batch_size = 0;
  expect_nvmm_error([&] { (void)extract_nvmm_frame_info(&surface); }, "empty batch size");

  surface = make_surface(params);
  surface.mem_type = 0;
  expect_nvmm_error([&] { (void)extract_nvmm_frame_info(&surface); }, "wrong memory type");

  surface = make_surface(params);
  surface.num_filled = 0;
  expect_nvmm_error([&] { (void)extract_nvmm_frame_info(&surface); }, "empty batch");

  surface = make_surface(params);
  surface.surface_list = nullptr;
  expect_nvmm_error([&] { (void)extract_nvmm_frame_info(&surface); }, "null surface list");

  params = make_params();
  params.plane_params.num_planes = 1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_nvmm_frame_info(&surface); }, "missing UV plane");

  params = make_params();
  params.buffer_desc = -1;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_nvmm_frame_info(&surface); }, "negative dmabuf fd");

  params = make_params();
  params.plane_params.offset[1] = std::numeric_limits<std::uint32_t>::max();
  params.plane_params.pitch[1] = 2;
  params.plane_params.height[1] = 2;
  surface = make_surface(params);
  expect_nvmm_error([&] { (void)extract_nvmm_frame_info(&surface); }, "total size overflow");
}

} // namespace

int main() {
  nvmm_extraction_matches_rust_layout();
  nvmm_validation_rejects_bad_metadata();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
