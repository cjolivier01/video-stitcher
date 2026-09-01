#include "reco/calibrate/gpu_features.hpp"
#include "reco/calibrate/pipeline.hpp"
#include "reco/core/cuda_backend.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

using namespace reco::calibrate;
using namespace reco::core;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
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

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

bool address_sanitizer_build() {
#if defined(__SANITIZE_ADDRESS__)
  return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
  return true;
#else
  return false;
#endif
#else
  return false;
#endif
}

struct DeviceFrame {
  CudaDeviceBuffer pixels;
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  [[nodiscard]] GpuGrayFrame view(std::uint16_t applied_rotation_degrees = 0) const {
    return {.ptr = pixels.ptr(),
            .pitch = width,
            .width = width,
            .height = height,
            .color_range = YuvColorRange::Full,
            .applied_rotation_degrees = applied_rotation_degrees};
  }
};

std::uint8_t texture_value(std::uint32_t global_x, std::uint32_t y) {
  const std::uint32_t cell_x = global_x / 6U;
  const std::uint32_t cell_y = y / 6U;
  std::uint32_t hash = cell_x * 0x9e3779b9U + cell_y * 0x85ebca6bU;
  hash ^= hash >> 16U;
  hash *= 0x7feb352dU;
  hash ^= hash >> 15U;
  return (hash & 1U) == 0U ? std::uint8_t{7} : std::uint8_t{241};
}

DeviceFrame make_frame(CudaBackend& backend, std::uint32_t width, std::uint32_t height,
                       std::uint32_t global_x_offset) {
  std::vector<std::uint8_t> host(static_cast<std::size_t>(width) * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      host[static_cast<std::size_t>(y) * width + x] = texture_value(x + global_x_offset, y);
    }
  }
  auto pixels = backend.allocate(host.size());
  backend.copy_host_to_device_2d({.src = host.data(),
                                  .src_pitch = width,
                                  .dst = pixels.ptr(),
                                  .dst_pitch = width,
                                  .width_bytes = width,
                                  .height = height});
  return {.pixels = std::move(pixels), .width = width, .height = height};
}

struct CopyShape {
  std::size_t width_bytes = 0;
  std::size_t height = 0;
};

struct DeviceToHostTrace final : CudaBackendTraceSink {
  void device_to_host_copy_submitted(std::size_t width_bytes,
                                     std::size_t height) noexcept override {
    if (count == copies.size()) {
      overflow = true;
      return;
    }
    copies[count++] = {.width_bytes = width_bytes, .height = height};
  }

  std::array<CopyShape, 16> copies{};
  std::size_t count = 0;
  bool overflow = false;
};

CameraParams camera(std::uint32_t width, std::uint32_t height) {
  return {.width = width,
          .height = height,
          .fx = 1.0e10,
          .fy = 1.0e10,
          .cx = static_cast<double>(width) * 0.5,
          .cy = static_cast<double>(height) * 0.5,
          .d = {0.0, 0.0, 0.0, 0.0}};
}

CalibrationConfig test_config() {
  CalibrationConfig config;
  config.num_frames = 1;
  config.akaze.threshold = 0.0001;
  config.akaze.max_keypoints = 128;
  config.akaze.detect_y_min = 0.15;
  config.akaze.detect_y_max = 0.85;
  config.matching.lowe_ratio = 1.0;
  config.matching.min_matches = 8;
  config.matching.spatial_x_threshold = 0.5;
  config.matching.spatial_x_inner = 0.0;
  config.matching.spatial_y_low = 0.15;
  config.matching.spatial_y_high = 0.85;
  config.matching.max_y_disparity = 0.02;
  config.matching.ransac_threshold = 1.0;
  config.optimizer.max_iters = 100;
  return config;
}

void calibration_stays_gpu_resident(CudaBackend& backend, std::uint32_t width,
                                    std::uint32_t height) {
  const auto left = make_frame(backend, width, height, 0);
  const auto right = make_frame(backend, width, height, width / 2U);
  auto trace = std::make_shared<DeviceToHostTrace>();
  auto observed = backend.with_trace_sink(trace);
  const std::array pairs{GpuCalibrationFramePairView{.left = left.view(), .right = right.view()}};

  const auto result =
      run_gpu_calibration_frames(observed, std::span<const GpuCalibrationFramePairView>{pairs},
                                 camera(width, height), camera(width, height), test_config(), 7);
  expect_eq(result.frames_used, 1U, "synthetic calibration frame count");
  expect_true(result.total_matches >= 8U, "synthetic calibration match count");
  expect_eq(result.calibration.sync_offset, 7, "synthetic calibration sync offset");
  expect_true(result.quality.has_value(), "synthetic calibration quality metrics");
  expect_true(std::isfinite(result.residual_error), "synthetic calibration residual is finite");

  expect_true(!trace->overflow, "device-to-host trace capacity");
  expect_eq(trace->count, 8U, "only scalar counts and compact matches reach the host");
  std::size_t scalar_copies = 0;
  std::size_t compact_copies = 0;
  for (std::size_t index = 0; index < trace->count; ++index) {
    const auto shape = trace->copies[index];
    expect_eq(shape.height, 1U, "calibration readback is a compact linear transfer");
    if (shape.width_bytes == sizeof(std::uint32_t)) {
      ++scalar_copies;
    } else {
      ++compact_copies;
      expect_eq(shape.width_bytes,
                result.per_frame.front().post_ratio_test * sizeof(GpuMatchedPoint),
                "accepted-match readback has no unused descriptor capacity");
    }
    expect_true(shape.width_bytes < static_cast<std::size_t>(width) * height,
                "calibration never reads a frame back to the host");
  }
  expect_eq(scalar_copies, 7U, "candidate, feature, and compact-match scalar counts are read back");
  expect_eq(compact_copies, 1U, "one accepted-match buffer is read back");
}

void calibration_returns_intrinsics_in_rotated_frame_coordinates(CudaBackend& backend) {
  constexpr std::uint32_t width = 640;
  constexpr std::uint32_t height = 360;
  const auto left = make_frame(backend, width, height, 0);
  const auto right = make_frame(backend, width, height, width / 2U);
  const std::array pairs{
      GpuCalibrationFramePairView{.left = left.view(180), .right = right.view(180)}};
  auto params = camera(width, height);
  params.cx = 311.25;
  params.cy = 172.75;

  const auto result = run_gpu_calibration_frames(
      backend, std::span<const GpuCalibrationFramePairView>{pairs}, params, params, test_config());
  expect_eq(result.calibration.left.cx, 328.75,
            "left calibration cx follows the applied 180-degree rotation");
  expect_eq(result.calibration.left.cy, 187.25,
            "left calibration cy follows the applied 180-degree rotation");
  expect_eq(result.calibration.right.cx, 328.75,
            "right calibration cx follows the applied 180-degree rotation");
  expect_eq(result.calibration.right.cy, 187.25,
            "right calibration cy follows the applied 180-degree rotation");
}

} // namespace

int main() {
  static_assert(sizeof(GpuMatchedPoint) == 28);
  if (address_sanitizer_build() && !require_cuda()) {
    std::cerr << "SKIP: CUDA calibration pipeline test is skipped under ASan unless required\n";
    return EXIT_SUCCESS;
  }
  if (!CudaBackend::is_available()) {
    const auto error = CudaBackend::availability_error();
    if (require_cuda()) {
      std::cerr << "FAIL: CUDA unavailable: " << error << '\n';
      return EXIT_FAILURE;
    }
    std::cerr << "SKIP: CUDA unavailable: " << error << '\n';
    return EXIT_SUCCESS;
  }

  try {
    auto backend = CudaBackend::create();
    calibration_stays_gpu_resident(backend, 640, 360);
    calibration_stays_gpu_resident(backend, 2560, 1440);
    calibration_returns_intrinsics_in_rotated_frame_coordinates(backend);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
