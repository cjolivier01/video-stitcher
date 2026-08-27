#include "reco/calibrate/gpu_features.hpp"
#include "reco/core/cuda_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
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

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_invalid_argument(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

std::string_view test_shard() {
  const char* value = std::getenv("RECO_GPU_FEATURE_TEST_SHARD");
  return value == nullptr || value[0] == '\0' ? std::string_view("all") : std::string_view(value);
}

bool shard_enabled(std::string_view selected, std::string_view shard) {
  return selected == "all" || selected == shard;
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

void upload(CudaBackend& backend, const void* source, std::size_t bytes,
            const CudaDeviceBuffer& destination) {
  backend.copy_host_to_device_2d({.src = source,
                                  .src_pitch = bytes,
                                  .dst = destination.ptr(),
                                  .dst_pitch = bytes,
                                  .width_bytes = bytes,
                                  .height = 1});
}

struct DeviceFrame {
  CudaDeviceBuffer pixels;
  std::size_t pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  [[nodiscard]] GpuGrayFrame view(YuvColorRange range = YuvColorRange::Full) const {
    return {.ptr = pixels.ptr(),
            .pitch = pitch,
            .width = width,
            .height = height,
            .color_range = range};
  }
};

std::vector<std::uint8_t> texture(std::uint32_t width, std::uint32_t height, std::size_t pitch,
                                  std::uint8_t low, std::uint8_t high, std::uint8_t padding) {
  std::vector<std::uint8_t> pixels(pitch * height, padding);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t cell_x = x / 6U;
      const std::uint32_t cell_y = y / 6U;
      std::uint32_t hash = cell_x * 0x9e3779b9U + cell_y * 0x85ebca6bU;
      hash ^= hash >> 16U;
      hash *= 0x7feb352dU;
      hash ^= hash >> 15U;
      pixels[static_cast<std::size_t>(y) * pitch + x] = (hash & 1U) == 0U ? low : high;
    }
  }
  return pixels;
}

DeviceFrame make_frame(CudaBackend& backend, std::uint32_t width, std::uint32_t height,
                       std::size_t pitch, std::uint8_t low = 0, std::uint8_t high = 255,
                       std::uint8_t padding = 0) {
  auto pixels = texture(width, height, pitch, low, high, padding);
  auto device = backend.allocate(pitch * height);
  backend.memset_d8(device, padding);
  backend.copy_host_to_device_2d({.src = pixels.data(),
                                  .src_pitch = pitch,
                                  .dst = device.ptr(),
                                  .dst_pitch = pitch,
                                  .width_bytes = width,
                                  .height = height});
  return {.pixels = std::move(device), .pitch = pitch, .width = width, .height = height};
}

DeviceFrame make_blank_frame(CudaBackend& backend, std::uint32_t width, std::uint32_t height,
                             std::size_t pitch, std::uint8_t value) {
  auto device = backend.allocate(pitch * height);
  backend.memset_d8(device, value);
  return {.pixels = std::move(device), .pitch = pitch, .width = width, .height = height};
}

DeviceFrame make_frame_from_pixels(CudaBackend& backend, std::vector<std::uint8_t> pixels,
                                   std::uint32_t width, std::uint32_t height) {
  const auto bytes = static_cast<std::size_t>(width) * height;
  expect_eq(pixels.size(), bytes, "host frame fixture size");
  auto device = backend.allocate(bytes);
  upload(backend, pixels.data(), bytes, device);
  return {.pixels = std::move(device), .pitch = width, .width = width, .height = height};
}

std::vector<std::uint8_t> rotate_clockwise(const std::vector<std::uint8_t>& source,
                                           std::uint32_t size) {
  std::vector<std::uint8_t> result(source.size());
  for (std::uint32_t y = 0; y < size; ++y) {
    for (std::uint32_t x = 0; x < size; ++x) {
      result[static_cast<std::size_t>(x) * size + (size - 1U - y)] =
          source[static_cast<std::size_t>(y) * size + x];
    }
  }
  return result;
}

std::uint32_t feature_count(CudaBackend& backend, const GpuFeatureSet& features) {
  const auto view = features.view();
  std::uint32_t count = std::numeric_limits<std::uint32_t>::max();
  backend.synchronize();
  backend.copy_device_to_host_2d({.dst = &count,
                                  .dst_pitch = sizeof(count),
                                  .src = view.count,
                                  .src_pitch = sizeof(count),
                                  .width_bytes = sizeof(count),
                                  .height = 1});
  return count;
}

std::vector<Descriptor> feature_descriptors(CudaBackend& backend, const GpuFeatureSet& features) {
  const auto count = feature_count(backend, features);
  std::vector<Descriptor> descriptors(count);
  if (count == 0U) {
    return descriptors;
  }
  const auto bytes = descriptors.size() * sizeof(Descriptor);
  backend.copy_device_to_host_2d({.dst = descriptors.data(),
                                  .dst_pitch = bytes,
                                  .src = features.view().descriptors,
                                  .src_pitch = bytes,
                                  .width_bytes = bytes,
                                  .height = 1});
  return descriptors;
}

std::uint64_t descriptor_hash(const Descriptor& descriptor) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto byte : descriptor) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<GpuFeaturePoint> download_points(CudaBackend& backend, const GpuFeatureSet& features) {
  const auto count = feature_count(backend, features);
  std::vector<GpuFeaturePoint> points(count);
  if (count != 0U) {
    backend.copy_device_to_host_2d({.dst = points.data(),
                                    .dst_pitch = points.size() * sizeof(GpuFeaturePoint),
                                    .src = features.view().points,
                                    .src_pitch = points.size() * sizeof(GpuFeaturePoint),
                                    .width_bytes = points.size() * sizeof(GpuFeaturePoint),
                                    .height = 1});
  }
  return points;
}

GpuAkazeConfig detector_config(std::uint32_t max_keypoints = 32) {
  GpuAkazeConfig config;
  config.max_keypoints = max_keypoints;
  config.threshold = 0.0001F;
  config.border_margin = 0;
  return config;
}

void expect_equivalent_features(CudaBackend& backend, const GpuAkazePipeline& pipeline,
                                const GpuFeatureSet& left, const GpuFeatureSet& right,
                                std::string_view message) {
  const auto left_count = feature_count(backend, left);
  const auto right_count = feature_count(backend, right);
  expect_eq(right_count, left_count, message);
  const auto matches = pipeline.match(left.view(), right.view(), 1.0F);
  expect_eq(matches.size(), static_cast<std::size_t>(left_count), message);
  for (const auto& match : matches) {
    expect_eq(match.distance, 0U, message);
    expect_near(match.left.x, match.right.x, 1.0e-4F, message);
    expect_near(match.left.y, match.right.y, 1.0e-4F, message);
    expect_near(match.left.response, match.right.response, 1.0e-6F, message);
  }
}

struct DeviceFeatures {
  CudaDeviceBuffer points;
  CudaDeviceBuffer descriptors;
  CudaDeviceBuffer count;
  std::uint32_t capacity = 0;

  [[nodiscard]] GpuFeatureView view() const {
    return {.points = points.ptr(),
            .descriptors = descriptors.ptr(),
            .count = count.ptr(),
            .capacity = capacity};
  }
};

DeviceFeatures make_features(CudaBackend& backend, std::span<const GpuFeaturePoint> points,
                             std::span<const Descriptor> descriptors, std::uint32_t capacity = 0) {
  expect_eq(points.size(), descriptors.size(), "feature fixture point/descriptor count");
  if (capacity == 0) {
    capacity = static_cast<std::uint32_t>(points.size());
  }
  expect_true(capacity >= points.size(), "feature fixture capacity");

  auto device_points =
      backend.allocate(static_cast<std::size_t>(capacity) * sizeof(GpuFeaturePoint));
  auto device_descriptors = backend.allocate(static_cast<std::size_t>(capacity) * kDescriptorBytes);
  auto device_count = backend.allocate(sizeof(std::uint32_t));
  backend.memset_d8(device_points, 0);
  backend.memset_d8(device_descriptors, 0);
  if (!points.empty()) {
    upload(backend, points.data(), points.size_bytes(), device_points);
    upload(backend, descriptors.data(), descriptors.size_bytes(), device_descriptors);
  }
  const auto count = static_cast<std::uint32_t>(points.size());
  upload(backend, &count, sizeof(count), device_count);
  return {.points = std::move(device_points),
          .descriptors = std::move(device_descriptors),
          .count = std::move(device_count),
          .capacity = capacity};
}

GpuFeaturePoint point(float x) {
  return {.x = x, .y = x + 0.25F, .response = x + 0.5F, .size = 1.0F};
}

Descriptor filled_descriptor(std::uint8_t value) {
  Descriptor descriptor{};
  descriptor.fill(value);
  return descriptor;
}

void flip_bit(Descriptor& descriptor, std::size_t bit) {
  descriptor[bit / 8U] ^= static_cast<std::uint8_t>(1U << (bit % 8U));
}

void api_validation(CudaBackend& backend, const GpuAkazePipeline& pipeline,
                    const DeviceFrame& valid_frame) {
  GpuFeatureSet empty;
  expect_eq(empty.capacity(), 0U, "default feature set capacity");
  expect_eq(empty.view().points, 0U, "default feature set point pointer");
  expect_eq(empty.view().descriptors, 0U, "default feature set descriptor pointer");
  expect_eq(empty.view().count, 0U, "default feature set count pointer");

  const auto config = detector_config();
  expect_invalid_argument(
      [&] {
        (void)pipeline.detect({.ptr = 0,
                               .pitch = valid_frame.pitch,
                               .width = valid_frame.width,
                               .height = valid_frame.height},
                              config);
      },
      "detector null frame pointer");
  expect_invalid_argument(
      [&] {
        auto frame = valid_frame.view();
        frame.width = 0;
        (void)pipeline.detect(frame, config);
      },
      "detector zero width");
  expect_invalid_argument(
      [&] {
        auto frame = valid_frame.view();
        frame.pitch = frame.width - 1U;
        (void)pipeline.detect(frame, config);
      },
      "detector narrow pitch");
  if constexpr (sizeof(std::size_t) == sizeof(std::uint64_t)) {
    expect_invalid_argument(
        [&] {
          auto frame = valid_frame.view();
          frame.pitch = std::numeric_limits<std::size_t>::max();
          frame.height = std::numeric_limits<std::uint32_t>::max();
          (void)pipeline.detect(frame, config);
        },
        "detector visible span overflow");
  }
  expect_invalid_argument(
      [&] {
        auto frame = valid_frame.view();
        frame.ptr = std::numeric_limits<CudaDevicePtr>::max() - 8U;
        (void)pipeline.detect(frame, config);
      },
      "detector pointer range overflow");
  expect_invalid_argument(
      [&] {
        auto frame = valid_frame.view();
        frame.color_range = static_cast<YuvColorRange>(99);
        (void)pipeline.detect(frame, config);
      },
      "detector invalid luma range");

  auto invalid_config = config;
  invalid_config.max_keypoints = 0;
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector zero output limit");
  invalid_config = config;
  invalid_config.max_keypoints = kMaxGpuAkazeFeatures + 1U;
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector output limit is launch-bounded");
  invalid_config = config;
  invalid_config.max_detection_width = 0;
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector zero resize limit");
  invalid_config = config;
  invalid_config.threshold = 0.0F;
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector zero threshold");
  invalid_config = config;
  invalid_config.threshold = std::numeric_limits<float>::quiet_NaN();
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector non-finite threshold");
  invalid_config = config;
  invalid_config.num_sublevels = 0;
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector zero sublevels");
  invalid_config = config;
  invalid_config.max_octaves = 0;
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector zero octaves");
  invalid_config = config;
  invalid_config.border_margin = kMaxCalibrationDimension + 1U;
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector oversized border margin");
  invalid_config = config;
  invalid_config.use_region = true;
  invalid_config.region = {.x_min = 0.75F, .x_max = 0.25F, .y_min = 0.0F, .y_max = 1.0F};
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector inverted ROI");
  invalid_config = config;
  invalid_config.use_region = true;
  invalid_config.region = {.x_min = -0.01F, .x_max = 1.0F, .y_min = 0.0F, .y_max = 1.0F};
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector ROI outside normalized bounds");
  invalid_config = config;
  invalid_config.use_region = true;
  invalid_config.region = {.x_min = 0.0F,
                           .x_max = 1.0F,
                           .y_min = std::numeric_limits<float>::quiet_NaN(),
                           .y_max = 1.0F};
  expect_invalid_argument([&] { (void)pipeline.detect(valid_frame.view(), invalid_config); },
                          "detector non-finite ROI");
  invalid_config = config;
  invalid_config.use_region = true;
  invalid_config.region = {.x_min = 0.0F, .x_max = 1.0F, .y_min = 0.0F, .y_max = 1.0F};
  invalid_config.max_detection_width = std::numeric_limits<std::uint32_t>::max();
  expect_invalid_argument(
      [&] {
        (void)pipeline.detect({.ptr = 1U,
                               .pitch = std::numeric_limits<std::uint32_t>::max(),
                               .width = std::numeric_limits<std::uint32_t>::max(),
                               .height = 1U},
                              invalid_config);
      },
      "unit ROI endpoint at maximum uint32 dimension remains defined");
  invalid_config = config;
  invalid_config.lowe_ratio = 0.0F;
  expect_invalid_argument(
      [&] {
        (void)pipeline.detect_and_match(valid_frame.view(), valid_frame.view(), invalid_config);
      },
      "detect-and-match zero Lowe ratio");

  const std::array fixture_points{point(1.0F), point(2.0F)};
  const std::array fixture_descriptors{filled_descriptor(0), filled_descriptor(255)};
  const auto fixture = make_features(backend, fixture_points, fixture_descriptors);
  expect_invalid_argument([&] { (void)pipeline.match({}, fixture.view(), 1.0F); },
                          "matcher empty left view");
  auto bad_view = fixture.view();
  bad_view.descriptors = 0;
  expect_invalid_argument([&] { (void)pipeline.match(fixture.view(), bad_view, 1.0F); },
                          "matcher null descriptors");
  bad_view = fixture.view();
  bad_view.count = 0;
  expect_invalid_argument([&] { (void)pipeline.match(bad_view, fixture.view(), 1.0F); },
                          "matcher null count");
  bad_view = fixture.view();
  bad_view.capacity = kMaxGpuAkazeFeatures + 1U;
  expect_invalid_argument([&] { (void)pipeline.match(bad_view, fixture.view(), 1.0F); },
                          "matcher input capacity is launch-bounded");
  bad_view = fixture.view();
  bad_view.points = std::numeric_limits<CudaDevicePtr>::max() - 3U;
  expect_invalid_argument([&] { (void)pipeline.match(bad_view, fixture.view(), 1.0F); },
                          "matcher point pointer range overflow");
  bad_view = fixture.view();
  bad_view.descriptors = std::numeric_limits<CudaDevicePtr>::max() - 7U;
  expect_invalid_argument([&] { (void)pipeline.match(fixture.view(), bad_view, 1.0F); },
                          "matcher descriptor pointer range overflow");
  bad_view = fixture.view();
  bad_view.count = std::numeric_limits<CudaDevicePtr>::max() - 3U;
  expect_invalid_argument([&] { (void)pipeline.match(bad_view, fixture.view(), 1.0F); },
                          "matcher count pointer range overflow");
  expect_invalid_argument([&] { (void)pipeline.match(fixture.view(), fixture.view(), 0.0F); },
                          "matcher zero Lowe ratio");
  expect_invalid_argument([&] { (void)pipeline.match(fixture.view(), fixture.view(), 1.01F); },
                          "matcher Lowe ratio above one");
  expect_invalid_argument(
      [&] {
        (void)pipeline.match(fixture.view(), fixture.view(),
                             std::numeric_limits<double>::quiet_NaN());
      },
      "matcher non-finite Lowe ratio");
}

void pipeline_owns_backend_lifetime(CudaBackend& allocation_backend, const DeviceFrame& frame) {
  const auto make_pipeline = [] {
    auto short_lived_backend = CudaBackend::create();
    return std::make_unique<GpuAkazePipeline>(short_lived_backend);
  };
  const auto pipeline = make_pipeline();
  const auto features = pipeline->detect(frame.view(), detector_config(8));
  expect_true(feature_count(allocation_backend, features) > 0U,
              "pipeline remains usable after constructor backend is destroyed");
}

void blank_image_has_no_features(CudaBackend& backend, const GpuAkazePipeline& pipeline) {
  const auto blank = make_blank_frame(backend, 160, 96, 176, 127);
  const auto features = pipeline.detect(blank.view(), detector_config());
  expect_eq(features.capacity(), 32U, "blank detector capacity");
  expect_eq(feature_count(backend, features), 0U, "blank detector feature count");
  expect_true(pipeline.match(features.view(), features.view(), 1.0F).empty(),
              "blank detector self-match");
}

void detector_is_bounded_and_deterministic(CudaBackend& backend, const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t width = 240;
  constexpr std::uint32_t height = 144;
  const auto packed = make_frame(backend, width, height, width);
  const auto padded_zero = make_frame(backend, width, height, 256, 0, 255, 0);
  const auto padded_ones = make_frame(backend, width, height, 256, 0, 255, 255);
  const auto config = detector_config();

  const auto packed_features = pipeline.detect(packed.view(), config);
  const auto padded_zero_features = pipeline.detect(padded_zero.view(), config);
  const auto padded_ones_features = pipeline.detect(padded_ones.view(), config);
  const auto repeated_features = pipeline.detect(packed.view(), config);
  const auto count = feature_count(backend, packed_features);
  expect_true(count > 3U, "textured detector produces enough features to exercise cap");
  expect_true(count <= config.max_keypoints, "detector output does not exceed cap");
  expect_eq(packed_features.capacity(), config.max_keypoints, "detector allocation capacity");
  expect_equivalent_features(backend, pipeline, packed_features, padded_zero_features,
                             "packed and zero-padded feature invariance");
  expect_equivalent_features(backend, pipeline, padded_zero_features, padded_ones_features,
                             "padding byte invariance");
  expect_equivalent_features(backend, pipeline, packed_features, repeated_features,
                             "detector deterministic repeat");

  const auto capped = pipeline.detect(packed.view(), detector_config(3));
  expect_eq(capped.capacity(), 3U, "small detector cap allocation");
  expect_eq(feature_count(backend, capped), 3U, "small detector cap enforced");
}

void luma_range_boundaries_are_equivalent(CudaBackend& backend, const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t width = 240;
  constexpr std::uint32_t height = 144;
  const auto full = make_frame(backend, width, height, 256, 0, 255, 73);
  const auto limited = make_frame(backend, width, height, 256, 16, 235, 191);
  const auto config = detector_config();
  const auto full_features = pipeline.detect(full.view(YuvColorRange::Full), config);
  const auto limited_features = pipeline.detect(limited.view(YuvColorRange::Limited), config);
  expect_true(feature_count(backend, full_features) > 0U, "full-range boundary texture detected");
  expect_equivalent_features(backend, pipeline, full_features, limited_features,
                             "full and limited luma endpoint equivalence");
}

void resize_boundary_maps_to_source_coordinates(CudaBackend& backend,
                                                const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t height = 96;
  const auto at_limit = make_frame(backend, 1920, height, 1936, 0, 255, 37);
  const auto over_limit = make_frame(backend, 1921, height, 1952, 0, 255, 219);
  auto config = detector_config(16);
  config.max_detection_width = 1920;

  for (const auto* frame : {&at_limit, &over_limit}) {
    const auto features = pipeline.detect(frame->view(), config);
    const auto count = feature_count(backend, features);
    expect_true(count > 0U, "resize-boundary detector produces features");
    expect_true(count <= config.max_keypoints, "resize-boundary detector observes cap");
    const auto matches = pipeline.match(features.view(), features.view(), 1.0F);
    expect_eq(matches.size(), static_cast<std::size_t>(count),
              "resize-boundary self-match covers detector output");
    for (const auto& match : matches) {
      expect_true(match.left.x >= 0.0F && match.left.x < static_cast<float>(frame->width),
                  "resize-boundary x maps to source extent");
      expect_true(match.left.y >= 0.0F && match.left.y < static_cast<float>(frame->height),
                  "resize-boundary y maps to source extent");
      expect_near(match.left.x, match.right.x, 1.0e-4F, "resize-boundary self-match x");
      expect_near(match.left.y, match.right.y, 1.0e-4F, "resize-boundary self-match y");
    }
  }
}

void triangle_downscale_matches_rust_golden(CudaBackend& backend,
                                            const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t width = 3840;
  constexpr std::uint32_t height = 192;
  const auto frame = make_frame(backend, width, height, width);
  auto config = detector_config(16);
  config.max_detection_width = 1920;
  const auto features = pipeline.detect(frame.view(), config);
  const auto points = download_points(backend, features);
  const auto descriptors = feature_descriptors(backend, features);
  expect_eq(points.size(), 16U, "Triangle golden feature count");
  expect_eq(descriptors.size(), 16U, "Triangle golden descriptor count");

  struct GoldenPoint {
    float x;
    float y;
    float response;
  };
  constexpr std::array<GoldenPoint, 16> rust_golden{{
      {3490.589111328F, 130.976364136F, 0.078354820609F},
      {953.287475586F, 77.089065552F, 0.073465585709F},
      {2177.023437500F, 106.958404541F, 0.072100788355F},
      {1012.978332520F, 98.228263855F, 0.070630028844F},
      {1355.276611328F, 131.111114502F, 0.068403765559F},
      {2870.102294922F, 86.022186279F, 0.068134702742F},
      {3158.328369141F, 91.927497864F, 0.066853381693F},
      {3143.578857422F, 94.641464233F, 0.065145090222F},
      {292.626037598F, 95.571952820F, 0.065043181181F},
      {2452.126708984F, 85.024482727F, 0.064136497676F},
      {828.253112793F, 96.716560364F, 0.063492082059F},
      {2896.912353516F, 77.008834839F, 0.062778681517F},
      {803.351623535F, 101.150360107F, 0.060621485114F},
      {904.586303711F, 58.997024536F, 0.060412760824F},
      {3274.332275391F, 95.149536133F, 0.060264542699F},
      {1930.680053711F, 88.560333252F, 0.059688717127F},
  }};
  constexpr std::array<std::uint64_t, 16> rust_descriptor_hashes{
      0x99f653f77cb98a73ULL, 0x3d3e8369d72ef5d5ULL, 0x40cd7da97675c37cULL, 0x81b4800b8888c3f6ULL,
      0xccdff312e6eee8e1ULL, 0xdd6e82f6407f81eeULL, 0x8f307f8eed986d7cULL, 0x8e82ac0dd59cd1aaULL,
      0x6bc27d7b0c3e15d5ULL, 0xbdc1eede28da1165ULL, 0x1b420c6acfccd414ULL, 0x9b680d4cdabf0d45ULL,
      0x56d1b7460e2f4672ULL, 0x32d837829dfed7f0ULL, 0x8ba3aaad96de75a4ULL, 0xf849fab312ab82e0ULL,
  };
  for (std::size_t index = 0; index < std::min(points.size(), rust_golden.size()); ++index) {
    expect_near(points[index].x, rust_golden[index].x, 2.0e-4F, "Triangle Rust-golden feature x");
    expect_near(points[index].y, rust_golden[index].y, 2.0e-4F, "Triangle Rust-golden feature y");
    expect_near(points[index].response, rust_golden[index].response, 1.0e-7F,
                "Triangle Rust-golden feature response");
    if (index < descriptors.size()) {
      expect_eq(descriptor_hash(descriptors[index]), rust_descriptor_hashes[index],
                "Triangle Rust-golden descriptor");
    }
  }
}

void crop_and_non_power_resize_matches_rust_golden(CudaBackend& backend,
                                                   const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t width = 5003;
  constexpr std::uint32_t height = 384;
  const auto frame = make_frame(backend, width, height, 5024);
  auto config = detector_config(5);
  config.max_detection_width = 1920;
  config.use_region = true;
  config.region = {.x_min = 0.25F, .x_max = 0.75F, .y_min = 0.2F, .y_max = 0.8F};
  const auto features = pipeline.detect(frame.view(), config);
  const auto points = download_points(backend, features);
  const auto descriptors = feature_descriptors(backend, features);
  expect_eq(points.size(), 5U, "crop-resize Rust-golden feature count");
  expect_eq(descriptors.size(), 5U, "crop-resize Rust-golden descriptor count");

  struct GoldenFeature {
    float x;
    float y;
    float response;
    std::uint64_t descriptor_hash;
  };
  constexpr std::array<GoldenFeature, 5> rust_golden{{
      {2081.472167968750F, 191.047607421875F, 0.125226318836F, 0x13c72e999d478afcULL},
      {3491.037841796875F, 131.094146728516F, 0.118464715779F, 0x2344d1de8c293859ULL},
      {2177.248779296875F, 107.127502441406F, 0.113023869693F, 0xda39e887f2406c31ULL},
      {2135.087158203125F, 167.164428710938F, 0.110071659088F, 0xa3afa2b01fabbf11ULL},
      {1355.444580078125F, 131.047775268555F, 0.108458556235F, 0x026495fbdf9cb253ULL},
  }};
  for (std::size_t index = 0; index < std::min(points.size(), rust_golden.size()); ++index) {
    expect_near(points[index].x, rust_golden[index].x, 4.0e-4F,
                "crop-resize Rust-golden feature x");
    expect_near(points[index].y, rust_golden[index].y, 4.0e-4F,
                "crop-resize Rust-golden feature y");
    expect_near(points[index].response, rust_golden[index].response, 1.0e-7F,
                "crop-resize Rust-golden feature response");
    if (index < descriptors.size()) {
      expect_eq(descriptor_hash(descriptors[index]), rust_golden[index].descriptor_hash,
                "crop-resize Rust-golden descriptor");
    }
  }
}

void single_sublevel_multi_octave_matches_rust_golden(CudaBackend& backend,
                                                      const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t size = 192;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(size) * size);
  for (std::uint32_t y = 0; y < size; ++y) {
    for (std::uint32_t x = 0; x < size; ++x) {
      std::uint32_t hash = x * 0x9e3779b9U + y * 0x85ebca6bU;
      hash ^= hash >> 16U;
      hash *= 0x7feb352dU;
      hash ^= hash >> 15U;
      pixels[static_cast<std::size_t>(y) * size + x] = static_cast<std::uint8_t>(hash >> 24U);
    }
  }
  const auto frame = make_frame_from_pixels(backend, std::move(pixels), size, size);
  auto config = detector_config(8);
  config.num_sublevels = 1;
  config.max_octaves = 2;
  const auto features = pipeline.detect(frame.view(), config);
  const auto points = download_points(backend, features);
  const auto descriptors = feature_descriptors(backend, features);

  struct GoldenFeature {
    float x;
    float y;
    float response;
    std::uint64_t descriptor_hash;
  };
  constexpr std::array<GoldenFeature, 8> rust_golden{{
      {145.971405029297F, 91.091178894043F, 0.007582044229F, 0xaad27300bf98bf85ULL},
      {139.686935424805F, 70.186004638672F, 0.006565546151F, 0x5543f2d204f5066aULL},
      {112.508705139160F, 114.866546630859F, 0.006493865512F, 0x5ef3c7c53a732e1aULL},
      {109.104331970215F, 50.350242614746F, 0.006122364663F, 0x63912f75a777a366ULL},
      {31.625604629517F, 117.065246582031F, 0.005789659452F, 0x1fac9f6c2818eca1ULL},
      {139.708633422852F, 99.887496948242F, 0.005627152510F, 0x9d13b0957795803dULL},
      {96.377853393555F, 29.344636917114F, 0.005588544067F, 0x00a4e28fdc353062ULL},
      {113.278808593750F, 77.035087585449F, 0.005495154765F, 0x5edb53ab09ceee26ULL},
  }};
  expect_eq(points.size(), rust_golden.size(), "single-sublevel Rust-golden feature count");
  expect_eq(descriptors.size(), rust_golden.size(), "single-sublevel Rust-golden descriptor count");
  for (std::size_t index = 0; index < std::min(points.size(), rust_golden.size()); ++index) {
    expect_near(points[index].x, rust_golden[index].x, 2.0e-4F,
                "single-sublevel Rust-golden feature x");
    expect_near(points[index].y, rust_golden[index].y, 2.0e-4F,
                "single-sublevel Rust-golden feature y");
    expect_near(points[index].response, rust_golden[index].response, 1.0e-7F,
                "single-sublevel Rust-golden feature response");
    if (index < descriptors.size()) {
      expect_eq(descriptor_hash(descriptors[index]), rust_golden[index].descriptor_hash,
                "single-sublevel Rust-golden descriptor");
    }
  }
}

void adversarial_selection_is_bounded_at_resolution(CudaBackend& backend,
                                                    const GpuAkazePipeline& pipeline,
                                                    std::uint32_t width, std::uint32_t height,
                                                    std::size_t pitch,
                                                    std::chrono::seconds maximum_latency) {
  std::vector<std::uint8_t> pixels(pitch * height, 0);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      std::uint32_t hash = x * 0x9e3779b9U + y * 0x85ebca6bU;
      hash ^= hash >> 16U;
      hash *= 0x7feb352dU;
      hash ^= hash >> 15U;
      pixels[static_cast<std::size_t>(y) * pitch + x] = static_cast<std::uint8_t>(hash >> 24U);
    }
  }
  auto device = backend.allocate(pitch * height);
  backend.copy_host_to_device_2d({.src = pixels.data(),
                                  .src_pitch = pitch,
                                  .dst = device.ptr(),
                                  .dst_pitch = pitch,
                                  .width_bytes = width,
                                  .height = height});
  const DeviceFrame frame{
      .pixels = std::move(device), .pitch = pitch, .width = width, .height = height};
  auto config = detector_config(64);
  config.threshold = 1.0e-8F;
  config.max_detection_width = width;
  config.num_sublevels = 2;
  config.max_octaves = 3;

  const auto started = std::chrono::steady_clock::now();
  const auto features = pipeline.detect(frame.view(), config);
  backend.synchronize();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  expect_eq(feature_count(backend, features), config.max_keypoints,
            "adversarial detector reaches its bounded output cap");
  expect_true(elapsed < maximum_latency,
              "adversarial selection completes without watchdog-scale latency");
}

void full_resolution_adversarial_selection_is_bounded(CudaBackend& backend,
                                                      const GpuAkazePipeline& pipeline) {
  adversarial_selection_is_bounded_at_resolution(backend, pipeline, 1920, 1080, 1936,
                                                 std::chrono::seconds(10));
}

void racecheck_adversarial_selection_is_bounded(CudaBackend& backend,
                                                const GpuAkazePipeline& pipeline) {
  adversarial_selection_is_bounded_at_resolution(backend, pipeline, 480, 270, 496,
                                                 std::chrono::seconds(5));
}

void roi_boundaries_are_enforced(CudaBackend& backend, const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t width = 320;
  constexpr std::uint32_t height = 192;
  const auto frame = make_frame(backend, width, height, 336, 0, 255, 0xA5);
  auto config = detector_config(32);
  config.use_region = true;
  config.region = {.x_min = 0.25F, .x_max = 0.75F, .y_min = 0.25F, .y_max = 0.75F};
  const auto features = pipeline.detect(frame.view(), config);
  const auto count = feature_count(backend, features);
  expect_true(count > 0U, "ROI detector produces features");
  const auto matches = pipeline.match(features.view(), features.view(), 1.0F);
  expect_eq(matches.size(), static_cast<std::size_t>(count),
            "ROI self-match covers detector output");
  const float x_min = config.region.x_min * static_cast<float>(width);
  const float x_max = config.region.x_max * static_cast<float>(width);
  const float y_min = config.region.y_min * static_cast<float>(height);
  const float y_max = config.region.y_max * static_cast<float>(height);
  for (const auto& match : matches) {
    expect_true(match.left.x >= x_min && match.left.x <= x_max, "ROI inclusive x boundaries");
    expect_true(match.left.y >= y_min && match.left.y <= y_max, "ROI inclusive y boundaries");
  }

  config.region = {.x_min = 0.0F, .x_max = 1.0F, .y_min = 0.0F, .y_max = 1.0F};
  const auto full_roi = pipeline.detect(frame.view(), config);
  expect_true(feature_count(backend, full_roi) > 0U, "unit ROI boundaries accepted");
}

void rotated_descriptors_match_known_vectors(CudaBackend& backend,
                                             const GpuAkazePipeline& pipeline) {
  constexpr std::uint32_t size = 192;
  auto pixels = texture(size, size, size, 7, 241, 0);
  const auto config = detector_config(8);
  constexpr std::array<std::uint64_t, 4> expected_hashes{
      0x3e73e22b19638e8fULL,
      0xa90e0344e8c1aa8aULL,
      0x46358dee951959e8ULL,
      0xa686a8518d294ee7ULL,
  };
  std::array<std::uint64_t, 4> hashes{};
  for (std::size_t rotation = 0; rotation < hashes.size(); ++rotation) {
    const auto frame = make_frame_from_pixels(backend, pixels, size, size);
    const auto features = pipeline.detect(frame.view(), config);
    const auto descriptors = feature_descriptors(backend, features);
    expect_eq(descriptors.size(), static_cast<std::size_t>(config.max_keypoints),
              "rotated golden descriptor count");
    if (!descriptors.empty()) {
      hashes[rotation] = descriptor_hash(descriptors.front());
    }
    for (const auto& descriptor : descriptors) {
      expect_eq(static_cast<unsigned int>(descriptor[60] & 0xC0U), 0U,
                "M-LDB bits above populated bit 485 are zero");
      expect_eq(static_cast<unsigned int>(descriptor[61]), 0U,
                "M-LDB byte 61 is outside the 486 populated bits");
      expect_eq(static_cast<unsigned int>(descriptor[62]), 0U,
                "M-LDB byte 62 is outside the 486 populated bits");
      expect_eq(static_cast<unsigned int>(descriptor[63]), 0U,
                "M-LDB byte 63 is outside the 486 populated bits");
    }
    pixels = rotate_clockwise(pixels, size);
  }

  for (std::size_t rotation = 0; rotation < hashes.size(); ++rotation) {
    expect_eq(hashes[rotation], expected_hashes[rotation], "rotated M-LDB descriptor known vector");
  }
}

void matcher_ties_lowe_and_crosscheck(CudaBackend& backend, const GpuAkazePipeline& pipeline) {
  Descriptor one_bit_a{};
  Descriptor one_bit_b{};
  flip_bit(one_bit_a, 0);
  flip_bit(one_bit_b, 1);
  const std::array tie_left_points{point(10.0F)};
  const std::array<Descriptor, 1> tie_left_descriptors{Descriptor{}};
  const std::array tie_right_points{point(20.0F), point(30.0F)};
  const std::array tie_right_descriptors{one_bit_a, one_bit_b};
  const auto tie_left = make_features(backend, tie_left_points, tie_left_descriptors);
  const auto tie_right = make_features(backend, tie_right_points, tie_right_descriptors);

  const auto tie_without_lowe = pipeline.match(tie_left.view(), tie_right.view(), 1.0F);
  expect_eq(tie_without_lowe.size(), 1U, "equal-distance match retained when Lowe is disabled");
  if (tie_without_lowe.size() == 1) {
    expect_eq(tie_without_lowe[0].distance, 1U, "equal-distance winner distance");
    expect_near(tie_without_lowe[0].right.x, 20.0F, 0.0F, "equal-distance lowest train index wins");
  }
  expect_true(pipeline.match(tie_left.view(), tie_right.view(), 0.75F).empty(),
              "equal best is included as second-best for Lowe rejection");

  const std::array duplicate_left_points{point(40.0F), point(50.0F)};
  const std::array duplicate_left_descriptors{Descriptor{}, Descriptor{}};
  const std::array asymmetric_right_points{point(60.0F), point(70.0F)};
  const std::array asymmetric_right_descriptors{Descriptor{}, filled_descriptor(255)};
  const auto duplicate_left =
      make_features(backend, duplicate_left_points, duplicate_left_descriptors);
  const auto asymmetric_right =
      make_features(backend, asymmetric_right_points, asymmetric_right_descriptors);
  expect_true(pipeline.match(duplicate_left.view(), asymmetric_right.view(), 0.75F).empty(),
              "reverse Lowe failure removes forward-accepted match");
  const auto asymmetric_no_lowe =
      pipeline.match(duplicate_left.view(), asymmetric_right.view(), 1.0F);
  expect_eq(asymmetric_no_lowe.size(), 1U, "Lowe bypass still applies mutual cross-check");
  if (asymmetric_no_lowe.size() == 1) {
    expect_near(asymmetric_no_lowe[0].left.x, 40.0F, 0.0F, "cross-check keeps only reverse winner");
  }

  Descriptor close{};
  flip_bit(close, 0);
  const std::array cross_left_points{point(80.0F), point(90.0F)};
  const std::array cross_left_descriptors{Descriptor{}, close};
  const std::array cross_right_points{point(100.0F)};
  const std::array<Descriptor, 1> cross_right_descriptors{Descriptor{}};
  const auto cross_left = make_features(backend, cross_left_points, cross_left_descriptors);
  const auto cross_right = make_features(backend, cross_right_points, cross_right_descriptors);
  const auto cross_matches = pipeline.match(cross_left.view(), cross_right.view(), 1.0F);
  expect_eq(cross_matches.size(), 1U, "non-mutual nearest match removed");
  if (cross_matches.size() == 1) {
    expect_near(cross_matches[0].left.x, 80.0F, 0.0F, "mutual nearest match retained");
  }
}

void matcher_order_is_stable(CudaBackend& backend, const GpuAkazePipeline& pipeline) {
  const Descriptor left_zero{};
  const auto left_ones = filled_descriptor(255);
  const auto left_alternating = filled_descriptor(0xAA);
  auto right_zero = left_zero;
  auto right_ones = left_ones;
  auto right_alternating = left_alternating;
  for (std::size_t bit = 0; bit < 3; ++bit) {
    flip_bit(right_zero, bit);
    flip_bit(right_alternating, bit);
  }
  flip_bit(right_ones, 0);

  const std::array left_points{point(10.0F), point(20.0F), point(30.0F)};
  const std::array left_descriptors{left_zero, left_ones, left_alternating};
  const std::array right_points{point(110.0F), point(120.0F), point(130.0F)};
  const std::array right_descriptors{right_zero, right_ones, right_alternating};
  const auto left = make_features(backend, left_points, left_descriptors, 7);
  const auto right = make_features(backend, right_points, right_descriptors, 5);

  const auto first = pipeline.match(left.view(), right.view(), 1.0F);
  const auto second = pipeline.match(left.view(), right.view(), 1.0F);
  expect_eq(first.size(), 3U, "stable-order fixture match count");
  expect_eq(second.size(), first.size(), "stable-order deterministic repeat count");
  const std::array expected_left_x{20.0F, 10.0F, 30.0F};
  const std::array<std::uint32_t, 3> expected_distance{1U, 3U, 3U};
  for (std::size_t i = 0; i < std::min(first.size(), expected_left_x.size()); ++i) {
    expect_near(first[i].left.x, expected_left_x[i], 0.0F,
                "matches sorted by distance then left index");
    expect_eq(first[i].distance, expected_distance[i], "stable-order match distance");
    expect_near(second[i].left.x, first[i].left.x, 0.0F, "matcher deterministic repeat order");
    expect_eq(second[i].distance, first[i].distance, "matcher deterministic repeat distance");
  }
}

void matcher_preserves_double_ratio_boundaries(CudaBackend& backend,
                                               const GpuAkazePipeline& pipeline) {
  Descriptor right_best{};
  Descriptor right_second{};
  flip_bit(right_best, 0);
  for (std::size_t bit = 0; bit < 3; ++bit) {
    flip_bit(right_second, bit);
  }
  const std::array left_points{point(1.0F), point(2.0F)};
  const std::array left_descriptors{Descriptor{}, filled_descriptor(255)};
  const std::array right_points{point(3.0F), point(4.0F)};
  const std::array right_descriptors{right_best, right_second};
  const auto left = make_features(backend, left_points, left_descriptors);
  const auto right = make_features(backend, right_points, right_descriptors);
  constexpr double boundary = 1.0 / 3.0;
  const auto below = std::nextafter(boundary, 0.0);
  const auto above = std::nextafter(std::nextafter(boundary, 1.0), 1.0);
  expect_true(pipeline.match(left.view(), right.view(), below).empty(),
              "Lowe ratio immediately below an integer distance boundary rejects");
  const auto accepted = pipeline.match(left.view(), right.view(), above);
  expect_eq(accepted.size(), 1U,
            "Lowe ratio immediately above an integer distance boundary accepts");
  if (accepted.size() == 1U) {
    expect_eq(accepted.front().distance, 1U, "double-boundary accepted match distance");
    expect_near(accepted.front().left.x, 1.0F, 0.0F, "double-boundary accepted match identity");
  }
}

} // namespace

int main() {
  static_assert(sizeof(GpuFeaturePoint) == 24);
  static_assert(sizeof(Descriptor) == kDescriptorBytes);

  if (address_sanitizer_build() && !require_cuda()) {
    std::cerr << "SKIP: CUDA AKAZE tests are skipped under ASan unless explicitly required\n";
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

  const auto shard = test_shard();
  if (shard != "all" && shard != "contracts" && shard != "detection" && shard != "golden" &&
      shard != "triangle" && shard != "crop" && shard != "contrast" && shard != "selection" &&
      shard != "selection-race" && shard != "selection-full" && shard != "rotation" &&
      shard != "matching") {
    std::cerr << "FAIL: unknown CUDA feature test shard: " << shard << '\n';
    return EXIT_FAILURE;
  }

  try {
    auto backend = CudaBackend::create();
    GpuAkazePipeline pipeline(backend);
    if (shard_enabled(shard, "contracts")) {
      const auto validation_frame = make_frame(backend, 160, 96, 176);
      api_validation(backend, pipeline, validation_frame);
      pipeline_owns_backend_lifetime(backend, validation_frame);
      blank_image_has_no_features(backend, pipeline);
    }
    if (shard_enabled(shard, "detection")) {
      detector_is_bounded_and_deterministic(backend, pipeline);
      luma_range_boundaries_are_equivalent(backend, pipeline);
      resize_boundary_maps_to_source_coordinates(backend, pipeline);
      roi_boundaries_are_enforced(backend, pipeline);
    }
    if (shard_enabled(shard, "golden") || shard == "triangle") {
      triangle_downscale_matches_rust_golden(backend, pipeline);
    }
    if (shard_enabled(shard, "golden") || shard == "crop") {
      crop_and_non_power_resize_matches_rust_golden(backend, pipeline);
    }
    if (shard_enabled(shard, "golden") || shard == "contrast") {
      single_sublevel_multi_octave_matches_rust_golden(backend, pipeline);
    }
    if (shard == "selection-full") {
      full_resolution_adversarial_selection_is_bounded(backend, pipeline);
    }
    if (shard == "all" || shard == "selection" || shard == "selection-race") {
      racecheck_adversarial_selection_is_bounded(backend, pipeline);
    }
    if (shard_enabled(shard, "golden") || shard == "rotation") {
      rotated_descriptors_match_known_vectors(backend, pipeline);
    }
    if (shard_enabled(shard, "matching")) {
      matcher_ties_lowe_and_crosscheck(backend, pipeline);
      matcher_order_is_stable(backend, pipeline);
      matcher_preserves_double_ratio_boundaries(backend, pipeline);
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
