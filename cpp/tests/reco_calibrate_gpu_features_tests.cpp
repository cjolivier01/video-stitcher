#include "reco/calibrate/gpu_features.hpp"
#include "reco/core/cuda_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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
  expect_invalid_argument([&] { (void)pipeline.match(fixture.view(), fixture.view(), 0.0F); },
                          "matcher zero Lowe ratio");
  expect_invalid_argument([&] { (void)pipeline.match(fixture.view(), fixture.view(), 1.01F); },
                          "matcher Lowe ratio above one");
  expect_invalid_argument(
      [&] {
        (void)pipeline.match(fixture.view(), fixture.view(),
                             std::numeric_limits<float>::quiet_NaN());
      },
      "matcher non-finite Lowe ratio");
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

  try {
    auto backend = CudaBackend::create();
    GpuAkazePipeline pipeline(backend);
    const auto validation_frame = make_frame(backend, 160, 96, 176);
    api_validation(backend, pipeline, validation_frame);
    blank_image_has_no_features(backend, pipeline);
    detector_is_bounded_and_deterministic(backend, pipeline);
    luma_range_boundaries_are_equivalent(backend, pipeline);
    resize_boundary_maps_to_source_coordinates(backend, pipeline);
    roi_boundaries_are_enforced(backend, pipeline);
    matcher_ties_lowe_and_crosscheck(backend, pipeline);
    matcher_order_is_stable(backend, pipeline);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
