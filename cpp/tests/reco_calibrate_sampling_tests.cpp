#include "reco/calibrate/sampling.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace reco::calibrate;

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

void select_frame_indices_match_rust() {
  const auto basic = select_frame_indices(1000, 30.0, 5, 0.0, 0.0);
  expect_eq(basic.size(), 5U, "basic sample count");
  expect_eq(basic[0], 140U, "basic first midpoint");
  expect_eq(basic[4], 860U, "basic last midpoint");
  for (std::size_t i = 1; i < basic.size(); ++i) {
    expect_true(basic[i - 1] < basic[i], "basic samples are sorted");
  }

  const auto skipped = select_frame_indices(1000, 30.0, 5, 10.0, 5.0);
  expect_eq(skipped.size(), 5U, "skip sample count");
  expect_eq(skipped[0], 355U, "skip first midpoint");
  expect_eq(skipped[4], 795U, "skip last midpoint");

  const auto short_video = select_frame_indices(10, 30.0, 5, 0.0, 0.0);
  expect_eq(short_video.size(), 5U, "short sample count");
  expect_eq(short_video[0], 0U, "short first sample");
  expect_eq(short_video[4], 8U, "short last sample");

  expect_true(select_frame_indices(0, 30.0, 5, 0.0, 0.0).empty(), "zero frames");
  expect_true(select_frame_indices(100, 30.0, 0, 0.0, 0.0).empty(), "zero samples");
  expect_true(select_frame_indices(100, 0.0, 5, 0.0, 0.0).empty(), "zero fps");

  const auto skipped_all = select_frame_indices(100, 30.0, 5, 10.0, 0.0);
  expect_eq(skipped_all.size(), 1U, "empty usable range falls back to midpoint");
  expect_eq(skipped_all[0], 50U, "empty usable midpoint");
}

void synchronized_indices_stay_in_stream_bounds() {
  const auto [positive_left, positive_right] =
      select_synchronized_frame_indices(100, 80, 10.0, 3, 0.0, 0.0, 7);
  expect_eq(positive_left.size(), 3U, "positive sync pair count");
  expect_eq(positive_right.size(), positive_left.size(), "positive sync vectors align");
  for (std::size_t index = 0; index < positive_left.size(); ++index) {
    expect_eq(positive_right[index], positive_left[index] + 7U,
              "positive sync offsets right indices");
    expect_true(positive_left[index] < 100U, "positive sync left index in range");
    expect_true(positive_right[index] < 80U, "positive sync right index in range");
  }

  const auto [negative_left, negative_right] =
      select_synchronized_frame_indices(60, 100, 10.0, 3, 0.0, 0.0, -9);
  expect_eq(negative_left.size(), 3U, "negative sync pair count");
  for (std::size_t index = 0; index < negative_left.size(); ++index) {
    expect_eq(negative_left[index], negative_right[index] + 9U,
              "negative sync offsets left indices");
    expect_true(negative_left[index] < 60U, "negative sync left index in range");
    expect_true(negative_right[index] < 100U, "negative sync right index in range");
  }

  const auto [empty_left, empty_right] =
      select_synchronized_frame_indices(5, 7, 30.0, 2, 0.0, 0.0, 7);
  expect_true(empty_left.empty() && empty_right.empty(), "offset beyond stream has no pairs");
}

void downscale_matches_rust() {
  GrayFrame small{.data = std::vector<std::uint8_t>(100 * 100, 128), .width = 100, .height = 100};
  const auto identity = downscale_if_needed(small, 1920);
  expect_eq(identity.width, 100U, "identity width");
  expect_eq(identity.height, 100U, "identity height");
  expect_eq(identity.data.size(), 10000U, "identity data size");
  expect_eq(identity.data[0], 128U, "identity data value");

  GrayFrame frame{.data = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150},
                  .width = 4,
                  .height = 4};
  const auto scaled = downscale_if_needed(frame, 2);
  expect_eq(scaled.width, 2U, "downscale width");
  expect_eq(scaled.height, 2U, "downscale height");
  expect_eq(scaled.data.size(), 4U, "downscale data size");
  expect_eq(scaled.data[0], 25U, "top-left box average");
  expect_eq(scaled.data[1], 45U, "top-right box average");
  expect_eq(scaled.data[2], 105U, "bottom-left box average");
  expect_eq(scaled.data[3], 125U, "bottom-right box average");

  bool zero_target_threw = false;
  try {
    (void)downscale_if_needed(frame, 0);
  } catch (const std::invalid_argument&) {
    zero_target_threw = true;
  }
  expect_true(zero_target_threw, "zero target width throws");

  bool identity_zero_target_threw = false;
  try {
    (void)downscale_if_needed(GrayFrame{.data = {}, .width = 0, .height = 0}, 0);
  } catch (const std::invalid_argument&) {
    identity_zero_target_threw = true;
  }
  expect_true(identity_zero_target_threw, "identity path zero target width throws");

  bool short_buffer_threw = false;
  try {
    (void)downscale_if_needed(GrayFrame{.data = {1, 2, 3}, .width = 4, .height = 4}, 2);
  } catch (const std::out_of_range&) {
    short_buffer_threw = true;
  }
  expect_true(short_buffer_threw, "short buffer throws");

  bool identity_short_buffer_threw = false;
  try {
    (void)downscale_if_needed(GrayFrame{.data = {1, 2, 3}, .width = 4, .height = 4}, 8);
  } catch (const std::out_of_range&) {
    identity_short_buffer_threw = true;
  }
  expect_true(identity_short_buffer_threw, "identity path short buffer throws");
}

void applied_rotation_transforms_off_center_intrinsics() {
  const reco::core::CameraParams camera{.width = 1920,
                                        .height = 1080,
                                        .fx = 970.0,
                                        .fy = 965.0,
                                        .cx = 931.25,
                                        .cy = 517.75,
                                        .d = {0.1, -0.02, 0.003, -0.0004}};
  const auto unchanged = camera_params_after_applied_rotation(camera, 0);
  expect_eq(unchanged.cx, camera.cx, "zero rotation preserves cx");
  expect_eq(unchanged.cy, camera.cy, "zero rotation preserves cy");

  const auto rotated = camera_params_after_applied_rotation(camera, 180);
  expect_eq(rotated.width, camera.width, "180-degree rotation preserves width");
  expect_eq(rotated.height, camera.height, "180-degree rotation preserves height");
  expect_eq(rotated.fx, camera.fx, "180-degree rotation preserves fx");
  expect_eq(rotated.fy, camera.fy, "180-degree rotation preserves fy");
  expect_eq(rotated.cx, 988.75, "180-degree rotation mirrors cx about the image center");
  expect_eq(rotated.cy, 562.25, "180-degree rotation mirrors cy about the image center");
  expect_true(rotated.d == camera.d, "180-degree rotation preserves radial distortion");

  bool unsupported_threw = false;
  try {
    (void)camera_params_after_applied_rotation(camera, 90);
  } catch (const std::invalid_argument&) {
    unsupported_threw = true;
  }
  expect_true(unsupported_threw, "dimension-swapping camera rotation fails closed");
}

} // namespace

int main() {
  select_frame_indices_match_rust();
  synchronized_indices_stay_in_stream_bounds();
  downscale_matches_rust();
  applied_rotation_transforms_off_center_intrinsics();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
