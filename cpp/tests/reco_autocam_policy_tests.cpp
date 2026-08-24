#include "reco/autocam/coaster.hpp"
#include "reco/autocam/roi_filter.hpp"
#include "reco/autocam/sweep_panner.hpp"
#include "reco/autocam/tracking_mode.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace reco::autocam;

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
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

Detection detection(reco::core::CameraId camera, std::uint16_t class_id, float cx, float cy,
                    float w, float h) {
  return {.camera = camera,
          .class_id = class_id,
          .confidence = 0.9F,
          .center_x = cx,
          .center_y = cy,
          .width = w,
          .height = h};
}

reco::core::FieldRoi full_roi() {
  return {
      .left = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}},
      .right = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}},
  };
}

reco::core::FieldRoi small_roi() {
  return {
      .left = {{0.2, 0.2}, {0.8, 0.2}, {0.8, 0.8}, {0.2, 0.8}},
      .right = {{0.2, 0.2}, {0.8, 0.2}, {0.8, 0.8}, {0.2, 0.8}},
  };
}

void tracking_mode_defaults_match_rust() {
  expect_true(default_tracking_mode() == TrackingMode::Field, "default tracking mode");
  expect_true(TrackingMode{} == TrackingMode::Field, "direct default tracking mode");
  expect_true(parse_tracking_mode("field") == TrackingMode::Field, "parse field");
  expect_true(parse_tracking_mode("BALL") == TrackingMode::Ball, "parse ball");
  expect_true(parse_tracking_mode("sweep") == TrackingMode::Sweep, "parse sweep");
  expect_true(!parse_tracking_mode("manual").has_value(), "unknown tracking mode rejected");
  expect_eq(tracking_mode_name(TrackingMode::Sweep), std::string_view("sweep"), "mode name");
}

void coaster_lifecycle_matches_rust() {
  Coaster never_seen(5);
  expect_true(never_seen.step_without_fresh() == CoastStatus::Lost, "never seen is lost");

  Coaster c(3);
  expect_true(c.accept_fresh() == CoastStatus::Tracking, "fresh tracking");
  expect_true(c.step_without_fresh() == CoastStatus::Coasting, "coast one");
  expect_true(c.step_without_fresh() == CoastStatus::Coasting, "coast two");
  expect_true(c.step_without_fresh() == CoastStatus::Coasting, "coast three");
  expect_true(c.step_without_fresh() == CoastStatus::Lost, "coast exhausted");

  Coaster zero(0);
  expect_true(zero.accept_fresh() == CoastStatus::Tracking, "zero budget fresh");
  expect_true(zero.step_without_fresh() == CoastStatus::Lost, "zero budget lost");

  Coaster reset(2);
  expect_true(reset.accept_fresh() == CoastStatus::Tracking, "reset fresh");
  expect_true(reset.step_without_fresh() == CoastStatus::Coasting, "reset coast one");
  expect_true(reset.step_without_fresh() == CoastStatus::Coasting, "reset coast two");
  expect_true(reset.accept_fresh() == CoastStatus::Tracking, "fresh reacquires");
  expect_true(reset.step_without_fresh() == CoastStatus::Coasting, "coast after reset");
  expect_eq(reset.frames_coasting(), 1U, "frames coasting counter");
}

void roi_filter_matches_rust_anchor_policy() {
  const auto full = full_roi();
  expect_eq(filter_by_roi({detection(reco::core::CameraId::Left, 0, 0.5F, 0.4F, 0.1F, 0.2F)}, full)
                .size(),
            1U, "inside roi passes");

  const auto small = small_roi();
  expect_true(
      filter_by_roi({detection(reco::core::CameraId::Left, 0, 0.05F, 0.05F, 0.1F, 0.2F)}, small)
          .empty(),
      "outside roi filtered");

  reco::core::FieldRoi degenerate;
  expect_eq(
      filter_by_roi({detection(reco::core::CameraId::Left, 0, 0.5F, 0.5F, 0.1F, 0.2F)}, degenerate)
          .size(),
      1U, "degenerate roi passes all");

  reco::core::FieldRoi split{
      .left = {{0.2, 0.2}, {0.8, 0.2}, {0.8, 0.8}, {0.2, 0.8}},
      .right = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}},
  };
  expect_true(
      filter_by_roi({detection(reco::core::CameraId::Left, 0, 0.05F, 0.05F, 0.1F, 0.2F)}, split)
          .empty(),
      "left polygon filters");
  expect_eq(
      filter_by_roi({detection(reco::core::CameraId::Right, 0, 0.05F, 0.05F, 0.1F, 0.2F)}, split)
          .size(),
      1U, "right polygon passes");

  const auto ball = detection(reco::core::CameraId::Left, 0, 0.5F, 0.7F, 0.1F, 0.3F);
  const auto player = detection(reco::core::CameraId::Left, 1, 0.5F, 0.7F, 0.1F, 0.3F);
  std::unordered_map<std::uint16_t, RoiAnchor> anchors{{1, RoiAnchor::Bottom}};
  const auto filtered = filter_by_roi({ball, player}, small, anchors, RoiAnchor::Center);
  expect_eq(filtered.size(), 1U, "per-class anchor filters player");
  expect_eq(filtered[0].class_id, 0U, "ball survives center anchor");

  const auto top_outside = detection(reco::core::CameraId::Left, 1, 0.5F, 0.3F, 0.1F, 0.3F);
  expect_true(!passes_roi_anchor(RoiAnchor::Top, top_outside, small.left),
              "top anchor rejects when top samples fall outside");
}

void sweep_panner_matches_rust_phase_policy() {
  const auto cal = reco::core::MatchCalibration{};
  const PanContext origin{.frame_index = 0, .timestamp_ms = 0.0, .calibration = &cal};
  const WorldState world;
  const SweepPanner panner(0.8F, 10.0F);
  const auto first = panner.decide(world, origin);
  expect_near(first.yaw, 0.0F, 1.0e-6F, "sweep starts at origin");
  expect_true(first.fov_degrees.has_value() && *first.fov_degrees == 50.0F, "default sweep fov");

  const auto quarter = panner.decide(
      world, PanContext{.frame_index = 75, .timestamp_ms = 2500.0, .calibration = &cal});
  expect_near(quarter.yaw, 0.8F, 0.05F, "quarter cycle reaches range");

  for (std::uint64_t i = 0; i < 300; ++i) {
    const auto out = panner.decide(world, PanContext{.frame_index = i, .calibration = &cal});
    expect_true(std::abs(out.yaw) <= 0.800001F, "sweep stays in range");
  }

  const auto fixed = panner.with_fov(42.0F).decide(world, origin);
  expect_true(fixed.fov_degrees.has_value() && *fixed.fov_degrees == 42.0F, "fov override");

  const auto zoomed_panner = panner.with_zoom(30.0F, 70.0F, 8.0F);
  const auto zoomed = zoomed_panner.decide(
      world, PanContext{.frame_index = 60, .timestamp_ms = 2000.0, .calibration = &cal});
  expect_true(zoomed.fov_degrees.has_value(), "zoom fov present");
  expect_near(*zoomed.fov_degrees, 70.0F, 0.05F, "zoom quarter cycle reaches max fov");
  const auto zoom_then_fov = zoomed_panner.with_fov(42.0F).decide(
      world, PanContext{.frame_index = 60, .timestamp_ms = 2000.0, .calibration = &cal});
  expect_near(*zoom_then_fov.fov_degrees, 70.0F, 0.05F, "with_fov does not disable zoom");

  WorldState with_ball;
  with_ball.ball = reco::core::TrackedEntity{.yaw = 1.5F, .pitch = 0.3F};
  const auto a = panner.decide(with_ball, origin);
  const auto b = panner.decide(world, origin);
  expect_near(a.yaw, b.yaw, 1.0e-6F, "sweep ignores world state");
}

} // namespace

int main() {
  tracking_mode_defaults_match_rust();
  coaster_lifecycle_matches_rust();
  roi_filter_matches_rust_anchor_policy();
  sweep_panner_matches_rust_phase_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
