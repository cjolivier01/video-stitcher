#include "reco/autocam/ball_tracker.hpp"
#include "reco/autocam/class_provider.hpp"
#include "reco/autocam/coaster.hpp"
#include "reco/autocam/roi_filter.hpp"
#include "reco/autocam/sweep_panner.hpp"
#include "reco/autocam/tracking_mode.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
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

reco::core::MappedDetection mapped_detection(reco::core::CameraId camera, std::uint16_t class_id,
                                             float yaw, float pitch, float confidence) {
  return {
      .camera = camera,
      .class_id = class_id,
      .confidence = confidence,
      .camera_center_x = 0.5F,
      .camera_center_y = 0.5F,
      .camera_size_x = 0.05F,
      .camera_size_y = 0.05F,
      .position = reco::core::ViewportPosition{.yaw = yaw, .pitch = pitch},
  };
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

void class_provider_matches_rust_stateless_policy() {
  ClassProvider provider(0);
  std::vector<reco::core::MappedDetection> detections{
      mapped_detection(reco::core::CameraId::Left, 0, 0.0F, 0.0F, 0.9F),
      mapped_detection(reco::core::CameraId::Left, 0, 0.5F, 0.0F, 0.8F),
      mapped_detection(reco::core::CameraId::Right, 0, -0.5F, 0.0F, 0.7F),
  };
  auto out = provider.update(detections, 0.0);
  expect_eq(out.size(), 3U, "class provider emits all configured detections");
  expect_true(out[0].state == reco::core::TrackState::Tracking, "provider state tracking");
  expect_near(out[2].yaw, -0.5F, 1.0e-6F, "provider yaw passthrough");
  expect_true(out[2].origin == reco::core::CameraId::Right, "provider origin passthrough");

  auto same = provider.update(detections, 16.7);
  expect_eq(same.size(), out.size(), "provider deterministic count");
  expect_near(same[1].yaw, out[1].yaw, 1.0e-6F, "provider deterministic yaw");

  auto other = mapped_detection(reco::core::CameraId::Left, 5, 0.3F, 0.0F, 0.9F);
  auto keep = mapped_detection(reco::core::CameraId::Left, 0, 0.1F, 0.0F, 0.9F);
  auto filtered = provider.update({other, keep}, 0.0);
  expect_eq(filtered.size(), 1U, "provider filters other classes");
  expect_near(filtered[0].yaw, 0.1F, 1.0e-6F, "provider kept class yaw");

  keep.position = std::nullopt;
  expect_true(provider.update({keep}, 0.0).empty(), "provider skips missing position");
  expect_eq(provider.class_id(), 0U, "provider class id");
}

void ball_tracker_matches_rust_singleton_policy() {
  BallTracker empty(0);
  expect_true(empty.update({}, 0.0).empty(), "empty detections produce nothing");

  BallTracker tracker(0);
  auto first =
      tracker.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.1F, 0.8F)}, 0.0);
  expect_eq(first.size(), 1U, "first detection emits");
  expect_true(first[0].state == reco::core::TrackState::Tracking, "first detection tracking");
  expect_near(first[0].yaw, 0.2F, 1.0e-6F, "first yaw");
  expect_true(first[0].origin == reco::core::CameraId::Left, "first origin");

  BallTracker wrong_class(32);
  expect_true(
      wrong_class.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.1F, 0.8F)}, 0.0)
          .empty(),
      "wrong class ignored");

  auto no_pos = mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.1F, 0.8F);
  no_pos.position = std::nullopt;
  expect_true(BallTracker(0).update({no_pos}, 0.0).empty(), "missing position ignored");

  auto coasting = BallTracker(0).with_max_coast_frames(2);
  (void)coasting.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.1F, 0.8F)}, 0.0);
  expect_true(coasting.update({}, 16.6)[0].state == reco::core::TrackState::Coasting, "coast one");
  expect_true(coasting.update({}, 33.3)[0].state == reco::core::TrackState::Coasting, "coast two");
  auto lost = coasting.update({}, 50.0);
  expect_true(lost[0].state == reco::core::TrackState::Lost, "coast then lost");
  expect_true(coasting.update({}, 66.6).empty(), "lost emits once");

  auto gated = BallTracker(0).with_max_jump_rad(0.1F);
  (void)gated.update({mapped_detection(reco::core::CameraId::Left, 0, 0.0F, 0.0F, 0.9F)}, 0.0);
  auto jump =
      gated.update({mapped_detection(reco::core::CameraId::Left, 0, 1.0F, 0.0F, 0.9F)}, 16.6);
  expect_true(jump[0].state == reco::core::TrackState::Coasting, "big jump rejected");
  expect_near(jump[0].yaw, 0.0F, 1.0e-6F, "coast holds last yaw");

  auto handoff = BallTracker(0).with_max_jump_rad(0.3F);
  (void)handoff.update({mapped_detection(reco::core::CameraId::Left, 0, 0.15F, 0.0F, 0.8F)}, 0.0);
  auto right =
      handoff.update({mapped_detection(reco::core::CameraId::Right, 0, 0.20F, 0.0F, 0.75F)}, 16.6);
  expect_true(right[0].state == reco::core::TrackState::Tracking, "cross camera handoff");
  expect_true(right[0].origin == reco::core::CameraId::Right, "handoff origin");

  auto anchored = BallTracker(0).with_player_anchor_rad(0.1F);
  anchored.set_players({reco::core::TrackedEntity{.id = 1,
                                                  .class_id = 0,
                                                  .yaw = 1.0F,
                                                  .pitch = 0.0F,
                                                  .confidence = 0.9F,
                                                  .state = reco::core::TrackState::Tracking,
                                                  .age_frames = 5,
                                                  .origin = reco::core::CameraId::Right}});
  expect_true(
      anchored.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.0F, 0.9F)}, 0.0)
          .empty(),
      "player anchor rejects far ball");
  auto no_anchor = BallTracker(0).with_player_anchor_rad(0.1F);
  expect_eq(
      no_anchor.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.0F, 0.9F)}, 0.0)
          .size(),
      1U, "no players anchor is noop");

  WorldState world;
  world.players = {reco::core::TrackedEntity{.id = 7,
                                             .class_id = 0,
                                             .yaw = 1.0F,
                                             .pitch = 0.0F,
                                             .confidence = 0.9F,
                                             .state = reco::core::TrackState::Tracking,
                                             .age_frames = 3,
                                             .origin = reco::core::CameraId::Right}};
  auto observed = BallTracker(0).with_player_anchor_rad(0.1F);
  observed.observe_world(world);
  expect_true(
      observed.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.0F, 0.9F)}, 0.0)
          .empty(),
      "observe_world populates anchors");
  observed.observe_world(WorldState{});
  expect_eq(
      observed.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.0F, 0.9F)}, 0.0)
          .size(),
      1U, "empty world clears anchors");

  auto chooser = BallTracker(0).with_max_jump_rad(1.0F);
  (void)chooser.update({mapped_detection(reco::core::CameraId::Left, 0, 0.0F, 0.0F, 0.9F)}, 0.0);
  auto chosen =
      chooser.update({mapped_detection(reco::core::CameraId::Left, 0, 0.40F, 0.0F, 0.95F),
                      mapped_detection(reco::core::CameraId::Left, 0, 0.05F, 0.0F, 0.55F)},
                     16.6);
  expect_near(chosen[0].yaw, 0.05F, 1.0e-6F, "closer candidate beats confidence");
  expect_eq(BallTracker(32).class_id(), 32U, "ball tracker class id");

  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_jump = BallTracker(0).with_max_jump_rad(nan);
  (void)nan_jump.update({mapped_detection(reco::core::CameraId::Left, 0, 0.0F, 0.0F, 0.9F)}, 0.0);
  auto rejected =
      nan_jump.update({mapped_detection(reco::core::CameraId::Left, 0, 0.01F, 0.0F, 0.9F)}, 16.6);
  expect_true(rejected[0].state == reco::core::TrackState::Coasting,
              "NaN max jump clamps to zero");
  expect_near(rejected[0].yaw, 0.0F, 1.0e-6F, "NaN max jump holds last pose");

  auto nan_anchor = BallTracker(0).with_player_anchor_rad(nan);
  nan_anchor.set_players({reco::core::TrackedEntity{.id = 1,
                                                    .class_id = 0,
                                                    .yaw = 0.2F,
                                                    .pitch = 0.0F,
                                                    .confidence = 0.9F,
                                                    .state = reco::core::TrackState::Tracking,
                                                    .age_frames = 5,
                                                    .origin = reco::core::CameraId::Right}});
  expect_eq(
      nan_anchor.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.0F, 0.9F)}, 0.0)
          .size(),
      1U, "NaN player anchor clamps to exact-match zero radius");

  auto nan_score = BallTracker(0);
  auto first_nan =
      nan_score.update({mapped_detection(reco::core::CameraId::Left, 0, 0.2F, 0.0F, nan)}, 0.0);
  expect_eq(first_nan.size(), 1U, "sole NaN confidence candidate is still selected");
  expect_true(std::isnan(first_nan[0].confidence), "NaN confidence passthrough");
}

} // namespace

int main() {
  tracking_mode_defaults_match_rust();
  coaster_lifecycle_matches_rust();
  roi_filter_matches_rust_anchor_policy();
  sweep_panner_matches_rust_phase_policy();
  class_provider_matches_rust_stateless_policy();
  ball_tracker_matches_rust_singleton_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
