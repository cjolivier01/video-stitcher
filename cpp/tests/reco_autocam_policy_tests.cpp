#include "reco/autocam/ball_tracker.hpp"
#include "reco/autocam/class_provider.hpp"
#include "reco/autocam/coaster.hpp"
#include "reco/autocam/field_panner.hpp"
#include "reco/autocam/file_panner.hpp"
#include "reco/autocam/roi_filter.hpp"
#include "reco/autocam/sweep_panner.hpp"
#include "reco/autocam/tracking_mode.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <variant>
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

reco::core::TrackedEntity tracked_player(float yaw, float pitch, std::uint64_t id,
                                         float confidence = 0.9F) {
  return {.id = id,
          .class_id = 0,
          .yaw = yaw,
          .pitch = pitch,
          .confidence = confidence,
          .state = reco::core::TrackState::Tracking,
          .age_frames = 5,
          .origin = reco::core::CameraId::Left};
}

reco::core::TrackedEntity tracked_ball(float yaw, float pitch) {
  return {.id = 0,
          .class_id = 32,
          .yaw = yaw,
          .pitch = pitch,
          .confidence = 0.8F,
          .state = reco::core::TrackState::Tracking,
          .age_frames = 1,
          .origin = reco::core::CameraId::Left};
}

WorldState tight_world() {
  return {.players = {tracked_player(0.28F, 0.0F, 1),
                      tracked_player(0.32F, 0.0F, 2),
                      tracked_player(0.36F, 0.0F, 3),
                      tracked_player(0.40F, 0.0F, 4),
                      tracked_player(0.44F, 0.0F, 5)}};
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

void file_panner_matches_rust_csv_policy() {
  const auto path = std::filesystem::temp_directory_path() / "reco_cpp_file_panner_policy.csv";
  {
    std::ofstream file(path);
    file << "frame,yaw,pitch,fov\n";
    file << "short\n";
    file << "\n";
    file << "0,0.1,0.2,45\n";
    file << "2,0.3,0.4,\n";
    file << "2,0.5,0.6,not-a-fov\n";
  }

  auto panner = FilePanner::from_csv(path);
  const WorldState world;
  expect_eq(panner.pose_count(), 2U, "file panner loaded distinct frames");

  auto frame0 = panner.decide(world, PanContext{.frame_index = 0});
  expect_near(frame0.yaw, 0.1F, 1.0e-6F, "file panner frame 0 yaw");
  expect_near(frame0.pitch, 0.2F, 1.0e-6F, "file panner frame 0 pitch");
  expect_true(frame0.fov_degrees.has_value() && *frame0.fov_degrees == 45.0F,
              "file panner parses fov");

  auto held = panner.decide(world, PanContext{.frame_index = 1});
  expect_near(held.yaw, 0.1F, 1.0e-6F, "file panner holds last yaw");
  expect_true(held.fov_degrees.has_value() && *held.fov_degrees == 45.0F,
              "file panner holds last fov");

  auto duplicate = panner.decide(world, PanContext{.frame_index = 2});
  expect_near(duplicate.yaw, 0.5F, 1.0e-6F, "file panner duplicate frame overwrites");
  expect_near(duplicate.pitch, 0.6F, 1.0e-6F, "file panner duplicate pitch");
  expect_true(!duplicate.fov_degrees.has_value(), "invalid optional fov becomes absent");

  std::filesystem::remove(path);

  const auto invalid_path =
      std::filesystem::temp_directory_path() / "reco_cpp_file_panner_policy_invalid.csv";
  {
    std::ofstream file(invalid_path);
    file << "frame,yaw,pitch,fov\n";
    file << "1,bad,0.2,45\n";
  }
  bool threw = false;
  try {
    (void)FilePanner::from_csv(invalid_path);
  } catch (const std::exception&) {
    threw = true;
  }
  expect_true(threw, "file panner propagates required parse errors");
  std::filesystem::remove(invalid_path);

  const auto negative_path =
      std::filesystem::temp_directory_path() / "reco_cpp_file_panner_policy_negative.csv";
  {
    std::ofstream file(negative_path);
    file << "frame,yaw,pitch,fov\n";
    file << "-1,0.1,0.2,45\n";
  }
  threw = false;
  try {
    (void)FilePanner::from_csv(negative_path);
  } catch (const std::exception&) {
    threw = true;
  }
  expect_true(threw, "file panner rejects negative frame indices");
  std::filesystem::remove(negative_path);
}

void field_panner_matches_rust_policy() {
  auto sanitized = FieldPannerConfig{
      .keep_fraction = -0.2F,
      .fov_tight = 60.0F,
      .fov_wide = 30.0F,
      .ball_weight = 1.5F,
      .lookahead_reactivity = 100.0F,
  }.sanitized();
  expect_true(sanitized.fov_tight <= sanitized.fov_wide, "field config orders fov range");
  expect_near(sanitized.fov_tight, 30.0F, 1.0e-6F, "field config tight fov");
  expect_near(sanitized.fov_wide, 60.0F, 1.0e-6F, "field config wide fov");
  expect_true(sanitized.ball_weight >= 0.0F && sanitized.ball_weight <= 1.0F,
              "field config clamps ball weight");
  expect_true(sanitized.keep_fraction > 0.0F, "field config clamps keep fraction");
  const float nan = std::numeric_limits<float>::quiet_NaN();
  FieldPannerConfig nan_config;
  nan_config.fov_tight = nan;
  nan_config.fov_wide = 30.0F;
  nan_config.fov_default = nan;
  nan_config.cluster_bandwidth_rad = nan;
  nan_config.dead_zone_rad = nan;
  nan_config.ball_max_dist_from_cluster = nan;
  nan_config.lead_gain = nan;
  nan_config.fov_alpha = nan;
  const auto sanitized_nan = nan_config.sanitized();
  expect_true(std::isfinite(sanitized_nan.fov_tight), "field config finite tight fov after NaN");
  expect_true(std::isfinite(sanitized_nan.fov_wide), "field config finite wide fov after NaN");
  expect_true(std::isnan(sanitized_nan.fov_default), "field config clamp preserves NaN default fov");
  expect_true(std::isfinite(sanitized_nan.cluster_bandwidth_rad),
              "field config finite bandwidth after NaN");
  expect_true(std::isfinite(sanitized_nan.dead_zone_rad), "field config finite dead zone after NaN");
  expect_true(std::isfinite(sanitized_nan.ball_max_dist_from_cluster),
              "field config finite ball gate after NaN");
  expect_true(std::isfinite(sanitized_nan.lead_gain), "field config finite lead gain after NaN");
  expect_true(std::isnan(sanitized_nan.fov_alpha), "field config clamp preserves NaN fov alpha");
  auto nan_range = FieldPanner(30.0F).with_fov_range(nan, 30.0F);
  expect_true(std::isfinite(nan_range.target_fov(0.2F, 0.0F, 0.0F)), "NaN fov range sanitized");

  expect_true(FieldPannerConfig::from_preset_name("broadcast").has_value(), "broadcast preset");
  expect_true(FieldPannerConfig::from_preset_name("ACTION").has_value(), "action preset");
  expect_true(!FieldPannerConfig::from_preset_name("nope").has_value(), "unknown preset");
  expect_true(FieldPannerConfig::frame_all().framing == FramingMode::FrameAll, "frame_all preset");

  FieldPanner trim =
      FieldPanner::with_config(30.0F, FieldPannerConfig{.cluster_mode = ClusterMode::TrimmedMean,
                                                        .keep_fraction = 0.6F});
  std::vector<FieldPanner::Point> points{{0.30F, 0.0F, 1.0F},
                                         {0.32F, 0.0F, 1.0F},
                                         {0.34F, 0.0F, 1.0F},
                                         {1.50F, 0.0F, 1.0F},
                                         {1.60F, 0.0F, 1.0F}};
  auto kept = trim.cluster_and_trim(points);
  expect_eq(kept.size(), 3U, "field trim keeps configured fraction");
  expect_true(std::all_of(kept.begin(), kept.end(),
                          [](const auto& point) { return std::get<0>(point) < 0.5F; }),
              "field trim excludes far outliers");
  FieldPannerConfig nan_keep_config;
  nan_keep_config.cluster_mode = ClusterMode::TrimmedMean;
  nan_keep_config.keep_fraction = nan;
  nan_keep_config.min_cluster = 2;
  auto nan_keep = FieldPanner::with_config(30.0F, nan_keep_config).cluster_and_trim(points);
  expect_eq(nan_keep.size(), 2U, "NaN keep fraction floors to min_cluster");

  FieldPanner density = FieldPanner::with_config(
      30.0F, FieldPannerConfig{.cluster_mode = ClusterMode::Density, .cluster_bandwidth_rad = 0.35F});
  std::vector<FieldPanner::Point> bimodal{{-0.05F, 0.0F, 1.0F}, {-0.02F, 0.0F, 1.0F},
                                          {0.00F, 0.0F, 1.0F},  {0.03F, 0.0F, 1.0F},
                                          {0.05F, 0.0F, 1.0F},  {0.02F, 0.0F, 1.0F},
                                          {1.35F, 0.0F, 1.0F},  {1.40F, 0.0F, 1.0F},
                                          {1.45F, 0.0F, 1.0F}};
  auto core = density.densest_cluster(bimodal);
  expect_eq(core.size(), 6U, "density keeps dominant near group");
  expect_true(std::all_of(core.begin(), core.end(),
                          [](const auto& point) { return std::get<0>(point) < 0.5F; }),
              "density excludes far block");

  const PanContext ctx{.frame_index = 0};
  FieldPannerConfig follow_config;
  follow_config.dead_zone_rad = 0.0F;
  follow_config.ball_weight = 0.0F;
  auto follow = FieldPanner::with_config(30.0F, follow_config);
  auto world = tight_world();
  auto out = follow.decide(world, ctx);
  for (std::uint64_t i = 1; i < 200; ++i) {
    out = follow.decide(world, PanContext{.frame_index = i});
  }
  expect_true(out.yaw > 0.38F && out.yaw < 0.45F, "field follows player centroid with edge push");
  expect_true(out.fov_degrees.has_value(), "field panner emits fov");

  auto hold = FieldPanner(30.0F);
  hold.set_pose_for_test(0.3F, 0.05F);
  auto held = hold.decide(WorldState{}, ctx);
  expect_near(held.yaw, 0.3F, 1.0e-6F, "field holds yaw without target");
  expect_near(held.pitch, 0.05F, 1.0e-6F, "field holds pitch without target");

  auto ball_only = FieldPanner::with_config(30.0F, FieldPannerConfig{.dead_zone_rad = 0.0F});
  WorldState ball_world;
  ball_world.ball = tracked_ball(0.5F, 0.1F);
  auto ball_out = ball_only.decide(ball_world, ctx);
  for (std::uint64_t i = 1; i < 300; ++i) {
    ball_out = ball_only.decide(ball_world, PanContext{.frame_index = i});
  }
  expect_true(std::abs(ball_out.yaw - 0.5F) < 0.05F, "field follows ball without cluster");
  expect_true(std::abs(ball_out.pitch - 0.1F) < 0.05F, "field follows ball pitch without cluster");

  auto blended = FieldPanner(30.0F).with_ball_weight(0.3F);
  auto blend_world = tight_world();
  blend_world.ball = tracked_ball(0.8F, 0.0F);
  auto blend = blended.decide(blend_world, ctx);
  for (std::uint64_t i = 1; i < 200; ++i) {
    blend = blended.decide(blend_world, PanContext{.frame_index = i});
  }
  expect_true(blend.yaw > 0.414F && blend.yaw < 0.8F, "field ball blend pulls but does not dominate");

  auto lost_ball = FieldPanner::with_config(30.0F, follow_config).with_ball_weight(0.3F);
  auto lost_world = tight_world();
  auto lost = tracked_ball(0.8F, 0.0F);
  lost.state = reco::core::TrackState::Lost;
  lost_world.ball = lost;
  auto lost_out = lost_ball.decide(lost_world, ctx);
  for (std::uint64_t i = 1; i < 200; ++i) {
    lost_out = lost_ball.decide(lost_world, PanContext{.frame_index = i});
  }
  expect_true(std::abs(lost_out.yaw - 0.414F) < 0.03F, "field ignores lost ball in blend");

  FieldPannerConfig lead_config;
  lead_config.lookahead_reactivity = 1.0F;
  lead_config.lead_gain = 1.0F;
  lead_config.dead_zone_rad = 0.0F;
  auto no_lead = FieldPanner::with_config(30.0F, lead_config);
  auto with_lead = FieldPanner::with_config(30.0F, lead_config);
  WorldState current_action{
      .players = {tracked_player(-0.02F, 0.0F, 1), tracked_player(0.02F, 0.0F, 2),
                  tracked_player(0.0F, 0.0F, 3)}};
  WorldState future_action{
      .players = {tracked_player(0.48F, 0.0F, 1), tracked_player(0.52F, 0.0F, 2),
                  tracked_player(0.50F, 0.0F, 3)}};
  auto no_lead_out = no_lead.decide(current_action, ctx);
  auto lead_out = with_lead.decide_with_lookahead(current_action, {future_action}, ctx);
  for (std::uint64_t i = 1; i < 20; ++i) {
    no_lead_out = no_lead.decide(current_action, PanContext{.frame_index = i});
    lead_out =
        with_lead.decide_with_lookahead(current_action, {future_action}, PanContext{.frame_index = i});
  }
  expect_true(lead_out.yaw > no_lead_out.yaw, "field lookahead aims ahead of current action");

  FieldPannerConfig nan_runtime_config;
  nan_runtime_config.dead_zone_rad = 0.0F;
  nan_runtime_config.lookahead_reactivity = nan;
  nan_runtime_config.velocity_alpha = nan;
  auto nan_runtime = FieldPanner::with_config(30.0F, nan_runtime_config);
  auto nan_runtime_out =
      nan_runtime.decide_with_lookahead(current_action, {future_action}, PanContext{.frame_index = 0});
  expect_true(std::isfinite(nan_runtime_out.yaw) && std::isfinite(nan_runtime_out.pitch),
              "field NaN runtime min/max config does not poison pose");

  auto frame_all = FieldPanner::with_config(
      30.0F, FieldPannerConfig{.dead_zone_rad = 0.0F, .framing = FramingMode::FrameAll});
  WorldState frame_world{
      .players = {tracked_player(0.0F, 0.0F, 1), tracked_player(0.1F, 0.0F, 2),
                  tracked_player(0.5F, 0.0F, 3)}};
  auto frame_out = frame_all.decide(frame_world, ctx);
  for (std::uint64_t i = 1; i < 400; ++i) {
    frame_out = frame_all.decide(frame_world, PanContext{.frame_index = i});
  }
  expect_true(std::abs(frame_out.yaw - 0.25F) < 0.02F, "frame-all aims at bbox midpoint");
  expect_true(frame_all.target_fov(0.30F, 0.0F, 0.0F) > frame_all.target_fov(0.05F, 0.0F, 0.0F),
              "frame-all fov grows with extent");

  FieldPannerConfig weighted_config;
  weighted_config.cluster_mode = ClusterMode::TrimmedMean;
  weighted_config.keep_fraction = 1.0F;
  weighted_config.dead_zone_rad = 0.0F;
  weighted_config.ball_weight = 0.0F;
  weighted_config.confidence_weighted = true;
  auto weighted = FieldPanner::with_config(30.0F, weighted_config);
  FieldPannerConfig unweighted_config = weighted_config;
  unweighted_config.confidence_weighted = false;
  auto unweighted = FieldPanner::with_config(30.0F, unweighted_config);
  WorldState confidence_world{
      .players = {tracked_player(0.0F, 0.0F, 1, 0.2F), tracked_player(0.4F, 0.0F, 2, 0.95F)}};
  auto weighted_out = weighted.decide(confidence_world, ctx);
  auto unweighted_out = unweighted.decide(confidence_world, ctx);
  for (std::uint64_t i = 1; i < 400; ++i) {
    weighted_out = weighted.decide(confidence_world, PanContext{.frame_index = i});
    unweighted_out = unweighted.decide(confidence_world, PanContext{.frame_index = i});
  }
  expect_true(weighted_out.yaw > unweighted_out.yaw, "confidence weighting pulls toward confident player");

  FieldPannerConfig lock_config;
  lock_config.dead_zone_rad = 0.0F;
  lock_config.ball_weight = 0.0F;
  lock_config.lock_pitch = true;
  lock_config.locked_pitch_rad = 0.0F;
  auto lock = FieldPanner::with_config(30.0F, lock_config);
  WorldState pitched{
      .players = {tracked_player(0.30F, 0.2F, 1), tracked_player(0.34F, 0.2F, 2),
                  tracked_player(0.38F, 0.2F, 3)}};
  auto locked = lock.decide(pitched, ctx);
  for (std::uint64_t i = 1; i < 400; ++i) {
    locked = lock.decide(pitched, PanContext{.frame_index = i});
  }
  expect_true(std::abs(locked.pitch) < 1.0e-3F, "lock pitch holds tilt");
  expect_true(locked.yaw > 0.2F, "lock pitch still pans yaw");

  auto nan_panner = FieldPanner(30.0F);
  nan_panner.set_pose_for_test(0.3F, 0.05F);
  WorldState nan_world{.players = {tracked_player(std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::quiet_NaN(), 1),
                                   tracked_player(std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::quiet_NaN(), 2)}};
  auto nan_out = nan_panner.decide(nan_world, ctx);
  expect_true(std::isfinite(nan_out.yaw) && std::isfinite(nan_out.pitch), "field rejects NaN aim");
  expect_near(nan_out.yaw, 0.3F, 1.0e-6F, "field keeps yaw on NaN");

  auto fov_nan = FieldPanner(30.0F);
  const float baseline_fov = fov_nan.current_fov();
  expect_true(std::isnan(fov_nan.target_fov(nan, nan, nan)), "field target fov preserves NaN inputs");
  auto fov_nan_out = fov_nan.decide(nan_world, ctx);
  expect_true(fov_nan_out.fov_degrees.has_value() && std::isfinite(*fov_nan_out.fov_degrees),
              "field fov does not latch NaN");
  expect_near(*fov_nan_out.fov_degrees, baseline_fov, 1.0e-6F, "field fov holds on NaN cluster");

  auto debug_panner = FieldPanner::with_config(30.0F, follow_config);
  (void)debug_panner.decide(world, PanContext{.frame_index = 42});
  auto debug = debug_panner.debug_event(42);
  expect_true(debug.has_value(), "field debug event present after cluster");
  if (debug.has_value()) {
    const auto* event = std::get_if<reco::core::PannerDebugEvent>(&debug->variant());
    expect_true(event != nullptr, "field debug event variant");
    if (event != nullptr) {
      expect_eq(event->frame_index, 42U, "field debug frame index");
      expect_eq(event->n_players, 5U, "field debug player count");
      expect_true(std::isfinite(event->fov_target), "field debug fov target finite");
    }
  }
  (void)debug_panner.decide(WorldState{}, PanContext{.frame_index = 43});
  expect_true(!debug_panner.debug_event(43).has_value(), "field debug clears without cluster");
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
  file_panner_matches_rust_csv_policy();
  field_panner_matches_rust_policy();
  class_provider_matches_rust_stateless_policy();
  ball_tracker_matches_rust_singleton_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
