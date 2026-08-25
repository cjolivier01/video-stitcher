#include "reco/core/pipeline_event.hpp"
#include "reco/core/projection.hpp"
#include "reco/core/replay_buffer.hpp"
#include "reco/core/telemetry.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

using namespace reco::core;
using namespace std::chrono_literals;

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
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

MatchCalibration projection_calibration() {
  MatchCalibration cal;
  cal.left = {3840, 2160, 1796.32, 1797.22, 1919.37, 1063.17, {0.0342, 0.0677, -0.0741, 0.0299}};
  cal.right = cal.left;
  cal.layout.camera_axis_offset = 0.2398;
  cal.layout.intersect = 0.5446;
  cal.layout.x_ty = 0.00476;
  cal.layout.x_rz = 0.00753;
  cal.layout.z_rx = -0.00431;
  return cal;
}

std::vector<std::array<double, 2>> unit_square() {
  return {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
}

void point_in_polygon_matches_rust_ray_casting() {
  expect_true(point_in_polygon({0.5, 0.5}, unit_square()), "square center inside");
  expect_true(!point_in_polygon({1.5, 0.5}, unit_square()), "right outside square");
  expect_true(!point_in_polygon({-0.1, 0.5}, unit_square()), "left outside square");
  expect_true(!point_in_polygon({0.5, -0.1}, unit_square()), "below square");
  expect_true(!point_in_polygon({0.5, 1.1}, unit_square()), "above square");

  const std::vector<std::array<double, 2>> triangle{{0.0, 0.0}, {1.0, 0.0}, {0.5, 1.0}};
  expect_true(point_in_polygon({0.5, 0.3}, triangle), "triangle inside");
  expect_true(!point_in_polygon({0.9, 0.8}, triangle), "triangle outside");

  const std::vector<std::array<double, 2>> l_shape{{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.5},
                                                   {0.5, 0.5}, {0.5, 1.0}, {0.0, 1.0}};
  expect_true(point_in_polygon({0.75, 0.25}, l_shape), "l-shape bottom arm inside");
  expect_true(point_in_polygon({0.25, 0.75}, l_shape), "l-shape top arm inside");
  expect_true(!point_in_polygon({0.75, 0.75}, l_shape), "l-shape concave cutout outside");
  expect_true(!point_in_polygon({0.5, 0.5}, {}), "empty polygon outside");
  expect_true(!point_in_polygon({0.5, 0.5}, {{0.0, 0.0}}), "single point polygon outside");
  expect_true(!point_in_polygon({0.5, 0.5}, {{0.0, 0.0}, {1.0, 1.0}}), "two point polygon outside");
  expect_true(point_in_polygon({0.001, 0.5}, unit_square()), "near left edge inside");
  expect_true(point_in_polygon({0.999, 0.5}, unit_square()), "near right edge inside");

  expect_true(point_in_polygon({0.0, 0.5}, unit_square()),
              "left boundary matches Rust ray casting as inside");
  expect_true(!point_in_polygon({1.0, 0.5}, unit_square()),
              "right boundary matches Rust ray casting as outside");
  expect_true(point_in_polygon({0.5, 0.0}, unit_square()),
              "bottom boundary matches Rust ray casting as inside");
  expect_true(!point_in_polygon({0.5, 1.0}, unit_square()),
              "top boundary matches Rust ray casting as outside");
}

float vec_norm(const Vec3& vector) {
  return std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

void virtual_camera_roundtrips_yaw_pitch() {
  const std::array<std::array<float, 3>, 3> camera_positions{
      std::array<float, 3>{0.24F, 0.0F, 0.24F},
      std::array<float, 3>{0.3F, 0.0F, 0.2F},
      std::array<float, 3>{0.1F, 0.0F, 0.5F},
  };
  const float yaws[] = {-1.2F, -0.6F, -0.2F, 0.0F, 0.2F, 0.6F, 1.2F};
  const float pitches[] = {-0.9F, -0.4F, -0.1F, 0.0F, 0.1F, 0.4F, 0.9F};
  for (const auto& camera_position : camera_positions) {
    const VirtualCamera camera(camera_position);
    for (const float yaw : yaws) {
      for (const float pitch : pitches) {
        const Vec3 direction = camera.yaw_pitch_to_direction(yaw, pitch);
        expect_near(vec_norm(direction), 1.0F, 1.0e-5F, "virtual camera direction is unit");
        const auto position = camera.direction_to_yaw_pitch(direction);
        expect_near(position.yaw, yaw, 1.0e-4F, "virtual camera yaw roundtrip");
        expect_near(position.pitch, pitch, 1.0e-4F, "virtual camera pitch roundtrip");
        expect_true(!position.fov_degrees.has_value(), "direction decomposition has no fov");
      }
    }
  }
}

void scene_geometry_matches_rust_layout_policy() {
  PlaneLayout layout;
  layout.camera_axis_offset = 0.25;
  layout.intersect = 0.5;
  const auto geom = SceneGeometry::from_layout_with_aspect(layout, 16.0F / 9.0F);
  expect_near(geom.left_position[2], 0.25F, 1.0e-5F, "left half offset");
  expect_near(geom.right_position[0], 0.25F, 1.0e-5F, "right half offset");
  expect_near(geom.camera_position[0], 0.25F, 1.0e-5F, "camera x");
  expect_near(geom.camera_position[2], 0.25F, 1.0e-5F, "camera z");
  expect_near(geom.left_rotation[1], 1.57079637F, 1.0e-5F, "left y rotation");
  expect_near(geom.plane_aspect, 16.0F / 9.0F, 1.0e-5F, "scene aspect");

  layout.camera_axis_offset = 0.24;
  layout.intersect = 0.55;
  layout.x_ty = 0.005;
  layout.x_rz = 0.008;
  layout.z_rx = -0.004;
  layout.x_rx = 0.002;
  layout.z_rz = -0.003;
  const auto corrected = SceneGeometry::from_layout_with_aspect(layout, 4.0F / 3.0F);
  expect_near(corrected.right_position[1], 0.005F, 1.0e-5F, "right y correction");
  expect_near(corrected.right_rotation[0], 0.002F, 1.0e-5F, "right x rotation");
  expect_near(corrected.right_rotation[2], 0.008F, 1.0e-5F, "right z rotation");
  expect_near(corrected.left_rotation[0], -0.004F, 1.0e-5F, "left x rotation");
  expect_near(corrected.left_rotation[2], -0.003F, 1.0e-5F, "left z rotation");
  expect_near(corrected.plane_aspect, 4.0F / 3.0F, 1.0e-5F, "custom aspect");
}

void plane_uv_world_roundtrips() {
  const auto cal = projection_calibration();
  const auto scene = SceneGeometry::from_layout_with_aspect(
      cal.layout, static_cast<float>(cal.left.width) / static_cast<float>(cal.left.height));
  const Vec3 left_golden = plane_uv_to_world({0.25, 0.75}, CameraId::Left, scene);
  expect_near(left_golden.x, -0.0F, 1.0e-6F, "rust golden left world x");
  expect_near(left_golden.y, -0.069773085F, 1.0e-6F, "rust golden left world y");
  expect_near(left_golden.z, 0.353001863F, 1.0e-6F, "rust golden left world z");
  const Vec3 right_golden = plane_uv_to_world({0.25, 0.75}, CameraId::Right, scene);
  expect_near(right_golden.x, 0.103232987F, 1.0e-6F, "rust golden right world x");
  expect_near(right_golden.y, -0.066491745F, 1.0e-6F, "rust golden right world y");
  expect_near(right_golden.z, 0.0F, 1.0e-6F, "rust golden right world z");

  const double uv_steps[] = {-0.3, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.4};
  for (const CameraId camera : {CameraId::Left, CameraId::Right}) {
    for (const double u : uv_steps) {
      for (const double v : uv_steps) {
        const Vec3 world = plane_uv_to_world({u, v}, camera, scene);
        const auto back = world_to_plane_uv(world, camera, scene);
        expect_true(back.has_value(), "world to plane inverse exists");
        if (back.has_value()) {
          expect_near(back->first, u, 2.0e-5, "plane uv x roundtrip");
          expect_near(back->second, v, 2.0e-5, "plane uv y roundtrip");
        }
      }
    }
  }
}

void kb4_fisheye_roundtrips() {
  const auto params = projection_calibration().left;
  const double cx = params.cx / static_cast<double>(params.width);
  const double cy = params.cy / static_cast<double>(params.height);
  const auto center = inverse_fisheye(cx, cy, params);
  expect_true(center.has_value(), "optical center inverse exists");
  if (center.has_value()) {
    expect_near(center->first, cx, 1.0e-6, "optical center x fixed point");
    expect_near(center->second, cy, 1.0e-6, "optical center y fixed point");
  }

  for (int ix = 0; ix <= 10; ++ix) {
    for (int iy = 0; iy <= 10; ++iy) {
      const double nx = 0.1 + 0.8 * (static_cast<double>(ix) / 10.0);
      const double ny = 0.1 + 0.8 * (static_cast<double>(iy) / 10.0);
      const auto plane = inverse_fisheye(nx, ny, params);
      expect_true(plane.has_value(), "inverse fisheye converges inside valid area");
      if (plane.has_value()) {
        const auto back = forward_fisheye(plane->first, plane->second, params);
        expect_near(back.first, nx, 1.0e-6, "fisheye x roundtrip");
        expect_near(back.second, ny, 1.0e-6, "fisheye y roundtrip");
      }
    }
  }

  const CameraParams zero = {1920, 1080, 960.0, 540.0, 960.0, 540.0, {0.0, 0.0, 0.0, 0.0}};
  const auto identity = inverse_fisheye(0.5, 0.5, zero);
  expect_true(identity.has_value(), "zero distortion center inverse exists");
  if (identity.has_value()) {
    expect_near(identity->first, 0.5, 1.0e-6, "zero distortion center x");
    expect_near(identity->second, 0.5, 1.0e-6, "zero distortion center y");
  }
  expect_near(kb4_forward_scale(0.0, params.d), 1.0, 0.0, "kb4 zero radius scale");
}

void camera_panorama_projection_roundtrips() {
  const auto cal = projection_calibration();
  const auto scene = SceneGeometry::from_layout_with_aspect(
      cal.layout, static_cast<float>(cal.left.width) / static_cast<float>(cal.left.height));
  const float cx = static_cast<float>(cal.left.cx) / static_cast<float>(cal.left.width);
  const float cy = static_cast<float>(cal.left.cy) / static_cast<float>(cal.left.height);
  const auto optical_center = camera_to_panorama(CameraId::Left, cx, cy, cal, scene);
  expect_true(optical_center.has_value(), "optical center maps to panorama");
  if (optical_center.has_value()) {
    expect_true(std::isfinite(optical_center->yaw), "optical center yaw finite");
    expect_true(std::isfinite(optical_center->pitch), "optical center pitch finite");
  }

  const auto left_center = camera_to_panorama(CameraId::Left, 0.5F, 0.5F, cal, scene);
  const auto right_center = camera_to_panorama(CameraId::Right, 0.5F, 0.5F, cal, scene);
  expect_true(left_center.has_value() && right_center.has_value(), "camera centers map");
  if (left_center.has_value() && right_center.has_value()) {
    expect_true(std::abs(left_center->yaw - right_center->yaw) > 0.01F,
                "left and right centers differ in yaw");
  }

  const auto left_golden = camera_to_panorama(CameraId::Left, 0.3F, 0.4F, cal, scene);
  expect_true(left_golden.has_value(), "rust golden left camera point maps");
  if (left_golden.has_value()) {
    expect_near(left_golden->yaw, 1.158160448F, 1.0e-6F, "rust golden left yaw");
    expect_near(left_golden->pitch, 0.116523206F, 1.0e-6F, "rust golden left pitch");
    const auto back =
        panorama_to_camera(left_golden->yaw, left_golden->pitch, CameraId::Left, cal, scene);
    expect_true(back.has_value(), "rust golden left maps back");
    if (back.has_value()) {
      expect_near(back->first, 0.299999893F, 1.5e-6F, "rust golden left back x");
      expect_near(back->second, 0.400000006F, 1.5e-6F, "rust golden left back y");
    }
  }
  const auto right_golden = camera_to_panorama(CameraId::Right, 0.7F, 0.6F, cal, scene);
  expect_true(right_golden.has_value(), "rust golden right camera point maps");
  if (right_golden.has_value()) {
    expect_near(right_golden->yaw, -1.160059094F, 1.0e-6F, "rust golden right yaw");
    expect_near(right_golden->pitch, -0.094546765F, 1.0e-6F, "rust golden right pitch");
    const auto back =
        panorama_to_camera(right_golden->yaw, right_golden->pitch, CameraId::Right, cal, scene);
    expect_true(back.has_value(), "rust golden right maps back");
    if (back.has_value()) {
      expect_near(back->first, 0.700000107F, 1.5e-6F, "rust golden right back x");
      expect_near(back->second, 0.600000024F, 1.5e-6F, "rust golden right back y");
    }
  }

  const float steps[] = {0.3F, 0.38F, 0.46F, 0.54F, 0.62F, 0.7F};
  for (const CameraId camera : {CameraId::Left, CameraId::Right}) {
    for (const float nx : steps) {
      for (const float ny : steps) {
        const auto pos = camera_to_panorama(camera, nx, ny, cal, scene);
        expect_true(pos.has_value(), "camera to panorama succeeds inside coverage");
        if (!pos.has_value()) {
          continue;
        }
        const auto back = panorama_to_camera(pos->yaw, pos->pitch, camera, cal, scene);
        expect_true(back.has_value(), "panorama to camera returns same camera");
        if (back.has_value()) {
          expect_near(back->first, nx, 1.5e-3F, "camera x roundtrip");
          expect_near(back->second, ny, 1.5e-3F, "camera y roundtrip");
        }
      }
    }
  }
}

void viewport_bounds_match_rust_policy() {
  const auto cal = projection_calibration();
  const auto scene = SceneGeometry::from_layout_with_aspect(
      cal.layout, static_cast<float>(cal.left.width) / static_cast<float>(cal.left.height));
  const auto bounds = viewport_bounds(40.0F, cal, scene, 16.0F / 9.0F);
  expect_near(bounds.min_yaw, -0.853967249F, 1.0e-6F, "rust golden bounds40 min yaw");
  expect_near(bounds.max_yaw, 0.844241679F, 1.0e-6F, "rust golden bounds40 max yaw");
  expect_near(bounds.min_pitch, -0.063555419F, 1.0e-6F, "rust golden bounds40 min pitch");
  expect_near(bounds.max_pitch, 0.077253968F, 1.0e-6F, "rust golden bounds40 max pitch");
  expect_true(bounds.min_yaw < bounds.max_yaw, "viewport yaw range valid");
  expect_true(bounds.min_pitch < bounds.max_pitch, "viewport pitch range valid");
  expect_true(bounds.max_yaw - bounds.min_yaw > 0.01F, "viewport yaw range non-trivial");

  const auto narrow = viewport_bounds(30.0F, cal, scene, 16.0F / 9.0F);
  const auto wide = viewport_bounds(60.0F, cal, scene, 16.0F / 9.0F);
  expect_near(wide.min_yaw, -0.587049007F, 1.0e-6F, "rust golden bounds60 min yaw");
  expect_near(wide.max_yaw, 0.577323437F, 1.0e-6F, "rust golden bounds60 max yaw");
  expect_near(wide.min_pitch, 0.006849274F, 1.0e-6F, "rust golden bounds60 min pitch");
  expect_near(wide.max_pitch, 0.006849274F, 1.0e-6F, "rust golden bounds60 max pitch");
  expect_true(wide.min_yaw >= narrow.min_yaw, "wide fov min yaw tightens");
  expect_true(wide.max_yaw <= narrow.max_yaw, "wide fov max yaw tightens");

  const ViewportPosition clamped =
      bounds.clamp({bounds.max_yaw + 1.0F, bounds.min_pitch - 1.0F, 90.0F});
  expect_near(clamped.yaw, bounds.max_yaw, 0.0F, "bounds clamp yaw");
  expect_near(clamped.pitch, bounds.min_pitch, 0.0F, "bounds clamp pitch");
  expect_true(clamped.fov_degrees.has_value() && *clamped.fov_degrees == 90.0F,
              "bounds clamp preserves fov");
}

struct CountingSink final : TelemetrySink {
  int calls = 0;
  std::uint64_t last_frames = 0;

  void on_snapshot(const TelemetrySnapshot& snapshot) override {
    ++calls;
    last_frames = snapshot.frames_processed;
  }
};

void telemetry_collector_matches_rust_policy() {
  TelemetryCollector collector;
  collector.set_gpu_name("Test GPU");
  collector.set_encoder_name("Test Encoder");
  collector.set_decode_mode("nvdec");
  auto sink = std::make_unique<CountingSink>();
  auto* sink_ptr = sink.get();
  collector.set_sink(std::move(sink), 0);

  for (int i = 0; i < 300; ++i) {
    collector.record_frame(FrameTiming{
        .decode = 2ms,
        .upload = 1ms,
        .stitch = 4ms,
        .readback = 3ms,
        .submit = 9ms,
        .detection = 5ms,
        .tracking = std::nullopt,
        .total = std::chrono::milliseconds(i + 1),
    });
  }
  collector.record_detections(6, 3, true);
  collector.record_detections(2, 4, false);
  const auto snapshot = collector.snapshot();
  expect_eq(snapshot.frames_processed, 300ULL, "telemetry frame count");
  expect_near(snapshot.avg_decode_ms, 2.0F, 0.01F, "decode average");
  expect_near(snapshot.avg_upload_ms, 1.0F, 0.01F, "upload average");
  expect_near(snapshot.avg_submit_ms, 9.0F, 0.01F, "submit average");
  expect_near(snapshot.avg_total_ms, 172.5F, 0.01F, "ring average uses latest 256 frames");
  expect_near(snapshot.p99_total_ms, 298.0F, 0.01F, "p99 index formula");
  expect_near(snapshot.max_frame_ms, 300.0F, 0.01F, "max frame");
  expect_true(snapshot.bottleneck == PipelineStage::Submit, "submit bottleneck");
  expect_eq(snapshot.total_detections, 8ULL, "total detections");
  expect_near(snapshot.detections_per_frame, 8.0F / 300.0F, 1.0e-6F, "detections per frame");
  expect_near(snapshot.ball_presence_pct, 100.0F / 300.0F, 1.0e-6F, "ball presence pct");
  expect_eq(snapshot.active_tracks, 4U, "active tracks");
  expect_eq(snapshot.gpu_name, std::string_view("Test GPU"), "gpu name");
  expect_eq(sink_ptr->calls, 300, "sink interval maxes to one");
  expect_eq(sink_ptr->last_frames, 300ULL, "sink receives latest frame count");

  const auto summary = session_summary_text(snapshot);
  expect_true(summary.find("Session Summary") != std::string::npos, "summary title");
  expect_true(summary.find("Test GPU") != std::string::npos, "summary gpu");
  expect_true(summary.find("Bottleneck: submit") != std::string::npos, "summary bottleneck");
  expect_true(summary.find("Ball present:  0% of frames") != std::string::npos,
              "summary ball percentage has zero decimals");

  TelemetrySnapshot encode_snapshot = snapshot;
  encode_snapshot.avg_encode_worker_ms = 7.5F;
  encode_snapshot.backpressure_stalls = 2;
  encode_snapshot.backpressure_ms = 3.25F;
  const auto encode_summary = session_summary_text(encode_snapshot);
  expect_true(
      encode_summary.find("Encode:    7.5 ms (overlapped); backpressure 2 stalls / 3.2 ms") !=
          std::string::npos,
      "summary includes encode backpressure line");

  TelemetryCollector tie_collector;
  tie_collector.record_frame(FrameTiming{
      .decode = 10ms,
      .upload = std::nullopt,
      .stitch = std::nullopt,
      .readback = std::nullopt,
      .submit = 10ms,
      .detection = std::nullopt,
      .tracking = std::nullopt,
      .total = 20ms,
  });
  expect_true(tie_collector.snapshot().bottleneck == PipelineStage::Submit,
              "bottleneck ties choose later stage like Rust max_by");
}

void pipeline_events_match_rust_frame_index_and_timing() {
  const PipelineEvent events[] = {
      PipelineEvent(FrameStartEvent{.frame_index = 1, .timestamp_ms = 0.0}),
      PipelineEvent(DetectionsRawEvent{.frame_index = 2, .detections = {}}),
      PipelineEvent(WorldStateEvent{.frame_index = 3, .timestamp_ms = 1.0}),
      PipelineEvent(PanDecisionEvent{.frame_index = 4, .pose = {}}),
      PipelineEvent(PannerDebugEvent{.frame_index = 5}),
      PipelineEvent(PosePresentedEvent{.frame_index = 6, .pose = {}}),
      PipelineEvent(FrameCompleteEvent{.frame_index = 7, .timestamp_ms = 2.0}),
  };
  for (std::size_t i = 0; i < std::size(events); ++i) {
    expect_eq(events[i].frame_index(), static_cast<std::uint64_t>(i + 1), "event frame index");
  }

  const auto timing = FrameTimingMicros::from_frame_timing(FrameTiming{
      .decode = 1000us,
      .upload = std::nullopt,
      .stitch = 500us,
      .readback = std::nullopt,
      .submit = 2000us,
      .detection = std::nullopt,
      .tracking = std::nullopt,
      .total = 5000us,
  });
  expect_eq(timing.decode_us.value_or(0), 1000U, "decode micros");
  expect_true(!timing.upload_us.has_value(), "upload micros absent");
  expect_eq(timing.stitch_us.value_or(0), 500U, "stitch micros");
  expect_eq(timing.submit_us.value_or(0), 2000U, "submit micros");
  expect_eq(timing.total_us, 5000U, "total micros");
  const auto overflow = FrameTimingMicros::from_frame_timing(
      FrameTiming{.total = std::chrono::microseconds{
                      static_cast<long long>(std::numeric_limits<std::uint32_t>::max()) + 2LL}});
  expect_eq(overflow.total_us, 1U, "timing micros overflow truncates like Rust as u32");
  expect_eq(camera_id_name(CameraId::Left), std::string_view("L"), "left display");
  expect_eq(camera_id_name(CameraId::Right), std::string_view("R"), "right display");
}

ReplayFrame replay_frame(std::uint8_t value, std::chrono::nanoseconds captured_at) {
  return {.rgba = std::vector<std::uint8_t>(4, value), .captured_at = captured_at, .pose = {}};
}

void replay_buffer_matches_rust_policy() {
  ReplayBuffer buffer(2s);
  for (std::uint8_t i = 0; i < 5; ++i) {
    buffer.push(replay_frame(i, std::chrono::seconds(i)));
  }
  expect_eq(buffer.len(), 3U, "replay trim length");
  expect_eq(buffer.frames().front().rgba[0], static_cast<std::uint8_t>(2), "replay first kept");
  expect_eq(buffer.latest()->rgba[0], static_cast<std::uint8_t>(4), "replay latest");
  expect_eq(buffer.oldest()->rgba[0], static_cast<std::uint8_t>(2), "replay oldest");
  expect_eq(buffer.buffered_duration(), 2s, "replay buffered duration");

  ReplayBuffer boundary(100ms);
  boundary.push(replay_frame(0, 0ms));
  boundary.push(replay_frame(1, 100ms));
  expect_eq(boundary.len(), 2U, "replay retain boundary inclusive");

  ReplayBuffer snapshots(10s);
  for (std::uint8_t i = 0; i < 3; ++i) {
    snapshots.push(replay_frame(i, std::chrono::milliseconds(i * 100)));
  }
  const auto snap = snapshots.snapshot();
  expect_eq(snap.size(), 3U, "snapshot size");
  expect_eq(snap.front().rgba[0], static_cast<std::uint8_t>(0), "snapshot first");
  expect_eq(snap.back().rgba[0], static_cast<std::uint8_t>(2), "snapshot last");
  expect_eq(snapshots.len(), 3U, "snapshot does not drain");
  const auto drained = snapshots.take();
  expect_eq(drained.size(), 3U, "take size");
  expect_true(snapshots.empty(), "take drains");
  snapshots.push(replay_frame(9, 1s));
  snapshots.clear();
  expect_true(snapshots.empty(), "clear drains");
  expect_eq(snapshots.max_duration(), 10s, "clear preserves duration");
}

} // namespace

int main() {
  point_in_polygon_matches_rust_ray_casting();
  virtual_camera_roundtrips_yaw_pitch();
  scene_geometry_matches_rust_layout_policy();
  plane_uv_world_roundtrips();
  kb4_fisheye_roundtrips();
  camera_panorama_projection_roundtrips();
  viewport_bounds_match_rust_policy();
  telemetry_collector_matches_rust_policy();
  pipeline_events_match_rust_frame_index_and_timing();
  replay_buffer_matches_rust_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
