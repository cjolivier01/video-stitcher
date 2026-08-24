#include "reco/core/pipeline_event.hpp"
#include "reco/core/projection.hpp"
#include "reco/core/replay_buffer.hpp"
#include "reco/core/telemetry.hpp"

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
  telemetry_collector_matches_rust_policy();
  pipeline_events_match_rust_frame_index_and_timing();
  replay_buffer_matches_rust_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
