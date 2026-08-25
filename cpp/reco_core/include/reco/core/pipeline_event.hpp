#pragma once

#include "reco/core/telemetry.hpp"
#include "reco/core/viewport_position.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace reco::core {

enum class CameraId {
  Left,
  Right,
};

[[nodiscard]] std::string camera_id_name(CameraId camera);

struct MappedDetection {
  CameraId camera = CameraId::Left;
  std::uint16_t class_id = 0;
  float confidence = 0.0F;
  float camera_center_x = 0.0F;
  float camera_center_y = 0.0F;
  float camera_size_x = 0.0F;
  float camera_size_y = 0.0F;
  std::optional<ViewportPosition> position;
};

enum class TrackState {
  Tracking,
  Coasting,
  Lost,
};

struct TrackedEntity {
  std::uint64_t id = 0;
  std::uint16_t class_id = 0;
  float yaw = 0.0F;
  float pitch = 0.0F;
  float confidence = 0.0F;
  TrackState state = TrackState::Tracking;
  std::uint64_t age_frames = 0;
  CameraId origin = CameraId::Left;
};

struct FrameTimingMicros {
  std::optional<std::uint32_t> decode_us;
  std::optional<std::uint32_t> upload_us;
  std::optional<std::uint32_t> stitch_us;
  std::optional<std::uint32_t> readback_us;
  std::optional<std::uint32_t> submit_us;
  std::optional<std::uint32_t> detection_us;
  std::uint32_t total_us = 0;

  [[nodiscard]] static FrameTimingMicros from_frame_timing(const FrameTiming& timing);
};

struct FrameStartEvent {
  std::uint64_t frame_index = 0;
  double timestamp_ms = 0.0;
};
struct DetectionsRawEvent {
  std::uint64_t frame_index = 0;
  std::vector<MappedDetection> detections;
};
struct WorldStateEvent {
  std::uint64_t frame_index = 0;
  double timestamp_ms = 0.0;
  std::vector<TrackedEntity> players;
  std::optional<TrackedEntity> ball;
};
struct PanDecisionEvent {
  std::uint64_t frame_index = 0;
  ViewportPosition pose;
};
struct PannerDebugEvent {
  std::uint64_t frame_index = 0;
  float cluster_yaw = 0.0F;
  float cluster_pitch = 0.0F;
  float cluster_spread = 0.0F;
  std::uint32_t n_players = 0;
  bool ball_near_cluster = false;
  float ball_presence = 0.0F;
  float effective_ball_weight = 0.0F;
  float target_yaw = 0.0F;
  float target_pitch = 0.0F;
  float fov_target = 0.0F;
};
struct PosePresentedEvent {
  std::uint64_t frame_index = 0;
  ViewportPosition pose;
};
struct FrameCompleteEvent {
  std::uint64_t frame_index = 0;
  double timestamp_ms = 0.0;
  FrameTimingMicros timing;
  std::uint32_t detection_count = 0;
  std::uint32_t active_tracks = 0;
  bool ball_present = false;
};

class PipelineEvent {
public:
  using Variant =
      std::variant<FrameStartEvent, DetectionsRawEvent, WorldStateEvent, PanDecisionEvent,
                   PannerDebugEvent, PosePresentedEvent, FrameCompleteEvent>;

  explicit PipelineEvent(Variant event);
  [[nodiscard]] std::uint64_t frame_index() const;
  [[nodiscard]] const Variant& variant() const;

private:
  Variant event_;
};

} // namespace reco::core
