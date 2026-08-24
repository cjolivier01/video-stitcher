#include "reco/io/jsonl_sink.hpp"

#include <optional>
#include <type_traits>
#include <utility>

namespace reco::io {
namespace {

nlohmann::json pose_to_json(const reco::core::ViewportPosition& pose) {
  nlohmann::json json{{"yaw", pose.yaw}, {"pitch", pose.pitch}};
  if (pose.fov_degrees.has_value()) {
    json["fov_degrees"] = *pose.fov_degrees;
  } else {
    json["fov_degrees"] = nullptr;
  }
  return json;
}

nlohmann::json timing_to_json(const reco::core::FrameTimingMicros& timing) {
  auto optional_u32 = [](std::optional<std::uint32_t> value) -> nlohmann::json {
    if (!value.has_value()) {
      return nullptr;
    }
    return *value;
  };
  return {
      {"decode_us", optional_u32(timing.decode_us)},
      {"upload_us", optional_u32(timing.upload_us)},
      {"stitch_us", optional_u32(timing.stitch_us)},
      {"readback_us", optional_u32(timing.readback_us)},
      {"submit_us", optional_u32(timing.submit_us)},
      {"detection_us", optional_u32(timing.detection_us)},
      {"total_us", timing.total_us},
  };
}

std::string camera_id_json_name(reco::core::CameraId camera) {
  switch (camera) {
  case reco::core::CameraId::Left:
    return "Left";
  case reco::core::CameraId::Right:
    return "Right";
  }
  return "Left";
}

nlohmann::json detection_to_json(const reco::core::MappedDetection& detection) {
  nlohmann::json json{
      {"camera", camera_id_json_name(detection.camera)},
      {"class_id", detection.class_id},
      {"confidence", detection.confidence},
      {"camera_center", {detection.camera_center_x, detection.camera_center_y}},
      {"camera_size", {detection.camera_size_x, detection.camera_size_y}},
  };
  if (detection.position.has_value()) {
    json["position"] = pose_to_json(*detection.position);
  } else {
    json["position"] = nullptr;
  }
  return json;
}

std::string track_state_name(reco::core::TrackState state) {
  switch (state) {
  case reco::core::TrackState::Tracking:
    return "Tracking";
  case reco::core::TrackState::Coasting:
    return "Coasting";
  case reco::core::TrackState::Lost:
    return "Lost";
  }
  return "Tracking";
}

nlohmann::json entity_to_json(const reco::core::TrackedEntity& entity) {
  return {
      {"id", entity.id},
      {"class_id", entity.class_id},
      {"yaw", entity.yaw},
      {"pitch", entity.pitch},
      {"confidence", entity.confidence},
      {"state", track_state_name(entity.state)},
      {"age_frames", entity.age_frames},
      {"origin", camera_id_json_name(entity.origin)},
  };
}

} // namespace

nlohmann::json pipeline_event_to_json(const reco::core::PipelineEvent& event) {
  return std::visit(
      [](const auto& value) -> nlohmann::json {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, reco::core::FrameStartEvent>) {
          return {{"kind", "frame_start"},
                  {"frame_index", value.frame_index},
                  {"timestamp_ms", value.timestamp_ms}};
        } else if constexpr (std::is_same_v<T, reco::core::DetectionsRawEvent>) {
          nlohmann::json detections = nlohmann::json::array();
          for (const auto& detection : value.detections) {
            detections.push_back(detection_to_json(detection));
          }
          return {{"kind", "detections_raw"},
                  {"frame_index", value.frame_index},
                  {"detections", std::move(detections)}};
        } else if constexpr (std::is_same_v<T, reco::core::WorldStateEvent>) {
          nlohmann::json players = nlohmann::json::array();
          for (const auto& player : value.players) {
            players.push_back(entity_to_json(player));
          }
          return {{"kind", "world_state"},
                  {"frame_index", value.frame_index},
                  {"timestamp_ms", value.timestamp_ms},
                  {"players", std::move(players)},
                  {"ball",
                   value.ball.has_value() ? entity_to_json(*value.ball) : nlohmann::json(nullptr)}};
        } else if constexpr (std::is_same_v<T, reco::core::PanDecisionEvent>) {
          return {{"kind", "pan_decision"},
                  {"frame_index", value.frame_index},
                  {"pose", pose_to_json(value.pose)}};
        } else if constexpr (std::is_same_v<T, reco::core::PannerDebugEvent>) {
          return {{"kind", "panner_debug"},
                  {"frame_index", value.frame_index},
                  {"cluster_yaw", value.cluster_yaw},
                  {"cluster_pitch", value.cluster_pitch},
                  {"cluster_spread", value.cluster_spread},
                  {"n_players", value.n_players},
                  {"ball_near_cluster", value.ball_near_cluster},
                  {"ball_presence", value.ball_presence},
                  {"effective_ball_weight", value.effective_ball_weight},
                  {"target_yaw", value.target_yaw},
                  {"target_pitch", value.target_pitch},
                  {"fov_target", value.fov_target}};
        } else if constexpr (std::is_same_v<T, reco::core::PosePresentedEvent>) {
          return {{"kind", "pose_presented"},
                  {"frame_index", value.frame_index},
                  {"pose", pose_to_json(value.pose)}};
        } else {
          return {{"kind", "frame_complete"},
                  {"frame_index", value.frame_index},
                  {"timestamp_ms", value.timestamp_ms},
                  {"timing", timing_to_json(value.timing)},
                  {"detection_count", value.detection_count},
                  {"active_tracks", value.active_tracks},
                  {"ball_present", value.ball_present}};
        }
      },
      event.variant());
}

JsonlSink::JsonlSink(const std::filesystem::path& path)
    : writer_(path, std::ios::binary | std::ios::trunc) {}

void JsonlSink::emit(const reco::core::PipelineEvent& event) {
  if (!writer_) {
    ++write_failures_;
    return;
  }
  writer_ << pipeline_event_to_json(event).dump() << '\n';
  if (!writer_) {
    ++write_failures_;
  }
}

void JsonlSink::flush() {
  writer_.flush();
  if (!writer_) {
    ++write_failures_;
  }
}

} // namespace reco::io
