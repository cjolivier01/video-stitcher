#include "reco/autocam/class_provider.hpp"

namespace reco::autocam {

ClassProvider::ClassProvider(std::uint16_t class_id) : class_id_(class_id) {}

std::vector<reco::core::TrackedEntity>
ClassProvider::update(const std::vector<reco::core::MappedDetection>& detections, double) {
  std::vector<reco::core::TrackedEntity> out;
  out.reserve(detections.size());
  for (const auto& detection : detections) {
    if (detection.class_id != class_id_ || !detection.position.has_value()) {
      continue;
    }
    out.push_back({
        .id = 0,
        .class_id = class_id_,
        .yaw = detection.position->yaw,
        .pitch = detection.position->pitch,
        .confidence = detection.confidence,
        .state = reco::core::TrackState::Tracking,
        .age_frames = 0,
        .origin = detection.camera,
    });
  }
  return out;
}

} // namespace reco::autocam
