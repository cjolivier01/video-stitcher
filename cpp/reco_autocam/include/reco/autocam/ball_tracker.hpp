#pragma once

#include "reco/autocam/coaster.hpp"
#include "reco/autocam/sweep_panner.hpp"
#include "reco/core/pipeline_event.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace reco::autocam {

inline constexpr float kDefaultMaxJumpRad = 0.35F;
inline constexpr std::uint32_t kDefaultCoastFrames = 20;
inline constexpr float kDefaultPlayerAnchorRad = 0.20F;

class BallTracker {
public:
  explicit BallTracker(std::uint16_t class_id);

  [[nodiscard]] BallTracker with_max_jump_rad(float rad) const;
  [[nodiscard]] BallTracker with_max_coast_frames(std::uint32_t frames) const;
  [[nodiscard]] BallTracker with_player_anchor_rad(float rad) const;

  void set_players(const std::vector<reco::core::TrackedEntity>& players);
  void observe_world(const WorldState& world);

  [[nodiscard]] std::vector<reco::core::TrackedEntity>
  update(const std::vector<reco::core::MappedDetection>& detections, double timestamp_ms);

  [[nodiscard]] std::uint16_t class_id() const { return class_id_; }

private:
  struct LastKnown {
    float yaw = 0.0F;
    float pitch = 0.0F;
    reco::core::CameraId origin = reco::core::CameraId::Left;
  };

  [[nodiscard]] std::optional<float> score(const reco::core::MappedDetection& detection) const;
  [[nodiscard]] bool passes_player_anchor(float yaw, float pitch) const;

  std::uint16_t class_id_ = 0;
  Coaster coaster_{kDefaultCoastFrames};
  std::optional<LastKnown> last_;
  float max_jump_rad_ = kDefaultMaxJumpRad;
  float player_anchor_max_rad_ = kDefaultPlayerAnchorRad;
  std::vector<std::pair<float, float>> current_players_;
  std::uint64_t age_frames_ = 0;
};

} // namespace reco::autocam
