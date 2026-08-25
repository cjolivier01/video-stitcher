#pragma once

#include "reco/core/calibration.hpp"
#include "reco/core/pipeline_event.hpp"
#include "reco/core/viewport_position.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace reco::autocam {

struct WorldState {
  std::vector<reco::core::TrackedEntity> players;
  std::optional<reco::core::TrackedEntity> ball;
};

struct PanContext {
  std::uint64_t frame_index = 0;
  double timestamp_ms = 0.0;
  reco::core::ViewportPosition previous_position;
  const reco::core::MatchCalibration* calibration = nullptr;
};

class SweepPanner {
public:
  SweepPanner(float yaw_range, float cycle_secs);

  [[nodiscard]] SweepPanner with_fov(float fov_degrees) const;
  [[nodiscard]] SweepPanner with_zoom(float fov_min, float fov_max, float cycle_secs) const;
  [[nodiscard]] SweepPanner with_fps(float fps) const;

  [[nodiscard]] reco::core::ViewportPosition decide(const WorldState& world,
                                                    const PanContext& context) const;

private:
  float yaw_range_ = 0.0F;
  float cycle_secs_ = 0.1F;
  float fov_degrees_ = 50.0F;
  float fov_min_ = 0.0F;
  float fov_max_ = 0.0F;
  float zoom_cycle_secs_ = 0.0F;
  float fps_ = 30.0F;
};

} // namespace reco::autocam
