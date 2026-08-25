#include "reco/autocam/sweep_panner.hpp"

#include <algorithm>
#include <cmath>

namespace reco::autocam {
namespace {

constexpr float kTau = 6.2831853071795864769F;

} // namespace

SweepPanner::SweepPanner(float yaw_range, float cycle_secs)
    : yaw_range_(yaw_range), cycle_secs_(std::max(cycle_secs, 0.1F)) {}

SweepPanner SweepPanner::with_fov(float fov_degrees) const {
  auto copy = *this;
  copy.fov_degrees_ = fov_degrees;
  return copy;
}

SweepPanner SweepPanner::with_zoom(float fov_min, float fov_max, float cycle_secs) const {
  auto copy = *this;
  copy.fov_min_ = fov_min;
  copy.fov_max_ = fov_max;
  copy.zoom_cycle_secs_ = std::max(cycle_secs, 0.1F);
  return copy;
}

SweepPanner SweepPanner::with_fps(float fps) const {
  auto copy = *this;
  copy.fps_ = std::max(fps, 1.0F);
  return copy;
}

reco::core::ViewportPosition SweepPanner::decide(const WorldState&,
                                                 const PanContext& context) const {
  const float t = static_cast<float>(context.frame_index) / fps_;
  const float yaw_phase = std::sin(t * kTau / cycle_secs_);
  float fov = fov_degrees_;
  if (zoom_cycle_secs_ > 0.0F) {
    const float zoom_phase = std::sin(t * kTau / zoom_cycle_secs_);
    const float mid = (fov_min_ + fov_max_) * 0.5F;
    const float amp = (fov_max_ - fov_min_) * 0.5F;
    fov = mid + zoom_phase * amp;
  }
  return {.yaw = yaw_phase * yaw_range_, .pitch = 0.0F, .fov_degrees = fov};
}

} // namespace reco::autocam
