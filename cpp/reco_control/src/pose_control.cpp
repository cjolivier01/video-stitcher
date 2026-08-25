#include "reco/control/pose_control.hpp"

#include <algorithm>

#include "reco/core/rig_correction.hpp"

namespace reco::control {
namespace {

float radians(float degrees) { return degrees * 0.017453292519943295769F; }

} // namespace

PoseControl::PoseControl(PoseControlConfig config) : config_(config) {
  const auto rest = config_.rest_pose;
  const float fov = rest.fov_degrees.value_or(75.0F);
  target_yaw_rad_ = rest.yaw;
  target_pitch_rad_ = rest.pitch;
  target_fov_deg_ = fov;
  current_yaw_rad_ = rest.yaw;
  current_pitch_rad_ = rest.pitch;
  current_fov_deg_ = fov;
}

void PoseControl::apply_drag(float dx_pixels, float dy_pixels) {
  const float raw_dx = config_.invert_drag_x ? dx_pixels : -dx_pixels;
  const float raw_dy = config_.invert_drag_y ? -dy_pixels : dy_pixels;
  target_yaw_rad_ += radians(raw_dx * config_.drag_deg_per_pixel);
  target_pitch_rad_ += radians(raw_dy * config_.drag_deg_per_pixel);
}

void PoseControl::apply_wheel(float ticks) {
  set_target_fov(target_fov_deg_ - ticks * config_.wheel_fov_per_tick);
}

void PoseControl::apply_hotkey(HotkeyIntent intent) {
  switch (intent) {
  case HotkeyIntent::YawLeft:
    target_yaw_rad_ -= config_.hotkey_yaw_step_rad;
    break;
  case HotkeyIntent::YawRight:
    target_yaw_rad_ += config_.hotkey_yaw_step_rad;
    break;
  case HotkeyIntent::PitchUp:
    target_pitch_rad_ += config_.hotkey_pitch_step_rad;
    break;
  case HotkeyIntent::PitchDown:
    target_pitch_rad_ -= config_.hotkey_pitch_step_rad;
    break;
  case HotkeyIntent::ZoomIn:
    set_target_fov(target_fov_deg_ - config_.hotkey_fov_step_deg);
    break;
  case HotkeyIntent::ZoomOut:
    set_target_fov(target_fov_deg_ + config_.hotkey_fov_step_deg);
    break;
  case HotkeyIntent::Reset:
    target_yaw_rad_ = config_.rest_pose.yaw;
    target_pitch_rad_ = config_.rest_pose.pitch;
    target_fov_deg_ = config_.rest_pose.fov_degrees.value_or(target_fov_deg_);
    break;
  case HotkeyIntent::ToggleConstrained:
    break;
  }
}

void PoseControl::set_target(reco::core::ViewportPosition pose) {
  target_yaw_rad_ = pose.yaw;
  target_pitch_rad_ = pose.pitch;
  if (pose.fov_degrees.has_value()) {
    set_target_fov(*pose.fov_degrees);
  }
}

void PoseControl::snap_to_rest() {
  const auto rest = config_.rest_pose;
  const float fov = rest.fov_degrees.value_or(current_fov_deg_);
  target_yaw_rad_ = rest.yaw;
  target_pitch_rad_ = rest.pitch;
  target_fov_deg_ = fov;
  current_yaw_rad_ = rest.yaw;
  current_pitch_rad_ = rest.pitch;
  current_fov_deg_ = fov;
}

void PoseControl::set_target_fov(float fov_deg) {
  const float min_fov = std::min(config_.fov_min_degrees, config_.fov_max_degrees);
  target_fov_deg_ = std::clamp(fov_deg, min_fov, config_.fov_max_degrees);
}

void PoseControl::tick() { tick_with(config_.smoothing); }

void PoseControl::tick_with(float smoothing) {
  const float s = std::clamp(smoothing, 0.0F, 1.0F);
  current_yaw_rad_ += (target_yaw_rad_ - current_yaw_rad_) * s;
  current_pitch_rad_ += (target_pitch_rad_ - current_pitch_rad_) * s;
  current_fov_deg_ += (target_fov_deg_ - current_fov_deg_) * s;
}

reco::core::ViewportPosition PoseControl::target_pose() const {
  return {target_yaw_rad_, target_pitch_rad_, target_fov_deg_};
}

reco::core::ViewportPosition PoseControl::current_pose() const {
  return {current_yaw_rad_, current_pitch_rad_, current_fov_deg_};
}

reco::core::ViewportPosition PoseControl::render_pose(float rig_tilt) const {
  return {current_yaw_rad_, reco::core::render_pitch(current_yaw_rad_, current_pitch_rad_, rig_tilt),
          current_fov_deg_};
}

float PoseControl::current_yaw_rad() const { return current_yaw_rad_; }
float PoseControl::current_pitch_rad() const { return current_pitch_rad_; }
float PoseControl::current_fov_deg() const { return current_fov_deg_; }
const PoseControlConfig& PoseControl::config() const { return config_; }
void PoseControl::set_config(PoseControlConfig config) { config_ = config; }

void PoseControl::set_fov_max_degrees(float max_deg) {
  config_.fov_max_degrees = max_deg;
  target_fov_deg_ = std::min(target_fov_deg_, max_deg);
  current_fov_deg_ = std::min(current_fov_deg_, max_deg);
}

} // namespace reco::control

