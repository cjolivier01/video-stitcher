#pragma once

#include "reco/control/intents.hpp"
#include "reco/core/viewport_position.hpp"

namespace reco::control {

struct PoseControlConfig {
  float drag_deg_per_pixel = 0.1F;
  float wheel_fov_per_tick = 3.0F;
  float smoothing = 0.3F;
  float fov_min_degrees = 20.0F;
  float fov_max_degrees = 150.0F;
  bool invert_drag_x = false;
  bool invert_drag_y = false;
  float hotkey_yaw_step_rad = 0.0872664626F;
  float hotkey_pitch_step_rad = 0.0872664626F;
  float hotkey_fov_step_deg = 5.0F;
  reco::core::ViewportPosition rest_pose{};
};

class PoseControl {
public:
  explicit PoseControl(PoseControlConfig config = {});

  void apply_drag(float dx_pixels, float dy_pixels);
  void apply_wheel(float ticks);
  void apply_hotkey(HotkeyIntent intent);
  void set_target(reco::core::ViewportPosition pose);
  void snap_to_rest();
  void set_target_fov(float fov_deg);
  void tick();
  void tick_with(float smoothing);
  [[nodiscard]] reco::core::ViewportPosition target_pose() const;
  [[nodiscard]] reco::core::ViewportPosition current_pose() const;
  [[nodiscard]] reco::core::ViewportPosition render_pose(float rig_tilt) const;
  [[nodiscard]] float current_yaw_rad() const;
  [[nodiscard]] float current_pitch_rad() const;
  [[nodiscard]] float current_fov_deg() const;
  [[nodiscard]] const PoseControlConfig& config() const;
  void set_config(PoseControlConfig config);
  void set_fov_max_degrees(float max_deg);

private:
  float target_yaw_rad_ = 0.0F;
  float target_pitch_rad_ = 0.0F;
  float target_fov_deg_ = 75.0F;
  float current_yaw_rad_ = 0.0F;
  float current_pitch_rad_ = 0.0F;
  float current_fov_deg_ = 75.0F;
  PoseControlConfig config_;
};

} // namespace reco::control

