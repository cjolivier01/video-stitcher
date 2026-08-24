#pragma once

#include "reco/core/viewport_position.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace reco::core {

struct Vec3 {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
};

[[nodiscard]] bool point_in_polygon(std::array<double, 2> point,
                                    const std::vector<std::array<double, 2>>& polygon);

class VirtualCamera {
public:
  explicit VirtualCamera(const std::array<float, 3>& camera_position);

  [[nodiscard]] static Vec3 world_up();
  [[nodiscard]] ViewportPosition direction_to_yaw_pitch(const Vec3& direction) const;
  [[nodiscard]] Vec3 yaw_pitch_to_direction(float yaw, float pitch) const;

  Vec3 eye;
  Vec3 base_forward;
  Vec3 base_right;
};

} // namespace reco::core
