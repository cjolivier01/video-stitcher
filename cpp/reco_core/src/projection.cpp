#include "reco/core/projection.hpp"

#include <algorithm>
#include <cmath>

namespace reco::core {
namespace {

Vec3 operator-(const Vec3& lhs, const Vec3& rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3 operator*(const Vec3& vector, float scalar) {
  return {vector.x * scalar, vector.y * scalar, vector.z * scalar};
}

float dot(const Vec3& lhs, const Vec3& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
  return {
      lhs.y * rhs.z - lhs.z * rhs.y,
      lhs.z * rhs.x - lhs.x * rhs.z,
      lhs.x * rhs.y - lhs.y * rhs.x,
  };
}

float norm(const Vec3& vector) { return std::sqrt(dot(vector, vector)); }

Vec3 normalize(const Vec3& vector) {
  const float length = norm(vector);
  if (length <= 1.0e-6F) {
    return {};
  }
  return vector * (1.0F / length);
}

} // namespace

bool point_in_polygon(std::array<double, 2> point,
                      const std::vector<std::array<double, 2>>& polygon) {
  const std::size_t n = polygon.size();
  if (n < 3) {
    return false;
  }

  const auto [px, py] = point;
  bool inside = false;
  std::size_t j = n - 1;
  for (std::size_t i = 0; i < n; ++i) {
    const auto [xi, yi] = polygon[i];
    const auto [xj, yj] = polygon[j];
    if (((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}

Vec3 VirtualCamera::world_up() { return {0.0F, 1.0F, 0.0F}; }

VirtualCamera::VirtualCamera(const std::array<float, 3>& camera_position)
    : eye{camera_position[0], camera_position[1], camera_position[2]} {
  base_forward = normalize({-eye.x, -eye.y, -eye.z});
  base_right = normalize(cross(base_forward, world_up()));
}

ViewportPosition VirtualCamera::direction_to_yaw_pitch(const Vec3& direction) const {
  ViewportPosition position;
  position.fov_degrees = std::nullopt;
  position.pitch = std::asin(std::clamp(direction.y, -1.0F, 1.0F));

  const Vec3 horizontal{direction.x, 0.0F, direction.z};
  const float horizontal_length = norm(horizontal);
  if (horizontal_length > 1.0e-6F) {
    const Vec3 h = horizontal * (1.0F / horizontal_length);
    const float cos_yaw = std::clamp(dot(h, base_forward), -1.0F, 1.0F);
    const float sin_yaw = -dot(h, base_right);
    position.yaw = std::atan2(sin_yaw, cos_yaw);
  }
  return position;
}

Vec3 VirtualCamera::yaw_pitch_to_direction(float yaw, float pitch) const {
  const float cos_pitch = std::cos(pitch);
  const Vec3 horizontal =
      (base_forward * (cos_pitch * std::cos(yaw))) - (base_right * (cos_pitch * std::sin(yaw)));
  return {horizontal.x, std::sin(pitch), horizontal.z};
}

} // namespace reco::core
