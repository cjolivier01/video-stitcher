#pragma once

#include "reco/core/viewport_position.hpp"

#include "reco/core/calibration.hpp"
#include "reco/core/pipeline_event.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace reco::core {

struct Vec3 {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
};

struct SceneGeometry {
  std::array<float, 3> left_position{};
  std::array<float, 3> left_rotation{};
  std::array<float, 3> right_position{};
  std::array<float, 3> right_rotation{};
  std::array<float, 3> camera_position{};
  float plane_width = 1.0F;
  float plane_aspect = 16.0F / 9.0F;

  [[nodiscard]] static SceneGeometry from_layout_with_aspect(const PlaneLayout& layout,
                                                             float aspect);
};

struct ViewportBounds {
  float min_yaw = 0.0F;
  float max_yaw = 0.0F;
  float min_pitch = 0.0F;
  float max_pitch = 0.0F;

  [[nodiscard]] ViewportPosition clamp(ViewportPosition position) const;
};

[[nodiscard]] bool point_in_polygon(std::array<double, 2> point,
                                    const std::vector<std::array<double, 2>>& polygon);
[[nodiscard]] double kb4_theta_d(double theta, const std::array<double, 4>& d);
[[nodiscard]] double kb4_theta_d_prime(double theta, const std::array<double, 4>& d);
[[nodiscard]] double kb4_forward_scale(double r, const std::array<double, 4>& d);
[[nodiscard]] std::pair<double, double> forward_fisheye(double uv_x, double uv_y,
                                                        const CameraParams& params);
[[nodiscard]] std::optional<std::pair<double, double>> inverse_fisheye(double dist_x, double dist_y,
                                                                       const CameraParams& params);
[[nodiscard]] Vec3 plane_uv_to_world(std::pair<double, double> uv, CameraId camera,
                                     const SceneGeometry& scene);
[[nodiscard]] std::optional<std::pair<double, double>>
world_to_plane_uv(Vec3 world, CameraId camera, const SceneGeometry& scene);
[[nodiscard]] std::optional<ViewportPosition>
camera_to_panorama(CameraId camera, float norm_x, float norm_y, const MatchCalibration& calibration,
                   const SceneGeometry& scene);
[[nodiscard]] std::optional<std::pair<float, float>>
panorama_to_camera(float yaw, float pitch, CameraId camera, const MatchCalibration& calibration,
                   const SceneGeometry& scene);
[[nodiscard]] ViewportBounds viewport_bounds(float fov_degrees, const MatchCalibration& calibration,
                                             const SceneGeometry& scene, float aspect);

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
