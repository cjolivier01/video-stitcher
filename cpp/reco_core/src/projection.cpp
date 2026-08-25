#include "reco/core/projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace reco::core {
namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kPiOver2 = kPi * 0.5F;

Vec3 operator+(const Vec3& lhs, const Vec3& rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

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

struct Mat4 {
  float m[4][4]{};
};

Mat4 identity() {
  Mat4 result;
  for (int i = 0; i < 4; ++i) {
    result.m[i][i] = 1.0F;
  }
  return result;
}

Mat4 multiply(const Mat4& lhs, const Mat4& rhs) {
  Mat4 result;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      for (int k = 0; k < 4; ++k) {
        result.m[row][col] += lhs.m[row][k] * rhs.m[k][col];
      }
    }
  }
  return result;
}

Mat4 translation(const std::array<float, 3>& position) {
  Mat4 result = identity();
  result.m[0][3] = position[0];
  result.m[1][3] = position[1];
  result.m[2][3] = position[2];
  return result;
}

Mat4 rotation_x(float angle) {
  Mat4 result = identity();
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  result.m[1][1] = c;
  result.m[1][2] = -s;
  result.m[2][1] = s;
  result.m[2][2] = c;
  return result;
}

Mat4 rotation_y(float angle) {
  Mat4 result = identity();
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  result.m[0][0] = c;
  result.m[0][2] = s;
  result.m[2][0] = -s;
  result.m[2][2] = c;
  return result;
}

Mat4 rotation_z(float angle) {
  Mat4 result = identity();
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  result.m[0][0] = c;
  result.m[0][1] = -s;
  result.m[1][0] = s;
  result.m[1][1] = c;
  return result;
}

Mat4 euler(float rx, float ry, float rz) {
  return multiply(rotation_z(rz), multiply(rotation_y(ry), rotation_x(rx)));
}

Vec3 transform_point(const Mat4& matrix, Vec3 point) {
  return {
      matrix.m[0][0] * point.x + matrix.m[0][1] * point.y + matrix.m[0][2] * point.z +
          matrix.m[0][3],
      matrix.m[1][0] * point.x + matrix.m[1][1] * point.y + matrix.m[1][2] * point.z +
          matrix.m[1][3],
      matrix.m[2][0] * point.x + matrix.m[2][1] * point.y + matrix.m[2][2] * point.z +
          matrix.m[2][3],
  };
}

Vec3 transform_vector(const Mat4& matrix, Vec3 vector) {
  return {
      matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y + matrix.m[0][2] * vector.z,
      matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y + matrix.m[1][2] * vector.z,
      matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y + matrix.m[2][2] * vector.z,
  };
}

Mat4 model_matrix(CameraId camera, const SceneGeometry& scene) {
  if (camera == CameraId::Left) {
    const auto base = euler(0.0F, scene.left_rotation[1], scene.left_rotation[2]);
    const auto roll = euler(scene.left_rotation[0], 0.0F, 0.0F);
    return multiply(translation(scene.left_position), multiply(roll, base));
  }
  return multiply(translation(scene.right_position),
                  euler(scene.right_rotation[0], scene.right_rotation[1], scene.right_rotation[2]));
}

} // namespace

SceneGeometry SceneGeometry::from_layout_with_aspect(const PlaneLayout& layout, float aspect) {
  SceneGeometry geometry;
  geometry.plane_width = 1.0F;
  geometry.plane_aspect = aspect;
  const float half_offset =
      (geometry.plane_width / 2.0F) * (1.0F - static_cast<float>(layout.intersect));
  geometry.left_position = {0.0F, 0.0F, half_offset};
  geometry.left_rotation = {static_cast<float>(layout.z_rx), kPiOver2,
                            static_cast<float>(layout.z_rz)};
  geometry.right_position = {half_offset, static_cast<float>(layout.x_ty), 0.0F};
  geometry.right_rotation = {static_cast<float>(layout.x_rx), 0.0F,
                             static_cast<float>(layout.x_rz)};
  geometry.camera_position = {static_cast<float>(layout.camera_axis_offset), 0.0F,
                              static_cast<float>(layout.camera_axis_offset)};
  return geometry;
}

ViewportPosition ViewportBounds::clamp(ViewportPosition position) const {
  position.yaw = std::clamp(position.yaw, min_yaw, max_yaw);
  position.pitch = std::clamp(position.pitch, min_pitch, max_pitch);
  return position;
}

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

double kb4_theta_d(double theta, const std::array<double, 4>& d) {
  const double theta2 = theta * theta;
  const double theta4 = theta2 * theta2;
  const double theta6 = theta4 * theta2;
  const double theta8 = theta4 * theta4;
  return theta * (1.0 + d[0] * theta2 + d[1] * theta4 + d[2] * theta6 + d[3] * theta8);
}

double kb4_theta_d_prime(double theta, const std::array<double, 4>& d) {
  const double theta2 = theta * theta;
  const double theta4 = theta2 * theta2;
  const double theta6 = theta4 * theta2;
  const double theta8 = theta4 * theta4;
  return 1.0 + 3.0 * d[0] * theta2 + 5.0 * d[1] * theta4 + 7.0 * d[2] * theta6 +
         9.0 * d[3] * theta8;
}

double kb4_forward_scale(double r, const std::array<double, 4>& d) {
  if (r < 1.0e-10) {
    return 1.0;
  }
  return kb4_theta_d(std::atan(r), d) / r;
}

std::pair<double, double> forward_fisheye(double uv_x, double uv_y, const CameraParams& params) {
  const double w = params.width;
  const double h = params.height;
  const double fx = params.fx / w;
  const double fy = params.fy / h;
  const double cx = params.cx / w;
  const double cy = params.cy / h;
  const double x = (uv_x - cx) / fx;
  const double y = (uv_y - cy) / fy;
  const double r = std::sqrt(x * x + y * y);
  if (r < 1.0e-12) {
    return {cx, cy};
  }
  const double scale = kb4_forward_scale(r, params.d);
  return {fx * x * scale + cx, fy * y * scale + cy};
}

std::optional<std::pair<double, double>> inverse_fisheye(double dist_x, double dist_y,
                                                         const CameraParams& params) {
  constexpr int kMaxIterations = 20;
  constexpr double kConvergenceEpsilon = 1.0e-10;
  const double w = params.width;
  const double h = params.height;
  const double fx = params.fx / w;
  const double fy = params.fy / h;
  const double cx = params.cx / w;
  const double cy = params.cy / h;
  const double dx = (dist_x - cx) / fx;
  const double dy = (dist_y - cy) / fy;
  const double theta_d = std::sqrt(dx * dx + dy * dy);
  if (theta_d < 1.0e-12) {
    return std::pair{cx, cy};
  }
  double theta = theta_d;
  for (int i = 0; i < kMaxIterations; ++i) {
    const double f = kb4_theta_d(theta, params.d) - theta_d;
    const double f_prime = kb4_theta_d_prime(theta, params.d);
    if (std::abs(f_prime) < 1.0e-15) {
      return std::nullopt;
    }
    const double delta = f / f_prime;
    theta -= delta;
    if (std::abs(delta) < kConvergenceEpsilon) {
      break;
    }
  }
  const double r = std::tan(theta);
  const double scale = std::abs(theta) < 1.0e-12 ? 1.0 : theta_d / r;
  if (!std::isfinite(scale)) {
    return std::nullopt;
  }
  return std::pair{fx * (dx / scale) + cx, fy * (dy / scale) + cy};
}

Vec3 plane_uv_to_world(std::pair<double, double> uv, CameraId camera, const SceneGeometry& scene) {
  const float tex_u = static_cast<float>((uv.first + 0.5) / 2.0);
  const float tex_v = static_cast<float>((uv.second + 0.5) / 2.0);
  const Vec3 local{tex_u - 0.5F, (0.5F - tex_v) / scene.plane_aspect, 0.0F};
  return transform_point(model_matrix(camera, scene), local);
}

std::optional<std::pair<double, double>> world_to_plane_uv(Vec3 world, CameraId camera,
                                                           const SceneGeometry& scene) {
  const Mat4 model = model_matrix(camera, scene);
  const Vec3 origin = transform_point(model, {0.0F, 0.0F, 0.0F});
  const Vec3 right = transform_vector(model, {1.0F, 0.0F, 0.0F});
  const Vec3 up = transform_vector(model, {0.0F, 1.0F, 0.0F});
  const Vec3 rel = world - origin;
  const float right_len2 = dot(right, right);
  const float up_len2 = dot(up, up);
  if (right_len2 <= 1.0e-12F || up_len2 <= 1.0e-12F) {
    return std::nullopt;
  }
  const float local_x = dot(rel, right) / right_len2;
  const float local_y = dot(rel, up) / up_len2;
  const float tex_u = local_x / scene.plane_width + 0.5F;
  const float tex_v = 0.5F - local_y * scene.plane_aspect / scene.plane_width;
  return std::pair{static_cast<double>(tex_u * 2.0F - 0.5F),
                   static_cast<double>(tex_v * 2.0F - 0.5F)};
}

std::optional<ViewportPosition> camera_to_panorama(CameraId camera, float norm_x, float norm_y,
                                                   const MatchCalibration& calibration,
                                                   const SceneGeometry& scene) {
  const CameraParams& params = camera == CameraId::Left ? calibration.left : calibration.right;
  const auto plane_uv = inverse_fisheye(norm_x, norm_y, params);
  if (!plane_uv.has_value()) {
    return std::nullopt;
  }
  const Vec3 world = plane_uv_to_world(*plane_uv, camera, scene);
  const Vec3 cam_pos{scene.camera_position[0], scene.camera_position[1], scene.camera_position[2]};
  return VirtualCamera(scene.camera_position).direction_to_yaw_pitch(normalize(world - cam_pos));
}

std::optional<std::pair<float, float>> panorama_to_camera(float yaw, float pitch, CameraId camera,
                                                          const MatchCalibration& calibration,
                                                          const SceneGeometry& scene) {
  const CameraParams& params = camera == CameraId::Left ? calibration.left : calibration.right;
  const VirtualCamera virtual_camera(scene.camera_position);
  const Vec3 dir = virtual_camera.yaw_pitch_to_direction(yaw, pitch);
  const Vec3 cam_pos = virtual_camera.eye;
  const Mat4 model = model_matrix(camera, scene);
  const Vec3 plane_origin = transform_point(model, {0.0F, 0.0F, 0.0F});
  const Vec3 plane_normal = normalize(transform_vector(model, {0.0F, 0.0F, 1.0F}));
  const float denom = dot(plane_normal, dir);
  if (std::abs(denom) < 1.0e-6F) {
    return std::nullopt;
  }
  const float t = dot(plane_origin - cam_pos, plane_normal) / denom;
  if (t <= 0.0F) {
    return std::nullopt;
  }
  const Vec3 hit = cam_pos + (dir * t);
  const auto uv = world_to_plane_uv(hit, camera, scene);
  if (!uv.has_value()) {
    return std::nullopt;
  }
  const double tex_u = (uv->first + 0.5) * 0.5;
  const double tex_v = (uv->second + 0.5) * 0.5;
  if (tex_u < 0.0 || tex_u > 1.0 || tex_v < 0.0 || tex_v > 1.0) {
    return std::nullopt;
  }
  const auto norm = forward_fisheye(uv->first, uv->second, params);
  if (norm.first < 0.0 || norm.first > 1.0 || norm.second < 0.0 || norm.second > 1.0) {
    return std::nullopt;
  }
  return std::pair{static_cast<float>(norm.first), static_cast<float>(norm.second)};
}

ViewportBounds viewport_bounds(float fov_degrees, const MatchCalibration& calibration,
                               const SceneGeometry& scene, float aspect) {
  const float half_vfov = (fov_degrees * 0.5F) * kPi / 180.0F;
  const float half_hfov = std::atan(std::tan(half_vfov) * aspect);
  const float corner_hfov = std::atan(std::tan(half_hfov) / std::cos(half_vfov));
  const float corner_vfov = std::atan(std::tan(half_vfov) / std::cos(half_hfov));
  constexpr std::uint32_t kEdgeSteps = 40;
  constexpr float kLo = 0.02F;
  constexpr float kHi = 0.98F;
  std::vector<std::pair<float, float>> frontier;
  frontier.reserve((kEdgeSteps + 1) * 8);
  for (const CameraId camera : {CameraId::Left, CameraId::Right}) {
    for (std::uint32_t i = 0; i <= kEdgeSteps; ++i) {
      const float t = kLo + (kHi - kLo) * (static_cast<float>(i) / static_cast<float>(kEdgeSteps));
      for (const auto [nx, ny] :
           {std::pair{kLo, t}, std::pair{kHi, t}, std::pair{t, kLo}, std::pair{t, kHi}}) {
        if (const auto pos = camera_to_panorama(camera, nx, ny, calibration, scene);
            pos.has_value()) {
          frontier.push_back({pos->yaw, pos->pitch});
        }
      }
    }
  }
  if (frontier.empty()) {
    return {};
  }
  float pitch_min = std::numeric_limits<float>::max();
  float pitch_max = std::numeric_limits<float>::lowest();
  float yaw_min = std::numeric_limits<float>::max();
  float yaw_max = std::numeric_limits<float>::lowest();
  for (const auto [yaw, pitch] : frontier) {
    pitch_min = std::min(pitch_min, pitch);
    pitch_max = std::max(pitch_max, pitch);
    yaw_min = std::min(yaw_min, yaw);
    yaw_max = std::max(yaw_max, yaw);
  }
  constexpr std::size_t kBins = 20;
  constexpr std::size_t kMinPointsPerBin = 4;
  const float pitch_bin_size = (pitch_max - pitch_min) / static_cast<float>(kBins);
  float bound_min_yaw = std::numeric_limits<float>::lowest();
  float bound_max_yaw = std::numeric_limits<float>::max();
  for (std::size_t bin = 0; bin < kBins; ++bin) {
    const float bin_lo = pitch_min + static_cast<float>(bin) * pitch_bin_size;
    const float bin_hi = bin_lo + pitch_bin_size;
    float yaw_lo = std::numeric_limits<float>::max();
    float yaw_hi = std::numeric_limits<float>::lowest();
    std::size_t count = 0;
    for (const auto [yaw, pitch] : frontier) {
      if (pitch >= bin_lo && pitch < bin_hi) {
        yaw_lo = std::min(yaw_lo, yaw);
        yaw_hi = std::max(yaw_hi, yaw);
        ++count;
      }
    }
    if (count >= kMinPointsPerBin) {
      bound_min_yaw = std::max(bound_min_yaw, yaw_lo + corner_hfov);
      bound_max_yaw = std::min(bound_max_yaw, yaw_hi - corner_hfov);
    }
  }
  const float yaw_bin_size = (yaw_max - yaw_min) / static_cast<float>(kBins);
  float bound_min_pitch = std::numeric_limits<float>::lowest();
  float bound_max_pitch = std::numeric_limits<float>::max();
  for (std::size_t bin = 0; bin < kBins; ++bin) {
    const float bin_lo = yaw_min + static_cast<float>(bin) * yaw_bin_size;
    const float bin_hi = bin_lo + yaw_bin_size;
    float p_lo = std::numeric_limits<float>::max();
    float p_hi = std::numeric_limits<float>::lowest();
    std::size_t count = 0;
    for (const auto [yaw, pitch] : frontier) {
      if (yaw >= bin_lo && yaw < bin_hi) {
        p_lo = std::min(p_lo, pitch);
        p_hi = std::max(p_hi, pitch);
        ++count;
      }
    }
    if (count >= kMinPointsPerBin) {
      bound_min_pitch = std::max(bound_min_pitch, p_lo + corner_vfov);
      bound_max_pitch = std::min(bound_max_pitch, p_hi - corner_vfov);
    }
  }
  if (bound_min_yaw == std::numeric_limits<float>::lowest()) {
    bound_min_yaw = yaw_min + corner_hfov;
  }
  if (bound_max_yaw == std::numeric_limits<float>::max()) {
    bound_max_yaw = yaw_max - corner_hfov;
  }
  if (bound_min_pitch == std::numeric_limits<float>::lowest()) {
    bound_min_pitch = pitch_min + corner_vfov;
  }
  if (bound_max_pitch == std::numeric_limits<float>::max()) {
    bound_max_pitch = pitch_max - corner_vfov;
  }
  if (bound_min_yaw > bound_max_yaw) {
    const float mid = (bound_min_yaw + bound_max_yaw) * 0.5F;
    bound_min_yaw = mid;
    bound_max_yaw = mid;
  }
  if (bound_min_pitch > bound_max_pitch) {
    const float mid = (bound_min_pitch + bound_max_pitch) * 0.5F;
    bound_min_pitch = mid;
    bound_max_pitch = mid;
  }
  return {bound_min_yaw, bound_max_yaw, bound_min_pitch, bound_max_pitch};
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
