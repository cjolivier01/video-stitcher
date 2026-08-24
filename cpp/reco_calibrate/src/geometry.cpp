#include "reco/calibrate/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace reco::calibrate {
namespace {

Vec3d add(Vec3d a, Vec3d b) { return {.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z}; }
Vec3d sub(Vec3d a, Vec3d b) { return {.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z}; }
Vec3d mul(double scalar, Vec3d v) { return {.x = scalar * v.x, .y = scalar * v.y, .z = scalar * v.z}; }
double dot(Vec3d a, Vec3d b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double norm(Vec3d v) { return std::sqrt(dot(v, v)); }
Vec3d normalize(Vec3d v) {
  const double n = norm(v);
  return n < 1.0e-15 ? Vec3d{} : mul(1.0 / n, v);
}

double rust_max(double lhs, double rhs) {
  if (std::isnan(lhs)) {
    return rhs;
  }
  if (std::isnan(rhs)) {
    return lhs;
  }
  return std::max(lhs, rhs);
}

Vec3d mat_mul(const std::array<std::array<double, 3>, 3>& m, Vec3d v) {
  return {.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
          .y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
          .z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
}

std::vector<double> per_point_seam_weighted_errors_full(const std::vector<MatchedPoint>& points,
                                                        const OptParams& params,
                                                        const SeamWeightConfig& config) {
  const Vec3d camera{params.cam_d, 0.0, params.cam_d};
  const auto transformed = apply_transformations(points, params);
  const double left_cam_seam = 1.0 - params.intersect / 2.0;
  const double right_cam_seam = params.intersect / 2.0;
  const double sx = rust_max(config.sigma_x, 1.0e-6);
  const double sy = rust_max(config.sigma_y, 1.0e-6);
  const double inv_2sigma_sq = 1.0 / (2.0 * sx * sx);
  const double inv_2sigma_y_sq = 1.0 / (2.0 * sy * sy);

  std::vector<double> errors;
  errors.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& x_pt = transformed.x_plane[i];
    const auto& z_pt = transformed.z_plane[i];
    const double dl = points[i].left_pixel_nx - right_cam_seam;
    const double dr = points[i].right_pixel_nx - left_cam_seam;
    const double w_horiz =
        0.5 * (std::exp(-dl * dl * inv_2sigma_sq) + std::exp(-dr * dr * inv_2sigma_sq));
    const double yl = points[i].left[1] - config.y_center;
    const double yr = points[i].right[1] - config.y_center;
    const double w_vert =
        0.5 * (std::exp(-yl * yl * inv_2sigma_y_sq) + std::exp(-yr * yr * inv_2sigma_y_sq));
    const double w = w_horiz * w_vert;
    double err = 0.0;

    const Vec3d dir_x = sub(x_pt, camera);
    if (std::abs(dir_x.x) > 1.0e-15) {
      const double t = -camera.x / dir_x.x;
      if (t > 0.0) {
        const Vec3d hit = add(camera, mul(t, dir_x));
        const double dy = hit.y - z_pt.y;
        const double dz = hit.z - z_pt.z;
        err += w * (dy * dy + dz * dz);
      } else {
        err += w * 1.0e6;
      }
    }

    const Vec3d dir_z = sub(z_pt, camera);
    if (std::abs(dir_z.z) > 1.0e-15) {
      const double t = -camera.z / dir_z.z;
      if (t > 0.0) {
        const Vec3d hit = add(camera, mul(t, dir_z));
        const double dx = hit.x - x_pt.x;
        const double dy = hit.y - x_pt.y;
        err += w * (dx * dx + dy * dy);
      } else {
        err += w * 1.0e6;
      }
    }
    errors.push_back(err);
  }
  return errors;
}

} // namespace

OptParams OptParams::from_5param(const std::array<double, 5>& values) {
  return {.x_ty = values[0],
          .intersect = values[1],
          .cam_d = values[2],
          .x_rz = values[3],
          .z_rx = values[4]};
}

OptParams OptParams::from_6param(const std::array<double, 6>& values) {
  return {.x_ty = values[0],
          .intersect = values[1],
          .cam_d = values[2],
          .x_rz = values[3],
          .z_rx = values[4],
          .z_rz = values[5]};
}

std::array<double, 5> OptParams::to_5param() const {
  return {x_ty, intersect, cam_d, x_rz, z_rx};
}

std::array<double, 6> OptParams::to_6param() const {
  return {x_ty, intersect, cam_d, x_rz, z_rx, z_rz.value_or(0.0)};
}

SeamWeightConfig SeamWeightConfig::from_sigma(double sigma) {
  return {.sigma_x = sigma, .sigma_y = 0.08, .y_center = -0.05};
}

Vec3d to_3d_x_plane(std::array<double, 2> point) {
  return {.x = point[0], .y = -point[1], .z = 0.0};
}

Vec3d to_3d_z_plane(std::array<double, 2> point) {
  return {.x = 0.0, .y = -point[1], .z = -point[0]};
}

std::array<std::array<double, 3>, 3> rotation_matrix(double rx, double ry, double rz) {
  const double sx = std::sin(rx);
  const double cx = std::cos(rx);
  const double sy = std::sin(ry);
  const double cy = std::cos(ry);
  const double sz = std::sin(rz);
  const double cz = std::cos(rz);
  return {{
      {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx},
      {sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx},
      {-sy, cy * sx, cy * cx},
  }};
}

TransformedPoints apply_transformations(const std::vector<MatchedPoint>& points,
                                        const OptParams& params) {
  const double half_offset = kPlaneWidth / 2.0 * (1.0 - params.intersect);
  const auto r_x_plane = rotation_matrix(0.0, 0.0, params.x_rz);
  const Vec3d t_x_plane{half_offset, params.x_ty, 0.0};
  const auto r_z_plane = rotation_matrix(params.z_rx, 0.0, params.z_rz.value_or(0.0));
  const Vec3d t_z_plane{0.0, 0.0, half_offset};

  TransformedPoints out;
  out.x_plane.reserve(points.size());
  out.z_plane.reserve(points.size());
  for (const auto& point : points) {
    out.x_plane.push_back(add(mat_mul(r_x_plane, to_3d_x_plane(point.left)), t_x_plane));
    out.z_plane.push_back(add(mat_mul(r_z_plane, to_3d_z_plane(point.right)), t_z_plane));
  }
  return out;
}

double reprojection_error(const std::vector<MatchedPoint>& points, const OptParams& params) {
  const Vec3d camera{params.cam_d, 0.0, params.cam_d};
  const auto transformed = apply_transformations(points, params);
  double total = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& x_pt = transformed.x_plane[i];
    const auto& z_pt = transformed.z_plane[i];
    const Vec3d dir_x = sub(x_pt, camera);
    if (std::abs(dir_x.x) > 1.0e-15) {
      const double t = -camera.x / dir_x.x;
      if (t > 0.0) {
        const Vec3d hit = add(camera, mul(t, dir_x));
        const double dy = hit.y - z_pt.y;
        const double dz = hit.z - z_pt.z;
        total += dy * dy + dz * dz;
      } else {
        total += 1.0e6;
      }
    }
    const Vec3d dir_z = sub(z_pt, camera);
    if (std::abs(dir_z.z) > 1.0e-15) {
      const double t = -camera.z / dir_z.z;
      if (t > 0.0) {
        const Vec3d hit = add(camera, mul(t, dir_z));
        const double dx = hit.x - x_pt.x;
        const double dy = hit.y - x_pt.y;
        total += dx * dx + dy * dy;
      } else {
        total += 1.0e6;
      }
    }
  }
  return total;
}

std::vector<double> per_point_reprojection_error(const std::vector<MatchedPoint>& points,
                                                 const OptParams& params) {
  const Vec3d camera{params.cam_d, 0.0, params.cam_d};
  const auto transformed = apply_transformations(points, params);
  std::vector<double> errors;
  errors.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& x_pt = transformed.x_plane[i];
    const auto& z_pt = transformed.z_plane[i];
    double err = 0.0;
    const Vec3d dir_x = sub(x_pt, camera);
    if (std::abs(dir_x.x) > 1.0e-15) {
      const double t = -camera.x / dir_x.x;
      if (t > 0.0) {
        const Vec3d hit = add(camera, mul(t, dir_x));
        const double dy = hit.y - z_pt.y;
        const double dz = hit.z - z_pt.z;
        err += dy * dy + dz * dz;
      } else {
        errors.push_back(1.0e6);
        continue;
      }
    }
    const Vec3d dir_z = sub(z_pt, camera);
    if (std::abs(dir_z.z) > 1.0e-15) {
      const double t = -camera.z / dir_z.z;
      if (t > 0.0) {
        const Vec3d hit = add(camera, mul(t, dir_z));
        const double dx = hit.x - x_pt.x;
        const double dy = hit.y - x_pt.y;
        err += dx * dx + dy * dy;
      } else {
        errors.push_back(1.0e6);
        continue;
      }
    }
    errors.push_back(err);
  }
  return errors;
}

double trimmed_reprojection_error(const std::vector<MatchedPoint>& points, const OptParams& params,
                                  double trim_fraction) {
  if (points.empty()) {
    return 0.0;
  }
  auto errors = per_point_reprojection_error(points, params);
  std::sort(errors.begin(), errors.end(),
            [](double a, double b) { return std::isnan(a) || std::isnan(b) ? false : a < b; });
  const auto raw_keep = static_cast<std::size_t>(
      std::ceil((1.0 - trim_fraction) * static_cast<double>(errors.size())));
  const auto keep = std::max<std::size_t>(1, std::min(raw_keep, errors.size()));
  return std::accumulate(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(keep), 0.0);
}

double angular_error(const std::vector<MatchedPoint>& points, const OptParams& params) {
  const Vec3d camera{params.cam_d, 0.0, params.cam_d};
  const auto transformed = apply_transformations(points, params);
  double total = 0.0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Vec3d dx = sub(transformed.x_plane[i], camera);
    const Vec3d dz = sub(transformed.z_plane[i], camera);
    if (norm(dx) < 1.0e-15 || norm(dz) < 1.0e-15) {
      continue;
    }
    const double d = std::clamp(dot(normalize(dx), normalize(dz)), -1.0, 1.0);
    total += std::acos(d);
  }
  return total;
}

std::vector<double> per_point_seam_weighted_errors(const std::vector<MatchedPoint>& points,
                                                   const OptParams& params, double sigma) {
  return per_point_seam_weighted_errors_full(points, params, SeamWeightConfig::from_sigma(sigma));
}

double seam_weighted_reprojection_error(const std::vector<MatchedPoint>& points,
                                        const OptParams& params, double sigma) {
  const auto errors = per_point_seam_weighted_errors(points, params, sigma);
  return std::accumulate(errors.begin(), errors.end(), 0.0);
}

double trimmed_seam_weighted_reprojection_error(const std::vector<MatchedPoint>& points,
                                                const OptParams& params, double sigma,
                                                double trim_fraction) {
  if (points.empty()) {
    return 0.0;
  }
  auto errors = per_point_seam_weighted_errors(points, params, sigma);
  std::sort(errors.begin(), errors.end(),
            [](double a, double b) { return std::isnan(a) || std::isnan(b) ? false : a < b; });
  const auto raw_keep = static_cast<std::size_t>(
      std::ceil((1.0 - trim_fraction) * static_cast<double>(errors.size())));
  const auto keep = std::max<std::size_t>(1, std::min(raw_keep, errors.size()));
  return std::accumulate(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(keep), 0.0);
}

std::array<double, 2> normalize_to_plane(double px, double py, std::uint32_t img_w,
                                         std::uint32_t img_h) {
  const double w = static_cast<double>(std::max<std::uint32_t>(img_w, 1));
  const double h = static_cast<double>(std::max<std::uint32_t>(img_h, 1));
  return {(px / w - 0.5) * kPlaneWidth, (py / h - 0.5) * kPlaneWidth * (h / w)};
}

} // namespace reco::calibrate
