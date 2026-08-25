#include "reco/calibrate/optimizer.hpp"

#include "reco/calibrate/geometry.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace reco::calibrate;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

std::vector<MatchedPoint> synthetic_points(const OptParams& true_params, std::size_t n) {
  const double half_offset = kPlaneWidth / 2.0 * (1.0 - true_params.intersect);
  const Vec3d cam{true_params.cam_d, 0.0, true_params.cam_d};
  std::vector<MatchedPoint> points;
  points.reserve(n);
  const auto grid = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));

  for (std::size_t iy = 0; iy < grid; ++iy) {
    for (std::size_t ix = 0; ix < grid; ++ix) {
      if (points.size() >= n) {
        break;
      }
      const double fx = (static_cast<double>(ix) + 0.5) / static_cast<double>(grid);
      const double fy = (static_cast<double>(iy) + 0.5) / static_cast<double>(grid);
      const double yaw = -0.9 + fx * 0.5;
      const double pitch = (fy - 0.5) * 0.4;
      Vec3d d{yaw, pitch, yaw - 0.3};
      const double norm = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
      d.x /= norm;
      d.y /= norm;
      d.z /= norm;
      if (std::abs(d.z) < 1.0e-10 || std::abs(d.x) < 1.0e-10) {
        continue;
      }
      const double t_x = -cam.z / d.z;
      const double t_z = -cam.x / d.x;
      if (t_x < 0.0 || t_z < 0.0) {
        continue;
      }
      const Vec3d hit_x{cam.x + t_x * d.x, cam.y + t_x * d.y, cam.z + t_x * d.z};
      const Vec3d hit_z{cam.x + t_z * d.x, cam.y + t_z * d.y, cam.z + t_z * d.z};
      points.push_back(MatchedPoint::from_planes({hit_x.x - half_offset,
                                                  -(hit_x.y - true_params.x_ty)},
                                                 {-(hit_z.z - half_offset), -hit_z.y}));
    }
  }
  return points;
}

void parameter_helpers_match_rust() {
  const auto bounds = active_bounds(false, false, false);
  expect_eq(bounds.size(), 5U, "5d bounds count");
  expect_near(bounds[0].first, 0.1, 1.0e-12, "cam_d lower bound");
  expect_near(bounds[4].second, 0.3, 1.0e-12, "z_rx upper bound");
  expect_eq(active_bounds(true, false, false).size(), 6U, "x_rx adds bound");
  expect_eq(active_bounds(false, true, false).size(), 4U, "locked cam_d removes bound");
  expect_eq(active_bounds(false, false, true).size(), 4U, "locked z_rx removes bound");

  const auto params = params_from_vec({0.225, 0.5, 0.01, 0.02, -0.03, 0.04});
  expect_near(params.cam_d, 0.225, 1.0e-12, "params cam_d");
  expect_true(params.x_rx.has_value(), "params x_rx present");
  expect_near(*params.x_rx, 0.04, 1.0e-12, "params x_rx value");

  const auto locked = params_from_vec_locked({0.7, 0.01, 0.02, -0.03});
  expect_near(locked.cam_d, 0.15, 1.0e-12, "locked cam_d derived");
  expect_near(locked.intersect, 0.7, 1.0e-12, "locked intersect");

  const auto no_zrx = params_from_vec_no_zrx({0.225, 0.5, 0.01, 0.02, 0.04});
  expect_near(no_zrx.z_rx, 0.0, 1.0e-12, "no zrx fixed");
  expect_true(no_zrx.x_rx.has_value(), "no zrx x_rx present");
}

void penalty_and_simplex_match_rust() {
  const auto bounds = active_bounds(false, false, false);
  expect_near(bounds_penalty({0.225, 0.5, 0.0, 0.0, 0.0}, bounds), 0.0, 1.0e-15,
              "inside bounds has zero penalty");
  expect_true(bounds_penalty({0.05, 0.5, 0.0, 0.0, 0.0}, bounds) > 0.0,
              "outside bounds has penalty");

  const auto simplex = build_simplex({0.225, 0.5, 0.0, 0.0, 0.0}, bounds);
  expect_eq(simplex.size(), 6U, "5d simplex size");
  for (const auto& vertex : simplex) {
    for (std::size_t i = 0; i < vertex.size(); ++i) {
      expect_true(vertex[i] >= bounds[i].first - 1.0e-10 &&
                      vertex[i] <= bounds[i].second + 1.0e-10,
                  "simplex vertex inside bounds");
    }
  }

  const auto bounds6 = active_bounds(true, false, false);
  auto simplex6 = build_simplex({0.225, 0.5, 0.0, 0.0, 0.0, 0.0}, bounds6);
  expect_eq(simplex6.size(), 7U, "6d simplex size");
}

void optimizer_recovers_known_params() {
  const OptParams true_params{
      .x_ty = 0.01, .intersect = 0.5, .cam_d = 0.225, .x_rz = 0.0, .z_rx = 0.0};
  const auto points = synthetic_points(true_params, 50);
  expect_true(points.size() >= 10, "synthetic point count");
  expect_true(reprojection_error(points, true_params) < 0.001,
              "synthetic points match ground truth");

  CalibrationConfig config;
  config.optimizer.max_iters = 800;
  const auto result = optimize(points, config);
  expect_near(result.layout.camera_axis_offset, 0.225, 0.06, "optimized cam_d");
  expect_near(result.layout.intersect, 0.5, 0.12, "optimized intersect");
  expect_near(result.layout.x_ty, 0.01, 0.03, "optimized x_ty");

  const Optimizer& backend = NelderMeadOptimizer();
  const auto via_backend = backend.optimize(points, config);
  expect_near(via_backend.layout.camera_axis_offset, 0.225, 0.06,
              "optimizer interface cam_d");
}

void optimizer_handles_small_rotations() {
  const OptParams true_params{
      .x_ty = 0.005, .intersect = 0.55, .cam_d = 0.24, .x_rz = 0.01, .z_rx = -0.005};
  const auto points = synthetic_points(true_params, 50);

  CalibrationConfig config;
  config.optimizer.max_iters = 800;
  const auto result = optimize(points, config);
  expect_near(result.layout.camera_axis_offset, 0.24, 0.06, "small rotation cam_d");
  expect_near(result.layout.intersect, 0.55, 0.12, "small rotation intersect");
}

void optimizer_honors_locks_and_xrx() {
  const OptParams true_params{
      .x_ty = 0.0, .intersect = 0.5, .cam_d = 0.25, .x_rz = 0.0, .z_rx = 0.0};
  const auto points = synthetic_points(true_params, 30);
  CalibrationConfig config;
  config.optimizer.max_iters = 200;
  config.optimizer.lock_cam_d = true;
  config.optimizer.lock_z_rx = true;
  config.optimizer.enable_x_rx = true;
  config.imu_xrx_seed = 0.02;
  const auto result = optimize(points, config);
  expect_near(result.layout.camera_axis_offset, 0.5 * (1.0 - result.layout.intersect), 1.0e-12,
              "locked cam_d derived from intersect");
  expect_near(result.layout.z_rx, 0.0, 1.0e-12, "locked z_rx layout");
}

} // namespace

int main() {
  parameter_helpers_match_rust();
  penalty_and_simplex_match_rust();
  optimizer_recovers_known_params();
  optimizer_handles_small_rotations();
  optimizer_honors_locks_and_xrx();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
