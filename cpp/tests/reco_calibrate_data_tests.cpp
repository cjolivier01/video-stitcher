#include "reco/calibrate/geometry.hpp"
#include "reco/calibrate/types.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
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

OptParams good_params() {
  return {.x_ty = 0.0, .intersect = 0.5, .cam_d = 0.25, .x_rz = 0.0, .z_rx = 0.0};
}

void config_defaults_and_validation_match_rust() {
  CalibrationConfig config;
  expect_eq(config.num_frames, 2U, "default num_frames");
  expect_eq(config.use_imu_rotation_seeds, false, "default imu seeds disabled");
  expect_near(config.akaze.threshold, 0.0001, 1.0e-12, "default akaze threshold");
  expect_eq(config.akaze.max_keypoints, 2000U, "default max keypoints");
  expect_near(config.matching.lowe_ratio, 0.75, 1.0e-12, "default lowe ratio");
  expect_near(config.matching.max_y_disparity, 0.08, 1.0e-12, "default y disparity");
  expect_near(config.optimizer.seam_sigma, 0.08, 1.0e-12, "default seam sigma");
  expect_eq(config.optimizer.max_iters, 5000U, "default optimizer iters");
  expect_true(!config.validate().has_value(), "default config validates");

  config.num_frames = 0;
  expect_true(config.validate().value_or("").find("num_frames") != std::string::npos,
              "invalid num_frames");
  config = CalibrationConfig{};
  config.skip_start_secs = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("skip durations") != std::string::npos,
              "invalid nan skip start");
  config = CalibrationConfig{};
  config.skip_end_secs = -0.1;
  expect_true(config.validate().value_or("").find("skip durations") != std::string::npos,
              "invalid skip end");
  config = CalibrationConfig{};
  config.matching.lowe_ratio = 0.0;
  expect_true(config.validate().value_or("").find("lowe_ratio") != std::string::npos,
              "invalid lowe ratio");
  config = CalibrationConfig{};
  config.matching.lowe_ratio = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("lowe_ratio") != std::string::npos,
              "invalid nan lowe ratio");
  config = CalibrationConfig{};
  config.matching.ransac_threshold = 0.0;
  expect_true(config.validate().value_or("").find("ransac_threshold") != std::string::npos,
              "invalid ransac");
  config = CalibrationConfig{};
  config.matching.ransac_threshold = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("ransac_threshold") != std::string::npos,
              "invalid nan ransac");
  config = CalibrationConfig{};
  config.akaze.max_keypoints = 0;
  expect_true(config.validate().value_or("").find("max_keypoints") != std::string::npos,
              "invalid max keypoints");
  config = CalibrationConfig{};
  config.akaze.detect_y_min = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("detect_y") != std::string::npos,
              "invalid nan detect y min");
  config = CalibrationConfig{};
  config.akaze.detect_y_min = 0.9;
  config.akaze.detect_y_max = 0.1;
  expect_true(config.validate().value_or("").find("detect_y") != std::string::npos,
              "invalid detect y order");
  config = CalibrationConfig{};
  config.akaze.threshold = 0.0;
  expect_true(config.validate().value_or("").find("akaze_threshold") != std::string::npos,
              "invalid akaze threshold");
  config = CalibrationConfig{};
  config.akaze.threshold = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("akaze_threshold") != std::string::npos,
              "invalid nan akaze threshold");
  config = CalibrationConfig{};
  config.optimizer.trim_fraction = 1.1;
  expect_true(config.validate().value_or("").find("trim_fraction") != std::string::npos,
              "invalid trim fraction");
  config = CalibrationConfig{};
  config.optimizer.trim_fraction = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("trim_fraction") != std::string::npos,
              "invalid nan trim fraction");
  config = CalibrationConfig{};
  config.matching.spatial_x_threshold = -0.1;
  expect_true(config.validate().value_or("").find("spatial_x_threshold") != std::string::npos,
              "invalid spatial x");
  config = CalibrationConfig{};
  config.matching.spatial_x_threshold = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("spatial_x_threshold") != std::string::npos,
              "invalid nan spatial x");
  config = CalibrationConfig{};
  config.matching.spatial_x_inner = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("spatial_x_inner") != std::string::npos,
              "invalid nan spatial x inner");
  config = CalibrationConfig{};
  config.matching.spatial_y_low = 0.9;
  config.matching.spatial_y_high = 0.1;
  expect_true(config.validate().value_or("").find("spatial_y") != std::string::npos,
              "invalid spatial y order");
  config = CalibrationConfig{};
  config.matching.max_y_disparity = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("max_y_disparity") != std::string::npos,
              "invalid nan max y disparity");
  config = CalibrationConfig{};
  config.imu_xrz_seed = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("IMU") != std::string::npos,
              "invalid nan imu seed");
  config = CalibrationConfig{};
  config.optimizer.seam_sigma = 0.0;
  expect_true(config.validate().value_or("").find("seam_sigma") != std::string::npos,
              "invalid seam sigma");
  config = CalibrationConfig{};
  config.optimizer.seam_sigma = std::numeric_limits<double>::quiet_NaN();
  expect_true(config.validate().value_or("").find("seam_sigma") != std::string::npos,
              "invalid nan seam sigma");

  expect_eq(calibration_step_name(CalibrationStep::DetectingProfiles),
            std::string("DetectingProfiles"), "calibration step display");
  CalibrationProgress progress{.step = CalibrationStep::Optimizing, .detail = "refining"};
  expect_eq(calibration_step_name(progress.step), std::string("Optimizing"),
            "calibration progress step");
  expect_eq(progress.detail, std::string("refining"), "calibration progress detail");
}

void matched_point_and_params_match_rust() {
  const auto point = MatchedPoint::from_planes({0.1, 0.2}, {0.3, 0.4});
  expect_near(point.left[0], 0.1, 1.0e-12, "matched point left");
  expect_near(point.right[1], 0.4, 1.0e-12, "matched point right");
  expect_near(point.left_pixel_nx, 0.5, 1.0e-12, "matched point default left pixel");
  expect_near(point.right_pixel_nx, 0.5, 1.0e-12, "matched point default right pixel");

  const auto params = OptParams{
      .x_ty = 0.01, .intersect = 0.55, .cam_d = 0.24, .x_rz = 0.008, .z_rx = -0.004, .z_rz = 0.003};
  const auto p5 = OptParams::from_5param(params.to_5param());
  expect_near(p5.x_ty, params.x_ty, 1.0e-15, "5 param x_ty");
  expect_near(p5.intersect, params.intersect, 1.0e-15, "5 param intersect");
  const auto p6 = OptParams::from_6param(params.to_6param());
  expect_true(p6.z_rz.has_value(), "6 param z_rz");
  expect_near(*p6.z_rz, *params.z_rz, 1.0e-15, "6 param z_rz value");
}

void geometry_matches_rust_reference_tests() {
  const auto x = to_3d_x_plane({0.3, 0.1});
  expect_near(x.x, 0.3, 1.0e-10, "x plane x");
  expect_near(x.y, -0.1, 1.0e-10, "x plane y");
  expect_near(x.z, 0.0, 1.0e-10, "x plane z");
  const auto z = to_3d_z_plane({0.2, 0.05});
  expect_near(z.x, 0.0, 1.0e-10, "z plane x");
  expect_near(z.y, -0.05, 1.0e-10, "z plane y");
  expect_near(z.z, -0.2, 1.0e-10, "z plane z");

  const auto identity = rotation_matrix(0.0, 0.0, 0.0);
  expect_near(identity[0][0], 1.0, 1.0e-10, "identity 00");
  expect_near(identity[1][1], 1.0, 1.0e-10, "identity 11");
  expect_near(identity[2][2], 1.0, 1.0e-10, "identity 22");
  const auto rz90 = rotation_matrix(0.0, 0.0, 1.5707963267948966);
  expect_near(rz90[0][0], 0.0, 1.0e-10, "rz90 cos");
  expect_near(rz90[1][0], 1.0, 1.0e-10, "rz90 sin");

  const auto center = normalize_to_plane(960.0, 540.0, 1920, 1080);
  expect_near(center[0], 0.0, 1.0e-10, "normalize center x");
  expect_near(center[1], 0.0, 1.0e-10, "normalize center y");
  const auto top_left = normalize_to_plane(0.0, 0.0, 1920, 1080);
  expect_near(top_left[0], -0.5, 1.0e-10, "normalize top left x");
  expect_near(top_left[1], -0.28125, 1.0e-10, "normalize top left y");
}

void reprojection_costs_match_rust_properties() {
  const std::vector<MatchedPoint> perfect{MatchedPoint::from_planes({0.0, 0.0}, {0.0, 0.0})};
  const OptParams perfect_params{
      .x_ty = 0.0, .intersect = 1.0, .cam_d = 0.25, .x_rz = 0.0, .z_rx = 0.0};
  expect_near(reprojection_error(perfect, perfect_params), 0.0, 1.0e-6,
              "perfect reprojection error");
  expect_near(angular_error(perfect, perfect_params), 0.0, 1.0e-6, "perfect angular error");

  const std::vector<MatchedPoint> points{MatchedPoint::from_planes({-0.1, 0.0}, {0.5, 0.0}),
                                         MatchedPoint::from_planes({-0.2, 0.05}, {0.4, 0.05})};
  auto bad = good_params();
  bad.x_ty = 0.3;
  bad.x_rz = 0.2;
  expect_true(reprojection_error(points, bad) > reprojection_error(points, good_params()),
              "reprojection increases with misalignment");
  expect_true(angular_error(points, bad) > angular_error(points, good_params()),
              "angular error increases with misalignment");

  const auto per_point = per_point_reprojection_error(points, good_params());
  expect_eq(per_point.size(), points.size(), "per point error count");
  expect_true(trimmed_reprojection_error({}, good_params(), 0.5) == 0.0,
              "empty trimmed reprojection");

  const std::vector<MatchedPoint> mixed_ray{MatchedPoint::from_planes({0.5, 0.1}, {0.5, 0.0})};
  const auto mixed_errors = per_point_reprojection_error(mixed_ray, good_params());
  expect_eq(mixed_errors.size(), 1U, "mixed ray error count");
  expect_near(mixed_errors[0], 1.0e6, 1.0e-6, "per-point invalid ray returns penalty");
  expect_true(reprojection_error(mixed_ray, good_params()) > 1.0e6,
              "aggregate invalid ray still adds valid backward error");
}

void seam_weighting_matches_rust_properties() {
  const auto params = good_params();
  const MatchedPoint near_seam{
      .left = {0.1, 0.0}, .right = {0.1, 0.0}, .left_pixel_nx = 0.25, .right_pixel_nx = 0.75};
  const MatchedPoint far_from_seam{
      .left = {0.1, 0.0}, .right = {0.1, 0.0}, .left_pixel_nx = 0.8, .right_pixel_nx = 0.2};
  const double near_error = seam_weighted_reprojection_error({near_seam}, params, 0.08);
  const double far_error = seam_weighted_reprojection_error({far_from_seam}, params, 0.08);
  expect_true(near_error > far_error, "near seam weighs more");
  expect_true(far_error < near_error * 1.0e-6, "far seam nearly zero weighted");
  expect_true(trimmed_seam_weighted_reprojection_error({}, params, 0.08, 0.5) == 0.0,
              "empty trimmed seam weighted");
  const double nan_sigma = seam_weighted_reprojection_error(
      {near_seam}, params, std::numeric_limits<double>::quiet_NaN());
  const double tiny_sigma = seam_weighted_reprojection_error({near_seam}, params, 1.0e-6);
  expect_near(nan_sigma, tiny_sigma, 1.0e-12, "nan seam sigma clamps like Rust max");
}

} // namespace

int main() {
  config_defaults_and_validation_match_rust();
  matched_point_and_params_match_rust();
  geometry_matches_rust_reference_tests();
  reprojection_costs_match_rust_properties();
  seam_weighting_matches_rust_properties();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
