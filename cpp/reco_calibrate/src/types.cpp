#include "reco/calibrate/types.hpp"

#include <cmath>
#include <sstream>

namespace reco::calibrate {

namespace {

bool finite_in_range(double value, double min, double max, bool min_exclusive = false) {
  return std::isfinite(value) && (min_exclusive ? value > min : value >= min) && value <= max;
}

} // namespace

MatchedPoint MatchedPoint::from_planes(std::array<double, 2> left_plane,
                                       std::array<double, 2> right_plane) {
  return {.left = left_plane, .right = right_plane, .left_pixel_nx = 0.5, .right_pixel_nx = 0.5};
}

std::optional<std::string> CalibrationConfig::validate() const {
  if (num_frames == 0) {
    return "num_frames must be >= 1";
  }
  if (!std::isfinite(skip_start_secs) || !std::isfinite(skip_end_secs) || skip_start_secs < 0.0 ||
      skip_end_secs < 0.0) {
    return "skip durations must be finite and non-negative";
  }
  if (!finite_in_range(matching.lowe_ratio, 0.0, 1.0, true)) {
    std::ostringstream out;
    out << "lowe_ratio must be in (0, 1], got " << matching.lowe_ratio;
    return out.str();
  }
  if (!std::isfinite(matching.ransac_threshold) || matching.ransac_threshold <= 0.0) {
    std::ostringstream out;
    out << "ransac_threshold must be > 0, got " << matching.ransac_threshold;
    return out.str();
  }
  if (akaze.max_keypoints == 0) {
    return "max_keypoints must be >= 1";
  }
  if (!finite_in_range(akaze.detect_y_min, 0.0, 1.0) ||
      !finite_in_range(akaze.detect_y_max, 0.0, 1.0) || akaze.detect_y_min >= akaze.detect_y_max) {
    return "detect_y range must be ordered within [0, 1]";
  }
  if (!std::isfinite(akaze.threshold) || akaze.threshold <= 0.0) {
    std::ostringstream out;
    out << "akaze_threshold must be > 0, got " << akaze.threshold;
    return out.str();
  }
  if (!finite_in_range(optimizer.trim_fraction, 0.0, 1.0)) {
    std::ostringstream out;
    out << "trim_fraction must be in [0, 1], got " << optimizer.trim_fraction;
    return out.str();
  }
  if (!finite_in_range(matching.spatial_x_threshold, 0.0, 1.0)) {
    std::ostringstream out;
    out << "spatial_x_threshold must be in [0, 1], got " << matching.spatial_x_threshold;
    return out.str();
  }
  if (!finite_in_range(matching.spatial_x_inner, 0.0, 1.0)) {
    std::ostringstream out;
    out << "spatial_x_inner must be in [0, 1], got " << matching.spatial_x_inner;
    return out.str();
  }
  if (!finite_in_range(matching.spatial_y_low, 0.0, 1.0) ||
      !finite_in_range(matching.spatial_y_high, 0.0, 1.0) ||
      matching.spatial_y_low > matching.spatial_y_high) {
    return "spatial_y range must be ordered within [0, 1]";
  }
  if (!finite_in_range(matching.max_y_disparity, 0.0, 1.0)) {
    std::ostringstream out;
    out << "max_y_disparity must be in [0, 1], got " << matching.max_y_disparity;
    return out.str();
  }
  if ((imu_xrz_seed.has_value() && !std::isfinite(*imu_xrz_seed)) ||
      (imu_xrx_seed.has_value() && !std::isfinite(*imu_xrx_seed)) ||
      (imu_zrx_seed.has_value() && !std::isfinite(*imu_zrx_seed))) {
    return "IMU rotation seeds must be finite";
  }
  if (!std::isfinite(optimizer.seam_sigma) || optimizer.seam_sigma <= 0.0) {
    std::ostringstream out;
    out << "seam_sigma must be > 0, got " << optimizer.seam_sigma;
    return out.str();
  }
  return std::nullopt;
}

std::string calibration_step_name(CalibrationStep step) {
  switch (step) {
  case CalibrationStep::Probing:
    return "Probing";
  case CalibrationStep::DetectingProfiles:
    return "DetectingProfiles";
  case CalibrationStep::AudioSync:
    return "AudioSync";
  case CalibrationStep::ExtractingFrames:
    return "ExtractingFrames";
  case CalibrationStep::Undistorting:
    return "Undistorting";
  case CalibrationStep::FeatureMatching:
    return "FeatureMatching";
  case CalibrationStep::Optimizing:
    return "Optimizing";
  }
  return "?";
}

} // namespace reco::calibrate
