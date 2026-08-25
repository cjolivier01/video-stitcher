#include "reco/calibrate/types.hpp"

#include <cmath>
#include <sstream>

namespace reco::calibrate {

MatchedPoint MatchedPoint::from_planes(std::array<double, 2> left_plane,
                                       std::array<double, 2> right_plane) {
  return {.left = left_plane, .right = right_plane, .left_pixel_nx = 0.5, .right_pixel_nx = 0.5};
}

std::optional<std::string> CalibrationConfig::validate() const {
  if (num_frames == 0) {
    return "num_frames must be >= 1";
  }
  if (matching.lowe_ratio <= 0.0 || matching.lowe_ratio > 1.0) {
    std::ostringstream out;
    out << "lowe_ratio must be in (0, 1], got " << matching.lowe_ratio;
    return out.str();
  }
  if (matching.ransac_threshold <= 0.0) {
    std::ostringstream out;
    out << "ransac_threshold must be > 0, got " << matching.ransac_threshold;
    return out.str();
  }
  if (akaze.max_keypoints == 0) {
    return "max_keypoints must be >= 1";
  }
  if (akaze.threshold <= 0.0) {
    std::ostringstream out;
    out << "akaze_threshold must be > 0, got " << akaze.threshold;
    return out.str();
  }
  if (std::isnan(optimizer.trim_fraction) || optimizer.trim_fraction < 0.0 ||
      optimizer.trim_fraction > 1.0) {
    std::ostringstream out;
    out << "trim_fraction must be in [0, 1], got " << optimizer.trim_fraction;
    return out.str();
  }
  if (matching.spatial_x_threshold < 0.0 || matching.spatial_x_threshold > 1.0) {
    std::ostringstream out;
    out << "spatial_x_threshold must be in [0, 1], got " << matching.spatial_x_threshold;
    return out.str();
  }
  if (optimizer.seam_sigma <= 0.0) {
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
