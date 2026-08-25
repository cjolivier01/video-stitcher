#pragma once

#include "reco/core/calibration.hpp"
#include "reco/core/source.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace reco::calibrate {

struct GrayFrame {
  std::vector<std::uint8_t> data;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

using YuvFrame = reco::core::YuvFrame;

struct MatchedPoint {
  std::array<double, 2> left{};
  std::array<double, 2> right{};
  double left_pixel_nx = 0.0;
  double right_pixel_nx = 0.0;

  [[nodiscard]] static MatchedPoint from_planes(std::array<double, 2> left,
                                                std::array<double, 2> right);
};

struct FrameMatches {
  std::vector<MatchedPoint> points;
  std::size_t keypoints_left = 0;
  std::size_t keypoints_right = 0;
  std::size_t min_descriptors = 0;
  std::size_t post_ratio_test = 0;
  std::size_t post_spatial_filter = 0;
  std::size_t post_ransac = 0;
};

struct AkazeConfig {
  double threshold = 0.0001;
  std::size_t max_keypoints = 2000;
  double detect_y_min = 0.05;
  double detect_y_max = 0.95;
};

struct MatchConfig {
  double lowe_ratio = 0.75;
  std::size_t min_matches = 6;
  double spatial_x_threshold = 0.5;
  double spatial_x_inner = 0.0;
  double spatial_y_low = 0.2;
  double spatial_y_high = 0.8;
  double max_y_disparity = 0.08;
  double ransac_threshold = 1.0;
};

struct OptimizerConfig {
  bool lock_cam_d = false;
  bool lock_z_rx = false;
  bool enable_x_rx = false;
  double seam_sigma = 0.08;
  double trim_fraction = 0.3;
  std::size_t max_iters = 5000;
};

struct CalibrationConfig {
  std::size_t num_frames = 2;
  double skip_start_secs = 0.0;
  double skip_end_secs = 0.0;
  bool use_imu_rotation_seeds = false;
  std::optional<double> imu_xrz_seed;
  std::optional<double> imu_xrx_seed;
  std::optional<double> imu_zrx_seed;
  AkazeConfig akaze;
  MatchConfig matching;
  OptimizerConfig optimizer;

  [[nodiscard]] std::optional<std::string> validate() const;
};

enum class CalibrationStep {
  Probing,
  DetectingProfiles,
  AudioSync,
  ExtractingFrames,
  Undistorting,
  FeatureMatching,
  Optimizing,
};

[[nodiscard]] std::string calibration_step_name(CalibrationStep step);

struct CalibrationProgress {
  CalibrationStep step = CalibrationStep::Probing;
  std::string detail;
};

enum class ProfileSource {
  Database,
  File,
  AutoDetected,
  Fallback,
};

struct LensProfileInfo {
  std::string camera;
  std::string lens;
  ProfileSource source = ProfileSource::Database;
  std::optional<std::string> path;
};

struct LensProfileSummary {
  std::string camera;
  std::string lens;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct CalibrationQuality {
  double mean_reprojection_error = 0.0;
  double trimmed_reprojection_error = 0.0;
  double angular_error = 0.0;
};

struct CalibrationResult {
  reco::core::MatchCalibration calibration;
  std::size_t total_matches = 0;
  std::size_t frames_used = 0;
  double residual_error = 0.0;
  double confidence = 0.0;
  std::vector<FrameMatches> per_frame;
  std::optional<LensProfileInfo> left_lens_profile;
  std::optional<LensProfileInfo> right_lens_profile;
  std::optional<CalibrationQuality> quality;
};

} // namespace reco::calibrate
