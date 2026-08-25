#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace reco::core {

inline constexpr std::uint32_t kMaxCalibrationDimension = 8192;
inline constexpr std::int64_t kMaxSyncOffsetFrames = 100000;
inline constexpr std::uint64_t kMaxCalibrationFileSize = 1048576;

struct CameraParams {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::array<double, 4> d{};
};

struct PlaneLayout {
  double camera_axis_offset = 0.0;
  double intersect = 0.0;
  double x_ty = 0.0;
  double x_rz = 0.0;
  double z_rx = 0.0;
  double x_rx = 0.0;
  double z_rz = 0.0;
};

struct FieldRoi {
  std::vector<std::array<double, 2>> left;
  std::vector<std::array<double, 2>> right;
};

struct MatchCalibration {
  CameraParams left;
  CameraParams right;
  PlaneLayout layout;
  double rig_tilt = 0.0;
  double rig_roll = 0.0;
  std::int64_t sync_offset = 0;
  std::optional<FieldRoi> field_roi;
  float lens_correction_amount = 1.0F;
  float blend_width = 0.05F;

  [[nodiscard]] std::string validate() const;
};

std::optional<MatchCalibration> parse_match_calibration_json(std::string_view json);
std::optional<MatchCalibration> load_match_calibration_file(const std::string& path,
                                                            std::string* error = nullptr);
std::string calibration_to_json(const MatchCalibration& calibration);

} // namespace reco::core
