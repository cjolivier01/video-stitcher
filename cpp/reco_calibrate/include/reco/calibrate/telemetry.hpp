#pragma once

#include "reco/core/calibration.hpp"

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace reco::calibrate {

struct ImuSample {
  double t = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  [[nodiscard]] double magnitude() const;
};

struct TelemetryData {
  std::string camera_type;
  std::optional<std::string> camera_model;
  std::vector<ImuSample> gyro;
  std::vector<ImuSample> accel;
  std::optional<reco::core::CameraParams> lens_profile;
  std::vector<std::pair<double, std::array<double, 4>>> quaternions;
  std::optional<std::string> lens_info;
};

[[nodiscard]] std::optional<double> estimate_sync_offset(const TelemetryData& left,
                                                         const TelemetryData& right);
[[nodiscard]] std::optional<std::array<double, 3>> gravity_vector(const TelemetryData& data,
                                                                  double skip_secs);
[[nodiscard]] std::optional<std::array<double, 3>>
differential_orientation(const TelemetryData& left, const TelemetryData& right, double skip_secs);
[[nodiscard]] std::optional<double> rig_tilt(const TelemetryData& data, double skip_secs);

[[nodiscard]] std::vector<double> normalize_signal(const std::vector<double>& signal);
[[nodiscard]] std::vector<double> resample_signal(const std::vector<std::pair<double, double>>& signal,
                                                  double t_start, double rate, std::size_t n);

} // namespace reco::calibrate
