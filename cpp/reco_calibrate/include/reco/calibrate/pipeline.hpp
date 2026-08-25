#pragma once

#include "reco/calibrate/types.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace reco::calibrate {

struct CalibrationVideoInput {
  std::string path;
  std::optional<std::string> lens_profile;
};

struct GpuCalibrationRequest {
  CalibrationVideoInput left;
  CalibrationVideoInput right;
  CalibrationConfig config;
  bool no_auto_imu = false;
  bool auto_sync = true;
  std::int64_t manual_sync_offset = 0;
  std::optional<std::string> debug_dir;
  std::string output = "match.json";
};

struct BackendProbe {
  bool available = false;
  std::string detail;
};

struct CalibrationBackendStatus {
  BackendProbe cuda;
  BackendProbe gstreamer;
  BackendProbe npp;
  BackendProbe nvbufsurface;
};

struct CalibrationPlanStep {
  CalibrationStep step = CalibrationStep::Probing;
  bool requires_gpu_residency = true;
  std::string detail;
};

struct CalibrationExecutionPlan {
  std::vector<CalibrationPlanStep> steps;
  bool gpu_resident = true;
  bool ready = false;
  std::optional<std::string> blocked_reason;
};

class CalibrationExecutionError : public std::runtime_error {
public:
  explicit CalibrationExecutionError(const std::string& message) : std::runtime_error(message) {}
};

[[nodiscard]] std::optional<std::string>
validate_gpu_calibration_request(const GpuCalibrationRequest& request);
[[nodiscard]] CalibrationBackendStatus probe_calibration_backends();
[[nodiscard]] CalibrationExecutionPlan
build_gpu_calibration_plan(const GpuCalibrationRequest& request,
                           const CalibrationBackendStatus& backends);
[[nodiscard]] std::string describe_calibration_plan(const CalibrationExecutionPlan& plan);
[[nodiscard]] CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request,
                                                    const CalibrationBackendStatus& backends);
[[nodiscard]] CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request);

} // namespace reco::calibrate
