#pragma once

#include "reco/calibrate/gpu_undistort.hpp"
#include "reco/calibrate/types.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/nvmm.hpp"

#include <cstdint>
#include <optional>
#include <span>
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
  std::string probe_worker;
  std::uint64_t probe_timeout_ns = 30'000'000'000ULL;
  reco::io::NvbufSurfaceAbi nvbufsurface_abi = reco::io::NvbufSurfaceAbi::DeepStream7_1;
};

/// Non-owning pair of CUDA-resident luma frames used by calibration.
struct GpuCalibrationFramePairView {
  GpuGrayFrame left;
  GpuGrayFrame right;
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
  bool gpu_resident = false;
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
/// Runs undistortion, AKAZE, filtering, and optimization on device-resident frame pairs.
/// Only compact accepted correspondences and scalar counts cross to host memory.
[[nodiscard]] CalibrationResult run_gpu_calibration_frames(
    reco::core::CudaBackend& backend, std::span<const GpuCalibrationFramePairView> frames,
    const reco::core::CameraParams& left_params, const reco::core::CameraParams& right_params,
    const CalibrationConfig& config, std::int64_t sync_offset = 0);
/// Sequentially consumes selected frames from two GPU decode sources.
/// At most one copied CUDA luma frame per source is retained at a time.
[[nodiscard]] CalibrationResult run_gpu_calibration_sources(
    reco::core::CudaBackend& backend, reco::io::GpuFileDecodeSource& left_source,
    reco::io::GpuFileDecodeSource& right_source, std::span<const std::uint64_t> left_indices,
    std::span<const std::uint64_t> right_indices, const reco::core::CameraParams& left_params,
    const reco::core::CameraParams& right_params, const CalibrationConfig& config,
    std::int64_t sync_offset = 0);
[[nodiscard]] CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request,
                                                    const CalibrationBackendStatus& backends);
[[nodiscard]] CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request);

} // namespace reco::calibrate
