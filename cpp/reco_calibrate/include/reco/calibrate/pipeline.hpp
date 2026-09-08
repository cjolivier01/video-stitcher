#pragma once

#include "reco/calibrate/gpu_undistort.hpp"
#include "reco/calibrate/types.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/nvmm.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace reco::io {
struct GpuVideoProbe;
}

namespace reco::calibrate {

struct CalibrationVideoInput {
  std::string path;
  std::optional<std::string> lens_profile;
  /// Stable descriptor-backed path used for opening and identity checks.
  /// The original path remains the explicit codec/container format hint.
  std::optional<std::string> retained_path;
  /// Metadata captured when the media descriptor was first pinned.
  std::optional<CalibrationFileIdentity> expected_identity;
  /// Metadata captured when the selected lens profile was first pinned.
  std::optional<CalibrationFileIdentity> lens_profile_expected_identity;
};

struct GpuCalibrationRequest {
  CalibrationVideoInput left;
  CalibrationVideoInput right;
  CalibrationConfig config;
  // Must be explicitly enabled when automatic IMU extraction is unavailable.
  bool no_auto_imu = false;
  bool auto_sync = true;
  std::int64_t manual_sync_offset = 0;
  std::optional<std::string> debug_dir;
  std::string output = "match.json";
  /// Legacy deployed probe-worker path retained for request-protocol compatibility.
  /// File calibration probes inside the already isolated calibration worker.
  std::string probe_worker;
  std::uint64_t probe_timeout_ns = 30'000'000'000ULL;
  /// Absolute path to the deployed isolated calibration worker executable.
  std::string calibration_worker_path;
  /// End-to-end budget for worker startup, calibration, IPC, and teardown.
  std::uint64_t calibration_timeout_ns = 600'000'000'000ULL;
  /// Maximum resident host memory admitted for the worker; process descendants are denied.
  std::uint64_t calibration_host_memory_limit_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
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
/// Validates probe metadata for indexed calibration. Rejects estimated frame counts and
/// requires complete selected-video caps and indexed-cadence proof through end of stream.
[[nodiscard]] std::optional<std::string>
validate_gpu_calibration_probe_metadata(const reco::io::GpuVideoProbe& probe);
/// Rejects an output path that lexically or physically identifies a video input or selected lens
/// profile.
[[nodiscard]] std::optional<std::string> validate_calibration_output_identity(
    const std::filesystem::path& left_input, const std::filesystem::path& right_input,
    const std::filesystem::path& output, std::span<const std::filesystem::path> lens_profiles = {});
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
/// Runs file-backed calibration in a bounded worker process. The worker path must be absolute.
[[nodiscard]] CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request,
                                                    const CalibrationBackendStatus& backends);
[[nodiscard]] CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request);

} // namespace reco::calibrate
