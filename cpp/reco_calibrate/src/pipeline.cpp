#include "reco/calibrate/pipeline.hpp"

#include "reco/core/cuda_backend.hpp"
#include "reco/detect/npp_interop.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gstreamer.hpp"

#include <cmath>
#include <sstream>
#include <utility>

namespace reco::calibrate {
namespace {

bool finite_in_range(double value, double min, double max, bool min_exclusive = false) {
  return std::isfinite(value) && (min_exclusive ? value > min : value >= min) && value <= max;
}

BackendProbe probe_from_runtime(const reco::io::RuntimeProbe& probe) {
  return {.available = probe.available, .detail = probe.available ? probe.library : probe.error};
}

void append_step(CalibrationExecutionPlan& plan, CalibrationStep step, std::string detail) {
  plan.steps.push_back(CalibrationPlanStep{
      .step = step, .requires_gpu_residency = true, .detail = std::move(detail)});
}

std::optional<std::string> validate_decode_path(std::string_view path, std::string_view label) {
  const std::string path_text(path);
  const reco::io::GpuFileDecodeConfig config{
      .path = path_text,
      .codec = reco::io::gpu_decode_codec_for_path(path_text),
      .elementary_stream = reco::io::gpu_decode_path_is_elementary_stream(path_text),
      .container = reco::io::gpu_decode_container_for_path(path_text)};
  if (const auto error = reco::io::validate_gpu_file_decode_config(config); error.has_value()) {
    return std::string(label) + " " + *error;
  }
  return std::nullopt;
}

std::string gpu_decode_detail(const GpuCalibrationRequest& request) {
  const reco::io::GpuFileDecodeConfig left{
      .path = request.left.path,
      .codec = reco::io::gpu_decode_codec_for_path(request.left.path),
      .elementary_stream = reco::io::gpu_decode_path_is_elementary_stream(request.left.path),
      .container = reco::io::gpu_decode_container_for_path(request.left.path)};
  const reco::io::GpuFileDecodeConfig right{
      .path = request.right.path,
      .codec = reco::io::gpu_decode_codec_for_path(request.right.path),
      .elementary_stream = reco::io::gpu_decode_path_is_elementary_stream(request.right.path),
      .container = reco::io::gpu_decode_container_for_path(request.right.path)};
  return "left: " + reco::io::build_gstreamer_gpu_file_decode_pipeline(left) +
         " | right: " + reco::io::build_gstreamer_gpu_file_decode_pipeline(right);
}

} // namespace

std::optional<std::string> validate_gpu_calibration_request(const GpuCalibrationRequest& request) {
  if (request.left.path.empty()) {
    return "left video path is required";
  }
  if (request.right.path.empty()) {
    return "right video path is required";
  }
  if (request.output.empty()) {
    return "output calibration path is required";
  }
  if (const auto error = validate_decode_path(request.left.path, "left"); error.has_value()) {
    return *error;
  }
  if (const auto error = validate_decode_path(request.right.path, "right"); error.has_value()) {
    return *error;
  }
  if (const auto error = request.config.validate(); error.has_value()) {
    return *error;
  }
  if (request.config.num_frames == 0) {
    return "calibration requires at least one frame";
  }
  if (!std::isfinite(request.config.skip_start_secs) ||
      !std::isfinite(request.config.skip_end_secs) || request.config.skip_start_secs < 0.0 ||
      request.config.skip_end_secs < 0.0) {
    return "calibration skip durations must be non-negative";
  }
  if (!finite_in_range(request.config.matching.lowe_ratio, 0.0, 1.0, true)) {
    return "calibration Lowe ratio must be in (0, 1]";
  }
  if (!std::isfinite(request.config.akaze.threshold) || request.config.akaze.threshold <= 0.0) {
    return "calibration AKAZE threshold must be positive";
  }
  if (!finite_in_range(request.config.akaze.detect_y_min, 0.0, 1.0) ||
      !finite_in_range(request.config.akaze.detect_y_max, 0.0, 1.0) ||
      request.config.akaze.detect_y_min >= request.config.akaze.detect_y_max) {
    return "calibration detection Y range must be ordered within [0, 1]";
  }
  return std::nullopt;
}

CalibrationBackendStatus probe_calibration_backends() {
  CalibrationBackendStatus status;
  status.cuda.available = reco::core::CudaBackend::is_available();
  status.cuda.detail = status.cuda.available ? "CUDA driver/runtime available"
                                             : reco::core::CudaBackend::availability_error();
  status.gstreamer = probe_from_runtime(reco::io::probe_gstreamer_runtime());
  status.npp.available = reco::detect::is_npp_available();
  status.npp.detail = status.npp.available ? "NPP image primitives available"
                                           : reco::detect::npp_availability_error();
  status.nvbufsurface = probe_from_runtime(reco::io::probe_nvbufsurface_runtime());
  return status;
}

CalibrationExecutionPlan build_gpu_calibration_plan(const GpuCalibrationRequest& request,
                                                    const CalibrationBackendStatus& backends) {
  CalibrationExecutionPlan plan;
  if (const auto error = validate_gpu_calibration_request(request); error.has_value()) {
    plan.ready = false;
    plan.blocked_reason = *error;
    return plan;
  }

  append_step(plan, CalibrationStep::Probing, "probe video metadata without materializing frames");
  append_step(plan, CalibrationStep::DetectingProfiles,
              request.left.lens_profile.has_value() ? "load explicit left/right lens profiles"
                                                    : "detect lens profiles from metadata");
  append_step(plan, CalibrationStep::AudioSync,
              request.auto_sync ? "derive sync on the GPU pipeline timeline"
                                : "use explicit frame sync offset");
  append_step(plan, CalibrationStep::ExtractingFrames,
              "decode calibration samples into CUDA/NVMM surfaces via " +
                  gpu_decode_detail(request));
  append_step(plan, CalibrationStep::Undistorting, "run device-resident fisheye Y-plane undistort");
  append_step(plan, CalibrationStep::FeatureMatching,
              "detect and match AKAZE features without CPU frame readback");
  append_step(plan, CalibrationStep::Optimizing,
              "optimize stitch layout and write v1-compatible calibration JSON");

  if (!backends.cuda.available) {
    plan.blocked_reason = "CUDA is required for C++ calibration: " + backends.cuda.detail;
  } else if (!backends.gstreamer.available) {
    plan.blocked_reason =
        "GStreamer is required for C++ GPU calibration video ingest: " + backends.gstreamer.detail;
  } else if (!backends.npp.available) {
    plan.blocked_reason =
        "NPP is required for C++ GPU calibration resize/color interop: " + backends.npp.detail;
  } else {
    plan.blocked_reason =
        "C++ GPU AKAZE detection/matching execution is not ported yet; refusing CPU fallback";
  }
  plan.ready = !plan.blocked_reason.has_value();
  plan.gpu_resident = plan.ready;
  return plan;
}

std::string describe_calibration_plan(const CalibrationExecutionPlan& plan) {
  std::ostringstream out;
  out << "GPU calibration plan";
  if (plan.gpu_resident) {
    out << " (device-resident)";
  }
  out << ":\n";
  for (const auto& step : plan.steps) {
    out << "  - " << calibration_step_name(step.step);
    if (!step.detail.empty()) {
      out << ": " << step.detail;
    }
    out << '\n';
  }
  if (plan.blocked_reason.has_value()) {
    out << "blocked: " << *plan.blocked_reason << '\n';
  }
  return out.str();
}

CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request,
                                      const CalibrationBackendStatus& backends) {
  const auto plan = build_gpu_calibration_plan(request, backends);
  if (!plan.ready) {
    throw CalibrationExecutionError(plan.blocked_reason.value_or("GPU calibration is unavailable"));
  }
  throw CalibrationExecutionError(
      "C++ GPU calibration reached an unexpected ready state before execution was ported");
}

CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request) {
  return run_gpu_calibration(request, probe_calibration_backends());
}

} // namespace reco::calibrate
