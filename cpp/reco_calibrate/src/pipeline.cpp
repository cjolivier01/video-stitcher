#include "reco/calibrate/pipeline.hpp"

#include "reco/calibrate/geometry.hpp"
#include "reco/calibrate/gpu_features.hpp"
#include "reco/calibrate/lens_database.hpp"
#include "reco/calibrate/optimizer.hpp"
#include "reco/calibrate/ransac.hpp"
#include "reco/calibrate/sampling.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/detect/npp_interop.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gpu_video_probe.hpp"
#include "reco/io/gstreamer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

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

reco::io::GpuFileDecodeConfig calibration_decode_config(const std::string& path) {
  return {.path = path,
          .codec = reco::io::gpu_decode_codec_for_path(path),
          .elementary_stream = reco::io::gpu_decode_path_is_elementary_stream(path),
          .container = reco::io::gpu_decode_container_for_path(path),
          .max_buffers = 2,
          .drop = false};
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

void validate_calibration_frame(const GpuGrayFrame& frame, std::string_view label) {
  if (frame.ptr == 0) {
    throw std::invalid_argument(std::string(label) + " calibration frame pointer is null");
  }
  if (frame.width == 0 || frame.height == 0) {
    throw std::invalid_argument(std::string(label) + " calibration frame dimensions are zero");
  }
  if (frame.width > reco::core::kMaxCalibrationDimension ||
      frame.height > reco::core::kMaxCalibrationDimension) {
    throw std::invalid_argument(std::string(label) +
                                " calibration frame dimensions exceed the supported limit");
  }
  if (frame.pitch < frame.width) {
    throw std::invalid_argument(std::string(label) +
                                " calibration frame pitch is smaller than its width");
  }
  switch (frame.color_range) {
  case reco::core::YuvColorRange::Limited:
  case reco::core::YuvColorRange::Full:
    return;
  }
  throw std::invalid_argument(std::string(label) + " calibration frame color range is invalid");
}

bool spatial_match_is_accepted(const GpuMatchedPoint& match, std::uint32_t left_width,
                               std::uint32_t left_height, std::uint32_t right_width,
                               std::uint32_t right_height, const CalibrationConfig& config) {
  const auto& matching = config.matching;
  const double left_x = match.left.x;
  const double left_y = match.left.y;
  const double right_x = match.right.x;
  const double right_y = match.right.y;
  const double average_height =
      (static_cast<double>(left_height) + static_cast<double>(right_height)) * 0.5;
  return left_x >= matching.spatial_x_threshold * left_width &&
         left_x <= (1.0 - matching.spatial_x_inner) * left_width &&
         right_x >= matching.spatial_x_inner * right_width &&
         right_x <= matching.spatial_x_threshold * right_width &&
         left_y >= matching.spatial_y_low * left_height &&
         left_y <= matching.spatial_y_high * left_height &&
         right_y >= matching.spatial_y_low * right_height &&
         right_y <= matching.spatial_y_high * right_height &&
         std::abs(left_y - right_y) <= matching.max_y_disparity * average_height;
}

GpuAkazeConfig akaze_config_for_side(const CalibrationConfig& config, bool left) {
  if (config.akaze.max_keypoints > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("calibration maximum keypoint count exceeds the GPU ABI");
  }
  const auto threshold = static_cast<float>(config.akaze.threshold);
  if (!std::isfinite(threshold) || !(threshold > 0.0F)) {
    throw std::invalid_argument("calibration AKAZE threshold is outside the GPU float range");
  }
  const auto inner = static_cast<float>(config.matching.spatial_x_inner);
  const auto overlap = static_cast<float>(config.matching.spatial_x_threshold);
  GpuAkazeConfig result;
  result.max_keypoints = static_cast<std::uint32_t>(config.akaze.max_keypoints);
  result.threshold = threshold;
  result.lowe_ratio = config.matching.lowe_ratio;
  result.region = {.x_min = left ? overlap : inner,
                   .x_max = left ? 1.0F - inner : 1.0F - overlap,
                   .y_min = static_cast<float>(config.akaze.detect_y_min),
                   .y_max = static_cast<float>(config.akaze.detect_y_max)};
  result.use_region = true;
  return result;
}

std::pair<std::uint32_t, std::uint32_t> calibration_working_dimensions(std::uint32_t width,
                                                                       std::uint32_t height) {
  constexpr std::uint32_t kTargetWidth = 1280;
  const auto factor = std::max<std::uint32_t>(width / kTargetWidth, 1U);
  return {width / factor, height / factor};
}

std::optional<FrameMatches>
process_gpu_frame_pair(GpuAkazePipeline& akaze, const GpuGrayFrame& left, const GpuGrayFrame& right,
                       const GpuAkazeConfig& left_config, const GpuAkazeConfig& right_config,
                       const CalibrationConfig& config) {
  auto left_features = akaze.detect(left, left_config);
  auto right_features = akaze.detect(right, right_config);
  const auto left_count = akaze.feature_count(left_features.view());
  const auto right_count = akaze.feature_count(right_features.view());
  if (left_count == 0 || right_count == 0) {
    return std::nullopt;
  }

  const auto raw_matches =
      akaze.match(left_features.view(), right_features.view(), config.matching.lowe_ratio);
  if (raw_matches.size() < config.matching.min_matches) {
    return std::nullopt;
  }
  std::vector<GpuMatchedPoint> spatial_matches;
  spatial_matches.reserve(raw_matches.size());
  for (const auto& match : raw_matches) {
    if (spatial_match_is_accepted(match, left.width, left.height, right.width, right.height,
                                  config)) {
      spatial_matches.push_back(match);
    }
  }
  if (spatial_matches.size() < config.matching.min_matches) {
    return std::nullopt;
  }

  std::vector<Point2d> left_points;
  std::vector<Point2d> right_points;
  left_points.reserve(spatial_matches.size());
  right_points.reserve(spatial_matches.size());
  for (const auto& match : spatial_matches) {
    left_points.push_back({static_cast<double>(match.left.x), static_cast<double>(match.left.y)});
    right_points.push_back(
        {static_cast<double>(match.right.x), static_cast<double>(match.right.y)});
  }

  std::vector<std::size_t> inliers;
  try {
    inliers = ransac_fundamental(left_points, right_points, config.matching.ransac_threshold, 2000);
  } catch (const std::exception&) {
    return std::nullopt;
  }
  if (inliers.size() < config.matching.min_matches) {
    return std::nullopt;
  }

  FrameMatches result;
  result.keypoints_left = left_count;
  result.keypoints_right = right_count;
  result.min_descriptors = std::min(left_count, right_count);
  result.post_ratio_test = raw_matches.size();
  result.post_spatial_filter = spatial_matches.size();
  result.post_ransac = inliers.size();
  result.points.reserve(inliers.size());
  for (const auto index : inliers) {
    if (index >= spatial_matches.size()) {
      throw std::runtime_error("calibration RANSAC returned an out-of-range inlier index");
    }
    const auto& match = spatial_matches[index];
    result.points.push_back(
        {.left = normalize_to_plane(match.right.x, match.right.y, right.width, right.height),
         .right = normalize_to_plane(match.left.x, match.left.y, left.width, left.height),
         .left_pixel_nx = static_cast<double>(match.right.x) / right.width,
         .right_pixel_nx = static_cast<double>(match.left.x) / left.width});
  }
  return result;
}

CalibrationResult run_gpu_calibration_frame_provider(
    reco::core::CudaBackend& backend, std::size_t frame_count,
    const std::function<GpuCalibrationFramePairView(std::size_t)>& frame_provider,
    const reco::core::CameraParams& left_params, const reco::core::CameraParams& right_params,
    const CalibrationConfig& config, std::int64_t sync_offset) {
  if (const auto error = config.validate(); error.has_value()) {
    throw std::invalid_argument("invalid GPU calibration config: " + *error);
  }
  if (sync_offset < -reco::core::kMaxSyncOffsetFrames ||
      sync_offset > reco::core::kMaxSyncOffsetFrames) {
    throw std::invalid_argument("GPU calibration sync offset exceeds the supported limit");
  }
  if (frame_count == 0) {
    throw CalibrationExecutionError("GPU calibration has no frame pairs");
  }

  const auto first = frame_provider(0);
  validate_calibration_frame(first.left, "left");
  validate_calibration_frame(first.right, "right");
  const auto left_width = first.left.width;
  const auto left_height = first.left.height;
  const auto right_width = first.right.width;
  const auto right_height = first.right.height;
  const auto [left_work_width, left_work_height] =
      calibration_working_dimensions(left_width, left_height);
  const auto [right_work_width, right_work_height] =
      calibration_working_dimensions(right_width, right_height);

  GpuCalibrationUndistorter left_undistorter(
      backend,
      {.camera = left_params, .output_width = left_work_width, .output_height = left_work_height});
  GpuCalibrationUndistorter right_undistorter(backend, {.camera = right_params,
                                                        .output_width = right_work_width,
                                                        .output_height = right_work_height});
  std::optional<reco::core::CudaPitchedAllocation> left_resized_storage;
  std::optional<reco::core::CudaPitchedAllocation> right_resized_storage;
  if (left_work_width != left_width || left_work_height != left_height) {
    left_resized_storage = backend.allocate_pitched(left_work_width, left_work_height, 16);
  }
  if (right_work_width != right_width || right_work_height != right_height) {
    right_resized_storage = backend.allocate_pitched(right_work_width, right_work_height, 16);
  }
  auto left_storage = backend.allocate_pitched(left_work_width, left_work_height, 16);
  auto right_storage = backend.allocate_pitched(right_work_width, right_work_height, 16);
  GpuGrayFrame left_undistorted{.ptr = left_storage.buffer.ptr(),
                                .pitch = left_storage.pitch,
                                .width = left_work_width,
                                .height = left_work_height};
  GpuGrayFrame right_undistorted{.ptr = right_storage.buffer.ptr(),
                                 .pitch = right_storage.pitch,
                                 .width = right_work_width,
                                 .height = right_work_height};
  GpuAkazePipeline akaze(backend);
  const auto left_config = akaze_config_for_side(config, true);
  const auto right_config = akaze_config_for_side(config, false);

  std::vector<FrameMatches> successful_frames;
  successful_frames.reserve(frame_count);
  const auto resize_for_features =
      [&backend](const GpuGrayFrame& frame,
                 const std::optional<reco::core::CudaPitchedAllocation>& resized_storage,
                 std::uint32_t work_width, std::uint32_t work_height) {
        if (!resized_storage.has_value()) {
          return frame;
        }
        reco::detect::npp_resize_c1(frame.ptr, frame.pitch, frame.width, frame.height,
                                    resized_storage->buffer.ptr(), resized_storage->pitch,
                                    work_width, work_height);
        backend.synchronize();
        return GpuGrayFrame{.ptr = resized_storage->buffer.ptr(),
                            .pitch = resized_storage->pitch,
                            .width = work_width,
                            .height = work_height,
                            .color_range = frame.color_range};
      };
  const auto process_pair = [&](const GpuCalibrationFramePairView& frame) {
    validate_calibration_frame(frame.left, "left");
    validate_calibration_frame(frame.right, "right");
    if (frame.left.width != left_width || frame.left.height != left_height ||
        frame.right.width != right_width || frame.right.height != right_height) {
      throw std::invalid_argument("GPU calibration frame dimensions changed between pairs");
    }
    left_undistorted.color_range = frame.left.color_range;
    right_undistorted.color_range = frame.right.color_range;
    const auto left_feature_frame =
        resize_for_features(frame.left, left_resized_storage, left_work_width, left_work_height);
    const auto right_feature_frame = resize_for_features(frame.right, right_resized_storage,
                                                         right_work_width, right_work_height);
    left_undistorter.undistort_y(left_feature_frame, left_undistorted);
    right_undistorter.undistort_y(right_feature_frame, right_undistorted);
    if (auto result = process_gpu_frame_pair(akaze, left_undistorted, right_undistorted,
                                             left_config, right_config, config);
        result.has_value()) {
      successful_frames.push_back(std::move(*result));
    }
  };
  process_pair(first);
  for (std::size_t index = 1; index < frame_count; ++index) {
    process_pair(frame_provider(index));
  }
  if (successful_frames.empty()) {
    throw CalibrationExecutionError("GPU calibration found no usable frame pairs");
  }

  std::vector<MatchedPoint> all_points;
  for (const auto& frame : successful_frames) {
    all_points.insert(all_points.end(), frame.points.begin(), frame.points.end());
  }
  if (all_points.size() < config.matching.min_matches) {
    throw CalibrationExecutionError("GPU calibration has only " +
                                    std::to_string(all_points.size()) +
                                    " inlier matches; at least " +
                                    std::to_string(config.matching.min_matches) + " are required");
  }

  const auto optimized = optimize(all_points, config);
  const OptParams best_params{.x_ty = optimized.layout.x_ty,
                              .intersect = optimized.layout.intersect,
                              .cam_d = optimized.layout.camera_axis_offset,
                              .x_rz = optimized.layout.x_rz,
                              .z_rx = optimized.layout.z_rx};
  const auto confidence = std::min(static_cast<double>(all_points.size()) / 50.0, 1.0);
  return {
      .calibration = {.left = left_params,
                      .right = right_params,
                      .layout = optimized.layout,
                      .rig_tilt = 0.0,
                      .rig_roll = 0.0,
                      .sync_offset = sync_offset,
                      .field_roi = std::nullopt,
                      .lens_correction_amount = 1.0F,
                      .blend_width = 0.05F},
      .total_matches = all_points.size(),
      .frames_used = successful_frames.size(),
      .residual_error = optimized.residual_error,
      .confidence = confidence,
      .per_frame = std::move(successful_frames),
      .quality = CalibrationQuality{
          .mean_reprojection_error = reprojection_error(all_points, best_params),
          .trimmed_reprojection_error = trimmed_reprojection_error(all_points, best_params, 0.2),
          .angular_error = angular_error(all_points, best_params)}};
}

void validate_probe(const reco::io::GpuVideoProbe& probe, std::string_view label) {
  if (probe.width == 0 || probe.height == 0 || probe.width > reco::core::kMaxCalibrationDimension ||
      probe.height > reco::core::kMaxCalibrationDimension) {
    throw CalibrationExecutionError(std::string(label) + " video has unsupported probe dimensions");
  }
  if (!std::isfinite(probe.fps) || probe.fps <= 0.0 || probe.fps_numerator == 0 ||
      probe.fps_denominator == 0) {
    throw CalibrationExecutionError(std::string(label) +
                                    " video has no usable constant frame rate");
  }
  if (probe.total_frames == 0) {
    throw CalibrationExecutionError(std::string(label) + " video contains no indexed frames");
  }
  if (!probe.indexed_sampling_cadence_verified) {
    throw CalibrationExecutionError(std::string(label) +
                                    " video cadence is not verified for indexed sampling");
  }
}

bool matching_frame_rates(const reco::io::GpuVideoProbe& left,
                          const reco::io::GpuVideoProbe& right) {
  return static_cast<std::uint64_t>(left.fps_numerator) * right.fps_denominator ==
         static_cast<std::uint64_t>(right.fps_numerator) * left.fps_denominator;
}

std::pair<reco::core::CameraParams, reco::core::CameraParams>
load_calibration_profiles(const GpuCalibrationRequest& request) {
  if (!request.left.lens_profile.has_value()) {
    throw CalibrationExecutionError(
        "automatic lens profile detection is not ported; provide --left-profile");
  }
  try {
    auto left = load_lens_from_file(*request.left.lens_profile);
    auto right = request.right.lens_profile.has_value()
                     ? load_lens_from_file(*request.right.lens_profile)
                     : left;
    return {std::move(left), std::move(right)};
  } catch (const std::exception& error) {
    throw CalibrationExecutionError(std::string("failed to load calibration lens profile: ") +
                                    error.what());
  }
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
  if (request.manual_sync_offset < -reco::core::kMaxSyncOffsetFrames ||
      request.manual_sync_offset > reco::core::kMaxSyncOffsetFrames) {
    return "calibration sync offset exceeds the supported limit";
  }
  if (request.right.lens_profile.has_value() && !request.left.lens_profile.has_value()) {
    return "right lens profile requires a left lens profile";
  }
  if (!request.probe_worker.empty() && !std::filesystem::path(request.probe_worker).is_absolute()) {
    return "GPU video probe worker path must be absolute";
  }
  constexpr std::uint64_t kSecondNs = 1'000'000'000ULL;
  if (request.probe_timeout_ns < kSecondNs || request.probe_timeout_ns > 3600ULL * kSecondNs) {
    return "GPU video probe timeout must be between one second and one hour";
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
  } else if (!backends.nvbufsurface.available) {
    plan.blocked_reason = "NvBufSurface is required for zero-copy C++ GPU calibration ingest: " +
                          backends.nvbufsurface.detail;
  } else if (request.probe_worker.empty()) {
    plan.blocked_reason = "GPU video probe worker path is required for calibration";
  } else if (!request.left.lens_profile.has_value()) {
    plan.blocked_reason = "automatic lens profile detection is not ported; provide --left-profile";
  } else if (request.auto_sync) {
    plan.blocked_reason =
        "automatic IMU/audio sync extraction is not ported; use --no-auto-sync and --sync-offset";
  } else if (request.debug_dir.has_value()) {
    plan.blocked_reason = "GPU calibration debug image export is not ported; omit --debug-dir";
  } else {
    plan.blocked_reason.reset();
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

CalibrationResult run_gpu_calibration_frames(reco::core::CudaBackend& backend,
                                             std::span<const GpuCalibrationFramePairView> frames,
                                             const reco::core::CameraParams& left_params,
                                             const reco::core::CameraParams& right_params,
                                             const CalibrationConfig& config,
                                             std::int64_t sync_offset) {
  return run_gpu_calibration_frame_provider(
      backend, frames.size(), [&frames](std::size_t index) { return frames[index]; }, left_params,
      right_params, config, sync_offset);
}

CalibrationResult run_gpu_calibration_sources(
    reco::core::CudaBackend& backend, reco::io::GpuFileDecodeSource& left_source,
    reco::io::GpuFileDecodeSource& right_source, std::span<const std::uint64_t> left_indices,
    std::span<const std::uint64_t> right_indices, const reco::core::CameraParams& left_params,
    const reco::core::CameraParams& right_params, const CalibrationConfig& config,
    std::int64_t sync_offset) {
  if (left_indices.empty() || left_indices.size() != right_indices.size()) {
    throw std::invalid_argument(
        "GPU calibration source indices must be non-empty and have equal length");
  }
  GpuCalibrationFrameReader left_reader(backend, left_source);
  GpuCalibrationFrameReader right_reader(backend, right_source);
  std::optional<GpuCalibrationFrame> left_frame;
  std::optional<GpuCalibrationFrame> right_frame;
  return run_gpu_calibration_frame_provider(
      backend, left_indices.size(),
      [&](std::size_t index) {
        left_frame = left_reader.read(left_indices[index]);
        right_frame = right_reader.read(right_indices[index]);
        return GpuCalibrationFramePairView{.left = left_frame->view(),
                                           .right = right_frame->view()};
      },
      left_params, right_params, config, sync_offset);
}

CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request,
                                      const CalibrationBackendStatus& backends) {
  const auto plan = build_gpu_calibration_plan(request, backends);
  if (!plan.ready) {
    throw CalibrationExecutionError(plan.blocked_reason.value_or("GPU calibration is unavailable"));
  }

  try {
    const auto left_config = calibration_decode_config(request.left.path);
    const auto right_config = calibration_decode_config(request.right.path);
    const auto worker = std::filesystem::path(request.probe_worker);
    const auto left_probe =
        reco::io::probe_gpu_video(left_config, worker, request.probe_timeout_ns);
    const auto right_probe =
        reco::io::probe_gpu_video(right_config, worker, request.probe_timeout_ns);
    validate_probe(left_probe, "left");
    validate_probe(right_probe, "right");
    if (!matching_frame_rates(left_probe, right_probe)) {
      throw CalibrationExecutionError(
          "left and right calibration videos must have the same constant frame rate");
    }

    const auto [left_indices, right_indices] = select_synchronized_frame_indices(
        left_probe.total_frames, right_probe.total_frames, left_probe.fps,
        request.config.num_frames, request.config.skip_start_secs, request.config.skip_end_secs,
        request.manual_sync_offset);
    if (left_indices.empty() || left_indices.size() != right_indices.size()) {
      throw CalibrationExecutionError("calibration videos have no synchronized sample range");
    }

    auto [left_params, right_params] = load_calibration_profiles(request);
    auto backend = reco::core::CudaBackend::create();
    auto left_source =
        reco::io::open_gstreamer_gpu_file_decode_source(left_config, request.nvbufsurface_abi);
    auto right_source =
        reco::io::open_gstreamer_gpu_file_decode_source(right_config, request.nvbufsurface_abi);
    auto result = run_gpu_calibration_sources(backend, *left_source, *right_source, left_indices,
                                              right_indices, left_params, right_params,
                                              request.config, request.manual_sync_offset);
    result.left_lens_profile = LensProfileInfo{
        .camera = "", .lens = "", .source = ProfileSource::File, .path = request.left.lens_profile};
    result.right_lens_profile =
        LensProfileInfo{.camera = "",
                        .lens = "",
                        .source = request.right.lens_profile.has_value() ? ProfileSource::File
                                                                         : ProfileSource::Fallback,
                        .path = request.right.lens_profile.has_value() ? request.right.lens_profile
                                                                       : request.left.lens_profile};
    return result;
  } catch (const CalibrationExecutionError&) {
    throw;
  } catch (const std::exception& error) {
    throw CalibrationExecutionError(std::string("GPU calibration failed: ") + error.what());
  }
}

CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request) {
  return run_gpu_calibration(request, probe_calibration_backends());
}

} // namespace reco::calibrate
