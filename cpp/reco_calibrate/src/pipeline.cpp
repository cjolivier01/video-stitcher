#include "reco/calibrate/pipeline.hpp"

#include "calibration_worker_internal.hpp"
#include "gpu_video_probe_internal.hpp"
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
#include "stable_media_file.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace reco::calibrate {
namespace {

constexpr std::size_t kMaximumCalibrationRequestPathBytes = 16U * 1024U;
constexpr std::size_t kMaximumCalibrationRequestFrames = 256;
constexpr std::uint64_t kMinimumCalibrationWorkerMemoryBytes = 16ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaximumCalibrationWorkerMemoryBytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;

bool invalid_protocol_path(std::string_view value) {
  return value.size() > kMaximumCalibrationRequestPathBytes ||
         value.find('\0') != std::string_view::npos;
}

const std::string& calibration_open_path(const CalibrationVideoInput& input) {
  return input.retained_path.has_value() ? *input.retained_path : input.path;
}

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
  const auto calibrated_left_params =
      camera_params_after_applied_rotation(left_params, first.left.applied_rotation_degrees);
  const auto calibrated_right_params =
      camera_params_after_applied_rotation(right_params, first.right.applied_rotation_degrees);
  GpuCalibrationUndistorter left_undistorter(
      backend,
      {.camera = calibrated_left_params, .output_width = left_width, .output_height = left_height});
  GpuCalibrationUndistorter right_undistorter(backend, {.camera = calibrated_right_params,
                                                        .output_width = right_width,
                                                        .output_height = right_height});
  auto left_storage = backend.allocate_pitched(left_width, left_height, 16);
  auto right_storage = backend.allocate_pitched(right_width, right_height, 16);
  GpuGrayFrame left_undistorted{.ptr = left_storage.buffer.ptr(),
                                .pitch = left_storage.pitch,
                                .width = left_width,
                                .height = left_height};
  GpuGrayFrame right_undistorted{.ptr = right_storage.buffer.ptr(),
                                 .pitch = right_storage.pitch,
                                 .width = right_width,
                                 .height = right_height};
  GpuAkazePipeline akaze(backend);
  const auto left_config =
      fit_gpu_akaze_workspace(left_width, left_height, akaze_config_for_side(config, true));
  const auto right_config =
      fit_gpu_akaze_workspace(right_width, right_height, akaze_config_for_side(config, false));

  std::vector<FrameMatches> successful_frames;
  successful_frames.reserve(frame_count);
  const auto process_pair = [&](const GpuCalibrationFramePairView& frame) {
    validate_calibration_frame(frame.left, "left");
    validate_calibration_frame(frame.right, "right");
    if (frame.left.width != left_width || frame.left.height != left_height ||
        frame.right.width != right_width || frame.right.height != right_height) {
      throw std::invalid_argument("GPU calibration frame dimensions changed between pairs");
    }
    if (frame.left.applied_rotation_degrees != first.left.applied_rotation_degrees ||
        frame.right.applied_rotation_degrees != first.right.applied_rotation_degrees) {
      throw std::invalid_argument("GPU calibration frame rotation changed between pairs");
    }
    left_undistorted.color_range = frame.left.color_range;
    right_undistorted.color_range = frame.right.color_range;
    left_undistorter.undistort_y(frame.left, left_undistorted);
    right_undistorter.undistort_y(frame.right, right_undistorted);
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
      .calibration = {.left = calibrated_left_params,
                      .right = calibrated_right_params,
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
  if (const auto error = validate_gpu_calibration_probe_metadata(probe); error.has_value()) {
    throw CalibrationExecutionError(std::string(label) + " video " + *error);
  }
}

bool matching_frame_rates(const reco::io::GpuVideoProbe& left,
                          const reco::io::GpuVideoProbe& right) {
  return static_cast<std::uint64_t>(left.fps_numerator) * right.fps_denominator ==
         static_cast<std::uint64_t>(right.fps_numerator) * left.fps_denominator;
}

std::pair<reco::core::CameraParams, reco::core::CameraParams>
load_calibration_profiles(const detail::StableLensProfileFile& left_profile,
                          const detail::StableLensProfileFile* right_profile) {
  try {
    auto left = load_lens_from_file(left_profile.retained_path().string());
    auto right = right_profile != nullptr
                     ? load_lens_from_file(right_profile->retained_path().string())
                     : left;
    return {std::move(left), std::move(right)};
  } catch (const std::exception& error) {
    throw CalibrationExecutionError(std::string("failed to load calibration lens profile: ") +
                                    error.what());
  }
}

class StableCalibrationProfiles {
public:
  explicit StableCalibrationProfiles(const GpuCalibrationRequest& request) {
    if (request.left.lens_profile.has_value()) {
      left_.emplace(*request.left.lens_profile);
    }
    if (request.right.lens_profile.has_value()) {
      right_.emplace(*request.right.lens_profile);
    }
  }

  [[nodiscard]] std::pair<reco::core::CameraParams, reco::core::CameraParams> load() const {
    if (!left_.has_value()) {
      throw CalibrationExecutionError(
          "automatic lens profile detection is not ported; provide --left-profile");
    }
    return load_calibration_profiles(*left_, right_ ? &*right_ : nullptr);
  }

  void verify_unchanged() const {
    if (left_.has_value()) {
      left_->verify_unchanged();
    }
    if (right_.has_value()) {
      right_->verify_unchanged();
    }
  }

private:
  std::optional<detail::StableLensProfileFile> left_;
  std::optional<detail::StableLensProfileFile> right_;
};

std::vector<std::filesystem::path> lens_profile_paths(const GpuCalibrationRequest& request) {
  std::vector<std::filesystem::path> profiles;
  if (request.left.lens_profile.has_value()) {
    profiles.emplace_back(*request.left.lens_profile);
  }
  if (request.right.lens_profile.has_value()) {
    profiles.emplace_back(*request.right.lens_profile);
  }
  return profiles;
}

} // namespace

std::optional<std::string>
validate_gpu_calibration_probe_metadata(const reco::io::GpuVideoProbe& probe) {
  if (probe.width == 0 || probe.height == 0 || probe.width > reco::core::kMaxCalibrationDimension ||
      probe.height > reco::core::kMaxCalibrationDimension) {
    return "has unsupported probe dimensions";
  }
  if (!std::isfinite(probe.fps) || probe.fps <= 0.0 || probe.fps_numerator == 0 ||
      probe.fps_denominator == 0) {
    return "has no usable constant frame rate";
  }
  if (probe.total_frames == 0) {
    return "contains no indexed frames";
  }
  if (probe.timestamp_multiplicity == 0U ||
      probe.timestamp_multiplicity > reco::io::kMaximumIndexedTimestampMultiplicity) {
    return "has an unsupported timestamp multiplicity";
  }
  if (!probe.first_stream_time_ns.has_value()) {
    return "has no presentation stream-time origin";
  }
  if (probe.total_frames_is_estimated) {
    return "has no exact compressed-frame count";
  }
  if (!probe.selected_stream_caps_verified) {
    return "has no full-stream selected-video caps proof";
  }
  if (!probe.indexed_sampling_cadence_verified) {
    return "cadence is ambiguous for indexed sampling";
  }
  if (probe.total_frames % probe.timestamp_multiplicity != 0U) {
    return "ends within an indexed timestamp group";
  }
  return std::nullopt;
}

std::optional<std::string> validate_calibration_output_identity(
    const std::filesystem::path& left_input, const std::filesystem::path& right_input,
    const std::filesystem::path& output, std::span<const std::filesystem::path> lens_profiles) {
  const auto normalize = [](const std::filesystem::path& path, std::string_view label,
                            std::filesystem::path& normalized) -> std::optional<std::string> {
    std::error_code error;
    normalized = std::filesystem::absolute(path, error);
    if (error) {
      return "cannot resolve " + std::string(label) + " for identity checking: " + error.message();
    }
    normalized = normalized.lexically_normal();
    return std::nullopt;
  };

  std::filesystem::path normalized_left;
  std::filesystem::path normalized_right;
  std::filesystem::path normalized_output;
  std::vector<std::filesystem::path> normalized_profiles;
  if (const auto error = normalize(left_input, "left video path", normalized_left);
      error.has_value()) {
    return error;
  }
  if (const auto error = normalize(right_input, "right video path", normalized_right);
      error.has_value()) {
    return error;
  }
  if (const auto error = normalize(output, "output calibration path", normalized_output);
      error.has_value()) {
    return error;
  }
  normalized_profiles.reserve(lens_profiles.size());
  for (std::size_t index = 0; index < lens_profiles.size(); ++index) {
    std::filesystem::path normalized_profile;
    const auto label = index == 0 ? "left lens profile path" : "right lens profile path";
    if (const auto error = normalize(lens_profiles[index], label, normalized_profile);
        error.has_value()) {
      return error;
    }
    normalized_profiles.push_back(std::move(normalized_profile));
  }

  const auto alias_error = [](std::string_view label) {
    return "output calibration path identifies the " + std::string(label);
  };
  if (normalized_output == normalized_left) {
    return alias_error("left video input");
  }
  if (normalized_output == normalized_right) {
    return alias_error("right video input");
  }
  for (std::size_t index = 0; index < normalized_profiles.size(); ++index) {
    if (normalized_output == normalized_profiles[index]) {
      return alias_error(index == 0 ? "left lens profile" : "right lens profile");
    }
  }

  std::error_code error;
  const auto output_status = std::filesystem::status(normalized_output, error);
  if (error == std::errc::no_such_file_or_directory) {
    return std::nullopt;
  }
  if (error) {
    return "cannot inspect output calibration path identity: " + error.message();
  }
  if (!std::filesystem::exists(output_status)) {
    return std::nullopt;
  }

  const auto equivalent_to_input = [&](const std::filesystem::path& input,
                                       std::string_view label) -> std::optional<std::string> {
    error.clear();
    const bool equivalent = std::filesystem::equivalent(normalized_output, input, error);
    if (error) {
      return "cannot compare output calibration path with " + std::string(label) +
             " identity: " + error.message();
    }
    return equivalent ? std::optional<std::string>(alias_error(label)) : std::nullopt;
  };
  if (const auto alias = equivalent_to_input(normalized_left, "left video input");
      alias.has_value()) {
    return alias;
  }
  if (const auto alias = equivalent_to_input(normalized_right, "right video input");
      alias.has_value()) {
    return alias;
  }
  for (std::size_t index = 0; index < normalized_profiles.size(); ++index) {
    if (const auto alias = equivalent_to_input(
            normalized_profiles[index], index == 0 ? "left lens profile" : "right lens profile");
        alias.has_value()) {
      return alias;
    }
  }
  return std::nullopt;
}

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
  if (invalid_protocol_path(request.left.path) || invalid_protocol_path(request.right.path) ||
      invalid_protocol_path(request.output) ||
      (request.left.lens_profile.has_value() &&
       invalid_protocol_path(*request.left.lens_profile)) ||
      (request.right.lens_profile.has_value() &&
       invalid_protocol_path(*request.right.lens_profile)) ||
      (request.left.retained_path.has_value() &&
       invalid_protocol_path(*request.left.retained_path)) ||
      (request.right.retained_path.has_value() &&
       invalid_protocol_path(*request.right.retained_path)) ||
      (request.debug_dir.has_value() && invalid_protocol_path(*request.debug_dir))) {
    return "calibration paths must not contain NUL bytes or exceed 16384 bytes";
  }
  if (const auto error = validate_decode_path(request.left.path, "left"); error.has_value()) {
    return *error;
  }
  if (const auto error = validate_decode_path(request.right.path, "right"); error.has_value()) {
    return *error;
  }
  if ((request.left.retained_path.has_value() &&
       (request.left.retained_path->empty() ||
        !std::filesystem::path(*request.left.retained_path).is_absolute())) ||
      (request.right.retained_path.has_value() &&
       (request.right.retained_path->empty() ||
        !std::filesystem::path(*request.right.retained_path).is_absolute()))) {
    return "retained calibration video paths must be non-empty and absolute";
  }
  const auto profiles = lens_profile_paths(request);
  if (const auto error = validate_calibration_output_identity(calibration_open_path(request.left),
                                                              calibration_open_path(request.right),
                                                              request.output, profiles);
      error.has_value()) {
    return *error;
  }
  if (const auto error = request.config.validate(); error.has_value()) {
    return *error;
  }
  if (request.config.num_frames == 0) {
    return "calibration requires at least one frame";
  }
  if (request.config.num_frames > kMaximumCalibrationRequestFrames) {
    return "calibration frame count exceeds the isolated worker limit of 256";
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
  if (invalid_protocol_path(request.probe_worker)) {
    return "GPU video probe worker path must not contain NUL bytes or exceed 16384 bytes";
  }
  if (!request.calibration_worker_path.empty() &&
      !std::filesystem::path(request.calibration_worker_path).is_absolute()) {
    return "GPU calibration worker path must be absolute";
  }
  if (invalid_protocol_path(request.calibration_worker_path)) {
    return "GPU calibration worker path must not contain NUL bytes or exceed 16384 bytes";
  }
  constexpr std::uint64_t kSecondNs = 1'000'000'000ULL;
  if (request.probe_timeout_ns < kSecondNs || request.probe_timeout_ns > 3600ULL * kSecondNs) {
    return "GPU video probe timeout must be between one second and one hour";
  }
  if (request.calibration_timeout_ns < kSecondNs ||
      request.calibration_timeout_ns > 24ULL * 3600ULL * kSecondNs) {
    return "GPU calibration timeout must be between one second and 24 hours";
  }
  if (request.calibration_host_memory_limit_bytes < kMinimumCalibrationWorkerMemoryBytes ||
      request.calibration_host_memory_limit_bytes > kMaximumCalibrationWorkerMemoryBytes) {
    return "GPU calibration host memory limit must be between 16 MiB and 64 GiB";
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
  } else if (!backends.nvbufsurface.available) {
    plan.blocked_reason = "NvBufSurface is required for zero-copy C++ GPU calibration ingest: " +
                          backends.nvbufsurface.detail;
  } else if (request.calibration_worker_path.empty()) {
    plan.blocked_reason = "GPU calibration worker path is required for calibration";
  } else if (!request.left.lens_profile.has_value()) {
    plan.blocked_reason = "automatic lens profile detection is not ported; provide --left-profile";
  } else if (!request.no_auto_imu) {
    plan.blocked_reason = "automatic IMU extraction is not ported; explicitly use --no-auto-imu";
  } else if (request.auto_sync) {
    plan.blocked_reason =
        "automatic audio sync extraction is not ported; use --no-auto-sync and --sync-offset";
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

CalibrationResult detail::run_gpu_calibration_in_process(const GpuCalibrationRequest& request,
                                                         const CalibrationBackendStatus& backends) {
  const auto runtime_abi = reco::io::discover_nvbufsurface_abi();
  if (runtime_abi != request.nvbufsurface_abi) {
    throw CalibrationExecutionError(
        "calibration worker NvBufSurface ABI does not match the supervisor runtime");
  }
  const auto plan = build_gpu_calibration_plan(request, backends);
  if (!plan.ready) {
    throw CalibrationExecutionError(plan.blocked_reason.value_or("GPU calibration is unavailable"));
  }

  try {
    auto left_config = calibration_decode_config(request.left.path);
    auto right_config = calibration_decode_config(request.right.path);
    StableMediaFile left_media(calibration_open_path(request.left));
    StableMediaFile right_media(calibration_open_path(request.right));
    StableCalibrationProfiles profiles(request);
    left_config.path = left_media.decode_path().string();
    right_config.path = right_media.decode_path().string();
    const auto left_probe = reco::io::detail::probe_gpu_video_in_process(
        left_config, request.probe_timeout_ns,
        reco::io::detail::GpuVideoProbePolicy::ExhaustiveIndexedCadence);
    const auto right_probe = reco::io::detail::probe_gpu_video_in_process(
        right_config, request.probe_timeout_ns,
        reco::io::detail::GpuVideoProbePolicy::ExhaustiveIndexedCadence);
    left_media.verify_unchanged();
    right_media.verify_unchanged();
    validate_probe(left_probe, "left");
    validate_probe(right_probe, "right");
    if (!matching_frame_rates(left_probe, right_probe)) {
      throw CalibrationExecutionError(
          "left and right calibration videos must have the same constant frame rate");
    }
    left_config.read_timeout_ns = request.probe_timeout_ns;
    right_config.read_timeout_ns = request.probe_timeout_ns;
    left_config.indexed_fps_numerator = left_probe.fps_numerator;
    left_config.indexed_fps_denominator = left_probe.fps_denominator;
    left_config.indexed_timestamp_multiplicity = left_probe.timestamp_multiplicity;
    left_config.indexed_stream_time_origin_ns = left_probe.first_stream_time_ns;
    right_config.indexed_fps_numerator = right_probe.fps_numerator;
    right_config.indexed_fps_denominator = right_probe.fps_denominator;
    right_config.indexed_timestamp_multiplicity = right_probe.timestamp_multiplicity;
    right_config.indexed_stream_time_origin_ns = right_probe.first_stream_time_ns;

    const auto [left_indices, right_indices] = select_synchronized_frame_indices(
        left_probe.total_frames, right_probe.total_frames, left_probe.fps,
        request.config.num_frames, request.config.skip_start_secs, request.config.skip_end_secs,
        request.manual_sync_offset);
    if (left_indices.empty() || left_indices.size() != right_indices.size()) {
      throw CalibrationExecutionError("calibration videos have no synchronized sample range");
    }
    auto [left_params, right_params] = profiles.load();
    auto backend = reco::core::CudaBackend::create();
    left_config.start_frame_index = left_indices.front();
    right_config.start_frame_index = right_indices.front();
    auto left_source = reco::io::open_gstreamer_gpu_file_decode_source(std::move(left_config),
                                                                       request.nvbufsurface_abi);
    auto right_source = reco::io::open_gstreamer_gpu_file_decode_source(std::move(right_config),
                                                                        request.nvbufsurface_abi);
    auto result = run_gpu_calibration_sources(backend, *left_source, *right_source, left_indices,
                                              right_indices, left_params, right_params,
                                              request.config, request.manual_sync_offset);
    left_media.verify_unchanged();
    right_media.verify_unchanged();
    profiles.verify_unchanged();
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

CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request,
                                      const CalibrationBackendStatus& backends) {
  const auto plan = build_gpu_calibration_plan(request, backends);
  if (!plan.ready) {
    throw CalibrationExecutionError(plan.blocked_reason.value_or("GPU calibration is unavailable"));
  }
  return detail::run_gpu_calibration_supervised(request);
}

CalibrationResult run_gpu_calibration(const GpuCalibrationRequest& request) {
  if (const auto error = validate_gpu_calibration_request(request); error.has_value()) {
    throw CalibrationExecutionError(*error);
  }
  if (request.calibration_worker_path.empty()) {
    throw CalibrationExecutionError("GPU calibration worker path is required for file execution");
  }
  return detail::run_gpu_calibration_supervised(request);
}

} // namespace reco::calibrate
