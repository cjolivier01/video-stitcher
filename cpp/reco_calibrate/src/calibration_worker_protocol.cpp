#include "calibration_worker_protocol.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace reco::calibrate::detail {
namespace {

constexpr std::array<char, 4> kMagic = {'R', 'C', 'A', 'L'};
constexpr std::size_t kMaximumProfileTextBytes = 4U * 1024U;

class Writer {
public:
  void u8(std::uint8_t value) {
    require_capacity(1);
    bytes_.push_back(static_cast<char>(value));
  }

  void boolean(bool value) { u8(value ? 1U : 0U); }

  void u16(std::uint16_t value) {
    u8(static_cast<std::uint8_t>(value >> 8U));
    u8(static_cast<std::uint8_t>(value));
  }

  void u32(std::uint32_t value) {
    u16(static_cast<std::uint16_t>(value >> 16U));
    u16(static_cast<std::uint16_t>(value));
  }

  void u64(std::uint64_t value) {
    u32(static_cast<std::uint32_t>(value >> 32U));
    u32(static_cast<std::uint32_t>(value));
  }

  void i64(std::int64_t value) { u64(std::bit_cast<std::uint64_t>(value)); }

  void floating(double value) { u64(std::bit_cast<std::uint64_t>(value)); }

  void floating(float value) { u32(std::bit_cast<std::uint32_t>(value)); }

  void text(std::string_view value, std::size_t maximum, std::string_view description) {
    if (value.size() > maximum || value.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw CalibrationExecutionError(std::string(description) + " exceeds the protocol limit");
    }
    u32(static_cast<std::uint32_t>(value.size()));
    require_capacity(value.size());
    bytes_.append(value);
  }

  void optional_text(const std::optional<std::string>& value, std::size_t maximum,
                     std::string_view description) {
    boolean(value.has_value());
    if (value.has_value()) {
      text(*value, maximum, description);
    }
  }

  [[nodiscard]] std::string take() && { return std::move(bytes_); }

private:
  void require_capacity(std::size_t count) const {
    constexpr auto maximum_payload_bytes =
        kMaximumCalibrationWorkerFrameBytes - kCalibrationWorkerFrameHeaderBytes;
    if (count > maximum_payload_bytes - bytes_.size()) {
      throw CalibrationExecutionError("calibration worker payload exceeds the protocol limit");
    }
  }

  std::string bytes_;
};

class Reader {
public:
  explicit Reader(std::string_view bytes) : bytes_(bytes) {}

  [[nodiscard]] std::uint8_t u8(std::string_view description) {
    require(1, description);
    return static_cast<std::uint8_t>(static_cast<unsigned char>(bytes_[offset_++]));
  }

  [[nodiscard]] bool boolean(std::string_view description) {
    const auto value = u8(description);
    if (value > 1U) {
      throw CalibrationExecutionError(std::string(description) + " is not a protocol boolean");
    }
    return value != 0U;
  }

  [[nodiscard]] std::uint16_t u16(std::string_view description) {
    const auto high = u8(description);
    const auto low = u8(description);
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(high) << 8U) | low);
  }

  [[nodiscard]] std::uint32_t u32(std::string_view description) {
    const auto high = u16(description);
    const auto low = u16(description);
    return (static_cast<std::uint32_t>(high) << 16U) | low;
  }

  [[nodiscard]] std::uint64_t u64(std::string_view description) {
    const auto high = u32(description);
    const auto low = u32(description);
    return (static_cast<std::uint64_t>(high) << 32U) | low;
  }

  [[nodiscard]] std::int64_t i64(std::string_view description) {
    return std::bit_cast<std::int64_t>(u64(description));
  }

  [[nodiscard]] double floating64(std::string_view description) {
    return std::bit_cast<double>(u64(description));
  }

  [[nodiscard]] float floating32(std::string_view description) {
    return std::bit_cast<float>(u32(description));
  }

  [[nodiscard]] std::string text(std::size_t maximum, std::string_view description) {
    const auto size = u32(description);
    if (size > maximum) {
      throw CalibrationExecutionError(std::string(description) + " exceeds the protocol limit");
    }
    require(size, description);
    std::string result(bytes_.substr(offset_, size));
    offset_ += size;
    return result;
  }

  [[nodiscard]] std::optional<std::string> optional_text(std::size_t maximum,
                                                         std::string_view description) {
    if (!boolean(description)) {
      return std::nullopt;
    }
    return text(maximum, description);
  }

  void require_elements(std::size_t count, std::size_t element_size,
                        std::string_view description) const {
    if (element_size == 0 || count > (bytes_.size() - offset_) / element_size) {
      throw CalibrationExecutionError(std::string(description) + " is truncated");
    }
  }

  void finish(std::string_view description) const {
    if (offset_ != bytes_.size()) {
      throw CalibrationExecutionError(std::string(description) + " contains trailing bytes");
    }
  }

private:
  void require(std::size_t count, std::string_view description) const {
    if (count > bytes_.size() - offset_) {
      throw CalibrationExecutionError(std::string(description) + " is truncated");
    }
  }

  std::string_view bytes_;
  std::size_t offset_ = 0;
};

std::string frame(CalibrationWorkerMessage message, std::string payload) {
  if (payload.empty() ||
      payload.size() > kMaximumCalibrationWorkerFrameBytes - kCalibrationWorkerFrameHeaderBytes) {
    throw CalibrationExecutionError("calibration worker payload exceeds the protocol limit");
  }
  Writer header;
  for (const char byte : kMagic) {
    header.u8(static_cast<std::uint8_t>(byte));
  }
  header.u16(kCalibrationWorkerProtocolVersion);
  header.u16(static_cast<std::uint16_t>(message));
  header.u32(static_cast<std::uint32_t>(payload.size()));
  auto result = std::move(header).take();
  result += payload;
  return result;
}

std::pair<CalibrationWorkerMessage, std::string_view> parse_frame(std::string_view value) {
  if (value.size() < kCalibrationWorkerFrameHeaderBytes ||
      value.size() > kMaximumCalibrationWorkerFrameBytes) {
    throw CalibrationExecutionError("calibration worker frame size is invalid");
  }
  CalibrationWorkerFrameHeader header{};
  std::copy_n(value.data(), header.size(), header.data());
  const auto decoded = decode_calibration_worker_header(header);
  if (decoded.payload_size != value.size() - header.size()) {
    throw CalibrationExecutionError("calibration worker frame length does not match its header");
  }
  return {decoded.message, value.substr(header.size())};
}

void write_camera(Writer& writer, const reco::core::CameraParams& camera) {
  writer.u32(camera.width);
  writer.u32(camera.height);
  writer.floating(camera.fx);
  writer.floating(camera.fy);
  writer.floating(camera.cx);
  writer.floating(camera.cy);
  for (const auto coefficient : camera.d) {
    writer.floating(coefficient);
  }
}

reco::core::CameraParams read_camera(Reader& reader) {
  reco::core::CameraParams camera;
  camera.width = reader.u32("camera width");
  camera.height = reader.u32("camera height");
  camera.fx = reader.floating64("camera fx");
  camera.fy = reader.floating64("camera fy");
  camera.cx = reader.floating64("camera cx");
  camera.cy = reader.floating64("camera cy");
  for (auto& coefficient : camera.d) {
    coefficient = reader.floating64("camera distortion");
  }
  return camera;
}

void write_roi_points(Writer& writer, const std::vector<std::array<double, 2>>& points) {
  if (points.size() > kMaximumCalibrationWorkerRoiPoints) {
    throw CalibrationExecutionError("calibration result field ROI exceeds the protocol limit");
  }
  writer.u32(static_cast<std::uint32_t>(points.size()));
  for (const auto& point : points) {
    writer.floating(point[0]);
    writer.floating(point[1]);
  }
}

std::vector<std::array<double, 2>> read_roi_points(Reader& reader) {
  const auto count = reader.u32("field ROI point count");
  if (count > kMaximumCalibrationWorkerRoiPoints) {
    throw CalibrationExecutionError("calibration result field ROI exceeds the protocol limit");
  }
  std::vector<std::array<double, 2>> points;
  points.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    points.push_back({reader.floating64("field ROI x"), reader.floating64("field ROI y")});
  }
  return points;
}

void write_profile(Writer& writer, const std::optional<LensProfileInfo>& profile) {
  writer.boolean(profile.has_value());
  if (!profile.has_value()) {
    return;
  }
  writer.text(profile->camera, kMaximumProfileTextBytes, "lens profile camera");
  writer.text(profile->lens, kMaximumProfileTextBytes, "lens profile lens");
  writer.u8(static_cast<std::uint8_t>(profile->source));
  writer.optional_text(profile->path, kMaximumCalibrationWorkerPathBytes, "lens profile path");
}

std::optional<LensProfileInfo> read_profile(Reader& reader) {
  if (!reader.boolean("lens profile presence")) {
    return std::nullopt;
  }
  LensProfileInfo profile;
  profile.camera = reader.text(kMaximumProfileTextBytes, "lens profile camera");
  profile.lens = reader.text(kMaximumProfileTextBytes, "lens profile lens");
  const auto source = reader.u8("lens profile source");
  if (source > static_cast<std::uint8_t>(ProfileSource::Fallback)) {
    throw CalibrationExecutionError("calibration worker returned an invalid lens profile source");
  }
  profile.source = static_cast<ProfileSource>(source);
  profile.path = reader.optional_text(kMaximumCalibrationWorkerPathBytes, "lens profile path");
  return profile;
}

std::size_t read_size(Reader& reader, std::string_view description) {
  const auto value = reader.u64(description);
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw CalibrationExecutionError(std::string(description) + " exceeds the host size limit");
  }
  return static_cast<std::size_t>(value);
}

bool matched_point_is_finite(const MatchedPoint& point) {
  return std::isfinite(point.left[0]) && std::isfinite(point.left[1]) &&
         std::isfinite(point.right[0]) && std::isfinite(point.right[1]) &&
         std::isfinite(point.left_pixel_nx) && std::isfinite(point.right_pixel_nx);
}

void write_matched_point(Writer& writer, const MatchedPoint& point) {
  writer.floating(point.left[0]);
  writer.floating(point.left[1]);
  writer.floating(point.right[0]);
  writer.floating(point.right[1]);
  writer.floating(point.left_pixel_nx);
  writer.floating(point.right_pixel_nx);
}

MatchedPoint read_matched_point(Reader& reader) {
  MatchedPoint point;
  point.left[0] = reader.floating64("left correspondence x");
  point.left[1] = reader.floating64("left correspondence y");
  point.right[0] = reader.floating64("right correspondence x");
  point.right[1] = reader.floating64("right correspondence y");
  point.left_pixel_nx = reader.floating64("left correspondence pixel x");
  point.right_pixel_nx = reader.floating64("right correspondence pixel x");
  if (!matched_point_is_finite(point)) {
    throw CalibrationExecutionError("calibration worker returned a non-finite correspondence");
  }
  return point;
}

void validate_finite_result(const CalibrationResult& result) {
  if (!std::isfinite(result.residual_error) || !std::isfinite(result.confidence) ||
      (result.quality.has_value() && (!std::isfinite(result.quality->mean_reprojection_error) ||
                                      !std::isfinite(result.quality->trimmed_reprojection_error) ||
                                      !std::isfinite(result.quality->angular_error)))) {
    throw CalibrationExecutionError("calibration worker returned non-finite result metrics");
  }
  if (const auto error = result.calibration.validate(); !error.empty()) {
    throw CalibrationExecutionError("calibration worker returned an invalid calibration: " + error);
  }
}

void validate_result_counts(const CalibrationResult& result) {
  std::size_t summarized_matches = 0;
  std::size_t correspondence_count = 0;
  for (const auto& summary : result.per_frame) {
    if (summary.min_descriptors != std::min(summary.keypoints_left, summary.keypoints_right) ||
        summary.post_ratio_test > summary.min_descriptors ||
        summary.post_spatial_filter > summary.post_ratio_test ||
        summary.post_ransac > summary.post_spatial_filter ||
        summary.points.size() > summary.post_ransac ||
        summary.post_ransac > std::numeric_limits<std::size_t>::max() - summarized_matches) {
      throw CalibrationExecutionError("calibration worker returned inconsistent result counts");
    }
    if (summary.points.size() > kMaximumCalibrationWorkerCorrespondences - correspondence_count) {
      throw CalibrationExecutionError("calibration result has too many correspondences");
    }
    if (!std::all_of(summary.points.begin(), summary.points.end(), matched_point_is_finite)) {
      throw CalibrationExecutionError("calibration worker returned a non-finite correspondence");
    }
    summarized_matches += summary.post_ransac;
    correspondence_count += summary.points.size();
  }
  if (result.frames_used != result.per_frame.size() || result.total_matches == 0 ||
      result.total_matches != summarized_matches) {
    throw CalibrationExecutionError("calibration worker returned inconsistent result counts");
  }
}

} // namespace

DecodedCalibrationWorkerHeader
decode_calibration_worker_header(const CalibrationWorkerFrameHeader& header) {
  Reader reader(std::string_view(header.data(), header.size()));
  for (const char expected : kMagic) {
    if (reader.u8("calibration worker frame magic") != static_cast<std::uint8_t>(expected)) {
      throw CalibrationExecutionError("calibration worker frame magic is invalid");
    }
  }
  const auto version = reader.u16("calibration worker protocol version");
  if (version != kCalibrationWorkerProtocolVersion) {
    throw CalibrationExecutionError("unsupported calibration worker protocol version");
  }
  const auto raw_message = reader.u16("calibration worker message type");
  if (raw_message < static_cast<std::uint16_t>(CalibrationWorkerMessage::Request) ||
      raw_message > static_cast<std::uint16_t>(CalibrationWorkerMessage::Failure)) {
    throw CalibrationExecutionError("calibration worker message type is invalid");
  }
  const auto payload_size = reader.u32("calibration worker payload size");
  reader.finish("calibration worker frame header");
  if (payload_size == 0 ||
      payload_size > kMaximumCalibrationWorkerFrameBytes - kCalibrationWorkerFrameHeaderBytes) {
    throw CalibrationExecutionError("calibration worker payload size is invalid");
  }
  return {.message = static_cast<CalibrationWorkerMessage>(raw_message),
          .payload_size = payload_size};
}

std::string encode_calibration_worker_request(const GpuCalibrationRequest& request) {
  Writer writer;
  const auto write_input = [&writer](const CalibrationVideoInput& input) {
    writer.text(input.path, kMaximumCalibrationWorkerPathBytes, "calibration video path");
    writer.optional_text(input.lens_profile, kMaximumCalibrationWorkerPathBytes,
                         "calibration lens profile path");
    writer.optional_text(input.retained_path, kMaximumCalibrationWorkerPathBytes,
                         "retained calibration video path");
  };
  write_input(request.left);
  write_input(request.right);
  writer.u64(request.config.num_frames);
  writer.floating(request.config.skip_start_secs);
  writer.floating(request.config.skip_end_secs);
  writer.boolean(request.config.use_imu_rotation_seeds);
  const auto optional_double = [&writer](const std::optional<double>& value) {
    writer.boolean(value.has_value());
    if (value.has_value()) {
      writer.floating(*value);
    }
  };
  optional_double(request.config.imu_xrz_seed);
  optional_double(request.config.imu_xrx_seed);
  optional_double(request.config.imu_zrx_seed);
  writer.floating(request.config.akaze.threshold);
  writer.u64(request.config.akaze.max_keypoints);
  writer.floating(request.config.akaze.detect_y_min);
  writer.floating(request.config.akaze.detect_y_max);
  writer.floating(request.config.matching.lowe_ratio);
  writer.u64(request.config.matching.min_matches);
  writer.floating(request.config.matching.spatial_x_threshold);
  writer.floating(request.config.matching.spatial_x_inner);
  writer.floating(request.config.matching.spatial_y_low);
  writer.floating(request.config.matching.spatial_y_high);
  writer.floating(request.config.matching.max_y_disparity);
  writer.floating(request.config.matching.ransac_threshold);
  writer.boolean(request.config.optimizer.lock_cam_d);
  writer.boolean(request.config.optimizer.lock_z_rx);
  writer.boolean(request.config.optimizer.enable_x_rx);
  writer.floating(request.config.optimizer.seam_sigma);
  writer.floating(request.config.optimizer.trim_fraction);
  writer.u64(request.config.optimizer.max_iters);
  writer.boolean(request.no_auto_imu);
  writer.boolean(request.auto_sync);
  writer.i64(request.manual_sync_offset);
  writer.optional_text(request.debug_dir, kMaximumCalibrationWorkerPathBytes,
                       "calibration debug directory");
  writer.text(request.output, kMaximumCalibrationWorkerPathBytes, "calibration output path");
  writer.text(request.probe_worker, kMaximumCalibrationWorkerPathBytes, "video probe worker path");
  writer.u64(request.probe_timeout_ns);
  writer.text(request.calibration_worker_path, kMaximumCalibrationWorkerPathBytes,
              "calibration worker path");
  writer.u64(request.calibration_timeout_ns);
  writer.u64(request.calibration_host_memory_limit_bytes);
  writer.u32(static_cast<std::uint32_t>(request.nvbufsurface_abi));
  return frame(CalibrationWorkerMessage::Request, std::move(writer).take());
}

GpuCalibrationRequest decode_calibration_worker_request(std::string_view value) {
  const auto [message, payload] = parse_frame(value);
  if (message != CalibrationWorkerMessage::Request) {
    throw CalibrationExecutionError("calibration worker expected a request frame");
  }
  Reader reader(payload);
  GpuCalibrationRequest request;
  const auto read_input = [&reader](CalibrationVideoInput& input) {
    input.path = reader.text(kMaximumCalibrationWorkerPathBytes, "calibration video path");
    input.lens_profile =
        reader.optional_text(kMaximumCalibrationWorkerPathBytes, "calibration lens profile path");
    input.retained_path =
        reader.optional_text(kMaximumCalibrationWorkerPathBytes, "retained calibration video path");
  };
  read_input(request.left);
  read_input(request.right);
  request.config.num_frames = read_size(reader, "calibration frame count");
  request.config.skip_start_secs = reader.floating64("calibration skip start");
  request.config.skip_end_secs = reader.floating64("calibration skip end");
  request.config.use_imu_rotation_seeds = reader.boolean("IMU seed policy");
  const auto optional_double = [&reader](std::string_view description) -> std::optional<double> {
    if (!reader.boolean(description)) {
      return std::nullopt;
    }
    return reader.floating64(description);
  };
  request.config.imu_xrz_seed = optional_double("IMU xrz seed");
  request.config.imu_xrx_seed = optional_double("IMU xrx seed");
  request.config.imu_zrx_seed = optional_double("IMU zrx seed");
  request.config.akaze.threshold = reader.floating64("AKAZE threshold");
  request.config.akaze.max_keypoints = read_size(reader, "AKAZE keypoint limit");
  request.config.akaze.detect_y_min = reader.floating64("AKAZE Y minimum");
  request.config.akaze.detect_y_max = reader.floating64("AKAZE Y maximum");
  request.config.matching.lowe_ratio = reader.floating64("Lowe ratio");
  request.config.matching.min_matches = read_size(reader, "minimum match count");
  request.config.matching.spatial_x_threshold = reader.floating64("spatial X threshold");
  request.config.matching.spatial_x_inner = reader.floating64("spatial X inner limit");
  request.config.matching.spatial_y_low = reader.floating64("spatial Y low limit");
  request.config.matching.spatial_y_high = reader.floating64("spatial Y high limit");
  request.config.matching.max_y_disparity = reader.floating64("maximum Y disparity");
  request.config.matching.ransac_threshold = reader.floating64("RANSAC threshold");
  request.config.optimizer.lock_cam_d = reader.boolean("optimizer camera distance lock");
  request.config.optimizer.lock_z_rx = reader.boolean("optimizer zrx lock");
  request.config.optimizer.enable_x_rx = reader.boolean("optimizer xrx policy");
  request.config.optimizer.seam_sigma = reader.floating64("optimizer seam sigma");
  request.config.optimizer.trim_fraction = reader.floating64("optimizer trim fraction");
  request.config.optimizer.max_iters = read_size(reader, "optimizer iteration limit");
  request.no_auto_imu = reader.boolean("automatic IMU policy");
  request.auto_sync = reader.boolean("automatic sync policy");
  request.manual_sync_offset = reader.i64("manual sync offset");
  request.debug_dir =
      reader.optional_text(kMaximumCalibrationWorkerPathBytes, "calibration debug directory");
  request.output = reader.text(kMaximumCalibrationWorkerPathBytes, "calibration output path");
  request.probe_worker = reader.text(kMaximumCalibrationWorkerPathBytes, "video probe worker path");
  request.probe_timeout_ns = reader.u64("video probe timeout");
  request.calibration_worker_path =
      reader.text(kMaximumCalibrationWorkerPathBytes, "calibration worker path");
  request.calibration_timeout_ns = reader.u64("calibration timeout");
  request.calibration_host_memory_limit_bytes = reader.u64("calibration host memory limit");
  const auto abi = reader.u32("NvBufSurface ABI");
  if (abi != static_cast<std::uint32_t>(reco::io::NvbufSurfaceAbi::DeepStream7_1) &&
      abi != static_cast<std::uint32_t>(reco::io::NvbufSurfaceAbi::DeepStream9_1)) {
    throw CalibrationExecutionError("calibration worker request has an invalid NvBufSurface ABI");
  }
  request.nvbufsurface_abi = static_cast<reco::io::NvbufSurfaceAbi>(abi);
  reader.finish("calibration worker request");
  if (const auto error = validate_gpu_calibration_request(request); error.has_value()) {
    throw CalibrationExecutionError("invalid calibration worker request: " + *error);
  }
  return request;
}

std::string encode_calibration_worker_success(const CalibrationResult& result) {
  validate_finite_result(result);
  if (result.per_frame.size() > kMaximumCalibrationWorkerResultFrames) {
    throw CalibrationExecutionError("calibration result has too many frame summaries");
  }
  validate_result_counts(result);
  Writer writer;
  write_camera(writer, result.calibration.left);
  write_camera(writer, result.calibration.right);
  const auto& layout = result.calibration.layout;
  writer.floating(layout.camera_axis_offset);
  writer.floating(layout.intersect);
  writer.floating(layout.x_ty);
  writer.floating(layout.x_rz);
  writer.floating(layout.z_rx);
  writer.floating(layout.x_rx);
  writer.floating(layout.z_rz);
  writer.floating(result.calibration.rig_tilt);
  writer.floating(result.calibration.rig_roll);
  writer.i64(result.calibration.sync_offset);
  writer.boolean(result.calibration.field_roi.has_value());
  if (result.calibration.field_roi.has_value()) {
    write_roi_points(writer, result.calibration.field_roi->left);
    write_roi_points(writer, result.calibration.field_roi->right);
  }
  writer.floating(result.calibration.lens_correction_amount);
  writer.floating(result.calibration.blend_width);
  writer.u64(result.total_matches);
  writer.u64(result.frames_used);
  writer.floating(result.residual_error);
  writer.floating(result.confidence);
  writer.u32(static_cast<std::uint32_t>(result.per_frame.size()));
  for (const auto& summary : result.per_frame) {
    writer.u64(summary.keypoints_left);
    writer.u64(summary.keypoints_right);
    writer.u64(summary.min_descriptors);
    writer.u64(summary.post_ratio_test);
    writer.u64(summary.post_spatial_filter);
    writer.u64(summary.post_ransac);
    writer.u32(static_cast<std::uint32_t>(summary.points.size()));
    for (const auto& point : summary.points) {
      write_matched_point(writer, point);
    }
  }
  write_profile(writer, result.left_lens_profile);
  write_profile(writer, result.right_lens_profile);
  writer.boolean(result.quality.has_value());
  if (result.quality.has_value()) {
    writer.floating(result.quality->mean_reprojection_error);
    writer.floating(result.quality->trimmed_reprojection_error);
    writer.floating(result.quality->angular_error);
  }
  return frame(CalibrationWorkerMessage::Success, std::move(writer).take());
}

std::string encode_calibration_worker_failure(std::string_view message) {
  Writer writer;
  writer.text(message.substr(0, kMaximumCalibrationWorkerErrorBytes),
              kMaximumCalibrationWorkerErrorBytes, "calibration worker error");
  return frame(CalibrationWorkerMessage::Failure, std::move(writer).take());
}

CalibrationResult decode_calibration_worker_response(std::string_view value) {
  const auto [message, payload] = parse_frame(value);
  if (message == CalibrationWorkerMessage::Request) {
    throw CalibrationExecutionError("calibration worker returned a request frame");
  }
  Reader reader(payload);
  if (message == CalibrationWorkerMessage::Failure) {
    const auto error = reader.text(kMaximumCalibrationWorkerErrorBytes, "calibration worker error");
    reader.finish("calibration worker failure");
    throw CalibrationExecutionError("calibration worker failed: " + error);
  }

  CalibrationResult result;
  result.calibration.left = read_camera(reader);
  result.calibration.right = read_camera(reader);
  auto& layout = result.calibration.layout;
  layout.camera_axis_offset = reader.floating64("camera axis offset");
  layout.intersect = reader.floating64("camera intersection");
  layout.x_ty = reader.floating64("layout x translation");
  layout.x_rz = reader.floating64("layout xrz");
  layout.z_rx = reader.floating64("layout zrx");
  layout.x_rx = reader.floating64("layout xrx");
  layout.z_rz = reader.floating64("layout zrz");
  result.calibration.rig_tilt = reader.floating64("rig tilt");
  result.calibration.rig_roll = reader.floating64("rig roll");
  result.calibration.sync_offset = reader.i64("calibration sync offset");
  if (reader.boolean("field ROI presence")) {
    result.calibration.field_roi =
        reco::core::FieldRoi{.left = read_roi_points(reader), .right = read_roi_points(reader)};
  }
  result.calibration.lens_correction_amount = reader.floating32("lens correction amount");
  result.calibration.blend_width = reader.floating32("blend width");
  result.total_matches = read_size(reader, "total match count");
  result.frames_used = read_size(reader, "used frame count");
  result.residual_error = reader.floating64("residual error");
  result.confidence = reader.floating64("calibration confidence");
  const auto frame_count = reader.u32("frame summary count");
  if (frame_count > kMaximumCalibrationWorkerResultFrames) {
    throw CalibrationExecutionError("calibration worker returned too many frame summaries");
  }
  result.per_frame.reserve(frame_count);
  std::size_t correspondence_count = 0;
  for (std::uint32_t index = 0; index < frame_count; ++index) {
    FrameMatches summary;
    summary.keypoints_left = read_size(reader, "left keypoint count");
    summary.keypoints_right = read_size(reader, "right keypoint count");
    summary.min_descriptors = read_size(reader, "descriptor count");
    summary.post_ratio_test = read_size(reader, "ratio match count");
    summary.post_spatial_filter = read_size(reader, "spatial match count");
    summary.post_ransac = read_size(reader, "RANSAC match count");
    const auto point_count = reader.u32("correspondence count");
    if (point_count > summary.post_ransac) {
      throw CalibrationExecutionError("calibration worker returned inconsistent result counts");
    }
    if (point_count > kMaximumCalibrationWorkerCorrespondences - correspondence_count) {
      throw CalibrationExecutionError("calibration worker returned too many correspondences");
    }
    reader.require_elements(point_count, kCalibrationWorkerMatchedPointBytes,
                            "calibration worker correspondence payload");
    summary.points.reserve(point_count);
    for (std::uint32_t point_index = 0; point_index < point_count; ++point_index) {
      summary.points.push_back(read_matched_point(reader));
    }
    correspondence_count += point_count;
    result.per_frame.push_back(std::move(summary));
  }
  result.left_lens_profile = read_profile(reader);
  result.right_lens_profile = read_profile(reader);
  if (reader.boolean("calibration quality presence")) {
    result.quality = CalibrationQuality{
        .mean_reprojection_error = reader.floating64("mean reprojection error"),
        .trimmed_reprojection_error = reader.floating64("trimmed reprojection error"),
        .angular_error = reader.floating64("angular error")};
  }
  reader.finish("calibration worker response");
  validate_finite_result(result);
  validate_result_counts(result);
  return result;
}

} // namespace reco::calibrate::detail
