#include "reco/io/gpu_decode.hpp"
#include "reco/core/path.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace reco::io {
namespace {

bool contains_pipeline_metacharacter(std::string_view value) {
  for (const char ch : value) {
    if (ch == '!' || ch == '\n' || ch == '\r' || ch == '\0') {
      return true;
    }
  }
  return false;
}

std::string quote_gstreamer_property(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('"');
  for (const char ch : value) {
    if (ch == '\\' || ch == '"') {
      quoted.push_back('\\');
    }
    quoted.push_back(ch);
  }
  quoted.push_back('"');
  return quoted;
}

std::string_view parser_for_codec(GpuDecodeCodec codec) {
  switch (codec) {
  case GpuDecodeCodec::H264:
    return "h264parse";
  case GpuDecodeCodec::Hevc:
    return "h265parse";
  }
  return "h264parse";
}

std::string lowercase(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

} // namespace

std::string_view gpu_decode_codec_name(GpuDecodeCodec codec) {
  switch (codec) {
  case GpuDecodeCodec::H264:
    return "h264";
  case GpuDecodeCodec::Hevc:
    return "hevc";
  }
  return "h264";
}

std::string_view gpu_decode_container_demuxer(GpuDecodeContainer container) {
  switch (container) {
  case GpuDecodeContainer::QuickTime:
    return "qtdemux";
  case GpuDecodeContainer::Matroska:
    return "matroskademux";
  case GpuDecodeContainer::MpegTs:
    return "tsdemux";
  }
  return "qtdemux";
}

GpuDecodeCodec gpu_decode_codec_for_path(std::string_view path) {
  const auto ext = lowercase(reco::core::path_from_utf8(path).extension().string());
  if (ext == ".h265" || ext == ".hevc" || ext == ".265") {
    return GpuDecodeCodec::Hevc;
  }
  return GpuDecodeCodec::H264;
}

bool gpu_decode_path_is_elementary_stream(std::string_view path) {
  const auto ext = lowercase(reco::core::path_from_utf8(path).extension().string());
  return ext == ".h264" || ext == ".264" || ext == ".h265" || ext == ".hevc" || ext == ".265";
}

std::optional<GpuDecodeContainer> gpu_decode_container_for_path(std::string_view path) {
  const auto ext = lowercase(reco::core::path_from_utf8(path).extension().string());
  if (ext == ".mp4" || ext == ".mov" || ext == ".m4v") {
    return GpuDecodeContainer::QuickTime;
  }
  if (ext == ".mkv") {
    return GpuDecodeContainer::Matroska;
  }
  if (ext == ".ts" || ext == ".mts" || ext == ".m2ts") {
    return GpuDecodeContainer::MpegTs;
  }
  return std::nullopt;
}

std::optional<std::string> validate_gpu_file_decode_config(const GpuFileDecodeConfig& config) {
  if (config.path.empty()) {
    return "GPU file decode path is required";
  }
  if (contains_pipeline_metacharacter(config.path)) {
    return "GPU file decode path contains GStreamer pipeline metacharacters";
  }
  if (config.max_buffers == 0) {
    return "GPU file decode max_buffers must be non-zero";
  }
  if (config.read_timeout_ns < 100'000'000ULL || config.read_timeout_ns > 3'600'000'000'000ULL) {
    return "GPU file decode read timeout must be between 100 ms and one hour";
  }
  if (config.indexed_fps_numerator.has_value() != config.indexed_fps_denominator.has_value()) {
    return "GPU file decode indexed cadence requires both numerator and denominator";
  }
  if (config.indexed_fps_numerator.value_or(1U) == 0U ||
      config.indexed_fps_denominator.value_or(1U) == 0U) {
    return "GPU file decode indexed cadence must be non-zero";
  }
  if (config.indexed_timestamp_multiplicity == 0U ||
      config.indexed_timestamp_multiplicity > kMaximumIndexedTimestampMultiplicity) {
    return "GPU file decode indexed timestamp multiplicity is out of range";
  }
  if (config.indexed_timestamp_multiplicity != 1U && !config.indexed_fps_numerator.has_value()) {
    return "GPU file decode timestamp multiplicity requires indexed cadence";
  }
  if (config.start_frame_index.has_value() && !config.indexed_fps_numerator.has_value()) {
    return "GPU file decode start frame requires indexed cadence";
  }
  if (config.start_frame_index.has_value() && !config.indexed_stream_time_origin_ns.has_value()) {
    return "GPU file decode start frame requires the probed stream-time origin";
  }
  if (config.indexed_stream_time_origin_ns.has_value() &&
      !config.indexed_fps_numerator.has_value()) {
    return "GPU file decode stream-time origin requires indexed cadence";
  }
  if (!config.elementary_stream && !config.container.has_value()) {
    return "GPU file decode container is unsupported";
  }
  return std::nullopt;
}

void GpuFileDecodeSource::seek_to_frame(std::uint64_t) {
  throw GpuDecodeError("GPU file decode source does not support indexed seeking");
}

std::optional<std::string> validate_gpu_stereo_decode_config(const GpuStereoDecodeConfig& config) {
  if (config.sync_offset < -kMaximumGpuStereoSyncOffset ||
      config.sync_offset > kMaximumGpuStereoSyncOffset) {
    return "GPU stereo decode sync offset must be between -100000 and 100000 frames";
  }
  if (config.queue_capacity < kMinimumGpuStereoQueueCapacity ||
      config.queue_capacity > kMaximumGpuStereoQueueCapacity) {
    return "GPU stereo decode queue capacity must be between 1 and 16";
  }
  return std::nullopt;
}

std::optional<std::string> validate_gpu_decoded_frame(const GpuDecodedFrame& frame) {
  if (!frame.owner) {
    return "GPU decoded frame must retain its decoder buffer owner";
  }
  if (const auto error = validate_nvmm_frame_info(frame.nvmm); error.has_value()) {
    return "GPU decoded frame is invalid: " + *error;
  }
  if (frame.visible_width == 0 || frame.visible_height == 0) {
    return "GPU decoded frame visible dimensions must be non-zero";
  }
  if ((frame.visible_width % 2U) != 0 || (frame.visible_height % 2U) != 0) {
    return "GPU decoded frame visible NV12 dimensions must be even";
  }
  if (frame.visible_width > frame.nvmm.width || frame.visible_height > frame.nvmm.height) {
    return "GPU decoded frame visible dimensions exceed the NVMM allocation";
  }
  if (frame.rotation_degrees != 0U && frame.rotation_degrees != 90U &&
      frame.rotation_degrees != 180U && frame.rotation_degrees != 270U) {
    return "GPU decoded frame rotation must be 0, 90, 180, or 270 degrees";
  }
  return std::nullopt;
}

NvmmCudaFrame map_gpu_decoded_frame_to_cuda(const GpuDecodedFrame& frame) {
  if (const auto error = validate_gpu_decoded_frame(frame); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  auto mapped = map_nvmm_frame_to_cuda(frame.nvmm, frame.owner);
  mapped.width = frame.visible_width;
  mapped.height = frame.visible_height;
  return mapped;
}

CudaNv12FrameLease map_gpu_decoded_frame_to_cuda_lease(const GpuDecodedFrame& frame) {
  auto mapped = map_gpu_decoded_frame_to_cuda(frame);
  core::CudaPitchedPlaneView y_plane(
      mapped.y_ptr, mapped.y_accessible_bytes, mapped.y_pitch, mapped.width, mapped.height,
      static_cast<core::CudaContextId>(mapped.context_id), mapped.device_ordinal);
  core::CudaPitchedPlaneView uv_plane(
      mapped.uv_ptr, mapped.uv_accessible_bytes, mapped.uv_pitch, mapped.width, mapped.height / 2U,
      static_cast<core::CudaContextId>(mapped.context_id), mapped.device_ordinal);
  core::CudaNv12FrameView view(std::move(y_plane), std::move(uv_plane), mapped.width, mapped.height,
                               mapped.color_matrix, mapped.color_range);
  return CudaNv12FrameLease(std::move(mapped), std::move(view));
}

std::string build_gstreamer_gpu_file_decode_pipeline(const GpuFileDecodeConfig& config) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  std::ostringstream pipeline;
  pipeline << "filesrc location=" << quote_gstreamer_property(config.path) << " ! ";
  if (config.elementary_stream) {
    pipeline << parser_for_codec(config.codec);
  } else {
    pipeline << gpu_decode_container_demuxer(*config.container)
             << " ! capsfilter caps=\"video/x-h264;video/x-h265\" ! parsebin";
  }
  pipeline << " ! identity name=display_info silent=true"
           << " ! nvv4l2decoder"
           << " ! nvvideoconvert compute-hw=1 bl-output=false disable-passthrough=true"
           << " ! video/x-raw(memory:NVMM),format=NV12"
           << " ! identity name=output_info silent=true"
           << " ! appsink name=sink emit-signals=false sync=false max-buffers="
           << config.max_buffers << " drop=" << (config.drop ? "true" : "false");
  return pipeline.str();
}

GpuDecodeReadResult make_gpu_decode_eos() {
  return {.status = GpuDecodeFrameStatus::EndOfStream, .frame = std::nullopt};
}

GpuDecodeReadResult make_gpu_decode_frame(GpuDecodedFrame frame) {
  if (const auto error = validate_gpu_decoded_frame(frame); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  return {.status = GpuDecodeFrameStatus::Frame, .frame = std::move(frame)};
}

} // namespace reco::io
