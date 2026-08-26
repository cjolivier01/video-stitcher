#include "reco/io/gpu_decode.hpp"

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

std::string_view caps_for_codec(GpuDecodeCodec codec) {
  switch (codec) {
  case GpuDecodeCodec::H264:
    return "video/x-h264";
  case GpuDecodeCodec::Hevc:
    return "video/x-h265";
  }
  return "video/x-h264";
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
  const auto ext = lowercase(std::filesystem::path(std::string(path)).extension().string());
  if (ext == ".h265" || ext == ".hevc" || ext == ".265") {
    return GpuDecodeCodec::Hevc;
  }
  return GpuDecodeCodec::H264;
}

bool gpu_decode_path_is_elementary_stream(std::string_view path) {
  const auto ext = lowercase(std::filesystem::path(std::string(path)).extension().string());
  return ext == ".h264" || ext == ".264" || ext == ".h265" || ext == ".hevc" || ext == ".265";
}

std::optional<GpuDecodeContainer> gpu_decode_container_for_path(std::string_view path) {
  const auto ext = lowercase(std::filesystem::path(std::string(path)).extension().string());
  if (ext == ".mp4" || ext == ".mov" || ext == ".m4v") {
    return GpuDecodeContainer::QuickTime;
  }
  if (ext == ".mkv" || ext == ".webm") {
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
  if (!config.elementary_stream && !config.container.has_value()) {
    return "GPU file decode container is unsupported";
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
  return std::nullopt;
}

NvmmCudaFrame map_gpu_decoded_frame_to_cuda(const GpuDecodedFrame& frame) {
  if (const auto error = validate_gpu_decoded_frame(frame); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  return map_nvmm_frame_to_cuda(frame.nvmm, frame.owner);
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
    pipeline << gpu_decode_container_demuxer(*config.container) << " ! "
             << caps_for_codec(config.codec) << " ! " << parser_for_codec(config.codec);
  }
  pipeline << " ! nvv4l2decoder"
           << " ! nvvideoconvert compute-hw=1 bl-output=false disable-passthrough=true"
           << " ! video/x-raw(memory:NVMM),format=NV12"
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
