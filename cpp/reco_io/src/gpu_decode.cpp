#include "reco/io/gpu_decode.hpp"

#include <sstream>
#include <stdexcept>

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
  return std::nullopt;
}

std::string build_gstreamer_gpu_file_decode_pipeline(const GpuFileDecodeConfig& config) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  std::ostringstream pipeline;
  pipeline << "filesrc location=" << quote_gstreamer_property(config.path) << " ! qtdemux ! "
           << parser_for_codec(config.codec)
           << " ! nvv4l2decoder ! video/x-raw(memory:NVMM),format=NV12"
           << " ! appsink name=sink emit-signals=false sync=false max-buffers="
           << config.max_buffers << " drop=" << (config.drop ? "true" : "false");
  return pipeline.str();
}

} // namespace reco::io
