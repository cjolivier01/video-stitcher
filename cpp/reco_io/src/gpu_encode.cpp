#include "reco/io/gpu_encode.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace reco::io {
namespace {

std::string quote_gstreamer_property(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 2U);
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

std::string_view parser_factory(Codec codec) {
  switch (codec) {
  case Codec::H264:
    return "h264parse config-interval=-1";
  case Codec::HEVC:
    return "h265parse config-interval=-1";
  case Codec::AV1:
    return "av1parse";
  }
  throw std::invalid_argument("unsupported GPU encode codec");
}

std::string muxer_description(Format format) {
  switch (format) {
  case Format::Mp4:
    return "mp4mux name=mux";
  case Format::Mp4Fragmented:
    return "mp4mux name=mux fragment-duration=1000 streamable=true";
  case Format::Mkv:
    return "matroskamux name=mux streamable=true";
  case Format::Mov:
    return "qtmux name=mux";
  case Format::Flv:
    return "flvmux name=mux streamable=true";
  }
  throw std::invalid_argument("unsupported GPU encode container");
}

std::uint32_t target_bitrate(const GpuEncodeConfig& config) {
  const auto pixel_rate = static_cast<long double>(config.width) * config.height *
                          config.fps_numerator / config.fps_denominator;
  long double bits_per_pixel = 0.10L;
  switch (config.quality) {
  case Quality::Fast:
    bits_per_pixel = 0.065L;
    break;
  case Quality::Balanced:
    bits_per_pixel = 0.10L;
    break;
  case Quality::High:
    bits_per_pixel = 0.16L;
    break;
  }
  if (config.quality_value.has_value()) {
    bits_per_pixel = 0.04L + static_cast<long double>(*config.quality_value) * 0.002L;
  }
  const auto raw = pixel_rate * bits_per_pixel;
  return static_cast<std::uint32_t>(std::clamp<long double>(raw, 500'000.0L, 120'000'000.0L));
}

bool valid_timeout(std::chrono::milliseconds timeout) {
  return timeout >= std::chrono::milliseconds(1) && timeout <= std::chrono::hours(1);
}

} // namespace

std::optional<std::string> validate_gpu_encode_config(const GpuEncodeConfig& config) {
  const bool has_output_path = !config.output_path.empty();
  if (has_output_path == config.output_descriptor.has_value()) {
    return "GPU encode requires exactly one output path or descriptor";
  }
  if (config.output_path.find('\0') != std::string::npos) {
    return "GPU encode output path contains an embedded NUL";
  }
  if (config.output_descriptor.has_value() && *config.output_descriptor < 0) {
    return "GPU encode output descriptor must be non-negative";
  }
  if (config.width == 0 || config.height == 0 || (config.width % 2U) != 0 ||
      (config.height % 2U) != 0) {
    return "GPU encode dimensions must be non-zero and even";
  }
  if (config.width > 16'384U || config.height > 16'384U) {
    return "GPU encode dimensions exceed the supported 16384-pixel limit";
  }
  if (config.fps_numerator == 0 || config.fps_denominator == 0) {
    return "GPU encode frame rate must be a positive rational";
  }
  if (static_cast<std::uint64_t>(config.fps_numerator) >
      static_cast<std::uint64_t>(config.fps_denominator) * 1'000U) {
    return "GPU encode frame rate exceeds 1000 fps";
  }
  if (config.device_ordinal != 0U) {
    return "GPU encode currently requires CUDA device ordinal zero";
  }
  if (config.pool_capacity < kMinimumGpuEncodePoolCapacity ||
      config.pool_capacity > kMaximumGpuEncodePoolCapacity) {
    return "GPU encode pool capacity must be between 8 and 16";
  }
  if (!valid_timeout(config.acquire_timeout) || !valid_timeout(config.startup_timeout) ||
      !valid_timeout(config.finalize_timeout)) {
    return "GPU encode timeouts must be between 1 millisecond and 1 hour";
  }
  if (config.quality_value.has_value() && *config.quality_value > 100U) {
    return "GPU encode quality value must be between 0 and 100";
  }
  if (config.encoder.has_value() &&
      *config.encoder != gstreamer_hardware_encoder_factory(config.codec)) {
    return "GPU encode factory must be the NVIDIA hardware encoder for the selected codec";
  }
  if (config.audio_caps.has_value() &&
      (config.audio_caps->empty() || config.audio_caps->size() > 16U * 1024U ||
       config.audio_caps->find('\0') != std::string::npos)) {
    return "GPU encode audio caps must be non-empty, NUL-free, and at most 16 KiB";
  }
  return std::nullopt;
}

std::string_view gstreamer_hardware_encoder_factory(Codec codec) {
  switch (codec) {
  case Codec::H264:
    return "nvv4l2h264enc";
  case Codec::HEVC:
    return "nvv4l2h265enc";
  case Codec::AV1:
    return "nvv4l2av1enc";
  }
  throw std::invalid_argument("unsupported GPU encode codec");
}

std::string build_gstreamer_gpu_encode_pipeline(const GpuEncodeConfig& config) {
  if (const auto error = validate_gpu_encode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }

  std::ostringstream pipeline;
  pipeline << "appsrc name=source is-live=false format=time do-timestamp=false block=false "
              "max-buffers="
           << config.pool_capacity
           << " caps=\"video/x-raw(memory:NVMM),format=(string)NV12,width=(int)" << config.width
           << ",height=(int)" << config.height << ",framerate=(fraction)" << config.fps_numerator
           << '/' << config.fps_denominator
           << ",colorimetry=(string)bt709,chroma-site=(string)mpeg2\""
           << " ! queue max-size-buffers=" << config.pool_capacity
           << " max-size-bytes=0 max-size-time=0"
           << " ! " << gstreamer_hardware_encoder_factory(config.codec)
           << " name=encoder gpu-id=" << config.device_ordinal
           << " bitrate=" << target_bitrate(config) << " ! " << parser_factory(config.codec)
           << " ! mux. ";
  if (config.audio_caps.has_value()) {
    pipeline << "appsrc name=audio_source is-live=false format=time do-timestamp=false block=false "
                "max-buffers=32 max-bytes=33554432 max-time=0 caps="
             << quote_gstreamer_property(*config.audio_caps)
             << " ! queue max-size-buffers=32 max-size-bytes=16777216 max-size-time=0 ! mux. ";
  }
  pipeline << muxer_description(config.format) << " ! ";
  if (config.output_descriptor.has_value()) {
    pipeline << "fdsink fd=" << *config.output_descriptor << " sync=false async=false";
  } else {
    pipeline << "filesink location=" << quote_gstreamer_property(config.output_path)
             << " sync=false async=false";
  }
  return pipeline.str();
}

} // namespace reco::io
