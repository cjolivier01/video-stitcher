#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace reco::io {

enum class GpuDecodeCodec {
  H264,
  Hevc,
};

enum class GpuDecodeContainer {
  QuickTime,
  Matroska,
  MpegTs,
};

struct GpuFileDecodeConfig {
  std::string path;
  GpuDecodeCodec codec = GpuDecodeCodec::H264;
  bool elementary_stream = false;
  std::optional<GpuDecodeContainer> container;
  std::uint32_t max_buffers = 4;
  bool drop = false;
};

[[nodiscard]] std::string_view gpu_decode_codec_name(GpuDecodeCodec codec);
[[nodiscard]] std::string_view gpu_decode_container_demuxer(GpuDecodeContainer container);
[[nodiscard]] GpuDecodeCodec gpu_decode_codec_for_path(std::string_view path);
[[nodiscard]] bool gpu_decode_path_is_elementary_stream(std::string_view path);
[[nodiscard]] std::optional<GpuDecodeContainer>
gpu_decode_container_for_path(std::string_view path);
[[nodiscard]] std::optional<std::string>
validate_gpu_file_decode_config(const GpuFileDecodeConfig& config);
[[nodiscard]] std::string
build_gstreamer_gpu_file_decode_pipeline(const GpuFileDecodeConfig& config);

} // namespace reco::io
