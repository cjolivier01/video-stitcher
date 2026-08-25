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

struct GpuFileDecodeConfig {
  std::string path;
  GpuDecodeCodec codec = GpuDecodeCodec::H264;
  std::uint32_t max_buffers = 4;
  bool drop = true;
};

[[nodiscard]] std::string_view gpu_decode_codec_name(GpuDecodeCodec codec);
[[nodiscard]] std::optional<std::string>
validate_gpu_file_decode_config(const GpuFileDecodeConfig& config);
[[nodiscard]] std::string build_gstreamer_gpu_file_decode_pipeline(const GpuFileDecodeConfig& config);

} // namespace reco::io
