#pragma once

#include "reco/io/nvmm.hpp"

#include <cstdint>
#include <memory>
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

enum class GpuDecodeFrameStatus {
  Frame,
  EndOfStream,
};

struct GpuDecodedFrame {
  NvmmFrameInfo nvmm;
  std::shared_ptr<void> owner;
  std::uint64_t frame_index = 0;
  std::int64_t pts_ns = 0;
  std::int64_t duration_ns = 0;
};

struct GpuDecodeReadResult {
  GpuDecodeFrameStatus status = GpuDecodeFrameStatus::EndOfStream;
  std::optional<GpuDecodedFrame> frame;
};

class GpuFileDecodeSource {
public:
  virtual ~GpuFileDecodeSource() = default;

  [[nodiscard]] virtual const GpuFileDecodeConfig& config() const = 0;
  [[nodiscard]] virtual std::string_view pipeline() const = 0;
  [[nodiscard]] virtual bool gpu_resident() const = 0;
  [[nodiscard]] virtual GpuDecodeReadResult read() = 0;
};

[[nodiscard]] std::string_view gpu_decode_codec_name(GpuDecodeCodec codec);
[[nodiscard]] std::string_view gpu_decode_container_demuxer(GpuDecodeContainer container);
[[nodiscard]] GpuDecodeCodec gpu_decode_codec_for_path(std::string_view path);
[[nodiscard]] bool gpu_decode_path_is_elementary_stream(std::string_view path);
[[nodiscard]] std::optional<GpuDecodeContainer>
gpu_decode_container_for_path(std::string_view path);
[[nodiscard]] std::optional<std::string>
validate_gpu_file_decode_config(const GpuFileDecodeConfig& config);
[[nodiscard]] std::optional<std::string> validate_gpu_decoded_frame(const GpuDecodedFrame& frame);
[[nodiscard]] NvmmCudaFrame map_gpu_decoded_frame_to_cuda(const GpuDecodedFrame& frame);
[[nodiscard]] std::string
build_gstreamer_gpu_file_decode_pipeline(const GpuFileDecodeConfig& config);
[[nodiscard]] GpuDecodeReadResult make_gpu_decode_eos();
[[nodiscard]] GpuDecodeReadResult make_gpu_decode_frame(GpuDecodedFrame frame);

} // namespace reco::io
