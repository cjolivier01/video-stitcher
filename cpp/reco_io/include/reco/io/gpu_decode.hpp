#pragma once

#include "reco/io/nvmm.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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
  // Elementary streams require an explicit parser. Supported containers
  // select H264 or HEVC from their video pad at runtime.
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
  std::uint32_t visible_width = 0;
  std::uint32_t visible_height = 0;
  std::shared_ptr<void> owner;
  std::uint64_t frame_index = 0;
  std::optional<std::uint64_t> pts_ns;
  std::optional<std::uint64_t> duration_ns;
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

/// Failure while loading or consuming a GPU-resident GStreamer decode stream.
class GpuDecodeError : public std::runtime_error {
public:
  explicit GpuDecodeError(std::string message) : std::runtime_error(std::move(message)) {}
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
/// Opens an NVDEC/NVMM appsink source using the selected DeepStream surface ABI.
[[nodiscard]] std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_file_decode_source(GpuFileDecodeConfig config, NvbufSurfaceAbi abi);

} // namespace reco::io
