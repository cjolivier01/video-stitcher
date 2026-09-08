#pragma once

#include "reco/core/cuda_frame.hpp"
#include "reco/io/nvmm.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace reco::io {

inline constexpr std::uint32_t kMaximumIndexedTimestampMultiplicity = 1024;
inline constexpr std::int64_t kMaximumGpuStereoSyncOffset = 100'000;
inline constexpr std::uint32_t kMinimumGpuStereoQueueCapacity = 1;
inline constexpr std::uint32_t kMaximumGpuStereoQueueCapacity = 16;

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
  // Bounds one appsink read, including repeated poll wakeups.
  std::uint64_t read_timeout_ns = 30'000'000'000ULL;
  // When set, decoded frame indices are derived from presentation stream time
  // instead of a synthetic pull counter. Calibration uses this to detect
  // dropped AUs.
  std::optional<std::uint32_t> indexed_fps_numerator;
  std::optional<std::uint32_t> indexed_fps_denominator;
  // Number of consecutive access units that share each presentation timestamp.
  // The probe proves this value is constant before indexed calibration uses it.
  std::uint32_t indexed_timestamp_multiplicity = 1;
  // Presentation stream time of absolute frame zero, as returned by the GPU
  // video probe. Required for an indexed seek so nonzero media timelines are
  // not relabeled as though they began at zero.
  std::optional<std::uint64_t> indexed_stream_time_origin_ns;
  // Optional first frame for an accurate, bounded GStreamer seek before
  // decoding starts. Requires indexed cadence so emitted indices retain their
  // absolute stream positions after the seek.
  std::optional<std::uint64_t> start_frame_index;
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
  // Clockwise display rotation carried by the selected video stream.
  std::uint16_t rotation_degrees = 0;
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
  /// Idempotently interrupts a blocked read and prevents further decoding.
  virtual void request_stop() noexcept = 0;
  /// Accurately seeks an indexed source without rebuilding its decode pipeline.
  virtual void seek_to_frame(std::uint64_t frame_index);
};

/// Failure while loading or consuming a GPU-resident GStreamer decode stream.
class GpuDecodeError : public std::runtime_error {
public:
  explicit GpuDecodeError(std::string message) : std::runtime_error(std::move(message)) {}
};

/// Side of a persistent stereo decode failure.
enum class GpuDecodeSide {
  Left,
  Right,
};

/// Failure from one side of a persistent stereo decode session.
class GpuStereoDecodeError : public GpuDecodeError {
public:
  GpuStereoDecodeError(GpuDecodeSide side, std::string message);

  /// Returns the source that failed.
  [[nodiscard]] GpuDecodeSide side() const noexcept { return side_; }

private:
  GpuDecodeSide side_;
};

/// Bounds and temporal alignment for a persistent stereo decode session.
struct GpuStereoDecodeConfig {
  // Positive offsets pair left N with right N + offset; negative offsets pair
  // left N + abs(offset) with right N.
  std::int64_t sync_offset = 0;
  // Maximum number of retained decoder-buffer owners in each side's queue.
  std::uint32_t queue_capacity = 4;
};

/// Terminal or frame-producing state returned by a stereo session read.
enum class GpuStereoDecodeStatus {
  FramePair,
  EndOfStream,
  Stopped,
};

/// GPU-resident frames aligned for one stereo render operation.
struct GpuDecodedFramePair {
  GpuDecodedFrame left;
  GpuDecodedFrame right;
};

/// Result of one persistent stereo session read.
struct GpuStereoDecodeReadResult {
  GpuStereoDecodeStatus status = GpuStereoDecodeStatus::EndOfStream;
  std::optional<GpuDecodedFramePair> frames;
};

/// Concurrently decodes and aligns two persistent GPU-resident file sources.
class GpuStereoDecodeSession {
public:
  GpuStereoDecodeSession(std::unique_ptr<GpuFileDecodeSource> left,
                         std::unique_ptr<GpuFileDecodeSource> right,
                         GpuStereoDecodeConfig config = {});
  ~GpuStereoDecodeSession();

  GpuStereoDecodeSession(const GpuStereoDecodeSession&) = delete;
  GpuStereoDecodeSession& operator=(const GpuStereoDecodeSession&) = delete;

  /// Returns the next aligned pair, EOS, or the explicit-stop state.
  [[nodiscard]] GpuStereoDecodeReadResult read();
  /// Idempotently stops both sources and wakes blocked reads.
  void request_stop() noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Move-only owner of an NVMM CUDA mapping and its validated NV12 render view.
class CudaNv12FrameLease final {
public:
  CudaNv12FrameLease(CudaNv12FrameLease&&) noexcept = default;
  CudaNv12FrameLease& operator=(CudaNv12FrameLease&&) noexcept = default;
  CudaNv12FrameLease(const CudaNv12FrameLease&) = delete;
  CudaNv12FrameLease& operator=(const CudaNv12FrameLease&) = delete;

  /// Driver-validated NV12 view usable by CUDA renderers while this lease remains alive.
  [[nodiscard]] const core::CudaNv12FrameView& view() const noexcept { return view_; }
  /// Underlying mapping metadata and retained decoder/runtime owners.
  [[nodiscard]] const NvmmCudaFrame& mapping() const noexcept { return mapping_; }

private:
  CudaNv12FrameLease(NvmmCudaFrame mapping, core::CudaNv12FrameView view)
      : mapping_(std::move(mapping)), view_(std::move(view)) {}

  NvmmCudaFrame mapping_;
  core::CudaNv12FrameView view_;

  friend CudaNv12FrameLease map_gpu_decoded_frame_to_cuda_lease(const GpuDecodedFrame& frame);
};

[[nodiscard]] std::string_view gpu_decode_codec_name(GpuDecodeCodec codec);
[[nodiscard]] std::string_view gpu_decode_container_demuxer(GpuDecodeContainer container);
[[nodiscard]] GpuDecodeCodec gpu_decode_codec_for_path(std::string_view path);
[[nodiscard]] bool gpu_decode_path_is_elementary_stream(std::string_view path);
[[nodiscard]] std::optional<GpuDecodeContainer>
gpu_decode_container_for_path(std::string_view path);
[[nodiscard]] std::optional<std::string>
validate_gpu_file_decode_config(const GpuFileDecodeConfig& config);
[[nodiscard]] std::optional<std::string>
validate_gpu_stereo_decode_config(const GpuStereoDecodeConfig& config);
[[nodiscard]] std::optional<std::string> validate_gpu_decoded_frame(const GpuDecodedFrame& frame);
[[nodiscard]] NvmmCudaFrame map_gpu_decoded_frame_to_cuda(const GpuDecodedFrame& frame);
/// Maps one decoded frame and derives a render view exclusively from CUDA-verified provenance.
[[nodiscard]] CudaNv12FrameLease map_gpu_decoded_frame_to_cuda_lease(const GpuDecodedFrame& frame);
[[nodiscard]] std::string
build_gstreamer_gpu_file_decode_pipeline(const GpuFileDecodeConfig& config);
[[nodiscard]] GpuDecodeReadResult make_gpu_decode_eos();
[[nodiscard]] GpuDecodeReadResult make_gpu_decode_frame(GpuDecodedFrame frame);
/// Opens an NVDEC/NVMM appsink source using the selected DeepStream surface ABI.
[[nodiscard]] std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_file_decode_source(GpuFileDecodeConfig config, NvbufSurfaceAbi abi);
/// Opens an NVDEC/NVMM source bound to the retained surface runtime.
[[nodiscard]] std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_file_decode_source(GpuFileDecodeConfig config,
                                      std::shared_ptr<const NvbufSurfaceRuntime> runtime);

} // namespace reco::io
