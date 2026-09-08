#pragma once

#include "reco/core/cuda_frame.hpp"
#include "reco/io/audio_passthrough.hpp"
#include "reco/io/nvmm.hpp"
#include "reco/io/output.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace reco::io {

// Current NVENC generations may retain seven input surfaces before releasing the first.
inline constexpr std::uint32_t kMinimumGpuEncodePoolCapacity = 8;
inline constexpr std::uint32_t kMaximumGpuEncodePoolCapacity = 16;

/// Fixed video and lifecycle settings for one GPU-resident encoder session.
struct GpuEncodeConfig {
  /// Path written by GStreamer when `output_descriptor` is not supplied.
  std::string output_path;
  /// Optional borrowed seekable descriptor written by `fdsink`; exactly one output is required.
  std::optional<int> output_descriptor;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_numerator = 0;
  std::uint32_t fps_denominator = 0;
  Codec codec = Codec::H264;
  Quality quality = Quality::Balanced;
  Format format = Format::Mp4;
  /// Optional exact NVIDIA GStreamer encoder factory. Software factories are rejected.
  std::optional<std::string> encoder;
  /// Optional parser-negotiated compressed audio caps for packet passthrough.
  std::optional<std::string> audio_caps;
  /// Optional 0-100 quality control, translated to a bounded hardware bitrate.
  std::optional<std::uint8_t> quality_value;
  std::uint32_t device_ordinal = 0;
  std::uint32_t pool_capacity = 8;
  std::chrono::milliseconds acquire_timeout = std::chrono::seconds(30);
  std::chrono::milliseconds startup_timeout = std::chrono::seconds(10);
  std::chrono::milliseconds finalize_timeout = std::chrono::seconds(30);
};

/// Non-throwing observability for output-pool residency and backpressure tests.
class GpuEncodeTraceSink {
public:
  virtual ~GpuEncodeTraceSink() = default;
  virtual void surface_allocated() noexcept {}
  virtual void surface_acquired() noexcept {}
  virtual void surface_submitted() noexcept {}
  virtual void surface_released() noexcept {}
};

/// Failure while allocating, feeding, or finalizing a GPU encoder pipeline.
class GpuEncodeError : public std::runtime_error {
public:
  explicit GpuEncodeError(std::string message) : std::runtime_error(std::move(message)) {}
};

/// Exclusive writable lease for one bounded NVMM output surface.
class GpuEncodeFrameLease final {
public:
  GpuEncodeFrameLease(GpuEncodeFrameLease&&) noexcept;
  GpuEncodeFrameLease& operator=(GpuEncodeFrameLease&&) noexcept;
  GpuEncodeFrameLease(const GpuEncodeFrameLease&) = delete;
  GpuEncodeFrameLease& operator=(const GpuEncodeFrameLease&) = delete;
  ~GpuEncodeFrameLease();

  /// CUDA NV12 view valid until the lease is submitted or destroyed.
  [[nodiscard]] const core::CudaNv12FrameView& view() const;
  [[nodiscard]] explicit operator bool() const noexcept;

private:
  struct State;
  explicit GpuEncodeFrameLease(std::unique_ptr<State> state);

  std::unique_ptr<State> state_;

  friend class GpuVideoEncodeSession;
};

/// Bounded `memory:NVMM` appsrc session feeding only NVIDIA hardware encoders.
class GpuVideoEncodeSession final {
public:
  /// Opens the production GStreamer and DeepStream runtime bindings.
  [[nodiscard]] static GpuVideoEncodeSession
  open(GpuEncodeConfig config, std::shared_ptr<GpuEncodeTraceSink> trace_sink = {});
  /// Opens against an already retained NvBufSurface runtime.
  [[nodiscard]] static GpuVideoEncodeSession
  open(GpuEncodeConfig config, std::shared_ptr<const NvbufSurfaceRuntime> runtime,
       std::shared_ptr<GpuEncodeTraceSink> trace_sink = {});

  GpuVideoEncodeSession(const GpuVideoEncodeSession&) = delete;
  GpuVideoEncodeSession& operator=(const GpuVideoEncodeSession&) = delete;
  GpuVideoEncodeSession(GpuVideoEncodeSession&&) noexcept;
  GpuVideoEncodeSession& operator=(GpuVideoEncodeSession&&) noexcept;
  ~GpuVideoEncodeSession();

  /// Waits for a free pool surface, or throws on timeout or a sticky pipeline error.
  [[nodiscard]] GpuEncodeFrameLease acquire_frame();
  /// Transfers a rendered frame to appsrc. GStreamer retains the surface until downstream release.
  void submit_frame(GpuEncodeFrameLease&& frame, std::uint64_t pts_ns, std::uint64_t duration_ns);
  /// Copies one bounded compressed audio packet to the muxer without decoding or re-encoding it.
  void submit_audio_packet(CompressedAudioPacket packet);
  /// Sends EOS and waits for the encoder and muxer to finalize successfully.
  void finish();
  /// Idempotently stops the pipeline without publishing a partial output.
  void abort() noexcept;

  [[nodiscard]] const GpuEncodeConfig& config() const;
  [[nodiscard]] std::string_view pipeline() const;

private:
  struct Impl;
  explicit GpuVideoEncodeSession(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::optional<std::string> validate_gpu_encode_config(const GpuEncodeConfig& config);
[[nodiscard]] std::string_view gstreamer_hardware_encoder_factory(Codec codec);
[[nodiscard]] std::string build_gstreamer_gpu_encode_pipeline(const GpuEncodeConfig& config);
/// Verifies one parser-selected compressed video access unit in an isolated worker.
///
/// The worker uses only a demuxer, parser, compressed caps filters, and an appsink. It never
/// instantiates a video decoder or materializes pixels. H.264, HEVC, and AV1 in MP4, fragmented
/// MP4, Matroska, QuickTime, and FLV containers are supported.
void verify_muxed_gpu_video_output(const std::filesystem::path& path, Codec codec, Format format,
                                   const std::filesystem::path& probe_worker,
                                   std::chrono::milliseconds timeout = std::chrono::seconds(10));

} // namespace reco::io
