#pragma once

#include "reco/io/gpu_decode.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace reco::io {

/// Parser-derived metadata used to select GPU calibration frames.
struct GpuVideoProbe {
  /// Parser-visible video width in pixels.
  std::uint32_t width = 0;
  /// Parser-visible video height in pixels.
  std::uint32_t height = 0;
  /// Frame-rate numerator from constant parser timing, parser caps, or the
  /// explicit 30 fps fallback used when both are unavailable.
  std::uint32_t fps_numerator = 0;
  /// Frame-rate denominator matching `fps_numerator`.
  std::uint32_t fps_denominator = 0;
  /// Frame rate as a floating-point value for Rust-compatible selection math.
  double fps = 0.0;
  /// Parsed or explicitly estimated duration in nanoseconds.
  std::uint64_t duration_ns = 0;
  /// EOS-proven compressed access-unit count when available, otherwise a
  /// bounded duration/timestamp estimate.
  std::uint64_t total_frames = 0;
  /// Whether `duration_ns` uses fallback, inferred terminal timing, or bounded
  /// correlation rather than an EOS-proven final AU duration.
  bool duration_is_estimated = false;
  /// Whether `total_frames` comes from bounded duration/timestamp correlation
  /// instead of an EOS-proven compressed access-unit count.
  bool total_frames_is_estimated = false;
  /// True only when every selected-stream access unit was observed through EOS
  /// with unchanged codec, geometry, and timing caps.
  bool selected_stream_caps_verified = false;
  /// True only when every selected-stream presentation timestamp was checked
  /// against one constant cadence. Callers must require this before converting
  /// calibration timestamps to frame indices.
  bool indexed_sampling_cadence_verified = false;
};

/// Failure while probing a GPU-decodable video without materializing frames.
class GpuVideoProbeError : public std::runtime_error {
public:
  explicit GpuVideoProbeError(std::string message) : std::runtime_error(std::move(message)) {}
};

/// Uses the production GStreamer demux/parser topology to probe a local input.
///
/// The pipeline stops before NVDEC and never produces a decoded frame. The
/// selected stream's duration is correlated with bounded seeks over compressed
/// access-unit timestamps. Short dense timestamp streams are scanned to EOS and
/// report an exact AU count. Longer, sparse, or incomplete timing stays bounded
/// and exposes explicit count, caps, and cadence verification state.
/// Matroska compressed-buffer durations preserve the exact selected-track
/// rational when GStreamer normalizes negotiated caps.
/// Dense timestamp evidence that proves variable frame rate is rejected
/// because indexed calibration sampling requires a constant cadence.
/// Sparse or incomplete timestamp evidence is returned with
/// `indexed_sampling_cadence_verified == false` and must not drive frame-index
/// selection without an additional cadence proof at the sampling boundary.
/// `worker_path` must be the absolute path of the deployed
/// `reco_video_probe_worker` executable.
/// The worker is isolated so a blocked native multimedia call can be killed
/// with its process-owned resources. Once the operating-system process-creation
/// call returns, the timeout covers guardian/worker loading, IPC, probing,
/// termination, and reaping, with a bounded cleanup reserve. Kernel process
/// creation itself is synchronous and cannot be preempted by this API, so
/// `worker_path` should name a trusted local executable. The timeout must be
/// between one second and one hour, inclusive. Each admitted worker reserves
/// its 512 MiB allowance against a 2 GiB process-wide aggregate; calls beyond
/// that budget fail before another process is launched.
[[nodiscard]] GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config,
                                            const std::filesystem::path& worker_path,
                                            std::uint64_t timeout_ns);

} // namespace reco::io
