#pragma once

#include "reco/io/gpu_decode.hpp"

#include <cstdint>
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
  /// Whether `duration_ns` is a fallback or an uncorrelated container estimate.
  bool duration_is_estimated = false;
  /// Whether `total_frames` comes from bounded duration/timestamp correlation
  /// instead of an EOS-proven compressed access-unit count.
  bool total_frames_is_estimated = false;
};

/// Failure while probing a GPU-decodable video without materializing frames.
class GpuVideoProbeError : public std::runtime_error {
public:
  explicit GpuVideoProbeError(std::string message) : std::runtime_error(std::move(message)) {}
};

/// Uses the production GStreamer demux/parser topology to probe a local input.
///
/// The pipeline stops before NVDEC and never produces a decoded frame. The
/// selected stream's duration is correlated with bounded seeks over
/// compressed access-unit timestamps. The parser scans a bounded AU prefix to
/// EOS when practical; longer streams expose an explicitly estimated count.
/// The timeout must be between one second and one hour, inclusive.
[[nodiscard]] GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config,
                                            std::uint64_t timeout_ns);

} // namespace reco::io
