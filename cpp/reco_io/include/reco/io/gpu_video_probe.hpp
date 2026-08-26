#pragma once

#include "reco/io/gpu_decode.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace reco::io {

/// Parser-derived metadata used to select GPU calibration frames.
struct GpuVideoProbe {
  /// Coded video width in pixels.
  std::uint32_t width = 0;
  /// Coded video height in pixels.
  std::uint32_t height = 0;
  /// Exact frame-rate numerator reported by the parser.
  std::uint32_t fps_numerator = 0;
  /// Exact frame-rate denominator reported by the parser.
  std::uint32_t fps_denominator = 0;
  /// Frame rate as a floating-point value for Rust-compatible selection math.
  double fps = 0.0;
  /// Parsed or explicitly estimated duration in nanoseconds.
  std::uint64_t duration_ns = 0;
  /// Truncated `duration * fps` frame-count estimate.
  std::uint64_t total_frames = 0;
  /// Whether `duration_ns` uses the Rust-compatible 60-second fallback.
  bool duration_is_estimated = false;
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
/// access-unit timestamps. The timeout must be between one second and one hour,
/// inclusive.
[[nodiscard]] GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config,
                                            std::uint64_t timeout_ns);

} // namespace reco::io
