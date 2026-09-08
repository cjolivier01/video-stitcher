#pragma once

#include "reco/io/stable_media_file.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace reco::io {

/// One video segment whose compressed audio may be passed through.
struct AudioPassthroughSegment {
  /// User-facing input path used for explicit container selection and diagnostics.
  std::string path;
  /// Optional retained authority. An independent cursor is acquired only while this segment is
  /// active, and a pathname substitute is never consumed.
  std::shared_ptr<const StableMediaFile> stable_source;
  /// Probed video duration used to preserve the joined video timeline when audio is absent.
  std::uint64_t video_duration_ns = 0;
};

/// Bounded configuration for compressed audio demuxed without decoding or re-encoding.
struct AudioPassthroughConfig {
  /// Ordered recording segments and their authoritative video durations.
  std::vector<AudioPassthroughSegment> segments;
  /// Source timestamp trimmed from the first segment and rebased to output timestamp zero.
  std::uint64_t start_time_ns = 0;
  /// Maximum wait for one compressed packet or terminal pipeline state.
  std::chrono::milliseconds read_timeout = std::chrono::seconds(30);
};

/// One parsed compressed audio access unit and its nanosecond timeline metadata.
struct CompressedAudioPacket {
  std::vector<std::byte> bytes;
  std::optional<std::uint64_t> pts_ns;
  std::optional<std::uint64_t> dts_ns;
  std::uint64_t duration_ns = 0;
  std::uint32_t flags = 0;

  /// Earliest available packet timestamp, used for bounded video-duration clipping.
  [[nodiscard]] std::optional<std::uint64_t> timestamp_ns() const noexcept;
};

enum class AudioPassthroughStatus {
  Packet,
  EndOfStream,
};

struct AudioPassthroughReadResult {
  AudioPassthroughStatus status = AudioPassthroughStatus::EndOfStream;
  std::optional<CompressedAudioPacket> packet;
};

/// Failure while demuxing compressed audio packets for stream-copy muxing.
class AudioPassthroughError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// GStreamer compressed-audio reader. Video pads are never decoded or materialized.
class AudioPassthroughSource final {
public:
  [[nodiscard]] static AudioPassthroughSource open(AudioPassthroughConfig config);

  AudioPassthroughSource(const AudioPassthroughSource&) = delete;
  AudioPassthroughSource& operator=(const AudioPassthroughSource&) = delete;
  AudioPassthroughSource(AudioPassthroughSource&&) noexcept;
  AudioPassthroughSource& operator=(AudioPassthroughSource&&) noexcept;
  ~AudioPassthroughSource();

  /// Parsed caps copied verbatim to the encoder audio appsrc, or empty when no segment has audio.
  [[nodiscard]] std::optional<std::string> caps() const;
  /// Returns the next rebased packet, advancing across compatible recording segments.
  [[nodiscard]] AudioPassthroughReadResult read();
  /// Idempotently interrupts native reads and releases segment resources.
  void request_stop() noexcept;

private:
  struct Impl;
  explicit AudioPassthroughSource(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::optional<std::string>
validate_audio_passthrough_config(const AudioPassthroughConfig& config);
[[nodiscard]] std::string build_gstreamer_audio_passthrough_pipeline(std::string_view path);

} // namespace reco::io
