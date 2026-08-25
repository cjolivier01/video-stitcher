#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace reco::io {

enum class Codec {
  H264,
  HEVC,
  AV1,
};

[[nodiscard]] std::optional<Codec> parse_codec(std::string_view value);
[[nodiscard]] std::string_view codec_name(Codec codec);

enum class Quality {
  Balanced,
  Fast,
  High,
};

[[nodiscard]] std::optional<Quality> parse_quality(std::string_view value);
[[nodiscard]] std::string_view quality_name(Quality quality);

struct CrfBitrate {
  std::uint8_t value = 0;
};

using Bitrate = std::variant<Quality, CrfBitrate>;

[[nodiscard]] Bitrate default_bitrate();

enum class Format {
  Mp4,
  Mp4Fragmented,
  Mkv,
  Mov,
  Flv,
};

[[nodiscard]] std::optional<Format> parse_format(std::string_view value);
[[nodiscard]] Format format_for_output(std::string_view path_or_url);
[[nodiscard]] bool is_streaming_format(Format format);
[[nodiscard]] std::string_view format_name(Format format);

struct CopyAudioFrom {
  std::size_t input_index = 0;
};

struct DisableAudio {};

using AudioMode = std::variant<CopyAudioFrom, DisableAudio>;

[[nodiscard]] AudioMode default_audio_mode();

} // namespace reco::io
