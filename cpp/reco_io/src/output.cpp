#include "reco/io/output.hpp"
#include "reco/core/path.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace reco::io {
namespace {

std::string lowercase(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

} // namespace

std::optional<Codec> parse_codec(std::string_view value) {
  const auto s = lowercase(value);
  if (s == "h264" || s == "avc" || s == "h.264" || s == "x264") {
    return Codec::H264;
  }
  if (s == "hevc" || s == "h265" || s == "h.265" || s == "x265") {
    return Codec::HEVC;
  }
  if (s == "av1" || s == "svt-av1" || s == "libaom-av1") {
    return Codec::AV1;
  }
  return std::nullopt;
}

std::string_view codec_name(Codec codec) {
  switch (codec) {
  case Codec::H264:
    return "h264";
  case Codec::HEVC:
    return "hevc";
  case Codec::AV1:
    return "av1";
  }
  return "h264";
}

std::optional<Quality> parse_quality(std::string_view value) {
  const auto s = lowercase(value);
  if (s == "fast" || s == "low") {
    return Quality::Fast;
  }
  if (s == "balanced" || s == "medium") {
    return Quality::Balanced;
  }
  if (s == "high" || s == "slow") {
    return Quality::High;
  }
  return std::nullopt;
}

std::string_view quality_name(Quality quality) {
  switch (quality) {
  case Quality::Fast:
    return "fast";
  case Quality::Balanced:
    return "balanced";
  case Quality::High:
    return "high";
  }
  return "balanced";
}

Bitrate default_bitrate() { return Quality::Balanced; }

std::optional<Format> parse_format(std::string_view value) {
  const auto s = lowercase(value);
  if (s == "mp4") {
    return Format::Mp4;
  }
  if (s == "fmp4" || s == "mp4-fragmented" || s == "mp4_fragmented") {
    return Format::Mp4Fragmented;
  }
  if (s == "mkv" || s == "matroska") {
    return Format::Mkv;
  }
  if (s == "mov" || s == "quicktime") {
    return Format::Mov;
  }
  if (s == "flv") {
    return Format::Flv;
  }
  return std::nullopt;
}

Format format_for_output(std::string_view path_or_url) {
  if (starts_with(path_or_url, "rtmp://") || starts_with(path_or_url, "rtmps://")) {
    return Format::Flv;
  }
  if (starts_with(path_or_url, "srt://")) {
    return Format::Mkv;
  }

  const auto ext = lowercase(reco::core::path_from_utf8(path_or_url).extension().string());
  if (ext == ".mkv") {
    return Format::Mkv;
  }
  if (ext == ".mov") {
    return Format::Mov;
  }
  if (ext == ".flv") {
    return Format::Flv;
  }
  return Format::Mp4;
}

bool is_streaming_format(Format format) { return format == Format::Flv; }

std::string_view format_name(Format format) {
  switch (format) {
  case Format::Mp4:
    return "mp4";
  case Format::Mp4Fragmented:
    return "mp4-fragmented";
  case Format::Mkv:
    return "mkv";
  case Format::Mov:
    return "mov";
  case Format::Flv:
    return "flv";
  }
  return "mp4";
}

AudioMode default_audio_mode() { return CopyAudioFrom{0}; }

} // namespace reco::io
