#include "reco/core/source.hpp"
#include "reco/io/output.hpp"
#include "reco/io/stacked_video.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

using namespace reco::io;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

reco::core::YuvFrame fill(std::uint32_t w, std::uint32_t h, std::uint8_t y, std::uint8_t u,
                          std::uint8_t v) {
  reco::core::YuvFrame frame;
  frame.width = w;
  frame.height = h;
  frame.timestamp_us = 12345;
  frame.y.assign(static_cast<std::size_t>(w) * h, y);
  frame.u.assign(static_cast<std::size_t>(w / 2) * (h / 2), u);
  frame.v.assign(static_cast<std::size_t>(w / 2) * (h / 2), v);
  return frame;
}

GridLayout must_layout(std::variant<GridLayout, std::string_view> result) {
  if (auto* layout = std::get_if<GridLayout>(&result)) {
    return *layout;
  }
  std::cerr << "FAIL: expected valid layout, got " << std::get<std::string_view>(result) << '\n';
  ++failures;
  return std::get<GridLayout>(GridLayout::grid(2, 2, 1, 1));
}

void output_parsing_matches_rust_aliases() {
  expect_true(parse_codec("h264") == Codec::H264, "h264 codec");
  expect_true(parse_codec("AVC") == Codec::H264, "avc codec alias");
  expect_true(parse_codec("h.265") == Codec::HEVC, "hevc codec alias");
  expect_true(parse_codec("libaom-av1") == Codec::AV1, "av1 codec alias");
  expect_true(!parse_codec("vp9").has_value(), "unknown codec rejected");
  expect_eq(codec_name(Codec::HEVC), std::string_view("hevc"), "codec display");

  expect_true(parse_quality("low") == Quality::Fast, "quality low alias");
  expect_true(parse_quality("medium") == Quality::Balanced, "quality medium alias");
  expect_true(parse_quality("slow") == Quality::High, "quality slow alias");
  expect_true(!parse_quality("lossless").has_value(), "unknown quality rejected");
  expect_true(Quality{} == Quality::Balanced, "default quality");
  expect_true(std::holds_alternative<Quality>(Bitrate{}), "direct default bitrate quality");
  expect_true(std::get<Quality>(Bitrate{}) == Quality::Balanced, "direct default bitrate");
  expect_true(std::holds_alternative<Quality>(default_bitrate()), "default bitrate quality");
  expect_true(std::get<Quality>(default_bitrate()) == Quality::Balanced, "default bitrate");
}

void format_detection_matches_rust_policy() {
  expect_true(parse_format("mp4") == Format::Mp4, "mp4 format");
  expect_true(parse_format("fmp4") == Format::Mp4Fragmented, "fragmented mp4 alias");
  expect_true(parse_format("matroska") == Format::Mkv, "mkv alias");
  expect_true(parse_format("quicktime") == Format::Mov, "mov alias");
  expect_true(!parse_format("webm").has_value(), "unknown format rejected");

  expect_true(format_for_output("rtmp://example/live") == Format::Flv, "rtmp is flv");
  expect_true(format_for_output("rtmps://example/live") == Format::Flv, "rtmps is flv");
  expect_true(format_for_output("srt://example:9000") == Format::Mkv, "srt is mkv");
  expect_true(format_for_output("clip.MKV") == Format::Mkv, "mkv extension");
  expect_true(format_for_output("clip.mov") == Format::Mov, "mov extension");
  expect_true(format_for_output("clip.flv") == Format::Flv, "flv extension");
  expect_true(format_for_output("clip.unknown") == Format::Mp4, "unknown extension defaults mp4");
  expect_true(is_streaming_format(Format::Flv), "flv streams");
  expect_true(!is_streaming_format(Format::Mkv), "mkv does not stream in Rust policy");
  expect_eq(format_name(Format::Mp4Fragmented), std::string_view("mp4-fragmented"), "format name");
  expect_true(std::holds_alternative<CopyAudioFrom>(default_audio_mode()), "default audio copy");
  expect_eq(std::get<CopyAudioFrom>(default_audio_mode()).input_index, 0U, "default audio input");
}

void layout_validation_matches_rust_guards() {
  const auto layout = must_layout(GridLayout::vstack(1920, 1080, 2));
  expect_eq(layout.packed_width(), 1920U, "vstack width");
  expect_eq(layout.packed_height(), 2160U, "vstack height");
  expect_eq(layout.capacity(), 2U, "vstack capacity");

  const auto hstack = must_layout(GridLayout::hstack(640, 480, 3));
  expect_eq(hstack.packed_width(), 1920U, "hstack width");
  expect_eq(hstack.packed_height(), 480U, "hstack height");
  expect_eq(hstack.capacity(), 3U, "hstack capacity");

  expect_true(std::holds_alternative<std::string_view>(GridLayout::vstack(1921, 1080, 2)),
              "odd width rejected");
  expect_true(std::holds_alternative<std::string_view>(GridLayout::vstack(1920, 1079, 2)),
              "odd height rejected");
  expect_true(std::holds_alternative<std::string_view>(GridLayout::vstack(0, 1080, 2)),
              "zero width rejected");
  expect_true(std::holds_alternative<std::string_view>(GridLayout::vstack(1920, 1080, 0)),
              "zero tiles rejected");
}

void pack_unpack_round_trips_identity() {
  const auto layout = must_layout(GridLayout::vstack(64, 64, 2));
  const auto a = fill(64, 64, 100, 120, 140);
  const auto b = fill(64, 64, 200, 60, 90);

  auto packed_result = pack_yuv420p(layout, {&a, &b});
  expect_true(std::holds_alternative<reco::core::YuvFrame>(packed_result), "pack succeeded");
  const auto packed = std::get<reco::core::YuvFrame>(packed_result);
  expect_eq(packed.width, 64U, "packed width");
  expect_eq(packed.height, 128U, "packed height");

  auto unpacked_result = unpack_yuv420p(layout, packed);
  expect_true(std::holds_alternative<std::vector<reco::core::YuvFrame>>(unpacked_result),
              "unpack succeeded");
  const auto unpacked = std::get<std::vector<reco::core::YuvFrame>>(unpacked_result);
  expect_eq(unpacked.size(), 2U, "unpacked count");
  expect_true(unpacked[0].y == a.y, "first y plane");
  expect_true(unpacked[0].u == a.u, "first u plane");
  expect_true(unpacked[0].v == a.v, "first v plane");
  expect_true(unpacked[1].y == b.y, "second y plane");
  expect_true(unpacked[1].u == b.u, "second u plane");
  expect_true(unpacked[1].v == b.v, "second v plane");
}

void empty_tiles_get_grey_fill() {
  const auto layout = must_layout(GridLayout::vstack(64, 64, 3));
  const auto a = fill(64, 64, 10, 20, 30);
  const auto b = fill(64, 64, 40, 50, 60);

  const auto packed = std::get<reco::core::YuvFrame>(pack_yuv420p(layout, {&a, &b, nullptr}));
  const auto unpacked = std::get<std::vector<reco::core::YuvFrame>>(unpack_yuv420p(layout, packed));
  expect_eq(unpacked.size(), 3U, "unpacked count with empty");
  expect_true(std::all_of(unpacked[2].y.begin(), unpacked[2].y.end(),
                          [](std::uint8_t value) { return value == 128; }),
              "empty y plane grey");
  expect_true(std::all_of(unpacked[2].u.begin(), unpacked[2].u.end(),
                          [](std::uint8_t value) { return value == 128; }),
              "empty u plane grey");
  expect_true(std::all_of(unpacked[2].v.begin(), unpacked[2].v.end(),
                          [](std::uint8_t value) { return value == 128; }),
              "empty v plane grey");
}

void stack_errors_match_rust_cases() {
  const auto layout = must_layout(GridLayout::vstack(64, 64, 2));
  auto a = fill(64, 64, 0, 0, 0);
  const auto wrong = fill(128, 64, 0, 0, 0);

  auto too_many = std::get<StackError>(pack_yuv420p(layout, {&a, &a, &a}));
  expect_true(std::holds_alternative<TooManyTiles>(too_many), "too many tiles");
  expect_eq(std::get<TooManyTiles>(too_many).capacity, 2U, "too many capacity");
  expect_eq(std::get<TooManyTiles>(too_many).got, 3U, "too many got");

  auto bad_dims = std::get<StackError>(pack_yuv420p(layout, {&a, &wrong}));
  expect_true(std::holds_alternative<TileDimensionMismatch>(bad_dims), "tile dimension mismatch");
  expect_eq(std::get<TileDimensionMismatch>(bad_dims).index, 1U, "dimension mismatch index");

  a.u.pop_back();
  auto bad_plane = std::get<StackError>(pack_yuv420p(layout, {&a}));
  expect_true(std::holds_alternative<PlaneSizeMismatch>(bad_plane), "plane size mismatch");
  expect_eq(std::get<PlaneSizeMismatch>(bad_plane).plane, std::string_view("u"),
            "plane mismatch label");

  const auto small = fill(64, 64, 0, 0, 0);
  auto bad_packed = std::get<StackError>(unpack_yuv420p(layout, small));
  expect_true(std::holds_alternative<PackedDimensionMismatch>(bad_packed),
              "packed dimension mismatch");
}

void grid_3x3_round_trips_nine_tiles() {
  const auto layout = must_layout(GridLayout::grid(32, 32, 3, 3));
  std::vector<reco::core::YuvFrame> tiles;
  std::vector<const reco::core::YuvFrame*> slots;
  tiles.reserve(9);
  slots.reserve(9);
  for (std::uint8_t i = 0; i < 9; ++i) {
    tiles.push_back(fill(32, 32, static_cast<std::uint8_t>(i * 20), 128, 128));
    slots.push_back(&tiles.back());
  }

  const auto packed = std::get<reco::core::YuvFrame>(pack_yuv420p(layout, slots));
  expect_eq(packed.width, 96U, "3x3 packed width");
  expect_eq(packed.height, 96U, "3x3 packed height");
  const auto unpacked = std::get<std::vector<reco::core::YuvFrame>>(unpack_yuv420p(layout, packed));
  expect_eq(unpacked.size(), 9U, "3x3 unpacked count");
  for (std::size_t i = 0; i < unpacked.size(); ++i) {
    expect_eq(unpacked[i].y[0], static_cast<std::uint8_t>(i * 20), "3x3 tile identity");
  }
}

void timestamp_follows_first_nonempty_tile() {
  const auto layout = must_layout(GridLayout::vstack(64, 64, 3));
  auto a = fill(64, 64, 0, 0, 0);
  auto b = fill(64, 64, 0, 0, 0);
  a.timestamp_us = 1000;
  b.timestamp_us = 2000;
  const auto packed = std::get<reco::core::YuvFrame>(pack_yuv420p(layout, {nullptr, &a, &b}));
  expect_eq(packed.timestamp_us, 1000, "first non-empty timestamp");
}

} // namespace

int main() {
  static_assert(std::is_copy_constructible_v<StackError>, "StackError is copyable");
  output_parsing_matches_rust_aliases();
  format_detection_matches_rust_policy();
  layout_validation_matches_rust_guards();
  pack_unpack_round_trips_identity();
  empty_tiles_get_grey_fill();
  stack_errors_match_rust_cases();
  grid_3x3_round_trips_nine_tiles();
  timestamp_follows_first_nonempty_tile();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
