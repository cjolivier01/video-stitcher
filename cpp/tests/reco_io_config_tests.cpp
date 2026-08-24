#include "reco/core/source.hpp"
#include "reco/io/jsonl_sink.hpp"
#include "reco/io/output.hpp"
#include "reco/io/settings.hpp"
#include "reco/io/stacked_video.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

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

struct DummySettings {
  bool flag = false;
  std::uint32_t count = 0;
  std::string name;
};

void to_json(nlohmann::json& json, const DummySettings& settings) {
  json = nlohmann::json{
      {"flag", settings.flag},
      {"count", settings.count},
      {"name", settings.name},
  };
}

void from_json(const nlohmann::json& json, DummySettings& settings) {
  settings.flag = json.value("flag", false);
  settings.count = json.value("count", 0U);
  settings.name = json.value("name", std::string{});
}

class ScopedConfigDir {
public:
  ScopedConfigDir() {
    path_ = std::filesystem::temp_directory_path() / "reco_io_cpp_settings_tests";
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
#if defined(_WIN32)
    _putenv_s("RECO_CONFIG_DIR", path_.string().c_str());
#else
    setenv("RECO_CONFIG_DIR", path_.string().c_str(), 1);
#endif
  }

  ~ScopedConfigDir() {
#if defined(_WIN32)
    _putenv_s("RECO_CONFIG_DIR", "");
#else
    unsetenv("RECO_CONFIG_DIR");
#endif
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

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

void settings_round_trip_and_namespace_guards_match_rust() {
  ScopedConfigDir config;
  const DummySettings expected{true, 42, "test"};
  const auto save_error = save_settings("round_trip_test", expected);
  expect_true(!save_error.has_value(), "settings save succeeds");

  auto loaded_result = load_settings<DummySettings>("round_trip_test");
  expect_true(std::holds_alternative<DummySettings>(loaded_result), "settings load succeeds");
  const auto loaded = std::get<DummySettings>(loaded_result);
  expect_true(loaded.flag == expected.flag, "settings flag roundtrip");
  expect_eq(loaded.count, expected.count, "settings count roundtrip");
  expect_true(loaded.name == expected.name, "settings name roundtrip");

  auto exists_before = settings_exists("missing_probe");
  expect_true(std::holds_alternative<bool>(exists_before), "missing exists query succeeds");
  expect_true(!std::get<bool>(exists_before), "missing settings returns false");
  expect_true(load_settings_or_default<DummySettings>("missing_probe").count == 0,
              "missing load defaults");
  auto missing_load = load_settings<DummySettings>("missing_probe");
  expect_true(std::holds_alternative<SettingsError>(missing_load), "missing load errors");
  if (auto* error = std::get_if<SettingsError>(&missing_load)) {
    expect_true(error->io_kind == std::errc::no_such_file_or_directory,
                "missing load preserves not found");
  }

  auto exists_after = settings_exists("round_trip_test");
  expect_true(std::holds_alternative<bool>(exists_after), "saved exists query succeeds");
  expect_true(std::get<bool>(exists_after), "saved settings exists");

  auto removed = delete_settings("round_trip_test");
  expect_true(std::holds_alternative<bool>(removed), "delete query succeeds");
  expect_true(std::get<bool>(removed), "delete removes file");
  auto removed_again = delete_settings("round_trip_test");
  expect_true(std::holds_alternative<bool>(removed_again), "second delete succeeds");
  expect_true(!std::get<bool>(removed_again), "second delete reports false");

  for (const std::string_view ns : {"has space", "../etc", "UPPER", ""}) {
    auto bad = settings_path(ns);
    expect_true(std::holds_alternative<SettingsError>(bad), "bad namespace rejected");
    if (auto* error = std::get_if<SettingsError>(&bad)) {
      expect_true(error->kind == SettingsErrorKind::BadNamespace, "bad namespace error kind");
    }
  }
  expect_true(settings_error_kind_name(SettingsErrorKind::Serialize) == "serialize",
              "settings error kind label");

  std::filesystem::create_directory(config.path() / "dir_probe.json");
  auto dir_load = load_settings<DummySettings>("dir_probe");
  expect_true(std::holds_alternative<SettingsError>(dir_load), "directory load errors");
  if (auto* error = std::get_if<SettingsError>(&dir_load)) {
    expect_true(error->io_kind == std::errc::is_a_directory, "directory load kind");
  }
  auto dir_delete = delete_settings("dir_probe");
  expect_true(std::holds_alternative<SettingsError>(dir_delete), "directory delete errors");
  if (auto* error = std::get_if<SettingsError>(&dir_delete)) {
    expect_true(error->io_kind == std::errc::is_a_directory, "directory delete kind");
  }
}

void recent_files_match_rust_mru_policy() {
  RecentFiles recent(4);
  recent.push("/a");
  recent.push("/b");
  recent.push("/c");
  recent.push("/b");
  expect_eq(recent.size(), 3U, "recent files dedup size");
  expect_true(recent.entries()[0] == std::filesystem::path("/b"), "dedup moves to front");
  expect_true(recent.entries()[1] == std::filesystem::path("/c"), "middle entry");
  expect_true(recent.entries()[2] == std::filesystem::path("/a"), "last entry");

  RecentFiles capped(2);
  capped.push("/1");
  capped.push("/2");
  capped.push("/3");
  expect_eq(capped.size(), 2U, "recent files cap");
  expect_true(capped.entries()[0] == std::filesystem::path("/3"), "cap newest");
  expect_true(capped.entries()[1] == std::filesystem::path("/2"), "cap second newest");

  capped.remove("/3");
  expect_eq(capped.size(), 1U, "recent remove");
  capped.clear();
  expect_true(capped.empty(), "recent clear");

  const nlohmann::json json = recent;
  const auto roundtrip = json.get<RecentFiles>();
  expect_eq(roundtrip.size(), 3U, "recent json roundtrip size");
  expect_true(roundtrip.entries()[0] == std::filesystem::path("/b"), "recent json order");
}

void jsonl_sink_writes_one_pipeline_event_per_line() {
  const auto path = std::filesystem::temp_directory_path() / "reco_io_cpp_events.jsonl";
  std::filesystem::remove(path);
  {
    JsonlSink sink(path);
    expect_true(sink.ok(), "jsonl sink opens");
    for (std::uint64_t i = 0; i < 5; ++i) {
      sink.emit(reco::core::PipelineEvent(
          reco::core::FrameStartEvent{.frame_index = i, .timestamp_ms = i * 16.6}));
    }
    sink.flush();
    expect_eq(sink.write_failures(), 0ULL, "jsonl sink no write failures");
  }

  std::ifstream input(path);
  std::string line;
  std::vector<std::string> lines;
  while (std::getline(input, line)) {
    lines.push_back(line);
  }
  expect_eq(lines.size(), 5U, "jsonl line count");
  for (std::size_t i = 0; i < lines.size(); ++i) {
    const auto parsed = nlohmann::json::parse(lines[i]);
    expect_true(parsed["kind"] == "frame_start", "jsonl kind");
    expect_true(parsed["frame_index"] == i, "jsonl frame index");
  }
  std::filesystem::remove(path);
}

void pipeline_event_json_serializes_core_event_vocabulary() {
  reco::core::FrameTimingMicros timing;
  timing.decode_us = 10;
  timing.total_us = 99;
  auto event = reco::core::PipelineEvent(reco::core::FrameCompleteEvent{
      .frame_index = 7,
      .timestamp_ms = 116.2,
      .timing = timing,
      .detection_count = 3,
      .active_tracks = 2,
      .ball_present = true,
  });
  const auto json = pipeline_event_to_json(event);
  expect_true(json["kind"] == "frame_complete", "frame complete kind");
  expect_true(json["frame_index"] == 7, "frame complete index");
  expect_true(json["timing"]["decode_us"] == 10, "frame complete timing optional");
  expect_true(json["timing"]["upload_us"].is_null(), "frame complete missing optional is null");
  expect_true(json["timing"]["total_us"] == 99, "frame complete total");
  expect_true(json["ball_present"] == true, "frame complete ball flag");

  reco::core::MappedDetection detection;
  detection.camera = reco::core::CameraId::Right;
  detection.class_id = 32;
  detection.confidence = 0.75F;
  detection.camera_center_x = 0.25F;
  detection.camera_center_y = 0.5F;
  detection.camera_size_x = 0.1F;
  detection.camera_size_y = 0.2F;
  detection.position = reco::core::ViewportPosition{.yaw = 0.3F, .pitch = -0.1F};
  detection.position->fov_degrees = std::nullopt;
  const auto detections_json = pipeline_event_to_json(reco::core::PipelineEvent(
      reco::core::DetectionsRawEvent{.frame_index = 8, .detections = {detection}}));
  expect_true(detections_json["kind"] == "detections_raw", "detections kind");
  expect_true(detections_json["detections"][0]["camera"] == "Right", "camera enum schema");
  expect_true(detections_json["detections"][0]["camera_center"].is_array(),
              "camera center tuple schema");
  expect_true(detections_json["detections"][0]["camera_size"].is_array(),
              "camera size tuple schema");
  expect_true(detections_json["detections"][0]["position"]["fov_degrees"].is_null(),
              "position optional fov schema");

  reco::core::TrackedEntity entity;
  entity.id = 2;
  entity.class_id = 1;
  entity.state = reco::core::TrackState::Coasting;
  entity.origin = reco::core::CameraId::Left;
  const auto world_json =
      pipeline_event_to_json(reco::core::PipelineEvent(reco::core::WorldStateEvent{
          .frame_index = 9, .timestamp_ms = 1.0, .players = {entity}, .ball = entity}));
  expect_true(world_json["players"][0]["state"] == "Coasting", "track state enum schema");
  expect_true(world_json["players"][0]["origin"] == "Left", "origin enum schema");
  expect_true(world_json["ball"]["state"] == "Coasting", "optional entity schema");

  auto pose = reco::core::ViewportPosition{.yaw = 0.1F, .pitch = 0.2F};
  pose.fov_degrees = 55.0F;
  const auto pan_json = pipeline_event_to_json(
      reco::core::PipelineEvent(reco::core::PanDecisionEvent{.frame_index = 10, .pose = pose}));
  expect_true(pan_json["kind"] == "pan_decision", "pan decision kind");
  expect_true(pan_json["pose"]["fov_degrees"] == 55.0F, "pan decision pose fov");

  const auto debug_json = pipeline_event_to_json(
      reco::core::PipelineEvent(reco::core::PannerDebugEvent{.frame_index = 11,
                                                             .cluster_yaw = 0.1F,
                                                             .cluster_pitch = 0.2F,
                                                             .cluster_spread = 0.3F,
                                                             .n_players = 4,
                                                             .ball_near_cluster = true,
                                                             .ball_presence = 0.5F,
                                                             .effective_ball_weight = 0.6F,
                                                             .target_yaw = 0.7F,
                                                             .target_pitch = 0.8F,
                                                             .fov_target = 60.0F}));
  expect_true(debug_json["kind"] == "panner_debug", "panner debug kind");
  expect_true(debug_json["n_players"] == 4, "panner debug players");

  const auto presented_json = pipeline_event_to_json(
      reco::core::PipelineEvent(reco::core::PosePresentedEvent{.frame_index = 12, .pose = pose}));
  expect_true(presented_json["kind"] == "pose_presented", "pose presented kind");

  const auto default_pose_json = pipeline_event_to_json(reco::core::PipelineEvent(
      reco::core::PosePresentedEvent{.frame_index = 13, .pose = reco::core::ViewportPosition{}}));
  expect_true(default_pose_json["pose"]["fov_degrees"].is_null(), "default pose fov is null");
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
  settings_round_trip_and_namespace_guards_match_rust();
  recent_files_match_rust_mru_policy();
  jsonl_sink_writes_one_pipeline_event_per_line();
  pipeline_event_json_serializes_core_event_vocabulary();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
