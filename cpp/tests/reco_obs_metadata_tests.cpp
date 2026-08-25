#include "reco/obs/metadata.hpp"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace reco::obs;

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

const PropertyDescriptor* find_property(const std::vector<PropertyDescriptor>& properties,
                                        std::string_view key) {
  for (const auto& property : properties) {
    if (property.key == key) {
      return &property;
    }
  }
  return nullptr;
}

void metadata_matches_rust_module_and_source_info() {
  static_assert(kLibobsApiVersion == (30U << 24U));
  expect_eq(kModuleName, std::string_view("reco-obs"), "module name");
  expect_eq(kModuleDescription,
            std::string_view("GPU-accelerated panoramic video stitcher powered by Reco"),
            "module description");

  const auto metadata = source_metadata();
  expect_eq(metadata.id, std::string_view("reco_stitcher"), "source id");
  expect_eq(metadata.name, std::string_view("Reco Panorama Stitcher"), "source name");
  expect_true(metadata.type == SourceType::kInput, "source type");
  expect_true(metadata.flags.video, "source video flag");
  expect_true(metadata.flags.interaction, "source interaction flag");
  expect_true(metadata.icon == SourceIcon::kCamera, "source icon");
}

void source_properties_match_obs_properties_callback() {
  const auto properties = source_properties(true);
  expect_eq(properties.size(), 13U, "property count with replay");

  const auto* left = find_property(properties, kLeftSourceKey);
  expect_true(left != nullptr, "left source property exists");
  if (left != nullptr) {
    expect_eq(left->label, std::string("Left camera source"), "left source label");
    expect_true(left->kind == PropertyKind::kSourceList, "left source kind");
  }

  const auto* format = find_property(properties, kInputFormatKey);
  expect_true(format != nullptr, "input format property exists");
  if (format != nullptr) {
    expect_true(format->kind == PropertyKind::kStringList, "input format kind");
    expect_eq(format->choices.size(), 2U, "input format choice count");
    if (format->choices.size() == 2) {
      expect_eq(format->choices[0].label, std::string("I420 (Media Source, V4L2)"),
                "i420 choice label");
      expect_eq(format->choices[0].value, std::string("i420"), "i420 choice value");
      expect_eq(format->choices[1].label, std::string("BGRA (Browser Source, Screen Capture)"),
                "bgra choice label");
      expect_eq(format->choices[1].value, std::string("bgra"), "bgra choice value");
    }
  }

  const auto* config = find_property(properties, kConfigPathKey);
  expect_true(config != nullptr, "config path property exists");
  if (config != nullptr) {
    expect_true(config->kind == PropertyKind::kPathOpen, "config path kind");
    expect_eq(config->filter.value_or(""), std::string("JSON files (*.json)"),
              "config path filter");
  }

  const auto* output_width = find_property(properties, kOutputWidthKey);
  expect_true(output_width != nullptr, "output width property exists");
  if (output_width != nullptr && output_width->range.has_value()) {
    expect_eq(output_width->range->min, 320.0, "output width min");
    expect_eq(output_width->range->max, 7680.0, "output width max");
    expect_eq(output_width->range->step, 1.0, "output width step");
  } else {
    expect_true(false, "output width range");
  }

  const auto* pitch = find_property(properties, kPitchKey);
  expect_true(pitch != nullptr, "pitch property exists");
  if (pitch != nullptr && pitch->range.has_value()) {
    expect_eq(pitch->range->min, -90.0, "pitch min");
    expect_eq(pitch->range->max, 90.0, "pitch max");
    expect_eq(pitch->range->step, 0.1, "pitch step");
  } else {
    expect_true(false, "pitch range");
  }

  const auto* replay_path = find_property(properties, kReplayPathKey);
  expect_true(replay_path != nullptr, "replay path property exists");
  if (replay_path != nullptr) {
    expect_true(replay_path->kind == PropertyKind::kPathSave, "replay path kind");
    expect_eq(replay_path->filter.value_or(""), std::string("Matroska video (*.mkv)"),
              "replay path filter");
  }

  const auto* replay_mode = find_property(properties, kReplayModeKey);
  expect_true(replay_mode != nullptr, "replay mode property exists");
  if (replay_mode != nullptr) {
    expect_eq(replay_mode->choices.size(), 2U, "replay mode choice count");
    if (replay_mode->choices.size() == 2) {
      expect_eq(replay_mode->choices[0].label, std::string("Follow OBS Record / Stream"),
                "follow OBS choice label");
      expect_eq(replay_mode->choices[0].value, std::string("follow_obs"),
                "follow OBS choice value");
      expect_eq(replay_mode->choices[1].label, std::string("Record independently (advanced)"),
                "always choice label");
      expect_eq(replay_mode->choices[1].value, std::string("always"), "always choice value");
    }
  }

  const auto without_replay = source_properties();
  expect_eq(without_replay.size(), 10U, "property count without replay");
  expect_true(find_property(without_replay, kReplayEnabledKey) == nullptr,
              "no replay enabled property without replay");
}

void defaults_match_obs_callback() {
  const auto defaults = source_defaults();
  expect_eq(defaults.output_width, 1920U, "default output width");
  expect_eq(defaults.output_height, 1080U, "default output height");
  expect_eq(defaults.input_width, 1920U, "default input width");
  expect_eq(defaults.input_height, 1080U, "default input height");
  expect_true(defaults.config_path.empty(), "default config path");
  expect_true(defaults.input_format == InputFormat::kYuv420p, "default input format");
  expect_true(!defaults.replay_enabled, "default replay disabled");
  expect_true(defaults.replay_path.empty(), "default replay path");
  expect_eq(defaults.replay_mode, std::string("follow_obs"), "default replay mode");
}

void upstream_source_classification_matches_warning_branches() {
  expect_true(classify_upstream_source({.video = false, .async_video = false}) ==
                  UpstreamSourceStatus::kNoVideo,
              "no video source");
  expect_true(classify_upstream_source({.video = true, .async_video = false}) ==
                  UpstreamSourceStatus::kSyncVideo,
              "sync video source");
  expect_true(classify_upstream_source({.video = true, .async_video = true}) ==
                  UpstreamSourceStatus::kAsyncVideo,
              "async video source");
}

void input_format_parser_matches_apply_settings() {
  expect_true(parse_input_format(std::nullopt).format == InputFormat::kYuv420p,
              "null input format");
  expect_true(!parse_input_format(std::nullopt).used_fallback, "null input format no fallback");
  expect_true(parse_input_format("").format == InputFormat::kYuv420p, "empty input format");
  expect_true(!parse_input_format("").used_fallback, "empty input format no fallback");
  expect_true(parse_input_format("i420").format == InputFormat::kYuv420p, "i420 input format");
  expect_true(!parse_input_format("i420").used_fallback, "i420 input format no fallback");
  expect_true(parse_input_format("bgra").format == InputFormat::kBgra, "bgra input format");
  expect_true(!parse_input_format("bgra").used_fallback, "bgra input format no fallback");

  const auto unknown = parse_input_format("nv12");
  expect_true(unknown.format == InputFormat::kYuv420p, "unknown input format fallback");
  expect_true(unknown.used_fallback, "unknown input format fallback status");
}

void replay_mode_parser_is_safe_by_default() {
  expect_true(replay_mode_follows_obs(std::nullopt), "null replay mode follows OBS");
  expect_true(replay_mode_follows_obs(""), "empty replay mode follows OBS");
  expect_true(replay_mode_follows_obs("follow_obs"), "follow_obs replay mode");
  expect_true(replay_mode_follows_obs("future_mode"), "unknown replay mode follows OBS");
  expect_true(!replay_mode_follows_obs("always"), "always replay mode");
}

void raw_settings_apply_positive_dimensions_and_defaults() {
  RawSettings raw;
  raw.config_path = "/tmp/calibration.json";
  raw.output_width = 3840;
  raw.output_height = 2160;
  raw.input_width = 0;
  raw.input_height = -720;
  raw.input_format = "bgra";
  raw.left_source = "Media Left";
  raw.right_source = "Media Right";
  raw.yaw_degrees = 12.5;
  raw.pitch_degrees = -3.0;
  raw.replay_enabled = true;
  raw.replay_path = "/tmp/replay.mkv";
  raw.replay_mode = "always";

  const auto parsed = parse_settings(raw);
  expect_eq(parsed.config_path, std::string("/tmp/calibration.json"), "parsed config path");
  expect_eq(parsed.output_width, 3840U, "parsed output width");
  expect_eq(parsed.output_height, 2160U, "parsed output height");
  expect_eq(parsed.input_width, 1920U, "zero input width keeps default");
  expect_eq(parsed.input_height, 1080U, "negative input height keeps default");
  expect_true(parsed.input_format == InputFormat::kBgra, "parsed input format");
  expect_true(!parsed.input_format_used_fallback, "known input format no fallback");
  expect_eq(parsed.left_source, std::string("Media Left"), "parsed left source");
  expect_eq(parsed.right_source, std::string("Media Right"), "parsed right source");
  expect_eq(parsed.yaw_degrees, 12.5, "parsed yaw");
  expect_eq(parsed.pitch_degrees, -3.0, "parsed pitch");
  expect_true(parsed.replay_enabled, "parsed replay enabled");
  expect_eq(parsed.replay_path, std::string("/tmp/replay.mkv"), "parsed replay path");
  expect_true(!parsed.replay_follow_obs, "parsed replay mode");
}

void unknown_format_records_warning_equivalent_status() {
  RawSettings raw;
  raw.input_format = "nv12";

  const auto parsed = parse_settings(raw);
  expect_true(parsed.input_format == InputFormat::kYuv420p, "unknown setting format fallback");
  expect_true(parsed.input_format_used_fallback, "unknown setting format fallback status");
}

void source_state_transition_matches_update_logic() {
  SourceState state;
  state.warned_unsupported_format = true;

  RawSettings initial;
  initial.config_path = "a.json";
  initial.output_width = 2560;
  initial.output_height = 1440;
  initial.input_width = 0;
  initial.input_height = -1;
  initial.input_format = "bgra";
  initial.left_source = "Left";
  initial.right_source = "Right";
  initial.yaw_degrees = 10.0;
  initial.pitch_degrees = -2.0;
  initial.replay_enabled = true;
  initial.replay_path = "replay.mkv";
  initial.replay_mode = "always";

  const auto first = apply_settings(state, initial);
  expect_true(first.config_changed, "config change detected");
  expect_true(first.output_dimensions_changed, "output dimensions changed");
  expect_true(!first.input_dimensions_changed, "invalid input dimensions ignored");
  expect_true(first.input_format_changed, "format change detected");
  expect_true(first.left_source_changed, "left source changed");
  expect_true(first.right_source_changed, "right source changed");
  expect_true(first.left_source_resolve_requested, "left source resolve requested");
  expect_true(first.right_source_resolve_requested, "right source resolve requested");
  expect_true(first.pose_target_changed, "pose target changed");
  expect_true(first.unsupported_format_warning_reset, "warning reset on format/source change");
  expect_true(first.reload_calibration, "reload calibration on config change");
  expect_true(first.rebuild_pipeline, "rebuild pipeline on config/dim/format change");
  expect_true(!first.update_replay_recorder, "replay recorder waits for rebuild path");
  expect_true(!first.input_format_used_fallback, "known format no fallback");
  expect_eq(state.config_path, std::string("a.json"), "state config path");
  expect_eq(state.output_width, 2560U, "state output width");
  expect_eq(state.output_height, 1440U, "state output height");
  expect_eq(state.input_width, 1920U, "state input width unchanged");
  expect_eq(state.input_height, 1080U, "state input height unchanged");
  expect_true(state.input_format == InputFormat::kBgra, "state input format");
  expect_eq(state.left_source, std::string("Left"), "state left source");
  expect_eq(state.right_source, std::string("Right"), "state right source");
  expect_eq(state.yaw_degrees, 10.0, "state yaw");
  expect_eq(state.pitch_degrees, -2.0, "state pitch");
  expect_true(!state.warned_unsupported_format, "state warning reset");
  expect_true(state.replay_enabled, "state replay enabled");
  expect_eq(state.replay_path, std::string("replay.mkv"), "state replay path");
  expect_true(!state.replay_follow_obs, "state replay follows always mode");

  state.left_source_resolved = true;
  state.right_source_resolved = true;
  RawSettings replay_only;
  replay_only.input_format = "bgra";
  replay_only.left_source = "Left";
  replay_only.right_source = "Right";
  replay_only.replay_mode = "follow_obs";
  const auto replay = apply_settings(state, replay_only);
  expect_true(!replay.rebuild_pipeline, "replay-only update does not rebuild");
  expect_true(replay.update_replay_recorder, "replay-only update refreshes recorder");
  expect_true(state.replay_follow_obs, "state replay follows OBS");

  state.left_source_resolved = false;
  state.right_source_resolved = false;
  state.warned_unsupported_format = true;
  RawSettings retry_sources;
  retry_sources.input_format = "bgra";
  retry_sources.left_source = "Left";
  retry_sources.right_source = "Right";
  const auto retry = apply_settings(state, retry_sources);
  expect_true(!retry.left_source_changed, "left source name unchanged");
  expect_true(!retry.right_source_changed, "right source name unchanged");
  expect_true(retry.left_source_resolve_requested, "left unresolved source retry");
  expect_true(retry.right_source_resolve_requested, "right unresolved source retry");
  expect_true(retry.unsupported_format_warning_reset, "retry resets warning");
  expect_true(!state.warned_unsupported_format, "retry warning reset state");

  state.calibration_present = true;
  state.core_present = false;
  RawSettings retry_pipeline;
  retry_pipeline.input_format = "bgra";
  const auto pipeline_retry = apply_settings(state, retry_pipeline);
  expect_true(pipeline_retry.rebuild_pipeline, "calibrated missing core retries pipeline");
  expect_true(!pipeline_retry.update_replay_recorder, "pipeline retry skips replay-only path");

  RawSettings unknown_format;
  unknown_format.input_format = "nv12";
  const auto fallback = apply_settings(state, unknown_format);
  expect_true(fallback.input_format_used_fallback, "state format fallback status");
  expect_true(fallback.input_format_changed, "fallback to yuv changes bgra state");
  expect_true(fallback.rebuild_pipeline, "format fallback rebuilds when state changes");
  expect_true(state.input_format == InputFormat::kYuv420p, "fallback state format");
}

void log_levels_match_obs_constants() {
  expect_eq(obs_log_level(LogLevel::kError), 100, "error log level");
  expect_eq(obs_log_level(LogLevel::kWarn), 200, "warn log level");
  expect_eq(obs_log_level(LogLevel::kInfo), 300, "info log level");
  expect_eq(obs_log_level(LogLevel::kDebug), 400, "debug log level");
  expect_eq(obs_log_level(LogLevel::kTrace), 400, "trace log level");
}

} // namespace

int main() {
  metadata_matches_rust_module_and_source_info();
  source_properties_match_obs_properties_callback();
  defaults_match_obs_callback();
  upstream_source_classification_matches_warning_branches();
  input_format_parser_matches_apply_settings();
  replay_mode_parser_is_safe_by_default();
  raw_settings_apply_positive_dimensions_and_defaults();
  unknown_format_records_warning_equivalent_status();
  source_state_transition_matches_update_logic();
  log_levels_match_obs_constants();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
