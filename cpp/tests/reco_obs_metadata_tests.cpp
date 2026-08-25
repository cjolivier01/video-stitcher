#include "reco/obs/metadata.hpp"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

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
  defaults_match_obs_callback();
  input_format_parser_matches_apply_settings();
  replay_mode_parser_is_safe_by_default();
  raw_settings_apply_positive_dimensions_and_defaults();
  unknown_format_records_warning_equivalent_status();
  log_levels_match_obs_constants();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
