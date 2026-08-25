#include "reco/obs/metadata.hpp"

namespace reco::obs {

SourceMetadata source_metadata() { return {}; }

SourceDefaults source_defaults() { return {}; }

InputFormatParseResult parse_input_format(std::optional<std::string_view> value) {
  if (!value.has_value() || value->empty() || *value == kInputFormatI420) {
    return {.format = InputFormat::kYuv420p, .used_fallback = false};
  }
  if (*value == kInputFormatBgra) {
    return {.format = InputFormat::kBgra, .used_fallback = false};
  }
  return {.format = InputFormat::kYuv420p, .used_fallback = true};
}

bool replay_mode_follows_obs(std::optional<std::string_view> value) {
  return !value.has_value() || *value != kReplayModeAlways;
}

ParsedSettings parse_settings(const RawSettings& raw, const SourceDefaults& defaults) {
  ParsedSettings parsed;
  parsed.config_path = raw.config_path.value_or(defaults.config_path);
  parsed.output_width = raw.output_width.has_value() && *raw.output_width > 0
                            ? static_cast<std::uint32_t>(*raw.output_width)
                            : defaults.output_width;
  parsed.output_height = raw.output_height.has_value() && *raw.output_height > 0
                             ? static_cast<std::uint32_t>(*raw.output_height)
                             : defaults.output_height;
  parsed.input_width = raw.input_width.has_value() && *raw.input_width > 0
                           ? static_cast<std::uint32_t>(*raw.input_width)
                           : defaults.input_width;
  parsed.input_height = raw.input_height.has_value() && *raw.input_height > 0
                            ? static_cast<std::uint32_t>(*raw.input_height)
                            : defaults.input_height;
  const auto input_format =
      parse_input_format(raw.input_format.has_value()
                             ? std::optional<std::string_view>(*raw.input_format)
                             : std::nullopt);
  parsed.input_format = input_format.format;
  parsed.input_format_used_fallback = input_format.used_fallback;
  parsed.left_source = raw.left_source.value_or("");
  parsed.right_source = raw.right_source.value_or("");
  parsed.yaw_degrees = raw.yaw_degrees.value_or(0.0);
  parsed.pitch_degrees = raw.pitch_degrees.value_or(0.0);
  parsed.replay_enabled = raw.replay_enabled.value_or(defaults.replay_enabled);
  parsed.replay_path = raw.replay_path.value_or(defaults.replay_path);
  parsed.replay_follow_obs =
      replay_mode_follows_obs(raw.replay_mode.has_value()
                                  ? std::optional<std::string_view>(*raw.replay_mode)
                                  : std::optional<std::string_view>(defaults.replay_mode));
  return parsed;
}

int obs_log_level(LogLevel level) {
  switch (level) {
  case LogLevel::kError:
    return 100;
  case LogLevel::kWarn:
    return 200;
  case LogLevel::kInfo:
    return 300;
  case LogLevel::kDebug:
  case LogLevel::kTrace:
    return 400;
  }
  return 400;
}

} // namespace reco::obs
