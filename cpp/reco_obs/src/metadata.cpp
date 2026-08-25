#include "reco/obs/metadata.hpp"

namespace reco::obs {

SourceMetadata source_metadata() { return {}; }

SourceDefaults source_defaults() { return {}; }

std::vector<PropertyDescriptor> source_properties(bool include_replay) {
  std::vector<PropertyDescriptor> properties = {
      {.key = kLeftSourceKey, .label = "Left camera source", .kind = PropertyKind::kSourceList},
      {.key = kRightSourceKey, .label = "Right camera source", .kind = PropertyKind::kSourceList},
      {.key = kInputFormatKey,
       .label = "Input format",
       .kind = PropertyKind::kStringList,
       .choices = {{"I420 (Media Source, V4L2)", std::string(kInputFormatI420)},
                   {"BGRA (Browser Source, Screen Capture)", std::string(kInputFormatBgra)}}},
      {.key = kConfigPathKey,
       .label = "Calibration file",
       .kind = PropertyKind::kPathOpen,
       .filter = "JSON files (*.json)"},
      {.key = kOutputWidthKey,
       .label = "Output width",
       .kind = PropertyKind::kInteger,
       .range = NumericRange{.min = 320.0, .max = 7680.0, .step = 1.0}},
      {.key = kOutputHeightKey,
       .label = "Output height",
       .kind = PropertyKind::kInteger,
       .range = NumericRange{.min = 240.0, .max = 4320.0, .step = 1.0}},
      {.key = kInputWidthKey,
       .label = "Input width (per camera)",
       .kind = PropertyKind::kInteger,
       .range = NumericRange{.min = 320.0, .max = 7680.0, .step = 1.0}},
      {.key = kInputHeightKey,
       .label = "Input height (per camera)",
       .kind = PropertyKind::kInteger,
       .range = NumericRange{.min = 240.0, .max = 4320.0, .step = 1.0}},
      {.key = kYawKey,
       .label = "Camera yaw (degrees)",
       .kind = PropertyKind::kFloat,
       .range = NumericRange{.min = -180.0, .max = 180.0, .step = 0.1}},
      {.key = kPitchKey,
       .label = "Camera pitch (degrees)",
       .kind = PropertyKind::kFloat,
       .range = NumericRange{.min = -90.0, .max = 90.0, .step = 0.1}},
  };

  if (include_replay) {
    properties.push_back({.key = kReplayEnabledKey,
                          .label = "Record replay (stacked video)",
                          .kind = PropertyKind::kBoolean});
    properties.push_back({.key = kReplayPathKey,
                          .label = "Replay output path (.mkv)",
                          .kind = PropertyKind::kPathSave,
                          .filter = "Matroska video (*.mkv)"});
    properties.push_back({.key = kReplayModeKey,
                          .label = "Replay trigger",
                          .kind = PropertyKind::kStringList,
                          .choices = {{"Follow OBS Record / Stream",
                                       std::string(kReplayModeFollowObs)},
                                      {"Record independently (advanced)",
                                       std::string(kReplayModeAlways)}}});
  }

  return properties;
}

UpstreamSourceStatus classify_upstream_source(UpstreamSourceFlags flags) {
  if (!flags.video) {
    return UpstreamSourceStatus::kNoVideo;
  }
  if (!flags.async_video) {
    return UpstreamSourceStatus::kSyncVideo;
  }
  return UpstreamSourceStatus::kAsyncVideo;
}

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
