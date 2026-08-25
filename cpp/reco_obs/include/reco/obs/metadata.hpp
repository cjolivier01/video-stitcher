#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace reco::obs {

inline constexpr std::uint32_t kLibobsApiVersion = 30U << 24U;
inline constexpr std::string_view kModuleName = "reco-obs";
inline constexpr std::string_view kModuleDescription =
    "GPU-accelerated panoramic video stitcher powered by Reco";
inline constexpr std::string_view kSourceId = "reco_stitcher";
inline constexpr std::string_view kSourceName = "Reco Panorama Stitcher";

inline constexpr std::string_view kConfigPathKey = "config_path";
inline constexpr std::string_view kOutputWidthKey = "output_width";
inline constexpr std::string_view kOutputHeightKey = "output_height";
inline constexpr std::string_view kInputWidthKey = "input_width";
inline constexpr std::string_view kInputHeightKey = "input_height";
inline constexpr std::string_view kYawKey = "yaw";
inline constexpr std::string_view kPitchKey = "pitch";
inline constexpr std::string_view kLeftSourceKey = "left_source";
inline constexpr std::string_view kRightSourceKey = "right_source";
inline constexpr std::string_view kInputFormatKey = "input_format";
inline constexpr std::string_view kReplayEnabledKey = "replay_enabled";
inline constexpr std::string_view kReplayPathKey = "replay_path";
inline constexpr std::string_view kReplayModeKey = "replay_mode";

inline constexpr std::string_view kInputFormatI420 = "i420";
inline constexpr std::string_view kInputFormatBgra = "bgra";
inline constexpr std::string_view kReplayModeFollowObs = "follow_obs";
inline constexpr std::string_view kReplayModeAlways = "always";

enum class SourceType {
  kInput,
};

enum class SourceIcon {
  kCamera,
};

enum class InputFormat {
  kYuv420p,
  kBgra,
};

enum class LogLevel {
  kError,
  kWarn,
  kInfo,
  kDebug,
  kTrace,
};

struct SourceFlags {
  bool video = true;
  bool interaction = true;
};

struct SourceMetadata {
  std::string_view id = kSourceId;
  std::string_view name = kSourceName;
  SourceType type = SourceType::kInput;
  SourceFlags flags;
  SourceIcon icon = SourceIcon::kCamera;
};

struct SourceDefaults {
  std::uint32_t output_width = 1920;
  std::uint32_t output_height = 1080;
  std::uint32_t input_width = 1920;
  std::uint32_t input_height = 1080;
  std::string config_path;
  InputFormat input_format = InputFormat::kYuv420p;
  bool replay_enabled = false;
  std::string replay_path;
  std::string replay_mode = std::string(kReplayModeFollowObs);
};

struct InputFormatParseResult {
  InputFormat format = InputFormat::kYuv420p;
  bool used_fallback = false;
};

struct ParsedSettings {
  std::string config_path;
  std::uint32_t output_width = 1920;
  std::uint32_t output_height = 1080;
  std::uint32_t input_width = 1920;
  std::uint32_t input_height = 1080;
  InputFormat input_format = InputFormat::kYuv420p;
  bool input_format_used_fallback = false;
  std::string left_source;
  std::string right_source;
  double yaw_degrees = 0.0;
  double pitch_degrees = 0.0;
  bool replay_enabled = false;
  std::string replay_path;
  bool replay_follow_obs = true;
};

struct RawSettings {
  std::optional<std::string> config_path;
  std::optional<std::int64_t> output_width;
  std::optional<std::int64_t> output_height;
  std::optional<std::int64_t> input_width;
  std::optional<std::int64_t> input_height;
  std::optional<std::string> input_format;
  std::optional<std::string> left_source;
  std::optional<std::string> right_source;
  std::optional<double> yaw_degrees;
  std::optional<double> pitch_degrees;
  std::optional<bool> replay_enabled;
  std::optional<std::string> replay_path;
  std::optional<std::string> replay_mode;
};

[[nodiscard]] SourceMetadata source_metadata();
[[nodiscard]] SourceDefaults source_defaults();
[[nodiscard]] InputFormatParseResult parse_input_format(std::optional<std::string_view> value);
[[nodiscard]] bool replay_mode_follows_obs(std::optional<std::string_view> value);
[[nodiscard]] ParsedSettings parse_settings(const RawSettings& raw,
                                            const SourceDefaults& defaults = source_defaults());
[[nodiscard]] int obs_log_level(LogLevel level);

} // namespace reco::obs
