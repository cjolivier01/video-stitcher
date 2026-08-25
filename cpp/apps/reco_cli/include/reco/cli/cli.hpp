#pragma once

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace reco::cli {

struct WxH {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct ParseError {
  std::string message;
};

struct StitchCommand {
  std::string left;
  std::string right;
  std::string calibration;
  std::string output = "output.mp4";
  std::uint32_t width = 1920;
  std::uint32_t height = 1080;
  std::optional<double> start_time;
  std::optional<double> end_time;
  std::optional<std::uint64_t> max_frames;
  std::optional<std::string> encoder;
  std::string codec = "h264";
  std::string quality = "balanced";
  float blend = 0.15F;
  std::int64_t sync_offset = 0;
  std::optional<std::string> model;
  std::uint64_t detection_interval = 1;
  double lookahead = 1.5;
  std::string tracking = "field";
  std::optional<std::uint8_t> quality_value;
  std::optional<std::string> preset;
  std::optional<std::string> container;
  std::optional<std::string> replay;
  std::optional<WxH> replay_scale;
  bool allow_no_tracking = false;
  bool no_zero_copy = false;
  std::optional<std::string> events;
  std::optional<std::string> trajectory;
  std::optional<std::string> panner_config;
  std::optional<std::string> panner_preset;
};

struct PreviewCommand {
  std::string left;
  std::string right;
  std::string calibration;
  std::uint32_t width = 1280;
  std::uint32_t height = 720;
  std::int64_t sync_offset = 0;
  float blend = 0.15F;
  float rig_tilt = 0.0F;
};

struct CalibrateCommand {
  std::string left;
  std::string right;
  std::optional<std::string> left_profile;
  std::optional<std::string> right_profile;
  std::size_t frames = 2;
  bool no_auto_imu = false;
  bool auto_sync = true;
  std::int64_t sync_offset = 0;
  double skip_start = 0.0;
  double skip_end = 0.0;
  double akaze_threshold = 0.0001;
  double lowe_ratio = 0.75;
  double detect_x = 0.5;
  double detect_y_min = 0.25;
  double detect_y_max = 0.85;
  bool lock_cam_d = false;
  bool lock_z_rx = false;
  double trim = 0.3;
  double seam_sigma = 0.08;
  std::optional<std::string> debug_dir;
  std::string output = "match.json";
};

struct InfoCommand {};
struct HelpCommand {};

using Command =
    std::variant<StitchCommand, PreviewCommand, CalibrateCommand, InfoCommand, HelpCommand>;

[[nodiscard]] std::variant<float, ParseError> parse_blend(std::string_view value);
[[nodiscard]] std::variant<WxH, ParseError> parse_wxh(std::string_view value);
[[nodiscard]] std::variant<Command, ParseError> parse_args(const std::vector<std::string>& args);
[[nodiscard]] std::string_view command_name(const Command& command);
[[nodiscard]] std::string help_text();
int run_command(const Command& command, std::ostream& out, std::ostream& err);

} // namespace reco::cli
