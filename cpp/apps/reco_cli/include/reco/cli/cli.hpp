#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <optional>
#include <span>
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

struct CameraCommand {
  std::string left_device;
  std::string right_device;
  std::string calibration;
  std::string output;
  std::uint32_t capture_width = 3840;
  std::uint32_t capture_height = 2160;
  std::uint32_t capture_fps = 30;
  std::uint32_t width = 1920;
  std::uint32_t height = 1080;
  std::optional<std::string> encoder;
  std::string codec = "h264";
  std::string quality = "fast";
  float blend = 0.15F;
  std::optional<std::uint64_t> max_frames;
  std::optional<double> end_time;
  std::optional<std::string> model;
  std::uint64_t detection_interval = 1;
  std::optional<std::uint8_t> quality_value;
  std::optional<std::string> preset;
  std::optional<std::string> container;
  std::optional<std::string> stream_url;
  std::string tracking = "field";
  bool unconstrained = false;
  std::optional<std::string> replay;
  std::optional<WxH> replay_scale;
  bool v4l2_direct = false;
  std::uint32_t exposure = 780;
  std::uint32_t sensor_gain = 16;
  bool live_calibrate = false;
  std::size_t calibrate_frames = 8;
  std::optional<std::string> left_lens_profile;
};

struct LibcameraCommand {
  std::uint32_t left_camera = 0;
  std::uint32_t right_camera = 1;
  std::string calibration;
  std::string output;
  std::uint32_t capture_width = 1920;
  std::uint32_t capture_height = 1080;
  std::uint32_t capture_fps = 30;
  std::uint32_t width = 1920;
  std::uint32_t height = 1080;
  std::optional<std::string> encoder;
  std::string codec = "h264";
  std::string quality = "fast";
  float blend = 0.15F;
  std::optional<std::uint64_t> max_frames;
  std::optional<double> end_time;
  std::optional<std::string> model;
  std::uint64_t detection_interval = 1;
  std::optional<std::uint8_t> quality_value;
  std::optional<std::string> preset;
};

struct GoproCommand {
  std::optional<std::string> serial;
  std::optional<std::string> url;
  bool start = false;
  bool stop = false;
  bool sports_preset = false;
};

struct InfoCommand {};
struct HelpCommand {};

using Command = std::variant<StitchCommand, PreviewCommand, CalibrateCommand, CameraCommand,
                             LibcameraCommand, GoproCommand, InfoCommand, HelpCommand>;

[[nodiscard]] std::variant<float, ParseError> parse_blend(std::string_view value);
[[nodiscard]] std::variant<WxH, ParseError> parse_wxh(std::string_view value);
[[nodiscard]] std::variant<Command, ParseError> parse_args(const std::vector<std::string>& args);
[[nodiscard]] std::string_view command_name(const Command& command);
[[nodiscard]] std::string help_text();
int run_command(const Command& command, std::ostream& out, std::ostream& err,
                const std::filesystem::path& executable_path = {});

namespace detail {

/// Resolves the deployed video probe worker for CLI startup and hardening tests.
[[nodiscard]] std::optional<std::filesystem::path>
resolve_video_probe_worker(const std::filesystem::path& executable_path);

/// Resolves the deployed calibration worker for CLI startup and hardening tests.
[[nodiscard]] std::optional<std::filesystem::path>
resolve_calibration_worker(const std::filesystem::path& executable_path);

/// Publishes serialized calibration JSON after rechecking that it cannot replace an input or
/// selected lens profile. The trailing controls are deterministic race hooks for publication
/// tests; production callers leave them at their defaults.
void write_calibration_json_atomically(std::string_view json,
                                       const std::filesystem::path& destination,
                                       const std::filesystem::path& left_input,
                                       const std::filesystem::path& right_input,
                                       const std::function<void()>& before_publish = {},
                                       std::span<const std::filesystem::path> lens_profiles = {},
                                       const std::function<void()>& before_commit = {},
                                       bool force_rename_fallback = false,
                                       const std::function<void()>& after_publish = {},
                                       const std::function<void()>& on_lock_contention = {});

} // namespace detail

} // namespace reco::cli
