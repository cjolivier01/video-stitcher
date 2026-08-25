#include "reco/cli/cli.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using namespace reco::cli;

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

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

Command expect_command(std::variant<Command, ParseError> parsed, std::string_view message) {
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    std::cerr << "FAIL: " << message << " unexpected error=" << error->message << '\n';
    ++failures;
    return HelpCommand{};
  }
  return std::get<Command>(std::move(parsed));
}

void expect_error(const std::variant<Command, ParseError>& parsed, std::string_view message) {
  if (!std::holds_alternative<ParseError>(parsed)) {
    std::cerr << "FAIL: " << message << " expected parse error\n";
    ++failures;
  }
}

void validators_match_rust() {
  const auto blend = parse_blend("0.25");
  expect_true(std::holds_alternative<float>(blend), "blend parses");
  expect_near(std::get<float>(blend), 0.25F, 1.0e-6F, "blend value");
  expect_true(std::holds_alternative<float>(parse_blend("0.0")), "blend lower inclusive");
  expect_true(std::holds_alternative<float>(parse_blend("1.0")), "blend upper inclusive");
  expect_true(std::holds_alternative<ParseError>(parse_blend("-0.1")), "blend below rejected");
  expect_true(std::holds_alternative<ParseError>(parse_blend("1.1")), "blend above rejected");
  expect_true(std::holds_alternative<ParseError>(parse_blend("NaN")), "blend nan rejected");
  expect_true(std::holds_alternative<ParseError>(parse_blend("0.5x")), "blend suffix rejected");

  const auto wxh = parse_wxh("1280x720");
  expect_true(std::holds_alternative<WxH>(wxh), "wxh parses");
  expect_eq(std::get<WxH>(wxh).width, 1280U, "wxh width");
  expect_eq(std::get<WxH>(wxh).height, 720U, "wxh height");
  expect_true(std::holds_alternative<WxH>(parse_wxh("856X480")), "wxh uppercase separator");
  expect_true(std::holds_alternative<ParseError>(parse_wxh("0x720")), "zero width rejected");
  expect_true(std::holds_alternative<ParseError>(parse_wxh("854x480")), "width alignment rejected");
  expect_true(std::holds_alternative<ParseError>(parse_wxh("1280x721")), "height parity rejected");
}

void stitch_parse_matches_rust_defaults() {
  const auto command =
      expect_command(parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json",
                                 "--sync-offset", "-3", "--replay-scale", "1280x720",
                                 "--quality-value", "80", "--allow-no-tracking", "--no-zero-copy"}),
                     "stitch parse");
  const auto* stitch = std::get_if<StitchCommand>(&command);
  expect_true(stitch != nullptr, "stitch variant");
  if (stitch == nullptr)
    return;
  expect_eq(stitch->left, std::string("left.mp4"), "stitch left");
  expect_eq(stitch->right, std::string("right.mp4"), "stitch right");
  expect_eq(stitch->calibration, std::string("match.json"), "stitch calibration");
  expect_eq(stitch->output, std::string("output.mp4"), "stitch output default");
  expect_eq(stitch->width, 1920U, "stitch width default");
  expect_eq(stitch->height, 1080U, "stitch height default");
  expect_near(stitch->blend, 0.15F, 1.0e-6F, "stitch blend default");
  expect_eq(stitch->sync_offset, -3, "stitch negative sync");
  expect_eq(stitch->codec, std::string("h264"), "stitch codec default");
  expect_eq(stitch->quality, std::string("balanced"), "stitch quality default");
  expect_eq(stitch->tracking, std::string("field"), "stitch tracking default");
  expect_true(stitch->replay_scale.has_value(), "stitch replay scale");
  expect_eq(stitch->replay_scale->width, 1280U, "stitch replay width");
  expect_eq(stitch->quality_value.value_or(0), 80U, "stitch quality value");
  expect_true(stitch->allow_no_tracking, "stitch allow no tracking");
  expect_true(stitch->no_zero_copy, "stitch no zero copy");
}

void preview_and_calibrate_parse_matches_rust_defaults() {
  const auto preview_command =
      expect_command(parse_args({"preview", "l.mp4", "r.mp4", "--calibration", "match.json",
                                 "--rig-tilt", "-2.5"}),
                     "preview parse");
  const auto* preview = std::get_if<PreviewCommand>(&preview_command);
  expect_true(preview != nullptr, "preview variant");
  if (preview != nullptr) {
    expect_eq(preview->width, 1280U, "preview width default");
    expect_eq(preview->height, 720U, "preview height default");
    expect_near(preview->blend, 0.15F, 1.0e-6F, "preview blend default");
    expect_near(preview->rig_tilt, -2.5F, 1.0e-6F, "preview negative rig tilt");
  }

  const auto preview_unbounded_blend = expect_command(
      parse_args({"preview", "l.mp4", "r.mp4", "-c", "match.json", "--blend", "1.1"}),
      "preview unbounded blend parse");
  const auto* unbounded_preview = std::get_if<PreviewCommand>(&preview_unbounded_blend);
  expect_true(unbounded_preview != nullptr, "preview unbounded blend variant");
  if (unbounded_preview != nullptr) {
    expect_near(unbounded_preview->blend, 1.1F, 1.0e-6F, "preview blend is raw f32");
  }

  const auto preview_nan_blend = expect_command(
      parse_args({"preview", "l.mp4", "r.mp4", "-c", "match.json", "--blend", "NaN"}),
      "preview nan blend parse");
  const auto* nan_preview = std::get_if<PreviewCommand>(&preview_nan_blend);
  expect_true(nan_preview != nullptr, "preview nan blend variant");
  if (nan_preview != nullptr) {
    expect_true(std::isnan(nan_preview->blend), "preview blend accepts nan like raw f32");
  }

  const auto calibrate_command =
      expect_command(parse_args({"calibrate", "l.mp4", "r.mp4", "--frames", "8", "--no-auto-imu",
                                 "--no-auto-sync", "--output", "out.json"}),
                     "calibrate parse");
  const auto* calibrate = std::get_if<CalibrateCommand>(&calibrate_command);
  expect_true(calibrate != nullptr, "calibrate variant");
  if (calibrate != nullptr) {
    expect_eq(calibrate->frames, 8U, "calibrate frames");
    expect_true(calibrate->no_auto_imu, "calibrate no auto imu");
    expect_true(!calibrate->auto_sync, "calibrate no auto sync");
    expect_eq(calibrate->output, std::string("out.json"), "calibrate output");
    expect_near(static_cast<float>(calibrate->akaze_threshold), 0.0001F, 1.0e-8F,
                "calibrate akaze default");
    expect_near(static_cast<float>(calibrate->lowe_ratio), 0.75F, 1.0e-6F,
                "calibrate lowe default");
  }
}

void live_command_parse_matches_rust_defaults() {
  const auto camera_command = expect_command(
      parse_args({"camera", "--left-device", "0", "--right-device", "1", "-c", "match.json", "-o",
                  "out.mp4", "--stream-url", "rtmp://example/live", "--replay-scale", "1280x720",
                  "--live-calibrate", "--left-lens-profile", "lens.json"}),
      "camera parse");
  const auto* camera = std::get_if<CameraCommand>(&camera_command);
  expect_true(camera != nullptr, "camera variant");
  if (camera != nullptr) {
    expect_eq(camera->left_device, std::string("0"), "camera left device");
    expect_eq(camera->right_device, std::string("1"), "camera right device");
    expect_eq(camera->capture_width, 3840U, "camera capture width default");
    expect_eq(camera->capture_height, 2160U, "camera capture height default");
    expect_eq(camera->capture_fps, 30U, "camera fps default");
    expect_eq(camera->width, 1920U, "camera output width default");
    expect_eq(camera->height, 1080U, "camera output height default");
    expect_eq(camera->quality, std::string("fast"), "camera quality default");
    expect_eq(camera->tracking, std::string("field"), "camera tracking default");
    expect_true(camera->stream_url.has_value(), "camera stream url");
    expect_true(camera->replay_scale.has_value(), "camera replay scale");
    expect_true(camera->live_calibrate, "camera live calibrate");
    expect_eq(camera->calibrate_frames, 8U, "camera calibrate frames default");
    expect_eq(camera->exposure, 780U, "camera exposure default");
    expect_eq(camera->sensor_gain, 16U, "camera sensor gain default");
    expect_eq(camera->left_lens_profile.value_or(""), std::string("lens.json"),
              "camera lens profile");
  }

  const auto libcamera_command = expect_command(
      parse_args({"libcamera", "-c", "match.json", "-o", "out.mp4", "--left-camera", "2",
                  "--right-camera", "3", "--quality-value", "70", "--preset", "fast"}),
      "libcamera parse");
  const auto* libcamera = std::get_if<LibcameraCommand>(&libcamera_command);
  expect_true(libcamera != nullptr, "libcamera variant");
  if (libcamera != nullptr) {
    expect_eq(libcamera->left_camera, 2U, "libcamera left camera");
    expect_eq(libcamera->right_camera, 3U, "libcamera right camera");
    expect_eq(libcamera->capture_width, 1920U, "libcamera capture width default");
    expect_eq(libcamera->capture_height, 1080U, "libcamera capture height default");
    expect_eq(libcamera->quality, std::string("fast"), "libcamera quality default");
    expect_eq(libcamera->quality_value.value_or(0), 70U, "libcamera quality value");
    expect_eq(libcamera->preset.value_or(""), std::string("fast"), "libcamera preset");
  }

  const auto gopro_command = expect_command(
      parse_args({"gopro", "--serial", "123", "--start", "--sports-preset"}), "gopro parse");
  const auto* gopro = std::get_if<GoproCommand>(&gopro_command);
  expect_true(gopro != nullptr, "gopro variant");
  if (gopro != nullptr) {
    expect_eq(gopro->serial.value_or(""), std::string("123"), "gopro serial");
    expect_true(gopro->start, "gopro start");
    expect_true(!gopro->stop, "gopro stop default");
    expect_true(gopro->sports_preset, "gopro sports preset");
  }
}

void parse_errors_are_reported() {
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4"}), "missing calibration");
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4", "-c"}), "missing option value");
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4", "-c", "--unknown"}),
               "string value cannot be another option");
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json", "--output",
                           "--codec", "h264"}),
               "string value cannot consume valid option");
  expect_error(
      parse_args({"preview", "left.mp4", "right.mp4", "-c", "match.json", "--width", "1920x"}),
      "numeric suffix rejected");
  expect_error(
      parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json", "--quality-value", "256"}),
      "quality value u8 range rejected");
  expect_error(
      parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json", "--lookahead", "-1"}),
      "lookahead does not allow hyphen value");
  expect_error(parse_args({"calibrate", "left.mp4", "right.mp4", "--skip-start", "-1"}),
               "skip start does not allow hyphen value");
  expect_error(parse_args({"info", "--verbose"}), "info rejects options");
  expect_error(parse_args({"camera", "--right-device", "1", "-c", "match.json", "-o", "out.mp4"}),
               "camera requires left device");
  expect_error(parse_args({"libcamera", "-c", "match.json"}), "libcamera requires output");
  expect_error(parse_args({"gopro", "--bogus"}), "gopro rejects unknown option");

  const auto help = expect_command(parse_args({"--help"}), "help parse");
  expect_true(std::holds_alternative<HelpCommand>(help), "help variant");
  const auto stitch_help = expect_command(parse_args({"stitch", "--help"}), "stitch help parse");
  expect_true(std::holds_alternative<HelpCommand>(stitch_help), "stitch help variant");
  const auto partial_stitch_help =
      expect_command(parse_args({"stitch", "left.mp4", "--help"}), "partial stitch help parse");
  expect_true(std::holds_alternative<HelpCommand>(partial_stitch_help),
              "partial stitch help variant");
  const auto camera_help =
      expect_command(parse_args({"camera", "--left-device", "0", "--help"}), "camera help parse");
  expect_true(std::holds_alternative<HelpCommand>(camera_help), "camera help variant");
}

void command_execution_dispatches_available_stages() {
  std::ostringstream out;
  std::ostringstream err;
  const auto help_status = run_command(HelpCommand{}, out, err);
  expect_eq(help_status, 0, "help exits success");
  expect_true(out.str().find("Usage:") != std::string::npos, "help writes usage");
  expect_true(err.str().empty(), "help writes no stderr");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  const auto info_status = run_command(InfoCommand{}, out, err);
  expect_eq(info_status, 0, "info exits success");
  expect_true(out.str().find("Reco C++ capability report") != std::string::npos,
              "info writes report heading");
  expect_true(out.str().find("CUDA:") != std::string::npos, "info writes CUDA probe");
  expect_true(out.str().find("GStreamer:") != std::string::npos, "info writes GStreamer probe");
  expect_true(out.str().find("AI providers:") != std::string::npos, "info writes AI probe");
  expect_true(err.str().empty(), "info writes no stderr");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  StitchCommand stitch{.left = "left.mp4", .right = "right.mp4", .calibration = "match.json"};
  const auto stitch_status = run_command(Command{stitch}, out, err);
  expect_eq(stitch_status, 2, "stitch exits blocked");
  expect_true(out.str().empty(), "blocked stitch writes no stdout");
  expect_true(err.str().find("execution is not ported yet") != std::string::npos,
              "blocked stitch explains staged execution");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  CalibrateCommand calibrate{.left = "left.mp4", .right = "right.mp4"};
  const auto calibrate_status = run_command(Command{calibrate}, out, err);
  expect_eq(calibrate_status, 2, "calibrate exits blocked until GPU pipeline is complete");
  expect_true(out.str().find("GPU calibration plan") != std::string::npos,
              "calibrate writes GPU plan");
  expect_true(out.str().find("FeatureMatching") != std::string::npos,
              "calibrate writes AKAZE stage");
  expect_true(err.str().find("error:") != std::string::npos, "calibrate writes stderr error");
}

} // namespace

int main() {
  validators_match_rust();
  stitch_parse_matches_rust_defaults();
  preview_and_calibrate_parse_matches_rust_defaults();
  live_command_parse_matches_rust_defaults();
  parse_errors_are_reported();
  command_execution_dispatches_available_stages();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
