#include "reco/cli/cli.hpp"

#include "reco/calibrate/pipeline.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/detect/coreml_session.hpp"
#include "reco/detect/npp_interop.hpp"
#include "reco/detect/ort_session.hpp"
#include "reco/detect/probe.hpp"
#include "reco/io/gstreamer.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>

namespace reco::cli {
namespace {

template <typename T>
std::variant<T, ParseError> parse_integral(std::string_view value, std::string_view name) {
  if (value.empty()) {
    return ParseError{std::string(name) + " requires a value"};
  }
  T parsed{};
  const char* begin = value.data();
  const char* end = value.data() + value.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end) {
    return ParseError{"invalid " + std::string(name) + " " + std::string(value)};
  }
  return parsed;
}

std::variant<double, ParseError> parse_double(std::string_view value, std::string_view name) {
  std::string text(value);
  char* end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (text.empty() || end != text.c_str() + text.size()) {
    return ParseError{"invalid " + std::string(name) + " " + text};
  }
  return parsed;
}

std::variant<float, ParseError> parse_float(std::string_view value, std::string_view name) {
  std::string text(value);
  char* end = nullptr;
  const float parsed = std::strtof(text.c_str(), &end);
  if (text.empty() || end != text.c_str() + text.size()) {
    return ParseError{"invalid " + std::string(name) + " " + text};
  }
  return parsed;
}

class Cursor {
public:
  explicit Cursor(const std::vector<std::string>& args) : args_(args) {}

  [[nodiscard]] bool empty() const { return index_ >= args_.size(); }
  [[nodiscard]] const std::string& peek() const { return args_[index_]; }
  [[nodiscard]] std::string take() { return args_[index_++]; }

  [[nodiscard]] std::variant<std::string, ParseError> value(std::string_view option,
                                                            bool allow_hyphen_value = false) {
    if (empty()) {
      return ParseError{"missing value for " + std::string(option)};
    }
    if (!allow_hyphen_value && !peek().empty() && peek().front() == '-') {
      return ParseError{"missing value for " + std::string(option)};
    }
    return take();
  }

private:
  const std::vector<std::string>& args_;
  std::size_t index_ = 0;
};

template <typename T>
bool assign_or_error(std::variant<T, ParseError>&& parsed, T& out, std::optional<ParseError>& err) {
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    err = *error;
    return false;
  }
  out = std::get<T>(std::move(parsed));
  return true;
}

std::optional<std::string> take_value_or_error(Cursor& cursor, std::string_view option,
                                               std::optional<ParseError>& err,
                                               bool allow_hyphen_value = false) {
  auto value = cursor.value(option, allow_hyphen_value);
  if (const auto* error = std::get_if<ParseError>(&value)) {
    err = *error;
    return std::nullopt;
  }
  return std::get<std::string>(std::move(value));
}

void write_probe(std::ostream& out, std::string_view label, bool available,
                 std::string_view detail) {
  out << "  " << label << ": " << (available ? "available" : "unavailable");
  if (!detail.empty()) {
    out << " (" << detail << ")";
  }
  out << '\n';
}

template <typename T, typename Parser>
bool assign_next_or_error(Cursor& cursor, std::string_view option, Parser parser, T& out,
                          std::optional<ParseError>& err, bool allow_hyphen_value = false) {
  auto value = take_value_or_error(cursor, option, err, allow_hyphen_value);
  if (!value.has_value()) {
    return false;
  }
  return assign_or_error(parser(*value), out, err);
}

template <typename T>
bool assign_optional_or_error(std::variant<T, ParseError>&& parsed, std::optional<T>& out,
                              std::optional<ParseError>& err) {
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    err = *error;
    return false;
  }
  out = std::get<T>(std::move(parsed));
  return true;
}

template <typename T, typename Parser>
bool assign_next_optional_or_error(Cursor& cursor, std::string_view option, Parser parser,
                                   std::optional<T>& out, std::optional<ParseError>& err,
                                   bool allow_hyphen_value = false) {
  auto value = take_value_or_error(cursor, option, err, allow_hyphen_value);
  if (!value.has_value()) {
    return false;
  }
  return assign_optional_or_error(parser(*value), out, err);
}

std::variant<StitchCommand, ParseError> parse_stitch(Cursor& cursor) {
  StitchCommand command;
  std::vector<std::string> positionals;
  std::optional<ParseError> err;

  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--start-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.start_time, err))
        return *err;
    } else if (arg == "--end-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.end_time, err))
        return *err;
    } else if (arg == "--max-frames") {
      if (!assign_next_optional_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.max_frames, err))
        return *err;
    } else if (arg == "--encoder") {
      if (!assign_optional_or_error(next(arg), command.encoder, err))
        return *err;
    } else if (arg == "--codec") {
      if (!assign_or_error(next(arg), command.codec, err))
        return *err;
    } else if (arg == "--quality") {
      if (!assign_or_error(next(arg), command.quality, err))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(cursor, arg, parse_blend, command.blend, err)) {
        return *err;
      }
    } else if (arg == "--sync-offset") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::int64_t>(v, arg); },
              command.sync_offset, err, true))
        return *err;
    } else if (arg == "--model") {
      if (!assign_optional_or_error(next(arg), command.model, err))
        return *err;
    } else if (arg == "--detection-interval") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.detection_interval, err))
        return *err;
    } else if (arg == "--lookahead") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.lookahead, err))
        return *err;
    } else if (arg == "--tracking") {
      if (!assign_or_error(next(arg), command.tracking, err))
        return *err;
    } else if (arg == "--quality-value") {
      std::uint32_t value = 0;
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); }, value,
              err))
        return *err;
      if (value > std::numeric_limits<std::uint8_t>::max()) {
        return ParseError{"invalid --quality-value " + std::to_string(value)};
      }
      command.quality_value = static_cast<std::uint8_t>(value);
    } else if (arg == "--preset") {
      if (!assign_optional_or_error(next(arg), command.preset, err))
        return *err;
    } else if (arg == "--container") {
      if (!assign_optional_or_error(next(arg), command.container, err))
        return *err;
    } else if (arg == "--replay") {
      if (!assign_optional_or_error(next(arg), command.replay, err))
        return *err;
    } else if (arg == "--replay-scale") {
      if (!assign_next_optional_or_error(cursor, arg, parse_wxh, command.replay_scale, err)) {
        return *err;
      }
    } else if (arg == "--allow-no-tracking") {
      command.allow_no_tracking = true;
    } else if (arg == "--no-zero-copy") {
      command.no_zero_copy = true;
    } else if (arg == "--events") {
      if (!assign_optional_or_error(next(arg), command.events, err))
        return *err;
    } else if (arg == "--trajectory") {
      if (!assign_optional_or_error(next(arg), command.trajectory, err))
        return *err;
    } else if (arg == "--panner-config") {
      if (!assign_optional_or_error(next(arg), command.panner_config, err))
        return *err;
    } else if (arg == "--panner-preset") {
      if (!assign_optional_or_error(next(arg), command.panner_preset, err))
        return *err;
    } else if (!arg.empty() && arg.front() == '-') {
      return ParseError{"unknown stitch option " + arg};
    } else {
      positionals.push_back(arg);
    }
  }

  if (positionals.size() != 2) {
    return ParseError{"stitch requires LEFT and RIGHT inputs"};
  }
  if (command.calibration.empty()) {
    return ParseError{"stitch requires -c/--calibration"};
  }
  command.left = positionals[0];
  command.right = positionals[1];
  return command;
}

std::variant<PreviewCommand, ParseError> parse_preview(Cursor& cursor) {
  PreviewCommand command;
  std::vector<std::string> positionals;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--sync-offset") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::int64_t>(v, arg); },
              command.sync_offset, err, true))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_float(v, arg); }, command.blend,
              err))
        return *err;
    } else if (arg == "--rig-tilt") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_float(v, arg); },
              command.rig_tilt, err, true))
        return *err;
    } else if (!arg.empty() && arg.front() == '-') {
      return ParseError{"unknown preview option " + arg};
    } else {
      positionals.push_back(arg);
    }
  }
  if (positionals.size() != 2) {
    return ParseError{"preview requires LEFT and RIGHT inputs"};
  }
  if (command.calibration.empty()) {
    return ParseError{"preview requires -c/--calibration"};
  }
  command.left = positionals[0];
  command.right = positionals[1];
  return command;
}

std::variant<CameraCommand, ParseError> parse_camera(Cursor& cursor) {
  CameraCommand command;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--left-device") {
      if (!assign_or_error(next(arg), command.left_device, err))
        return *err;
    } else if (arg == "--right-device") {
      if (!assign_or_error(next(arg), command.right_device, err))
        return *err;
    } else if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (arg == "--capture-width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_width, err))
        return *err;
    } else if (arg == "--capture-height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_height, err))
        return *err;
    } else if (arg == "--capture-fps") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_fps, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--encoder") {
      if (!assign_optional_or_error(next(arg), command.encoder, err))
        return *err;
    } else if (arg == "--codec") {
      if (!assign_or_error(next(arg), command.codec, err))
        return *err;
    } else if (arg == "--quality") {
      if (!assign_or_error(next(arg), command.quality, err))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(cursor, arg, parse_blend, command.blend, err)) {
        return *err;
      }
    } else if (arg == "--max-frames") {
      if (!assign_next_optional_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.max_frames, err))
        return *err;
    } else if (arg == "--end-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.end_time, err))
        return *err;
    } else if (arg == "--model") {
      if (!assign_optional_or_error(next(arg), command.model, err))
        return *err;
    } else if (arg == "--detection-interval") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.detection_interval, err))
        return *err;
    } else if (arg == "--quality-value") {
      std::uint32_t value = 0;
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); }, value,
              err))
        return *err;
      if (value > std::numeric_limits<std::uint8_t>::max()) {
        return ParseError{"invalid --quality-value " + std::to_string(value)};
      }
      command.quality_value = static_cast<std::uint8_t>(value);
    } else if (arg == "--preset") {
      if (!assign_optional_or_error(next(arg), command.preset, err))
        return *err;
    } else if (arg == "--container") {
      if (!assign_optional_or_error(next(arg), command.container, err))
        return *err;
    } else if (arg == "--stream-url") {
      if (!assign_optional_or_error(next(arg), command.stream_url, err))
        return *err;
    } else if (arg == "--tracking") {
      if (!assign_or_error(next(arg), command.tracking, err))
        return *err;
    } else if (arg == "--unconstrained") {
      command.unconstrained = true;
    } else if (arg == "--replay") {
      if (!assign_optional_or_error(next(arg), command.replay, err))
        return *err;
    } else if (arg == "--replay-scale") {
      if (!assign_next_optional_or_error(cursor, arg, parse_wxh, command.replay_scale, err)) {
        return *err;
      }
    } else if (arg == "--v4l2-direct") {
      command.v4l2_direct = true;
    } else if (arg == "--exposure") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.exposure, err))
        return *err;
    } else if (arg == "--sensor-gain") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.sensor_gain, err))
        return *err;
    } else if (arg == "--live-calibrate") {
      command.live_calibrate = true;
    } else if (arg == "--calibrate-frames") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::size_t>(v, arg); },
              command.calibrate_frames, err))
        return *err;
    } else if (arg == "--left-lens-profile") {
      if (!assign_optional_or_error(next(arg), command.left_lens_profile, err))
        return *err;
    } else {
      return ParseError{"unknown camera option " + arg};
    }
  }
  if (command.left_device.empty()) {
    return ParseError{"camera requires --left-device"};
  }
  if (command.right_device.empty()) {
    return ParseError{"camera requires --right-device"};
  }
  if (command.calibration.empty()) {
    return ParseError{"camera requires -c/--calibration"};
  }
  if (command.output.empty()) {
    return ParseError{"camera requires -o/--output"};
  }
  return command;
}

std::variant<LibcameraCommand, ParseError> parse_libcamera(Cursor& cursor) {
  LibcameraCommand command;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--left-camera") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.left_camera, err))
        return *err;
    } else if (arg == "--right-camera") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.right_camera, err))
        return *err;
    } else if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (arg == "--capture-width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_width, err))
        return *err;
    } else if (arg == "--capture-height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_height, err))
        return *err;
    } else if (arg == "--capture-fps") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_fps, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--encoder") {
      if (!assign_optional_or_error(next(arg), command.encoder, err))
        return *err;
    } else if (arg == "--codec") {
      if (!assign_or_error(next(arg), command.codec, err))
        return *err;
    } else if (arg == "--quality") {
      if (!assign_or_error(next(arg), command.quality, err))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(cursor, arg, parse_blend, command.blend, err)) {
        return *err;
      }
    } else if (arg == "--max-frames") {
      if (!assign_next_optional_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.max_frames, err))
        return *err;
    } else if (arg == "--end-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.end_time, err))
        return *err;
    } else if (arg == "--model") {
      if (!assign_optional_or_error(next(arg), command.model, err))
        return *err;
    } else if (arg == "--detection-interval") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.detection_interval, err))
        return *err;
    } else if (arg == "--quality-value") {
      std::uint32_t value = 0;
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); }, value,
              err))
        return *err;
      if (value > std::numeric_limits<std::uint8_t>::max()) {
        return ParseError{"invalid --quality-value " + std::to_string(value)};
      }
      command.quality_value = static_cast<std::uint8_t>(value);
    } else if (arg == "--preset") {
      if (!assign_optional_or_error(next(arg), command.preset, err))
        return *err;
    } else {
      return ParseError{"unknown libcamera option " + arg};
    }
  }
  if (command.calibration.empty()) {
    return ParseError{"libcamera requires -c/--calibration"};
  }
  if (command.output.empty()) {
    return ParseError{"libcamera requires -o/--output"};
  }
  return command;
}

std::variant<GoproCommand, ParseError> parse_gopro(Cursor& cursor) {
  GoproCommand command;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--serial") {
      if (!assign_optional_or_error(next(arg), command.serial, err))
        return *err;
    } else if (arg == "--url") {
      if (!assign_optional_or_error(next(arg), command.url, err))
        return *err;
    } else if (arg == "--start") {
      command.start = true;
    } else if (arg == "--stop") {
      command.stop = true;
    } else if (arg == "--sports-preset") {
      command.sports_preset = true;
    } else {
      return ParseError{"unknown gopro option " + arg};
    }
  }
  return command;
}

std::variant<CalibrateCommand, ParseError> parse_calibrate(Cursor& cursor) {
  CalibrateCommand command;
  std::vector<std::string> positionals;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--left-profile") {
      if (!assign_optional_or_error(next(arg), command.left_profile, err))
        return *err;
    } else if (arg == "--right-profile") {
      if (!assign_optional_or_error(next(arg), command.right_profile, err))
        return *err;
    } else if (arg == "--frames") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::size_t>(v, arg); },
              command.frames, err))
        return *err;
    } else if (arg == "--no-auto-imu") {
      command.no_auto_imu = true;
    } else if (arg == "--no-auto-sync") {
      command.auto_sync = false;
    } else if (arg == "--auto-sync") {
      std::string value;
      if (!assign_or_error(next(arg), value, err))
        return *err;
      if (value == "true") {
        command.auto_sync = true;
      } else if (value == "false") {
        command.auto_sync = false;
      } else {
        return ParseError{"--auto-sync expects true or false"};
      }
    } else if (arg == "--sync-offset") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::int64_t>(v, arg); },
              command.sync_offset, err, true))
        return *err;
    } else if (arg == "--skip-start") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.skip_start, err))
        return *err;
    } else if (arg == "--skip-end") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.skip_end, err))
        return *err;
    } else if (arg == "--akaze-threshold") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.akaze_threshold, err))
        return *err;
    } else if (arg == "--lowe-ratio") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.lowe_ratio, err))
        return *err;
    } else if (arg == "--detect-x") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.detect_x, err))
        return *err;
    } else if (arg == "--detect-y-min") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.detect_y_min, err))
        return *err;
    } else if (arg == "--detect-y-max") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.detect_y_max, err))
        return *err;
    } else if (arg == "--lock-cam-d") {
      command.lock_cam_d = true;
    } else if (arg == "--lock-z-rx") {
      command.lock_z_rx = true;
    } else if (arg == "--trim") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); }, command.trim,
              err)) {
        return *err;
      }
    } else if (arg == "--seam-sigma") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.seam_sigma, err))
        return *err;
    } else if (arg == "--debug-dir") {
      if (!assign_optional_or_error(next(arg), command.debug_dir, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (!arg.empty() && arg.front() == '-') {
      return ParseError{"unknown calibrate option " + arg};
    } else {
      positionals.push_back(arg);
    }
  }
  if (positionals.size() != 2) {
    return ParseError{"calibrate requires LEFT and RIGHT inputs"};
  }
  command.left = positionals[0];
  command.right = positionals[1];
  return command;
}

} // namespace

std::variant<float, ParseError> parse_blend(std::string_view value) {
  auto parsed = parse_float(value, "blend");
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    return *error;
  }
  const float blend = std::get<float>(parsed);
  if (!std::isfinite(blend) || blend < 0.0F || blend > 1.0F) {
    std::ostringstream out;
    out << blend << " is not in 0.0..=1.0";
    return ParseError{out.str()};
  }
  return blend;
}

std::variant<WxH, ParseError> parse_wxh(std::string_view value) {
  const auto split = value.find_first_of("xX");
  if (split == std::string_view::npos) {
    return ParseError{"expected WIDTHxHEIGHT, got " + std::string(value)};
  }
  auto width = parse_integral<std::uint32_t>(value.substr(0, split), "width");
  if (const auto* error = std::get_if<ParseError>(&width)) {
    return *error;
  }
  auto height = parse_integral<std::uint32_t>(value.substr(split + 1), "height");
  if (const auto* error = std::get_if<ParseError>(&height)) {
    return *error;
  }
  const WxH parsed{.width = std::get<std::uint32_t>(width),
                   .height = std::get<std::uint32_t>(height)};
  if (parsed.width == 0 || parsed.height == 0) {
    return ParseError{"dimensions must be > 0, got " + std::to_string(parsed.width) + "x" +
                      std::to_string(parsed.height)};
  }
  if (parsed.width % 4 != 0) {
    return ParseError{"width must be divisible by 4 (pack shader packs 4 pixels per u32 write), "
                      "got " +
                      std::to_string(parsed.width)};
  }
  if (parsed.height % 2 != 0) {
    return ParseError{"height must be even (YUV420P chroma subsampling), got " +
                      std::to_string(parsed.height)};
  }
  return parsed;
}

std::variant<Command, ParseError> parse_args(const std::vector<std::string>& args) {
  if (args.empty() || args[0] == "--help" || args[0] == "-h") {
    return Command{HelpCommand{}};
  }
  Cursor cursor(args);
  const std::string subcommand = cursor.take();
  if (subcommand == "stitch" || subcommand == "preview" || subcommand == "camera" ||
      subcommand == "libcamera" || subcommand == "calibrate" || subcommand == "gopro" ||
      subcommand == "info") {
    if (std::find(args.begin() + 1, args.end(), "--help") != args.end() ||
        std::find(args.begin() + 1, args.end(), "-h") != args.end()) {
      return Command{HelpCommand{}};
    }
  }
  if (subcommand == "stitch") {
    auto parsed = parse_stitch(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<StitchCommand>(std::move(parsed))};
  }
  if (subcommand == "preview") {
    auto parsed = parse_preview(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<PreviewCommand>(std::move(parsed))};
  }
  if (subcommand == "camera") {
    auto parsed = parse_camera(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<CameraCommand>(std::move(parsed))};
  }
  if (subcommand == "libcamera") {
    auto parsed = parse_libcamera(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<LibcameraCommand>(std::move(parsed))};
  }
  if (subcommand == "calibrate") {
    auto parsed = parse_calibrate(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<CalibrateCommand>(std::move(parsed))};
  }
  if (subcommand == "gopro") {
    auto parsed = parse_gopro(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<GoproCommand>(std::move(parsed))};
  }
  if (subcommand == "info") {
    if (!cursor.empty()) {
      return ParseError{"info does not accept positional arguments or options"};
    }
    return Command{InfoCommand{}};
  }
  return ParseError{"unknown command " + subcommand};
}

std::string_view command_name(const Command& command) {
  return std::visit(
      [](const auto& value) -> std::string_view {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, StitchCommand>)
          return "stitch";
        if constexpr (std::is_same_v<T, PreviewCommand>)
          return "preview";
        if constexpr (std::is_same_v<T, CalibrateCommand>)
          return "calibrate";
        if constexpr (std::is_same_v<T, CameraCommand>)
          return "camera";
        if constexpr (std::is_same_v<T, LibcameraCommand>)
          return "libcamera";
        if constexpr (std::is_same_v<T, GoproCommand>)
          return "gopro";
        if constexpr (std::is_same_v<T, InfoCommand>)
          return "info";
        return "help";
      },
      command);
}

std::string help_text() {
  return "Reco C++ CLI port\n\n"
         "Usage:\n"
         "  reco stitch LEFT RIGHT -c CALIBRATION [options]\n"
         "  reco preview LEFT RIGHT -c CALIBRATION [options]\n"
         "  reco camera --left-device DEV --right-device DEV -c CALIBRATION -o OUTPUT [options]\n"
         "  reco libcamera -c CALIBRATION -o OUTPUT [options]\n"
         "  reco calibrate LEFT RIGHT [options]\n"
         "  reco gopro [options]\n"
         "  reco info\n\n"
         "Runtime command execution is staged behind the remaining GPU/backend ports.";
}

int run_command(const Command& command, std::ostream& out, std::ostream& err) {
  if (std::holds_alternative<HelpCommand>(command)) {
    out << help_text() << '\n';
    return 0;
  }

  if (std::holds_alternative<InfoCommand>(command)) {
    out << "Reco C++ capability report\n";
    const bool cuda_available = reco::core::CudaBackend::is_available();
    if (cuda_available) {
      auto backend = reco::core::CudaBackend::create();
      const int devices = backend.device_count();
      out << "CUDA: available (" << devices << " device";
      if (devices != 1) {
        out << 's';
      }
      out << ")\n";
      for (int ordinal = 0; ordinal < devices; ++ordinal) {
        const auto device = backend.device_info(ordinal);
        out << "  cuda[" << ordinal << "]: " << device.name << '\n';
      }
    } else {
      out << "CUDA: unavailable (" << reco::core::CudaBackend::availability_error() << ")\n";
    }

    write_probe(out, "NPP", reco::detect::is_npp_available(),
                reco::detect::npp_availability_error());

    const auto gst = reco::io::probe_gstreamer_runtime();
    write_probe(out, "GStreamer", gst.available, gst.available ? gst.library : gst.error);
    const auto deepstream = reco::io::probe_deepstream_runtime();
    write_probe(out, "DeepStream", deepstream.available,
                deepstream.available ? deepstream.library : deepstream.error);
    const auto nvbuf = reco::io::probe_nvbufsurface_runtime();
    write_probe(out, "NvBufSurface", nvbuf.available,
                nvbuf.available ? nvbuf.library : nvbuf.error);

    const auto ort = reco::detect::probe_ort_runtime();
    write_probe(out, "ONNX Runtime", ort.available, ort.available ? ort.version : ort.error);
    const auto coreml = reco::detect::probe_coreml_runtime();
    write_probe(out, "CoreML", coreml.available, coreml.available ? coreml.provider : coreml.error);
    const auto ai = reco::detect::probe_execution_providers();
    out << "AI providers:";
    if (ai.providers.empty()) {
      out << " none";
    } else {
      for (const auto& provider : ai.providers) {
        out << ' ' << provider;
      }
    }
    out << '\n';
    out << "GPU frame inference: " << (ai.can_run_on_gpu_frames ? "available" : "unavailable")
        << '\n';
    return 0;
  }

  if (const auto* calibrate = std::get_if<CalibrateCommand>(&command)) {
    reco::calibrate::GpuCalibrationRequest request;
    request.left.path = calibrate->left;
    request.left.lens_profile = calibrate->left_profile;
    request.right.path = calibrate->right;
    request.right.lens_profile = calibrate->right_profile;
    request.config.num_frames = calibrate->frames;
    request.config.skip_start_secs = calibrate->skip_start;
    request.config.skip_end_secs = calibrate->skip_end;
    request.config.akaze.threshold = calibrate->akaze_threshold;
    request.config.akaze.detect_y_min = calibrate->detect_y_min;
    request.config.akaze.detect_y_max = calibrate->detect_y_max;
    request.config.matching.lowe_ratio = calibrate->lowe_ratio;
    request.config.matching.spatial_x_threshold = calibrate->detect_x;
    request.config.optimizer.lock_cam_d = calibrate->lock_cam_d;
    request.config.optimizer.lock_z_rx = calibrate->lock_z_rx;
    request.config.optimizer.trim_fraction = calibrate->trim;
    request.config.optimizer.seam_sigma = calibrate->seam_sigma;
    request.no_auto_imu = calibrate->no_auto_imu;
    request.auto_sync = calibrate->auto_sync;
    request.manual_sync_offset = calibrate->sync_offset;
    request.debug_dir = calibrate->debug_dir;
    request.output = calibrate->output;

    const auto backends = reco::calibrate::probe_calibration_backends();
    const auto plan = reco::calibrate::build_gpu_calibration_plan(request, backends);
    out << reco::calibrate::describe_calibration_plan(plan);
    if (!plan.ready) {
      err << "error: " << plan.blocked_reason.value_or("C++ GPU calibration is unavailable")
          << '\n';
      return 2;
    }

    try {
      (void)reco::calibrate::run_gpu_calibration(request, backends);
    } catch (const reco::calibrate::CalibrationExecutionError& error) {
      err << "error: " << error.what() << '\n';
      return 2;
    }
    return 0;
  }

  err << "error: C++ reco " << command_name(command)
      << " execution is not ported yet; GPU/runtime backend stages remain authoritative in Rust.\n";
  return 2;
}

} // namespace reco::cli
