#include "reco/control/gopro.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <optional>
#include <sstream>
#include <system_error>

namespace reco::control::gopro {
namespace {

std::optional<std::string_view> status_object(std::string_view json) {
  const auto status_pos = json.find("\"status\"");
  if (status_pos == std::string_view::npos) {
    return std::nullopt;
  }
  const auto object_start = json.find('{', status_pos);
  if (object_start == std::string_view::npos) {
    return std::nullopt;
  }
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t i = object_start; i < json.size(); ++i) {
    const char ch = json[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0) {
        return json.substr(object_start, i - object_start + 1);
      }
    }
  }
  return std::nullopt;
}

std::optional<unsigned long long> parse_status_u64(std::string_view json, unsigned int id) {
  const auto object = status_object(json);
  if (!object.has_value()) {
    return std::nullopt;
  }
  const std::string quoted_id = "\"" + std::to_string(id) + "\"";
  const auto pos = object->find(quoted_id);
  if (pos == std::string_view::npos) {
    return std::nullopt;
  }
  const auto colon = object->find(':', pos + quoted_id.size());
  if (colon == std::string_view::npos) {
    return std::nullopt;
  }
  std::size_t first = colon + 1;
  while (first < object->size() && std::isspace(static_cast<unsigned char>((*object)[first])) != 0) {
    ++first;
  }
  unsigned long long value = 0;
  const auto* begin = object->data() + first;
  const auto* end = object->data() + object->size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc()) {
    return std::nullopt;
  }
  return value;
}

std::optional<bool> parse_status_bool(std::string_view json, unsigned int id) {
  const auto value = parse_status_u64(json, id);
  if (!value.has_value()) {
    return std::nullopt;
  }
  return *value != 0;
}

} // namespace

std::optional<CameraStatus> parse_status_json(std::string_view json) {
  if (json.empty() || json.find('{') == std::string_view::npos) {
    return std::nullopt;
  }
  CameraStatus status;
  status.battery_present = parse_status_bool(json, 1);
  if (const auto value = parse_status_u64(json, 2)) {
    status.battery_bars = static_cast<int>(*value);
  }
  if (const auto value = parse_status_u64(json, 70)) {
    status.battery_percent = static_cast<int>(*value);
  }
  status.encoding = parse_status_bool(json, 10);
  if (const auto value = parse_status_u64(json, 13)) {
    status.encoding_duration_secs = static_cast<int>(*value);
  }
  status.busy = parse_status_bool(json, 8);
  status.overheating = parse_status_bool(json, 6);
  status.cold = parse_status_bool(json, 85);
  status.sd_remaining_kb = parse_status_u64(json, 54);
  status.gps_lock = parse_status_bool(json, 68);
  if (const auto value = parse_status_u64(json, 86)) {
    status.orientation = static_cast<int>(*value);
  }
  return status;
}

CommandResult parse_command_response(std::string_view json) {
  CommandResult result;
  result.ok = json.find("\"ok\":true") != std::string_view::npos ||
              json.find("\"success\":true") != std::string_view::npos;
  result.message = result.ok ? "ok" : "command failed";
  return result;
}

std::string connection_mode_name(ConnectionMode mode) {
  switch (mode) {
  case ConnectionMode::Usb:
    return "usb";
  case ConnectionMode::Wifi:
    return "wifi";
  }
  return "unknown";
}

std::optional<std::string> usb_base_url(std::string_view serial_suffix) {
  std::string suffix(serial_suffix);
  if (suffix.size() >= 3) {
    suffix = suffix.substr(suffix.size() - 3);
  }
  if (suffix.size() != 3 ||
      !std::all_of(suffix.begin(), suffix.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
    return std::nullopt;
  }
  return "http://172.2" + suffix.substr(0, 1) + ".1" + suffix.substr(1, 2) + ".51:8080";
}

std::string wifi_base_url() { return "http://10.5.5.9:8080"; }
std::string endpoint_state() { return "/gopro/camera/state"; }
std::string endpoint_info() { return "/gopro/camera/info"; }
std::string endpoint_shutter_start() { return "/gopro/camera/shutter/start"; }
std::string endpoint_shutter_stop() { return "/gopro/camera/shutter/stop"; }
std::string endpoint_keep_alive() { return "/gopro/camera/keep_alive"; }

std::string endpoint_set_setting(unsigned int setting_id, unsigned int option) {
  return "/gopro/camera/setting?setting=" + std::to_string(setting_id) +
         "&option=" + std::to_string(option);
}

std::string endpoint_set_preset_group(unsigned int group_id) {
  return "/gopro/camera/presets/set_group?id=" + std::to_string(group_id);
}

std::vector<std::string> sports_preset_endpoints(unsigned int resolution, unsigned int fps,
                                                 unsigned int lens) {
  return {
      endpoint_set_preset_group(1000),
      endpoint_set_setting(2, resolution),
      endpoint_set_setting(3, fps),
      endpoint_set_setting(121, lens),
      endpoint_set_setting(135, 0),
      endpoint_set_setting(150, 0),
      endpoint_set_setting(59, 0),
  };
}

std::string endpoint_webcam_start(unsigned int resolution, unsigned int fov, unsigned int port,
                                  std::string_view protocol) {
  return "/gopro/webcam/start?res=" + std::to_string(resolution) +
         "&fov=" + std::to_string(fov) + "&port=" + std::to_string(port) +
         "&protocol=" + std::string(protocol);
}

std::string endpoint_webcam_stop() { return "/gopro/webcam/stop"; }
std::string endpoint_webcam_status() { return "/gopro/webcam/status"; }
std::string endpoint_webcam_exit() { return "/gopro/webcam/exit"; }

std::string webcam_stream_address(std::string_view base_url, unsigned int port,
                                  std::string_view protocol) {
  if (protocol == "RTSP") {
    return "rtsp://" + std::string(base_url) + ":" + std::to_string(port) + "/live";
  }
  return "udp://0.0.0.0:" + std::to_string(port);
}

std::string endpoint_digital_zoom(unsigned int percent) {
  return "/gopro/camera/digital_zoom?percent=" + std::to_string(std::min(percent, 100U));
}

std::string endpoint_media_list() { return "/gopro/media/list"; }
std::string endpoint_media_last_captured() { return "/gopro/media/last_captured"; }

std::string endpoint_media_telemetry(std::string_view path) {
  return "/gopro/media/telemetry?path=" + std::string(path);
}

std::string endpoint_enable_turbo_transfer() { return "/gopro/media/turbo_transfer?p=1"; }
std::string endpoint_disable_turbo_transfer() { return "/gopro/media/turbo_transfer?p=0"; }
std::string endpoint_enable_wired_usb() { return "/gopro/camera/control/wired_usb?p=1"; }

} // namespace reco::control::gopro
