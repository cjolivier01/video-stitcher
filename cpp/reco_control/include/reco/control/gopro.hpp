#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace reco::control::gopro {

enum class ConnectionMode {
  Usb,
  Wifi,
};

struct CameraStatus {
  std::optional<bool> battery_present;
  std::optional<int> battery_bars;
  std::optional<bool> encoding;
  std::optional<int> battery_percent;
  std::optional<int> encoding_duration_secs;
  std::optional<bool> busy;
  std::optional<bool> overheating;
  std::optional<bool> cold;
  std::optional<unsigned long long> sd_remaining_kb;
  std::optional<bool> gps_lock;
  std::optional<int> orientation;
};

struct CommandResult {
  bool ok = false;
  std::string message;
};

std::optional<CameraStatus> parse_status_json(std::string_view json);
CommandResult parse_command_response(std::string_view json);
std::string connection_mode_name(ConnectionMode mode);
std::optional<std::string> usb_base_url(std::string_view serial_suffix);
std::string wifi_base_url();
std::string endpoint_state();
std::string endpoint_info();
std::string endpoint_shutter_start();
std::string endpoint_shutter_stop();
std::string endpoint_keep_alive();
std::string endpoint_set_setting(unsigned int setting_id, unsigned int option);
std::string endpoint_set_preset_group(unsigned int group_id);
std::vector<std::string> sports_preset_endpoints(unsigned int resolution, unsigned int fps,
                                                 unsigned int lens);
std::string endpoint_webcam_start(unsigned int resolution, unsigned int fov, unsigned int port,
                                  std::string_view protocol);
std::string endpoint_webcam_stop();
std::string endpoint_webcam_status();
std::string endpoint_webcam_exit();
std::string webcam_stream_address(std::string_view base_url, unsigned int port,
                                  std::string_view protocol);
std::string endpoint_digital_zoom(unsigned int percent);
std::string endpoint_media_list();
std::string endpoint_media_last_captured();
std::string endpoint_media_telemetry(std::string_view path);
std::string endpoint_enable_turbo_transfer();
std::string endpoint_disable_turbo_transfer();
std::string endpoint_enable_wired_usb();

} // namespace reco::control::gopro
