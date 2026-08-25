#include "reco/gui/settings.hpp"

#include <array>

namespace reco::gui {
namespace {

std::optional<std::filesystem::path> optional_path(const nlohmann::json& json, const char* key) {
  if (!json.contains(key) || json.at(key).is_null()) {
    return std::nullopt;
  }
  return std::filesystem::path(json.at(key).get<std::string>());
}

std::optional<std::pair<std::uint32_t, std::uint32_t>> optional_window_size(
    const nlohmann::json& json, const char* key) {
  if (!json.contains(key) || json.at(key).is_null()) {
    return std::nullopt;
  }
  const auto size = json.at(key).get<std::array<std::uint32_t, 2>>();
  return std::pair<std::uint32_t, std::uint32_t>{size[0], size[1]};
}

void put_optional_path(nlohmann::json& json, const char* key,
                       const std::optional<std::filesystem::path>& value) {
  if (value.has_value()) {
    json[key] = value->string();
  } else {
    json[key] = nullptr;
  }
}

} // namespace

GuiSettings GuiSettings::load() { return reco::io::load_settings_or_default<GuiSettings>("gui"); }

void GuiSettings::save() const { (void)reco::io::save_settings("gui", *this); }

void GuiSettings::push_left(std::filesystem::path path) {
  recent_left.push(std::move(path));
  save();
}

void GuiSettings::push_right(std::filesystem::path path) {
  recent_right.push(std::move(path));
  save();
}

void GuiSettings::push_calibration(std::filesystem::path path) {
  recent_calibration.push(std::move(path));
  save();
}

void to_json(nlohmann::json& json, const GuiSettings& settings) {
  json = nlohmann::json{
      {"recent_left", settings.recent_left},
      {"recent_right", settings.recent_right},
      {"recent_calibration", settings.recent_calibration},
      {"default_codec", settings.default_codec},
      {"default_quality", settings.default_quality},
      {"default_blend_width", settings.default_blend_width},
      {"window_maximized", settings.window_maximized},
      {"recording_codec", settings.recording_codec},
      {"recording_quality", settings.recording_quality},
      {"preview_aspect", settings.preview_aspect},
      {"telemetry_enabled", settings.telemetry_enabled},
      {"dark_mode", settings.dark_mode},
  };
  put_optional_path(json, "ai_model_path", settings.ai_model_path);
  put_optional_path(json, "recording_folder", settings.recording_folder);
  if (settings.window_size.has_value()) {
    json["window_size"] = nlohmann::json::array({settings.window_size->first,
                                                 settings.window_size->second});
  } else {
    json["window_size"] = nullptr;
  }
  if (settings.telemetry_client_id.has_value()) {
    json["telemetry_client_id"] = *settings.telemetry_client_id;
  } else {
    json["telemetry_client_id"] = nullptr;
  }
}

void from_json(const nlohmann::json& json, GuiSettings& settings) {
  settings = GuiSettings{};
  settings.recent_left = json.value("recent_left", reco::io::RecentFiles{});
  settings.recent_right = json.value("recent_right", reco::io::RecentFiles{});
  settings.recent_calibration = json.value("recent_calibration", reco::io::RecentFiles{});
  settings.default_codec = json.value("default_codec", std::string("h264"));
  settings.default_quality = json.value("default_quality", std::string("balanced"));
  settings.default_blend_width = json.value("default_blend_width", 0.05F);
  settings.ai_model_path = optional_path(json, "ai_model_path");
  settings.window_size = optional_window_size(json, "window_size");
  settings.window_maximized = json.value("window_maximized", false);
  settings.recording_codec = json.value("recording_codec", std::string("h264"));
  settings.recording_quality = json.value("recording_quality", std::string("balanced"));
  settings.recording_folder = optional_path(json, "recording_folder");
  settings.preview_aspect = json.value("preview_aspect", std::string("auto"));
  settings.telemetry_enabled = json.value("telemetry_enabled", false);
  if (json.contains("telemetry_client_id") && !json.at("telemetry_client_id").is_null()) {
    settings.telemetry_client_id = json.at("telemetry_client_id").get<std::string>();
  }
  settings.dark_mode = json.value("dark_mode", true);
}

} // namespace reco::gui
