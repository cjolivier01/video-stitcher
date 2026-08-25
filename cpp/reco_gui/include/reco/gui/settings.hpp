#pragma once

#include "reco/io/settings.hpp"

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace reco::gui {

inline constexpr const char* kSettingsNamespace = "gui";

struct GuiSettings {
  reco::io::RecentFiles recent_left;
  reco::io::RecentFiles recent_right;
  reco::io::RecentFiles recent_calibration;
  std::string default_codec = "h264";
  std::string default_quality = "balanced";
  float default_blend_width = 0.05F;
  std::optional<std::filesystem::path> ai_model_path;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> window_size;
  bool window_maximized = false;
  std::string recording_codec = "h264";
  std::string recording_quality = "balanced";
  std::optional<std::filesystem::path> recording_folder;
  std::string preview_aspect = "auto";
  bool telemetry_enabled = false;
  std::optional<std::string> telemetry_client_id;
  bool dark_mode = true;

  [[nodiscard]] static GuiSettings load();
  void save() const;
  void push_left(std::filesystem::path path);
  void push_right(std::filesystem::path path);
  void push_calibration(std::filesystem::path path);
};

void to_json(nlohmann::json& json, const GuiSettings& settings);
void from_json(const nlohmann::json& json, GuiSettings& settings);

} // namespace reco::gui
