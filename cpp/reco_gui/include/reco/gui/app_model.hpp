#pragma once

#include "reco/gui/settings.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace reco::gui {

enum class PreviewStatus {
  MissingInputs,
  WaitingForGpu,
  Ready,
};

[[nodiscard]] std::string_view preview_status_name(PreviewStatus status);

struct GuiFileSelection {
  std::optional<std::filesystem::path> left;
  std::optional<std::filesystem::path> right;
  std::optional<std::filesystem::path> calibration;

  [[nodiscard]] bool complete() const;
};

struct PreviewControls {
  float yaw = 0.0F;
  float pitch = 0.0F;
  float fov_degrees = 75.0F;
  float blend_width = 0.05F;
  bool playing = false;

  void reset_pose();
  void clamp();
};

struct ExportRequest {
  std::filesystem::path left;
  std::filesystem::path right;
  std::filesystem::path calibration;
  std::filesystem::path output;
  std::string codec;
  std::string quality;
  float blend_width = 0.05F;
  std::optional<std::filesystem::path> ai_model;
};

class GuiAppModel {
public:
  explicit GuiAppModel(GuiSettings settings = {});

  [[nodiscard]] const GuiSettings& settings() const;
  [[nodiscard]] const GuiFileSelection& files() const;
  [[nodiscard]] const PreviewControls& preview() const;
  [[nodiscard]] PreviewStatus preview_status() const;
  [[nodiscard]] bool can_export() const;
  [[nodiscard]] std::optional<ExportRequest> export_request(std::filesystem::path output) const;

  void set_left(std::filesystem::path path);
  void set_right(std::filesystem::path path);
  void set_calibration(std::filesystem::path path);
  void set_gpu_ready(bool ready);
  void set_preview_controls(PreviewControls controls);
  void set_default_codec(std::string codec);
  void set_default_quality(std::string quality);
  void set_default_blend_width(float blend_width);
  void set_ai_model(std::optional<std::filesystem::path> path);

private:
  GuiSettings settings_;
  GuiFileSelection files_;
  PreviewControls preview_;
  bool gpu_ready_ = false;
};

} // namespace reco::gui
