#include "reco/gui/app_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace reco::gui {
namespace {

constexpr float kMinFovDegrees = 20.0F;
constexpr float kMaxFovDegrees = 150.0F;
constexpr float kMinPitchDegrees = -89.0F;
constexpr float kMaxPitchDegrees = 89.0F;
constexpr float kMaxGuiBlendWidth = 0.3F;

std::filesystem::path require_output(std::filesystem::path output) {
  if (output.empty()) {
    throw std::invalid_argument("GUI export output path must not be empty");
  }
  return output;
}

} // namespace

std::string_view preview_status_name(PreviewStatus status) {
  switch (status) {
  case PreviewStatus::MissingInputs:
    return "missing_inputs";
  case PreviewStatus::WaitingForGpu:
    return "waiting_for_gpu";
  case PreviewStatus::Ready:
    return "ready";
  }
  return "unknown";
}

bool GuiFileSelection::complete() const {
  return left.has_value() && right.has_value() && calibration.has_value();
}

void PreviewControls::reset_pose() {
  yaw = 0.0F;
  pitch = 0.0F;
  fov_degrees = 75.0F;
}

void PreviewControls::clamp() {
  if (!std::isfinite(yaw)) {
    yaw = 0.0F;
  }
  if (!std::isfinite(pitch)) {
    pitch = 0.0F;
  }
  if (!std::isfinite(fov_degrees)) {
    fov_degrees = 75.0F;
  }
  if (!std::isfinite(blend_width)) {
    blend_width = 0.05F;
  }
  pitch = std::clamp(pitch, kMinPitchDegrees, kMaxPitchDegrees);
  fov_degrees = std::clamp(fov_degrees, kMinFovDegrees, kMaxFovDegrees);
  blend_width = std::clamp(blend_width, 0.0F, kMaxGuiBlendWidth);
}

GuiAppModel::GuiAppModel(GuiSettings settings)
    : settings_(std::move(settings)), preview_{.blend_width = settings_.default_blend_width} {
  preview_.clamp();
}

const GuiSettings& GuiAppModel::settings() const { return settings_; }

const GuiFileSelection& GuiAppModel::files() const { return files_; }

const PreviewControls& GuiAppModel::preview() const { return preview_; }

PreviewStatus GuiAppModel::preview_status() const {
  if (!files_.complete()) {
    return PreviewStatus::MissingInputs;
  }
  if (!gpu_ready_) {
    return PreviewStatus::WaitingForGpu;
  }
  return PreviewStatus::Ready;
}

bool GuiAppModel::can_export() const { return preview_status() == PreviewStatus::Ready; }

std::optional<ExportRequest> GuiAppModel::export_request(std::filesystem::path output) const {
  if (!can_export()) {
    return std::nullopt;
  }
  return ExportRequest{
      .left = *files_.left,
      .right = *files_.right,
      .calibration = *files_.calibration,
      .output = require_output(std::move(output)),
      .codec = settings_.default_codec,
      .quality = settings_.default_quality,
      .blend_width = preview_.blend_width,
      .ai_model = settings_.ai_model_path,
  };
}

void GuiAppModel::set_left(std::filesystem::path path) {
  const bool changed = files_.left != path;
  files_.left = std::move(path);
  if (files_.left->empty()) {
    files_.left.reset();
  }
  if (changed) {
    gpu_ready_ = false;
  }
}

void GuiAppModel::set_right(std::filesystem::path path) {
  const bool changed = files_.right != path;
  files_.right = std::move(path);
  if (files_.right->empty()) {
    files_.right.reset();
  }
  if (changed) {
    gpu_ready_ = false;
  }
}

void GuiAppModel::set_calibration(std::filesystem::path path) {
  const bool changed = files_.calibration != path;
  files_.calibration = std::move(path);
  if (files_.calibration->empty()) {
    files_.calibration.reset();
  }
  if (changed) {
    gpu_ready_ = false;
  }
}

void GuiAppModel::set_gpu_ready(bool ready) { gpu_ready_ = ready; }

void GuiAppModel::set_preview_controls(PreviewControls controls) {
  controls.clamp();
  preview_ = controls;
}

void GuiAppModel::set_default_codec(std::string codec) {
  if (codec.empty()) {
    throw std::invalid_argument("GUI default codec must not be empty");
  }
  settings_.default_codec = std::move(codec);
}

void GuiAppModel::set_default_quality(std::string quality) {
  if (quality.empty()) {
    throw std::invalid_argument("GUI default quality must not be empty");
  }
  settings_.default_quality = std::move(quality);
}

void GuiAppModel::set_default_blend_width(float blend_width) {
  PreviewControls controls = preview_;
  controls.blend_width = blend_width;
  controls.clamp();
  preview_ = controls;
  settings_.default_blend_width = controls.blend_width;
}

void GuiAppModel::set_ai_model(std::optional<std::filesystem::path> path) {
  settings_.ai_model_path = std::move(path);
  if (settings_.ai_model_path.has_value() && settings_.ai_model_path->empty()) {
    settings_.ai_model_path.reset();
  }
}

} // namespace reco::gui
