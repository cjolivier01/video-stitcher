#pragma once

#include "reco/gui/app_model.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace reco::gui {

struct GuiRuntimeProbe {
  bool cuda_available = false;
  std::string cuda_detail;
  bool gstreamer_available = false;
  std::string gstreamer_detail;
  bool npp_available = false;
  std::string npp_detail;
};

enum class PreviewRuntimeState {
  MissingInputs,
  MissingCuda,
  MissingGstreamer,
  MissingNpp,
  PreviewBridgeNotPorted,
  Ready,
};

struct PreviewRuntimeReadiness {
  PreviewRuntimeState state = PreviewRuntimeState::MissingInputs;
  bool ready = false;
  std::string detail;
};

[[nodiscard]] std::string_view preview_runtime_state_name(PreviewRuntimeState state);
[[nodiscard]] bool preview_runtime_probe_required(const GuiFileSelection& files);
[[nodiscard]] GuiRuntimeProbe probe_gui_runtime();
[[nodiscard]] PreviewRuntimeReadiness evaluate_preview_runtime(const GuiFileSelection& files,
                                                              const GuiRuntimeProbe& runtime);
[[nodiscard]] std::optional<std::string> export_blocked_reason(PreviewRuntimeReadiness readiness);

} // namespace reco::gui
