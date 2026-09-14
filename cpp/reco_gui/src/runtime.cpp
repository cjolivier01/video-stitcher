#include "reco/gui/runtime.hpp"

#include "reco/core/cuda_backend.hpp"
#include "reco/io/gstreamer.hpp"

#include <utility>

namespace reco::gui {

std::string_view preview_runtime_state_name(PreviewRuntimeState state) {
  switch (state) {
  case PreviewRuntimeState::MissingInputs:
    return "missing_inputs";
  case PreviewRuntimeState::MissingCuda:
    return "missing_cuda";
  case PreviewRuntimeState::MissingGstreamer:
    return "missing_gstreamer";
  case PreviewRuntimeState::Ready:
    return "ready";
  }
  return "unknown";
}

bool preview_runtime_probe_required(const GuiFileSelection& files) { return files.complete(); }

GuiRuntimeProbe probe_gui_runtime() {
  GuiRuntimeProbe probe;
  probe.cuda_available = reco::core::CudaBackend::is_available();
  probe.cuda_detail = probe.cuda_available ? "CUDA driver/runtime available"
                                           : reco::core::CudaBackend::availability_error();

  const auto gst = reco::io::probe_gstreamer_runtime();
  probe.gstreamer_available = gst.available;
  probe.gstreamer_detail = gst.available ? gst.library : gst.error;

  return probe;
}

PreviewRuntimeReadiness evaluate_preview_runtime(const GuiFileSelection& files,
                                                 const GuiRuntimeProbe& runtime) {
  if (!files.complete()) {
    return {.state = PreviewRuntimeState::MissingInputs,
            .ready = false,
            .detail = "select left video, right video, and calibration"};
  }
  if (!runtime.cuda_available) {
    return {.state = PreviewRuntimeState::MissingCuda,
            .ready = false,
            .detail = "CUDA is required for GPU preview: " + runtime.cuda_detail};
  }
  if (!runtime.gstreamer_available) {
    return {.state = PreviewRuntimeState::MissingGstreamer,
            .ready = false,
            .detail = "GStreamer is required for GPU preview ingest: " + runtime.gstreamer_detail};
  }
  return {.state = PreviewRuntimeState::Ready,
          .ready = true,
          .detail = "CUDA and GStreamer prerequisites are available"};
}

std::optional<std::string> export_blocked_reason(PreviewRuntimeReadiness readiness) {
  if (readiness.ready) {
    return std::nullopt;
  }
  return readiness.detail.empty() ? std::optional<std::string>("GPU preview is not ready")
                                  : std::optional<std::string>(std::move(readiness.detail));
}

} // namespace reco::gui
