#include "reco/gui/runtime.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

using namespace reco::gui;

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

GuiFileSelection complete_files() {
  return {.left = "/video/left.mp4",
          .right = "/video/right.mp4",
          .calibration = "/calibration/match.json"};
}

GuiRuntimeProbe ready_runtime() {
  return {.cuda_available = true,
          .cuda_detail = "cuda",
          .gstreamer_available = true,
          .gstreamer_detail = "gstreamer"};
}

void runtime_state_names_are_stable() {
  expect_eq(preview_runtime_state_name(PreviewRuntimeState::MissingInputs),
            std::string_view("missing_inputs"), "missing inputs state name");
  expect_eq(preview_runtime_state_name(PreviewRuntimeState::Ready), std::string_view("ready"),
            "ready state name");
}

void preview_readiness_is_ordered_by_user_action_then_backends() {
  auto runtime = ready_runtime();
  auto readiness = evaluate_preview_runtime({}, runtime);
  expect_true(!preview_runtime_probe_required({}), "missing inputs do not require backend probe");
  expect_true(readiness.state == PreviewRuntimeState::MissingInputs, "missing inputs first");
  expect_true(!readiness.ready, "missing inputs not ready");
  expect_true(preview_runtime_probe_required(complete_files()),
              "complete files require backend probe");

  runtime = ready_runtime();
  runtime.cuda_available = false;
  runtime.cuda_detail = "no cuda";
  readiness = evaluate_preview_runtime(complete_files(), runtime);
  expect_true(readiness.state == PreviewRuntimeState::MissingCuda,
              "missing CUDA before video libs");
  expect_true(readiness.detail.find("no cuda") != std::string::npos, "CUDA detail included");

  runtime = ready_runtime();
  runtime.gstreamer_available = false;
  runtime.gstreamer_detail = "no gst";
  readiness = evaluate_preview_runtime(complete_files(), runtime);
  expect_true(readiness.state == PreviewRuntimeState::MissingGstreamer, "missing GStreamer");
}

void available_gpu_prerequisites_allow_controller_startup() {
  const auto readiness = evaluate_preview_runtime(complete_files(), ready_runtime());
  expect_true(readiness.state == PreviewRuntimeState::Ready,
              "available prerequisites permit controller startup");
  expect_true(readiness.ready, "available prerequisites are ready");

  const auto blocked = export_blocked_reason(readiness);
  expect_true(!blocked.has_value(), "ready prerequisites do not block export");
}

} // namespace

int main() {
  runtime_state_names_are_stable();
  preview_readiness_is_ordered_by_user_action_then_backends();
  available_gpu_prerequisites_allow_controller_startup();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
