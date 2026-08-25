#include "reco/gui/app_model.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

template <typename Fn> void expect_invalid_argument(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

void preview_status_requires_inputs_and_gpu() {
  GuiAppModel model;
  expect_eq(preview_status_name(PreviewStatus::MissingInputs), std::string_view("missing_inputs"),
            "preview status name");
  expect_eq(preview_status_name(model.preview_status()), std::string_view("missing_inputs"),
            "empty preview status");
  expect_true(!model.can_export(), "empty model cannot export");

  model.set_left("/video/left.mp4");
  model.set_right("/video/right.mp4");
  model.set_calibration("/cal/match.json");
  expect_true(!model.can_export(), "complete files cannot export before gpu");
  expect_eq(preview_status_name(model.preview_status()), std::string_view("waiting_for_gpu"),
            "complete files wait for gpu");

  model.set_gpu_ready(true);
  expect_eq(preview_status_name(model.preview_status()), std::string_view("ready"),
            "gpu-ready preview status");
  expect_true(model.can_export(), "gpu-ready files can export");

  model.set_right("/video/right2.mp4");
  expect_eq(preview_status_name(model.preview_status()), std::string_view("waiting_for_gpu"),
            "file change invalidates gpu readiness");
  model.set_gpu_ready(true);
  expect_eq(preview_status_name(model.preview_status()), std::string_view("ready"),
            "gpu ready can be restored");

  model.set_left("");
  expect_eq(preview_status_name(model.preview_status()), std::string_view("missing_inputs"),
            "empty path clears selection");
}

void preview_controls_are_clamped() {
  GuiSettings settings;
  settings.default_blend_width = 0.25F;
  GuiAppModel model(settings);
  expect_eq(model.preview().blend_width, 0.25F, "preview blend initializes from settings");

  PreviewControls controls{
      .yaw = 1.0F, .pitch = 200.0F, .fov_degrees = 200.0F, .blend_width = 2.0F, .playing = true};
  model.set_preview_controls(controls);
  expect_eq(model.preview().yaw, 1.0F, "yaw preserved");
  expect_eq(model.preview().pitch, 89.0F, "pitch clamped");
  expect_eq(model.preview().fov_degrees, 150.0F, "fov clamped");
  expect_eq(model.preview().blend_width, 0.3F, "blend clamped");
  expect_true(model.preview().playing, "playing preserved");

  controls.reset_pose();
  expect_eq(controls.yaw, 0.0F, "reset yaw");
  expect_eq(controls.pitch, 0.0F, "reset pitch");
  expect_eq(controls.fov_degrees, 75.0F, "reset fov");
}

void export_request_uses_settings_and_selection() {
  GuiSettings settings;
  settings.default_codec = "hevc";
  settings.default_quality = "high";
  settings.default_blend_width = 0.2F;
  settings.ai_model_path = "/models/ball.onnx";
  GuiAppModel model(settings);
  expect_true(!model.export_request("out.mp4").has_value(), "incomplete export request");

  model.set_left("/video/left.mp4");
  model.set_right("/video/right.mp4");
  model.set_calibration("/cal/match.json");
  expect_true(!model.export_request("/exports/out.mp4").has_value(), "gpu-not-ready export");
  model.set_gpu_ready(true);
  const auto request = model.export_request("/exports/out.mp4");
  expect_true(request.has_value(), "complete export request");
  if (request.has_value()) {
    expect_eq(request->left.string(), std::string("/video/left.mp4"), "export left");
    expect_eq(request->right.string(), std::string("/video/right.mp4"), "export right");
    expect_eq(request->calibration.string(), std::string("/cal/match.json"), "export calibration");
    expect_eq(request->output.string(), std::string("/exports/out.mp4"), "export output");
    expect_eq(request->codec, std::string("hevc"), "export codec");
    expect_eq(request->quality, std::string("high"), "export quality");
    expect_eq(request->blend_width, 0.2F, "export blend");
    expect_eq(request->ai_model->string(), std::string("/models/ball.onnx"), "export ai model");
  }

  expect_invalid_argument([&] { (void)model.export_request(""); }, "empty export output");
  expect_invalid_argument([&] { model.set_default_codec(""); }, "empty codec");
  expect_invalid_argument([&] { model.set_default_quality(""); }, "empty quality");
  model.set_default_blend_width(2.0F);
  expect_eq(model.settings().default_blend_width, 0.3F, "default blend setter clamps settings");
  expect_eq(model.preview().blend_width, 0.3F, "default blend setter clamps preview");
  model.set_ai_model("");
  expect_true(!model.settings().ai_model_path.has_value(), "empty ai path clears");
}

} // namespace

int main() {
  preview_status_requires_inputs_and_gpu();
  preview_controls_are_clamped();
  export_request_uses_settings_and_selection();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
