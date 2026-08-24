#include "reco/control/gopro.hpp"
#include "reco/control/intent_translator.hpp"
#include "reco/control/keyboard.hpp"
#include "reco/control/pose_control.hpp"
#include "reco/core/rig_correction.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using namespace reco::control;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U>
void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual
              << " tolerance=" << tolerance << '\n';
    ++failures;
  }
}

PoseControl fresh_pose() { return PoseControl{}; }

void pose_defaults_match_rust_plan() {
  const PoseControlConfig cfg;
  expect_near(cfg.fov_min_degrees, 20.0F, 0.0F, "fov_min");
  expect_near(cfg.fov_max_degrees, 150.0F, 0.0F, "fov_max");
  expect_near(cfg.drag_deg_per_pixel, 0.1F, 0.0F, "drag sensitivity");
  expect_near(cfg.wheel_fov_per_tick, 3.0F, 0.0F, "wheel sensitivity");
  expect_true(cfg.rest_pose.fov_degrees.has_value(), "rest fov present");
  expect_near(*cfg.rest_pose.fov_degrees, 75.0F, 0.0F, "rest fov");
}

void pose_drag_right_decreases_yaw_with_configured_sensitivity() {
  auto pose = fresh_pose();
  pose.apply_drag(100.0F, 0.0F);
  expect_true(pose.target_pose().yaw < 0.0F, "drag right decreases yaw");
  expect_near(pose.target_pose().yaw, -10.0F * 0.017453292519943295769F, 1.0e-5F,
              "drag sensitivity math");
}

void pose_wheel_clamps_to_fov_range() {
  auto pose = fresh_pose();
  for (int i = 0; i < 1000; ++i) {
    pose.apply_wheel(1.0F);
  }
  expect_near(*pose.target_pose().fov_degrees, pose.config().fov_min_degrees, 0.0F,
              "wheel clamps min");
  for (int i = 0; i < 1000; ++i) {
    pose.apply_wheel(-1.0F);
  }
  expect_near(*pose.target_pose().fov_degrees, pose.config().fov_max_degrees, 0.0F,
              "wheel clamps max");
}

void pose_tick_eases_and_can_snap() {
  auto pose = fresh_pose();
  pose.apply_drag(100.0F, 50.0F);
  const float before = pose.current_yaw_rad();
  pose.tick();
  expect_true(before != pose.current_yaw_rad(), "tick moves yaw");
  expect_near(std::abs(pose.current_yaw_rad() - pose.target_pose().yaw),
              std::abs(before - pose.target_pose().yaw) * 0.7F, 1.0e-5F, "tick smoothing");
  pose.tick_with(1.0F);
  expect_near(pose.current_yaw_rad(), pose.target_pose().yaw, 1.0e-6F, "tick_with one snaps");
}

void pose_reset_and_fov_max_clamp() {
  auto pose = fresh_pose();
  pose.apply_drag(1000.0F, 500.0F);
  pose.set_target_fov(100.0F);
  pose.tick_with(1.0F);
  pose.set_fov_max_degrees(80.0F);
  expect_near(*pose.target_pose().fov_degrees, 80.0F, 0.0F, "target fov max clamp");
  expect_near(pose.current_fov_deg(), 80.0F, 0.0F, "current fov max clamp");
  pose.apply_hotkey(HotkeyIntent::Reset);
  expect_near(pose.target_pose().yaw, pose.config().rest_pose.yaw, 1.0e-6F, "reset yaw");
}

void render_pitch_matches_rust_rig_correction_formula() {
  constexpr float pi = 3.14159265358979323846F;
  const float tilt = 15.0F * pi / 180.0F;
  expect_near(reco::core::render_pitch(0.0F, 0.0F, tilt), -tilt, 1.0e-6F,
              "render pitch at yaw zero");
  expect_near(reco::core::render_pitch(pi / 2.0F, 0.0F, tilt), 0.0F, 1.0e-6F,
              "render pitch at yaw 90");
  expect_near(reco::core::render_pitch(pi, 0.0F, tilt), tilt, 1.0e-6F,
              "render pitch at yaw pi");
}

void keyboard_poll_drains_in_order_and_appends() {
  KeyboardTransport transport;
  expect_eq(KeyboardTransport::name(), "keyboard", "keyboard name");
  transport.push(HotkeyControlIntent{HotkeyIntent::ZoomIn});
  transport.push(CaptureControlIntent{CaptureIntent::Snapshot});
  expect_eq(transport.pending(), 2U, "pending before poll");

  std::vector<ControlIntent> out{HotkeyControlIntent{HotkeyIntent::YawLeft}};
  const auto drained = transport.poll(out);
  expect_eq(drained, 2U, "poll drained count");
  expect_eq(out.size(), 3U, "poll appends");
  expect_eq(transport.pending(), 0U, "poll drains queue");
  expect_true(std::holds_alternative<HotkeyControlIntent>(out[1]), "first drained hotkey");
  expect_true(std::holds_alternative<CaptureControlIntent>(out[2]), "second drained capture");
}

void translator_dispatches_pose_and_handlers() {
  auto pose = fresh_pose();
  std::optional<QualityIntent> quality;
  std::optional<CaptureIntent> capture;
  IntentTranslator translator(pose);
  translator.with_quality_handler([&](const QualityIntent& intent) { quality = intent; })
      .with_capture_handler([&](CaptureIntent intent) { capture = intent; });

  translator.dispatch(PoseControlIntent{PoseDeltaYawRad{0.1F}});
  expect_near(pose.target_pose().yaw, 0.1F, 1.0e-6F, "pose delta yaw");

  translator.dispatch(QualityControlIntent{QualitySetBitrate{5'000'000}});
  expect_true(quality.has_value(), "quality handler called");
  expect_true(std::holds_alternative<QualitySetBitrate>(*quality), "quality bitrate variant");
  expect_eq(std::get<QualitySetBitrate>(*quality).value, 5'000'000U, "quality bitrate value");

  translator.dispatch(CaptureControlIntent{CaptureIntent::Snapshot});
  expect_true(capture.has_value(), "capture handler called");
  expect_true(*capture == CaptureIntent::Snapshot, "capture snapshot value");
}

void intent_json_preserves_rust_serde_shape_for_golden_cases() {
  expect_eq(to_json(ControlIntent{HotkeyControlIntent{HotkeyIntent::ZoomIn}}),
            "{\"kind\":\"hotkey\",\"data\":\"zoom_in\"}", "hotkey serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseSetYawRad{1.0F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"set_yaw_rad\",\"value\":1.0}}",
            "set yaw serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseSetPitchRad{2.0F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"set_pitch_rad\",\"value\":2.0}}",
            "set pitch serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseSetFovDeg{60.0F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"set_fov_deg\",\"value\":60.0}}",
            "set fov serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseDeltaYawRad{0.1F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"delta_yaw_rad\",\"value\":0.1}}",
            "delta yaw serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseDeltaPitchRad{-0.2F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"delta_pitch_rad\",\"value\":-0.2}}",
            "delta pitch serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseDeltaFovDeg{3.0F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"delta_fov_deg\",\"value\":3.0}}",
            "delta fov serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseDeltaYawRad{1.0e-5F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"delta_yaw_rad\",\"value\":0.00001}}",
            "small float serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseDeltaYawRad{1.0e-6F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"delta_yaw_rad\",\"value\":0.000001}}",
            "smaller float serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseDeltaYawRad{1.0e-7F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"delta_yaw_rad\",\"value\":1e-7}}",
            "scientific small float serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseSetFovDeg{1.0e7F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"set_fov_deg\",\"value\":10000000.0}}",
            "large float serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseSetFovDeg{1.0e20F}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"set_fov_deg\",\"value\":1e+20}}",
            "scientific large float serde shape");
  expect_eq(to_json(ControlIntent{PoseControlIntent{PoseReset{}}}),
            "{\"kind\":\"pose\",\"data\":{\"action\":\"reset\"}}", "pose reset serde shape");
  expect_eq(to_json(ControlIntent{QualityControlIntent{QualitySetCodec{"h264"}}}),
            "{\"kind\":\"quality\",\"data\":{\"action\":\"set_codec\",\"value\":\"h264\"}}",
            "codec serde shape");
  expect_eq(to_json(ControlIntent{QualityControlIntent{QualitySetCodec{"h\\n264\tmain"}}}),
            "{\"kind\":\"quality\",\"data\":{\"action\":\"set_codec\",\"value\":\"h\\\\n264\\tmain\"}}",
            "codec string escaping");
  expect_eq(to_json(ControlIntent{QualityControlIntent{QualitySetBitrate{5'000'000}}}),
            "{\"kind\":\"quality\",\"data\":{\"action\":\"set_bitrate\",\"value\":5000000}}",
            "bitrate serde shape");
  expect_eq(to_json(ControlIntent{QualityControlIntent{QualitySetResolution{1920, 1080}}}),
            "{\"kind\":\"quality\",\"data\":{\"action\":\"set_resolution\",\"value\":{\"width\":1920,\"height\":1080}}}",
            "resolution serde shape");
  expect_eq(to_json(ControlIntent{QualityControlIntent{QualitySetPreset{"fast"}}}),
            "{\"kind\":\"quality\",\"data\":{\"action\":\"set_preset\",\"value\":\"fast\"}}",
            "preset serde shape");
  expect_eq(to_json(ControlIntent{QualityControlIntent{QualitySetCrf{23}}}),
            "{\"kind\":\"quality\",\"data\":{\"action\":\"set_crf\",\"value\":23}}",
            "crf serde shape");
  expect_eq(to_json(ControlIntent{CaptureControlIntent{CaptureIntent::SaveReplay}}),
            "{\"kind\":\"capture\",\"data\":\"save_replay\"}", "capture serde shape");
  expect_eq(to_json(ControlIntent{ModelSelectControlIntent{ModelSetDetectorModel{"ball.onnx"}}}),
            "{\"kind\":\"model_select\",\"data\":{\"action\":\"set_detector_model\",\"value\":\"ball.onnx\"}}",
            "model path serde shape");
  expect_eq(to_json(ControlIntent{ModelSelectControlIntent{ModelSetDetectionInterval{3}}}),
            "{\"kind\":\"model_select\",\"data\":{\"action\":\"set_detection_interval\",\"value\":3}}",
            "model interval serde shape");
  expect_eq(to_json(ControlIntent{ModelSelectControlIntent{ModelDisableDetection{}}}),
            "{\"kind\":\"model_select\",\"data\":{\"action\":\"disable_detection\"}}",
            "model disable serde shape");
}

void golden_fixture_is_wired_into_bazel_test() {
  std::ifstream input("cpp/tests/fixtures/reco_control_golden.json");
  expect_true(input.good(), "golden fixture is available as Bazel test data");
  std::string first_line;
  std::getline(input, first_line);
  expect_eq(first_line, to_json(ControlIntent{HotkeyControlIntent{HotkeyIntent::ZoomIn}}),
            "fixture first line matches hotkey golden");
}

void gopro_parses_mocked_status_and_command_responses() {
  const auto status =
      reco::control::gopro::parse_status_json("{\"status\":{\"1\":1,\"2\":3,\"6\":0,\"8\":0,"
                                              "\"10\":1,\"13\":42,\"54\":123456,"
                                              "\"68\":1,\"70\":87,\"85\":0,\"86\":2},"
                                              "\"settings\":{}}");
  expect_true(status.has_value(), "status parses");
  expect_true(status->battery_present.value_or(false), "battery present");
  expect_eq(*status->battery_bars, 3, "battery bars");
  expect_true(status->encoding.value_or(false), "encoding true");
  expect_true(status->battery_percent.has_value(), "battery present");
  expect_eq(*status->battery_percent, 87, "battery value");
  expect_eq(*status->encoding_duration_secs, 42, "encoding duration");
  expect_true(!status->busy.value_or(true), "busy false");
  expect_eq(*status->sd_remaining_kb, 123456ULL, "sd remaining");
  expect_true(status->gps_lock.value_or(false), "gps lock");
  expect_eq(*status->orientation, 2, "orientation");
  const auto settings_only =
      reco::control::gopro::parse_status_json("{\"status\":{},\"settings\":{\"10\":1}}");
  expect_true(settings_only.has_value(), "settings-only status parses");
  expect_true(!settings_only->encoding.has_value(), "status parser stays inside status object");

  const auto result = reco::control::gopro::parse_command_response("{\"ok\":true}");
  expect_true(result.ok, "command ok");
  expect_eq(result.message, "ok", "command message");
  expect_eq(reco::control::gopro::connection_mode_name(reco::control::gopro::ConnectionMode::Usb),
            "usb", "connection mode");
  expect_eq(*reco::control::gopro::usb_base_url("ABC051"), "http://172.20.151.51:8080",
            "usb base url");
  expect_true(!reco::control::gopro::usb_base_url("bad").has_value(), "bad usb suffix");
  expect_eq(reco::control::gopro::wifi_base_url(), "http://10.5.5.9:8080", "wifi url");
  expect_eq(reco::control::gopro::endpoint_state(), "/gopro/camera/state", "state endpoint");
  expect_eq(reco::control::gopro::endpoint_shutter_start(), "/gopro/camera/shutter/start",
            "shutter start endpoint");
  expect_eq(reco::control::gopro::endpoint_set_setting(2, 100),
            "/gopro/camera/setting?setting=2&option=100", "setting endpoint");
  const auto preset = reco::control::gopro::sports_preset_endpoints(100, 5, 0);
  expect_eq(preset.size(), 7U, "sports preset endpoint count");
  expect_eq(preset[4], "/gopro/camera/setting?setting=135&option=0", "hypersmooth off");
  expect_eq(reco::control::gopro::endpoint_webcam_start(12, 4, 8554, "RTSP"),
            "/gopro/webcam/start?res=12&fov=4&port=8554&protocol=RTSP", "webcam start");
  expect_eq(reco::control::gopro::endpoint_digital_zoom(150),
            "/gopro/camera/digital_zoom?percent=100", "digital zoom clamps");
  expect_eq(reco::control::gopro::endpoint_media_telemetry("/DCIM/100GOPRO/GX010001.MP4"),
            "/gopro/media/telemetry?path=/DCIM/100GOPRO/GX010001.MP4", "telemetry endpoint");
  expect_eq(reco::control::gopro::endpoint_enable_wired_usb(),
            "/gopro/camera/control/wired_usb?p=1", "wired usb endpoint");
}

} // namespace

int main() {
  pose_defaults_match_rust_plan();
  pose_drag_right_decreases_yaw_with_configured_sensitivity();
  pose_wheel_clamps_to_fov_range();
  pose_tick_eases_and_can_snap();
  pose_reset_and_fov_max_clamp();
  render_pitch_matches_rust_rig_correction_formula();
  keyboard_poll_drains_in_order_and_appends();
  translator_dispatches_pose_and_handlers();
  intent_json_preserves_rust_serde_shape_for_golden_cases();
  golden_fixture_is_wired_into_bazel_test();
  gopro_parses_mocked_status_and_command_responses();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
