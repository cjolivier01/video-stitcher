#include "reco/gui/settings.hpp"
#include "reco/gui/toast.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <thread>

#include <nlohmann/json.hpp>

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

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

class ScopedConfigDir {
public:
  ScopedConfigDir() {
    path_ = std::filesystem::temp_directory_path() / "reco_gui_cpp_settings_tests";
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
#if defined(_WIN32)
    _putenv_s("RECO_CONFIG_DIR", path_.string().c_str());
#else
    setenv("RECO_CONFIG_DIR", path_.string().c_str(), 1);
#endif
  }

  ~ScopedConfigDir() {
#if defined(_WIN32)
    _putenv_s("RECO_CONFIG_DIR", "");
#else
    unsetenv("RECO_CONFIG_DIR");
#endif
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

private:
  std::filesystem::path path_;
};

void settings_defaults_match_rust() {
  const GuiSettings settings;
  expect_eq(settings.default_codec, std::string("h264"), "default codec");
  expect_eq(settings.default_quality, std::string("balanced"), "default quality");
  expect_near(settings.default_blend_width, 0.05F, 1.0e-6F, "default blend");
  expect_true(settings.recent_left.empty(), "recent left empty");
  expect_eq(settings.recording_codec, std::string("h264"), "recording codec");
  expect_eq(settings.recording_quality, std::string("balanced"), "recording quality");
  expect_eq(settings.preview_aspect, std::string("auto"), "preview aspect");
  expect_true(!settings.telemetry_enabled, "telemetry disabled");
  expect_true(settings.dark_mode, "dark mode default");
}

void settings_missing_fields_use_defaults() {
  const auto json = nlohmann::json::parse(R"json({ "default_codec": "hevc" })json");
  const auto settings = json.get<GuiSettings>();
  expect_eq(settings.default_codec, std::string("hevc"), "loaded codec");
  expect_eq(settings.default_quality, std::string("balanced"), "missing quality default");
  expect_true(settings.recent_left.empty(), "missing recent left default");
  expect_true(settings.dark_mode, "missing dark mode default");

  const auto malformed = nlohmann::json::parse(R"json({ "window_size": [1280] })json");
  try {
    (void)malformed.get<GuiSettings>();
    expect_true(false, "malformed window size rejected");
  } catch (const nlohmann::json::exception&) {
    expect_true(true, "malformed window size rejected");
  }
}

void settings_round_trip_persists_gui_namespace() {
  ScopedConfigDir config;
  GuiSettings settings;
  settings.default_codec = "av1";
  settings.window_size = std::pair<std::uint32_t, std::uint32_t>{1440, 900};
  settings.ai_model_path = std::filesystem::path("/models/ball.onnx");
  settings.save();

  const auto loaded = GuiSettings::load();
  expect_eq(loaded.default_codec, std::string("av1"), "roundtrip codec");
  expect_true(loaded.window_size.has_value(), "roundtrip window size");
  expect_eq(loaded.window_size->first, 1440U, "roundtrip window width");
  expect_eq(loaded.ai_model_path->string(), std::string("/models/ball.onnx"),
            "roundtrip model path");

  settings.push_left(std::filesystem::path("/video/left.mp4"));
  const auto pushed = GuiSettings::load();
  expect_eq(pushed.recent_left.size(), 1U, "push left saved");
  expect_eq(pushed.recent_left.entries()[0].string(), std::string("/video/left.mp4"),
            "push left path");
}

void toast_manager_matches_rust() {
  using namespace std::chrono_literals;
  expect_eq(severity_name(Severity::Info), std::string_view("info"), "info severity");
  expect_eq(default_ttl(Severity::Warn), 7000ms, "warn ttl");

  ToastManager manager;
  const auto a = manager.push(Severity::Info, "A", "");
  const auto b = manager.push(Severity::Info, "B", "");
  const auto c = manager.push(Severity::Info, "C", "");
  expect_eq(a, 1, "toast id a");
  expect_eq(b, 2, "toast id b");
  expect_eq(c, 3, "toast id c");
  expect_eq(manager.size(), 3U, "toast size after push");

  ToastManager wrapping(4, std::numeric_limits<std::int32_t>::max());
  const auto max_id = wrapping.push(Severity::Info, "max", "");
  const auto min_id = wrapping.push(Severity::Info, "min", "");
  expect_eq(max_id, std::numeric_limits<std::int32_t>::max(), "toast max id");
  expect_eq(min_id, std::numeric_limits<std::int32_t>::min(), "toast wraps to min id");

  ToastManager zero_skip(4, -1);
  const auto minus_one = zero_skip.push(Severity::Info, "minus one", "");
  const auto one = zero_skip.push(Severity::Info, "one", "");
  expect_eq(minus_one, -1, "toast minus one id");
  expect_eq(one, 1, "toast skips zero id");

  manager.dismiss(a);
  expect_eq(manager.size(), 2U, "dismiss removes");
  manager.dismiss(999);
  expect_eq(manager.size(), 2U, "missing dismiss no-op");

  const auto latest = manager.latest();
  expect_true(latest.has_value(), "latest exists");
  expect_eq(latest->second, std::string_view("C"), "latest title");

  ToastManager capped(2);
  (void)capped.push(Severity::Info, "1", "");
  (void)capped.push(Severity::Info, "2", "");
  (void)capped.push(Severity::Info, "3", "");
  expect_eq(capped.size(), 2U, "cap evicts oldest");
  expect_eq(capped.entries()[0].title, std::string("2"), "cap first");
  expect_eq(capped.entries()[1].title, std::string("3"), "cap second");

  ToastManager expiring;
  (void)expiring.push_with_ttl(Severity::Info, "quick", "", 1ms);
  (void)expiring.push_with_ttl(Severity::Info, "slow", "", 60s);
  std::this_thread::sleep_for(20ms);
  expect_true(expiring.expire(std::chrono::steady_clock::now()), "expire changed");
  expect_eq(expiring.size(), 1U, "expire leaves slow");
}

} // namespace

int main() {
  settings_defaults_match_rust();
  settings_missing_fields_use_defaults();
  settings_round_trip_persists_gui_namespace();
  toast_manager_matches_rust();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
