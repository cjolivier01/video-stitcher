#include "reco/io/gpu_video_probe.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace reco::io;

namespace {

int failures = 0;

void expect_true(bool value, std::string_view message) {
  if (!value) {
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

template <typename Function>
void expect_probe_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const GpuVideoProbeError& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_invalid_argument(Function&& function, std::string_view fragment,
                             std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::filesystem::path find_fake_runtime_runfile() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || runfiles[0] == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    const auto filename = entry.path().filename().string();
    if (filename.find("fake_gstreamer_runtime") != std::string::npos &&
        (ends_with(filename, ".so") || ends_with(filename, ".dylib") ||
         ends_with(filename, ".dll"))) {
      return entry.path();
    }
  }
  throw std::runtime_error("fake GStreamer runtime runfile not found");
}

void set_environment(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void set_scenario(std::string_view value) {
  set_environment("RECO_FAKE_GST_SCENARIO", std::string(value));
}

GpuFileDecodeConfig config_for(const std::filesystem::path& path) {
  return {.path = path.string(), .container = GpuDecodeContainer::QuickTime};
}

std::vector<std::string> read_events(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::string> events;
  for (std::string line; std::getline(input, line);) {
    events.push_back(std::move(line));
  }
  return events;
}

void probe_contracts(const std::filesystem::path& video_path,
                     const std::filesystem::path& event_path) {
  constexpr std::uint64_t timeout_ns = 5'000'000'000ULL;
  set_scenario("probe-ok");
  std::filesystem::remove(event_path);
  const auto result = probe_gpu_video(config_for(video_path), timeout_ns);
  expect_eq(result.width, 3840U, "probe width");
  expect_eq(result.height, 2160U, "probe height");
  expect_eq(result.fps_numerator, 30'000U, "probe FPS numerator");
  expect_eq(result.fps_denominator, 1'001U, "probe FPS denominator");
  expect_true(std::abs(result.fps - 30'000.0 / 1'001.0) < 1e-12, "probe rational FPS");
  expect_eq(result.duration_ns, 10'000'000'000ULL, "probe duration");
  expect_eq(result.total_frames, 299ULL, "probe frame count truncates like Rust");
  expect_eq(result.video_stream_count, 1U, "probe moving-video stream count");
  expect_true(!result.duration_is_estimated, "known duration is not estimated");
  expect_true(result.discovery_complete, "successful discovery is complete");

  const auto events = read_events(event_path);
  expect_true(std::find(events.begin(), events.end(), "discover-uri") != events.end(),
              "probe invokes parser-backed discovery");
  expect_true(std::find(events.begin(), events.end(), "parse") == events.end(),
              "probe does not construct a decode pipeline");
  expect_true(std::find(events.begin(), events.end(), "pull") == events.end(),
              "probe does not materialize frames");

  set_scenario("probe-duration-unknown");
  const auto unknown_duration = probe_gpu_video(config_for(video_path), timeout_ns);
  expect_eq(unknown_duration.duration_ns, 60'000'000'000ULL,
            "unknown duration uses Rust-compatible 60-second estimate");
  expect_eq(unknown_duration.total_frames, 1'798ULL,
            "unknown duration estimate derives frame count from rational FPS");
  expect_true(unknown_duration.duration_is_estimated, "unknown duration is marked estimated");

  set_scenario("probe-duration-zero");
  expect_true(probe_gpu_video(config_for(video_path), timeout_ns).duration_is_estimated,
              "zero duration uses explicit estimate");

  set_scenario("probe-missing-plugins");
  const auto incomplete = probe_gpu_video(config_for(video_path), timeout_ns);
  expect_true(!incomplete.discovery_complete, "missing plugins preserve usable metadata verdict");

  set_scenario("probe-multiple");
  expect_eq(probe_gpu_video(config_for(video_path), timeout_ns).video_stream_count, 2U,
            "all moving-video streams counted");

  set_scenario("probe-image-then-video");
  const auto image_then_video = probe_gpu_video(config_for(video_path), timeout_ns);
  expect_eq(image_then_video.video_stream_count, 1U,
            "attached image is excluded from moving-video stream count");
  expect_eq(image_then_video.width, 3840U, "moving video selected after attached image");
}

void invalid_inputs_fail(const std::filesystem::path& video_path) {
  constexpr std::uint64_t timeout_ns = 5'000'000'000ULL;
  expect_invalid_argument([&] { (void)probe_gpu_video({}, timeout_ns); }, "path is required",
                          "empty path rejected");
  expect_invalid_argument([&] { (void)probe_gpu_video(config_for(video_path), 0); }, "timeout",
                          "zero timeout rejected");
  expect_invalid_argument(
      [&] {
        (void)probe_gpu_video(config_for(video_path), std::numeric_limits<std::uint64_t>::max());
      },
      "timeout", "infinite timeout rejected");

  expect_probe_error(
      [&] {
        (void)probe_gpu_video(config_for(video_path.parent_path() / "missing.mp4"), timeout_ns);
      },
      "not a readable regular file", "missing input rejected before discovery");

  for (const auto& [scenario_name, fragment] :
       std::array<std::pair<std::string_view, std::string_view>, 8>{
           {{"old-version", "1.10 or newer"},
            {"init-error", "fake initialization failure"},
            {"probe-uri-error", "fake URI conversion failure"},
            {"probe-new-error", "fake discoverer construction failure"},
            {"probe-discover-error", "fake discovery failure"},
            {"probe-timeout", "timed out"},
            {"probe-no-video", "no moving-video stream"},
            {"probe-image-only", "no moving-video stream"}}}) {
    set_scenario(scenario_name);
    expect_probe_error([&] { (void)probe_gpu_video(config_for(video_path), timeout_ns); }, fragment,
                       scenario_name);
  }

  set_scenario("probe-bad-fps");
  expect_probe_error([&] { (void)probe_gpu_video(config_for(video_path), timeout_ns); },
                     "invalid frame rate", "invalid FPS rejected");
  set_scenario("probe-bad-dimensions");
  expect_probe_error([&] { (void)probe_gpu_video(config_for(video_path), timeout_ns); },
                     "zero frame dimensions", "invalid dimensions rejected");
}

} // namespace

int main() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  const auto runtime = find_fake_runtime_runfile();
  set_environment("RECO_GSTREAMER_DYLIB_PATH", runtime.string());
  set_environment("RECO_GSTPBUTILS_DYLIB_PATH", runtime.string());
  set_environment("RECO_GLIB_DYLIB_PATH", runtime.string());
  set_environment("RECO_GOBJECT_DYLIB_PATH", runtime.string());

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto video_path = std::filesystem::temp_directory_path() /
                          ("reco_gpu_probe_" + std::to_string(unique) + ".mp4");
  const auto event_path = std::filesystem::temp_directory_path() /
                          ("reco_gpu_probe_events_" + std::to_string(unique) + ".txt");
  {
    std::ofstream video(video_path, std::ios::binary);
    video << "fake video container";
  }
  set_environment("RECO_FAKE_GST_EVENT_PATH", event_path.string());

  probe_contracts(video_path, event_path);
  invalid_inputs_fail(video_path);

  std::filesystem::remove(video_path);
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
