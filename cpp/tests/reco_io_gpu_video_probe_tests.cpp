#include "reco/io/gpu_video_probe.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
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

GpuFileDecodeConfig container_config(const std::filesystem::path& path) {
  return {.path = path.string(),
          .codec = GpuDecodeCodec::H264,
          .elementary_stream = false,
          .container = GpuDecodeContainer::QuickTime,
          .max_buffers = 4,
          .drop = false};
}

GpuFileDecodeConfig elementary_config(const std::filesystem::path& path, GpuDecodeCodec codec) {
  return {.path = path.string(),
          .codec = codec,
          .elementary_stream = true,
          .container = std::nullopt,
          .max_buffers = 4,
          .drop = false};
}

std::vector<std::string> read_events(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::string> events;
  for (std::string line; std::getline(input, line);) {
    events.push_back(std::move(line));
  }
  return events;
}

bool has_event(const std::vector<std::string>& events, std::string_view value) {
  return std::find(events.begin(), events.end(), value) != events.end();
}

void probe_contracts(const std::filesystem::path& video_path,
                     const std::filesystem::path& event_path) {
  constexpr std::uint64_t timeout_ns = 5'000'000'000ULL;
  set_scenario("probe-ok");
  std::filesystem::remove(event_path);
  const auto result = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(result.width, 3840U, "parser-visible width");
  expect_eq(result.height, 2160U, "parser-visible height");
  expect_eq(result.fps_numerator, 30'000U, "parser FPS numerator");
  expect_eq(result.fps_denominator, 1'001U, "parser FPS denominator");
  expect_true(std::abs(result.fps - 30'000.0 / 1'001.0) < 1e-12, "rational FPS");
  expect_eq(result.duration_ns, 10'000'000'000ULL, "queried duration");
  expect_eq(result.total_frames, 299ULL, "exact rational frame count truncates");
  expect_true(!result.duration_is_estimated, "known duration is not estimated");

  const auto events = read_events(event_path);
  expect_true(has_event(events, "parse-probe"), "probe constructs parser-only pipeline");
  expect_true(has_event(events, "probe-codec-filter"),
              "container probe filters out attached images and unsupported codecs");
  expect_true(has_event(events, "probe-parsebin"),
              "container probe uses the production parser selection topology");
  expect_true(has_event(events, "probe-decoder-caps"),
              "probe constrains parser output to the NVDEC sink contract");
  expect_true(has_event(events, "state-playing"),
              "probe runs only the backpressured compressed parser branch");
  expect_true(has_event(events, "state-null"), "probe resets pipeline before release");
  expect_true(!has_event(events, "decoder-element"), "probe does not construct a decoder");
  expect_true(!has_event(events, "pull"), "probe does not pull a decoded frame");
  expect_true(has_event(events, "pull-probe"),
              "probe seeks only compressed access-unit timestamps");
  expect_true(has_event(events, "sample-caps"),
              "probe reads caps from the retained compressed sample");
  expect_true(!has_event(events, "pad-current-caps"), "probe does not race mutable live-pad caps");
  expect_true(!has_event(events, "map"), "probe does not map frame memory");
  expect_true(!has_event(events, "raw-video-caps"), "probe never negotiates raw video caps");

  set_scenario("probe-duration-unknown");
  const auto unknown_duration = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_duration.duration_ns, 60'000'000'000ULL,
            "unknown duration uses Rust-compatible estimate");
  expect_eq(unknown_duration.total_frames, 1'798ULL,
            "fallback frame count uses exact rational arithmetic");
  expect_true(unknown_duration.duration_is_estimated, "unknown duration is marked estimated");

  set_scenario("probe-duration-zero");
  expect_true(probe_gpu_video(container_config(video_path), timeout_ns).duration_is_estimated,
              "zero duration uses explicit estimate");

  set_scenario("probe-exact-frame-count");
  expect_eq(probe_gpu_video(container_config(video_path), timeout_ns).total_frames, 3ULL,
            "exact rational arithmetic avoids floating-point off-by-one");

  set_scenario("probe-integral-frame-count");
  expect_eq(probe_gpu_video(container_config(video_path), timeout_ns).total_frames, 30'000ULL,
            "integral rational frame count is not rounded down");

  set_scenario("probe-frame-count-overflow");
  expect_eq(probe_gpu_video(container_config(video_path), timeout_ns).total_frames,
            276'424'736'369ULL, "large duration remains exact without intermediate overflow");

  set_scenario("probe-duration-mismatch");
  const auto selected_duration = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_true(selected_duration.duration_ns > 900'000'000ULL &&
                  selected_duration.duration_ns < 1'100'000'000ULL,
              "selected stream duration excludes longer unrelated tracks");
  expect_eq(selected_duration.total_frames, 30ULL,
            "selected stream frame count excludes longer unrelated tracks");
  expect_true(!selected_duration.duration_is_estimated,
              "timestamp-correlated selected duration is not estimated");

  set_scenario("probe-delayed-stream");
  const auto delayed_stream = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(delayed_stream.duration_ns, 1'000'000'000ULL,
            "selected stream duration excludes its delayed container start");
  expect_eq(delayed_stream.total_frames, 30ULL,
            "delayed selected stream reports only its own frame count");
  expect_true(!delayed_stream.duration_is_estimated,
              "delayed selected stream duration remains timestamp-correlated");

  set_scenario("probe-nonzero-origin");
  const auto nonzero_origin = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(nonzero_origin.duration_ns, 2'000'000'000ULL,
            "nonzero timeline origin does not shorten a duration span");
  expect_eq(nonzero_origin.total_frames, 60ULL,
            "short demux duration still searches the final access unit");

  set_scenario("probe-decode-order-origin");
  const auto decode_order_origin = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(decode_order_origin.duration_ns, 1'000'000'000ULL,
            "first decode-order access unit does not define the timeline origin");
  expect_eq(decode_order_origin.total_frames, 30ULL,
            "decode-ordered first PTS does not drop presentation frames");

  set_scenario("probe-caps-runahead");
  const auto sample_caps = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(sample_caps.width, 854U, "metadata width remains correlated with the selected sample");
  expect_eq(sample_caps.height, 480U,
            "metadata height remains correlated with the selected sample");
  expect_eq(sample_caps.fps_numerator, 24U,
            "metadata frame rate remains correlated with the selected sample");

  set_scenario("probe-unknown-pts");
  const auto unknown_pts = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_pts.duration_ns, 1'000'000'000ULL,
            "unknown-PTS short stream uses frame-count duration");
  expect_eq(unknown_pts.total_frames, 30ULL, "unknown-PTS access units remain valid frames");
  expect_true(unknown_pts.duration_is_estimated,
              "unknown-PTS frame-count duration is explicitly estimated");

  set_scenario("probe-one-frame-rounding");
  const auto one_frame = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(one_frame.total_frames, 1ULL, "nanosecond rounding cannot erase a proven frame");

  set_scenario("probe-inexact-caps-fps");
  const auto inferred_fps = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(inferred_fps.fps_numerator, 30U,
            "constant presentation timing corrects inexact container caps");
  expect_eq(inferred_fps.fps_denominator, 1U, "inferred constant frame rate is reduced exactly");
  expect_eq(inferred_fps.total_frames, 60ULL, "inexact container caps do not lose a proven frame");

  set_scenario("probe-seek-unsupported");
  const auto unseekable = probe_gpu_video(container_config(video_path), timeout_ns);
  expect_eq(unseekable.duration_ns, 10'000'000'000ULL,
            "unseekable parser retains bounded container-duration fallback");
  expect_true(unseekable.duration_is_estimated,
              "uncorrelated container duration is explicitly marked estimated");

  set_scenario("probe-seek-preroll");
  expect_eq(probe_gpu_video(container_config(video_path), timeout_ns).total_frames, 299ULL,
            "seek preroll outside the active segment is ignored");

  set_scenario("probe-ok");
  expect_eq(probe_gpu_video(container_config(video_path), 1'000'000'000ULL).width, 3840U,
            "one-second timeout boundary accepted");
  expect_eq(probe_gpu_video(container_config(video_path), 3'600'000'000'000ULL).width, 3840U,
            "one-hour timeout boundary accepted");
  std::filesystem::remove(event_path);
  expect_eq(probe_gpu_video(elementary_config(video_path, GpuDecodeCodec::H264), timeout_ns).width,
            3840U, "H.264 elementary stream uses explicit parser");
  expect_true(has_event(read_events(event_path), "probe-h264-parser"),
              "H.264 elementary probe constructs only the requested parser");
  std::filesystem::remove(event_path);
  expect_eq(probe_gpu_video(elementary_config(video_path, GpuDecodeCodec::Hevc), timeout_ns).width,
            3840U, "HEVC elementary stream uses explicit parser");
  expect_true(has_event(read_events(event_path), "probe-h265-parser"),
              "HEVC elementary probe constructs only the requested parser");
}

void invalid_inputs_fail(const std::filesystem::path& video_path,
                         const std::filesystem::path& event_path) {
  constexpr std::uint64_t timeout_ns = 5'000'000'000ULL;
  expect_invalid_argument([&] { (void)probe_gpu_video({}, timeout_ns); }, "path is required",
                          "empty path rejected");
  expect_invalid_argument([&] { (void)probe_gpu_video(container_config(video_path), 999'999'999); },
                          "one second", "sub-second timeout rejected");
  expect_invalid_argument(
      [&] { (void)probe_gpu_video(container_config(video_path), 3'600'000'000'001ULL); },
      "one hour", "over-one-hour timeout rejected");

  expect_probe_error(
      [&] {
        (void)probe_gpu_video(container_config(video_path.parent_path() / "missing.mp4"),
                              timeout_ns);
      },
      "not a readable regular file", "missing input rejected before probing");

  for (const auto& [scenario_name, fragment] :
       std::array<std::pair<std::string_view, std::string_view>, 18>{
           {{"old-version", "1.10 or newer"},
            {"init-error", "fake initialization failure"},
            {"probe-parse-error", "fake parse failure"},
            {"probe-missing-info", "metadata identity"},
            {"probe-missing-pad", "metadata pad"},
            {"probe-missing-sink", "compressed-stream sink"},
            {"probe-missing-bus", "message bus"},
            {"probe-state-error", "playing state"},
            {"probe-stream-error", "playing state"},
            {"probe-async-error", "fake parser failure"},
            {"probe-no-supported-video", "H.264 or HEVC"},
            {"probe-missing-sample-caps", "H.264 or HEVC"},
            {"probe-missing-caps-structure", "no structure"},
            {"probe-wrong-codec-caps", "decoder-compatible"},
            {"probe-unparsed-caps", "decoder-compatible"},
            {"probe-avc-caps", "decoder-compatible"},
            {"probe-nal-caps", "decoder-compatible"},
            {"probe-bad-dimensions", "invalid visible"}}}) {
    set_scenario(scenario_name);
    expect_probe_error([&] { (void)probe_gpu_video(container_config(video_path), timeout_ns); },
                       fragment, scenario_name);
  }

  for (const auto scenario_name : {"probe-timeout", "probe-pull-timeout"}) {
    set_scenario(scenario_name);
    expect_probe_error(
        [&] { (void)probe_gpu_video(container_config(video_path), 1'000'000'000ULL); }, "timed out",
        scenario_name);
  }

  set_scenario("probe-odd-dimensions");
  expect_probe_error([&] { (void)probe_gpu_video(container_config(video_path), timeout_ns); },
                     "incompatible with NV12", "odd parser-visible dimensions rejected");
  set_scenario("probe-bad-fps");
  expect_probe_error([&] { (void)probe_gpu_video(container_config(video_path), timeout_ns); },
                     "invalid frame rate", "zero-denominator FPS rejected");
  set_scenario("probe-high-fps");
  expect_probe_error([&] { (void)probe_gpu_video(container_config(video_path), timeout_ns); },
                     "implausible frame rate", "time-base artifact FPS rejected");

  set_scenario("probe-parse-partial-error");
  std::filesystem::remove(event_path);
  expect_probe_error([&] { (void)probe_gpu_video(container_config(video_path), timeout_ns); },
                     "fake partial parse failure", "partial parse failure rejected");
  expect_true(has_event(read_events(event_path), "unref-pipeline"),
              "partially constructed pipeline is released before error propagation");
}

} // namespace

int main() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  const auto runtime = find_fake_runtime_runfile();
  set_environment("RECO_GSTREAMER_DYLIB_PATH", runtime.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", runtime.string());
  set_environment("RECO_GLIB_DYLIB_PATH", runtime.string());

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
  invalid_inputs_fail(video_path, event_path);

  std::filesystem::remove(video_path);
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
