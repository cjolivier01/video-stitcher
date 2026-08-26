#include "reco/io/gpu_video_probe.hpp"

#include "gpu_video_probe_process_test.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <csignal>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <windows.h>
#endif
#if defined(__linux__)
#include <fcntl.h>
#endif

using namespace reco::io;

namespace {

int failures = 0;
std::filesystem::path probe_worker_path;
std::filesystem::path fake_probe_worker_path;

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

std::filesystem::path resolve_runfile(std::string path) {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
  const auto logical_path = std::string(workspace) + "/" + path;
  const auto resolved = std::filesystem::path(runfiles->Rlocation(logical_path));
  if (resolved.empty() || !std::filesystem::is_regular_file(resolved)) {
    throw std::runtime_error(path + " runfile not found");
  }
  return resolved;
}

std::filesystem::path executable_runfile(std::string path) {
#if defined(_WIN32)
  path += ".exe";
#endif
  return resolve_runfile(std::move(path));
}

void set_environment(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

std::optional<std::uint64_t>
wait_for_process_marker(const std::filesystem::path& path,
                        std::chrono::steady_clock::time_point deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream input(path);
    std::uint64_t process_id = 0;
    if (input >> process_id && process_id != 0) {
      return process_id;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

#if defined(_WIN32)
class WindowsHandle {
public:
  WindowsHandle() = default;
  explicit WindowsHandle(HANDLE handle) : handle_(handle) {}
  WindowsHandle(const WindowsHandle&) = delete;
  WindowsHandle& operator=(const WindowsHandle&) = delete;
  WindowsHandle(WindowsHandle&&) = delete;
  WindowsHandle& operator=(WindowsHandle&&) = delete;
  ~WindowsHandle() {
    if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
  }

  [[nodiscard]] HANDLE get() const { return handle_; }
  [[nodiscard]] explicit operator bool() const {
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
  }

private:
  HANDLE handle_ = nullptr;
};

std::filesystem::path current_test_executable() {
  std::vector<wchar_t> path(32'768);
  const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (length == 0 || length >= path.size()) {
    throw std::runtime_error("failed to resolve the current test executable");
  }
  return std::filesystem::path(std::wstring(path.data(), length));
}
#endif

void set_scenario(std::string_view value) {
  set_environment("RECO_FAKE_GST_SCENARIO", std::string(value));
}

GpuFileDecodeConfig container_config(const std::filesystem::path& path) {
#if defined(_WIN32)
  const auto encoded = path.u8string();
  const std::string path_string(reinterpret_cast<const char*>(encoded.data()), encoded.size());
#else
  const auto path_string = path.string();
#endif
  return {.path = path_string,
          .codec = GpuDecodeCodec::H264,
          .elementary_stream = false,
          .container = GpuDecodeContainer::QuickTime,
          .max_buffers = 4,
          .drop = false};
}

GpuFileDecodeConfig elementary_config(const std::filesystem::path& path, GpuDecodeCodec codec) {
#if defined(_WIN32)
  const auto encoded = path.u8string();
  const std::string path_string(reinterpret_cast<const char*>(encoded.data()), encoded.size());
#else
  const auto path_string = path.string();
#endif
  return {.path = path_string,
          .codec = codec,
          .elementary_stream = true,
          .container = std::nullopt,
          .max_buffers = 4,
          .drop = false};
}

GpuVideoProbe probe_video(const GpuFileDecodeConfig& config, std::uint64_t timeout_ns) {
  return reco::io::probe_gpu_video(config, probe_worker_path, timeout_ns);
}

#if defined(_WIN32)
int run_windows_parent_death_probe_caller() {
  const char* video_path = std::getenv("RECO_FAKE_PROBE_CALLER_VIDEO_PATH");
  const char* worker_path = std::getenv("RECO_FAKE_PROBE_CALLER_WORKER_PATH");
  if (video_path == nullptr || video_path[0] == '\0' || worker_path == nullptr ||
      worker_path[0] == '\0') {
    return EXIT_FAILURE;
  }
  try {
    (void)reco::io::probe_gpu_video(container_config(std::filesystem::path(video_path)),
                                    std::filesystem::path(worker_path), 30'000'000'000ULL);
  } catch (...) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#endif

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
  constexpr std::uint64_t timeout_ns = 60'000'000'000ULL;
  set_scenario("probe-ok");
  std::filesystem::remove(event_path);
  const auto result = probe_video(container_config(video_path), timeout_ns);
  expect_eq(result.width, 3840U, "parser-visible width");
  expect_eq(result.height, 2160U, "parser-visible height");
  expect_eq(result.fps_numerator, 30'000U, "parser FPS numerator");
  expect_eq(result.fps_denominator, 1'001U, "parser FPS denominator");
  expect_true(std::abs(result.fps - 30'000.0 / 1'001.0) < 1e-12, "rational FPS");
  expect_eq(result.duration_ns, 10'000'000'000ULL, "queried duration");
  expect_eq(result.total_frames, 300ULL, "EOS scan preserves the exact access-unit count");
  expect_true(!result.duration_is_estimated, "known duration is not estimated");
  expect_true(!result.total_frames_is_estimated, "EOS-proven frame count is exact");

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
  const auto unknown_duration = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_duration.duration_ns, 60'000'000'000ULL,
            "unknown duration uses Rust-compatible estimate");
  expect_eq(unknown_duration.total_frames, 1'798ULL,
            "fallback frame count uses exact rational arithmetic");
  expect_true(unknown_duration.duration_is_estimated, "unknown duration is marked estimated");
  expect_true(unknown_duration.total_frames_is_estimated,
              "unknown long-stream frame count is marked estimated");

  set_scenario("probe-duration-zero");
  expect_true(probe_video(container_config(video_path), timeout_ns).duration_is_estimated,
              "zero duration uses explicit estimate");

  set_scenario("probe-exact-frame-count");
  expect_eq(probe_video(container_config(video_path), timeout_ns).total_frames, 3ULL,
            "exact rational arithmetic avoids floating-point off-by-one");

  set_scenario("probe-integral-frame-count");
  expect_eq(probe_video(container_config(video_path), timeout_ns).total_frames, 30'000ULL,
            "integral rational frame count is not rounded down");

  set_scenario("probe-frame-count-overflow");
  expect_eq(probe_video(container_config(video_path), timeout_ns).total_frames, 276'424'736'369ULL,
            "large duration remains exact without intermediate overflow");

  set_scenario("probe-duration-mismatch");
  const auto selected_duration = probe_video(container_config(video_path), timeout_ns);
  expect_true(selected_duration.duration_ns > 900'000'000ULL &&
                  selected_duration.duration_ns < 1'100'000'000ULL,
              "selected stream duration excludes longer unrelated tracks");
  expect_eq(selected_duration.total_frames, 30ULL,
            "selected stream frame count excludes longer unrelated tracks");
  expect_true(!selected_duration.duration_is_estimated,
              "timestamp-correlated selected duration is not estimated");

  set_scenario("probe-long-video-shorter-than-container");
  const auto long_selected_stream = probe_video(container_config(video_path), timeout_ns);
  expect_true(long_selected_stream.duration_ns > 199'000'000'000ULL &&
                  long_selected_stream.duration_ns < 201'000'000'000ULL,
              "timing windows use the selected video duration instead of a longer container track");
  expect_eq(long_selected_stream.total_frames, 5'995ULL,
            "long selected-stream correlation excludes trailing unrelated tracks");

  set_scenario("probe-delayed-stream");
  const auto delayed_stream = probe_video(container_config(video_path), timeout_ns);
  expect_eq(delayed_stream.duration_ns, 1'000'000'000ULL,
            "selected stream duration excludes its delayed container start");
  expect_eq(delayed_stream.total_frames, 30ULL,
            "delayed selected stream reports only its own frame count");
  expect_true(!delayed_stream.duration_is_estimated,
              "delayed selected stream duration remains timestamp-correlated");

  set_scenario("probe-nonzero-origin");
  const auto nonzero_origin = probe_video(container_config(video_path), timeout_ns);
  expect_eq(nonzero_origin.duration_ns, 2'000'000'000ULL,
            "nonzero timeline origin does not shorten a duration span");
  expect_eq(nonzero_origin.total_frames, 60ULL,
            "short demux duration still searches the final access unit");

  set_scenario("probe-decode-order-origin");
  const auto decode_order_origin = probe_video(container_config(video_path), timeout_ns);
  expect_eq(decode_order_origin.duration_ns, 1'000'000'000ULL,
            "first decode-order access unit does not define the timeline origin");
  expect_eq(decode_order_origin.total_frames, 30ULL,
            "decode-ordered first PTS does not drop presentation frames");

  set_scenario("probe-caps-runahead");
  const auto sample_caps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(sample_caps.width, 854U, "metadata width remains correlated with the selected sample");
  expect_eq(sample_caps.height, 480U,
            "metadata height remains correlated with the selected sample");
  expect_eq(sample_caps.fps_numerator, 24U,
            "metadata frame rate remains correlated with the selected sample");

  set_scenario("probe-unknown-pts");
  const auto unknown_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_pts.duration_ns, 1'000'000'000ULL,
            "unknown-PTS short stream uses frame-count duration");
  expect_eq(unknown_pts.total_frames, 30ULL, "unknown-PTS access units remain valid frames");
  expect_true(unknown_pts.duration_is_estimated,
              "unknown-PTS frame-count duration is explicitly estimated");

  set_scenario("probe-one-frame-rounding");
  const auto one_frame = probe_video(container_config(video_path), timeout_ns);
  expect_eq(one_frame.total_frames, 1ULL, "nanosecond rounding cannot erase a proven frame");

  set_scenario("probe-inexact-caps-fps");
  const auto inferred_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(inferred_fps.fps_numerator, 30U,
            "constant presentation timing corrects inexact container caps");
  expect_eq(inferred_fps.fps_denominator, 1U, "inferred constant frame rate is reduced exactly");
  expect_eq(inferred_fps.total_frames, 60ULL, "inexact container caps do not lose a proven frame");

  set_scenario("probe-short-quantized-exact-30");
  const auto short_exact_30 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_exact_30.fps_numerator, 30U,
            "short quantized timing preserves an exact caps numerator");
  expect_eq(short_exact_30.fps_denominator, 1U,
            "short quantized timing preserves an exact caps denominator");
  expect_eq(short_exact_30.total_frames, 3ULL, "short quantized timing retains the exact AU count");

  set_scenario("probe-short-quantized-exact-5997");
  const auto short_exact_5997 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_exact_5997.fps_numerator, 5'997U,
            "short quantized near-canonical timing preserves the exact caps numerator");
  expect_eq(short_exact_5997.fps_denominator, 100U,
            "short quantized near-canonical timing preserves the exact caps denominator");
  expect_eq(short_exact_5997.total_frames, 3ULL,
            "short near-canonical timing retains the exact AU count");

  set_scenario("probe-long-unknown-pts");
  const auto long_unknown_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_unknown_pts.duration_ns, 3'000'000'000ULL,
            "long unknown-PTS stream uses its exact access-unit count");
  expect_eq(long_unknown_pts.total_frames, 90ULL,
            "unknown-PTS stream scanning continues beyond the timing window");
  expect_true(long_unknown_pts.duration_is_estimated,
              "long unknown-PTS duration remains explicitly estimated");
  expect_true(!long_unknown_pts.total_frames_is_estimated,
              "unknown-PTS EOS scan still proves the AU count");

  set_scenario("probe-mixed-prefix-pts");
  const auto mixed_prefix = probe_video(container_config(video_path), timeout_ns);
  expect_eq(mixed_prefix.duration_ns, 4'000'000'000ULL,
            "untimed prefix retains its nominal frame interval");
  expect_eq(mixed_prefix.total_frames, 120ULL, "untimed prefix does not shift the stream origin");
  expect_true(mixed_prefix.duration_is_estimated,
              "duration with an untimed prefix is explicitly estimated");

  set_scenario("probe-long-mixed-prefix-pts");
  const auto long_mixed_prefix = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_mixed_prefix.duration_ns, 20'000'000'000ULL,
            "long untimed prefix is included in correlated duration");
  expect_eq(long_mixed_prefix.total_frames, 600ULL,
            "long untimed prefix is included in correlated frame count");
  expect_true(long_mixed_prefix.duration_is_estimated,
              "long untimed-prefix duration remains explicitly estimated");
  expect_true(long_mixed_prefix.total_frames_is_estimated,
              "bounded long untimed-prefix count remains explicitly estimated");

  set_scenario("probe-reordered-untimed-prefix");
  const auto reordered_untimed_prefix = probe_video(container_config(video_path), timeout_ns);
  expect_eq(reordered_untimed_prefix.duration_ns, 20'000'000'000ULL,
            "reordered untimed prefix is counted once in correlated duration");
  expect_eq(reordered_untimed_prefix.total_frames, 600ULL,
            "reordered untimed prefix is counted once in correlated frame count");

  set_scenario("probe-unset-fps-inferred");
  const auto unset_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unset_fps.fps_numerator, 30U, "unset caps frame rate is inferred from timestamps");
  expect_eq(unset_fps.fps_denominator, 1U, "inferred unset-caps frame rate is exact");
  expect_eq(unset_fps.total_frames, 120ULL, "unset caps frame rate remains seek-countable");

  const auto untimed_elementary =
      probe_video(elementary_config(video_path, GpuDecodeCodec::H264), timeout_ns);
  expect_eq(untimed_elementary.fps_numerator, 30U,
            "timing-less elementary stream uses the explicit frame-rate fallback");
  expect_eq(untimed_elementary.total_frames, 120ULL,
            "timing-less elementary stream is counted through EOS");
  expect_true(untimed_elementary.duration_is_estimated,
              "timing-less elementary duration remains explicitly estimated");

  set_scenario("probe-long-untimed-elementary");
  const auto long_untimed_elementary =
      probe_video(elementary_config(video_path, GpuDecodeCodec::H264), timeout_ns);
  expect_true(long_untimed_elementary.total_frames >= 513ULL,
              "long elementary estimate preserves the observed AU lower bound");
  expect_true(long_untimed_elementary.total_frames_is_estimated,
              "long elementary AU count is explicitly estimated");
  expect_true(long_untimed_elementary.duration_is_estimated,
              "untimed elementary duration query is not treated as authoritative");

  set_scenario("probe-vfr-unset-fps");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "dense nonconstant timing is rejected when caps do not provide a constant rate");

  set_scenario("probe-vfr-late-transition");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "dense nonconstant timing cannot silently retain plausible average caps");

  set_scenario("probe-retimed-constant-pts");
  const auto retimed_constant = probe_video(container_config(video_path), timeout_ns);
  expect_eq(retimed_constant.fps_numerator, 30U,
            "whole-stream constant PTS timing overrides stale plausible caps");
  expect_eq(retimed_constant.total_frames, 150ULL,
            "retimed constant stream retains its EOS-proven AU count");

  set_scenario("probe-eos-vui-duration-mismatch");
  const auto corrected_duration = probe_video(container_config(video_path), timeout_ns);
  expect_eq(corrected_duration.fps_numerator, 30U,
            "EOS timestamp cadence replaces contradictory VUI caps");
  expect_eq(corrected_duration.duration_ns, 5'000'000'000ULL,
            "corrected cadence does not trust contradictory parser buffer durations");
  expect_eq(corrected_duration.total_frames, 150ULL,
            "corrected EOS duration retains the exact AU count");
  expect_true(corrected_duration.duration_is_estimated,
              "timestamp-derived corrected duration is explicitly estimated");

  set_scenario("probe-duplicate-pts-pairs");
  const auto duplicate_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_pts.fps_numerator, 50U,
            "duplicate PTS pairs cannot halve a plausible caps frame rate");
  expect_eq(duplicate_pts.total_frames, 200ULL,
            "duplicate PTS pairs retain their EOS-proven AU count");
  expect_eq(duplicate_pts.duration_ns, 4'000'000'000ULL,
            "duplicate PTS pairs retain their observed terminal span");

  set_scenario("probe-long-duplicate-pts-pairs");
  const auto long_duplicate_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_duplicate_pts.fps_numerator, 50U,
            "long duplicate PTS pairs retain their AU frame rate");
  expect_eq(long_duplicate_pts.duration_ns, 12'000'000'000ULL,
            "terminal duplicate PTS group retains its complete presentation span");
  expect_eq(long_duplicate_pts.total_frames, 600ULL,
            "bounded terminal correlation counts every duplicate-PTS AU");

  set_scenario("probe-duplicate-clustered-missing-groups");
  const auto duplicate_clustered_missing = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_clustered_missing.fps_numerator, 50U,
            "periodically untimed duplicate groups preserve the AU rate");
  expect_eq(duplicate_clustered_missing.fps_denominator, 1U,
            "periodically untimed duplicate groups preserve the rate denominator");
  expect_eq(duplicate_clustered_missing.total_frames, 200ULL,
            "periodically untimed duplicate groups retain their EOS-proven AU count");
  expect_eq(duplicate_clustered_missing.duration_ns, 4'000'000'000ULL,
            "periodically untimed duplicate groups retain caps-rate duration");

  set_scenario("probe-duplicate-pts-transition");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "dense duplicate PTS multiplicity transitions are rejected unconditionally");

  set_scenario("probe-duplicate-pts-reorder-cutoff");
  const auto duplicate_reorder_cutoff = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_reorder_cutoff.fps_numerator, 50U,
            "reordered cutoff group does not invalidate uniform duplicate PTS timing");
  expect_eq(duplicate_reorder_cutoff.fps_denominator, 1U,
            "reordered duplicate PTS timing retains its inferred denominator");
  expect_eq(duplicate_reorder_cutoff.total_frames, 1'000ULL,
            "reordered duplicate PTS timing extends to an EOS-proven AU count");
  expect_true(!duplicate_reorder_cutoff.total_frames_is_estimated,
              "reordered duplicate PTS count is exact after reaching EOS");

  set_scenario("probe-duplicate-pts-transition-untimed-tail");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "an untimed tail cannot excuse a dense duplicate PTS cadence transition");

  set_scenario("probe-duplicate-pts-larger-reorder-suffix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "larger duplicate groups in the bounded suffix are rejected as a cadence transition");

  set_scenario("probe-paired-au-missing-pts");
  const auto paired_missing_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(paired_missing_pts.fps_numerator, 50U,
            "uniformly missing paired-AU PTS values preserve the AU rate");
  expect_eq(paired_missing_pts.total_frames, 200ULL,
            "uniformly missing paired-AU PTS values retain the exact AU count");

  set_scenario("probe-reordered-periodic-missing-pts");
  const auto reordered_periodic_missing = probe_video(container_config(video_path), timeout_ns);
  expect_eq(reordered_periodic_missing.fps_numerator, 30U,
            "periodic missing PTS survives B-frame presentation reordering");
  expect_eq(reordered_periodic_missing.fps_denominator, 1U,
            "reordered sparse timing retains the inferred denominator");
  expect_eq(reordered_periodic_missing.total_frames, 200ULL,
            "reordered sparse timing retains the EOS-proven AU count");
  expect_true(reordered_periodic_missing.duration_ns >= 6'666'666'666ULL &&
                  reordered_periodic_missing.duration_ns <= 6'666'666'667ULL,
              "reordered sparse timing derives duration from the corrected frame rate");

  set_scenario("probe-long-reordered-periodic-missing-pts");
  const auto long_reordered_periodic_missing =
      probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_reordered_periodic_missing.fps_numerator, 30U,
            "bounded reordered sparse timing corrects misleading caps");
  expect_eq(long_reordered_periodic_missing.total_frames, 600ULL,
            "conflicting reordered sparse timing extends to an exact AU count");
  expect_true(!long_reordered_periodic_missing.total_frames_is_estimated,
              "adaptive conflicting-metadata scan reaches EOS when bounded");

  set_scenario("probe-bounded-caps-underestimate");
  const auto bounded_caps_underestimate = probe_video(container_config(video_path), timeout_ns);
  expect_eq(bounded_caps_underestimate.fps_numerator, 30U,
            "bounded timing overrides caps that cannot cover observed access units");
  expect_eq(bounded_caps_underestimate.total_frames, 600ULL,
            "bounded timing and container duration recover the complete frame estimate");

  set_scenario("probe-container-rate-over-vui");
  const auto container_rate = probe_video(container_config(video_path), timeout_ns);
  expect_eq(container_rate.fps_numerator, 15U,
            "constant container timestamps override nearby bitstream VUI caps");
  expect_eq(container_rate.fps_denominator, 1U,
            "container timing correction remains an exact rational");
  expect_eq(container_rate.total_frames, 600ULL,
            "container frame rate preserves the 600-frame duration count");

  set_scenario("probe-bounded-stale-caps");
  const auto stale_caps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(stale_caps.fps_numerator, 15U,
            "bounded constant timing overrides substantially stale caps");
  expect_eq(stale_caps.fps_denominator, 1U, "substantial stale-caps correction remains exact");
  expect_eq(stale_caps.total_frames, 600ULL,
            "stale high-rate caps cannot double the calibration frame count");

  set_scenario("probe-sparse-exact-30-gaps");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "fully timestamped doubled intervals are not reinterpreted as sparse timing");

  set_scenario("probe-clustered-missing-pts");
  const auto clustered_missing_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(clustered_missing_pts.fps_numerator, 30U,
            "clustered missing PTS values do not imply uniform AU multiplicity");
  expect_eq(clustered_missing_pts.fps_denominator, 1U,
            "clustered missing PTS values preserve the caps frame-rate denominator");
  expect_eq(clustered_missing_pts.total_frames, 120ULL,
            "clustered missing PTS values retain the exact AU count");
  expect_eq(clustered_missing_pts.duration_ns, 4'000'000'000ULL,
            "clustered missing PTS values retain the caps-rate duration");

  set_scenario("probe-exact-5997-fps");
  const auto exact_5997_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(exact_5997_fps.fps_numerator, 5'997U,
            "exact near-canonical caps are not snapped to an NTSC numerator");
  expect_eq(exact_5997_fps.fps_denominator, 100U,
            "exact near-canonical caps preserve their denominator");

  set_scenario("probe-container-exact-5997-parser-ntsc");
  const auto container_exact_5997 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(container_exact_5997.fps_numerator, 5'997U,
            "exact container cadence overrides normalized parser caps");
  expect_eq(container_exact_5997.fps_denominator, 100U,
            "exact container cadence preserves its noncanonical denominator");
  expect_eq(container_exact_5997.total_frames, 600ULL,
            "near-NTSC conflict extension retains the exact AU count");

  set_scenario("probe-quantized-no-vui-5994");
  const auto quantized_no_vui = probe_video(container_config(video_path), timeout_ns);
  expect_eq(quantized_no_vui.fps_numerator, 60'000U,
            "90 kHz quantized timing snaps to the standard NTSC numerator");
  expect_eq(quantized_no_vui.fps_denominator, 1'001U,
            "90 kHz quantized timing snaps to the standard NTSC denominator");
  expect_eq(quantized_no_vui.total_frames, 240ULL,
            "quantized no-VUI stream retains its EOS-proven AU count");

  set_scenario("probe-short-unset-fps-15");
  const auto short_unset_15 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_unset_15.fps_numerator, 15U,
            "short EOS-proven timing infers an exact noncanonical frame rate");
  expect_eq(short_unset_15.fps_denominator, 1U,
            "short EOS-proven noncanonical timing retains its denominator");
  expect_eq(short_unset_15.total_frames, 15ULL,
            "short noncanonical inference retains its EOS-proven AU count");

  set_scenario("probe-long-unset-fps-15");
  const auto long_unset_15 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_unset_15.fps_numerator, 15U,
            "long unset-caps stream extends to EOS before selecting a noncanonical rate");
  expect_eq(long_unset_15.fps_denominator, 1U,
            "long unset-caps stream preserves its exact denominator");
  expect_eq(long_unset_15.total_frames, 600ULL,
            "long unset-caps stream retains its EOS-proven AU count");

  set_scenario("probe-durationless-unseekable-15");
  const auto bounded_unset_15 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(bounded_unset_15.fps_numerator, 15U,
            "bounded constant timing selects a noncanonical rate without duration or EOS");
  expect_eq(bounded_unset_15.fps_denominator, 1U,
            "bounded durationless inference preserves its exact denominator");
  expect_eq(bounded_unset_15.total_frames, 4'097ULL,
            "durationless inference preserves the observed compressed-AU lower bound");
  expect_eq(bounded_unset_15.duration_ns, 273'133'333'334ULL,
            "durationless inference covers its observed compressed-AU lower bound");
  expect_true(bounded_unset_15.total_frames_is_estimated,
              "durationless bounded frame count remains explicitly estimated");

  set_scenario("probe-late-vfr-after-bounded-prefix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "terminal timing rejects a VFR transition after the bounded sequential prefix");

  set_scenario("probe-vfr-in-final-window");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "mixed cadence inside the terminal timing window is rejected");

  set_scenario("probe-low-amplitude-vfr");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "a 30-to-31 fps EOS transition fails cadence phase validation");

  set_scenario("probe-interior-vfr-recovery");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "a middle 30-to-15-to-30 cadence change is rejected by interior sampling");

  set_scenario("probe-interior-seek-gap");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "samples beyond an interior window cannot stand in for its missing timing evidence");

  set_scenario("probe-prefix-vfr-tail-cfr");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "a constant terminal cadence cannot excuse a dense nonconstant prefix");

  set_scenario("probe-gap-before-reorder-suffix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "a dense gap immediately before the reorder suffix remains part of prefix analysis");

  std::filesystem::remove(event_path);
  set_scenario("probe-terminal-60-eos");
  const auto terminal_60 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(terminal_60.fps_numerator, 60U,
            "60 fps terminal timing retains the exact caps numerator");
  expect_eq(terminal_60.total_frames, 1'200ULL,
            "60 fps terminal timing retains the correlated frame count");
  expect_true(has_event(read_events(event_path), "terminal-window-eos"),
              "the five-second 60 fps terminal window drains through EOS");

  set_scenario("probe-terminal-transition-after-256");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "a terminal cadence transition after 256 analyzed samples is detected before EOS");

  set_scenario("probe-terminal-duplicate-pts-transition");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "EOS-complete terminal timing rejects smaller duplicate-PTS groups in the final 32 AUs");

  set_scenario("probe-vfr-missing-durations");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "missing buffer durations do not hide dense nonconstant PTS timing");

  set_scenario("probe-dropped-frame-after-prefix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "a fully timestamped missing access-unit interval is rejected as nonconstant cadence");

  set_scenario("probe-reduced-cadence-after-prefix");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "reduced cadence after the prefix is rejected as variable frame rate");

  set_scenario("probe-bframe-cutoff");
  const auto reordered_cutoff = probe_video(container_config(video_path), timeout_ns);
  expect_eq(reordered_cutoff.fps_numerator, 30U,
            "timing lookahead closes a B-frame group at the analysis boundary");
  expect_eq(reordered_cutoff.fps_denominator, 1U,
            "B-frame timing inference keeps the reduced rational rate");
  expect_eq(reordered_cutoff.total_frames, 120ULL,
            "B-frame cutoff does not lose the delayed presentation frame");

  set_scenario("probe-quantized-timestamps");
  const auto quantized = probe_video(container_config(video_path), timeout_ns);
  expect_eq(quantized.total_frames, 100ULL,
            "seek-proven count survives coarse timestamp quantization");
  expect_eq(quantized.duration_ns, 4'170'833'334ULL,
            "quantized duration is clamped to the proven frame-count boundary");
  expect_true(!quantized.duration_is_estimated,
              "EOS-correlated quantized duration remains authoritative");
  expect_true(!quantized.total_frames_is_estimated,
              "quantized EOS scan proves its compressed AU count");

  set_scenario("probe-bad-fps");
  const auto invalid_caps_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(invalid_caps_fps.fps_numerator, 30U,
            "invalid caps rate is replaced by constant parser timing");
  expect_eq(invalid_caps_fps.fps_denominator, 1U,
            "invalid caps timing inference is reduced exactly");

  set_scenario("probe-estimated-count-lower-bound");
  const auto lower_bound = probe_video(container_config(video_path), timeout_ns);
  expect_eq(lower_bound.total_frames, 4'097ULL,
            "adaptive estimated count cannot fall below observed compressed AUs");
  expect_true(lower_bound.total_frames_is_estimated,
              "bounded lower-bound count remains explicitly estimated");

  set_scenario("probe-seek-unsupported");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); },
      "could not verify constant frame timing",
      "an unseekable long container fails closed when terminal timing cannot be verified");

  set_scenario("probe-seek-preroll");
  const auto seek_preroll = probe_video(container_config(video_path), timeout_ns);
  expect_eq(seek_preroll.total_frames, 5'995ULL,
            "seek preroll outside the active segment is ignored");
  expect_true(seek_preroll.total_frames_is_estimated,
              "bounded-seek count remains explicitly estimated");
  expect_true(seek_preroll.duration_is_estimated,
              "bounded terminal seek duration remains explicitly estimated");

  set_scenario("probe-seek-unknown-pts-preroll");
  const auto unknown_pts_preroll = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_pts_preroll.total_frames, 5'995ULL,
            "PTS-less seek preroll is skipped before duration correlation");
  expect_true(unknown_pts_preroll.total_frames_is_estimated,
              "PTS-less seek-preroll count remains explicitly estimated");
  expect_true(unknown_pts_preroll.duration_is_estimated,
              "PTS-less seek-preroll duration remains explicitly estimated");

  set_scenario("probe-seek-untimestamped-tail");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); },
      "could not verify constant frame timing",
      "an untimestamped terminal window cannot establish constant presentation timing");

  set_scenario("probe-seek-dts-reorder-tail");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "could not verify constant frame timing",
                     "DTS-only terminal evidence cannot prove constant presentation timing");

  set_scenario("probe-blocking-seek");
  const auto blocking_seek_started = std::chrono::steady_clock::now();
  expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                     "worker timed out after exceeding the configured timeout",
                     "blocking seek respects the probe deadline");
  const auto blocking_seek_elapsed = std::chrono::steady_clock::now() - blocking_seek_started;
  expect_true(blocking_seek_elapsed < std::chrono::milliseconds(1'800),
              "blocking seek returns before the synchronous runtime call completes");

  set_scenario("probe-blocking-duration-query");
  const auto blocking_query_started = std::chrono::steady_clock::now();
  expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                     "worker timed out after exceeding the configured timeout",
                     "blocking duration query respects the probe deadline");
  const auto blocking_query_elapsed = std::chrono::steady_clock::now() - blocking_query_started;
  expect_true(blocking_query_elapsed < std::chrono::milliseconds(1'800),
              "blocking duration query returns before the runtime call completes");

  set_scenario("probe-blocking-null-state");
  const auto blocking_teardown_started = std::chrono::steady_clock::now();
  expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                     "worker timed out after exceeding the configured timeout",
                     "blocking pipeline teardown respects the probe deadline");
  const auto blocking_teardown_elapsed =
      std::chrono::steady_clock::now() - blocking_teardown_started;
  expect_true(blocking_teardown_elapsed < std::chrono::milliseconds(1'800),
              "blocking pipeline teardown is reclaimed with the worker process");

  for (const auto scenario_name : {"probe-blocking-playing", "probe-blocking-pull"}) {
    set_scenario(scenario_name);
    const auto blocking_call_started = std::chrono::steady_clock::now();
    expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                       "worker timed out after exceeding the configured timeout", scenario_name);
    expect_true(std::chrono::steady_clock::now() - blocking_call_started <
                    std::chrono::milliseconds(1'800),
                std::string(scenario_name) + " is reclaimed with the worker process");
  }

  set_scenario("probe-ok");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      1'000'000'000ULL)
                .width,
            854U, "one-second timeout boundary accepted");
  expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      3'600'000'000'000ULL)
                .width,
            854U, "one-hour timeout boundary accepted");
  std::filesystem::remove(event_path);
  expect_eq(probe_video(elementary_config(video_path, GpuDecodeCodec::H264), timeout_ns).width,
            3840U, "H.264 elementary stream uses explicit parser");
  expect_true(has_event(read_events(event_path), "probe-h264-parser"),
              "H.264 elementary probe constructs only the requested parser");
  std::filesystem::remove(event_path);
  expect_eq(probe_video(elementary_config(video_path, GpuDecodeCodec::Hevc), timeout_ns).width,
            3840U, "HEVC elementary stream uses explicit parser");
  expect_true(has_event(read_events(event_path), "probe-h265-parser"),
              "HEVC elementary probe constructs only the requested parser");
}

void invalid_inputs_fail(const std::filesystem::path& video_path,
                         const std::filesystem::path& event_path) {
  constexpr std::uint64_t timeout_ns = 5'000'000'000ULL;
  expect_invalid_argument([&] { (void)probe_video({}, timeout_ns); }, "path is required",
                          "empty path rejected");
  expect_invalid_argument([&] { (void)probe_video(container_config(video_path), 999'999'999); },
                          "one second", "sub-second timeout rejected");
  expect_invalid_argument(
      [&] { (void)probe_video(container_config(video_path), 3'600'000'000'001ULL); }, "one hour",
      "over-one-hour timeout rejected");
  expect_invalid_argument(
      [&] { (void)reco::io::probe_gpu_video(container_config(video_path), {}, timeout_ns); },
      "worker path is required", "empty worker path rejected");
  expect_invalid_argument(
      [&] {
        (void)reco::io::probe_gpu_video(container_config(video_path), "relative-probe-worker",
                                        timeout_ns);
      },
      "must be absolute", "relative worker path rejected");
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(container_config(video_path),
                                        video_path.parent_path() / "missing-probe-worker",
                                        timeout_ns);
      },
      "failed to start video probe worker", "missing worker executable rejected");

  expect_probe_error(
      [&] {
        (void)probe_video(container_config(video_path.parent_path() / "missing.mp4"), timeout_ns);
      },
      "not a readable regular file", "missing input rejected before probing");

  for (const auto& [scenario_name, fragment] :
       std::array<std::pair<std::string_view, std::string_view>, 19>{
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
            {"probe-buffered-async-error", "fake parser failure"},
            {"probe-no-supported-video", "H.264 or HEVC"},
            {"probe-missing-sample-caps", "H.264 or HEVC"},
            {"probe-missing-caps-structure", "no structure"},
            {"probe-wrong-codec-caps", "decoder-compatible"},
            {"probe-unparsed-caps", "decoder-compatible"},
            {"probe-avc-caps", "decoder-compatible"},
            {"probe-nal-caps", "decoder-compatible"},
            {"probe-bad-dimensions", "invalid visible"}}}) {
    set_scenario(scenario_name);
    expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                       fragment, scenario_name);
  }

  for (const auto scenario_name : {"probe-timeout", "probe-pull-timeout"}) {
    set_scenario(scenario_name);
    expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                       "timed out", scenario_name);
  }

  set_scenario("probe-odd-dimensions");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "incompatible with NV12", "odd parser-visible dimensions rejected");
  set_scenario("probe-high-fps");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "implausible frame rate", "time-base artifact FPS rejected");

  set_scenario("probe-parse-partial-error");
  std::filesystem::remove(event_path);
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "fake partial parse failure", "partial parse failure rejected");
  expect_true(has_event(read_events(event_path), "unref-pipeline"),
              "partially constructed pipeline is released before error propagation");
}

void worker_ipc_failures_are_bounded(const std::filesystem::path& video_path) {
  auto large_config = container_config(video_path);
  large_config.path.assign(250'000, 'a');
  large_config.path += ".mp4";

  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");
  const auto blocked_input_started = std::chrono::steady_clock::now();
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(large_config, fake_probe_worker_path, 1'000'000'000ULL);
      },
      "timed out", "worker that never reads its request respects the probe deadline");
  expect_true(std::chrono::steady_clock::now() - blocked_input_started <
                  std::chrono::milliseconds(1'800),
              "blocked worker request IPC is reclaimed before its native sleep completes");

  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "close-input");
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(large_config, fake_probe_worker_path, 5'000'000'000ULL);
      },
      "video probe worker", "closed worker input is an exception rather than process SIGPIPE");

  const std::array<std::pair<std::string_view, std::string_view>, 11> invalid_workers{{
      {"crash", "exited abnormally"},
      {"malformed-response", "not valid CBOR"},
      {"deep-response", "nesting"},
      {"truncated-frame", "truncated IPC frame"},
      {"oversized-frame", "IPC frame length"},
      {"trailing-response", "trailing IPC bytes"},
      {"wrong-version", "unsupported protocol version"},
      {"wrapped-version", "out-of-range protocol_version"},
      {"invalid-metadata", "invalid metadata"},
      {"negative-metadata", "out-of-range width"},
      {"oversized-metadata", "out-of-range width"},
  }};
  for (const auto& [scenario, fragment] : invalid_workers) {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", std::string(scenario));
    expect_probe_error(
        [&] {
          (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                          5'000'000'000ULL);
        },
        fragment, std::string("fake worker response: ") + std::string(scenario));
  }
}

void delayed_supervision_cannot_launch_worker(const std::filesystem::path& video_path) {
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".delayed-worker-startup");
  std::filesystem::remove(marker_path);
  set_environment("RECO_FAKE_PROBE_STARTUP_MARKER_PATH", marker_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "startup-marker");
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_with_supervisor_start_delay_for_test(
            container_config(video_path), fake_probe_worker_path, 1'000'000'000ULL,
            1'250'000'000ULL);
      },
      "timed out", "delayed supervision returns at the public deadline");
  expect_true(!std::filesystem::exists(marker_path),
              "expired supervision delay never launches a worker");
  std::filesystem::remove(marker_path);
}

void launch_gate_prevents_post_timeout_process_start(const std::filesystem::path& video_path) {
  const auto lifecycle_path =
      video_path.parent_path() / (video_path.filename().string() + ".launch-lifecycle");
  std::filesystem::remove(lifecycle_path);
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", lifecycle_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  const auto started = std::chrono::steady_clock::now();
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_with_pre_worker_spawn_delay_for_test(
            container_config(video_path), fake_probe_worker_path, 1'000'000'000ULL,
            1'250'000'000ULL);
      },
      "timed out", "pre-worker-spawn delay reaches a bounded timeout");
  const auto elapsed = std::chrono::steady_clock::now() - started;
  expect_true(elapsed >= std::chrono::milliseconds(800) &&
                  elapsed < std::chrono::milliseconds(1'200),
              "pre-worker launch delay returns within the public timeout bound");
  const auto events = read_events(lifecycle_path);
  expect_true(!has_event(events, "worker"), "expired launch sequence does not spawn a worker");
  expect_true(!has_event(events, "guard"), "expired launch sequence does not spawn a guard");
  expect_true(!has_event(events, "request"), "expired launch sequence does not write a request");
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", "");
  std::filesystem::remove(lifecycle_path);
}

void auto_reaped_workers_are_supported(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  struct sigaction ignore_action{};
  struct sigaction previous_action{};
  ignore_action.sa_handler = SIG_IGN;
  sigemptyset(&ignore_action.sa_mask);
  if (sigaction(SIGCHLD, &ignore_action, &previous_action) != 0) {
    expect_true(false, "SIGCHLD auto-reap policy installs");
    return;
  }

  bool probe_succeeded = false;
  const auto descendant_path =
      video_path.parent_path() / (video_path.filename().string() + ".probe-descendant");
  std::filesystem::remove(descendant_path);
  bool descendant_probe_succeeded = false;
  try {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    probe_succeeded = reco::io::probe_gpu_video(container_config(video_path),
                                                fake_probe_worker_path, 5'000'000'000ULL)
                          .width == 854U;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: auto-reaped worker probe threw: " << error.what() << '\n';
    ++failures;
  }
  try {
    set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", descendant_path.string());
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata-with-descendant");
    descendant_probe_succeeded = reco::io::probe_gpu_video(container_config(video_path),
                                                           fake_probe_worker_path, 5'000'000'000ULL)
                                     .width == 854U;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: auto-reaped descendant probe threw: " << error.what() << '\n';
    ++failures;
  }
  if (sigaction(SIGCHLD, &previous_action, nullptr) != 0) {
    expect_true(false, "SIGCHLD policy restores");
  }
  expect_true(probe_succeeded, "auto-reaped worker returns its framed response");
  expect_true(descendant_probe_succeeded,
              "auto-reaped worker with a descendant returns its framed response");
  pid_t descendant = -1;
  {
    std::ifstream input(descendant_path);
    input >> descendant;
  }
  expect_true(descendant > 0, "auto-reaped worker records its descendant");
  bool descendant_exited = false;
  const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (descendant > 0 && std::chrono::steady_clock::now() < exit_deadline) {
    errno = 0;
    if (kill(descendant, 0) != 0 && errno == ESRCH) {
      descendant_exited = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  expect_true(descendant_exited, "ECHILD supervision kills the worker process group");
  std::filesystem::remove(descendant_path);
#else
  (void)video_path;
#endif
}

void windows_job_reclaims_worker_descendants(const std::filesystem::path& video_path) {
#if defined(_WIN32)
  const auto descendant_path =
      video_path.parent_path() / (video_path.filename().string() + ".probe-descendant");
  std::filesystem::remove(descendant_path);
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", descendant_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata-with-descendant");
  try {
    expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        5'000'000'000ULL)
                  .width,
              854U, "Windows worker with a descendant returns its framed response");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: Windows descendant probe threw: " << error.what() << '\n';
    ++failures;
  }

  const auto descendant = wait_for_process_marker(
      descendant_path, std::chrono::steady_clock::now() + std::chrono::seconds(2));
  expect_true(descendant.has_value(), "Windows worker records its descendant");
  if (descendant.has_value() && *descendant <= std::numeric_limits<DWORD>::max()) {
    SetLastError(ERROR_SUCCESS);
    WindowsHandle process(
        OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(*descendant)));
    const auto open_error = GetLastError();
    bool exited = !process && open_error == ERROR_INVALID_PARAMETER;
    if (process) {
      exited = WaitForSingleObject(process.get(), 2'000) == WAIT_OBJECT_0;
      if (!exited) {
        (void)TerminateProcess(process.get(), 1);
      }
    }
    expect_true(exited, "closing the Windows worker Job kills its descendant");
  }
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", "");
  std::filesystem::remove(descendant_path);
#else
  (void)video_path;
#endif
}

void caller_death_reclaims_worker_and_descendant(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".caller-worker");
  const auto descendant_marker =
      video_path.parent_path() / (video_path.filename().string() + ".caller-descendant");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(descendant_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", descendant_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input-with-descendant");
  const auto caller = ::fork();
  if (caller == 0) {
    try {
      (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      30'000'000'000ULL);
    } catch (...) {
    }
    std::_Exit(EXIT_SUCCESS);
  }
  expect_true(caller > 0, "POSIX parent-death probe caller starts");
  if (caller > 0) {
    const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    const auto worker = wait_for_process_marker(worker_marker, marker_deadline);
    const auto descendant = wait_for_process_marker(descendant_marker, marker_deadline);
    expect_true(worker.has_value(), "POSIX isolated worker starts before caller death");
    expect_true(descendant.has_value(), "POSIX worker descendant starts before caller death");
    (void)::kill(caller, SIGKILL);
    int caller_status = 0;
    while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
    }
    const auto expect_process_exit = [](const std::optional<std::uint64_t>& process_id,
                                        std::string_view message) {
      if (!process_id.has_value() ||
          *process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
        return;
      }
      const auto pid = static_cast<pid_t>(*process_id);
      bool exited = false;
      const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < exit_deadline) {
        errno = 0;
        if (::kill(pid, 0) != 0 && errno == ESRCH) {
          exited = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      expect_true(exited, message);
      if (!exited) {
        (void)::kill(pid, SIGKILL);
      }
    };
    expect_process_exit(worker, "POSIX worker exits when its caller dies");
    expect_process_exit(descendant, "POSIX worker descendant exits when its caller dies");
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", "");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(descendant_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#elif defined(_WIN32)
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".caller-worker");
  const auto descendant_marker =
      video_path.parent_path() / (video_path.filename().string() + ".caller-descendant");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(descendant_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", descendant_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input-with-descendant");
  set_environment("RECO_FAKE_PROBE_CALLER_VIDEO_PATH", video_path.string());
  set_environment("RECO_FAKE_PROBE_CALLER_WORKER_PATH", fake_probe_worker_path.string());

  const auto application = current_test_executable().native();
  auto command_line = L"\"" + application + L"\" --reco-parent-death-probe-caller";
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION caller_info{};
  const bool caller_started =
      CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                     CREATE_NO_WINDOW, nullptr, nullptr, &startup, &caller_info) != 0;
  expect_true(caller_started, "Windows parent-death probe caller starts with CreateProcessW");
  if (caller_started) {
    WindowsHandle caller_process(caller_info.hProcess);
    WindowsHandle caller_thread(caller_info.hThread);
    const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    const auto worker = wait_for_process_marker(worker_marker, marker_deadline);
    const auto descendant = wait_for_process_marker(descendant_marker, marker_deadline);
    expect_true(worker.has_value(), "Windows isolated worker starts before caller death");
    expect_true(descendant.has_value(), "Windows worker descendant starts before caller death");

    const auto open_process = [](const std::optional<std::uint64_t>& process_id) {
      if (!process_id.has_value() || *process_id > std::numeric_limits<DWORD>::max()) {
        return static_cast<HANDLE>(nullptr);
      }
      return OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(*process_id));
    };
    WindowsHandle worker_process(open_process(worker));
    WindowsHandle descendant_process(open_process(descendant));
    expect_true(worker_process && WaitForSingleObject(worker_process.get(), 0) == WAIT_TIMEOUT,
                "Windows isolated worker is live before caller death");
    expect_true(descendant_process &&
                    WaitForSingleObject(descendant_process.get(), 0) == WAIT_TIMEOUT,
                "Windows worker descendant is live before caller death");

    (void)TerminateProcess(caller_process.get(), 1);
    (void)WaitForSingleObject(caller_process.get(), 2'000);
    const bool worker_exited =
        worker_process && WaitForSingleObject(worker_process.get(), 2'000) == WAIT_OBJECT_0;
    expect_true(worker_exited, "Windows Job kill-on-close reclaims the worker after caller death");
    const bool descendant_exited =
        descendant_process && WaitForSingleObject(descendant_process.get(), 2'000) == WAIT_OBJECT_0;
    expect_true(descendant_exited,
                "Windows Job kill-on-close reclaims the descendant after caller death");
    if (worker_process && !worker_exited) {
      (void)TerminateProcess(worker_process.get(), 1);
    }
    if (descendant_process && !descendant_exited) {
      (void)TerminateProcess(descendant_process.get(), 1);
    }
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", "");
  set_environment("RECO_FAKE_PROBE_CALLER_VIDEO_PATH", "");
  set_environment("RECO_FAKE_PROBE_CALLER_WORKER_PATH", "");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(descendant_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#else
  (void)video_path;
#endif
}

void unrelated_descriptor_writer_does_not_delay_pipe_eof(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  int pipe_descriptors[2] = {-1, -1};
  if (::pipe(pipe_descriptors) != 0) {
    expect_true(false, "unrelated descriptor EOF pipe opens");
    return;
  }

  const auto lifecycle_path =
      video_path.parent_path() / (video_path.filename().string() + ".pipe-lifecycle");
  std::filesystem::remove(lifecycle_path);
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", lifecycle_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");
  std::exception_ptr probe_failure;
  std::thread probe([&] {
    try {
      (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      2'000'000'000ULL);
    } catch (...) {
      probe_failure = std::current_exception();
    }
  });

  bool worker_started = false;
  bool guard_started = false;
  const auto launch_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (std::chrono::steady_clock::now() < launch_deadline &&
         (!worker_started || !guard_started)) {
    const auto events = read_events(lifecycle_path);
    worker_started = has_event(events, "worker");
    guard_started = has_event(events, "guard");
    if (!worker_started || !guard_started) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  expect_true(worker_started, "pipe EOF regression observes the blocked worker");
  expect_true(guard_started, "pipe EOF regression observes the process-group guard");

  (void)::close(pipe_descriptors[1]);
  pipe_descriptors[1] = -1;
  pollfd read_end{.fd = pipe_descriptors[0], .events = POLLIN, .revents = 0};
  int poll_result = -1;
  do {
    poll_result = ::poll(&read_end, 1, 500);
  } while (poll_result < 0 && errno == EINTR);
  char byte = '\0';
  const auto read_result = poll_result > 0 ? ::read(pipe_descriptors[0], &byte, 1) : -1;
  expect_true(read_result == 0,
              "unrelated worker and guard descriptors do not keep a caller pipe open");
  (void)::close(pipe_descriptors[0]);
  probe.join();

  bool timed_out = false;
  try {
    if (probe_failure != nullptr) {
      std::rethrow_exception(probe_failure);
    }
  } catch (const GpuVideoProbeError& error) {
    timed_out = std::string_view(error.what()).find("timed out") != std::string_view::npos;
  } catch (...) {
  }
  expect_true(timed_out, "pipe EOF regression's blocked worker reaches its bounded timeout");
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(lifecycle_path);
#else
  (void)video_path;
#endif
}

void unrelated_descriptors_are_not_inherited(const std::filesystem::path& video_path) {
#if defined(__linux__)
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".forbidden-descriptor");
  {
    std::ofstream marker(marker_path);
    marker << "descriptor marker";
  }
  const auto marker_descriptor = ::open(marker_path.c_str(), O_RDONLY);
  expect_true(marker_descriptor >= 0, "descriptor-isolation marker opens");
  if (marker_descriptor >= 0) {
    set_environment("RECO_FAKE_PROBE_FORBIDDEN_PATH", marker_path.string());
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "descriptor-isolation");
    expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        5'000'000'000ULL)
                  .width,
              854U, "probe worker closes unrelated inherited descriptors");
    (void)::close(marker_descriptor);
  }
  std::filesystem::remove(marker_path);
#else
  (void)video_path;
#endif
}

void non_utf8_path_round_trips() {
#if defined(__linux__)
  auto filename = std::string("reco_gpu_probe_non_utf8_");
  filename.push_back(static_cast<char>(0xFF));
  filename += ".mp4";
  const auto path = std::filesystem::temp_directory_path() / filename;
  {
    std::ofstream video(path, std::ios::binary);
    video << "fake video container";
  }
  set_scenario("probe-ok");
  expect_eq(probe_video(container_config(path), 5'000'000'000ULL).width, 3840U,
            "non-UTF-8 POSIX path survives probe worker IPC");
  std::filesystem::remove(path);
#endif
}

void windows_unicode_path_round_trips() {
#if defined(_WIN32)
  const auto filename = std::u8string(u8"reco_gpu_probe_\u5f55\u50cf.mp4");
  const auto path = std::filesystem::temp_directory_path() / std::filesystem::path(filename);
  {
    std::ofstream video(path, std::ios::binary);
    video << "fake video container";
  }
  set_scenario("probe-ok");
  expect_eq(probe_video(container_config(path), 5'000'000'000ULL).width, 3840U,
            "UTF-8 Windows path survives probe worker validation and pipeline construction");
  std::filesystem::remove(path);
#endif
}

void windows_path_runtime_discovery(const std::filesystem::path& video_path,
                                    const std::filesystem::path& runtime) {
#if defined(_WIN32)
  const auto directory = std::filesystem::temp_directory_path() /
                         ("reco_gstreamer_path_" + std::to_string(GetCurrentProcessId()));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  for (const auto* name : {"gstreamer-1.0-0.dll", "gstapp-1.0-0.dll", "libglib-2.0-0.dll"}) {
    std::filesystem::copy_file(runtime, directory / name,
                               std::filesystem::copy_options::overwrite_existing);
  }
  const char* current_path = std::getenv("PATH");
  const std::string original_path = current_path == nullptr ? "" : current_path;
  set_environment("PATH", directory.string() + ";" + original_path);
  set_environment("RECO_GSTREAMER_DYLIB_PATH", "");
  set_environment("RECO_GSTAPP_DYLIB_PATH", "");
  set_environment("RECO_GLIB_DYLIB_PATH", "");
  try {
    set_scenario("probe-ok");
    expect_eq(probe_video(container_config(video_path), 5'000'000'000ULL).width, 3840U,
              "Windows probe resolves conventional GStreamer DLLs from PATH securely");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: Windows PATH GStreamer discovery threw: " << error.what() << '\n';
    ++failures;
  }
  set_environment("PATH", original_path);
  set_environment("RECO_GSTREAMER_DYLIB_PATH", runtime.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", runtime.string());
  set_environment("RECO_GLIB_DYLIB_PATH", runtime.string());
  std::filesystem::remove_all(directory);
#else
  (void)video_path;
  (void)runtime;
#endif
}

void parent_death_reclaims_worker(const std::filesystem::path& video_path) {
#if defined(__linux__)
  set_scenario("probe-blocking-playing");
  const auto caller = fork();
  if (caller == 0) {
    try {
      (void)probe_video(container_config(video_path), 5'000'000'000ULL);
    } catch (...) {
    }
    std::_Exit(0);
  }
  expect_true(caller > 0, "parent-death probe caller starts");
  if (caller <= 0) {
    return;
  }

  std::optional<pid_t> worker;
  const auto discovery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  const auto task_path = std::filesystem::path("/proc") / std::to_string(caller) / "task";
  while (std::chrono::steady_clock::now() < discovery_deadline && !worker.has_value()) {
    std::error_code directory_error;
    for (const auto& task : std::filesystem::directory_iterator(task_path, directory_error)) {
      std::ifstream children(task.path() / "children");
      pid_t child = -1;
      if (children >> child && child > 0) {
        worker = child;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  expect_true(worker.has_value(), "isolated worker starts before parent-death test");
  (void)kill(caller, SIGKILL);
  int caller_status = 0;
  while (waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
  }

  if (worker.has_value()) {
    const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < exit_deadline &&
           (kill(*worker, 0) == 0 || errno != ESRCH)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect_true(kill(*worker, 0) != 0 && errno == ESRCH,
                "worker process group exits when its parent dies");
  }
  set_scenario("probe-ok");
#else
  (void)video_path;
#endif
}

void expect_no_unreaped_children() {
#if !defined(_WIN32)
  int status = 0;
  errno = 0;
  const auto child = waitpid(-1, &status, WNOHANG);
  expect_true(child == -1 && errno == ECHILD, "all probe workers are reaped before API return");
#endif
}

} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  if (argc == 2 && std::string_view(argv[1]) == "--reco-parent-death-probe-caller") {
    return run_windows_parent_death_probe_caller();
  }
#else
  (void)argc;
  (void)argv;
#endif
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  const auto runtime = resolve_runfile("cpp/tests/libfake_gstreamer_runtime.so");
  probe_worker_path = executable_runfile("cpp/reco_io/reco_video_probe_worker");
  fake_probe_worker_path = executable_runfile("cpp/tests/fake_video_probe_worker");
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
  non_utf8_path_round_trips();
  windows_unicode_path_round_trips();
  windows_path_runtime_discovery(video_path, runtime);
  worker_ipc_failures_are_bounded(video_path);
  delayed_supervision_cannot_launch_worker(video_path);
  launch_gate_prevents_post_timeout_process_start(video_path);
  auto_reaped_workers_are_supported(video_path);
  windows_job_reclaims_worker_descendants(video_path);
  unrelated_descriptors_are_not_inherited(video_path);
  unrelated_descriptor_writer_does_not_delay_pipe_eof(video_path);
  parent_death_reclaims_worker(video_path);
  caller_death_reclaims_worker_and_descendant(video_path);
  expect_no_unreaped_children();

  std::filesystem::remove(video_path);
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
