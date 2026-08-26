#include "reco/io/gpu_video_probe.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <algorithm>
#include <array>
#include <cerrno>
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
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <csignal>
#include <sys/wait.h>
#endif
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
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
  const auto vfr_fallback = probe_video(container_config(video_path), timeout_ns);
  expect_eq(vfr_fallback.fps_numerator, 30U, "VFR stream uses explicit fallback frame rate");
  expect_eq(vfr_fallback.total_frames, 135ULL,
            "VFR stream preserves its exact compressed access-unit count");
  expect_eq(vfr_fallback.duration_ns, 4'720'000'000ULL,
            "VFR EOS scan preserves observed presentation duration");
  expect_true(!vfr_fallback.total_frames_is_estimated,
              "VFR EOS scan proves its compressed AU count");

  set_scenario("probe-vfr-late-transition");
  const auto late_vfr = probe_video(container_config(video_path), timeout_ns);
  expect_eq(late_vfr.fps_numerator, 45U,
            "late VFR transition retains the plausible average caps rate");
  expect_eq(late_vfr.total_frames, 270ULL,
            "late VFR transition is counted by access units instead of timestamp slots");
  expect_true(!late_vfr.total_frames_is_estimated, "late VFR EOS scan reports an exact count");

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
  const auto duplicate_transition = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_transition.fps_numerator, 25U,
            "duplicate PTS multiplicity must remain uniform across the bounded scan");
  expect_eq(duplicate_transition.fps_denominator, 1U,
            "duplicate PTS transition preserves the caps denominator");
  expect_eq(duplicate_transition.total_frames, 513ULL,
            "duplicate PTS transition preserves the observed AU lower bound");
  expect_true(duplicate_transition.total_frames_is_estimated,
              "bounded duplicate PTS transition count remains explicitly estimated");

  set_scenario("probe-duplicate-pts-reorder-cutoff");
  const auto duplicate_reorder_cutoff = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_reorder_cutoff.fps_numerator, 50U,
            "reordered cutoff group does not invalidate uniform duplicate PTS timing");
  expect_eq(duplicate_reorder_cutoff.fps_denominator, 1U,
            "reordered duplicate PTS timing retains its inferred denominator");
  expect_true(duplicate_reorder_cutoff.total_frames >= 513ULL,
              "reordered duplicate PTS estimate preserves the observed AU lower bound");
  expect_true(duplicate_reorder_cutoff.total_frames_is_estimated,
              "bounded reordered duplicate PTS count remains explicitly estimated");

  set_scenario("probe-duplicate-pts-transition-untimed-tail");
  const auto duplicate_transition_untimed_tail =
      probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_transition_untimed_tail.fps_numerator, 25U,
            "untimed tail does not hide an earlier duplicate PTS cadence transition");
  expect_eq(duplicate_transition_untimed_tail.fps_denominator, 1U,
            "duplicate transition before an untimed tail preserves the caps denominator");
  expect_eq(duplicate_transition_untimed_tail.total_frames, 513ULL,
            "duplicate transition before an untimed tail preserves the AU lower bound");

  set_scenario("probe-duplicate-pts-larger-reorder-suffix");
  const auto larger_reorder_suffix = probe_video(container_config(video_path), timeout_ns);
  expect_eq(larger_reorder_suffix.fps_numerator, 25U,
            "larger duplicate groups in the reorder suffix invalidate multiplicity");
  expect_eq(larger_reorder_suffix.fps_denominator, 1U,
            "larger reorder-suffix groups preserve the caps denominator");
  expect_eq(larger_reorder_suffix.total_frames, 513ULL,
            "larger reorder-suffix groups preserve the AU lower bound");

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
  const auto sparse_exact = probe_video(container_config(video_path), timeout_ns);
  expect_eq(sparse_exact.fps_numerator, 30U,
            "sparse exact timestamp grid wins over a gap-biased span average");
  expect_eq(sparse_exact.fps_denominator, 1U,
            "sparse timestamp gaps preserve the exact cadence denominator");
  expect_eq(sparse_exact.duration_ns, 20'100'000'000ULL,
            "sparse timestamp gaps preserve the presentation duration");
  expect_eq(sparse_exact.total_frames, 600ULL,
            "sparse timestamp gaps retain the EOS-proven AU count");

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

  set_scenario("probe-vfr-missing-durations");
  const auto missing_durations = probe_video(container_config(video_path), timeout_ns);
  expect_eq(missing_durations.duration_ns, 6'000'000'000ULL,
            "missing buffer durations use the observed terminal PTS interval");
  expect_eq(missing_durations.total_frames, 270ULL,
            "missing buffer durations do not change the EOS-proven AU count");
  expect_true(missing_durations.duration_is_estimated,
              "duration inferred from PTS intervals is explicitly estimated");

  set_scenario("probe-dropped-frame-after-prefix");
  const auto dropped_frame = probe_video(container_config(video_path), timeout_ns);
  expect_eq(dropped_frame.total_frames, 179ULL,
            "dropped AU after the timing prefix is not reconstructed as a timestamp slot");
  expect_true(!dropped_frame.total_frames_is_estimated,
              "dropped-frame EOS scan reports an exact count");

  set_scenario("probe-reduced-cadence-after-prefix");
  const auto reduced_cadence = probe_video(container_config(video_path), timeout_ns);
  expect_eq(reduced_cadence.fps_numerator, 45U,
            "reduced cadence retains the plausible average caps numerator");
  expect_eq(reduced_cadence.fps_denominator, 2U,
            "reduced cadence retains the plausible average caps denominator");
  expect_eq(reduced_cadence.total_frames, 135ULL,
            "reduced cadence after the prefix is counted through EOS");
  expect_true(!reduced_cadence.total_frames_is_estimated,
              "reduced-cadence EOS scan reports an exact count");

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
  const auto unseekable = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unseekable.duration_ns, 10'000'000'000ULL,
            "unseekable parser retains bounded container-duration fallback");
  expect_true(unseekable.duration_is_estimated,
              "uncorrelated container duration is explicitly marked estimated");
  expect_true(unseekable.total_frames_is_estimated,
              "unseekable long-stream frame count is explicitly estimated");

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

  std::filesystem::remove(event_path);
  set_scenario("probe-seek-untimestamped-tail");
  const auto untimestamped_tail = probe_video(container_config(video_path), timeout_ns);
  expect_eq(untimestamped_tail.duration_ns, 20'000'000'000ULL,
            "untimestamped seek tail falls back to the selected-stream duration");
  expect_eq(untimestamped_tail.total_frames, 600ULL,
            "untimestamped seek tail estimates frame count from validated caps");
  expect_true(untimestamped_tail.total_frames_is_estimated,
              "untimestamped seek-tail count remains explicitly estimated");
  expect_true(untimestamped_tail.duration_is_estimated,
              "untimestamped seek-tail duration remains explicitly estimated");
  const auto untimestamped_tail_events = read_events(event_path);
  expect_true(std::count(untimestamped_tail_events.begin(), untimestamped_tail_events.end(),
                         "pull-probe") <= 514,
              "untimestamped seek tail abandons correlation without a linear parser drain");

  set_scenario("probe-seek-dts-reorder-tail");
  const auto dts_reorder_tail = probe_video(container_config(video_path), timeout_ns);
  expect_eq(dts_reorder_tail.duration_ns, 20'000'000'000ULL,
            "DTS-only reordered seek tail retains the final presentation span");
  expect_eq(dts_reorder_tail.total_frames, 600ULL,
            "DTS-only reordered seek tail retains the final presentation frame");
  expect_true(dts_reorder_tail.total_frames_is_estimated,
              "DTS-correlated bounded count remains explicitly estimated");
  expect_true(dts_reorder_tail.duration_is_estimated,
              "DTS-correlated bounded duration remains explicitly estimated");

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
#if defined(__linux__)
  const auto descendant_path =
      video_path.parent_path() / (video_path.filename().string() + ".probe-descendant");
  std::filesystem::remove(descendant_path);
  bool descendant_probe_succeeded = false;
#endif
  try {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    probe_succeeded = reco::io::probe_gpu_video(container_config(video_path),
                                                fake_probe_worker_path, 5'000'000'000ULL)
                          .width == 854U;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: auto-reaped worker probe threw: " << error.what() << '\n';
    ++failures;
  }
#if defined(__linux__)
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
#endif
  if (sigaction(SIGCHLD, &previous_action, nullptr) != 0) {
    expect_true(false, "SIGCHLD policy restores");
  }
  expect_true(probe_succeeded, "auto-reaped worker returns its framed response");
#if defined(__linux__)
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
#endif
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

int main() {
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
  worker_ipc_failures_are_bounded(video_path);
  auto_reaped_workers_are_supported(video_path);
  unrelated_descriptors_are_not_inherited(video_path);
  parent_death_reclaims_worker(video_path);
  expect_no_unreaped_children();

  std::filesystem::remove(video_path);
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
