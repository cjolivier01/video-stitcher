#include "reco/core/performance.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

using namespace reco::core;

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

void calibration_samples_validate_gpu_residency() {
  CalibrationPerformanceSample sample{
      .frame_pairs = 10, .elapsed_seconds = 2.0, .gpu_busy_seconds = 1.5, .gpu_resident = true};
  expect_eq(calibration_pairs_per_second(sample), 5.0, "calibration pairs per second");
  expect_true(!validate_calibration_performance_sample(sample, "C++").has_value(),
              "valid calibration sample accepted");

  sample.gpu_resident = false;
  const auto error = validate_calibration_performance_sample(sample, "C++");
  expect_true(error.has_value(), "CPU calibration sample rejected");
  expect_true(error->find("not GPU-resident") != std::string::npos,
              "calibration residency error is explicit");
}

void playback_samples_validate_gpu_residency() {
  PlaybackPerformanceSample sample{.frames = 300,
                                   .elapsed_seconds = 5.0,
                                   .p95_frame_latency_seconds = 0.015,
                                   .dropped_frame_ratio = 0.001,
                                   .gpu_resident = true};
  expect_eq(playback_frames_per_second(sample), 60.0, "playback frames per second");
  expect_true(!validate_playback_performance_sample(sample, "C++").has_value(),
              "valid playback sample accepted");

  sample.dropped_frame_ratio = 1.1;
  const auto error = validate_playback_performance_sample(sample, "C++");
  expect_true(error.has_value(), "invalid playback drop ratio rejected");
}

void parity_reports_fail_on_regressions() {
  const CalibrationPerformanceSample rust_cal{
      .frame_pairs = 20, .elapsed_seconds = 2.0, .gpu_busy_seconds = 1.0, .gpu_resident = true};
  const CalibrationPerformanceSample cpp_slow{
      .frame_pairs = 20, .elapsed_seconds = 3.0, .gpu_busy_seconds = 1.0, .gpu_resident = true};
  auto report = compare_calibration_performance(rust_cal, cpp_slow);
  expect_true(!report.pass, "slow calibration fails parity");
  expect_true(!report.findings.empty(), "slow calibration reports finding");

  const PlaybackPerformanceSample rust_play{.frames = 600,
                                            .elapsed_seconds = 10.0,
                                            .p95_frame_latency_seconds = 0.016,
                                            .dropped_frame_ratio = 0.0,
                                            .gpu_resident = true};
  const PlaybackPerformanceSample cpp_drop{.frames = 600,
                                           .elapsed_seconds = 10.0,
                                           .p95_frame_latency_seconds = 0.016,
                                           .dropped_frame_ratio = 0.01,
                                           .gpu_resident = true};
  report = compare_playback_performance(rust_play, cpp_drop);
  expect_true(!report.pass, "dropped playback frames fail parity");
  expect_true(report.findings.front().find("dropped-frame") != std::string::npos,
              "drop finding is explicit");

  const PlaybackPerformanceSample cpp_latency{.frames = 600,
                                              .elapsed_seconds = 10.0,
                                              .p95_frame_latency_seconds = 0.025,
                                              .dropped_frame_ratio = 0.0,
                                              .gpu_resident = true};
  report = compare_playback_performance(rust_play, cpp_latency);
  expect_true(!report.pass, "high playback latency fails parity");
  expect_true(report.findings.front().find("latency") != std::string::npos,
              "latency finding is explicit");
}

void zero_baseline_latency_cannot_mask_regression() {
  const CalibrationPerformanceSample rust_cal{
      .frame_pairs = 20, .elapsed_seconds = 2.0, .gpu_busy_seconds = 0.0, .gpu_resident = true};
  const CalibrationPerformanceSample cpp_cal{
      .frame_pairs = 20, .elapsed_seconds = 2.0, .gpu_busy_seconds = 0.1, .gpu_resident = true};
  auto report = compare_calibration_performance(rust_cal, cpp_cal);
  expect_true(!report.pass, "nonzero C++ GPU busy time fails against zero Rust baseline");

  const PlaybackPerformanceSample rust_play{.frames = 600,
                                            .elapsed_seconds = 10.0,
                                            .p95_frame_latency_seconds = 0.0,
                                            .dropped_frame_ratio = 0.0,
                                            .gpu_resident = true};
  const PlaybackPerformanceSample cpp_play{.frames = 600,
                                           .elapsed_seconds = 10.0,
                                           .p95_frame_latency_seconds = 0.001,
                                           .dropped_frame_ratio = 0.0,
                                           .gpu_resident = true};
  report = compare_playback_performance(rust_play, cpp_play);
  expect_true(!report.pass, "nonzero C++ playback latency fails against zero Rust baseline");
}

void invalid_thresholds_cannot_pass_parity() {
  const CalibrationPerformanceSample rust_cal{
      .frame_pairs = 20, .elapsed_seconds = 2.0, .gpu_busy_seconds = 1.0, .gpu_resident = true};
  const CalibrationPerformanceSample cpp_cal{
      .frame_pairs = 20, .elapsed_seconds = 2.0, .gpu_busy_seconds = 1.0, .gpu_resident = true};

  auto thresholds = PerformanceParityThresholds{.min_speed_ratio = -1.0};
  auto report = compare_calibration_performance(rust_cal, cpp_cal, thresholds);
  expect_true(!report.pass, "negative speed threshold fails parity");
  expect_true(report.findings.front().find("min_speed_ratio") != std::string::npos,
              "invalid speed threshold is reported");

  thresholds = PerformanceParityThresholds{
      .min_speed_ratio = 0.95, .max_latency_ratio = std::numeric_limits<double>::quiet_NaN()};
  report = compare_calibration_performance(rust_cal, cpp_cal, thresholds);
  expect_true(!report.pass, "NaN latency threshold fails parity");

  const PlaybackPerformanceSample rust_play{.frames = 600,
                                            .elapsed_seconds = 10.0,
                                            .p95_frame_latency_seconds = 0.016,
                                            .dropped_frame_ratio = 0.0,
                                            .gpu_resident = true};
  const PlaybackPerformanceSample cpp_play{.frames = 600,
                                           .elapsed_seconds = 10.0,
                                           .p95_frame_latency_seconds = 0.016,
                                           .dropped_frame_ratio = 0.0,
                                           .gpu_resident = true};
  thresholds = PerformanceParityThresholds{.min_speed_ratio = 0.95,
                                           .max_latency_ratio = 1.05,
                                           .max_dropped_frame_ratio_delta =
                                               std::numeric_limits<double>::quiet_NaN()};
  report = compare_playback_performance(rust_play, cpp_play, thresholds);
  expect_true(!report.pass, "NaN drop threshold fails parity");
}

void parity_reports_pass_for_matching_gpu_samples() {
  const CalibrationPerformanceSample rust_cal{
      .frame_pairs = 20, .elapsed_seconds = 2.0, .gpu_busy_seconds = 1.0, .gpu_resident = true};
  const CalibrationPerformanceSample cpp_cal{
      .frame_pairs = 20, .elapsed_seconds = 2.05, .gpu_busy_seconds = 1.02, .gpu_resident = true};
  expect_true(compare_calibration_performance(rust_cal, cpp_cal).pass,
              "matching calibration passes parity");

  const PlaybackPerformanceSample rust_play{.frames = 600,
                                            .elapsed_seconds = 10.0,
                                            .p95_frame_latency_seconds = 0.016,
                                            .dropped_frame_ratio = 0.001,
                                            .gpu_resident = true};
  const PlaybackPerformanceSample cpp_play{.frames = 600,
                                           .elapsed_seconds = 10.2,
                                           .p95_frame_latency_seconds = 0.0165,
                                           .dropped_frame_ratio = 0.002,
                                           .gpu_resident = true};
  expect_true(compare_playback_performance(rust_play, cpp_play).pass,
              "matching playback passes parity");
}

} // namespace

int main() {
  calibration_samples_validate_gpu_residency();
  playback_samples_validate_gpu_residency();
  parity_reports_fail_on_regressions();
  zero_baseline_latency_cannot_mask_regression();
  invalid_thresholds_cannot_pass_parity();
  parity_reports_pass_for_matching_gpu_samples();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
