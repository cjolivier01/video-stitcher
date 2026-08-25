#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace reco::core {

struct CalibrationPerformanceSample {
  std::uint64_t frame_pairs = 0;
  double elapsed_seconds = 0.0;
  double gpu_busy_seconds = 0.0;
  bool gpu_resident = false;
};

struct PlaybackPerformanceSample {
  std::uint64_t frames = 0;
  double elapsed_seconds = 0.0;
  double p95_frame_latency_seconds = 0.0;
  double dropped_frame_ratio = 0.0;
  bool gpu_resident = false;
};

struct PerformanceParityThresholds {
  double min_speed_ratio = 0.95;
  double max_latency_ratio = 1.05;
  double max_dropped_frame_ratio_delta = 0.002;
};

struct PerformanceParityReport {
  bool pass = false;
  std::vector<std::string> findings;
};

[[nodiscard]] double calibration_pairs_per_second(const CalibrationPerformanceSample& sample);
[[nodiscard]] double playback_frames_per_second(const PlaybackPerformanceSample& sample);
[[nodiscard]] std::optional<std::string>
validate_calibration_performance_sample(const CalibrationPerformanceSample& sample,
                                        const char* label);
[[nodiscard]] std::optional<std::string>
validate_playback_performance_sample(const PlaybackPerformanceSample& sample, const char* label);
[[nodiscard]] std::optional<std::string>
validate_performance_parity_thresholds(const PerformanceParityThresholds& thresholds);
[[nodiscard]] PerformanceParityReport
compare_calibration_performance(const CalibrationPerformanceSample& rust_baseline,
                                const CalibrationPerformanceSample& cpp_sample,
                                const PerformanceParityThresholds& thresholds = {});
[[nodiscard]] PerformanceParityReport
compare_playback_performance(const PlaybackPerformanceSample& rust_baseline,
                             const PlaybackPerformanceSample& cpp_sample,
                             const PerformanceParityThresholds& thresholds = {});

} // namespace reco::core
