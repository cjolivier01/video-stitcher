#include "reco/core/performance.hpp"

#include <cmath>
#include <sstream>
#include <string>

namespace reco::core {
namespace {

std::string ratio_finding(const char* label, double actual, double required) {
  std::ostringstream out;
  out << label << " ratio " << actual << " is below required " << required;
  return out.str();
}

bool finite_positive(double value) { return std::isfinite(value) && value > 0.0; }

} // namespace

double calibration_pairs_per_second(const CalibrationPerformanceSample& sample) {
  if (!finite_positive(sample.elapsed_seconds) || sample.frame_pairs == 0) {
    return 0.0;
  }
  return static_cast<double>(sample.frame_pairs) / sample.elapsed_seconds;
}

double playback_frames_per_second(const PlaybackPerformanceSample& sample) {
  if (!finite_positive(sample.elapsed_seconds) || sample.frames == 0) {
    return 0.0;
  }
  return static_cast<double>(sample.frames) / sample.elapsed_seconds;
}

std::optional<std::string>
validate_calibration_performance_sample(const CalibrationPerformanceSample& sample,
                                        const char* label) {
  if (sample.frame_pairs == 0) {
    return std::string(label) + " calibration sample has no frame pairs";
  }
  if (!finite_positive(sample.elapsed_seconds)) {
    return std::string(label) + " calibration elapsed time must be finite and positive";
  }
  if (!finite_positive(sample.gpu_busy_seconds)) {
    return std::string(label) + " calibration GPU busy time must be finite and positive";
  }
  if (sample.gpu_busy_seconds > sample.elapsed_seconds) {
    return std::string(label) + " calibration GPU busy time exceeds elapsed time";
  }
  if (!sample.gpu_resident) {
    return std::string(label) + " calibration sample is not GPU-resident";
  }
  return std::nullopt;
}

std::optional<std::string>
validate_playback_performance_sample(const PlaybackPerformanceSample& sample, const char* label) {
  if (sample.frames == 0) {
    return std::string(label) + " playback sample has no frames";
  }
  if (!finite_positive(sample.elapsed_seconds)) {
    return std::string(label) + " playback elapsed time must be finite and positive";
  }
  if (!std::isfinite(sample.dropped_frame_ratio) || sample.dropped_frame_ratio < 0.0 ||
      sample.dropped_frame_ratio > 1.0) {
    return std::string(label) + " playback dropped frame ratio must be finite within [0, 1]";
  }
  if (!finite_positive(sample.p95_frame_latency_seconds)) {
    return std::string(label) + " playback p95 frame latency must be finite and positive";
  }
  if (sample.p95_frame_latency_seconds > sample.elapsed_seconds) {
    return std::string(label) + " playback p95 frame latency exceeds elapsed time";
  }
  if (!sample.gpu_resident) {
    return std::string(label) + " playback sample is not GPU-resident";
  }
  return std::nullopt;
}

std::optional<std::string>
validate_performance_parity_thresholds(const PerformanceParityThresholds& thresholds) {
  if (!std::isfinite(thresholds.min_speed_ratio) || thresholds.min_speed_ratio <= 0.0 ||
      thresholds.min_speed_ratio > 1.0) {
    return "performance parity min_speed_ratio must be finite within (0, 1]";
  }
  if (!std::isfinite(thresholds.max_latency_ratio) || thresholds.max_latency_ratio < 1.0) {
    return "performance parity max_latency_ratio must be finite and >= 1";
  }
  if (!std::isfinite(thresholds.max_dropped_frame_ratio_delta) ||
      thresholds.max_dropped_frame_ratio_delta < 0.0 ||
      thresholds.max_dropped_frame_ratio_delta > 1.0) {
    return "performance parity max_dropped_frame_ratio_delta must be finite within [0, 1]";
  }
  return std::nullopt;
}

PerformanceParityReport
compare_calibration_performance(const CalibrationPerformanceSample& rust_baseline,
                                const CalibrationPerformanceSample& cpp_sample,
                                const PerformanceParityThresholds& thresholds) {
  PerformanceParityReport report;
  if (const auto error = validate_performance_parity_thresholds(thresholds); error.has_value()) {
    report.findings.push_back(*error);
    return report;
  }
  if (const auto error = validate_calibration_performance_sample(rust_baseline, "Rust");
      error.has_value()) {
    report.findings.push_back(*error);
  }
  if (const auto error = validate_calibration_performance_sample(cpp_sample, "C++");
      error.has_value()) {
    report.findings.push_back(*error);
  }
  if (!report.findings.empty()) {
    return report;
  }

  const double rust_rate = calibration_pairs_per_second(rust_baseline);
  const double cpp_rate = calibration_pairs_per_second(cpp_sample);
  const double speed_ratio = cpp_rate / rust_rate;
  if (speed_ratio < thresholds.min_speed_ratio) {
    report.findings.push_back(
        ratio_finding("calibration throughput", speed_ratio, thresholds.min_speed_ratio));
  }
  if (rust_baseline.gpu_busy_seconds == 0.0 && cpp_sample.gpu_busy_seconds > 0.0) {
    report.findings.push_back("calibration GPU busy time regressed from zero in the Rust baseline");
  } else if (rust_baseline.gpu_busy_seconds > 0.0) {
    const double gpu_busy_ratio = cpp_sample.gpu_busy_seconds / rust_baseline.gpu_busy_seconds;
    if (gpu_busy_ratio > thresholds.max_latency_ratio) {
      std::ostringstream out;
      out << "calibration GPU busy ratio " << gpu_busy_ratio << " exceeds allowed "
          << thresholds.max_latency_ratio;
      report.findings.push_back(out.str());
    }
  }
  report.pass = report.findings.empty();
  return report;
}

PerformanceParityReport
compare_playback_performance(const PlaybackPerformanceSample& rust_baseline,
                             const PlaybackPerformanceSample& cpp_sample,
                             const PerformanceParityThresholds& thresholds) {
  PerformanceParityReport report;
  if (const auto error = validate_performance_parity_thresholds(thresholds); error.has_value()) {
    report.findings.push_back(*error);
    return report;
  }
  if (const auto error = validate_playback_performance_sample(rust_baseline, "Rust");
      error.has_value()) {
    report.findings.push_back(*error);
  }
  if (const auto error = validate_playback_performance_sample(cpp_sample, "C++");
      error.has_value()) {
    report.findings.push_back(*error);
  }
  if (!report.findings.empty()) {
    return report;
  }

  const double rust_rate = playback_frames_per_second(rust_baseline);
  const double cpp_rate = playback_frames_per_second(cpp_sample);
  const double speed_ratio = cpp_rate / rust_rate;
  if (speed_ratio < thresholds.min_speed_ratio) {
    report.findings.push_back(
        ratio_finding("playback throughput", speed_ratio, thresholds.min_speed_ratio));
  }
  if (rust_baseline.p95_frame_latency_seconds == 0.0 &&
      cpp_sample.p95_frame_latency_seconds > 0.0) {
    report.findings.push_back(
        "playback p95 frame latency regressed from zero in the Rust baseline");
  } else if (rust_baseline.p95_frame_latency_seconds > 0.0) {
    const double latency_ratio =
        cpp_sample.p95_frame_latency_seconds / rust_baseline.p95_frame_latency_seconds;
    if (latency_ratio > thresholds.max_latency_ratio) {
      std::ostringstream out;
      out << "playback p95 frame latency ratio " << latency_ratio << " exceeds allowed "
          << thresholds.max_latency_ratio;
      report.findings.push_back(out.str());
    }
  }
  const double drop_delta = cpp_sample.dropped_frame_ratio - rust_baseline.dropped_frame_ratio;
  if (drop_delta > thresholds.max_dropped_frame_ratio_delta) {
    std::ostringstream out;
    out << "playback dropped-frame delta " << drop_delta << " exceeds allowed "
        << thresholds.max_dropped_frame_ratio_delta;
    report.findings.push_back(out.str());
  }
  report.pass = report.findings.empty();
  return report;
}

} // namespace reco::core
