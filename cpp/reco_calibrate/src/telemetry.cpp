#include "reco/calibrate/telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace reco::calibrate {
namespace {

std::optional<double> rig_tilt_from_quaternions(const TelemetryData& data) {
  if (data.quaternions.size() < 10) {
    return std::nullopt;
  }

  const auto n = std::min<std::size_t>(data.quaternions.size(), 100);
  double aw = 0.0;
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
  const auto q0 = data.quaternions[0].second;
  for (std::size_t i = 0; i < n; ++i) {
    const auto q = data.quaternions[i].second;
    const double dot = q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2] + q[3] * q0[3];
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    aw += q[0] * sign;
    ax += q[1] * sign;
    ay += q[2] * sign;
    az += q[3] * sign;
  }
  const double inv_n = 1.0 / static_cast<double>(n);
  aw *= inv_n;
  ax *= inv_n;
  ay *= inv_n;
  az *= inv_n;

  const double len = std::sqrt(aw * aw + ax * ax + ay * ay + az * az);
  if (len < 1.0e-10) {
    return std::nullopt;
  }
  const double w = aw / len;
  const double x = ax / len;
  const double y = ay / len;
  const double z = az / len;

  const double gx = -2.0 * (x * y - w * z);
  const double gy = -(1.0 - 2.0 * (x * x + z * z));
  return std::atan2(gy, gx);
}

} // namespace

double ImuSample::magnitude() const { return std::sqrt(x * x + y * y + z * z); }

std::optional<double> estimate_sync_offset(const TelemetryData& left, const TelemetryData& right) {
  if (left.gyro.size() < 100 || right.gyro.size() < 100) {
    return std::nullopt;
  }

  std::vector<std::pair<double, double>> left_mag;
  std::vector<std::pair<double, double>> right_mag;
  left_mag.reserve(left.gyro.size());
  right_mag.reserve(right.gyro.size());
  for (const auto& sample : left.gyro) {
    left_mag.push_back({sample.t, sample.magnitude()});
  }
  for (const auto& sample : right.gyro) {
    right_mag.push_back({sample.t, sample.magnitude()});
  }

  constexpr double sample_rate = 200.0;
  const double left_duration = left_mag.back().first - left_mag.front().first;
  const double right_duration = right_mag.back().first - right_mag.front().first;
  const double duration = std::min({left_duration, right_duration, 30.0});
  if (!std::isfinite(duration) || duration < 0.0) {
    return std::nullopt;
  }
  const auto n = static_cast<std::size_t>(duration * sample_rate);
  if (n < 100) {
    return std::nullopt;
  }

  const auto left_resampled = resample_signal(left_mag, left_mag.front().first, sample_rate, n);
  const auto right_resampled = resample_signal(right_mag, right_mag.front().first, sample_rate, n);
  const auto left_norm = normalize_signal(left_resampled);
  const auto right_norm = normalize_signal(right_resampled);
  if (left_norm.empty() || right_norm.empty()) {
    return std::nullopt;
  }

  auto max_lag = static_cast<std::int64_t>(5.0 * sample_rate);
  max_lag = std::min(max_lag, static_cast<std::int64_t>(n) / 2);
  double best_corr = -std::numeric_limits<double>::infinity();
  std::int64_t best_lag = 0;
  for (std::int64_t lag = -max_lag; lag <= max_lag; ++lag) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < left_norm.size(); ++i) {
      const auto j = static_cast<std::int64_t>(i) + lag;
      if (j >= 0 && static_cast<std::size_t>(j) < right_norm.size()) {
        sum += left_norm[i] * right_norm[static_cast<std::size_t>(j)];
        ++count;
      }
    }
    if (count > 0) {
      const double corr = sum / static_cast<double>(count);
      if (corr > best_corr) {
        best_corr = corr;
        best_lag = lag;
      }
    }
  }

  if (best_corr < 0.7) {
    return std::nullopt;
  }
  return -(static_cast<double>(best_lag) / sample_rate);
}

std::optional<std::array<double, 3>> gravity_vector(const TelemetryData& data, double skip_secs) {
  if (data.accel.empty()) {
    return std::nullopt;
  }
  std::size_t start_idx = 0;
  if (skip_secs > 0.0) {
    const auto it = std::find_if(data.accel.begin(), data.accel.end(),
                                 [&](const auto& sample) { return sample.t >= skip_secs; });
    if (it != data.accel.end()) {
      start_idx = static_cast<std::size_t>(std::distance(data.accel.begin(), it));
    }
  }
  const auto end_idx = std::min(start_idx + 200, data.accel.size());
  if (start_idx >= end_idx) {
    return std::nullopt;
  }

  double gx = 0.0;
  double gy = 0.0;
  double gz = 0.0;
  for (std::size_t i = start_idx; i < end_idx; ++i) {
    gx += data.accel[i].x;
    gy += data.accel[i].y;
    gz += data.accel[i].z;
  }
  const double n = static_cast<double>(end_idx - start_idx);
  return std::array<double, 3>{gx / n, gy / n, gz / n};
}

std::optional<std::array<double, 3>> differential_orientation(const TelemetryData& left,
                                                              const TelemetryData& right,
                                                              double skip_secs) {
  const auto lg = gravity_vector(left, skip_secs);
  const auto rg = gravity_vector(right, skip_secs);
  if (!lg.has_value() || !rg.has_value()) {
    return std::nullopt;
  }

  const double left_roll = std::atan2((*lg)[2], (*lg)[0]);
  const double right_roll = std::atan2((*rg)[2], (*rg)[0]);
  const double roll_diff = right_roll - left_roll;

  const double left_pitch = std::atan2((*lg)[1], (*lg)[0]);
  const double right_pitch = std::atan2((*rg)[1], (*rg)[0]);
  const double pitch_diff = right_pitch - left_pitch;

  const double left_tilt = std::atan2((*lg)[2], (*lg)[0]);
  const double right_tilt = std::atan2((*rg)[2], (*rg)[0]);
  const double rig_tilt_avg = (left_tilt + right_tilt) / 2.0;
  const double tilt_diff = left_tilt - rig_tilt_avg;

  return std::array<double, 3>{roll_diff, pitch_diff, tilt_diff};
}

std::optional<double> rig_tilt(const TelemetryData& data, double skip_secs) {
  if (const auto g = gravity_vector(data, skip_secs); g.has_value()) {
    return std::atan2((*g)[1], (*g)[0]);
  }
  return rig_tilt_from_quaternions(data);
}

std::vector<double> normalize_signal(const std::vector<double>& signal) {
  const auto n = signal.size();
  if (n == 0) {
    return {};
  }
  const double mean =
      std::accumulate(signal.begin(), signal.end(), 0.0) / static_cast<double>(n);
  double variance = 0.0;
  for (const double value : signal) {
    const double d = value - mean;
    variance += d * d;
  }
  variance /= static_cast<double>(n);
  const double stddev = std::sqrt(variance);
  if (stddev < 1.0e-15) {
    return {};
  }

  std::vector<double> out;
  out.reserve(n);
  for (const double value : signal) {
    out.push_back((value - mean) / stddev);
  }
  return out;
}

std::vector<double> resample_signal(const std::vector<std::pair<double, double>>& signal,
                                    double t_start, double rate, std::size_t n) {
  std::vector<double> result;
  result.reserve(n);
  std::size_t src_idx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double t = t_start + static_cast<double>(i) / rate;
    while (src_idx + 1 < signal.size() && signal[src_idx + 1].first < t) {
      ++src_idx;
    }
    if (src_idx + 1 >= signal.size()) {
      result.push_back(signal.empty() ? 0.0 : signal.back().second);
      continue;
    }
    const auto [t0, v0] = signal[src_idx];
    const auto [t1, v1] = signal[src_idx + 1];
    const double dt = t1 - t0;
    if (dt > 0.0) {
      const double frac = (t - t0) / dt;
      result.push_back(v0 + frac * (v1 - v0));
    } else {
      result.push_back(v0);
    }
  }
  return result;
}

} // namespace reco::calibrate
