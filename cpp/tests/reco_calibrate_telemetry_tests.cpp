#include "reco/calibrate/telemetry.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

using namespace reco::calibrate;

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

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

TelemetryData telemetry_with_accel(std::vector<ImuSample> accel) {
  TelemetryData data;
  data.camera_type = "test";
  data.accel = std::move(accel);
  return data;
}

void signal_helpers_match_rust() {
  expect_near(ImuSample{.x = 3.0, .y = 4.0, .z = 12.0}.magnitude(), 13.0, 1.0e-12,
              "imu magnitude");
  expect_true(normalize_signal({1.0, 1.0, 1.0}).empty(), "constant signal normalizes empty");
  const auto norm = normalize_signal({1.0, 2.0, 3.0});
  expect_eq(norm.size(), 3U, "normalized size");
  expect_near(norm[0], -1.224744871391589, 1.0e-12, "normalized first");
  expect_near(norm[2], 1.224744871391589, 1.0e-12, "normalized last");

  const auto resampled = resample_signal({{0.0, 0.0}, {1.0, 10.0}, {2.0, 20.0}}, 0.0, 2.0, 5);
  expect_eq(resampled.size(), 5U, "resampled size");
  expect_near(resampled[1], 5.0, 1.0e-12, "resampled interpolation");
  expect_near(resampled[4], 20.0, 1.0e-12, "resampled tail");
}

void gravity_and_orientation_match_rust() {
  std::vector<ImuSample> left_samples;
  std::vector<ImuSample> right_samples;
  for (std::size_t i = 0; i < 250; ++i) {
    const double t = static_cast<double>(i) * 0.01;
    left_samples.push_back({.t = t, .x = 9.8, .y = 0.98, .z = 0.0});
    right_samples.push_back({.t = t, .x = 9.8, .y = 1.96, .z = 0.98});
  }

  const auto left = telemetry_with_accel(left_samples);
  const auto right = telemetry_with_accel(right_samples);
  const auto g = gravity_vector(left, 1.0);
  expect_true(g.has_value(), "gravity present");
  expect_near((*g)[0], 9.8, 1.0e-12, "gravity x");
  expect_near((*g)[1], 0.98, 1.0e-12, "gravity y");

  const auto diff = differential_orientation(left, right, 0.0);
  expect_true(diff.has_value(), "differential orientation present");
  expect_near((*diff)[0], std::atan2(0.98, 9.8) - std::atan2(0.0, 9.8), 1.0e-12,
              "roll diff");
  expect_near((*diff)[1], std::atan2(1.96, 9.8) - std::atan2(0.98, 9.8), 1.0e-12,
              "pitch diff");
  expect_near((*diff)[2],
              std::atan2(0.0, 9.8) -
                  (std::atan2(0.0, 9.8) + std::atan2(0.98, 9.8)) / 2.0,
              1.0e-12, "tilt diff");

  const auto tilt = rig_tilt(left, 0.0);
  expect_true(tilt.has_value(), "rig tilt from accel");
  expect_near(*tilt, std::atan2(0.98, 9.8), 1.0e-12, "rig tilt value");
  expect_true(!gravity_vector(TelemetryData{}, 0.0).has_value(), "missing accel no gravity");
}

void quaternion_tilt_matches_rust() {
  TelemetryData data;
  data.camera_type = "quat";
  for (std::size_t i = 0; i < 12; ++i) {
    data.quaternions.push_back({static_cast<double>(i) * 0.01, {1.0, 0.0, 0.0, 0.0}});
  }
  const auto tilt = rig_tilt(data, 0.0);
  expect_true(tilt.has_value(), "quaternion tilt present");
  expect_near(*tilt, -1.5707963267948966, 1.0e-12, "identity quaternion tilt");

  TelemetryData too_few;
  too_few.quaternions = {{0.0, {1.0, 0.0, 0.0, 0.0}}};
  expect_true(!rig_tilt(too_few, 0.0).has_value(), "too few quaternions no tilt");
}

void sync_offset_matches_rust_policy() {
  TelemetryData left;
  TelemetryData right;
  const auto signal = [](double t) {
    const double a = std::exp(-((t - 1.7) * (t - 1.7)) / 0.03);
    const double b = 0.6 * std::exp(-((t - 3.1) * (t - 3.1)) / 0.08);
    const double c = 0.2 * std::sin(4.7 * t);
    return 2.0 + a + b + c;
  };
  for (std::size_t i = 0; i < 1000; ++i) {
    const double t = static_cast<double>(i) / 200.0;
    left.gyro.push_back({.t = t, .x = signal(t), .y = 0.0, .z = 0.0});
    right.gyro.push_back({.t = t, .x = signal(t + 0.25), .y = 0.0, .z = 0.0});
  }
  const auto offset = estimate_sync_offset(left, right);
  expect_true(offset.has_value(), "sync offset present");
  expect_near(*offset, 0.25, 0.01, "sync offset value");

  TelemetryData short_data;
  short_data.gyro.resize(10);
  expect_true(!estimate_sync_offset(short_data, short_data).has_value(), "short gyro no sync");

  TelemetryData constant_left;
  TelemetryData constant_right;
  for (std::size_t i = 0; i < 200; ++i) {
    const double t = static_cast<double>(i) / 200.0;
    constant_left.gyro.push_back({.t = t, .x = 1.0, .y = 0.0, .z = 0.0});
    constant_right.gyro.push_back({.t = t, .x = 1.0, .y = 0.0, .z = 0.0});
  }
  expect_true(!estimate_sync_offset(constant_left, constant_right).has_value(),
              "constant gyro no sync");

  TelemetryData decreasing_left;
  TelemetryData decreasing_right;
  for (std::size_t i = 0; i < 200; ++i) {
    const double t = 1.0 - static_cast<double>(i) / 200.0;
    decreasing_left.gyro.push_back({.t = t, .x = static_cast<double>(i), .y = 0.0, .z = 0.0});
    decreasing_right.gyro.push_back({.t = t, .x = static_cast<double>(i), .y = 0.0, .z = 0.0});
  }
  expect_true(!estimate_sync_offset(decreasing_left, decreasing_right).has_value(),
              "decreasing timestamps no sync");

  TelemetryData nan_left = left;
  TelemetryData nan_right = right;
  nan_left.gyro.back().t = std::numeric_limits<double>::quiet_NaN();
  expect_true(!estimate_sync_offset(nan_left, nan_right).has_value(), "nan timestamps no sync");
}

} // namespace

int main() {
  signal_helpers_match_rust();
  gravity_and_orientation_match_rust();
  quaternion_tilt_matches_rust();
  sync_offset_matches_rust_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
