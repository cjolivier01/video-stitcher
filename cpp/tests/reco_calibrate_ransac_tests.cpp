#include "reco/calibrate/ransac.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
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

void ransac_with_perfect_points_matches_rust_property() {
  const std::vector<Point2d> pts1{{100.0, 100.0}, {200.0, 100.0}, {300.0, 200.0},
                                  {150.0, 300.0}, {400.0, 150.0}, {250.0, 250.0},
                                  {350.0, 350.0}, {450.0, 200.0}, {180.0, 180.0},
                                  {320.0, 120.0}};
  std::vector<Point2d> pts2;
  pts2.reserve(pts1.size());
  for (const auto& p : pts1) {
    pts2.push_back({p[0] + 20.0 + p[0] * 0.02, p[1] + p[0] * 0.01});
  }

  const auto inliers = ransac_fundamental(pts1, pts2, 5.0, 1000);
  expect_true(inliers.size() >= 8, "perfect synthetic data returns at least 8 inliers");
}

void ransac_with_outliers_matches_rust_property() {
  std::vector<Point2d> pts1;
  for (std::size_t i = 0; i < 20; ++i) {
    const double x = 100.0 + static_cast<double>(i % 5) * 80.0;
    const double y = 100.0 + static_cast<double>(i / 5) * 80.0;
    pts1.push_back({x, y});
  }
  std::vector<Point2d> pts2;
  pts2.reserve(pts1.size());
  for (const auto& p : pts1) {
    const double depth = 1.0 + p[0] * 0.001 + p[1] * 0.0005;
    pts2.push_back({p[0] + 20.0 / depth, p[1] + 5.0 * p[0] * 0.001 / depth});
  }
  for (std::size_t i = 0; i < 5; ++i) {
    pts2[i][0] += 500.0;
    pts2[i][1] -= 300.0;
  }

  const auto inliers = ransac_fundamental(pts1, pts2, 5.0, 2000);
  expect_true(inliers.size() >= 10, "outlier case keeps most good points");
}

void invalid_inputs_fail_loudly() {
  bool too_few_threw = false;
  try {
    (void)ransac_fundamental(std::vector<Point2d>(5, {1.0, 2.0}),
                             std::vector<Point2d>(5, {3.0, 4.0}), 1.0, 100);
  } catch (const std::invalid_argument&) {
    too_few_threw = true;
  }
  expect_true(too_few_threw, "too few points throws");

  bool mismatched_threw = false;
  try {
    (void)ransac_fundamental(std::vector<Point2d>(8, {1.0, 2.0}),
                             std::vector<Point2d>(9, {3.0, 4.0}), 1.0, 100);
  } catch (const std::invalid_argument&) {
    mismatched_threw = true;
  }
  expect_true(mismatched_threw, "mismatched point counts throw");
}

void sampson_degenerate_denominator_matches_rust() {
  const Matrix3d zero{};
  expect_eq(sampson_error(zero, {1.0, 2.0}, {3.0, 4.0}),
            std::numeric_limits<double>::max(), "zero matrix denominator returns max");
}

} // namespace

int main() {
  ransac_with_perfect_points_matches_rust_property();
  ransac_with_outliers_matches_rust_property();
  invalid_inputs_fail_loudly();
  sampson_degenerate_denominator_matches_rust();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
