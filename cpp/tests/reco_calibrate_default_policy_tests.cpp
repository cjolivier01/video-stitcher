#include "reco/calibrate/defaults.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

using namespace reco::calibrate;

namespace {

int failures = 0;

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

Descriptor descriptor(std::uint8_t fill) {
  Descriptor out{};
  out.fill(fill);
  return out;
}

OptParams good_params() {
  return {.x_ty = 0.0, .intersect = 0.5, .cam_d = 0.25, .x_rz = 0.0, .z_rx = 0.0};
}

void hamming_matcher_wraps_descriptor_matching() {
  Descriptor left{};
  Descriptor close{};
  close[0] = 0b00000011;
  const auto matches = HammingMatcher(0.7).match_features({left}, {close, descriptor(0xFF)});
  expect_eq(matches.size(), 1U, "hamming matcher match count");
  expect_eq(matches[0].distance, 2U, "hamming matcher distance");
}

void point_filters_match_rust_defaults() {
  const std::vector<MatchedPoint> points{
      MatchedPoint::from_planes({0.0, 0.0}, {0.0, 0.01}),
      MatchedPoint::from_planes({0.0, 0.0}, {0.0, 0.2}),
  };

  const auto passthrough = NoOpFilter().filter(points);
  expect_eq(passthrough.size(), 2U, "noop filter keeps all points");
  expect_near(passthrough[1].right[1], 0.2, 1.0e-12, "noop preserves data");

  const auto filtered = YDisparityFilter().filter(points);
  expect_eq(filtered.size(), 1U, "y disparity filter rejects large dy");
  expect_near(filtered[0].right[1], 0.01, 1.0e-12, "y disparity keeps small dy");
}

void costs_match_geometry_helpers() {
  const auto params = good_params();
  const std::vector<MatchedPoint> points{
      MatchedPoint{.left = {0.1, 0.0},
                   .right = {0.1, 0.0},
                   .left_pixel_nx = 0.25,
                   .right_pixel_nx = 0.75},
      MatchedPoint{.left = {0.2, 0.02},
                   .right = {0.15, 0.01},
                   .left_pixel_nx = 0.3,
                   .right_pixel_nx = 0.7},
  };

  const RawReprojectionCost raw_no_trim(0.0);
  expect_near(raw_no_trim.cost(points, params), reprojection_error(points, params), 1.0e-12,
              "raw cost delegates to reprojection error");
  expect_eq(raw_no_trim.per_point_cost(points, params).size(), 2U, "raw per point count");

  const RawReprojectionCost raw_trimmed(0.5);
  expect_near(raw_trimmed.cost(points, params), trimmed_reprojection_error(points, params, 0.5),
              1.0e-12, "raw trimmed cost delegates");

  const SeamWeightedCost seam_no_trim(0.08, 0.0);
  expect_near(seam_no_trim.cost(points, params),
              seam_weighted_reprojection_error(points, params, 0.08), 1.0e-12,
              "seam cost delegates");
  expect_eq(seam_no_trim.per_point_cost(points, params).size(), 2U, "seam per point count");

  const SeamWeightedCost seam_trimmed(0.08, 0.5);
  expect_near(seam_trimmed.cost(points, params),
              trimmed_seam_weighted_reprojection_error(points, params, 0.08, 0.5), 1.0e-12,
              "seam trimmed cost delegates");
}

} // namespace

int main() {
  hamming_matcher_wraps_descriptor_matching();
  point_filters_match_rust_defaults();
  costs_match_geometry_helpers();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
