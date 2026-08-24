#include "reco/calibrate/features.hpp"
#include "reco/calibrate/filter.hpp"

#include <cstdlib>
#include <iostream>
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

Descriptor descriptor(std::uint8_t fill) {
  Descriptor out{};
  out.fill(fill);
  return out;
}

KeyPoint keypoint(float x, float y) { return {.x = x, .y = y, .response = 1.0F}; }

void hamming_distance_matches_rust() {
  const auto a = descriptor(0xAB);
  const auto b = descriptor(0xAB);
  expect_eq(hamming_distance(a, b), 0U, "identical descriptor distance");

  Descriptor one{};
  one[0] = 1;
  expect_eq(hamming_distance(Descriptor{}, one), 1U, "one bit descriptor distance");

  expect_eq(hamming_distance(Descriptor{}, descriptor(0xFF)), 512U, "all bits descriptor distance");
}

void descriptor_matching_matches_rust_policy() {
  Descriptor d0{};
  Descriptor d_close{};
  d_close[0] = 0b00000011;
  const auto d_far = descriptor(0xFF);

  const auto matches = match_descriptors({d0}, {d_close, d_far}, 0.7);
  expect_eq(matches.size(), 1U, "ratio test keeps close match");
  expect_eq(matches[0].left_idx, 0U, "match left index");
  expect_eq(matches[0].right_idx, 0U, "match right index");
  expect_eq(matches[0].distance, 2U, "match distance");

  Descriptor l0{};
  Descriptor l1{};
  l1[0] = 0b00001111;
  Descriptor r0 = l1;
  Descriptor r1 = l0;
  const auto sorted = match_descriptors({l0, l1}, {r0, r1}, 1.0);
  expect_eq(sorted.size(), 2U, "cross-check keeps mutual best matches");
  expect_eq(sorted[0].distance, 0U, "matches sorted by distance");
  expect_true(sorted[0].left_idx != sorted[1].left_idx, "distinct left matches");

  Descriptor ambiguous_a{};
  Descriptor ambiguous_b{};
  const auto rejected = match_descriptors({ambiguous_a}, {ambiguous_a, ambiguous_b}, 0.7);
  expect_true(rejected.empty(), "ratio test rejects ambiguous nearest matches");

  Descriptor equal_l0{};
  Descriptor equal_l1{};
  equal_l1[0] = 0b00000110;
  Descriptor equal_r0{};
  equal_r0[0] = 0b00000001;
  Descriptor equal_r1{};
  equal_r1[0] = 0b00000111;
  const auto equal_distance = match_descriptors({equal_l0, equal_l1}, {equal_r0, equal_r1}, 1.0);
  expect_eq(equal_distance.size(), 2U, "equal-distance cross-check keeps both matches");
  if (equal_distance.size() == 2) {
    expect_eq(equal_distance[0].left_idx, 0U, "stable sort preserves first equal-distance match");
    expect_eq(equal_distance[1].left_idx, 1U, "stable sort preserves second equal-distance match");
  }
}

void spatial_filter_matches_rust_overlap_policy() {
  CalibrationConfig config;
  config.matching.min_matches = 1;
  config.matching.spatial_x_threshold = 0.4;
  config.matching.spatial_x_inner = 0.15;

  const std::vector<KeyPoint> kp_left{
      keypoint(100.0F, 540.0F),
      keypoint(900.0F, 540.0F),
      keypoint(1700.0F, 540.0F),
      keypoint(900.0F, 100.0F),
      keypoint(900.0F, 700.0F),
  };
  const std::vector<KeyPoint> kp_right{
      keypoint(100.0F, 540.0F),
      keypoint(500.0F, 540.0F),
      keypoint(900.0F, 540.0F),
      keypoint(500.0F, 100.0F),
      keypoint(500.0F, 900.0F),
  };
  const std::vector<RawMatch> matches{
      {.left_idx = 0, .right_idx = 0, .distance = 10},
      {.left_idx = 1, .right_idx = 1, .distance = 20},
      {.left_idx = 2, .right_idx = 2, .distance = 30},
      {.left_idx = 3, .right_idx = 3, .distance = 40},
      {.left_idx = 4, .right_idx = 4, .distance = 50},
  };

  const auto result = spatial_filter(matches, kp_left, kp_right, 1920, 1080, 1920, 1080, config);
  expect_eq(result.size(), 1U, "spatial filter keeps only overlap match");
  expect_eq(result[0].left_idx, 1U, "spatial filter left index");
  expect_eq(result[0].right_idx, 1U, "spatial filter right index");

  CalibrationConfig strict;
  strict.matching.min_matches = 100;
  const auto empty = spatial_filter({RawMatch{.left_idx = 0, .right_idx = 0, .distance = 10}},
                                    {keypoint(100.0F, 540.0F)}, {keypoint(500.0F, 540.0F)},
                                    1920, 1080, 1920, 1080, strict);
  expect_true(empty.empty(), "spatial filter does not fallback to raw matches");

  bool invalid_left_threw = false;
  try {
    (void)spatial_filter({RawMatch{.left_idx = 99, .right_idx = 0, .distance = 10}},
                         {keypoint(100.0F, 540.0F)}, {keypoint(500.0F, 540.0F)}, 1920, 1080,
                         1920, 1080, config);
  } catch (const std::out_of_range&) {
    invalid_left_threw = true;
  }
  expect_true(invalid_left_threw, "invalid left keypoint index throws");

  bool invalid_right_threw = false;
  try {
    (void)spatial_filter({RawMatch{.left_idx = 0, .right_idx = 99, .distance = 10}},
                         {keypoint(100.0F, 540.0F)}, {keypoint(500.0F, 540.0F)}, 1920, 1080,
                         1920, 1080, config);
  } catch (const std::out_of_range&) {
    invalid_right_threw = true;
  }
  expect_true(invalid_right_threw, "invalid right keypoint index throws");
}

} // namespace

int main() {
  hamming_distance_matches_rust();
  descriptor_matching_matches_rust_policy();
  spatial_filter_matches_rust_overlap_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
