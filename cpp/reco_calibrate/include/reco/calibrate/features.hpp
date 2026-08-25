#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reco::calibrate {

inline constexpr std::size_t kDescriptorBytes = 64;
using Descriptor = std::array<std::uint8_t, kDescriptorBytes>;

struct KeyPoint {
  float x = 0.0F;
  float y = 0.0F;
  float response = 0.0F;
};

struct RawMatch {
  std::size_t left_idx = 0;
  std::size_t right_idx = 0;
  std::uint32_t distance = 0;
};

struct DetectRegion {
  float x_min = 0.0F;
  float x_max = 1.0F;
  float y_min = 0.0F;
  float y_max = 1.0F;
};

[[nodiscard]] std::uint32_t hamming_distance(const Descriptor& a, const Descriptor& b);
[[nodiscard]] std::vector<RawMatch> match_descriptors(const std::vector<Descriptor>& left,
                                                      const std::vector<Descriptor>& right,
                                                      double ratio);

} // namespace reco::calibrate
