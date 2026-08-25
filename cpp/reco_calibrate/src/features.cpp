#include "reco/calibrate/features.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <optional>
#include <tuple>

namespace reco::calibrate {
namespace {

inline constexpr std::size_t kDescriptorWords = kDescriptorBytes / 8;

std::vector<std::tuple<std::size_t, std::size_t, std::uint32_t>>
find_matches_one_way(const std::vector<Descriptor>& query, const std::vector<Descriptor>& train,
                     double ratio) {
  const bool use_ratio_test = ratio < 1.0;
  std::vector<std::tuple<std::size_t, std::size_t, std::uint32_t>> matches;

  for (std::size_t q_idx = 0; q_idx < query.size(); ++q_idx) {
    std::uint32_t best_dist = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t second_dist = std::numeric_limits<std::uint32_t>::max();
    std::size_t best_idx = 0;

    for (std::size_t t_idx = 0; t_idx < train.size(); ++t_idx) {
      const auto dist = hamming_distance(query[q_idx], train[t_idx]);
      if (dist < best_dist) {
        second_dist = best_dist;
        best_dist = dist;
        best_idx = t_idx;
      } else if (dist < second_dist) {
        second_dist = dist;
      }
    }

    if (use_ratio_test) {
      if (second_dist > 0 &&
          static_cast<double>(best_dist) < ratio * static_cast<double>(second_dist)) {
        matches.push_back({q_idx, best_idx, best_dist});
      }
    } else if (best_dist < std::numeric_limits<std::uint32_t>::max()) {
      matches.push_back({q_idx, best_idx, best_dist});
    }
  }
  return matches;
}

} // namespace

std::uint32_t hamming_distance(const Descriptor& a, const Descriptor& b) {
  std::uint32_t dist = 0;
  for (std::size_t i = 0; i < kDescriptorWords; ++i) {
    std::uint64_t wa = 0;
    std::uint64_t wb = 0;
    for (std::size_t byte = 0; byte < 8; ++byte) {
      wa |= static_cast<std::uint64_t>(a[i * 8 + byte]) << (byte * 8);
      wb |= static_cast<std::uint64_t>(b[i * 8 + byte]) << (byte * 8);
    }
    dist += std::popcount(wa ^ wb);
  }
  return dist;
}

std::vector<RawMatch> match_descriptors(const std::vector<Descriptor>& left,
                                        const std::vector<Descriptor>& right, double ratio) {
  const auto forward = find_matches_one_way(left, right, ratio);
  const auto backward = find_matches_one_way(right, left, ratio);

  std::vector<std::optional<std::size_t>> right_to_left(right.size());
  for (const auto& [r_idx, l_idx, distance] : backward) {
    (void)distance;
    right_to_left[r_idx] = l_idx;
  }

  std::vector<RawMatch> matches;
  for (const auto& [l_idx, r_idx, distance] : forward) {
    if (r_idx < right_to_left.size() && right_to_left[r_idx] == l_idx) {
      matches.push_back({.left_idx = l_idx, .right_idx = r_idx, .distance = distance});
    }
  }
  std::stable_sort(matches.begin(), matches.end(),
                   [](const auto& a, const auto& b) { return a.distance < b.distance; });
  return matches;
}

} // namespace reco::calibrate
