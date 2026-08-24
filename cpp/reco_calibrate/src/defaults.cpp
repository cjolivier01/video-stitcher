#include "reco/calibrate/defaults.hpp"

#include "reco/calibrate/features.hpp"

#include <cmath>

namespace reco::calibrate {

HammingMatcher::HammingMatcher(double lowe_ratio) : lowe_ratio(lowe_ratio) {}

std::vector<RawMatch> HammingMatcher::match_features(const std::vector<Descriptor>& left,
                                                     const std::vector<Descriptor>& right) const {
  return match_descriptors(left, right, lowe_ratio);
}

std::vector<MatchedPoint> NoOpFilter::filter(const std::vector<MatchedPoint>& points) const {
  return points;
}

YDisparityFilter::YDisparityFilter(double max_disparity) : max_disparity(max_disparity) {}

std::vector<MatchedPoint> YDisparityFilter::filter(const std::vector<MatchedPoint>& points) const {
  std::vector<MatchedPoint> out;
  out.reserve(points.size());
  for (const auto& point : points) {
    const double dy = std::abs(point.left[1] - point.right[1]);
    if (dy < max_disparity) {
      out.push_back(point);
    }
  }
  return out;
}

SeamWeightedCost::SeamWeightedCost(double sigma, double trim_fraction)
    : sigma(sigma), trim_fraction(trim_fraction) {}

double SeamWeightedCost::cost(const std::vector<MatchedPoint>& points,
                              const OptParams& params) const {
  if (trim_fraction > 0.0) {
    return trimmed_seam_weighted_reprojection_error(points, params, sigma, trim_fraction);
  }
  return seam_weighted_reprojection_error(points, params, sigma);
}

std::vector<double> SeamWeightedCost::per_point_cost(const std::vector<MatchedPoint>& points,
                                                     const OptParams& params) const {
  return per_point_seam_weighted_errors(points, params, sigma);
}

RawReprojectionCost::RawReprojectionCost(double trim_fraction) : trim_fraction(trim_fraction) {}

double RawReprojectionCost::cost(const std::vector<MatchedPoint>& points,
                                 const OptParams& params) const {
  if (trim_fraction > 0.0) {
    return trimmed_reprojection_error(points, params, trim_fraction);
  }
  return reprojection_error(points, params);
}

std::vector<double> RawReprojectionCost::per_point_cost(const std::vector<MatchedPoint>& points,
                                                        const OptParams& params) const {
  return per_point_reprojection_error(points, params);
}

} // namespace reco::calibrate
