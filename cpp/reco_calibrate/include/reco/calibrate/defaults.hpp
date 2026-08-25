#pragma once

#include "reco/calibrate/traits.hpp"

namespace reco::calibrate {

class HammingMatcher final : public FeatureMatcher {
public:
  explicit HammingMatcher(double lowe_ratio = 0.75);

  [[nodiscard]] std::vector<RawMatch>
  match_features(const std::vector<Descriptor>& left,
                 const std::vector<Descriptor>& right) const override;

  double lowe_ratio = 0.75;
};

class NoOpFilter final : public PointFilter {
public:
  [[nodiscard]] std::vector<MatchedPoint>
  filter(const std::vector<MatchedPoint>& points) const override;
};

class YDisparityFilter final : public PointFilter {
public:
  explicit YDisparityFilter(double max_disparity = 0.08);

  [[nodiscard]] std::vector<MatchedPoint>
  filter(const std::vector<MatchedPoint>& points) const override;

  double max_disparity = 0.08;
};

class SeamWeightedCost final : public CostFunction {
public:
  SeamWeightedCost(double sigma = 0.08, double trim_fraction = 0.3);

  [[nodiscard]] double cost(const std::vector<MatchedPoint>& points,
                            const OptParams& params) const override;
  [[nodiscard]] std::vector<double>
  per_point_cost(const std::vector<MatchedPoint>& points,
                 const OptParams& params) const override;

  double sigma = 0.08;
  double trim_fraction = 0.3;
};

class RawReprojectionCost final : public CostFunction {
public:
  explicit RawReprojectionCost(double trim_fraction = 0.3);

  [[nodiscard]] double cost(const std::vector<MatchedPoint>& points,
                            const OptParams& params) const override;
  [[nodiscard]] std::vector<double>
  per_point_cost(const std::vector<MatchedPoint>& points,
                 const OptParams& params) const override;

  double trim_fraction = 0.3;
};

} // namespace reco::calibrate
