#pragma once

#include "reco/calibrate/features.hpp"
#include "reco/calibrate/geometry.hpp"
#include "reco/calibrate/types.hpp"

#include <vector>

namespace reco::calibrate {

class FeatureMatcher {
public:
  virtual ~FeatureMatcher() = default;
  [[nodiscard]] virtual std::vector<RawMatch>
  match_features(const std::vector<Descriptor>& left,
                 const std::vector<Descriptor>& right) const = 0;
};

class PointFilter {
public:
  virtual ~PointFilter() = default;
  [[nodiscard]] virtual std::vector<MatchedPoint>
  filter(const std::vector<MatchedPoint>& points) const = 0;
};

class CostFunction {
public:
  virtual ~CostFunction() = default;
  [[nodiscard]] virtual double cost(const std::vector<MatchedPoint>& points,
                                    const OptParams& params) const = 0;
  [[nodiscard]] virtual std::vector<double>
  per_point_cost(const std::vector<MatchedPoint>& points,
                 const OptParams& params) const = 0;
};

} // namespace reco::calibrate
