#pragma once

#include "reco/calibrate/geometry.hpp"
#include "reco/calibrate/types.hpp"

#include "reco/core/calibration.hpp"

#include <utility>
#include <vector>

namespace reco::calibrate {

inline constexpr double kSimplexPerturbation = 0.10;

struct OptimizeResult {
  reco::core::PlaneLayout layout;
  double residual_error = 0.0;
};

class Optimizer {
public:
  virtual ~Optimizer() = default;
  [[nodiscard]] virtual OptimizeResult optimize(const std::vector<MatchedPoint>& points,
                                                const CalibrationConfig& config) const = 0;
};

class NelderMeadOptimizer final : public Optimizer {
public:
  [[nodiscard]] OptimizeResult optimize(const std::vector<MatchedPoint>& points,
                                        const CalibrationConfig& config) const override;
};

[[nodiscard]] std::vector<std::pair<double, double>>
active_bounds(bool enable_x_rx, bool lock_cam_d, bool lock_z_rx);
[[nodiscard]] OptParams params_from_vec(const std::vector<double>& p);
[[nodiscard]] OptParams params_from_vec_no_zrx(const std::vector<double>& p);
[[nodiscard]] OptParams params_from_vec_locked(const std::vector<double>& p);
[[nodiscard]] OptParams params_from_vec_locked_no_zrx(const std::vector<double>& p);
[[nodiscard]] double bounds_penalty(const std::vector<double>& p,
                                    const std::vector<std::pair<double, double>>& bounds);
[[nodiscard]] std::vector<std::vector<double>>
build_simplex(const std::vector<double>& start,
              const std::vector<std::pair<double, double>>& bounds);
[[nodiscard]] OptimizeResult optimize(const std::vector<MatchedPoint>& points,
                                      const CalibrationConfig& config);

} // namespace reco::calibrate
