#include "reco/calibrate/optimizer.hpp"

#include "reco/calibrate/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace reco::calibrate {
namespace {

using Bounds = std::vector<std::pair<double, double>>;

inline constexpr std::pair<double, double> kBounds5[5] = {
    {0.1, 0.30}, {-0.0, 1.0}, {-0.1, 0.1}, {-0.3, 0.3}, {-0.3, 0.3}};
inline constexpr std::pair<double, double> kXRxBound{-0.3, 0.3};
inline constexpr double kPenaltyScale = 1.0e4;

const std::vector<std::vector<double>> kStarts5{{0.225, 0.5, 0.0, 0.0, 0.0},
                                                {0.15, 0.5, 0.0, 0.0, 0.0},
                                                {0.30, 0.5, 0.0, 0.0, 0.0},
                                                {0.225, 0.3, 0.0, 0.0, 0.0},
                                                {0.225, 0.7, 0.0, 0.0, 0.0},
                                                {0.20, 0.5, 0.0, 0.0, 0.0},
                                                {0.19, 0.65, 0.0, 0.0, 0.0},
                                                {0.15, 0.7, 0.0, 0.0, 0.0}};
const std::vector<std::vector<double>> kStarts4{{0.5, 0.0, 0.0, 0.0},
                                                {0.3, 0.0, 0.0, 0.0},
                                                {0.7, 0.0, 0.0, 0.0},
                                                {0.4, 0.0, 0.0, 0.0},
                                                {0.6, 0.0, 0.0, 0.0}};

struct CalibrationCost {
  const std::vector<MatchedPoint>& points;
  double sigma = 0.08;
  Bounds bounds;
  bool lock_cam_d = false;
  bool lock_z_rx = false;
  double trim_fraction = 0.3;

  [[nodiscard]] double operator()(const std::vector<double>& p) const {
    const auto params = [&]() {
      if (lock_cam_d && lock_z_rx) {
        return params_from_vec_locked_no_zrx(p);
      }
      if (lock_cam_d) {
        return params_from_vec_locked(p);
      }
      if (lock_z_rx) {
        return params_from_vec_no_zrx(p);
      }
      return params_from_vec(p);
    }();

    const double err =
        trim_fraction > 0.0
            ? trimmed_seam_weighted_reprojection_error(points, params, sigma, trim_fraction)
            : seam_weighted_reprojection_error(points, params, sigma);
    return err + bounds_penalty(p, bounds);
  }
};

double standard_deviation(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  double variance = 0.0;
  for (const double value : values) {
    const double d = value - mean;
    variance += d * d;
  }
  variance /= static_cast<double>(values.size());
  return std::sqrt(variance);
}

std::vector<double> centroid_except_worst(const std::vector<std::vector<double>>& simplex) {
  const std::size_t n = simplex.front().size();
  std::vector<double> centroid(n, 0.0);
  for (std::size_t i = 0; i < simplex.size() - 1; ++i) {
    for (std::size_t d = 0; d < n; ++d) {
      centroid[d] += simplex[i][d];
    }
  }
  for (double& value : centroid) {
    value /= static_cast<double>(simplex.size() - 1);
  }
  return centroid;
}

std::vector<double> combine(const std::vector<double>& a, const std::vector<double>& b,
                            double b_scale) {
  std::vector<double> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b_scale * (b[i] - a[i]);
  }
  return out;
}

std::optional<std::pair<std::vector<double>, double>>
run_nelder_mead(const CalibrationCost& cost, const std::vector<double>& start,
                std::size_t max_iters) {
  auto simplex = build_simplex(start, cost.bounds);
  if (simplex.empty()) {
    return std::nullopt;
  }
  std::vector<double> costs(simplex.size());
  for (std::size_t i = 0; i < simplex.size(); ++i) {
    costs[i] = cost(simplex[i]);
  }

  for (std::size_t iter = 0; iter < max_iters; ++iter) {
    std::vector<std::size_t> order(simplex.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return costs[a] < costs[b]; });

    std::vector<std::vector<double>> sorted_simplex;
    std::vector<double> sorted_costs;
    sorted_simplex.reserve(simplex.size());
    sorted_costs.reserve(costs.size());
    for (const auto idx : order) {
      sorted_simplex.push_back(std::move(simplex[idx]));
      sorted_costs.push_back(costs[idx]);
    }
    simplex = std::move(sorted_simplex);
    costs = std::move(sorted_costs);

    if (standard_deviation(costs) <= 1.0e-12) {
      break;
    }

    const std::size_t worst = simplex.size() - 1;
    const auto centroid = centroid_except_worst(simplex);
    const auto reflected = combine(centroid, simplex[worst], -1.0);
    const double reflected_cost = cost(reflected);

    if (reflected_cost < costs[0]) {
      const auto expanded = combine(centroid, reflected, 2.0);
      const double expanded_cost = cost(expanded);
      if (expanded_cost < reflected_cost) {
        simplex[worst] = expanded;
        costs[worst] = expanded_cost;
      } else {
        simplex[worst] = reflected;
        costs[worst] = reflected_cost;
      }
    } else if (reflected_cost < costs[worst - 1]) {
      simplex[worst] = reflected;
      costs[worst] = reflected_cost;
    } else {
      const bool outside = reflected_cost < costs[worst];
      const auto contracted =
          outside ? combine(centroid, reflected, 0.5) : combine(centroid, simplex[worst], 0.5);
      const double contracted_cost = cost(contracted);
      if (contracted_cost < (outside ? reflected_cost : costs[worst])) {
        simplex[worst] = contracted;
        costs[worst] = contracted_cost;
      } else {
        for (std::size_t i = 1; i < simplex.size(); ++i) {
          simplex[i] = combine(simplex[0], simplex[i], 0.5);
          costs[i] = cost(simplex[i]);
        }
      }
    }
  }

  std::size_t best = 0;
  for (std::size_t i = 1; i < costs.size(); ++i) {
    if (costs[i] < costs[best]) {
      best = i;
    }
  }
  return std::make_pair(simplex[best], costs[best]);
}

std::vector<std::vector<double>> starts_for_config(const CalibrationConfig& config) {
  const bool lock = config.optimizer.lock_cam_d;
  const bool lock_zrx = config.optimizer.lock_z_rx;
  const bool enable_xrx = config.optimizer.enable_x_rx;
  const double xrx_default = config.imu_xrx_seed.value_or(0.0);
  const double zrx_default = config.imu_zrx_seed.value_or(0.0);

  std::vector<std::vector<double>> starts = lock ? kStarts4 : kStarts5;
  for (auto& start : starts) {
    if (!lock_zrx) {
      const std::size_t zrx_idx = lock ? 3 : 4;
      if (zrx_idx < start.size()) {
        start[zrx_idx] = zrx_default;
      }
    } else {
      const std::size_t zrx_idx = lock ? 3 : 4;
      if (zrx_idx < start.size()) {
        start.erase(start.begin() + static_cast<std::ptrdiff_t>(zrx_idx));
      }
    }
    if (enable_xrx) {
      start.push_back(xrx_default);
    }
  }

  if (config.imu_xrz_seed.has_value()) {
    auto imu_start = lock ? kStarts4[0] : kStarts5[0];
    const std::size_t xrz_idx = lock ? 2 : 3;
    imu_start[xrz_idx] = *config.imu_xrz_seed;
    if (lock_zrx) {
      const std::size_t zrx_idx = lock ? 3 : 4;
      if (zrx_idx < imu_start.size()) {
        imu_start.erase(imu_start.begin() + static_cast<std::ptrdiff_t>(zrx_idx));
      }
    }
    if (enable_xrx) {
      imu_start.push_back(xrx_default);
    }
    starts.push_back(std::move(imu_start));
  }
  return starts;
}

} // namespace

std::vector<std::pair<double, double>> active_bounds(bool enable_x_rx, bool lock_cam_d,
                                                     bool lock_z_rx) {
  Bounds bounds;
  if (lock_cam_d) {
    bounds.assign(std::begin(kBounds5) + 1, std::end(kBounds5));
  } else {
    bounds.assign(std::begin(kBounds5), std::end(kBounds5));
  }
  if (lock_z_rx) {
    bounds.pop_back();
  }
  if (enable_x_rx) {
    bounds.push_back(kXRxBound);
  }
  return bounds;
}

OptParams params_from_vec(const std::vector<double>& p) {
  if (p.size() < 5) {
    throw std::invalid_argument("need at least 5 params");
  }
  return {.x_ty = p[2],
          .intersect = p[1],
          .cam_d = p[0],
          .x_rz = p[3],
          .z_rx = p[4],
          .z_rz = std::nullopt,
          .x_rx = p.size() > 5 ? std::optional<double>(p[5]) : std::nullopt};
}

OptParams params_from_vec_no_zrx(const std::vector<double>& p) {
  if (p.size() < 4) {
    throw std::invalid_argument("need at least 4 params");
  }
  return {.x_ty = p[2],
          .intersect = p[1],
          .cam_d = p[0],
          .x_rz = p[3],
          .z_rx = 0.0,
          .z_rz = std::nullopt,
          .x_rx = p.size() > 4 ? std::optional<double>(p[4]) : std::nullopt};
}

OptParams params_from_vec_locked(const std::vector<double>& p) {
  if (p.size() < 4) {
    throw std::invalid_argument("need at least 4 params");
  }
  const double intersect = p[0];
  return {.x_ty = p[1],
          .intersect = intersect,
          .cam_d = 0.5 * (1.0 - intersect),
          .x_rz = p[2],
          .z_rx = p[3],
          .z_rz = std::nullopt,
          .x_rx = p.size() > 4 ? std::optional<double>(p[4]) : std::nullopt};
}

OptParams params_from_vec_locked_no_zrx(const std::vector<double>& p) {
  if (p.size() < 3) {
    throw std::invalid_argument("need at least 3 params");
  }
  const double intersect = p[0];
  return {.x_ty = p[1],
          .intersect = intersect,
          .cam_d = 0.5 * (1.0 - intersect),
          .x_rz = p[2],
          .z_rx = 0.0,
          .z_rz = std::nullopt,
          .x_rx = p.size() > 3 ? std::optional<double>(p[3]) : std::nullopt};
}

double bounds_penalty(const std::vector<double>& p,
                      const std::vector<std::pair<double, double>>& bounds) {
  double penalty = 0.0;
  for (std::size_t i = 0; i < p.size() && i < bounds.size(); ++i) {
    const auto [lo, hi] = bounds[i];
    if (p[i] < lo) {
      const double d = lo - p[i];
      penalty += kPenaltyScale * d * d;
    } else if (p[i] > hi) {
      const double d = p[i] - hi;
      penalty += kPenaltyScale * d * d;
    }
  }
  return penalty;
}

std::vector<std::vector<double>>
build_simplex(const std::vector<double>& start,
              const std::vector<std::pair<double, double>>& bounds) {
  const std::size_t n = start.size();
  if (bounds.size() < n) {
    throw std::invalid_argument("bounds must cover every simplex dimension");
  }
  std::vector<std::vector<double>> vertices;
  vertices.reserve(n + 1);
  vertices.push_back(start);
  for (std::size_t i = 0; i < n; ++i) {
    auto vertex = start;
    const double range = bounds[i].second - bounds[i].first;
    const double delta = kSimplexPerturbation * range;
    if (vertex[i] + delta <= bounds[i].second) {
      vertex[i] += delta;
    } else {
      vertex[i] -= delta;
    }
    vertices.push_back(std::move(vertex));
  }
  return vertices;
}

OptimizeResult NelderMeadOptimizer::optimize(const std::vector<MatchedPoint>& points,
                                             const CalibrationConfig& config) const {
  const bool lock = config.optimizer.lock_cam_d;
  const bool lock_zrx = config.optimizer.lock_z_rx;
  const bool enable_xrx = config.optimizer.enable_x_rx;
  const auto bounds = active_bounds(enable_xrx, lock, lock_zrx);
  const CalibrationCost cost{.points = points,
                             .sigma = config.optimizer.seam_sigma,
                             .bounds = bounds,
                             .lock_cam_d = lock,
                             .lock_z_rx = lock_zrx,
                             .trim_fraction = config.optimizer.trim_fraction};

  std::optional<std::pair<std::vector<double>, double>> best;
  for (const auto& start : starts_for_config(config)) {
    const auto result = run_nelder_mead(cost, start, config.optimizer.max_iters);
    if (result.has_value() && (!best.has_value() || result->second < best->second)) {
      best = result;
    }
  }
  if (!best.has_value()) {
    throw std::runtime_error("optimizer failed");
  }

  const auto params = [&]() {
    if (lock && lock_zrx) {
      return params_from_vec_locked_no_zrx(best->first);
    }
    if (lock) {
      return params_from_vec_locked(best->first);
    }
    if (lock_zrx) {
      return params_from_vec_no_zrx(best->first);
    }
    return params_from_vec(best->first);
  }();

  return {.layout = {.camera_axis_offset = params.cam_d,
                     .intersect = params.intersect,
                     .x_ty = params.x_ty,
                     .x_rz = params.x_rz,
                     .z_rx = params.z_rx,
                     .x_rx = enable_xrx ? params.x_rx.value_or(0.0) : 0.0,
                     .z_rz = 0.0},
          .residual_error = best->second};
}

OptimizeResult optimize(const std::vector<MatchedPoint>& points, const CalibrationConfig& config) {
  return NelderMeadOptimizer().optimize(points, config);
}

} // namespace reco::calibrate
