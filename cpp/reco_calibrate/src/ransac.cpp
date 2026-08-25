#include "reco/calibrate/ransac.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>

namespace reco::calibrate {
namespace {

template <std::size_t N> struct EigenResult {
  std::array<double, N> values{};
  std::array<std::array<double, N>, N> vectors{};
};

template <std::size_t N>
EigenResult<N> jacobi_eigen_symmetric(std::array<std::array<double, N>, N> a) {
  std::array<std::array<double, N>, N> v{};
  for (std::size_t i = 0; i < N; ++i) {
    v[i][i] = 1.0;
  }

  for (std::size_t iter = 0; iter < 80 * N * N; ++iter) {
    std::size_t p = 0;
    std::size_t q = 1;
    double max_offdiag = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = i + 1; j < N; ++j) {
        const double value = std::abs(a[i][j]);
        if (value > max_offdiag) {
          max_offdiag = value;
          p = i;
          q = j;
        }
      }
    }
    if (max_offdiag < 1.0e-12) {
      break;
    }

    const double app = a[p][p];
    const double aqq = a[q][q];
    const double apq = a[p][q];
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = std::copysign(1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)), tau);
    const double c = 1.0 / std::sqrt(1.0 + t * t);
    const double s = t * c;

    for (std::size_t k = 0; k < N; ++k) {
      if (k != p && k != q) {
        const double akp = a[k][p];
        const double akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[p][k] = a[k][p];
        a[k][q] = s * akp + c * akq;
        a[q][k] = a[k][q];
      }
    }
    a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    for (std::size_t k = 0; k < N; ++k) {
      const double vkp = v[k][p];
      const double vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq;
      v[k][q] = s * vkp + c * vkq;
    }
  }

  EigenResult<N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result.values[i] = a[i][i];
    for (std::size_t j = 0; j < N; ++j) {
      result.vectors[j][i] = v[j][i];
    }
  }
  return result;
}

Matrix3d transpose(const Matrix3d& m) {
  Matrix3d out{};
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c) {
      out[r][c] = m[c][r];
    }
  }
  return out;
}

Matrix3d multiply(const Matrix3d& a, const Matrix3d& b) {
  Matrix3d out{};
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c) {
      for (std::size_t k = 0; k < 3; ++k) {
        out[r][c] += a[r][k] * b[k][c];
      }
    }
  }
  return out;
}

double frobenius_norm(const Matrix3d& m) {
  double sum = 0.0;
  for (const auto& row : m) {
    for (const double value : row) {
      sum += value * value;
    }
  }
  return std::sqrt(sum);
}

Matrix3d normalize_frobenius(const Matrix3d& m) {
  const double norm = frobenius_norm(m);
  if (norm < 1.0e-15) {
    return {};
  }
  Matrix3d out = m;
  for (auto& row : out) {
    for (double& value : row) {
      value /= norm;
    }
  }
  return out;
}

std::array<double, 3> mat_vec(const Matrix3d& m, const std::array<double, 3>& v) {
  return {m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
          m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
          m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]};
}

Matrix3d enforce_rank2(const Matrix3d& f) {
  const auto ft = transpose(f);
  const auto ftf = multiply(ft, f);
  const auto eig = jacobi_eigen_symmetric<3>(ftf);
  std::size_t min_idx = 0;
  for (std::size_t i = 1; i < 3; ++i) {
    if (eig.values[i] < eig.values[min_idx]) {
      min_idx = i;
    }
  }
  const double sigma = std::sqrt(std::max(0.0, eig.values[min_idx]));
  if (sigma < 1.0e-15) {
    return f;
  }

  const std::array<double, 3> v_min{eig.vectors[0][min_idx], eig.vectors[1][min_idx],
                                    eig.vectors[2][min_idx]};
  auto u_min = mat_vec(f, v_min);
  for (double& value : u_min) {
    value /= sigma;
  }

  Matrix3d out = f;
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c) {
      out[r][c] -= sigma * u_min[r] * v_min[c];
    }
  }
  return out;
}

std::optional<Matrix3d> estimate_fundamental_8pt(const std::vector<Point2d>& pts1,
                                                 const std::vector<Point2d>& pts2,
                                                 const std::vector<std::size_t>& sample) {
  const std::size_t n = sample.size();
  if (n < 8) {
    return std::nullopt;
  }

  double cx1 = 0.0;
  double cy1 = 0.0;
  double cx2 = 0.0;
  double cy2 = 0.0;
  for (const auto i : sample) {
    cx1 += pts1[i][0];
    cy1 += pts1[i][1];
    cx2 += pts2[i][0];
    cy2 += pts2[i][1];
  }
  const double inv_n = 1.0 / static_cast<double>(n);
  cx1 *= inv_n;
  cy1 *= inv_n;
  cx2 *= inv_n;
  cy2 *= inv_n;

  double d1 = 0.0;
  double d2 = 0.0;
  for (const auto i : sample) {
    const double dx1 = pts1[i][0] - cx1;
    const double dy1 = pts1[i][1] - cy1;
    const double dx2 = pts2[i][0] - cx2;
    const double dy2 = pts2[i][1] - cy2;
    d1 += std::sqrt(dx1 * dx1 + dy1 * dy1);
    d2 += std::sqrt(dx2 * dx2 + dy2 * dy2);
  }
  d1 *= inv_n;
  d2 *= inv_n;
  if (d1 < 1.0e-10 || d2 < 1.0e-10) {
    return std::nullopt;
  }

  const double s1 = std::sqrt(2.0) / d1;
  const double s2 = std::sqrt(2.0) / d2;
  const Matrix3d t1{{{s1, 0.0, -s1 * cx1}, {0.0, s1, -s1 * cy1}, {0.0, 0.0, 1.0}}};
  const Matrix3d t2{{{s2, 0.0, -s2 * cx2}, {0.0, s2, -s2 * cy2}, {0.0, 0.0, 1.0}}};

  std::array<std::array<double, 9>, 9> ata{};
  for (const auto i : sample) {
    const double x1 = (pts1[i][0] - cx1) * s1;
    const double y1 = (pts1[i][1] - cy1) * s1;
    const double x2 = (pts2[i][0] - cx2) * s2;
    const double y2 = (pts2[i][1] - cy2) * s2;
    const std::array<double, 9> row{x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1,
                                    y2,      x1,      y1, 1.0};
    for (std::size_t r = 0; r < 9; ++r) {
      for (std::size_t c = 0; c <= r; ++c) {
        ata[r][c] += row[r] * row[c];
      }
    }
  }
  for (std::size_t r = 0; r < 9; ++r) {
    for (std::size_t c = r + 1; c < 9; ++c) {
      ata[r][c] = ata[c][r];
    }
  }

  const auto eig = jacobi_eigen_symmetric<9>(ata);
  std::size_t min_idx = 0;
  for (std::size_t i = 1; i < 9; ++i) {
    if (eig.values[i] < eig.values[min_idx]) {
      min_idx = i;
    }
  }

  Matrix3d f_norm{{{eig.vectors[0][min_idx], eig.vectors[1][min_idx], eig.vectors[2][min_idx]},
                   {eig.vectors[3][min_idx], eig.vectors[4][min_idx], eig.vectors[5][min_idx]},
                   {eig.vectors[6][min_idx], eig.vectors[7][min_idx], eig.vectors[8][min_idx]}}};
  for (const auto& row : f_norm) {
    for (const double value : row) {
      if (std::isnan(value)) {
        return std::nullopt;
      }
    }
  }

  const Matrix3d f = multiply(multiply(transpose(t2), f_norm), t1);
  const Matrix3d rank2 = enforce_rank2(f);
  const double norm = frobenius_norm(rank2);
  if (norm < 1.0e-15) {
    return std::nullopt;
  }
  return normalize_frobenius(rank2);
}

std::vector<std::size_t> sample_indices(const std::vector<std::size_t>& indices,
                                        std::mt19937_64& rng) {
  auto shuffled = indices;
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  shuffled.resize(8);
  return shuffled;
}

} // namespace

double sampson_error(const Matrix3d& f, const Point2d& p1, const Point2d& p2) {
  const std::array<double, 3> x1{p1[0], p1[1], 1.0};
  const std::array<double, 3> x2{p2[0], p2[1], 1.0};
  const auto fx1 = mat_vec(f, x1);
  const auto ftx2 = mat_vec(transpose(f), x2);
  const double x2tfx1 = x2[0] * fx1[0] + x2[1] * fx1[1] + x2[2] * fx1[2];
  const double numerator = x2tfx1 * x2tfx1;
  const double denominator = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] +
                             ftx2[1] * ftx2[1];
  if (denominator < 1.0e-15) {
    return std::numeric_limits<double>::max();
  }
  return numerator / denominator;
}

std::vector<std::size_t> ransac_fundamental(const std::vector<Point2d>& pts1,
                                            const std::vector<Point2d>& pts2, double threshold,
                                            std::size_t max_iterations) {
  const std::size_t n = pts1.size();
  if (n != pts2.size()) {
    throw std::invalid_argument("point arrays must have equal length");
  }
  if (n < 8) {
    throw std::invalid_argument("need at least 8 point pairs");
  }

  const double threshold_sq = threshold * threshold;
  const std::size_t max_iters = max_iterations == 0 ? 2000 : max_iterations;
  std::mt19937_64 rng(42);
  std::vector<std::size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<std::size_t> best_inliers;
  std::size_t best_score = 0;
  for (std::size_t iter = 0; iter < max_iters; ++iter) {
    const auto sample = sample_indices(indices, rng);
    const auto f = estimate_fundamental_8pt(pts1, pts2, sample);
    if (!f.has_value()) {
      continue;
    }

    std::vector<std::size_t> inliers;
    for (std::size_t i = 0; i < n; ++i) {
      if (sampson_error(*f, pts1[i], pts2[i]) < threshold_sq) {
        inliers.push_back(i);
      }
    }
    if (inliers.size() > best_score) {
      best_score = inliers.size();
      best_inliers = std::move(inliers);
      if (best_score * 5 > n * 4) {
        break;
      }
    }
  }

  if (best_inliers.empty()) {
    throw std::runtime_error("RANSAC found no inliers");
  }

  if (const auto refined = estimate_fundamental_8pt(pts1, pts2, best_inliers);
      refined.has_value()) {
    std::vector<std::size_t> refined_inliers;
    for (std::size_t i = 0; i < n; ++i) {
      if (sampson_error(*refined, pts1[i], pts2[i]) < threshold_sq) {
        refined_inliers.push_back(i);
      }
    }
    if (refined_inliers.size() >= best_inliers.size()) {
      best_inliers = std::move(refined_inliers);
    }
  }

  return best_inliers;
}

} // namespace reco::calibrate
