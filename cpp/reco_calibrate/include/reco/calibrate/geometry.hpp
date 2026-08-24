#pragma once

#include "reco/calibrate/types.hpp"

#include <array>
#include <optional>
#include <vector>

namespace reco::calibrate {

inline constexpr double kPlaneWidth = 1.0;

struct Vec3d {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct OptParams {
  double x_ty = 0.0;
  double intersect = 0.0;
  double cam_d = 0.0;
  double x_rz = 0.0;
  double z_rx = 0.0;
  std::optional<double> z_rz;
  std::optional<double> x_rx;

  [[nodiscard]] static OptParams from_5param(const std::array<double, 5>& values);
  [[nodiscard]] static OptParams from_6param(const std::array<double, 6>& values);
  [[nodiscard]] std::array<double, 5> to_5param() const;
  [[nodiscard]] std::array<double, 6> to_6param() const;
};

struct TransformedPoints {
  std::vector<Vec3d> x_plane;
  std::vector<Vec3d> z_plane;
};

struct SeamWeightConfig {
  double sigma_x = 0.08;
  double sigma_y = 0.08;
  double y_center = -0.05;

  [[nodiscard]] static SeamWeightConfig from_sigma(double sigma);
};

[[nodiscard]] Vec3d to_3d_x_plane(std::array<double, 2> point);
[[nodiscard]] Vec3d to_3d_z_plane(std::array<double, 2> point);
[[nodiscard]] std::array<std::array<double, 3>, 3> rotation_matrix(double rx, double ry,
                                                                  double rz);
[[nodiscard]] TransformedPoints apply_transformations(const std::vector<MatchedPoint>& points,
                                                      const OptParams& params);
[[nodiscard]] double reprojection_error(const std::vector<MatchedPoint>& points,
                                        const OptParams& params);
[[nodiscard]] std::vector<double> per_point_reprojection_error(const std::vector<MatchedPoint>& points,
                                                               const OptParams& params);
[[nodiscard]] double trimmed_reprojection_error(const std::vector<MatchedPoint>& points,
                                                const OptParams& params, double trim_fraction);
[[nodiscard]] double angular_error(const std::vector<MatchedPoint>& points, const OptParams& params);
[[nodiscard]] std::vector<double> per_point_seam_weighted_errors(
    const std::vector<MatchedPoint>& points, const OptParams& params, double sigma);
[[nodiscard]] double seam_weighted_reprojection_error(const std::vector<MatchedPoint>& points,
                                                      const OptParams& params, double sigma);
[[nodiscard]] double trimmed_seam_weighted_reprojection_error(
    const std::vector<MatchedPoint>& points, const OptParams& params, double sigma,
    double trim_fraction);
[[nodiscard]] std::array<double, 2> normalize_to_plane(double px, double py, std::uint32_t img_w,
                                                       std::uint32_t img_h);

} // namespace reco::calibrate
