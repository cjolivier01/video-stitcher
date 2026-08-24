#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace reco::calibrate {

using Point2d = std::array<double, 2>;
using Matrix3d = std::array<std::array<double, 3>, 3>;

[[nodiscard]] std::vector<std::size_t> ransac_fundamental(const std::vector<Point2d>& pts1,
                                                          const std::vector<Point2d>& pts2,
                                                          double threshold,
                                                          std::size_t max_iterations);

[[nodiscard]] double sampson_error(const Matrix3d& f, const Point2d& p1, const Point2d& p2);

} // namespace reco::calibrate
