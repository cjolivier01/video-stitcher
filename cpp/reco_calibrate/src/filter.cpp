#include "reco/calibrate/filter.hpp"

#include <cmath>
#include <stdexcept>

namespace reco::calibrate {

std::vector<RawMatch> spatial_filter(const std::vector<RawMatch>& matches,
                                     const std::vector<KeyPoint>& kp_left,
                                     const std::vector<KeyPoint>& kp_right,
                                     std::uint32_t img_w_left, std::uint32_t img_h_left,
                                     std::uint32_t img_w_right, std::uint32_t img_h_right,
                                     const CalibrationConfig& config) {
  const double x_thresh_left = config.matching.spatial_x_threshold * static_cast<double>(img_w_left);
  const double x_thresh_right =
      config.matching.spatial_x_threshold * static_cast<double>(img_w_right);
  const double x_inner_left =
      (1.0 - config.matching.spatial_x_inner) * static_cast<double>(img_w_left);
  const double x_inner_right = config.matching.spatial_x_inner * static_cast<double>(img_w_right);
  const double y_low_left = config.matching.spatial_y_low * static_cast<double>(img_h_left);
  const double y_high_left = config.matching.spatial_y_high * static_cast<double>(img_h_left);
  const double y_low_right = config.matching.spatial_y_low * static_cast<double>(img_h_right);
  const double y_high_right = config.matching.spatial_y_high * static_cast<double>(img_h_right);
  const double avg_h = (static_cast<double>(img_h_left) + static_cast<double>(img_h_right)) / 2.0;
  const double max_y_disp = config.matching.max_y_disparity * avg_h;

  std::vector<RawMatch> out;
  for (const auto& match : matches) {
    if (match.left_idx >= kp_left.size()) {
      throw std::out_of_range("left keypoint index out of range");
    }
    if (match.right_idx >= kp_right.size()) {
      throw std::out_of_range("right keypoint index out of range");
    }
    const auto& left = kp_left[match.left_idx];
    const auto& right = kp_right[match.right_idx];
    const double left_x = left.x;
    const double left_y = left.y;
    const double right_x = right.x;
    const double right_y = right.y;
    const double y_disp = std::abs(left_y - right_y);
    if (left_x >= x_thresh_left && left_x <= x_inner_left && right_x >= x_inner_right &&
        right_x <= x_thresh_right && left_y >= y_low_left && left_y <= y_high_left &&
        right_y >= y_low_right && right_y <= y_high_right && y_disp <= max_y_disp) {
      out.push_back(match);
    }
  }
  return out;
}

} // namespace reco::calibrate
