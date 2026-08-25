#pragma once

#include "reco/calibrate/features.hpp"
#include "reco/calibrate/types.hpp"

#include <cstdint>
#include <vector>

namespace reco::calibrate {

[[nodiscard]] std::vector<RawMatch> spatial_filter(const std::vector<RawMatch>& matches,
                                                   const std::vector<KeyPoint>& kp_left,
                                                   const std::vector<KeyPoint>& kp_right,
                                                   std::uint32_t img_w_left,
                                                   std::uint32_t img_h_left,
                                                   std::uint32_t img_w_right,
                                                   std::uint32_t img_h_right,
                                                   const CalibrationConfig& config);

} // namespace reco::calibrate
