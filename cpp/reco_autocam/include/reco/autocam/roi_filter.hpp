#pragma once

#include "reco/core/calibration.hpp"
#include "reco/core/pipeline_event.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace reco::autocam {

struct Detection {
  reco::core::CameraId camera = reco::core::CameraId::Left;
  std::uint16_t class_id = 0;
  float confidence = 0.0F;
  float center_x = 0.0F;
  float center_y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
};

enum class RoiAnchor {
  Center,
  Bottom,
  Top,
};

[[nodiscard]] bool passes_roi_anchor(RoiAnchor anchor, const Detection& detection,
                                     const std::vector<std::array<double, 2>>& polygon);

[[nodiscard]] std::vector<Detection>
filter_by_roi(const std::vector<Detection>& detections, const reco::core::FieldRoi& roi,
              const std::unordered_map<std::uint16_t, RoiAnchor>& class_anchors = {},
              RoiAnchor default_anchor = RoiAnchor::Center);

} // namespace reco::autocam
