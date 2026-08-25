#include "reco/autocam/roi_filter.hpp"

#include "reco/core/projection.hpp"

namespace reco::autocam {

bool passes_roi_anchor(RoiAnchor anchor, const Detection& detection,
                       const std::vector<std::array<double, 2>>& polygon) {
  const double cx = detection.center_x;
  const double cy = detection.center_y;
  const double half_h = static_cast<double>(detection.height) * 0.5;
  const double quarter_h = static_cast<double>(detection.height) * 0.25;
  switch (anchor) {
  case RoiAnchor::Center:
    return reco::core::point_in_polygon({cx, cy}, polygon);
  case RoiAnchor::Bottom:
    return reco::core::point_in_polygon({cx, cy + half_h}, polygon) &&
           reco::core::point_in_polygon({cx, cy + quarter_h}, polygon);
  case RoiAnchor::Top:
    return reco::core::point_in_polygon({cx, cy - half_h}, polygon) &&
           reco::core::point_in_polygon({cx, cy - quarter_h}, polygon);
  }
  return false;
}

std::vector<Detection>
filter_by_roi(const std::vector<Detection>& detections, const reco::core::FieldRoi& roi,
              const std::unordered_map<std::uint16_t, RoiAnchor>& class_anchors,
              RoiAnchor default_anchor) {
  std::vector<Detection> out;
  out.reserve(detections.size());
  for (const auto& detection : detections) {
    const auto& polygon = detection.camera == reco::core::CameraId::Left ? roi.left : roi.right;
    if (polygon.size() < 3) {
      out.push_back(detection);
      continue;
    }
    const auto it = class_anchors.find(detection.class_id);
    const RoiAnchor anchor = it == class_anchors.end() ? default_anchor : it->second;
    if (passes_roi_anchor(anchor, detection, polygon)) {
      out.push_back(detection);
    }
  }
  return out;
}

} // namespace reco::autocam
