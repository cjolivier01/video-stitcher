#pragma once

#include "reco/core/pipeline_event.hpp"

#include <cstdint>
#include <vector>

namespace reco::autocam {

class ClassProvider {
public:
  explicit ClassProvider(std::uint16_t class_id);

  [[nodiscard]] std::vector<reco::core::TrackedEntity>
  update(const std::vector<reco::core::MappedDetection>& detections, double timestamp_ms);

  [[nodiscard]] std::uint16_t class_id() const { return class_id_; }

private:
  std::uint16_t class_id_ = 0;
};

} // namespace reco::autocam
