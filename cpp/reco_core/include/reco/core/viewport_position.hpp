#pragma once

#include <optional>

namespace reco::core {

struct ViewportPosition {
  float yaw = 0.0F;
  float pitch = 0.0F;
  std::optional<float> fov_degrees = 75.0F;

  friend bool operator==(const ViewportPosition&, const ViewportPosition&) = default;
};

} // namespace reco::core

