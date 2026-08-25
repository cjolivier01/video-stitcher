#include "reco/core/rig_correction.hpp"

#include <cmath>

namespace reco::core {

float render_pitch(float user_yaw, float user_pitch, float rig_tilt) {
  if (std::abs(rig_tilt) < 1.0e-6F) {
    return user_pitch;
  }
  return user_pitch - std::atan(std::cos(user_yaw) * std::tan(rig_tilt));
}

} // namespace reco::core
