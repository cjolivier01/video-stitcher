#include "reco/autocam/coaster.hpp"

namespace reco::autocam {

Coaster::Coaster(std::uint32_t max_coast_frames) : max_coast_frames_(max_coast_frames) {}

CoastStatus Coaster::accept_fresh() {
  frames_coasting_ = 0;
  ever_tracked_ = true;
  return CoastStatus::Tracking;
}

CoastStatus Coaster::step_without_fresh() {
  if (!ever_tracked_) {
    return CoastStatus::Lost;
  }
  if (frames_coasting_ < max_coast_frames_) {
    ++frames_coasting_;
    return CoastStatus::Coasting;
  }
  frames_coasting_ = max_coast_frames_;
  ever_tracked_ = false;
  return CoastStatus::Lost;
}

} // namespace reco::autocam
