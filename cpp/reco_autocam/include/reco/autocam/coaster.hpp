#pragma once

#include <cstdint>

namespace reco::autocam {

enum class CoastStatus {
  Tracking,
  Coasting,
  Lost,
};

class Coaster {
public:
  explicit Coaster(std::uint32_t max_coast_frames);

  [[nodiscard]] CoastStatus accept_fresh();
  [[nodiscard]] CoastStatus step_without_fresh();
  [[nodiscard]] std::uint32_t frames_coasting() const { return frames_coasting_; }

private:
  std::uint32_t max_coast_frames_ = 0;
  std::uint32_t frames_coasting_ = 0;
  bool ever_tracked_ = false;
};

} // namespace reco::autocam
