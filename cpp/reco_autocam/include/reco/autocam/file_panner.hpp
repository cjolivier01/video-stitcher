#pragma once

#include "reco/autocam/sweep_panner.hpp"
#include "reco/core/viewport_position.hpp"

#include <cstdint>
#include <filesystem>
#include <unordered_map>

namespace reco::autocam {

class FilePanner {
public:
  static FilePanner from_csv(const std::filesystem::path& path);

  [[nodiscard]] reco::core::ViewportPosition decide(const WorldState& world,
                                                    const PanContext& context);

  [[nodiscard]] std::size_t pose_count() const { return poses_.size(); }

private:
  std::unordered_map<std::uint64_t, reco::core::ViewportPosition> poses_;
  reco::core::ViewportPosition last_;
};

} // namespace reco::autocam
