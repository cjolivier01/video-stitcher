#pragma once

#include <optional>
#include <string_view>

namespace reco::autocam {

enum class TrackingMode {
  Field,
  Ball,
  Sweep,
};

[[nodiscard]] std::optional<TrackingMode> parse_tracking_mode(std::string_view value);
[[nodiscard]] std::string_view tracking_mode_name(TrackingMode mode);
[[nodiscard]] TrackingMode default_tracking_mode();

} // namespace reco::autocam
