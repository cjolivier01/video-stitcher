#include "reco/autocam/tracking_mode.hpp"

#include <algorithm>
#include <string>

namespace reco::autocam {
namespace {

std::string lowercase(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

} // namespace

std::optional<TrackingMode> parse_tracking_mode(std::string_view value) {
  const auto s = lowercase(value);
  if (s == "field") {
    return TrackingMode::Field;
  }
  if (s == "ball") {
    return TrackingMode::Ball;
  }
  if (s == "sweep") {
    return TrackingMode::Sweep;
  }
  return std::nullopt;
}

std::string_view tracking_mode_name(TrackingMode mode) {
  switch (mode) {
  case TrackingMode::Field:
    return "field";
  case TrackingMode::Ball:
    return "ball";
  case TrackingMode::Sweep:
    return "sweep";
  }
  return "field";
}

TrackingMode default_tracking_mode() { return TrackingMode::Field; }

} // namespace reco::autocam
