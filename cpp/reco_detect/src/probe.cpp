#include "reco/detect/probe.hpp"

namespace reco::detect {

std::string AiProbeResult::best_provider() const {
  return providers.empty() ? "unavailable" : providers.front();
}

bool AiProbeResult::is_available() const { return !providers.empty(); }

} // namespace reco::detect
