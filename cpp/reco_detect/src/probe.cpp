#include "reco/detect/probe.hpp"

#include "reco/detect/ort_session.hpp"

namespace reco::detect {

std::string AiProbeResult::best_provider() const {
  return providers.empty() ? "unavailable" : providers.front();
}

bool AiProbeResult::is_available() const { return !providers.empty(); }

AiProbeResult probe_execution_providers() {
  AiProbeResult result;
  const auto ort = probe_ort_runtime();
  if (!ort.available) {
    result.errors.push_back(ort.error);
    return result;
  }

  result.errors.push_back("ORT provider registration probe is not ported to C++ yet");
  return result;
}

} // namespace reco::detect
