#include "reco/detect/probe.hpp"

#include "reco/detect/coreml_session.hpp"
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
  } else {
    result.errors.push_back("ORT provider registration probe is not ported to C++ yet");
  }

  const auto coreml = probe_coreml_runtime();
  if (coreml.available) {
    result.errors.push_back("CoreML prediction bridge is not ported to C++ yet");
  } else if (!coreml.error.empty()) {
    result.errors.push_back(coreml.error);
  }
  return result;
}

} // namespace reco::detect
