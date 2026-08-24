#pragma once

#include <string>
#include <vector>

namespace reco::detect {

struct AiProbeResult {
  std::vector<std::string> providers;
  bool can_run_on_gpu_frames = false;
  std::vector<std::string> errors;

  [[nodiscard]] std::string best_provider() const;
  [[nodiscard]] bool is_available() const;
};

} // namespace reco::detect
