#include "reco/core/calibration.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size > reco::core::kMaxCalibrationFileSize) {
    return 0;
  }
  (void)reco::core::parse_match_calibration_json(
      std::string_view(reinterpret_cast<const char*>(data), size));
  return 0;
}

#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
int main() {
  const std::uint8_t smoke[] = {'{', '}'};
  return LLVMFuzzerTestOneInput(smoke, sizeof(smoke));
}
#endif
