#include "reco/detect/detectors.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size > 16 * 1024) {
    return 0;
  }
  (void)reco::detect::parse_names_dict_string(
      std::string_view(reinterpret_cast<const char*>(data), size));
  return 0;
}

#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
int main() {
  const std::uint8_t smoke[] = {'{', '0', ':', ' ', '\'', 'b', 'a', 'l', 'l', '\'', '}'};
  return LLVMFuzzerTestOneInput(smoke, sizeof(smoke));
}
#endif
