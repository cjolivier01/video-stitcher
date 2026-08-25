#include "reco/core/source.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size > 4096) {
    return 0;
  }
  std::string text(reinterpret_cast<const char*>(data), size);
  (void)reco::core::validate_input_path(std::filesystem::path(text));
  return 0;
}

#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
int main() {
  const std::uint8_t smoke[] = {'m', 'i', 's', 's', 'i', 'n', 'g'};
  return LLVMFuzzerTestOneInput(smoke, sizeof(smoke));
}
#endif
