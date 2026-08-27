#include "../src/gpu_video_probe_protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const auto payload = std::string_view(reinterpret_cast<const char*>(data), size);
  try {
    (void)reco::io::detail::decode_probe_response(payload);
  } catch (const reco::io::GpuVideoProbeError&) {
  } catch (const std::invalid_argument&) {
  }
  return 0;
}

#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
int main() {
  const std::uint8_t smoke[] = {0xA0};
  return LLVMFuzzerTestOneInput(smoke, sizeof(smoke));
}
#endif
