#include "calibration_worker_internal.hpp"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

bool parse_unsigned(const char* value, std::uint64_t* result) {
  if (value == nullptr || result == nullptr) {
    return false;
  }
  const std::string_view text(value);
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), *result);
  return error == std::errc{} && end == text.data() + text.size();
}

void close_descriptor(int descriptor) noexcept {
#if defined(_WIN32)
  (void)::_close(descriptor);
#else
  (void)::close(descriptor);
#endif
}

} // namespace

int main(int argc, char** argv) {
  using namespace reco::calibrate::detail;
  if (argc != 4 || argv == nullptr || argv[1] == nullptr || argv[2] == nullptr) {
    return EXIT_FAILURE;
  }
  std::uint64_t deadline = 0;
  if (!parse_unsigned(argv[3], &deadline)) {
    return EXIT_FAILURE;
  }
  const bool is_worker = std::string_view(argv[1]) == kCalibrationWorkerIpcArgument;
  const auto descriptor = connect_calibration_ipc(argv[2], deadline);
  if (descriptor < 0) {
    return EXIT_FAILURE;
  }
  if (is_worker && !install_calibration_worker_sandbox()) {
    close_descriptor(descriptor);
    return EXIT_FAILURE;
  }
  int result = EXIT_FAILURE;
  if (std::string_view(argv[1]) == kCalibrationGuardianArgument) {
    result = run_calibration_guardian_fd(descriptor, argv[0], deadline);
  } else if (is_worker) {
    result = run_calibration_worker_fd(descriptor, deadline);
  }
  close_descriptor(descriptor);
  return result;
}
