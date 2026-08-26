#include "gpu_video_probe_internal.hpp"
#include "gpu_video_probe_protocol.hpp"

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace reco::io::detail {
namespace {

std::string read_request() {
  std::string request;
  std::array<char, 4096> buffer{};
  while (std::cin) {
    std::cin.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = std::cin.gcount();
    if (count <= 0) {
      break;
    }
    const auto size = static_cast<std::size_t>(count);
    if (request.size() > kMaximumProbeIpcBytes - size) {
      throw GpuVideoProbeError("video probe worker request exceeds the IPC size limit");
    }
    request.append(buffer.data(), size);
  }
  if (std::cin.bad()) {
    throw GpuVideoProbeError("failed to read video probe worker request");
  }
  return request;
}

void start_parent_liveness_watch(int descriptor) {
#if defined(_WIN32)
  (void)descriptor;
#else
  std::thread([descriptor] {
    std::array<char, 1> ignored{};
    while (true) {
      const auto result = ::read(descriptor, ignored.data(), ignored.size());
      if (result > 0 || (result < 0 && errno == EINTR)) {
        continue;
      }
      (void)::kill(0, SIGKILL);
      std::_Exit(3);
    }
  }).detach();
#endif
}

} // namespace

int run_gpu_video_probe_worker(int parent_liveness_descriptor) {
  std::string response;
  try {
    const auto payload = read_request();
    start_parent_liveness_watch(parent_liveness_descriptor);
    const auto request = decode_probe_request(payload);
    response = encode_probe_success(probe_gpu_video_in_process(request.config, request.timeout_ns));
  } catch (const std::invalid_argument& error) {
    response = encode_probe_failure("invalid_argument", error.what());
  } catch (const GpuVideoProbeError& error) {
    response = encode_probe_failure("probe_error", error.what());
  } catch (const std::exception& error) {
    response = encode_probe_failure("worker_error", error.what());
  } catch (...) {
    response = encode_probe_failure("worker_error", "unknown video probe worker failure");
  }
  std::cout << response;
  std::cout.flush();
  return std::cout ? 0 : 2;
}

} // namespace reco::io::detail
