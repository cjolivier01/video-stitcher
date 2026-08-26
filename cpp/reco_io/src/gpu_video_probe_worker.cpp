#include "gpu_video_probe_internal.hpp"
#include "gpu_video_probe_protocol.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
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

void start_parent_liveness_watch(std::uint64_t expected_parent_pid) {
#if defined(_WIN32)
  (void)expected_parent_pid;
#else
  std::thread([expected_parent_pid] {
    while (true) {
      if (static_cast<std::uint64_t>(::getppid()) != expected_parent_pid) {
        (void)::kill(0, SIGKILL);
        std::_Exit(3);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
#endif
}

} // namespace

int run_gpu_video_probe_worker(std::uint64_t expected_parent_pid) {
  std::string response;
  try {
    start_parent_liveness_watch(expected_parent_pid);
    const auto payload = read_request();
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
