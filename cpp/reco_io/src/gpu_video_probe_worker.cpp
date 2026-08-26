#include "gpu_video_probe_internal.hpp"
#include "gpu_video_probe_protocol.hpp"

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
  const auto read_exact = [](char* destination, std::size_t size) {
    std::cin.read(destination, static_cast<std::streamsize>(size));
    if (std::cin.gcount() != static_cast<std::streamsize>(size)) {
      throw GpuVideoProbeError("video probe worker request has a truncated IPC frame");
    }
  };
  ProbeIpcFrameHeader header{};
  read_exact(header.data(), header.size());
  std::string request(decode_probe_ipc_frame_header(header), '\0');
  read_exact(request.data(), request.size());
  return request;
}

void write_response(std::string_view response) {
  const auto header = encode_probe_ipc_frame_header(response.size());
  std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::cout.write(response.data(), static_cast<std::streamsize>(response.size()));
  std::cout.flush();
  if (!std::cout) {
    throw GpuVideoProbeError("failed to write video probe worker response");
  }
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
  try {
    write_response(response);
    return 0;
  } catch (...) {
    return 2;
  }
}

} // namespace reco::io::detail
