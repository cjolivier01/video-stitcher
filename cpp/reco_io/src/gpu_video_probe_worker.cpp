#include "gpu_video_probe_internal.hpp"
#include "gpu_video_probe_protocol.hpp"

#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace reco::io::detail {
namespace {

#if !defined(_WIN32)
[[noreturn]] void kill_guarded_process_group() {
  (void)::kill(0, SIGKILL);
  std::_Exit(3);
}

bool close_unrelated_descriptors() {
#if defined(__linux__)
  if (auto* directory = ::opendir("/proc/self/fd"); directory != nullptr) {
    const auto directory_descriptor = ::dirfd(directory);
    std::vector<int> descriptors;
    while (const auto* entry = ::readdir(directory)) {
      const std::string_view name(entry->d_name);
      int descriptor = -1;
      const auto [end, error] = std::from_chars(name.data(), name.data() + name.size(), descriptor);
      if (error == std::errc{} && end == name.data() + name.size() && descriptor > 2 &&
          descriptor != directory_descriptor) {
        descriptors.push_back(descriptor);
      }
    }
    (void)::closedir(directory);
    for (const int descriptor : descriptors) {
      if (::close(descriptor) != 0 && errno != EINTR && errno != EBADF) {
        return false;
      }
    }
    return true;
  }
#endif
  const auto maximum_descriptor = ::sysconf(_SC_OPEN_MAX);
  if (maximum_descriptor < 0) {
    return false;
  }
  for (long descriptor = 3; descriptor < maximum_descriptor; ++descriptor) {
    if (::close(static_cast<int>(descriptor)) != 0 && errno != EINTR && errno != EBADF) {
      return false;
    }
  }
  return true;
}

int run_process_group_guard() {
  if (!close_unrelated_descriptors()) {
    kill_guarded_process_group();
  }
  constexpr char kReady = 'R';
#if defined(SO_NOSIGPIPE) && !defined(MSG_NOSIGNAL)
  const int suppress_sigpipe = 1;
  if (::setsockopt(STDOUT_FILENO, SOL_SOCKET, SO_NOSIGPIPE, &suppress_sigpipe,
                   sizeof(suppress_sigpipe)) != 0) {
    kill_guarded_process_group();
  }
#endif
  ssize_t written = -1;
  do {
    written = ::send(STDOUT_FILENO, &kReady, 1,
#if defined(MSG_NOSIGNAL)
                     MSG_NOSIGNAL
#else
                     0
#endif
    );
  } while (written < 0 && errno == EINTR);
  if (written != 1) {
    kill_guarded_process_group();
  }

  char value = '\0';
  while (true) {
    const auto received = ::read(STDIN_FILENO, &value, 1);
    if (received > 0 || (received < 0 && errno == EINTR)) {
      continue;
    }
    kill_guarded_process_group();
  }
}
#endif

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

#if !defined(_WIN32)
int run_gpu_video_probe_guard() { return run_process_group_guard(); }
#endif

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
