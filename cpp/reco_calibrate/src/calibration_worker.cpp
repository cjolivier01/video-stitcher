#include "calibration_worker_internal.hpp"

#include "calibration_worker_protocol.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <istream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/magic.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace reco::calibrate::detail {
namespace {

void read_stream_exact(std::istream& input, char* destination, std::size_t size) {
  input.read(destination, static_cast<std::streamsize>(size));
  if (input.gcount() != static_cast<std::streamsize>(size)) {
    throw CalibrationExecutionError("calibration worker request is truncated");
  }
}

void write_stream_frame(std::ostream& output, std::string_view frame) {
  output.write(frame.data(), static_cast<std::streamsize>(frame.size()));
  output.flush();
  if (!output) {
    throw CalibrationExecutionError("calibration worker could not write its response");
  }
}

using ReadExact = std::function<void(char*, std::size_t)>;
using WriteFrame = std::function<void(std::string_view)>;
using PrepareRequest = std::function<void(GpuCalibrationRequest&)>;

int run_calibration_worker_io(const ReadExact& read_exact, const WriteFrame& write_frame,
                              const PrepareRequest& prepare_request = {}) {
  try {
    CalibrationWorkerFrameHeader header{};
    read_exact(header.data(), header.size());
    const auto decoded = decode_calibration_worker_header(header);
    std::string request(header.data(), header.size());
    const auto payload_offset = request.size();
    request.resize(payload_offset + decoded.payload_size);
    read_exact(request.data() + payload_offset, decoded.payload_size);
    auto decoded_request = decode_calibration_worker_request(request);
    if (prepare_request) {
      prepare_request(decoded_request);
    }
    const auto result =
        run_gpu_calibration_in_process(decoded_request, probe_calibration_backends());
    write_frame(encode_calibration_worker_success(result));
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    try {
      write_frame(encode_calibration_worker_failure(error.what()));
    } catch (const std::exception&) {
      return EXIT_FAILURE;
    }
    return EXIT_FAILURE;
  } catch (...) {
    try {
      write_frame(encode_calibration_worker_failure("unknown calibration worker failure"));
    } catch (...) {
      return EXIT_FAILURE;
    }
    return EXIT_FAILURE;
  }
}

#if defined(__linux__)
using Clock = std::chrono::steady_clock;

std::optional<Clock::time_point> ipc_deadline(std::uint64_t nanoseconds) {
  using Duration = Clock::duration;
  const auto maximum = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Duration::max()).count());
  if (nanoseconds > maximum) {
    return std::nullopt;
  }
  return Clock::time_point(
      std::chrono::duration_cast<Duration>(std::chrono::nanoseconds(nanoseconds)));
}

bool wait_for_fd(int descriptor, short events, Clock::time_point deadline) {
  while (Clock::now() < deadline) {
    pollfd item{.fd = descriptor, .events = events, .revents = 0};
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    const auto timeout = static_cast<int>(std::max<std::int64_t>(1, remaining.count()));
    const auto result = ::poll(&item, 1, timeout);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if ((item.revents & events) != 0) {
      return true;
    }
    if (result <= 0 || (item.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return false;
    }
  }
  return false;
}

void read_fd_exact(int descriptor, char* destination, std::size_t size,
                   Clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < size) {
    const auto received = ::recv(descriptor, destination + offset, size - offset, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
        wait_for_fd(descriptor, POLLIN, deadline)) {
      continue;
    }
    if (received <= 0) {
      throw CalibrationExecutionError("calibration worker request channel is truncated");
    }
    offset += static_cast<std::size_t>(received);
  }
}

void write_fd_frame(int descriptor, std::string_view frame, Clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < frame.size()) {
    const auto written =
        ::send(descriptor, frame.data() + offset, frame.size() - offset, MSG_NOSIGNAL);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
        wait_for_fd(descriptor, POLLOUT, deadline)) {
      continue;
    }
    if (written <= 0) {
      throw CalibrationExecutionError("calibration worker could not write its response");
    }
    offset += static_cast<std::size_t>(written);
  }
}
#endif

} // namespace

int connect_calibration_ipc(std::string_view address, std::uint64_t deadline_nanoseconds) {
#if defined(__linux__)
  const auto deadline = ipc_deadline(deadline_nanoseconds);
  if (!deadline.has_value() || address.empty() || address.size() >= sizeof(sockaddr_un::sun_path)) {
    return -1;
  }
  const auto descriptor = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (descriptor < 0) {
    return -1;
  }
  constexpr int buffer_bytes = 4096;
  if (::setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0 ||
      ::setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) {
    (void)::close(descriptor);
    return -1;
  }
  sockaddr_un socket_address{};
  socket_address.sun_family = AF_UNIX;
  std::memcpy(socket_address.sun_path + 1, address.data(), address.size());
  const auto size = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + address.size());
  while (true) {
    if (::connect(descriptor, reinterpret_cast<const sockaddr*>(&socket_address), size) == 0) {
      return descriptor;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno != EINPROGRESS && errno != EAGAIN && errno != EALREADY) {
      (void)::close(descriptor);
      return -1;
    }
    if (!wait_for_fd(descriptor, POLLOUT, *deadline)) {
      (void)::close(descriptor);
      return -1;
    }
    int socket_error = 0;
    socklen_t error_size = sizeof(socket_error);
    if (::getsockopt(descriptor, SOL_SOCKET, SO_ERROR, &socket_error, &error_size) != 0 ||
        socket_error != 0) {
      (void)::close(descriptor);
      return -1;
    }
    return descriptor;
  }
#else
  (void)address;
  (void)deadline_nanoseconds;
  return -1;
#endif
}

namespace {

int receive_calibration_file_fd(int descriptor, std::uint64_t deadline_nanoseconds,
                                char expected_marker, bool require_owner, bool reject_set_id,
                                std::string_view description) {
#if defined(__linux__)
  const auto deadline = ipc_deadline(deadline_nanoseconds);
  if (!deadline.has_value() || !wait_for_fd(descriptor, POLLIN, *deadline)) {
    throw CalibrationExecutionError("calibration " + std::string(description) +
                                    " descriptor was not received");
  }
  char marker = '\0';
  std::array<char, CMSG_SPACE(sizeof(int) * 2U)> control{};
  iovec bytes{.iov_base = &marker, .iov_len = 1};
  msghdr message{};
  message.msg_iov = &bytes;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  ssize_t received = -1;
  do {
    received = ::recvmsg(descriptor, &message, MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);

  int received_descriptor = -1;
  bool invalid =
      received != 1 || marker != expected_marker || (message.msg_flags & MSG_CTRUNC) != 0;
  for (auto* header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
        header->cmsg_len < CMSG_LEN(sizeof(int))) {
      invalid = true;
      continue;
    }
    const auto count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    const auto* descriptors = reinterpret_cast<const int*>(CMSG_DATA(header));
    for (std::size_t index = 0; index < count; ++index) {
      if (received_descriptor < 0) {
        received_descriptor = descriptors[index];
      } else {
        invalid = true;
        (void)::close(descriptors[index]);
      }
    }
  }
  struct stat status{};
  if (received_descriptor < 0 || ::fstat(received_descriptor, &status) != 0 ||
      !S_ISREG(status.st_mode) || (require_owner && status.st_uid != ::geteuid()) ||
      (reject_set_id && (status.st_mode & (S_ISUID | S_ISGID)) != 0)) {
    invalid = true;
  }
  if (invalid) {
    if (received_descriptor >= 0) {
      (void)::close(received_descriptor);
    }
    throw CalibrationExecutionError("calibration " + std::string(description) +
                                    " descriptor is invalid");
  }
  return received_descriptor;
#else
  (void)descriptor;
  (void)deadline_nanoseconds;
  (void)expected_marker;
  (void)require_owner;
  (void)reject_set_id;
  (void)description;
  throw CalibrationExecutionError("calibration file descriptors require Linux");
#endif
}

} // namespace

int receive_calibration_admission_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
  return receive_calibration_file_fd(descriptor, deadline_nanoseconds, 'L', true, false,
                                     "admission");
}

int receive_calibration_executable_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
  return receive_calibration_file_fd(descriptor, deadline_nanoseconds, 'X', false, true,
                                     "executable");
}

int receive_calibration_left_input_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
  return receive_calibration_file_fd(descriptor, deadline_nanoseconds, 'I', false, false,
                                     "left input");
}

int receive_calibration_right_input_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
  return receive_calibration_file_fd(descriptor, deadline_nanoseconds, 'J', false, false,
                                     "right input");
}

int receive_calibration_left_profile_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
  return receive_calibration_file_fd(descriptor, deadline_nanoseconds, 'K', false, false,
                                     "left lens profile");
}

int receive_calibration_right_profile_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
  return receive_calibration_file_fd(descriptor, deadline_nanoseconds, 'M', false, false,
                                     "right lens profile");
}

int receive_calibration_cgroup_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
#if defined(__linux__)
  const auto deadline = ipc_deadline(deadline_nanoseconds);
  if (!deadline.has_value() || !wait_for_fd(descriptor, POLLIN, *deadline)) {
    throw CalibrationExecutionError("calibration cgroup descriptor was not received");
  }
  char marker = '\0';
  std::array<char, CMSG_SPACE(sizeof(int) * 2U)> control{};
  iovec bytes{.iov_base = &marker, .iov_len = 1};
  msghdr message{};
  message.msg_iov = &bytes;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  ssize_t received = -1;
  do {
    received = ::recvmsg(descriptor, &message, MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);

  int cgroup = -1;
  bool invalid = received != 1 || marker != 'C' || (message.msg_flags & MSG_CTRUNC) != 0;
  for (auto* header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
        header->cmsg_len < CMSG_LEN(sizeof(int))) {
      invalid = true;
      continue;
    }
    const auto count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    const auto* descriptors = reinterpret_cast<const int*>(CMSG_DATA(header));
    for (std::size_t index = 0; index < count; ++index) {
      if (cgroup < 0) {
        cgroup = descriptors[index];
      } else {
        invalid = true;
        (void)::close(descriptors[index]);
      }
    }
  }
  struct stat status{};
  struct statfs filesystem{};
  if (cgroup < 0 || ::fstat(cgroup, &status) != 0 || !S_ISDIR(status.st_mode) ||
      ::fstatfs(cgroup, &filesystem) != 0 ||
      static_cast<unsigned long>(filesystem.f_type) != CGROUP2_SUPER_MAGIC) {
    invalid = true;
  }
  if (invalid) {
    if (cgroup >= 0) {
      (void)::close(cgroup);
    }
    throw CalibrationExecutionError("calibration cgroup descriptor is invalid");
  }
  return cgroup;
#else
  (void)descriptor;
  (void)deadline_nanoseconds;
  throw CalibrationExecutionError("calibration cgroup descriptors require Linux");
#endif
}

int run_calibration_worker(std::istream& input, std::ostream& output) {
  return run_calibration_worker_io(
      [&input](char* destination, std::size_t size) {
        read_stream_exact(input, destination, size);
      },
      [&output](std::string_view frame) { write_stream_frame(output, frame); });
}

int run_calibration_worker_fd(int descriptor, std::uint64_t deadline_nanoseconds) {
#if defined(__linux__)
  if (descriptor < 0) {
    return EXIT_FAILURE;
  }
  int left_input = -1;
  int right_input = -1;
  int left_profile = -1;
  int right_profile = -1;
  try {
    const auto deadline = ipc_deadline(deadline_nanoseconds);
    if (!deadline.has_value()) {
      return EXIT_FAILURE;
    }
    const auto result = run_calibration_worker_io(
        [descriptor, deadline](char* destination, std::size_t size) {
          read_fd_exact(descriptor, destination, size, *deadline);
        },
        [descriptor, deadline](std::string_view frame) {
          write_fd_frame(descriptor, frame, *deadline);
        },
        [&](GpuCalibrationRequest& request) {
          const auto retained_path = [](int file_descriptor) {
            return (std::filesystem::path("/proc") / std::to_string(::getpid()) / "fd" /
                    std::to_string(file_descriptor))
                .string();
          };
          if (request.left.retained_path.has_value()) {
            left_input = receive_calibration_left_input_fd(descriptor, deadline_nanoseconds);
            request.left.retained_path = retained_path(left_input);
          }
          if (request.right.retained_path.has_value()) {
            right_input = receive_calibration_right_input_fd(descriptor, deadline_nanoseconds);
            request.right.retained_path = retained_path(right_input);
          }
          if (request.left.lens_profile.has_value()) {
            left_profile = receive_calibration_left_profile_fd(descriptor, deadline_nanoseconds);
            request.left.lens_profile = retained_path(left_profile);
          }
          if (request.right.lens_profile.has_value()) {
            right_profile = receive_calibration_right_profile_fd(descriptor, deadline_nanoseconds);
            request.right.lens_profile = retained_path(right_profile);
          }
        });
    if (left_input >= 0) {
      (void)::close(left_input);
    }
    if (right_input >= 0) {
      (void)::close(right_input);
    }
    if (left_profile >= 0) {
      (void)::close(left_profile);
    }
    if (right_profile >= 0) {
      (void)::close(right_profile);
    }
    return result;
  } catch (...) {
    if (left_input >= 0) {
      (void)::close(left_input);
    }
    if (right_input >= 0) {
      (void)::close(right_input);
    }
    if (left_profile >= 0) {
      (void)::close(left_profile);
    }
    if (right_profile >= 0) {
      (void)::close(right_profile);
    }
    return EXIT_FAILURE;
  }
#else
  (void)descriptor;
  (void)deadline_nanoseconds;
  return EXIT_FAILURE;
#endif
}

void read_calibration_worker_bytes_fd(int descriptor, std::span<char> destination,
                                      std::uint64_t deadline_nanoseconds) {
#if defined(__linux__)
  const auto deadline = ipc_deadline(deadline_nanoseconds);
  if (descriptor < 0 || !deadline.has_value()) {
    throw CalibrationExecutionError("calibration worker transport deadline is invalid");
  }
  read_fd_exact(descriptor, destination.data(), destination.size(), *deadline);
#else
  (void)descriptor;
  (void)destination;
  (void)deadline_nanoseconds;
  throw CalibrationExecutionError("calibration worker transport requires Linux");
#endif
}

void write_calibration_worker_bytes_fd(int descriptor, std::string_view bytes,
                                       std::uint64_t deadline_nanoseconds) {
#if defined(__linux__)
  const auto deadline = ipc_deadline(deadline_nanoseconds);
  if (descriptor < 0 || !deadline.has_value()) {
    throw CalibrationExecutionError("calibration worker transport deadline is invalid");
  }
  write_fd_frame(descriptor, bytes, *deadline);
#else
  (void)descriptor;
  (void)bytes;
  (void)deadline_nanoseconds;
  throw CalibrationExecutionError("calibration worker transport requires Linux");
#endif
}

} // namespace reco::calibrate::detail
