#include "gpu_video_probe_internal.hpp"
#include "gpu_video_probe_protocol.hpp"
#include "stable_media_file_internal.hpp"

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace reco::io::detail {
namespace {

#if defined(_WIN32)
class StartupAttributes {
public:
  StartupAttributes() {
    SIZE_T size = 0;
    (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    if (size == 0) {
      return;
    }
    storage_.resize(size);
    value_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
    if (InitializeProcThreadAttributeList(value_, 1, 0, &size) == 0) {
      value_ = nullptr;
    }
  }
  StartupAttributes(const StartupAttributes&) = delete;
  StartupAttributes& operator=(const StartupAttributes&) = delete;
  ~StartupAttributes() {
    if (value_ != nullptr) {
      DeleteProcThreadAttributeList(value_);
    }
  }

  [[nodiscard]] LPPROC_THREAD_ATTRIBUTE_LIST get() const { return value_; }

private:
  std::vector<unsigned char> storage_;
  LPPROC_THREAD_ATTRIBUTE_LIST value_ = nullptr;
};
#endif

struct ReceivedRequest {
  std::string payload;
#if !defined(_WIN32)
  int descriptor = -1;

  ~ReceivedRequest() {
    if (descriptor >= 0) {
      (void)::close(descriptor);
    }
  }

  [[nodiscard]] int release_descriptor() { return std::exchange(descriptor, -1); }
#endif
};

ReceivedRequest read_request() {
#if defined(_WIN32)
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
  return {.payload = std::move(request)};
#else
  ProbeIpcFrameHeader header{};
  std::array<char, CMSG_SPACE(sizeof(int))> control{};
  iovec vector{.iov_base = header.data(), .iov_len = header.size()};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  ssize_t received = -1;
  do {
    received = ::recvmsg(STDIN_FILENO, &message,
#if defined(MSG_CMSG_CLOEXEC)
                         MSG_CMSG_CLOEXEC
#else
                         0
#endif
    );
  } while (received < 0 && errno == EINTR);
  if (received <= 0 || (message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) != 0) {
    throw GpuVideoProbeError("video probe worker request has a truncated IPC frame");
  }

  int descriptor = -1;
  for (auto* header_value = CMSG_FIRSTHDR(&message); header_value != nullptr;
       header_value = CMSG_NXTHDR(&message, header_value)) {
    if (header_value->cmsg_level != SOL_SOCKET || header_value->cmsg_type != SCM_RIGHTS ||
        header_value->cmsg_len != CMSG_LEN(sizeof(int)) || descriptor >= 0) {
      throw GpuVideoProbeError("video probe worker request has invalid descriptor authority");
    }
    std::memcpy(&descriptor, CMSG_DATA(header_value), sizeof(descriptor));
  }
#if !defined(MSG_CMSG_CLOEXEC)
  if (descriptor >= 0) {
    const int flags = ::fcntl(descriptor, F_GETFD);
    if (flags < 0 || ::fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0) {
      const int saved_error = errno;
      (void)::close(descriptor);
      throw GpuVideoProbeError("video probe worker could not restrict its media descriptor: " +
                               std::string(std::strerror(saved_error)));
    }
  }
#endif

  const auto read_exact = [](char* destination, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      ssize_t amount = -1;
      do {
        amount = ::read(STDIN_FILENO, destination + offset, size - offset);
      } while (amount < 0 && errno == EINTR);
      if (amount <= 0) {
        throw GpuVideoProbeError("video probe worker request has a truncated IPC frame");
      }
      offset += static_cast<std::size_t>(amount);
    }
  };
  if (static_cast<std::size_t>(received) < header.size()) {
    read_exact(header.data() + received, header.size() - static_cast<std::size_t>(received));
  }
  ReceivedRequest request{.payload = std::string(decode_probe_ipc_frame_header(header), '\0'),
                          .descriptor = descriptor};
  read_exact(request.payload.data(), request.payload.size());
  return request;
#endif
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

} // namespace

#if defined(_WIN32)
int run_gpu_video_probe_guardian(std::intptr_t inherited_handle) {
  std::vector<wchar_t> executable(32'768);
  const auto length =
      GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  if (length == 0 || length >= executable.size()) {
    return 2;
  }
  const std::wstring application(executable.data(), length);
  auto command_line = L"\"" + application + L"\" --reco-video-probe-worker";
  if (inherited_handle > 0) {
    command_line += L" " + std::to_wstring(static_cast<std::uint64_t>(inherited_handle));
  }
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup.StartupInfo.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  startup.StartupInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  std::vector<HANDLE> inherited_handles{
      startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError};
  if (inherited_handle > 0) {
    inherited_handles.push_back(reinterpret_cast<HANDLE>(inherited_handle));
  }
  StartupAttributes attributes;
  if (attributes.get() == nullptr ||
      UpdateProcThreadAttribute(
          attributes.get(), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited_handles.data(),
          inherited_handles.size() * sizeof(inherited_handles.front()), nullptr, nullptr) == 0) {
    return 2;
  }
  startup.lpAttributeList = attributes.get();
  PROCESS_INFORMATION process{};
  if (CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                     nullptr, &startup.StartupInfo, &process) == 0) {
    return 2;
  }
  BOOL worker_in_job = FALSE;
  const bool contained =
      IsProcessInJob(process.hProcess, nullptr, &worker_in_job) != 0 && worker_in_job != FALSE;
  if (!contained || ResumeThread(process.hThread) == std::numeric_limits<DWORD>::max()) {
    (void)TerminateProcess(process.hProcess, 2);
    (void)CloseHandle(process.hThread);
    (void)CloseHandle(process.hProcess);
    return 2;
  }
  (void)CloseHandle(process.hThread);
  const auto wait_result = WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = 2;
  if (wait_result != WAIT_OBJECT_0 || GetExitCodeProcess(process.hProcess, &exit_code) == 0) {
    exit_code = 2;
  }
  (void)CloseHandle(process.hProcess);
  return exit_code <= static_cast<DWORD>(std::numeric_limits<int>::max())
             ? static_cast<int>(exit_code)
             : 2;
}
#endif

int run_gpu_video_probe_worker(std::intptr_t inherited_handle) {
  std::string response;
  try {
    auto received = read_request();
    auto request = decode_probe_request(received.payload);
    if (request.expects_stable_source) {
#if defined(_WIN32)
      if (inherited_handle <= 0) {
        throw GpuVideoProbeError("video probe worker did not inherit its stable media handle");
      }
      const int descriptor = _open_osfhandle(inherited_handle, _O_RDONLY | _O_BINARY);
      if (descriptor < 0) {
        throw GpuVideoProbeError("video probe worker could not adopt its stable media handle");
      }
      inherited_handle = -1;
#else
      if (received.descriptor < 0) {
        throw GpuVideoProbeError("video probe worker did not receive its stable media descriptor");
      }
      const int descriptor = received.release_descriptor();
#endif
      request.config.stable_source =
          adopt_stable_media_worker_descriptor(descriptor, request.config.path);
    } else {
#if defined(_WIN32)
      if (inherited_handle > 0) {
        throw GpuVideoProbeError("video probe worker received unexpected stable media authority");
      }
#else
      if (received.descriptor >= 0) {
        throw GpuVideoProbeError("video probe worker received unexpected stable media authority");
      }
#endif
    }
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
