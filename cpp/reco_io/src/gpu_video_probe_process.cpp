#include "reco/io/gpu_video_probe.hpp"

#include "gpu_video_probe_process_test.hpp"
#include "gpu_video_probe_protocol.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// Older Windows SDK headers omit the JobList convenience macro even though
// UpdateProcThreadAttribute accepts the documented attribute on supported OSes.
constexpr DWORD_PTR kProcThreadAttributeJobList = ProcThreadAttributeValue(13, FALSE, TRUE, FALSE);
#else
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

extern char** environ;
#endif

namespace reco::io {
namespace {

constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr std::uint64_t kMinimumProbeTimeoutNs = kNanosecondsPerSecond;
constexpr std::uint64_t kMaximumProbeTimeoutNs = 3'600ULL * kNanosecondsPerSecond;
constexpr auto kProcessPollInterval = std::chrono::milliseconds(2);
constexpr auto kMinimumTerminationReserve = std::chrono::milliseconds(50);
constexpr auto kMaximumTerminationReserve = std::chrono::milliseconds(250);
constexpr std::size_t kMaximumConcurrentProbeWorkers = 32;

struct SupervisorSlots {
  std::mutex mutex;
  std::size_t active = 0;
};

SupervisorSlots& supervisor_slots() {
  static auto* slots = new SupervisorSlots;
  return *slots;
}

class SupervisorSlot {
public:
  SupervisorSlot() {
    auto& slots = supervisor_slots();
    std::lock_guard lock(slots.mutex);
    if (slots.active >= kMaximumConcurrentProbeWorkers) {
      throw GpuVideoProbeError("all video probe worker supervisor slots are occupied");
    }
    ++slots.active;
  }
  SupervisorSlot(const SupervisorSlot&) = delete;
  SupervisorSlot& operator=(const SupervisorSlot&) = delete;
  ~SupervisorSlot() {
    auto& slots = supervisor_slots();
    std::lock_guard lock(slots.mutex);
    --slots.active;
  }
};

[[noreturn]] void throw_worker_timeout() {
  throw GpuVideoProbeError("video probe worker timed out after exceeding the configured timeout");
}

void require_worker_launch_active(std::chrono::steady_clock::time_point deadline) {
  if (std::chrono::steady_clock::now() >= deadline) {
    throw_worker_timeout();
  }
}

void wait_for_worker_launch_delay(std::chrono::nanoseconds delay,
                                  std::chrono::steady_clock::time_point deadline) {
  if (delay.count() <= 0) {
    require_worker_launch_active(deadline);
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline || delay >= deadline - now) {
    std::this_thread::sleep_until(deadline);
    throw_worker_timeout();
  }
  std::this_thread::sleep_for(delay);
  require_worker_launch_active(deadline);
}

#if defined(_WIN32)

class UniqueHandle {
public:
  UniqueHandle() = default;
  explicit UniqueHandle(HANDLE value) : value_(value) {}
  UniqueHandle(const UniqueHandle&) = delete;
  UniqueHandle& operator=(const UniqueHandle&) = delete;
  UniqueHandle(UniqueHandle&& other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
  UniqueHandle& operator=(UniqueHandle&& other) noexcept {
    if (this != &other) {
      reset(std::exchange(other.value_, nullptr));
    }
    return *this;
  }
  ~UniqueHandle() { reset(); }

  [[nodiscard]] HANDLE get() const { return value_; }
  [[nodiscard]] explicit operator bool() const {
    return value_ != nullptr && value_ != INVALID_HANDLE_VALUE;
  }
  HANDLE release() { return std::exchange(value_, nullptr); }
  void reset(HANDLE value = nullptr) {
    if (*this) {
      CloseHandle(value_);
    }
    value_ = value;
  }

private:
  HANDLE value_ = nullptr;
};

class StartupAttributeList {
public:
  explicit StartupAttributeList(DWORD attribute_count) {
    SIZE_T size = 0;
    (void)InitializeProcThreadAttributeList(nullptr, attribute_count, 0, &size);
    if (size == 0) {
      throw GpuVideoProbeError("failed to size video probe worker launch attributes");
    }
    storage_.resize(size);
    value_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
    if (InitializeProcThreadAttributeList(value_, attribute_count, 0, &size) == 0) {
      throw GpuVideoProbeError("failed to initialize video probe worker launch attributes");
    }
  }
  StartupAttributeList(const StartupAttributeList&) = delete;
  StartupAttributeList& operator=(const StartupAttributeList&) = delete;
  ~StartupAttributeList() {
    if (value_ != nullptr) {
      DeleteProcThreadAttributeList(value_);
    }
  }

  [[nodiscard]] LPPROC_THREAD_ATTRIBUTE_LIST get() const { return value_; }

private:
  std::vector<unsigned char> storage_;
  LPPROC_THREAD_ATTRIBUTE_LIST value_ = nullptr;
};

std::wstring utf8_to_wide(std::string_view value) {
  if (value.empty()) {
    return {};
  }
  if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw GpuVideoProbeError("video probe worker path is too long");
  }
  const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                        static_cast<int>(value.size()), nullptr, 0);
  if (size <= 0) {
    throw GpuVideoProbeError("video probe worker path is not valid UTF-8");
  }
  std::wstring result(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), result.data(), size) != size) {
    throw GpuVideoProbeError("failed to convert video probe worker path");
  }
  return result;
}

void write_all(HANDLE output, std::string_view payload, std::string_view description) {
  std::size_t offset = 0;
  while (offset < payload.size()) {
    const auto remaining =
        std::min<std::size_t>(payload.size() - offset, std::numeric_limits<DWORD>::max());
    DWORD written = 0;
    if (WriteFile(output, payload.data() + offset, static_cast<DWORD>(remaining), &written,
                  nullptr) == 0 ||
        written == 0) {
      throw GpuVideoProbeError("failed to write video probe worker " + std::string(description) +
                               " (Windows error " + std::to_string(GetLastError()) + ")");
    }
    offset += written;
  }
}

void write_request(HANDLE output, std::string_view request) {
  const auto header = detail::encode_probe_ipc_frame_header(request.size());
  write_all(output, std::string_view(header.data(), header.size()), "request frame header");
  write_all(output, request, "request");
}

void read_exact(HANDLE input, char* destination, std::size_t size,
                std::chrono::steady_clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < size) {
    DWORD available = 0;
    if (PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr) == 0) {
      throw GpuVideoProbeError("video probe worker response has a truncated IPC frame");
    }
    if (available == 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw_worker_timeout();
      }
      std::this_thread::sleep_for(kProcessPollInterval);
      continue;
    }
    const auto remaining = std::min<std::size_t>(size - offset, available);
    DWORD received = 0;
    if (ReadFile(input, destination + offset, static_cast<DWORD>(remaining), &received, nullptr) ==
            0 ||
        received == 0) {
      throw GpuVideoProbeError("video probe worker response has a truncated IPC frame");
    }
    offset += received;
  }
}

std::string read_response(HANDLE input, std::chrono::steady_clock::time_point deadline) {
  detail::ProbeIpcFrameHeader header{};
  read_exact(input, header.data(), header.size(), deadline);
  std::string response(detail::decode_probe_ipc_frame_header(header), '\0');
  read_exact(input, response.data(), response.size(), deadline);

  DWORD trailing = 0;
  if (PeekNamedPipe(input, nullptr, 0, nullptr, &trailing, nullptr) != 0 && trailing != 0) {
    throw GpuVideoProbeError("video probe worker response has trailing IPC bytes");
  }
  return response;
}

std::string run_probe_worker(const std::filesystem::path& worker_path, std::string_view request,
                             std::chrono::steady_clock::time_point deadline,
                             std::chrono::steady_clock::time_point cleanup_deadline,
                             std::chrono::nanoseconds pre_worker_spawn_delay) {
  require_worker_launch_active(deadline);
  SECURITY_ATTRIBUTES security{.nLength = sizeof(SECURITY_ATTRIBUTES),
                               .lpSecurityDescriptor = nullptr,
                               .bInheritHandle = TRUE};
  HANDLE child_stdin_raw = nullptr;
  HANDLE parent_stdin_raw = nullptr;
  if (CreatePipe(&child_stdin_raw, &parent_stdin_raw, &security,
                 static_cast<DWORD>(detail::kMaximumProbeIpcBytes)) == 0) {
    throw GpuVideoProbeError("failed to create video probe worker input pipe");
  }
  UniqueHandle child_stdin(child_stdin_raw);
  UniqueHandle parent_stdin(parent_stdin_raw);
  HANDLE parent_stdout_raw = nullptr;
  HANDLE child_stdout_raw = nullptr;
  if (CreatePipe(&parent_stdout_raw, &child_stdout_raw, &security,
                 static_cast<DWORD>(detail::kMaximumProbeIpcBytes)) == 0) {
    throw GpuVideoProbeError("failed to create video probe worker output pipe");
  }
  UniqueHandle parent_stdout(parent_stdout_raw);
  UniqueHandle child_stdout(child_stdout_raw);
  if (SetHandleInformation(parent_stdin.get(), HANDLE_FLAG_INHERIT, 0) == 0 ||
      SetHandleInformation(parent_stdout.get(), HANDLE_FLAG_INHERIT, 0) == 0) {
    throw GpuVideoProbeError("failed to restrict video probe worker pipe inheritance");
  }

  HANDLE child_stderr_raw = nullptr;
  const auto stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
  if (stderr_handle != nullptr && stderr_handle != INVALID_HANDLE_VALUE) {
    if (DuplicateHandle(GetCurrentProcess(), stderr_handle, GetCurrentProcess(), &child_stderr_raw,
                        0, TRUE, DUPLICATE_SAME_ACCESS) == 0) {
      throw GpuVideoProbeError("failed to duplicate video probe worker stderr");
    }
  } else {
    child_stderr_raw = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (child_stderr_raw == INVALID_HANDLE_VALUE) {
      throw GpuVideoProbeError("failed to open video probe worker stderr");
    }
  }
  UniqueHandle child_stderr(child_stderr_raw);

  UniqueHandle job(CreateJobObjectW(nullptr, nullptr));
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!job || SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits,
                                      sizeof(limits)) == 0) {
    throw GpuVideoProbeError("failed to configure video probe worker Job");
  }

  StartupAttributeList attributes(2);
  std::array<HANDLE, 3> inherited_handles{child_stdin.get(), child_stdout.get(),
                                          child_stderr.get()};
  if (UpdateProcThreadAttribute(attributes.get(), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                inherited_handles.data(), sizeof(inherited_handles), nullptr,
                                nullptr) == 0) {
    throw GpuVideoProbeError("failed to restrict video probe worker inherited handles");
  }
  std::array<HANDLE, 1> assigned_jobs{job.get()};
  if (UpdateProcThreadAttribute(attributes.get(), 0, kProcThreadAttributeJobList,
                                assigned_jobs.data(), sizeof(assigned_jobs), nullptr,
                                nullptr) == 0) {
    throw GpuVideoProbeError("failed to assign the video probe worker Job at launch");
  }

  const auto encoded_path = worker_path.u8string();
  const std::string utf8_path(reinterpret_cast<const char*>(encoded_path.data()),
                              encoded_path.size());
  const auto application = utf8_to_wide(utf8_path);
  auto command_line = L"\"" + application + L"\" --reco-video-probe-guardian";
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = child_stdin.get();
  startup.StartupInfo.hStdOutput = child_stdout.get();
  startup.StartupInfo.hStdError = child_stderr.get();
  PROCESS_INFORMATION process_info{};
  wait_for_worker_launch_delay(pre_worker_spawn_delay, deadline);
  if (CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                     nullptr, &startup.StartupInfo, &process_info) == 0) {
    throw GpuVideoProbeError("failed to start video probe worker (Windows error " +
                             std::to_string(GetLastError()) + ")");
  }
  UniqueHandle process(process_info.hProcess);
  UniqueHandle thread(process_info.hThread);
  const auto terminate_worker = [&] {
    (void)TerminateJobObject(job.get(), 1);
    (void)TerminateProcess(process.get(), 1);
    const auto now = std::chrono::steady_clock::now();
    if (now < cleanup_deadline) {
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(cleanup_deadline - now);
      const auto wait_ms = static_cast<DWORD>(
          std::clamp<std::uint64_t>(static_cast<std::uint64_t>(remaining.count()) + 1ULL, 1ULL,
                                    std::numeric_limits<DWORD>::max() - 1ULL));
      (void)WaitForSingleObject(process.get(), wait_ms);
    }
    job.reset();
  };
  BOOL worker_in_job = FALSE;
  if (IsProcessInJob(process.get(), job.get(), &worker_in_job) == 0 ||
      (worker_in_job == FALSE && AssignProcessToJobObject(job.get(), process.get()) == 0)) {
    terminate_worker();
    throw GpuVideoProbeError("failed to verify video probe worker Job containment");
  }
  if (ResumeThread(thread.get()) == std::numeric_limits<DWORD>::max()) {
    terminate_worker();
    throw GpuVideoProbeError("failed to release the video probe guardian");
  }
  thread.reset();
  if (std::chrono::steady_clock::now() >= deadline) {
    terminate_worker();
    throw_worker_timeout();
  }
  child_stdin.reset();
  child_stdout.reset();
  child_stderr.reset();
  if (std::chrono::steady_clock::now() >= deadline) {
    terminate_worker();
    throw_worker_timeout();
  }
  std::exception_ptr write_error;
  std::thread writer([input = std::move(parent_stdin), request, &write_error]() {
    try {
      write_request(input.get(), request);
    } catch (...) {
      write_error = std::current_exception();
    }
  });

  const auto terminate_and_join = [&] {
    terminate_worker();
    (void)CancelSynchronousIo(writer.native_handle());
    writer.join();
  };

  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    terminate_and_join();
    throw_worker_timeout();
  }
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
  const auto wait_ms = static_cast<DWORD>(
      std::clamp<std::uint64_t>(static_cast<std::uint64_t>(remaining.count()) + 1ULL, 1ULL,
                                std::numeric_limits<DWORD>::max() - 1ULL));
  const auto wait_result = WaitForSingleObject(process.get(), wait_ms);
  if (wait_result == WAIT_TIMEOUT) {
    terminate_and_join();
    throw_worker_timeout();
  }
  if (wait_result != WAIT_OBJECT_0) {
    terminate_and_join();
    throw GpuVideoProbeError("failed while waiting for video probe worker");
  }
  writer.join();
  if (TerminateJobObject(job.get(), 1) == 0) {
    job.reset();
  }
  if (write_error != nullptr) {
    std::rethrow_exception(write_error);
  }
  DWORD exit_code = 0;
  if (GetExitCodeProcess(process.get(), &exit_code) == 0 || exit_code != 0) {
    throw GpuVideoProbeError("video probe worker exited abnormally");
  }
  return read_response(parent_stdout.get(), deadline);
}

#else

class UniqueFd {
public:
  UniqueFd() = default;
  explicit UniqueFd(int value) : value_(value) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset(std::exchange(other.value_, -1));
    }
    return *this;
  }
  ~UniqueFd() { reset(); }

  [[nodiscard]] int get() const { return value_; }
  int release() { return std::exchange(value_, -1); }
  void reset(int value = -1) {
    if (value_ >= 0) {
      (void)::close(value_);
    }
    value_ = value;
  }

private:
  int value_ = -1;
};

void wait_for_socket(int socket, short events, std::chrono::steady_clock::time_point deadline) {
  while (true) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      throw_worker_timeout();
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    const auto timeout_ms = static_cast<int>(
        std::clamp<std::int64_t>(remaining.count() + 1, 1, std::numeric_limits<int>::max()));
    pollfd descriptor{.fd = socket, .events = events, .revents = 0};
    const auto result = ::poll(&descriptor, 1, timeout_ms);
    if (result > 0) {
      return;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result == 0) {
      throw_worker_timeout();
    }
    throw GpuVideoProbeError("failed while waiting for video probe worker IPC: " +
                             std::string(std::strerror(errno)));
  }
}

void make_nonblocking(int descriptor) {
  const auto flags = ::fcntl(descriptor, F_GETFL);
  if (flags < 0 || ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0) {
    throw GpuVideoProbeError("failed to configure video probe worker IPC: " +
                             std::string(std::strerror(errno)));
  }
}

void make_close_on_exec(int descriptor) {
  const auto flags = ::fcntl(descriptor, F_GETFD);
  if (flags < 0 || ::fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0) {
    throw GpuVideoProbeError("failed to restrict video probe worker IPC inheritance: " +
                             std::string(std::strerror(errno)));
  }
}

void create_socket_pair(int descriptors[2], std::string_view description) {
#if defined(SOCK_CLOEXEC)
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, descriptors) == 0) {
    return;
  }
  if (errno != EINVAL && errno != EPROTONOSUPPORT) {
    throw GpuVideoProbeError("failed to create video probe worker " + std::string(description) +
                             " socket: " + std::string(std::strerror(errno)));
  }
#endif
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) != 0) {
    throw GpuVideoProbeError("failed to create video probe worker " + std::string(description) +
                             " socket: " + std::string(std::strerror(errno)));
  }
  try {
    make_close_on_exec(descriptors[0]);
    make_close_on_exec(descriptors[1]);
  } catch (...) {
    const auto saved_error = errno;
    (void)::close(descriptors[0]);
    (void)::close(descriptors[1]);
    descriptors[0] = -1;
    descriptors[1] = -1;
    errno = saved_error;
    throw;
  }
}

void write_all(int output, std::string_view payload,
               std::chrono::steady_clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < payload.size()) {
    const auto written = ::send(output, payload.data() + offset, payload.size() - offset,
#if defined(MSG_NOSIGNAL)
                                MSG_NOSIGNAL
#else
                                0
#endif
    );
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      wait_for_socket(output, POLLOUT, deadline);
      continue;
    }
    if (written <= 0) {
      throw GpuVideoProbeError("failed to write video probe worker request: " +
                               std::string(std::strerror(errno)));
    }
    offset += static_cast<std::size_t>(written);
  }
}

void write_request(int output, std::string_view request,
                   std::chrono::steady_clock::time_point deadline) {
  const auto header = detail::encode_probe_ipc_frame_header(request.size());
  write_all(output, std::string_view(header.data(), header.size()), deadline);
  write_all(output, request, deadline);
}

void read_exact(int input, char* destination, std::size_t size,
                std::chrono::steady_clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < size) {
    const auto received = ::recv(input, destination + offset, size - offset, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      wait_for_socket(input, POLLIN, deadline);
      continue;
    }
    if (received < 0) {
      throw GpuVideoProbeError("failed to read video probe worker response: " +
                               std::string(std::strerror(errno)));
    }
    if (received == 0) {
      throw GpuVideoProbeError("video probe worker response has a truncated IPC frame");
    }
    offset += static_cast<std::size_t>(received);
  }
}

std::string read_response(int input, std::chrono::steady_clock::time_point deadline) {
  detail::ProbeIpcFrameHeader header{};
  read_exact(input, header.data(), header.size(), deadline);
  std::string response(detail::decode_probe_ipc_frame_header(header), '\0');
  read_exact(input, response.data(), response.size(), deadline);

  char trailing = '\0';
  ssize_t trailing_size = 0;
  do {
    trailing_size = ::recv(input, &trailing, 1, MSG_PEEK);
  } while (trailing_size < 0 && errno == EINTR);
  if (trailing_size > 0) {
    throw GpuVideoProbeError("video probe worker response has trailing IPC bytes");
  }
  if (trailing_size < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    throw GpuVideoProbeError("failed to inspect video probe worker response: " +
                             std::string(std::strerror(errno)));
  }
  return response;
}

constexpr int kGuardianWorkerInput = 3;
constexpr int kGuardianWorkerOutput = 4;
constexpr int kGuardianFirstUnusedDescriptor = 5;
constexpr char kGuardianReady = 'R';
constexpr char kGuardianLaunch = 'L';
constexpr char kGuardianStarted = 'S';
constexpr char kGuardianExited = 'E';
constexpr char kGuardianAcknowledge = 'A';
constexpr char kGuardianTerminate = 'T';

UniqueFd duplicate_for_guardian(int descriptor) {
#if defined(F_DUPFD_CLOEXEC)
  const auto duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, kGuardianFirstUnusedDescriptor);
#else
  const auto duplicate = ::fcntl(descriptor, F_DUPFD, kGuardianFirstUnusedDescriptor);
#endif
  if (duplicate < 0) {
    throw GpuVideoProbeError("failed to isolate video probe guardian descriptors: " +
                             std::string(std::strerror(errno)));
  }
#if !defined(F_DUPFD_CLOEXEC)
  try {
    make_close_on_exec(duplicate);
  } catch (...) {
    (void)::close(duplicate);
    throw;
  }
#endif
  return UniqueFd(duplicate);
}

void kill_worker_process_group(pid_t worker_pid) {
  if (worker_pid <= 0) {
    return;
  }
  (void)::kill(-worker_pid, SIGKILL);
  (void)::kill(worker_pid, SIGKILL);
}

void kill_worker_descendants(pid_t worker_pid) {
  if (worker_pid > 0) {
    (void)::kill(-worker_pid, SIGKILL);
  }
}

bool wait_for_process_exit(pid_t pid, int* status, std::chrono::steady_clock::time_point deadline) {
  while (true) {
    const auto result = ::waitpid(pid, status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      return true;
    }
    if (result < 0 && errno != EINTR) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(kProcessPollInterval);
  }
}

[[noreturn]] void guardian_exit(int status) { std::_Exit(status); }

bool guardian_write(int descriptor, const void* data, std::size_t size) {
  auto* bytes = static_cast<const char*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const auto written = ::send(descriptor, bytes + offset, size - offset,
#if defined(MSG_NOSIGNAL)
                                MSG_NOSIGNAL
#else
                                0
#endif
    );
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

void guardian_close_from(int descriptor, long maximum_descriptor) {
#if defined(__linux__) && defined(SYS_close_range)
  if (::syscall(SYS_close_range, static_cast<unsigned int>(descriptor),
                std::numeric_limits<unsigned int>::max(), 0U) == 0) {
    return;
  }
#elif defined(__APPLE__)
  ::closefrom(descriptor);
  return;
#endif
  for (long value = descriptor; value < maximum_descriptor; ++value) {
    (void)::close(static_cast<int>(value));
  }
}

void guardian_terminate_worker(pid_t worker_pid) {
  kill_worker_process_group(worker_pid);
  if (worker_pid <= 0) {
    return;
  }
  constexpr auto kGuardianReapAttempts = 25;
  constexpr timespec kReapPause{.tv_sec = 0, .tv_nsec = 2'000'000};
  for (int attempt = 0; attempt < kGuardianReapAttempts; ++attempt) {
    int status = 0;
    const auto result = ::waitpid(worker_pid, &status, WNOHANG);
    if (result == worker_pid || (result < 0 && errno == ECHILD)) {
      return;
    }
    if (result < 0 && errno != EINTR) {
      return;
    }
    (void)::nanosleep(&kReapPause, nullptr);
  }
}

std::size_t format_guardian_parent_argument(char* destination, std::size_t capacity, pid_t pid) {
  char reversed[32]{};
  auto value = static_cast<std::uint64_t>(pid);
  std::size_t digits = 0;
  do {
    reversed[digits++] = static_cast<char>('0' + (value % 10U));
    value /= 10U;
  } while (value != 0 && digits < sizeof(reversed));
  if (digits + 3U > capacity) {
    return 0;
  }
  for (std::size_t index = 0; index < digits; ++index) {
    destination[index] = reversed[digits - index - 1U];
  }
  destination[digits] = ':';
  destination[digits + 1U] = '1';
  destination[digits + 2U] = '\0';
  return digits + 2U;
}

[[noreturn]] void run_guardian_child(const char* executable, int control_descriptor,
                                     int worker_input_descriptor, int worker_output_descriptor,
                                     long maximum_descriptor) {
  if (::setpgid(0, 0) != 0 || ::dup2(control_descriptor, STDIN_FILENO) < 0 ||
      ::dup2(control_descriptor, STDOUT_FILENO) < 0 ||
      ::dup2(worker_input_descriptor, kGuardianWorkerInput) < 0 ||
      ::dup2(worker_output_descriptor, kGuardianWorkerOutput) < 0) {
    guardian_exit(2);
  }
  guardian_close_from(kGuardianFirstUnusedDescriptor, maximum_descriptor);
  struct sigaction child_action{};
  child_action.sa_handler = SIG_DFL;
  (void)::sigemptyset(&child_action.sa_mask);
  if (::sigaction(SIGCHLD, &child_action, nullptr) != 0 ||
      !guardian_write(STDOUT_FILENO, &kGuardianReady, 1)) {
    guardian_exit(2);
  }

  char command = '\0';
  ssize_t received = -1;
  do {
    received = ::recv(STDIN_FILENO, &command, 1, 0);
  } while (received < 0 && errno == EINTR);
  if (received != 1 || command != kGuardianLaunch) {
    guardian_exit(0);
  }

  const auto worker_pid = ::fork();
  if (worker_pid == 0) {
    if (::setpgid(0, 0) != 0 || ::dup2(kGuardianWorkerInput, STDIN_FILENO) < 0 ||
        ::dup2(kGuardianWorkerOutput, STDOUT_FILENO) < 0) {
      guardian_exit(127);
    }
    guardian_close_from(kGuardianWorkerInput, maximum_descriptor);
    char parent_argument[40]{};
    if (format_guardian_parent_argument(parent_argument, sizeof(parent_argument), ::getppid()) ==
        0) {
      guardian_exit(127);
    }
    char* const arguments[] = {const_cast<char*>(executable),
                               const_cast<char*>("--reco-video-probe-worker"), parent_argument,
                               nullptr};
    ::execve(executable, arguments, environ);
    guardian_exit(127);
  }
  if (worker_pid < 0) {
    guardian_exit(2);
  }
  (void)::setpgid(worker_pid, worker_pid);
  (void)::close(kGuardianWorkerInput);
  (void)::close(kGuardianWorkerOutput);
  const auto encoded_worker_pid = static_cast<std::uint64_t>(worker_pid);
  if (!guardian_write(STDOUT_FILENO, &kGuardianStarted, 1) ||
      !guardian_write(STDOUT_FILENO, &encoded_worker_pid, sizeof(encoded_worker_pid))) {
    guardian_terminate_worker(worker_pid);
    guardian_exit(2);
  }

  int worker_status = 0;
  while (true) {
    const auto wait_result = ::waitpid(worker_pid, &worker_status, WNOHANG);
    if (wait_result == worker_pid) {
      break;
    }
    if (wait_result < 0 && errno == ECHILD) {
      worker_status = 0;
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      guardian_terminate_worker(worker_pid);
      guardian_exit(2);
    }
    pollfd control{.fd = STDIN_FILENO, .events = POLLIN, .revents = 0};
    const auto poll_result = ::poll(&control, 1, 2);
    if (poll_result < 0 && errno == EINTR) {
      continue;
    }
    if (poll_result < 0) {
      guardian_terminate_worker(worker_pid);
      guardian_exit(2);
    }
    if (poll_result > 0) {
      command = '\0';
      do {
        received = ::recv(STDIN_FILENO, &command, 1, 0);
      } while (received < 0 && errno == EINTR);
      if (received != 1 || command == kGuardianTerminate) {
        guardian_terminate_worker(worker_pid);
        guardian_exit(0);
      }
    }
  }

  kill_worker_descendants(worker_pid);
  if (!guardian_write(STDOUT_FILENO, &kGuardianExited, 1) ||
      !guardian_write(STDOUT_FILENO, &worker_status, sizeof(worker_status))) {
    guardian_exit(2);
  }
  do {
    received = ::recv(STDIN_FILENO, &command, 1, 0);
  } while (received < 0 && errno == EINTR);
  guardian_exit(received == 1 && command == kGuardianAcknowledge ? 0 : 2);
}

class GuardianProcess {
public:
  GuardianProcess(pid_t pid, UniqueFd control, std::chrono::steady_clock::time_point deadline)
      : pid_(pid), control_(std::move(control)), deadline_(deadline) {}
  GuardianProcess(const GuardianProcess&) = delete;
  GuardianProcess& operator=(const GuardianProcess&) = delete;
  ~GuardianProcess() { terminate(); }

  [[nodiscard]] int control() const { return control_.get(); }
  void set_worker_pid(pid_t worker_pid) { worker_pid_ = worker_pid; }
  void mark_worker_exited() { worker_pid_ = -1; }

  bool finish() {
    if (pid_ <= 0) {
      return true;
    }
    const char acknowledge = kGuardianAcknowledge;
    (void)::send(control_.get(), &acknowledge, 1,
#if defined(MSG_NOSIGNAL)
                 MSG_NOSIGNAL
#else
                 0
#endif
    );
    control_.reset();
    int status = 0;
    const auto exited = wait_for_process_exit(pid_, &status, deadline_);
    if (exited) {
      pid_ = -1;
    }
    return exited && (status == 0 || (WIFEXITED(status) && WEXITSTATUS(status) == 0));
  }

private:
  void terminate() {
    if (pid_ <= 0) {
      return;
    }
    kill_worker_process_group(worker_pid_);
    const char terminate = kGuardianTerminate;
    (void)::send(control_.get(), &terminate, 1,
#if defined(MSG_NOSIGNAL)
                 MSG_NOSIGNAL
#else
                 0
#endif
    );
    control_.reset();
    int status = 0;
    const auto now = std::chrono::steady_clock::now();
    const auto maximum_grace = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::milliseconds(50));
    const auto grace_deadline =
        now < deadline_ ? now + std::min((deadline_ - now) / 2, maximum_grace) : now;
    if (!wait_for_process_exit(pid_, &status, grace_deadline)) {
      (void)::kill(pid_, SIGKILL);
      (void)wait_for_process_exit(pid_, &status, deadline_);
    }
    pid_ = -1;
  }

  pid_t pid_ = -1;
  pid_t worker_pid_ = -1;
  UniqueFd control_;
  std::chrono::steady_clock::time_point deadline_;
};

std::string run_probe_worker(const std::filesystem::path& worker_path, std::string_view request,
                             std::chrono::steady_clock::time_point deadline,
                             std::chrono::steady_clock::time_point cleanup_deadline,
                             std::chrono::nanoseconds pre_worker_spawn_delay) {
  require_worker_launch_active(deadline);
  int request_socket[2] = {-1, -1};
  int response_socket[2] = {-1, -1};
  int control_socket[2] = {-1, -1};
  create_socket_pair(request_socket, "input");
  UniqueFd child_input(request_socket[0]);
  UniqueFd parent_input(request_socket[1]);
  create_socket_pair(response_socket, "output");
  UniqueFd parent_output(response_socket[0]);
  UniqueFd child_output(response_socket[1]);
  create_socket_pair(control_socket, "guardian control");
  UniqueFd child_control(control_socket[0]);
  UniqueFd parent_control(control_socket[1]);
  make_nonblocking(parent_input.get());
  make_nonblocking(parent_output.get());
  make_nonblocking(parent_control.get());
#if defined(SO_NOSIGPIPE) && !defined(MSG_NOSIGNAL)
  const int suppress_sigpipe = 1;
  if (::setsockopt(parent_input.get(), SOL_SOCKET, SO_NOSIGPIPE, &suppress_sigpipe,
                   sizeof(suppress_sigpipe)) != 0) {
    throw GpuVideoProbeError("failed to suppress SIGPIPE for video probe worker IPC: " +
                             std::string(std::strerror(errno)));
  }
#endif

  const auto executable = worker_path.string();
  auto guard_control = duplicate_for_guardian(child_control.get());
  auto guard_input = duplicate_for_guardian(child_input.get());
  auto guard_output = duplicate_for_guardian(child_output.get());
  const auto maximum_descriptor = ::sysconf(_SC_OPEN_MAX);
  if (maximum_descriptor < kGuardianFirstUnusedDescriptor) {
    throw GpuVideoProbeError("failed to determine the video probe guardian descriptor limit");
  }
  wait_for_worker_launch_delay(pre_worker_spawn_delay, deadline);
  const auto guardian_pid = ::fork();
  if (guardian_pid == 0) {
    run_guardian_child(executable.c_str(), guard_control.get(), guard_input.get(),
                       guard_output.get(), maximum_descriptor);
  }
  if (guardian_pid < 0) {
    throw GpuVideoProbeError("failed to start video probe guardian: " +
                             std::string(std::strerror(errno)));
  }
  GuardianProcess guardian(guardian_pid, std::move(parent_control), cleanup_deadline);
  child_input.reset();
  child_output.reset();
  child_control.reset();
  guard_control.reset();
  guard_input.reset();
  guard_output.reset();
  char lifecycle = '\0';
  read_exact(guardian.control(), &lifecycle, 1, deadline);
  if (lifecycle != kGuardianReady) {
    throw GpuVideoProbeError("video probe guardian failed its readiness handshake");
  }
  write_all(guardian.control(), std::string_view(&kGuardianLaunch, 1), deadline);
  read_exact(guardian.control(), &lifecycle, 1, deadline);
  if (lifecycle != kGuardianStarted) {
    throw GpuVideoProbeError("video probe guardian failed its worker launch handshake");
  }
  std::uint64_t encoded_worker_pid = 0;
  read_exact(guardian.control(), reinterpret_cast<char*>(&encoded_worker_pid),
             sizeof(encoded_worker_pid), deadline);
  if (encoded_worker_pid == 0 ||
      encoded_worker_pid > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    throw GpuVideoProbeError("video probe guardian returned an invalid worker process ID");
  }
  guardian.set_worker_pid(static_cast<pid_t>(encoded_worker_pid));
  require_worker_launch_active(deadline);
  write_request(parent_input.get(), request, deadline);
  parent_input.reset();

  read_exact(guardian.control(), &lifecycle, 1, deadline);
  if (lifecycle != kGuardianExited) {
    throw GpuVideoProbeError("video probe guardian failed its worker exit handshake");
  }
  int status = 0;
  read_exact(guardian.control(), reinterpret_cast<char*>(&status), sizeof(status), deadline);
  guardian.mark_worker_exited();
  if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
    throw GpuVideoProbeError("failed to start video probe worker");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw GpuVideoProbeError("video probe worker exited abnormally");
  }
  auto response = read_response(parent_output.get(), deadline);
  if (!guardian.finish()) {
    throw GpuVideoProbeError("failed to reap video probe guardian before the configured timeout");
  }
  return response;
}

#endif

std::string run_probe_worker_bounded(std::filesystem::path worker_path, std::string request,
                                     std::chrono::steady_clock::time_point worker_deadline,
                                     std::chrono::steady_clock::time_point public_deadline,
                                     std::chrono::nanoseconds supervisor_start_delay,
                                     std::chrono::nanoseconds pre_worker_spawn_delay) {
  SupervisorSlot supervisor_slot;
  wait_for_worker_launch_delay(supervisor_start_delay, public_deadline);
  return run_probe_worker(worker_path, request, worker_deadline, public_deadline,
                          pre_worker_spawn_delay);
}

GpuVideoProbe probe_gpu_video_with_delays(const GpuFileDecodeConfig& config,
                                          const std::filesystem::path& worker_path,
                                          std::uint64_t timeout_ns,
                                          std::chrono::nanoseconds supervisor_start_delay,
                                          std::chrono::nanoseconds pre_worker_spawn_delay) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  if (timeout_ns < kMinimumProbeTimeoutNs || timeout_ns > kMaximumProbeTimeoutNs) {
    throw std::invalid_argument("video probe timeout must be between one second and one hour");
  }
  if (worker_path.empty()) {
    throw std::invalid_argument("video probe worker path is required");
  }
  if (!worker_path.is_absolute()) {
    throw std::invalid_argument("video probe worker path must be absolute");
  }
  const auto timeout = std::chrono::nanoseconds(timeout_ns);
  const auto termination_reserve =
      std::clamp(timeout / 10,
                 std::chrono::duration_cast<std::chrono::nanoseconds>(kMinimumTerminationReserve),
                 std::chrono::duration_cast<std::chrono::nanoseconds>(kMaximumTerminationReserve));
  const auto public_deadline = std::chrono::steady_clock::now() + timeout;
  const auto worker_deadline = public_deadline - termination_reserve;
  const auto request = detail::encode_probe_request(config, timeout_ns);
  return detail::decode_probe_response(
      run_probe_worker_bounded(worker_path, request, worker_deadline, public_deadline,
                               supervisor_start_delay, pre_worker_spawn_delay));
}

} // namespace

GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config,
                              const std::filesystem::path& worker_path, std::uint64_t timeout_ns) {
  return probe_gpu_video_with_delays(config, worker_path, timeout_ns,
                                     std::chrono::nanoseconds::zero(),
                                     std::chrono::nanoseconds::zero());
}

GpuVideoProbe detail::probe_gpu_video_with_supervisor_start_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t supervisor_start_delay_ns) {
  return probe_gpu_video_with_delays(config, worker_path, timeout_ns,
                                     std::chrono::nanoseconds(supervisor_start_delay_ns),
                                     std::chrono::nanoseconds::zero());
}

GpuVideoProbe detail::probe_gpu_video_with_pre_worker_spawn_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_worker_spawn_delay_ns) {
  return probe_gpu_video_with_delays(config, worker_path, timeout_ns,
                                     std::chrono::nanoseconds::zero(),
                                     std::chrono::nanoseconds(pre_worker_spawn_delay_ns));
}

} // namespace reco::io
