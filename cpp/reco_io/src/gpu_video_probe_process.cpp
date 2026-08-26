#include "reco/io/gpu_video_probe.hpp"

#include "gpu_video_probe_protocol.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__linux__)
#include <dlfcn.h>
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
  StartupAttributeList() {
    SIZE_T size = 0;
    (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    if (size == 0) {
      throw GpuVideoProbeError("failed to size video probe worker launch attributes");
    }
    storage_.resize(size);
    value_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
    if (InitializeProcThreadAttributeList(value_, 1, 0, &size) == 0) {
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
                             std::chrono::steady_clock::time_point deadline) {
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

  StartupAttributeList attributes;
  std::array<HANDLE, 3> inherited_handles{child_stdin.get(), child_stdout.get(),
                                          child_stderr.get()};
  if (UpdateProcThreadAttribute(attributes.get(), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                inherited_handles.data(), sizeof(inherited_handles), nullptr,
                                nullptr) == 0) {
    throw GpuVideoProbeError("failed to restrict video probe worker inherited handles");
  }

  const auto encoded_path = worker_path.u8string();
  const std::string utf8_path(reinterpret_cast<const char*>(encoded_path.data()),
                              encoded_path.size());
  const auto application = utf8_to_wide(utf8_path);
  auto command_line = L"\"" + application + L"\" --reco-video-probe-worker";
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = child_stdin.get();
  startup.StartupInfo.hStdOutput = child_stdout.get();
  startup.StartupInfo.hStdError = child_stderr.get();
  PROCESS_INFORMATION process_info{};
  if (CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                     nullptr, &startup.StartupInfo, &process_info) == 0) {
    throw GpuVideoProbeError("failed to start video probe worker (Windows error " +
                             std::to_string(GetLastError()) + ")");
  }
  UniqueHandle process(process_info.hProcess);
  UniqueHandle thread(process_info.hThread);
  UniqueHandle job(CreateJobObjectW(nullptr, nullptr));
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!job ||
      SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits,
                              sizeof(limits)) == 0 ||
      AssignProcessToJobObject(job.get(), process.get()) == 0) {
    TerminateProcess(process.get(), 1);
    throw GpuVideoProbeError("failed to isolate video probe worker process");
  }
  if (ResumeThread(thread.get()) == std::numeric_limits<DWORD>::max()) {
    TerminateJobObject(job.get(), 1);
    throw GpuVideoProbeError("failed to resume video probe worker process");
  }
  thread.reset();
  child_stdin.reset();
  child_stdout.reset();
  child_stderr.reset();
  std::exception_ptr write_error;
  std::thread writer([input = std::move(parent_stdin), request, &write_error]() {
    try {
      write_request(input.get(), request);
    } catch (...) {
      write_error = std::current_exception();
    }
  });

  const auto terminate_and_join = [&] {
    if (TerminateJobObject(job.get(), 1) == 0) {
      job.reset();
    }
    (void)WaitForSingleObject(process.get(), INFINITE);
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
  job.reset();
  writer.join();
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

class ChildProcess {
public:
  explicit ChildProcess(pid_t value) : value_(value) {}
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ~ChildProcess() {
    if (value_ <= 0) {
      return;
    }
    (void)::kill(-value_, SIGKILL);
    int status = 0;
    while (::waitpid(value_, &status, 0) < 0 && errno == EINTR) {
    }
  }

  [[nodiscard]] pid_t get() const { return value_; }
  void release() { value_ = -1; }

private:
  pid_t value_ = -1;
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
  const auto trailing_size = ::recv(input, &trailing, 1, MSG_PEEK);
  if (trailing_size > 0) {
    throw GpuVideoProbeError("video probe worker response has trailing IPC bytes");
  }
  if (trailing_size < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
    throw GpuVideoProbeError("failed to inspect video probe worker response: " +
                             std::string(std::strerror(errno)));
  }
  return response;
}

std::string run_probe_worker(const std::filesystem::path& worker_path, std::string_view request,
                             std::chrono::steady_clock::time_point deadline) {
  int request_socket[2] = {-1, -1};
  int response_socket[2] = {-1, -1};
  create_socket_pair(request_socket, "input");
  UniqueFd child_input(request_socket[0]);
  UniqueFd parent_input(request_socket[1]);
  create_socket_pair(response_socket, "output");
  UniqueFd parent_output(response_socket[0]);
  UniqueFd child_output(response_socket[1]);
  make_nonblocking(parent_input.get());
  make_nonblocking(parent_output.get());
#if defined(SO_NOSIGPIPE) && !defined(MSG_NOSIGNAL)
  const int suppress_sigpipe = 1;
  if (::setsockopt(parent_input.get(), SOL_SOCKET, SO_NOSIGPIPE, &suppress_sigpipe,
                   sizeof(suppress_sigpipe)) != 0) {
    throw GpuVideoProbeError("failed to suppress SIGPIPE for video probe worker IPC: " +
                             std::string(std::strerror(errno)));
  }
#endif

  posix_spawn_file_actions_t actions;
  const auto actions_result = posix_spawn_file_actions_init(&actions);
  if (actions_result != 0) {
    throw GpuVideoProbeError("failed to initialize video probe worker launch: " +
                             std::string(std::strerror(actions_result)));
  }
  const auto destroy_actions = [&actions] { posix_spawn_file_actions_destroy(&actions); };
  auto action_error = posix_spawn_file_actions_adddup2(&actions, child_input.get(), STDIN_FILENO);
  if (action_error == 0) {
    action_error = posix_spawn_file_actions_adddup2(&actions, child_output.get(), STDOUT_FILENO);
  }
  bool closes_all_unrelated_descriptors = false;
#if defined(__linux__)
  using AddCloseFrom = int (*)(posix_spawn_file_actions_t*, int);
  const auto add_close_from = reinterpret_cast<AddCloseFrom>(
      ::dlsym(RTLD_DEFAULT, "posix_spawn_file_actions_addclosefrom_np"));
  if (action_error == 0 && add_close_from != nullptr) {
    action_error = add_close_from(&actions, STDERR_FILENO + 1);
    closes_all_unrelated_descriptors = action_error == 0;
  }
#endif
  if (!closes_all_unrelated_descriptors) {
    for (const int descriptor :
         {child_input.get(), parent_input.get(), parent_output.get(), child_output.get()}) {
      if (action_error == 0 && descriptor != STDIN_FILENO && descriptor != STDOUT_FILENO) {
        action_error = posix_spawn_file_actions_addclose(&actions, descriptor);
      }
    }
  }
  if (action_error != 0) {
    destroy_actions();
    throw GpuVideoProbeError("failed to configure video probe worker launch: " +
                             std::string(std::strerror(action_error)));
  }

  posix_spawnattr_t attributes;
  const auto attributes_result = posix_spawnattr_init(&attributes);
  if (attributes_result != 0) {
    destroy_actions();
    throw GpuVideoProbeError("failed to initialize video probe worker attributes: " +
                             std::string(std::strerror(attributes_result)));
  }
  const auto destroy_attributes = [&attributes] { posix_spawnattr_destroy(&attributes); };
  short spawn_flags = POSIX_SPAWN_SETPGROUP;
#if defined(POSIX_SPAWN_CLOEXEC_DEFAULT)
  spawn_flags = static_cast<short>(spawn_flags | POSIX_SPAWN_CLOEXEC_DEFAULT);
  closes_all_unrelated_descriptors = true;
#endif
  auto attribute_error = posix_spawnattr_setpgroup(&attributes, 0);
  if (attribute_error == 0) {
    attribute_error = posix_spawnattr_setflags(&attributes, spawn_flags);
  }
  if (attribute_error != 0) {
    destroy_attributes();
    destroy_actions();
    throw GpuVideoProbeError("failed to isolate video probe worker process group: " +
                             std::string(std::strerror(attribute_error)));
  }

  const auto executable = worker_path.string();
  const auto parent_pid_argument = std::to_string(static_cast<std::uint64_t>(::getpid())) +
                                   (closes_all_unrelated_descriptors ? ":1" : ":0");
  char* const arguments[] = {const_cast<char*>(executable.c_str()),
                             const_cast<char*>("--reco-video-probe-worker"),
                             const_cast<char*>(parent_pid_argument.c_str()), nullptr};
  pid_t pid = -1;
  const auto spawn_result =
      ::posix_spawn(&pid, executable.c_str(), &actions, &attributes, arguments, environ);
  destroy_attributes();
  destroy_actions();
  if (spawn_result != 0) {
    throw GpuVideoProbeError("failed to start video probe worker: " +
                             std::string(std::strerror(spawn_result)));
  }
  ChildProcess child(pid);
  child_input.reset();
  child_output.reset();
  write_request(parent_input.get(), request, deadline);
  parent_input.reset();

  int status = 0;
  bool has_wait_status = false;
  while (true) {
    const auto wait_result = ::waitpid(child.get(), &status, WNOHANG);
    if (wait_result == child.get()) {
      (void)::kill(-child.get(), SIGKILL);
      child.release();
      has_wait_status = true;
      break;
    }
    if (wait_result < 0 && errno == ECHILD) {
      child.release();
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      throw GpuVideoProbeError("failed while waiting for video probe worker: " +
                               std::string(std::strerror(errno)));
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      (void)::kill(-child.get(), SIGKILL);
      while (::waitpid(child.get(), &status, 0) < 0 && errno == EINTR) {
      }
      child.release();
      throw_worker_timeout();
    }
    std::this_thread::sleep_for(kProcessPollInterval);
  }
  if (has_wait_status && (!WIFEXITED(status) || WEXITSTATUS(status) != 0)) {
    throw GpuVideoProbeError("video probe worker exited abnormally");
  }
  return read_response(parent_output.get(), deadline);
}

#endif

std::string run_probe_worker_bounded(std::filesystem::path worker_path, std::string request,
                                     std::chrono::steady_clock::time_point worker_deadline,
                                     std::chrono::steady_clock::time_point public_deadline) {
  auto supervisor_slot = std::make_shared<SupervisorSlot>();
  std::promise<std::string> result;
  auto future = result.get_future();
  std::thread supervisor([supervisor_slot = std::move(supervisor_slot),
                          worker_path = std::move(worker_path), request = std::move(request),
                          result = std::move(result), worker_deadline]() mutable {
    try {
      result.set_value(run_probe_worker(worker_path, request, worker_deadline));
    } catch (...) {
      result.set_exception(std::current_exception());
    }
  });
  if (future.wait_until(public_deadline) != std::future_status::ready) {
    supervisor.detach();
    throw_worker_timeout();
  }
  supervisor.join();
  return future.get();
}

} // namespace

GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config,
                              const std::filesystem::path& worker_path, std::uint64_t timeout_ns) {
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
      run_probe_worker_bounded(worker_path, request, worker_deadline, public_deadline));
}

} // namespace reco::io
