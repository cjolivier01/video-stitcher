#include "reco/io/gpu_video_probe.hpp"

#include "gpu_video_probe_process_test.hpp"
#include "gpu_video_probe_protocol.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
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
#include <pthread.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <copyfile.h>
#include <dlfcn.h>
#include <libproc.h>
#include <sys/sysctl.h>
#include <sys/xattr.h>
#endif

#if defined(__linux__)
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/memfd.h>
#include <linux/seccomp.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif

extern char** environ;
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) ||                         \
    __has_feature(memory_sanitizer)
#define RECO_PROBE_WIDE_ADDRESS_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define RECO_PROBE_WIDE_ADDRESS_SANITIZER 1
#endif

namespace reco::io {
#if !defined(_WIN32)
namespace detail {
int run_gpu_video_probe_supervisor(const char* executable, std::uint64_t pre_worker_report_delay_ns,
                                   std::uint64_t pre_guardian_exec_delay_ns,
                                   std::uint64_t caller_pid, bool has_marker);
int run_gpu_video_probe_guardian(const char* executable, std::uint64_t pre_worker_report_delay_ns);
} // namespace detail
#endif
namespace {

constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr std::uint64_t kMinimumProbeTimeoutNs = kNanosecondsPerSecond;
constexpr std::uint64_t kMaximumProbeTimeoutNs = 3'600ULL * kNanosecondsPerSecond;
constexpr auto kProcessPollInterval = std::chrono::milliseconds(2);
constexpr auto kMinimumTerminationReserve = std::chrono::milliseconds(50);
constexpr auto kMaximumTerminationReserve = std::chrono::milliseconds(250);
constexpr std::uint64_t kMaximumWorkerAddressSpaceBytes = 512ULL * 1024ULL * 1024ULL;
// Reserve each admitted worker's full allowance and leave process memory for
// the caller, guardians, IPC, and non-probe application state.
constexpr std::uint64_t kMaximumAggregateWorkerAddressSpaceBytes =
    2ULL * 1024ULL * 1024ULL * 1024ULL;
static_assert(kMaximumAggregateWorkerAddressSpaceBytes % kMaximumWorkerAddressSpaceBytes == 0);
constexpr std::size_t kMaximumConcurrentProbeWorkers =
    kMaximumAggregateWorkerAddressSpaceBytes / kMaximumWorkerAddressSpaceBytes;

#if !defined(_WIN32)
const pid_t probe_process_generation = ::getpid();

void require_probe_process_generation() {
  if (::getpid() != probe_process_generation) {
    throw GpuVideoProbeError(
        "video probing is unavailable after fork without exec; exec before starting a probe");
  }
}
#else
void require_probe_process_generation() {}
#endif

struct ProbeLaunchOptions {
  std::chrono::nanoseconds supervisor_start_delay{};
  std::chrono::nanoseconds pre_worker_spawn_delay{};
  std::filesystem::path pre_supervisor_exec_marker;
  std::chrono::nanoseconds pre_worker_report_delay{};
  std::chrono::nanoseconds pre_guardian_exec_delay{};
  std::filesystem::path pre_guardian_exec_marker;
};

struct SupervisorSlots {
  std::mutex mutex;
  std::uint64_t reserved_worker_address_space_bytes = 0;
};

SupervisorSlots& supervisor_slots() {
  static auto* slots = new SupervisorSlots;
  return *slots;
}

class SupervisorSlot {
public:
  SupervisorSlot() {
    require_probe_process_generation();
    auto& slots = supervisor_slots();
    std::lock_guard lock(slots.mutex);
    if (slots.reserved_worker_address_space_bytes >
        kMaximumAggregateWorkerAddressSpaceBytes - kMaximumWorkerAddressSpaceBytes) {
      throw GpuVideoProbeError("the aggregate video probe worker memory budget is exhausted");
    }
    slots.reserved_worker_address_space_bytes += kMaximumWorkerAddressSpaceBytes;
  }
  SupervisorSlot(const SupervisorSlot&) = delete;
  SupervisorSlot& operator=(const SupervisorSlot&) = delete;
  ~SupervisorSlot() {
    auto& slots = supervisor_slots();
    std::lock_guard lock(slots.mutex);
    slots.reserved_worker_address_space_bytes -= kMaximumWorkerAddressSpaceBytes;
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

bool windows_job_is_empty(HANDLE job) {
  JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting{};
  return QueryInformationJobObject(job, JobObjectBasicAccountingInformation, &accounting,
                                   sizeof(accounting), nullptr) != 0 &&
         accounting.ActiveProcesses == 0;
}

bool wait_for_windows_job_empty(HANDLE job, std::chrono::steady_clock::time_point deadline) {
  while (true) {
    if (windows_job_is_empty(job)) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(kProcessPollInterval);
  }
}

class DeferredWindowsJobReaper {
public:
  DeferredWindowsJobReaper() {
    std::thread([this] { run(); }).detach();
  }
  DeferredWindowsJobReaper(const DeferredWindowsJobReaper&) = delete;
  DeferredWindowsJobReaper& operator=(const DeferredWindowsJobReaper&) = delete;

  [[nodiscard]] bool add(HANDLE job, std::shared_ptr<SupervisorSlot> reservation) noexcept {
    std::lock_guard lock(mutex_);
    for (auto& entry : jobs_) {
      if (entry.job == nullptr) {
        entry.job = job;
        entry.reservation = std::move(reservation);
        condition_.notify_one();
        return true;
      }
    }
    return false;
  }

private:
  struct DeferredJob {
    HANDLE job = nullptr;
    std::shared_ptr<SupervisorSlot> reservation;
  };

  void run() {
    std::unique_lock lock(mutex_);
    while (true) {
      condition_.wait(lock, [this] {
        return std::any_of(jobs_.begin(), jobs_.end(),
                           [](const auto& entry) { return entry.job != nullptr; });
      });
      bool has_jobs = false;
      for (auto& entry : jobs_) {
        if (entry.job == nullptr) {
          continue;
        }
        has_jobs = true;
        if (windows_job_is_empty(entry.job)) {
          (void)CloseHandle(entry.job);
          entry.job = nullptr;
          entry.reservation.reset();
        }
      }
      if (has_jobs) {
        condition_.wait_for(lock, kProcessPollInterval);
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  std::array<DeferredJob, kMaximumConcurrentProbeWorkers> jobs_{};
};

DeferredWindowsJobReaper& deferred_windows_job_reaper() {
  static auto* reaper = new DeferredWindowsJobReaper;
  return *reaper;
}

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
                             const ProbeLaunchOptions& options,
                             std::shared_ptr<SupervisorSlot> reservation) {
  (void)options.pre_worker_report_delay;
  (void)options.pre_guardian_exec_delay;
  (void)options.pre_guardian_exec_marker;
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
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
                                            JOB_OBJECT_LIMIT_PROCESS_MEMORY |
                                            JOB_OBJECT_LIMIT_JOB_MEMORY;
  limits.ProcessMemoryLimit = static_cast<SIZE_T>(kMaximumWorkerAddressSpaceBytes);
  limits.JobMemoryLimit = static_cast<SIZE_T>(kMaximumWorkerAddressSpaceBytes);
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
  startup.lpAttributeList = attributes.get();
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = child_stdin.get();
  startup.StartupInfo.hStdOutput = child_stdout.get();
  startup.StartupInfo.hStdError = child_stderr.get();
  PROCESS_INFORMATION process_info{};
  wait_for_worker_launch_delay(options.pre_worker_spawn_delay, deadline);
  if (CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                     nullptr, &startup.StartupInfo, &process_info) == 0) {
    throw GpuVideoProbeError("failed to start video probe worker (Windows error " +
                             std::to_string(GetLastError()) + ")");
  }
  UniqueHandle process(process_info.hProcess);
  UniqueHandle thread(process_info.hThread);
  const auto retire_job = [&] {
    if (!job) {
      return true;
    }
    if (wait_for_windows_job_empty(job.get(), cleanup_deadline)) {
      job.reset();
      reservation.reset();
      return true;
    }
    if (deferred_windows_job_reaper().add(job.get(), reservation)) {
      (void)job.release();
      reservation.reset();
    }
    return false;
  };
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
    (void)retire_job();
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
  if (job) {
    (void)TerminateJobObject(job.get(), 1);
    (void)retire_job();
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

pthread_mutex_t fork_descriptor_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_once_t fork_descriptor_once = PTHREAD_ONCE_INIT;
int fork_descriptor_registration_error = 0;

void lock_fork_descriptors() { (void)::pthread_mutex_lock(&fork_descriptor_mutex); }
void unlock_fork_descriptors() { (void)::pthread_mutex_unlock(&fork_descriptor_mutex); }

void register_fork_descriptor_handlers() {
  fork_descriptor_registration_error =
      ::pthread_atfork(lock_fork_descriptors, unlock_fork_descriptors, unlock_fork_descriptors);
}

class ForkDescriptorLock {
public:
  ForkDescriptorLock() {
    const auto once_error =
        ::pthread_once(&fork_descriptor_once, register_fork_descriptor_handlers);
    if (once_error != 0 || fork_descriptor_registration_error != 0) {
      const auto error = once_error != 0 ? once_error : fork_descriptor_registration_error;
      throw GpuVideoProbeError("failed to coordinate video probe descriptor creation: " +
                               std::string(std::strerror(error)));
    }
    const auto lock_error = ::pthread_mutex_lock(&fork_descriptor_mutex);
    if (lock_error != 0) {
      throw GpuVideoProbeError("failed to lock video probe descriptor creation: " +
                               std::string(std::strerror(lock_error)));
    }
    locked_ = true;
  }
  ForkDescriptorLock(const ForkDescriptorLock&) = delete;
  ForkDescriptorLock& operator=(const ForkDescriptorLock&) = delete;
  ~ForkDescriptorLock() {
    if (locked_) {
      (void)::pthread_mutex_unlock(&fork_descriptor_mutex);
    }
  }

private:
  bool locked_ = false;
};

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
  ForkDescriptorLock lock;
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
constexpr int kGuardianWorkerAuthority = 5;
constexpr int kGuardianExecutable = 6;
constexpr int kGuardianFirstUnusedDescriptor = 7;
constexpr int kWorkerExecutable = 3;
constexpr int kPreMainWatchdogFirstUnusedDescriptor = 5;
constexpr int kPreMainWatchdogTarget = 3;
constexpr int kPreMainWatchdogReady = 4;
constexpr int kSupervisorControl = 3;
constexpr int kSupervisorWorkerInput = 4;
constexpr int kSupervisorWorkerOutput = 5;
constexpr int kSupervisorMarker = 6;
constexpr int kSupervisorWatchdogReady = 7;
constexpr int kSupervisorExecutable = 8;
constexpr int kSupervisorFirstUnusedDescriptor = 9;
constexpr char kGuardianReady = 'R';
constexpr char kGuardianLaunch = 'L';
constexpr char kGuardianStarted = 'S';
constexpr char kGuardianRelease = 'G';
constexpr char kGuardianExited = 'E';
constexpr char kGuardianAcknowledge = 'A';

void execute_pinned_probe(int descriptor, char* const arguments[],
                          char* const environment[]) noexcept {
#if defined(__APPLE__)
  // Darwin has no fexecve/execveat equivalent. The caller therefore executes a private,
  // high-entropy, non-writable snapshot and preserves its security xattrs. Deliberate mutation by
  // another process running as the same UID is outside the OS boundary available to an
  // unprivileged CLI; such a process can also modify other user-owned deployment artifacts.
  (void)descriptor;
  ::execve(arguments[0], arguments, environment);
#else
  ::fexecve(descriptor, arguments, environment);
#endif
}
constexpr char kGuardianTerminate = 'T';
constexpr char kGuardianLaunchFailed = 'F';
constexpr char kSupervisorCertified = 'C';
constexpr char kWorkerGroupCertified = 'C';

long descriptor_scan_limit();
[[noreturn]] void guardian_exit(int status);
bool guardian_close_from(int descriptor, long maximum_descriptor);
bool wait_for_process_exit(pid_t pid, int* status, std::chrono::steady_clock::time_point deadline);

struct GuardianLaunch {
  pid_t supervisor_pid = -1;
  pid_t watchdog_pid = -1;
  UniqueFd caller_lifetime;
};

GuardianLaunch spawn_guardian_process(const std::string& executable, int executable_descriptor,
                                      int control_descriptor, int worker_input_descriptor,
                                      int worker_output_descriptor,
                                      std::chrono::nanoseconds pre_worker_report_delay,
                                      std::chrono::nanoseconds pre_guardian_exec_delay,
                                      const std::filesystem::path& pre_guardian_exec_marker,
                                      std::shared_ptr<SupervisorSlot>* reservation);

bool guardian_watchdog_exit_is_fatal(bool memory_termination_sent, int wait_result, int wait_error,
                                     pid_t observed_pid, pid_t watchdog_pid) {
  return !memory_termination_sent && ((wait_result == 0 && observed_pid == watchdog_pid) ||
                                      (wait_result < 0 && wait_error != EINTR));
}

UniqueFd duplicate_close_on_exec(int descriptor, int minimum_descriptor) {
#if defined(F_DUPFD_CLOEXEC)
  const auto duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, minimum_descriptor);
#else
  ForkDescriptorLock lock;
  const auto duplicate = ::fcntl(descriptor, F_DUPFD, minimum_descriptor);
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

UniqueFd duplicate_for_guardian(int descriptor) {
  return duplicate_close_on_exec(descriptor, kGuardianFirstUnusedDescriptor);
}

UniqueFd duplicate_for_supervisor(int descriptor) {
  return duplicate_close_on_exec(descriptor, kSupervisorFirstUnusedDescriptor);
}

UniqueFd open_probe_executable(const std::filesystem::path& path) {
  int flags = O_CLOEXEC;
#if defined(__linux__) && defined(O_PATH)
  flags |= O_RDONLY;
#elif defined(__APPLE__)
  flags |= O_RDONLY;
#elif defined(O_EXEC)
  flags |= O_EXEC;
#else
  flags |= O_RDONLY;
#endif
  UniqueFd executable(::open(path.c_str(), flags));
  struct stat status{};
  if (executable.get() < 0) {
    throw GpuVideoProbeError("failed to start video probe worker: cannot open executable: " +
                             std::string(std::strerror(errno)));
  }
  if (::fstat(executable.get(), &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
      (status.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0 ||
      (status.st_mode & (S_ISUID | S_ISGID)) != 0) {
    throw GpuVideoProbeError(
        "failed to start video probe worker: executable must be a non-set-id regular file");
  }
  return executable;
}

#if defined(__linux__)
UniqueFd snapshot_linux_probe_executable(int source) {
#if !defined(SYS_memfd_create)
  (void)source;
  throw GpuVideoProbeError("sealed Linux video probe snapshots require memfd_create");
#else
  struct stat before{};
  constexpr std::uint64_t maximum_snapshot_bytes = 256ULL * 1024ULL * 1024ULL;
  if (::fstat(source, &before) != 0 || before.st_size <= 0 ||
      static_cast<std::uint64_t>(before.st_size) > maximum_snapshot_bytes) {
    throw GpuVideoProbeError("failed to snapshot Linux video probe worker: invalid size");
  }
  constexpr auto base_flags = static_cast<unsigned int>(MFD_CLOEXEC | MFD_ALLOW_SEALING);
  int raw_snapshot = -1;
#if defined(MFD_EXEC)
  raw_snapshot =
      static_cast<int>(::syscall(SYS_memfd_create, "reco-video-probe", base_flags | MFD_EXEC));
  if (raw_snapshot < 0 && errno == EINVAL) {
    raw_snapshot = static_cast<int>(::syscall(SYS_memfd_create, "reco-video-probe", base_flags));
  }
#else
  raw_snapshot = static_cast<int>(::syscall(SYS_memfd_create, "reco-video-probe", base_flags));
#endif
  UniqueFd snapshot(raw_snapshot);
  if (snapshot.get() < 0) {
    throw GpuVideoProbeError("failed to create sealed Linux video probe snapshot: " +
                             std::string(std::strerror(errno)));
  }

  std::array<char, 64U * 1024U> buffer{};
  off_t offset = 0;
  while (offset < before.st_size) {
    const auto remaining = static_cast<std::uint64_t>(before.st_size - offset);
    const auto requested = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(buffer.size())));
    ssize_t received = -1;
    do {
      received = ::pread(source, buffer.data(), requested, offset);
    } while (received < 0 && errno == EINTR);
    if (received <= 0) {
      throw GpuVideoProbeError("failed to read Linux video probe executable snapshot");
    }
    std::size_t written = 0;
    while (written < static_cast<std::size_t>(received)) {
      ssize_t count = -1;
      do {
        count = ::write(snapshot.get(), buffer.data() + written,
                        static_cast<std::size_t>(received) - written);
      } while (count < 0 && errno == EINTR);
      if (count <= 0) {
        throw GpuVideoProbeError("failed to write Linux video probe executable snapshot");
      }
      written += static_cast<std::size_t>(count);
    }
    offset += received;
  }

  struct stat after{};
  struct stat snapshot_status{};
  if (::fstat(source, &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec ||
      ::fstat(snapshot.get(), &snapshot_status) != 0 || snapshot_status.st_size != before.st_size) {
    throw GpuVideoProbeError("Linux video probe executable changed while it was snapshotted");
  }
  int seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
#if defined(F_SEAL_FUTURE_WRITE)
  seals |= F_SEAL_FUTURE_WRITE;
#endif
  if (::fchmod(snapshot.get(), 0500) != 0 || ::fcntl(snapshot.get(), F_ADD_SEALS, seals) != 0 ||
      (::fcntl(snapshot.get(), F_GET_SEALS) & seals) != seals) {
    throw GpuVideoProbeError("failed to seal Linux video probe executable snapshot: " +
                             std::string(std::strerror(errno)));
  }
  return snapshot;
#endif
}
#endif

#if defined(__APPLE__)
class MacProbeExecutableSnapshot final {
public:
  MacProbeExecutableSnapshot(int source, std::chrono::steady_clock::time_point deadline) {
    try {
      struct stat before{};
      constexpr std::uint64_t maximum_snapshot_bytes = 256ULL * 1024ULL * 1024ULL;
      if (::fstat(source, &before) != 0 || before.st_size <= 0 ||
          static_cast<std::uint64_t>(before.st_size) > maximum_snapshot_bytes) {
        throw GpuVideoProbeError(
            "failed to snapshot macOS video probe worker: invalid executable size");
      }

      const auto temporary_directory = std::filesystem::temp_directory_path();
      std::random_device random;
      constexpr char hexadecimal[] = "0123456789abcdef";
      int setup_error = EEXIST;
      for (int attempt = 0; attempt < 128 && setup_error == EEXIST; ++attempt) {
        std::string token(32U, '0');
        for (auto& digit : token) {
          digit = hexadecimal[random() & 0x0fU];
        }
        root_ = temporary_directory / ("reco-video-probe-" + token);
        executable_ = root_ / "probe-worker";
        setup_error = start_cleanup_helper(deadline);
        if (setup_error == EEXIST) {
          root_.clear();
          executable_.clear();
        }
      }
      if (setup_error != 0) {
        throw GpuVideoProbeError("failed to create macOS video probe snapshot directory: " +
                                 std::string(std::strerror(setup_error)));
      }
      UniqueFd output(
          ::open(executable_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
      if (output.get() < 0) {
        throw GpuVideoProbeError("failed to create macOS video probe snapshot: " +
                                 std::string(std::strerror(errno)));
      }

      constexpr std::size_t maximum_xattr_names_bytes = 1024U * 1024U;
      constexpr std::uint64_t maximum_xattr_value_bytes = 16ULL * 1024ULL * 1024ULL;
      const auto xattr_names_size = ::flistxattr(source, nullptr, 0, 0);
      if (xattr_names_size < 0 ||
          static_cast<std::uint64_t>(xattr_names_size) > maximum_xattr_names_bytes) {
        throw GpuVideoProbeError("failed to bound macOS video probe executable metadata");
      }
      std::vector<char> xattr_names(static_cast<std::size_t>(xattr_names_size));
      if (xattr_names_size > 0 &&
          ::flistxattr(source, xattr_names.data(), xattr_names.size(), 0) != xattr_names_size) {
        throw GpuVideoProbeError("failed to read macOS video probe executable metadata");
      }
      std::uint64_t xattr_value_bytes = 0;
      std::size_t name_offset = 0;
      while (name_offset < xattr_names.size()) {
        const auto remaining = xattr_names.size() - name_offset;
        const auto name_size = ::strnlen(xattr_names.data() + name_offset, remaining);
        if (name_size == 0 || name_size == remaining) {
          throw GpuVideoProbeError("macOS video probe executable metadata is malformed");
        }
        const auto value_size =
            ::fgetxattr(source, xattr_names.data() + name_offset, nullptr, 0, 0, 0);
        if (value_size < 0 || static_cast<std::uint64_t>(value_size) >
                                  maximum_xattr_value_bytes - xattr_value_bytes) {
          throw GpuVideoProbeError("macOS video probe executable metadata exceeds its limit");
        }
        xattr_value_bytes += static_cast<std::uint64_t>(value_size);
        name_offset += name_size + 1U;
      }
      if (::fcopyfile(source, output.get(), nullptr, COPYFILE_DATA | COPYFILE_XATTR) != 0) {
        throw GpuVideoProbeError("failed to copy macOS video probe executable snapshot: " +
                                 std::string(std::strerror(errno)));
      }

      struct stat after{};
      struct stat snapshot_status{};
      if (::fstat(source, &after) != 0 || before.st_dev != after.st_dev ||
          before.st_ino != after.st_ino || before.st_size != after.st_size ||
          before.st_mtimespec.tv_sec != after.st_mtimespec.tv_sec ||
          before.st_mtimespec.tv_nsec != after.st_mtimespec.tv_nsec ||
          before.st_ctimespec.tv_sec != after.st_ctimespec.tv_sec ||
          before.st_ctimespec.tv_nsec != after.st_ctimespec.tv_nsec ||
          ::fstat(output.get(), &snapshot_status) != 0 ||
          snapshot_status.st_size != before.st_size) {
        throw GpuVideoProbeError("macOS video probe executable changed while it was snapshotted");
      }
      if (::fsync(output.get()) != 0 || ::fchmod(output.get(), 0500) != 0 ||
          ::chmod(root_.c_str(), 0500) != 0) {
        throw GpuVideoProbeError("failed to seal macOS video probe executable snapshot: " +
                                 std::string(std::strerror(errno)));
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  MacProbeExecutableSnapshot(const MacProbeExecutableSnapshot&) = delete;
  MacProbeExecutableSnapshot& operator=(const MacProbeExecutableSnapshot&) = delete;
  ~MacProbeExecutableSnapshot() { cleanup(); }

  [[nodiscard]] const std::filesystem::path& executable() const { return executable_; }

private:
  [[nodiscard]] int start_cleanup_helper(std::chrono::steady_clock::time_point deadline) {
    int lifetime_descriptors[2] = {-1, -1};
    create_socket_pair(lifetime_descriptors, "macOS snapshot cleanup");
    UniqueFd helper_lifetime(lifetime_descriptors[0]);
    UniqueFd owner_lifetime(lifetime_descriptors[1]);
    const int suppress_sigpipe = 1;
    if (::setsockopt(helper_lifetime.get(), SOL_SOCKET, SO_NOSIGPIPE, &suppress_sigpipe,
                     sizeof(suppress_sigpipe)) != 0) {
      throw GpuVideoProbeError("failed to suppress macOS snapshot cleanup SIGPIPE: " +
                               std::string(std::strerror(errno)));
    }
    const auto maximum_descriptor = descriptor_scan_limit();
    const auto helper = ::fork();
    const auto fork_error = errno;
    if (helper == 0) {
      (void)::close(owner_lifetime.get());
      if (::dup2(helper_lifetime.get(), STDIN_FILENO) < 0 ||
          !guardian_close_from(STDOUT_FILENO, maximum_descriptor)) {
        guardian_exit(127);
      }
      const int setup_error = ::mkdir(root_.c_str(), 0700) == 0 ? 0 : errno;
      std::size_t offset = 0;
      while (offset < sizeof(setup_error)) {
        ssize_t written = -1;
        do {
          written = ::write(STDIN_FILENO, reinterpret_cast<const char*>(&setup_error) + offset,
                            sizeof(setup_error) - offset);
        } while (written < 0 && errno == EINTR);
        if (written <= 0) {
          if (setup_error == 0) {
            (void)::rmdir(root_.c_str());
          }
          guardian_exit(127);
        }
        offset += static_cast<std::size_t>(written);
      }
      if (setup_error != 0) {
        guardian_exit(0);
      }
      char lifetime = '\0';
      while (true) {
        const auto received = ::read(STDIN_FILENO, &lifetime, 1);
        if (received < 0 && errno == EINTR) {
          continue;
        }
        if (received <= 0) {
          break;
        }
      }
      (void)::chmod(root_.c_str(), 0700);
      (void)::unlink(executable_.c_str());
      (void)::rmdir(root_.c_str());
      guardian_exit(0);
    }
    if (helper < 0) {
      errno = fork_error;
      throw GpuVideoProbeError("failed to start macOS video probe snapshot cleanup: " +
                               std::string(std::strerror(errno)));
    }
    helper_lifetime.reset();
    cleanup_pid_ = helper;
    cleanup_lifetime_ = std::move(owner_lifetime);
    int setup_error = 0;
    try {
      read_exact(cleanup_lifetime_.get(), reinterpret_cast<char*>(&setup_error),
                 sizeof(setup_error), deadline);
    } catch (...) {
      stop_cleanup_helper();
      throw;
    }
    if (setup_error != 0) {
      stop_cleanup_helper();
      return setup_error;
    }
    helper_owns_root_ = true;
    return 0;
  }

  void stop_cleanup_helper() noexcept {
    cleanup_lifetime_.reset();
    if (cleanup_pid_ <= 0) {
      return;
    }
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    if (!wait_for_process_exit(cleanup_pid_, &status, deadline)) {
      (void)::kill(cleanup_pid_, SIGKILL);
      while (::waitpid(cleanup_pid_, &status, 0) < 0 && errno == EINTR) {
      }
    }
    cleanup_pid_ = -1;
  }

  void cleanup() noexcept {
    stop_cleanup_helper();
    if (helper_owns_root_ && !root_.empty()) {
      (void)::chmod(root_.c_str(), 0700);
      (void)::unlink(executable_.c_str());
      (void)::rmdir(root_.c_str());
    }
    helper_owns_root_ = false;
    executable_.clear();
    root_.clear();
  }

  std::filesystem::path root_;
  std::filesystem::path executable_;
  pid_t cleanup_pid_ = -1;
  UniqueFd cleanup_lifetime_;
  bool helper_owns_root_ = false;
};
#endif

void kill_worker_process_group(pid_t worker_pid) {
  if (worker_pid <= 0) {
    return;
  }
  (void)::kill(-worker_pid, SIGKILL);
  (void)::kill(worker_pid, SIGKILL);
}

bool worker_process_group_exists(pid_t worker_pid) {
  if (worker_pid <= 0) {
    return false;
  }
  errno = 0;
  return ::kill(-worker_pid, 0) == 0 || errno != ESRCH;
}

bool take_supervisor_certificate(UniqueFd* lifetime) noexcept {
  if (lifetime == nullptr || lifetime->get() < 0) {
    return false;
  }
  char certificate = '\0';
  ssize_t received = -1;
  do {
    received = ::recv(lifetime->get(), &certificate, 1, 0);
  } while (received < 0 && errno == EINTR);
  lifetime->reset();
  return received == 1 && certificate == kSupervisorCertified;
}

class DeferredProcessReaper {
public:
  DeferredProcessReaper() {
    std::thread([this] { run(); }).detach();
  }
  DeferredProcessReaper(const DeferredProcessReaper&) = delete;
  DeferredProcessReaper& operator=(const DeferredProcessReaper&) = delete;

  [[nodiscard]] bool add(pid_t pid, std::shared_ptr<SupervisorSlot>&& reservation,
                         UniqueFd&& certification = UniqueFd{}) noexcept {
    std::lock_guard lock(mutex_);
    for (auto& process : processes_) {
      if (process.pid <= 0 && !process.reservation) {
        process.pid = pid;
        process.reservation = std::move(reservation);
        process.certification = std::move(certification);
        condition_.notify_one();
        return true;
      }
    }
    return false;
  }

private:
  struct DeferredProcess {
    pid_t pid = -1;
    std::shared_ptr<SupervisorSlot> reservation;
    UniqueFd certification;
  };

  void run() {
    std::unique_lock lock(mutex_);
    while (true) {
      condition_.wait(lock, [this] {
        return std::any_of(processes_.begin(), processes_.end(),
                           [](const auto& process) { return process.pid > 0; });
      });
      bool has_processes = false;
      for (auto& item : processes_) {
        if (item.pid <= 0) {
          continue;
        }
        has_processes = true;
        int status = 0;
        const auto result = ::waitpid(item.pid, &status, WNOHANG);
        if (result == item.pid || (result < 0 && errno == ECHILD)) {
          if (!item.reservation || take_supervisor_certificate(&item.certification)) {
            item.reservation.reset();
          } else {
            item.certification.reset();
          }
          item.pid = -1;
        }
      }
      if (has_processes) {
        condition_.wait_for(lock, kProcessPollInterval);
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  // Four admitted probes can each defer a supervisor and its pre-main
  // watchdog. Fixed storage keeps timeout cleanup independent of allocation.
  std::array<DeferredProcess, kMaximumConcurrentProbeWorkers * 2U> processes_{};
};

DeferredProcessReaper& deferred_process_reaper() {
  static auto* reaper = new DeferredProcessReaper;
  return *reaper;
}

void retain_uncertified_reservation(std::shared_ptr<SupervisorSlot>&& reservation) noexcept {
  if (!reservation || deferred_process_reaper().add(-1, std::move(reservation))) {
    return;
  }
  static std::array<std::shared_ptr<SupervisorSlot>, kMaximumConcurrentProbeWorkers>
      retained_reservations;
  static std::mutex retained_mutex;
  std::lock_guard lock(retained_mutex);
  const auto empty = std::find(retained_reservations.begin(), retained_reservations.end(),
                               std::shared_ptr<SupervisorSlot>{});
  if (empty != retained_reservations.end()) {
    *empty = std::move(reservation);
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

[[noreturn]] void report_guardian_startup_error(int error) {
  const auto reported_error = error == 0 ? EIO : error;
  (void)guardian_write(STDOUT_FILENO, &kGuardianLaunchFailed, 1);
  (void)guardian_write(STDOUT_FILENO, &reported_error, sizeof(reported_error));
  guardian_exit(2);
}

bool guardian_pipe_write(int descriptor, char value) {
  ssize_t written = -1;
  do {
    written = ::write(descriptor, &value, 1);
  } while (written < 0 && errno == EINTR);
  return written == 1;
}

bool guardian_pipe_write_process_id(int descriptor, pid_t value) {
  ssize_t written = -1;
  do {
    written = ::write(descriptor, &value, sizeof(value));
  } while (written < 0 && errno == EINTR);
  return written == static_cast<ssize_t>(sizeof(value));
}

bool guardian_sleep(std::chrono::nanoseconds delay) {
  if (delay.count() <= 0) {
    return true;
  }
  constexpr auto kNanosecondsPerSecond = 1'000'000'000LL;
  timespec remaining{.tv_sec = static_cast<time_t>(delay.count() / kNanosecondsPerSecond),
                     .tv_nsec = static_cast<long>(delay.count() % kNanosecondsPerSecond)};
  while (::nanosleep(&remaining, &remaining) != 0) {
    if (errno != EINTR) {
      return false;
    }
  }
  return true;
}

bool guardian_close_from(int descriptor, long maximum_descriptor) {
#if defined(__linux__) && defined(SYS_getdents64)
  (void)maximum_descriptor;
#endif
#if defined(__linux__) && defined(SYS_close_range) && !defined(RECO_GUARDIAN_FORCE_DESCRIPTOR_SCAN)
  if (::syscall(SYS_close_range, static_cast<unsigned int>(descriptor),
                std::numeric_limits<unsigned int>::max(), 0U) == 0) {
    return true;
  }
#endif
#if defined(F_CLOSEM)
  if (::fcntl(descriptor, F_CLOSEM, 0) == 0) {
    return true;
  }
#endif
#if defined(__linux__) && defined(SYS_getdents64)
  const auto directory = ::open("/proc/self/fd", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) {
    return false;
  }
  struct LinuxDirectoryEntry {
    std::uint64_t inode;
    std::int64_t offset;
    unsigned short record_length;
    unsigned char type;
    char name[1];
  };
  alignas(std::uint64_t) std::array<char, 4096> entries{};
  while (true) {
    const auto count = ::syscall(SYS_getdents64, directory, entries.data(), entries.size());
    if (count == 0) {
      (void)::close(directory);
      return true;
    }
    if (count < 0) {
      const auto saved_error = errno;
      (void)::close(directory);
      errno = saved_error;
      return false;
    }
    std::size_t offset = 0;
    while (offset < static_cast<std::size_t>(count)) {
      const auto* entry = reinterpret_cast<const LinuxDirectoryEntry*>(entries.data() + offset);
      constexpr auto name_offset = offsetof(LinuxDirectoryEntry, name);
      if (entry->record_length <= name_offset ||
          entry->record_length > static_cast<std::size_t>(count) - offset) {
        (void)::close(directory);
        errno = EIO;
        return false;
      }
      int value = 0;
      bool numeric = entry->name[0] != '\0';
      for (std::size_t index = 0; index < entry->record_length - name_offset; ++index) {
        const auto character = entry->name[index];
        if (character == '\0') {
          break;
        }
        if (character < '0' || character > '9' ||
            value > (std::numeric_limits<int>::max() - (character - '0')) / 10) {
          numeric = false;
          break;
        }
        value = value * 10 + (character - '0');
      }
      if (numeric && value >= descriptor && value != directory) {
        (void)::close(value);
      }
      offset += entry->record_length;
    }
  }
#else
  for (long value = descriptor; value < maximum_descriptor; ++value) {
    (void)::close(static_cast<int>(value));
  }
  return true;
#endif
}

long descriptor_scan_limit() {
  long maximum = ::sysconf(_SC_OPEN_MAX);
  struct rlimit limits{};
  if (::getrlimit(RLIMIT_NOFILE, &limits) == 0 && limits.rlim_max != RLIM_INFINITY &&
      limits.rlim_max <= static_cast<rlim_t>(std::numeric_limits<long>::max())) {
    maximum = std::max(maximum, static_cast<long>(limits.rlim_max));
  }
#if defined(__APPLE__)
  int kernel_maximum = 0;
  std::size_t size = sizeof(kernel_maximum);
  if (::sysctlbyname("kern.maxfilesperproc", &kernel_maximum, &size, nullptr, 0) == 0 &&
      kernel_maximum > 0) {
    maximum = std::max(maximum, static_cast<long>(kernel_maximum));
  }
#endif
#if defined(__linux__) && defined(SYS_getdents64)
  // Linux closes the actual /proc/self/fd set when close_range is unavailable,
  // so the numeric fallback bound is irrelevant on this path.
  maximum = std::max(maximum, static_cast<long>(kGuardianFirstUnusedDescriptor));
#endif
  return maximum;
}

bool guardian_limit_worker_address_space() {
#if defined(__APPLE__)
  // Darwin rejects RLIMIT_DATA values below the process's existing VM map,
  // which is commonly larger than this worker budget before exec. The
  // process-group physical-footprint watchdog below remains enforceable.
  return true;
#elif defined(RECO_PROBE_WIDE_ADDRESS_SANITIZER)
  // Wide-address sanitizers reserve terabytes of shadow virtual memory before main.
  return true;
#else
  struct rlimit current{};
  if (::getrlimit(RLIMIT_AS, &current) != 0) {
    return false;
  }
  const auto requested = static_cast<rlim_t>(kMaximumWorkerAddressSpaceBytes);
  const auto existing = current.rlim_cur == RLIM_INFINITY ? requested : current.rlim_cur;
  const auto limit = std::min(existing, requested);
  if (limit == 0) {
    return false;
  }
  const struct rlimit bounded{.rlim_cur = limit, .rlim_max = limit};
  return ::setrlimit(RLIMIT_AS, &bounded) == 0;
#endif
}

bool guardian_restrict_worker_process_creation() {
#if defined(__linux__)
#if defined(__x86_64__)
  constexpr std::uint32_t kAuditArchitecture = AUDIT_ARCH_X86_64;
  constexpr std::uint32_t kX32SystemCallBit = 0x40000000U;
  constexpr std::size_t kFilterInstructionCount = 22;
#elif defined(__aarch64__)
  constexpr std::uint32_t kAuditArchitecture = AUDIT_ARCH_AARCH64;
  constexpr std::size_t kFilterInstructionCount = 20;
#else
  return false;
#endif
#if defined(SYS_fork)
  constexpr std::uint32_t kForkSystemCall = SYS_fork;
#else
  constexpr std::uint32_t kForkSystemCall = std::numeric_limits<std::uint32_t>::max();
#endif
#if defined(SYS_vfork)
  constexpr std::uint32_t kVforkSystemCall = SYS_vfork;
#else
  constexpr std::uint32_t kVforkSystemCall = std::numeric_limits<std::uint32_t>::max();
#endif
#if defined(SYS_clone3)
  constexpr std::uint32_t kClone3SystemCall = SYS_clone3;
#else
  constexpr std::uint32_t kClone3SystemCall = 435U;
#endif
  constexpr std::uint32_t kDenied = SECCOMP_RET_ERRNO | static_cast<std::uint32_t>(EPERM);
  constexpr std::uint32_t kUnavailable = SECCOMP_RET_ERRNO | static_cast<std::uint32_t>(ENOSYS);
  const std::array<sock_filter, kFilterInstructionCount> filter{{
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kAuditArchitecture, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
#if defined(__x86_64__)
      BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, kX32SystemCallBit, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
#endif
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kForkSystemCall, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, kDenied),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kVforkSystemCall, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, kDenied),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kClone3SystemCall, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, kUnavailable),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_setsid, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, kDenied),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_setpgid, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, kDenied),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone, 0, 4),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, CLONE_THREAD),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, CLONE_THREAD, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, kDenied),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  }};
  sock_fprog program{.len = static_cast<unsigned short>(filter.size()),
                     .filter = const_cast<sock_filter*>(filter.data())};
  return ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
         ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
#elif defined(__APPLE__)
  using SandboxInit = int (*)(const char*, std::uint64_t, char**);
  using SandboxFreeError = void (*)(char*);
  static const auto sandbox_init =
      reinterpret_cast<SandboxInit>(::dlsym(RTLD_DEFAULT, "sandbox_init"));
  static const auto sandbox_free_error =
      reinterpret_cast<SandboxFreeError>(::dlsym(RTLD_DEFAULT, "sandbox_free_error"));
  if (sandbox_init == nullptr) {
    return false;
  }
  constexpr const char* kProfile = "(version 1)\n(allow default)\n(deny process-fork)\n";
  char* error = nullptr;
  const auto result = sandbox_init(kProfile, 0, &error);
  if (error != nullptr && sandbox_free_error != nullptr) {
    sandbox_free_error(error);
  }
  if (result == 0) {
    return true;
  }
  if (::geteuid() == 0) {
    return false;
  }
  constexpr struct rlimit no_child_processes{.rlim_cur = 0, .rlim_max = 0};
  return ::setrlimit(RLIMIT_NPROC, &no_child_processes) == 0;
#else
  return false;
#endif
}

#if defined(__linux__)
struct LinuxProcessUsage {
  pid_t process_group = -1;
  std::uint64_t resident_pages = 0;
};

bool parse_linux_process_usage(const char* data, std::size_t size, LinuxProcessUsage* usage) {
  const char* close_parenthesis = nullptr;
  for (const char* cursor = data + size; cursor != data;) {
    --cursor;
    if (*cursor == ')') {
      close_parenthesis = cursor;
      break;
    }
  }
  if (close_parenthesis == nullptr || close_parenthesis + 3 >= data + size ||
      close_parenthesis[1] != ' ') {
    return false;
  }
  const char* cursor = close_parenthesis + 2;
  ++cursor; // state, field 3
  if (cursor >= data + size || *cursor != ' ') {
    return false;
  }
  ++cursor;
  std::int64_t process_group = -1;
  std::int64_t resident_pages = -1;
  for (int field = 4; field <= 24; ++field) {
    const char* end = cursor;
    while (end < data + size && *end != ' ' && *end != '\n') {
      ++end;
    }
    std::int64_t value = 0;
    const auto [parsed_end, error] = std::from_chars(cursor, end, value);
    if (error != std::errc{} || parsed_end != end) {
      return false;
    }
    if (field == 5) {
      process_group = value;
    } else if (field == 24) {
      resident_pages = value;
    }
    cursor = end;
    while (cursor < data + size && *cursor == ' ') {
      ++cursor;
    }
  }
  if (process_group <= 0 || process_group > std::numeric_limits<pid_t>::max() ||
      resident_pages < 0) {
    return false;
  }
  usage->process_group = static_cast<pid_t>(process_group);
  usage->resident_pages = static_cast<std::uint64_t>(resident_pages);
  return true;
}

struct LinuxWorkerGroupScan {
  bool valid = false;
  bool has_other_members = false;
  std::uint64_t resident_bytes = 0;
};

LinuxWorkerGroupScan scan_linux_worker_group(pid_t worker_pid) {
#if !defined(SYS_getdents64)
  (void)worker_pid;
  return {};
#else
  LinuxWorkerGroupScan result{};
  const auto directory = ::open("/proc", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) {
    return result;
  }
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    (void)::close(directory);
    return result;
  }
  struct LinuxDirectoryEntry {
    std::uint64_t inode;
    std::int64_t offset;
    unsigned short record_length;
    unsigned char type;
    char name[1];
  };
  alignas(std::uint64_t) std::array<char, 8192> entries{};
  bool valid = true;
  while (valid) {
    const auto count = ::syscall(SYS_getdents64, directory, entries.data(), entries.size());
    if (count == 0) {
      break;
    }
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      valid = false;
      break;
    }
    std::size_t offset = 0;
    while (offset < static_cast<std::size_t>(count)) {
      const auto* entry = reinterpret_cast<const LinuxDirectoryEntry*>(entries.data() + offset);
      constexpr auto name_offset = offsetof(LinuxDirectoryEntry, name);
      if (entry->record_length <= name_offset ||
          entry->record_length > static_cast<std::size_t>(count) - offset) {
        valid = false;
        break;
      }
      bool numeric = entry->name[0] != '\0';
      bool terminated = false;
      std::size_t name_length = 0;
      for (std::size_t index = 0; index < entry->record_length - name_offset; ++index) {
        const auto character = entry->name[index];
        if (character == '\0') {
          terminated = true;
          break;
        }
        if (character < '0' || character > '9') {
          numeric = false;
          break;
        }
        ++name_length;
      }
      if (numeric && terminated) {
        std::uint64_t encoded_process_id = 0;
        const auto [id_end, id_error] =
            std::from_chars(entry->name, entry->name + name_length, encoded_process_id);
        if (id_error != std::errc{} || id_end != entry->name + name_length ||
            encoded_process_id == 0 ||
            encoded_process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
          valid = false;
          break;
        }
        const auto process_id = static_cast<pid_t>(encoded_process_id);
        errno = 0;
        const auto process_group = ::getpgid(process_id);
        if (process_group < 0 && (errno == ENOENT || errno == ESRCH || errno == EPERM)) {
          offset += entry->record_length;
          continue;
        }
        if (process_group < 0) {
          valid = false;
          break;
        }
        if (process_group != worker_pid) {
          offset += entry->record_length;
          continue;
        }
        char relative_path[64]{};
        if (name_length + 6U >= sizeof(relative_path)) {
          valid = false;
          break;
        }
        std::memcpy(relative_path, entry->name, name_length);
        std::memcpy(relative_path + name_length, "/stat", 6U);
        const auto process_stat = ::openat(directory, relative_path, O_RDONLY | O_CLOEXEC);
        if (process_stat >= 0) {
          std::array<char, 4096> stat{};
          ssize_t stat_size = -1;
          do {
            stat_size = ::read(process_stat, stat.data(), stat.size());
          } while (stat_size < 0 && errno == EINTR);
          const auto read_error = errno;
          (void)::close(process_stat);
          if (stat_size > 0) {
            LinuxProcessUsage usage{};
            if (!parse_linux_process_usage(stat.data(), static_cast<std::size_t>(stat_size),
                                           &usage)) {
              valid = false;
              break;
            }
            if (usage.process_group != worker_pid) {
              offset += entry->record_length;
              continue;
            }
            result.has_other_members |= process_id != worker_pid;
            const auto page_bytes = static_cast<std::uint64_t>(page_size);
            if (usage.resident_pages >
                (std::numeric_limits<std::uint64_t>::max() - result.resident_bytes) / page_bytes) {
              valid = false;
              break;
            }
            result.resident_bytes += usage.resident_pages * page_bytes;
          }
          if (stat_size < 0 && read_error != ENOENT && read_error != ESRCH) {
            valid = false;
            break;
          }
        } else if (errno != ENOENT && errno != ESRCH) {
          valid = false;
          break;
        }
      }
      offset += entry->record_length;
    }
  }
  const auto close_error = ::close(directory);
  result.valid = valid && close_error == 0;
  return result;
#endif
}

bool guardian_worker_group_within_memory_limit(pid_t worker_pid) {
  const auto scan = scan_linux_worker_group(worker_pid);
  return scan.valid && scan.resident_bytes <= kMaximumWorkerAddressSpaceBytes;
}

bool guardian_worker_group_has_other_members(pid_t worker_pid) {
  const auto scan = scan_linux_worker_group(worker_pid);
  return !scan.valid || scan.has_other_members;
}
#elif defined(__APPLE__)
using ProcListPids = int (*)(std::uint32_t, std::uint32_t, void*, int);
using ProcPidRusage = int (*)(int, int, rusage_info_t*);

void* apple_proc_library() {
  static auto* library = ::dlopen("/usr/lib/libproc.dylib", RTLD_NOW | RTLD_LOCAL);
  return library;
}

ProcListPids apple_proc_listpids() {
  static const auto function = reinterpret_cast<ProcListPids>(
      apple_proc_library() != nullptr ? ::dlsym(apple_proc_library(), "proc_listpids") : nullptr);
  return function;
}

ProcPidRusage apple_proc_pid_rusage() {
  static const auto function = reinterpret_cast<ProcPidRusage>(
      apple_proc_library() != nullptr ? ::dlsym(apple_proc_library(), "proc_pid_rusage") : nullptr);
  return function;
}

bool guardian_worker_group_within_memory_limit(pid_t worker_pid) {
  std::array<pid_t, 4096> processes{};
  const auto proc_listpids = apple_proc_listpids();
  if (proc_listpids == nullptr) {
    return false;
  }
  errno = 0;
  const auto bytes = proc_listpids(PROC_PGRP_ONLY, static_cast<std::uint32_t>(worker_pid),
                                   processes.data(), static_cast<int>(sizeof(processes)));
  const auto list_error = errno;
  if (bytes < 0 || (bytes == 0 && list_error != 0) ||
      static_cast<std::size_t>(bytes) >= sizeof(processes)) {
    return false;
  }
  const auto proc_pid_rusage = apple_proc_pid_rusage();
  if (proc_pid_rusage == nullptr) {
    return false;
  }
  struct rusage_info_v2 worker_usage{};
  if (proc_pid_rusage(worker_pid, RUSAGE_INFO_V2,
                      reinterpret_cast<rusage_info_t*>(&worker_usage)) != 0) {
    errno = 0;
    return ::kill(worker_pid, 0) != 0 && errno == ESRCH;
  }
  std::uint64_t resident_bytes =
      std::max(worker_usage.ri_resident_size, worker_usage.ri_phys_footprint);
  if (resident_bytes > kMaximumWorkerAddressSpaceBytes) {
    return false;
  }
  const auto process_count = static_cast<std::size_t>(bytes) / sizeof(pid_t);
  for (std::size_t index = 0; index < process_count; ++index) {
    if (processes[index] <= 0 || processes[index] == worker_pid) {
      continue;
    }
    struct rusage_info_v2 usage{};
    if (proc_pid_rusage(processes[index], RUSAGE_INFO_V2,
                        reinterpret_cast<rusage_info_t*>(&usage)) != 0) {
      errno = 0;
      if (::kill(processes[index], 0) == 0 || errno != ESRCH) {
        return false;
      }
      continue;
    }
    const auto process_bytes = std::max(usage.ri_resident_size, usage.ri_phys_footprint);
    if (process_bytes > kMaximumWorkerAddressSpaceBytes - resident_bytes) {
      return false;
    }
    resident_bytes += process_bytes;
  }
  return true;
}

bool guardian_worker_group_has_other_members(pid_t worker_pid) {
  std::array<pid_t, 4096> processes{};
  const auto proc_listpids = apple_proc_listpids();
  if (proc_listpids == nullptr) {
    return true;
  }
  errno = 0;
  const auto bytes = proc_listpids(PROC_PGRP_ONLY, static_cast<std::uint32_t>(worker_pid),
                                   processes.data(), static_cast<int>(sizeof(processes)));
  const auto list_error = errno;
  if (bytes < 0 || (bytes == 0 && list_error != 0) ||
      static_cast<std::size_t>(bytes) >= sizeof(processes)) {
    return true;
  }
  const auto process_count = static_cast<std::size_t>(bytes) / sizeof(pid_t);
  for (std::size_t index = 0; index < process_count; ++index) {
    if (processes[index] > 0 && processes[index] != worker_pid) {
      return true;
    }
  }
  return false;
}
#else
bool guardian_worker_group_has_other_members(pid_t worker_pid) {
  (void)worker_pid;
  return false;
}
#endif

void guardian_wait_for_worker_group_cleanup(pid_t worker_pid) {
  constexpr timespec kGroupPollPause{.tv_sec = 0, .tv_nsec = 2'000'000};
  while (guardian_worker_group_has_other_members(worker_pid)) {
    (void)::nanosleep(&kGroupPollPause, nullptr);
  }
}

void guardian_terminate_worker(pid_t worker_pid, pid_t watchdog_pid = -1,
                               bool authority_reported = false) {
  kill_worker_process_group(worker_pid);
  if (watchdog_pid > 0) {
    (void)::kill(watchdog_pid, SIGKILL);
    int watchdog_status = 0;
    while (::waitpid(watchdog_pid, &watchdog_status, 0) < 0 && errno == EINTR) {
    }
  }
  if (worker_pid <= 0) {
    return;
  }
  siginfo_t worker_info{};
  int wait_result = -1;
  do {
    wait_result = ::waitid(P_PID, static_cast<id_t>(worker_pid), &worker_info, WEXITED | WNOWAIT);
  } while (wait_result < 0 && errno == EINTR);
  if (wait_result == 0 && worker_info.si_pid == worker_pid) {
    guardian_wait_for_worker_group_cleanup(worker_pid);
    if (authority_reported) {
      if (!guardian_pipe_write(kGuardianWorkerAuthority, kWorkerGroupCertified)) {
        guardian_exit(2);
      }
      (void)::close(kGuardianWorkerAuthority);
    }
    int worker_status = 0;
    while (::waitpid(worker_pid, &worker_status, 0) < 0 && errno == EINTR) {
    }
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

[[noreturn]] void run_pre_main_supervisor_watchdog(long maximum_descriptor) {
  if (!guardian_close_from(kPreMainWatchdogFirstUnusedDescriptor, maximum_descriptor)) {
    guardian_exit(127);
  }
  pid_t supervisor_pid = -1;
  std::size_t offset = 0;
  while (offset < sizeof(supervisor_pid)) {
    const auto received =
        ::read(kPreMainWatchdogTarget, reinterpret_cast<char*>(&supervisor_pid) + offset,
               sizeof(supervisor_pid) - offset);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      guardian_exit(0);
    }
    offset += static_cast<std::size_t>(received);
  }
  if (supervisor_pid <= 0) {
    guardian_exit(127);
  }

  while (true) {
    std::array<pollfd, 2> descriptors{{
        {.fd = STDIN_FILENO, .events = POLLIN, .revents = 0},
        {.fd = kPreMainWatchdogReady, .events = POLLIN, .revents = 0},
    }};
    const auto result = ::poll(descriptors.data(), descriptors.size(), -1);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      (void)::kill(-supervisor_pid, SIGKILL);
      (void)::kill(supervisor_pid, SIGKILL);
      (void)guardian_write(STDIN_FILENO, &kSupervisorCertified, 1);
      guardian_exit(127);
    }
    if (descriptors[1].revents != 0) {
      char ready = '\0';
      const auto received = ::read(kPreMainWatchdogReady, &ready, 1);
      if (received == 1 && ready == kGuardianReady &&
          guardian_pipe_write(kPreMainWatchdogReady, kGuardianAcknowledge)) {
        guardian_exit(0);
      }
      (void)::kill(-supervisor_pid, SIGKILL);
      (void)::kill(supervisor_pid, SIGKILL);
      (void)guardian_write(STDIN_FILENO, &kSupervisorCertified, 1);
      guardian_exit(127);
    }
    if (descriptors[0].revents != 0) {
      (void)::kill(-supervisor_pid, SIGKILL);
      (void)::kill(supervisor_pid, SIGKILL);
      (void)guardian_write(STDIN_FILENO, &kSupervisorCertified, 1);
      guardian_exit(0);
    }
  }
}

GuardianLaunch spawn_guardian_process(const std::string& executable, int executable_descriptor,
                                      int control_descriptor, int worker_input_descriptor,
                                      int worker_output_descriptor,
                                      std::chrono::nanoseconds pre_worker_report_delay,
                                      std::chrono::nanoseconds pre_guardian_exec_delay,
                                      const std::filesystem::path& pre_guardian_exec_marker,
                                      std::shared_ptr<SupervisorSlot>* reservation) {
  const auto maximum_descriptor = descriptor_scan_limit();
  if (maximum_descriptor < kSupervisorFirstUnusedDescriptor) {
    throw GpuVideoProbeError("failed to determine video probe guardian descriptor limit");
  }

  const auto encode_argument = [](std::array<char, 32>* destination, std::uint64_t value) {
    const auto [end, error] =
        std::to_chars(destination->data(), destination->data() + destination->size() - 1, value);
    if (error != std::errc{}) {
      throw GpuVideoProbeError("failed to encode video probe supervisor argument");
    }
    *end = '\0';
  };
  std::array<char, 32> worker_delay_text{};
  std::array<char, 32> guardian_delay_text{};
  std::array<char, 32> caller_pid_text{};
  encode_argument(&worker_delay_text, static_cast<std::uint64_t>(pre_worker_report_delay.count()));
  encode_argument(&guardian_delay_text,
                  static_cast<std::uint64_t>(pre_guardian_exec_delay.count()));
  const auto caller_pid = ::getpid();
  encode_argument(&caller_pid_text, static_cast<std::uint64_t>(caller_pid));
  std::array<char, 2> marker_text{pre_guardian_exec_marker.empty() ? '0' : '1', '\0'};
  char* const arguments[] = {const_cast<char*>(executable.c_str()),
                             const_cast<char*>("--reco-video-probe-supervisor"),
                             worker_delay_text.data(),
                             guardian_delay_text.data(),
                             caller_pid_text.data(),
                             marker_text.data(),
                             nullptr};
  std::vector<std::string> environment_storage;
  for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    if (std::string_view(*entry).rfind("RECO_VIDEO_PROBE_GUARDIAN_PROCESS=", 0) != 0 &&
        std::string_view(*entry).rfind("RECO_VIDEO_PROBE_PROCESS_ROLE=", 0) != 0) {
      environment_storage.emplace_back(*entry);
    }
  }
  environment_storage.emplace_back("RECO_VIDEO_PROBE_GUARDIAN_PROCESS=1");
  environment_storage.emplace_back("RECO_VIDEO_PROBE_PROCESS_ROLE=supervisor");
  std::vector<char*> guardian_environment;
  guardian_environment.reserve(environment_storage.size() + 1U);
  for (auto& entry : environment_storage) {
    guardian_environment.push_back(entry.data());
  }
  guardian_environment.push_back(nullptr);

  UniqueFd marker;
  if (!pre_guardian_exec_marker.empty()) {
    const auto marker_path = pre_guardian_exec_marker.string();
    UniqueFd opened_marker(
        ::open(marker_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    if (opened_marker.get() < 0) {
      throw GpuVideoProbeError("failed to create video probe guardian pre-exec marker: " +
                               std::string(std::strerror(errno)));
    }
    marker = duplicate_for_supervisor(opened_marker.get());
  }

  int lifetime_descriptors[2] = {-1, -1};
  create_socket_pair(lifetime_descriptors, "guardian lifetime");
  UniqueFd supervisor_lifetime(lifetime_descriptors[0]);
  UniqueFd caller_lifetime(lifetime_descriptors[1]);
  make_nonblocking(supervisor_lifetime.get());
  int watchdog_target_descriptors[2] = {-1, -1};
  int supervisor_arm_descriptors[2] = {-1, -1};
  int watchdog_ready_descriptors[2] = {-1, -1};
  create_socket_pair(watchdog_target_descriptors, "supervisor watchdog target");
  create_socket_pair(supervisor_arm_descriptors, "supervisor launch gate");
  create_socket_pair(watchdog_ready_descriptors, "supervisor watchdog readiness");
  UniqueFd watchdog_target(watchdog_target_descriptors[0]);
  UniqueFd caller_watchdog_target(watchdog_target_descriptors[1]);
  UniqueFd supervisor_arm(supervisor_arm_descriptors[0]);
  UniqueFd caller_supervisor_arm(supervisor_arm_descriptors[1]);
  UniqueFd watchdog_ready(watchdog_ready_descriptors[0]);
  UniqueFd supervisor_watchdog_ready(watchdog_ready_descriptors[1]);
#if defined(SO_NOSIGPIPE) && !defined(MSG_NOSIGNAL)
  const int suppress_lifetime_sigpipe = 1;
  if (::setsockopt(caller_lifetime.get(), SOL_SOCKET, SO_NOSIGPIPE, &suppress_lifetime_sigpipe,
                   sizeof(suppress_lifetime_sigpipe)) != 0) {
    throw GpuVideoProbeError("failed to suppress SIGPIPE for video probe guardian lifetime: " +
                             std::string(std::strerror(errno)));
  }
#endif
  auto supervisor_lifetime_descriptor = duplicate_for_supervisor(supervisor_lifetime.get());
  auto supervisor_executable = duplicate_for_supervisor(executable_descriptor);
  auto supervisor_control = duplicate_for_supervisor(control_descriptor);
  auto supervisor_worker_input = duplicate_for_supervisor(worker_input_descriptor);
  auto supervisor_worker_output = duplicate_for_supervisor(worker_output_descriptor);
  auto watchdog_lifetime_descriptor = duplicate_for_supervisor(supervisor_lifetime.get());
  auto watchdog_target_descriptor = duplicate_for_supervisor(watchdog_target.get());
  auto watchdog_ready_descriptor = duplicate_for_supervisor(watchdog_ready.get());
  auto supervisor_arm_descriptor = duplicate_for_supervisor(supervisor_arm.get());
  auto supervisor_watchdog_ready_descriptor =
      duplicate_for_supervisor(supervisor_watchdog_ready.get());
  const auto caller_lifetime_descriptor = caller_lifetime.get();
  const auto inherited_watchdog_target_writer = caller_watchdog_target.get();
  const auto inherited_supervisor_arm_writer = caller_supervisor_arm.get();
  char* const* const environment = guardian_environment.data();

  sigset_t blocked_mask{};
  sigset_t previous_mask{};
  if (sigfillset(&blocked_mask) != 0) {
    throw GpuVideoProbeError("failed to prepare video probe guardian signal mask: " +
                             std::string(std::strerror(errno)));
  }
  const auto block_error = ::pthread_sigmask(SIG_SETMASK, &blocked_mask, &previous_mask);
  if (block_error != 0) {
    throw GpuVideoProbeError("failed to block signals for video probe guardian launch: " +
                             std::string(std::strerror(block_error)));
  }

  const auto watchdog_pid = ::fork();
  const auto watchdog_fork_error = errno;
  if (watchdog_pid == 0) {
    (void)::close(caller_lifetime_descriptor);
    for (const auto [source, destination] :
         {std::pair(watchdog_lifetime_descriptor.get(), STDIN_FILENO),
          std::pair(watchdog_target_descriptor.get(), kPreMainWatchdogTarget),
          std::pair(watchdog_ready_descriptor.get(), kPreMainWatchdogReady)}) {
      if (::dup2(source, destination) < 0) {
        guardian_exit(127);
      }
    }
    run_pre_main_supervisor_watchdog(maximum_descriptor);
  }
  if (watchdog_pid < 0) {
    (void)::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr);
    errno = watchdog_fork_error;
    throw GpuVideoProbeError("failed to start video probe supervisor watchdog: " +
                             std::string(std::strerror(errno)));
  }

  const auto supervisor_pid = ::fork();
  const auto supervisor_fork_error = errno;
  if (supervisor_pid == 0) {
    const auto report_error = [&](int error) {
      const auto write_exact = [](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const char*>(data);
        std::size_t offset = 0;
        while (offset < size) {
          ssize_t written = -1;
          do {
            written = ::write(kSupervisorControl, bytes + offset, size - offset);
          } while (written < 0 && errno == EINTR);
          if (written <= 0) {
            guardian_exit(127);
          }
          offset += static_cast<std::size_t>(written);
        }
      };
      write_exact(&kGuardianLaunchFailed, 1);
      write_exact(&error, sizeof(error));
      (void)guardian_write(supervisor_lifetime_descriptor.get(), &kSupervisorCertified, 1);
      guardian_exit(127);
    };
    (void)::close(caller_lifetime_descriptor);
    (void)::close(inherited_watchdog_target_writer);
    (void)::close(inherited_supervisor_arm_writer);
    if (::dup2(supervisor_control.get(), kSupervisorControl) < 0) {
      guardian_exit(127);
    }
    if (::setpgid(0, 0) != 0) {
      report_error(errno);
    }
    char arm = '\0';
    ssize_t arm_size = -1;
    do {
      arm_size = ::read(supervisor_arm_descriptor.get(), &arm, 1);
    } while (arm_size < 0 && errno == EINTR);
    if (arm_size != 1 || arm != kGuardianRelease) {
      (void)guardian_write(supervisor_lifetime_descriptor.get(), &kSupervisorCertified, 1);
      guardian_exit(0);
    }
    for (const auto [source, destination] :
         {std::pair(supervisor_lifetime_descriptor.get(), STDIN_FILENO),
          std::pair(supervisor_executable.get(), kSupervisorExecutable),
          std::pair(supervisor_worker_input.get(), kSupervisorWorkerInput),
          std::pair(supervisor_worker_output.get(), kSupervisorWorkerOutput),
          std::pair(supervisor_watchdog_ready_descriptor.get(), kSupervisorWatchdogReady)}) {
      if (::dup2(source, destination) < 0) {
        report_error(errno);
      }
    }
    if (marker.get() >= 0 && ::dup2(marker.get(), kSupervisorMarker) < 0) {
      report_error(errno);
    }
    if (marker.get() < 0) {
      (void)::close(kSupervisorMarker);
    }
    if (!guardian_close_from(kSupervisorFirstUnusedDescriptor, maximum_descriptor)) {
      report_error(errno);
    }
    execute_pinned_probe(kSupervisorExecutable, arguments, environment);
    report_error(errno);
  }

  int supervisor_group_error = 0;
  if (supervisor_pid > 0 && ::setpgid(supervisor_pid, supervisor_pid) != 0 && errno != EACCES &&
      errno != ESRCH) {
    supervisor_group_error = errno;
  }

  bool watchdog_armed = false;
  if (supervisor_pid > 0 && supervisor_group_error == 0 &&
      guardian_write(caller_watchdog_target.get(), &supervisor_pid, sizeof(supervisor_pid)) &&
      guardian_write(caller_supervisor_arm.get(), &kGuardianRelease, 1)) {
    watchdog_armed = true;
  }
  caller_watchdog_target.reset();
  caller_supervisor_arm.reset();

  supervisor_lifetime.reset();
  const auto restore_error = ::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr);
  if (restore_error != 0 || supervisor_pid < 0 || supervisor_group_error != 0 || !watchdog_armed) {
    const char terminate = kGuardianTerminate;
    ssize_t written = -1;
    do {
      written = ::send(caller_lifetime.get(), &terminate, 1,
#if defined(MSG_NOSIGNAL)
                       MSG_NOSIGNAL
#else
                       0
#endif
      );
    } while (written < 0 && errno == EINTR);
    if (supervisor_pid > 0) {
      int status = 0;
      const auto cleanup_deadline = std::chrono::steady_clock::now() + kMaximumTerminationReserve;
      if (wait_for_process_exit(supervisor_pid, &status, cleanup_deadline)) {
        if (take_supervisor_certificate(&caller_lifetime)) {
          reservation->reset();
        } else {
          retain_uncertified_reservation(std::move(*reservation));
        }
      } else {
        (void)::kill(-supervisor_pid, SIGKILL);
        (void)::kill(supervisor_pid, SIGKILL);
        if (!deferred_process_reaper().add(supervisor_pid, std::move(*reservation),
                                           std::move(caller_lifetime))) {
          caller_lifetime.reset();
          retain_uncertified_reservation(std::move(*reservation));
        }
      }
    } else {
      caller_lifetime.reset();
    }
    int watchdog_status = 0;
    const auto watchdog_deadline = std::chrono::steady_clock::now() + kMaximumTerminationReserve;
    if (!wait_for_process_exit(watchdog_pid, &watchdog_status, watchdog_deadline)) {
      (void)::kill(watchdog_pid, SIGKILL);
      (void)deferred_process_reaper().add(watchdog_pid, nullptr);
    }
    if (restore_error != 0) {
      throw GpuVideoProbeError("failed to restore signals after video probe guardian launch: " +
                               std::string(std::strerror(restore_error)));
    }
    if (supervisor_pid < 0) {
      errno = supervisor_fork_error;
      throw GpuVideoProbeError("failed to start video probe worker guardian supervisor: " +
                               std::string(std::strerror(errno)));
    }
    if (supervisor_group_error != 0) {
      throw GpuVideoProbeError("failed to isolate video probe guardian supervisor: " +
                               std::string(std::strerror(supervisor_group_error)));
    }
    throw GpuVideoProbeError("failed to arm video probe supervisor watchdog");
  }
  return GuardianLaunch{.supervisor_pid = supervisor_pid,
                        .watchdog_pid = watchdog_pid,
                        .caller_lifetime = std::move(caller_lifetime)};
}

[[noreturn]] void run_supervisor_child(const char* executable,
                                       std::chrono::nanoseconds pre_worker_report_delay,
                                       std::chrono::nanoseconds pre_guardian_exec_delay,
                                       pid_t caller_pid, bool has_marker, long maximum_descriptor) {
  std::array<char, 32> delay_text{};
  const auto [delay_end, delay_error] =
      std::to_chars(delay_text.data(), delay_text.data() + delay_text.size() - 1,
                    static_cast<std::uint64_t>(pre_worker_report_delay.count()));
  if (delay_error != std::errc{}) {
    guardian_exit(2);
  }
  *delay_end = '\0';
  char* const arguments[] = {const_cast<char*>(executable),
                             const_cast<char*>("--reco-video-probe-guardian"), delay_text.data(),
                             nullptr};

  char initial_lifetime = '\0';
  ssize_t initial_lifetime_size = -1;
  do {
    initial_lifetime_size = ::read(STDIN_FILENO, &initial_lifetime, 1);
  } while (initial_lifetime_size < 0 && errno == EINTR);
  if (::getppid() != caller_pid || initial_lifetime_size >= 0 ||
      (errno != EAGAIN && errno != EWOULDBLOCK)) {
    (void)guardian_write(STDIN_FILENO, &kSupervisorCertified, 1);
    guardian_exit(0);
  }
  const auto report_supervisor_error = [](int error) {
    (void)guardian_write(kSupervisorControl, &kGuardianLaunchFailed, 1);
    (void)guardian_write(kSupervisorControl, &error, sizeof(error));
    (void)guardian_write(STDIN_FILENO, &kSupervisorCertified, 1);
    guardian_exit(2);
  };
  struct sigaction child_action{};
  child_action.sa_handler = SIG_DFL;
  (void)sigemptyset(&child_action.sa_mask);
  if (::sigaction(SIGCHLD, &child_action, nullptr) != 0) {
    report_supervisor_error(errno);
  }
#if defined(__linux__)
  if (::prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
    report_supervisor_error(errno);
  }
#endif
  int authority_pipe[2] = {-1, -1};
  if (::pipe(authority_pipe) != 0) {
    report_supervisor_error(errno);
  }
  int worker_authority[2] = {-1, -1};
  worker_authority[0] = ::fcntl(authority_pipe[0], F_DUPFD, kSupervisorFirstUnusedDescriptor);
  worker_authority[1] = ::fcntl(authority_pipe[1], F_DUPFD, kSupervisorFirstUnusedDescriptor);
  const auto authority_duplication_error = errno;
  (void)::close(authority_pipe[0]);
  (void)::close(authority_pipe[1]);
  if (worker_authority[0] < 0 || worker_authority[1] < 0) {
    (void)::close(worker_authority[0]);
    (void)::close(worker_authority[1]);
    report_supervisor_error(authority_duplication_error);
  }
  const auto authority_flags = ::fcntl(worker_authority[0], F_GETFL);
  if (authority_flags < 0 ||
      ::fcntl(worker_authority[0], F_SETFL, authority_flags | O_NONBLOCK) != 0) {
    report_supervisor_error(errno);
  }
  if (!guardian_pipe_write(kSupervisorWatchdogReady, kGuardianReady)) {
    report_supervisor_error(errno);
  }
  char watchdog_acknowledge = '\0';
  ssize_t watchdog_acknowledge_size = -1;
  do {
    watchdog_acknowledge_size = ::read(kSupervisorWatchdogReady, &watchdog_acknowledge, 1);
  } while (watchdog_acknowledge_size < 0 && errno == EINTR);
  if (watchdog_acknowledge_size != 1 || watchdog_acknowledge != kGuardianAcknowledge) {
    report_supervisor_error(errno == 0 ? EIO : errno);
  }
  (void)::close(kSupervisorWatchdogReady);

  const auto expected_supervisor_pid = ::getpid();
  const auto guardian_pid = ::fork();
  if (guardian_pid == 0) {
    (void)::close(worker_authority[0]);
    (void)::close(STDIN_FILENO);
    const auto report_error = [](int descriptor, int error) {
      (void)guardian_write(descriptor, &kGuardianLaunchFailed, 1);
      (void)guardian_write(descriptor, &error, sizeof(error));
      guardian_exit(127);
    };
    if (::setenv("RECO_VIDEO_PROBE_PROCESS_ROLE", "guardian", 1) != 0 ||
        ::dup2(kSupervisorControl, STDIN_FILENO) < 0 ||
        ::dup2(kSupervisorControl, STDOUT_FILENO) < 0) {
      report_error(kSupervisorControl, errno);
    }
    if (::setpgid(0, 0) != 0) {
      report_error(STDOUT_FILENO, errno);
    }
#if defined(__linux__)
    if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || ::getppid() != expected_supervisor_pid) {
      report_error(STDOUT_FILENO, errno == 0 ? ESRCH : errno);
    }
#endif
    if (::dup2(kSupervisorWorkerInput, kGuardianWorkerInput) < 0 ||
        ::dup2(kSupervisorWorkerOutput, kGuardianWorkerOutput) < 0 ||
        ::dup2(worker_authority[1], kGuardianWorkerAuthority) < 0) {
      report_error(STDOUT_FILENO, errno);
    }
    if (has_marker) {
      char process_text[40]{};
      const auto process_size =
          format_guardian_parent_argument(process_text, sizeof(process_text), ::getpid());
      if (process_size < 2U) {
        report_error(STDOUT_FILENO, EOVERFLOW);
      }
      const auto marker_size = process_size - 2U;
      std::size_t offset = 0;
      while (offset < marker_size) {
        ssize_t written = -1;
        do {
          written = ::write(kSupervisorMarker, process_text + offset, marker_size - offset);
        } while (written < 0 && errno == EINTR);
        if (written <= 0) {
          report_error(STDOUT_FILENO, errno);
        }
        offset += static_cast<std::size_t>(written);
      }
    }
    if (::dup2(kSupervisorExecutable, kGuardianExecutable) < 0) {
      report_error(STDOUT_FILENO, errno);
    }
    if (!guardian_close_from(kGuardianFirstUnusedDescriptor, maximum_descriptor)) {
      report_error(STDOUT_FILENO, errno);
    }
    if (!guardian_sleep(pre_guardian_exec_delay)) {
      report_error(STDOUT_FILENO, errno);
    }
    execute_pinned_probe(kGuardianExecutable, arguments, environ);
    report_error(STDOUT_FILENO, errno);
  }

  (void)::close(worker_authority[1]);
  if (guardian_pid < 0) {
    const auto launch_error = errno;
    (void)::close(worker_authority[0]);
    (void)guardian_write(kSupervisorControl, &kGuardianLaunchFailed, 1);
    (void)guardian_write(kSupervisorControl, &launch_error, sizeof(launch_error));
    (void)guardian_write(STDIN_FILENO, &kSupervisorCertified, 1);
    guardian_exit(2);
  }
  (void)::close(kSupervisorControl);
  (void)::close(kSupervisorWorkerInput);
  (void)::close(kSupervisorWorkerOutput);
  (void)::close(kSupervisorMarker);
  if (::dup2(worker_authority[0], STDOUT_FILENO) < 0) {
    (void)::kill(-guardian_pid, SIGKILL);
    (void)::kill(guardian_pid, SIGKILL);
    int status = 0;
    while (::waitpid(guardian_pid, &status, 0) < 0 && errno == EINTR) {
    }
    guardian_exit(2);
  }
  (void)::close(worker_authority[0]);
  (void)::close(STDERR_FILENO);
  if (!guardian_close_from(STDERR_FILENO + 1, maximum_descriptor)) {
    (void)::kill(-guardian_pid, SIGKILL);
    (void)::kill(guardian_pid, SIGKILL);
    int status = 0;
    while (::waitpid(guardian_pid, &status, 0) < 0 && errno == EINTR) {
    }
    guardian_exit(2);
  }

  constexpr timespec kSupervisorPollPause{.tv_sec = 0, .tv_nsec = 2'000'000};
  std::array<char, sizeof(pid_t) + 1U> worker_group_bytes{};
  std::size_t worker_group_size = 0;
  bool worker_authority_closed = false;
  bool worker_authority_invalid = false;
  const auto read_worker_authority = [&] {
    while (!worker_authority_closed && !worker_authority_invalid) {
      char byte = '\0';
      const auto received = ::read(STDOUT_FILENO, &byte, 1);
      if (received == 1) {
        if (worker_group_size >= worker_group_bytes.size()) {
          worker_authority_invalid = true;
        } else {
          worker_group_bytes[worker_group_size++] = byte;
        }
        continue;
      }
      if (received == 0) {
        worker_authority_closed = true;
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        worker_authority_invalid = true;
      }
      break;
    }
  };
  bool terminate_guardian = false;
  int supervisor_status = 2;
  while (true) {
    read_worker_authority();
    char lifetime = '\0';
    ssize_t lifetime_size = -1;
    do {
      lifetime_size = ::read(STDIN_FILENO, &lifetime, 1);
    } while (lifetime_size < 0 && errno == EINTR);
    if (::getppid() != caller_pid || lifetime_size == 0 || lifetime_size > 0 ||
        (lifetime_size < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
      terminate_guardian = true;
      (void)::kill(-guardian_pid, SIGKILL);
      (void)::kill(guardian_pid, SIGKILL);
    }

    int status = 0;
    const auto wait_result = ::waitpid(guardian_pid, &status, WNOHANG);
    if (wait_result == guardian_pid) {
      supervisor_status =
          terminate_guardian || (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : 2;
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      supervisor_status = terminate_guardian && errno == ECHILD ? 0 : 2;
      break;
    }
    (void)::nanosleep(&kSupervisorPollPause, nullptr);
  }

  while (!worker_authority_closed && !worker_authority_invalid) {
    read_worker_authority();
    if (!worker_authority_closed && !worker_authority_invalid) {
      (void)::nanosleep(&kSupervisorPollPause, nullptr);
    }
  }
  if (worker_authority_invalid || (worker_group_size != 0 && worker_group_size < sizeof(pid_t)) ||
      (worker_group_size == worker_group_bytes.size() &&
       worker_group_bytes.back() != kWorkerGroupCertified)) {
    // An incomplete authority report cannot prove that no worker group exists.
    while (true) {
      (void)::nanosleep(&kSupervisorPollPause, nullptr);
    }
  }

  pid_t worker_group = -1;
  if (worker_group_size >= sizeof(pid_t)) {
    std::memcpy(&worker_group, worker_group_bytes.data(), sizeof(worker_group));
    if (worker_group <= 0) {
      while (true) {
        (void)::nanosleep(&kSupervisorPollPause, nullptr);
      }
    }
    const bool cleanup_certified = worker_group_size == worker_group_bytes.size();
    if (!cleanup_certified) {
      kill_worker_process_group(worker_group);
    }
    while (!cleanup_certified && worker_process_group_exists(worker_group)) {
#if defined(__linux__)
      int worker_status = 0;
      while (::waitpid(-worker_group, &worker_status, WNOHANG) > 0) {
      }
#endif
      kill_worker_process_group(worker_group);
      (void)::nanosleep(&kSupervisorPollPause, nullptr);
    }
#if defined(__linux__)
    int worker_status = 0;
    while (::waitpid(-worker_group, &worker_status, WNOHANG) > 0) {
    }
#endif
  }
  if (!guardian_write(STDIN_FILENO, &kSupervisorCertified, 1)) {
    guardian_exit(2);
  }
  guardian_exit(supervisor_status);
}

[[noreturn]] void run_guardian_child(const char* executable, int control_descriptor,
                                     int worker_input_descriptor, int worker_output_descriptor,
                                     long maximum_descriptor,
                                     std::chrono::nanoseconds pre_worker_report_delay) {
  if (::unsetenv("RECO_VIDEO_PROBE_GUARDIAN_PROCESS") != 0 || ::setpgid(0, 0) != 0 ||
      ::dup2(control_descriptor, STDIN_FILENO) < 0 ||
      ::dup2(control_descriptor, STDOUT_FILENO) < 0 ||
      ::dup2(worker_input_descriptor, kGuardianWorkerInput) < 0 ||
      ::dup2(worker_output_descriptor, kGuardianWorkerOutput) < 0) {
    report_guardian_startup_error(errno);
  }
  if (!guardian_close_from(kGuardianFirstUnusedDescriptor, maximum_descriptor)) {
    report_guardian_startup_error(errno);
  }
  struct sigaction child_action{};
  child_action.sa_handler = SIG_DFL;
  (void)sigemptyset(&child_action.sa_mask);
  if (::sigaction(SIGCHLD, &child_action, nullptr) != 0) {
    report_guardian_startup_error(errno);
  }
  if (!guardian_write(STDOUT_FILENO, &kGuardianReady, 1)) {
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

#if defined(RECO_PROBE_WIDE_ADDRESS_SANITIZER)
  const char* current_asan_options = ::getenv("ASAN_OPTIONS");
  std::string asan_options = current_asan_options == nullptr ? "" : current_asan_options;
  if (!asan_options.empty()) {
    asan_options.push_back(':');
  }
  asan_options += "detect_leaks=0";
  if (::setenv("ASAN_OPTIONS", asan_options.c_str(), 1) != 0) {
    report_guardian_startup_error(errno);
  }
#endif
  if (!guardian_limit_worker_address_space()) {
    report_guardian_startup_error(errno);
  }
  if (::setenv("GST_REGISTRY_FORK", "no", 1) != 0 ||
      ::setenv("RECO_VIDEO_PROBE_PROCESS_ROLE", "worker", 1) != 0) {
    report_guardian_startup_error(errno);
  }
  int start_gate[2] = {-1, -1};
  int lifetime_gate[2] = {-1, -1};
  int watchdog_gate[2] = {-1, -1};
  if (::pipe(start_gate) != 0 || ::pipe(lifetime_gate) != 0 || ::pipe(watchdog_gate) != 0) {
    const auto pipe_error = errno;
    (void)::close(start_gate[0]);
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[0]);
    (void)::close(lifetime_gate[1]);
    (void)::close(watchdog_gate[0]);
    (void)::close(watchdog_gate[1]);
    report_guardian_startup_error(pipe_error);
  }
  const auto guardian_pid = ::getpid();
  const auto worker_pid = ::fork();
  const auto worker_fork_error = errno;
  if (worker_pid == 0) {
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[0]);
    (void)::close(lifetime_gate[1]);
    (void)::close(watchdog_gate[0]);
    (void)::close(watchdog_gate[1]);
    (void)::close(kGuardianWorkerAuthority);
#if defined(__linux__)
    if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || ::getppid() != guardian_pid) {
      guardian_exit(127);
    }
#endif
    if (::setpgid(0, 0) != 0 || ::dup2(kGuardianWorkerInput, STDIN_FILENO) < 0 ||
        ::dup2(kGuardianWorkerOutput, STDOUT_FILENO) < 0 ||
        ::dup2(kGuardianExecutable, kWorkerExecutable) < 0 ||
        ::fcntl(kWorkerExecutable, F_SETFD, FD_CLOEXEC) != 0) {
      guardian_exit(127);
    }
    if (!guardian_restrict_worker_process_creation()) {
      guardian_exit(127);
    }
    char release = '\0';
    ssize_t release_size = -1;
    do {
      release_size = ::read(start_gate[0], &release, 1);
    } while (release_size < 0 && errno == EINTR);
    if (release_size != 1 || release != kGuardianRelease) {
      guardian_exit(127);
    }
    (void)::close(start_gate[0]);
    if (!guardian_close_from(kWorkerExecutable + 1, maximum_descriptor)) {
      guardian_exit(127);
    }
    char parent_argument[40]{};
    if (format_guardian_parent_argument(parent_argument, sizeof(parent_argument), guardian_pid) ==
        0) {
      guardian_exit(127);
    }
    char* const arguments[] = {const_cast<char*>(executable),
                               const_cast<char*>("--reco-video-probe-worker"), parent_argument,
                               nullptr};
    execute_pinned_probe(kWorkerExecutable, arguments, environ);
    guardian_exit(127);
  }
  (void)::close(start_gate[0]);
  (void)::close(kGuardianExecutable);
  if (worker_pid < 0) {
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[0]);
    (void)::close(lifetime_gate[1]);
    (void)::close(watchdog_gate[0]);
    (void)::close(watchdog_gate[1]);
    report_guardian_startup_error(worker_fork_error);
  }
  if (::setpgid(worker_pid, worker_pid) != 0) {
    const auto group_error = errno;
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[0]);
    (void)::close(lifetime_gate[1]);
    (void)::close(watchdog_gate[0]);
    (void)::close(watchdog_gate[1]);
    guardian_terminate_worker(worker_pid);
    report_guardian_startup_error(group_error);
  }
  (void)::close(kGuardianWorkerInput);
  (void)::close(kGuardianWorkerOutput);

  const auto watchdog_pid = ::fork();
  if (watchdog_pid == 0) {
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[1]);
    (void)::close(watchdog_gate[1]);
    if (::dup2(lifetime_gate[0], STDIN_FILENO) < 0 || ::dup2(watchdog_gate[0], STDOUT_FILENO) < 0) {
      guardian_exit(127);
    }
    if (!guardian_close_from(STDERR_FILENO + 1, maximum_descriptor)) {
      guardian_exit(127);
    }
    char ready = '\0';
    ssize_t ready_size = -1;
    do {
      ready_size = ::read(STDOUT_FILENO, &ready, 1);
    } while (ready_size < 0 && errno == EINTR);
    if (ready_size != 1 || ready != kGuardianRelease) {
      guardian_exit(127);
    }
    char lifetime = '\0';
    ssize_t lifetime_size = -1;
    do {
      lifetime_size = ::read(STDIN_FILENO, &lifetime, 1);
    } while (lifetime_size < 0 && errno == EINTR);
    (void)::kill(0, SIGKILL);
    guardian_exit(127);
  }
  (void)::close(lifetime_gate[0]);
  (void)::close(watchdog_gate[0]);
  const auto watchdog_group_error =
      watchdog_pid < 0 || ::setpgid(watchdog_pid, worker_pid) != 0 ? errno : 0;
  if (watchdog_group_error != 0 || !guardian_pipe_write(watchdog_gate[1], kGuardianRelease)) {
    const auto startup_error = watchdog_group_error != 0 ? watchdog_group_error : EIO;
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[1]);
    (void)::close(watchdog_gate[1]);
    guardian_terminate_worker(worker_pid, watchdog_pid);
    report_guardian_startup_error(startup_error);
  }
  (void)::close(watchdog_gate[1]);
  const auto encoded_worker_pid = static_cast<std::uint64_t>(worker_pid);
  if (!guardian_sleep(pre_worker_report_delay) ||
      !guardian_pipe_write_process_id(kGuardianWorkerAuthority, worker_pid)) {
    const auto report_error = errno == 0 ? EIO : errno;
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[1]);
    guardian_terminate_worker(worker_pid, watchdog_pid);
    report_guardian_startup_error(report_error);
  }
  const bool authority_reported = true;
  if (!guardian_write(STDOUT_FILENO, &kGuardianStarted, 1) ||
      !guardian_write(STDOUT_FILENO, &encoded_worker_pid, sizeof(encoded_worker_pid))) {
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[1]);
    guardian_terminate_worker(worker_pid, watchdog_pid, authority_reported);
    guardian_exit(2);
  }

  command = '\0';
  do {
    received = ::recv(STDIN_FILENO, &command, 1, 0);
  } while (received < 0 && errno == EINTR);
  if (received != 1 || command != kGuardianRelease ||
      !guardian_pipe_write(start_gate[1], kGuardianRelease)) {
    (void)::close(start_gate[1]);
    (void)::close(lifetime_gate[1]);
    guardian_terminate_worker(worker_pid, watchdog_pid, authority_reported);
    guardian_exit(received == 1 && command == kGuardianTerminate ? 0 : 2);
  }
  (void)::close(start_gate[1]);

  int worker_status = 0;
  bool memory_termination_sent = false;
  while (true) {
    siginfo_t worker_info{};
    const auto wait_result =
        ::waitid(P_PID, static_cast<id_t>(worker_pid), &worker_info, WEXITED | WNOHANG | WNOWAIT);
    if (wait_result == 0 && worker_info.si_pid == worker_pid) {
      // Keep the zombie leader, and therefore its PID/process-group identity,
      // reserved until the in-group watchdog has killed every other member.
      (void)::close(lifetime_gate[1]);
      lifetime_gate[1] = -1;
      int watchdog_status = 0;
      while (::waitpid(watchdog_pid, &watchdog_status, 0) < 0 && errno == EINTR) {
      }
      guardian_wait_for_worker_group_cleanup(worker_pid);
      if (!guardian_pipe_write(kGuardianWorkerAuthority, kWorkerGroupCertified)) {
        guardian_exit(2);
      }
      (void)::close(kGuardianWorkerAuthority);
      pid_t reaped_worker = -1;
      do {
        reaped_worker = ::waitpid(worker_pid, &worker_status, 0);
      } while (reaped_worker < 0 && errno == EINTR);
      if (reaped_worker != worker_pid) {
        worker_status = -1;
      }
      break;
    }
    if (wait_result < 0 && errno == ECHILD) {
      (void)::close(lifetime_gate[1]);
      lifetime_gate[1] = -1;
      int watchdog_status = 0;
      while (::waitpid(watchdog_pid, &watchdog_status, 0) < 0 && errno == EINTR) {
      }
      guardian_wait_for_worker_group_cleanup(worker_pid);
      if (!guardian_pipe_write(kGuardianWorkerAuthority, kWorkerGroupCertified)) {
        guardian_exit(2);
      }
      (void)::close(kGuardianWorkerAuthority);
      worker_status = -1;
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      guardian_terminate_worker(worker_pid, watchdog_pid, authority_reported);
      guardian_exit(2);
    }
    siginfo_t watchdog_info{};
    const auto watchdog_wait = ::waitid(P_PID, static_cast<id_t>(watchdog_pid), &watchdog_info,
                                        WEXITED | WNOHANG | WNOWAIT);
    const auto watchdog_error = errno;
    if (guardian_watchdog_exit_is_fatal(memory_termination_sent, watchdog_wait, watchdog_error,
                                        watchdog_info.si_pid, watchdog_pid)) {
      guardian_terminate_worker(worker_pid, watchdog_pid, authority_reported);
      guardian_exit(2);
    }
#if defined(__linux__) || defined(__APPLE__)
    if (!memory_termination_sent && !guardian_worker_group_within_memory_limit(worker_pid)) {
      kill_worker_process_group(worker_pid);
      memory_termination_sent = true;
    }
#else
    (void)memory_termination_sent;
#endif
    pollfd control{.fd = STDIN_FILENO, .events = POLLIN, .revents = 0};
    const auto poll_result = ::poll(&control, 1, 2);
    if (poll_result < 0 && errno == EINTR) {
      continue;
    }
    if (poll_result < 0) {
      guardian_terminate_worker(worker_pid, watchdog_pid);
      guardian_exit(2);
    }
    if (poll_result > 0) {
      command = '\0';
      do {
        received = ::recv(STDIN_FILENO, &command, 1, 0);
      } while (received < 0 && errno == EINTR);
      if (received != 1 || command == kGuardianTerminate) {
        guardian_terminate_worker(worker_pid, watchdog_pid, authority_reported);
        guardian_exit(0);
      }
    }
  }

  (void)::close(lifetime_gate[1]);
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
  GuardianProcess(pid_t pid, pid_t watchdog_pid, UniqueFd control, UniqueFd caller_lifetime,
                  std::chrono::steady_clock::time_point deadline,
                  std::shared_ptr<SupervisorSlot> reservation)
      : pid_(pid), watchdog_pid_(watchdog_pid), control_(std::move(control)),
        caller_lifetime_(std::move(caller_lifetime)), deadline_(deadline),
        reservation_(std::move(reservation)) {}
  GuardianProcess(const GuardianProcess&) = delete;
  GuardianProcess& operator=(const GuardianProcess&) = delete;
  ~GuardianProcess() { terminate(); }

  [[nodiscard]] int control() const { return control_.get(); }

  bool finish() {
    if (pid_ <= 0) {
      return true;
    }
    const char acknowledge = kGuardianAcknowledge;
    ssize_t acknowledged = -1;
    do {
      acknowledged = ::send(control_.get(), &acknowledge, 1,
#if defined(MSG_NOSIGNAL)
                            MSG_NOSIGNAL
#else
                            0
#endif
      );
    } while (acknowledged < 0 && errno == EINTR);
    control_.reset();
    int status = 0;
    const auto exited = wait_for_process_exit(pid_, &status, deadline_);
    int watchdog_status = 0;
    const auto watchdog_exited =
        watchdog_pid_ <= 0 || wait_for_process_exit(watchdog_pid_, &watchdog_status, deadline_);
    if (exited) {
      const auto certified = take_supervisor_certificate(&caller_lifetime_);
      pid_ = -1;
      if (certified) {
        reservation_.reset();
      } else {
        retain_uncertified_reservation(std::move(reservation_));
      }
      certified_ = certified;
    }
    if (watchdog_exited) {
      watchdog_pid_ = -1;
    }
    return acknowledged == 1 && exited && certified_ && watchdog_exited &&
           (status == 0 || (WIFEXITED(status) && WEXITSTATUS(status) == 0));
  }

private:
  void terminate() {
    if (pid_ <= 0 && watchdog_pid_ <= 0) {
      return;
    }
    const char terminate = kGuardianTerminate;
    (void)::send(control_.get(), &terminate, 1,
#if defined(MSG_NOSIGNAL)
                 MSG_NOSIGNAL
#else
                 0
#endif
    );
    control_.reset();
    const char lifetime_terminate = kGuardianTerminate;
    ssize_t lifetime_written = -1;
    do {
      lifetime_written = ::send(caller_lifetime_.get(), &lifetime_terminate, 1,
#if defined(MSG_NOSIGNAL)
                                MSG_NOSIGNAL
#else
                                0
#endif
      );
    } while (lifetime_written < 0 && errno == EINTR);
    int status = 0;
    const auto now = std::chrono::steady_clock::now();
    const auto maximum_grace = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::milliseconds(50));
    const auto grace_deadline =
        now < deadline_ ? now + std::min((deadline_ - now) / 2, maximum_grace) : now;
    const auto supervisor_exited = pid_ <= 0 ||
                                   wait_for_process_exit(pid_, &status, grace_deadline) ||
                                   wait_for_process_exit(pid_, &status, deadline_);
    if (!supervisor_exited && !deferred_process_reaper().add(pid_, std::move(reservation_),
                                                             std::move(caller_lifetime_))) {
      // The fixed queue has two entries per admitted probe, so this is only
      // reachable after an internal accounting violation. Keep the process
      // killed and fail closed by retaining the reservation for process life.
      caller_lifetime_.reset();
      retain_uncertified_reservation(std::move(reservation_));
    } else if (supervisor_exited) {
      if (take_supervisor_certificate(&caller_lifetime_)) {
        reservation_.reset();
      } else {
        retain_uncertified_reservation(std::move(reservation_));
      }
    }
    int watchdog_status = 0;
    const auto watchdog_exited =
        watchdog_pid_ <= 0 || wait_for_process_exit(watchdog_pid_, &watchdog_status, deadline_);
    if (!watchdog_exited) {
      (void)::kill(watchdog_pid_, SIGKILL);
      (void)deferred_process_reaper().add(watchdog_pid_, nullptr);
    }
    pid_ = -1;
    watchdog_pid_ = -1;
  }

  pid_t pid_ = -1;
  pid_t watchdog_pid_ = -1;
  UniqueFd control_;
  UniqueFd caller_lifetime_;
  std::chrono::steady_clock::time_point deadline_;
  std::shared_ptr<SupervisorSlot> reservation_;
  bool certified_ = false;
};

std::string run_probe_worker(const std::filesystem::path& worker_path, std::string_view request,
                             std::chrono::steady_clock::time_point deadline,
                             std::chrono::steady_clock::time_point cleanup_deadline,
                             const ProbeLaunchOptions& options,
                             std::shared_ptr<SupervisorSlot> reservation) {
  require_worker_launch_active(deadline);
  (void)deferred_process_reaper();
  auto source_executable = open_probe_executable(worker_path);
#if defined(__linux__)
  auto pinned_executable = snapshot_linux_probe_executable(source_executable.get());
  const auto executable = worker_path.string();
#elif defined(__APPLE__)
  MacProbeExecutableSnapshot executable_snapshot(source_executable.get(), deadline);
  auto pinned_executable = open_probe_executable(executable_snapshot.executable());
  const auto executable = executable_snapshot.executable().string();
#else
  auto pinned_executable = std::move(source_executable);
  const auto executable = worker_path.string();
#endif
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
                   sizeof(suppress_sigpipe)) != 0 ||
      ::setsockopt(parent_control.get(), SOL_SOCKET, SO_NOSIGPIPE, &suppress_sigpipe,
                   sizeof(suppress_sigpipe)) != 0 ||
      ::setsockopt(child_control.get(), SOL_SOCKET, SO_NOSIGPIPE, &suppress_sigpipe,
                   sizeof(suppress_sigpipe)) != 0) {
    throw GpuVideoProbeError("failed to suppress SIGPIPE for video probe worker IPC: " +
                             std::string(std::strerror(errno)));
  }
#endif

  auto guard_control = duplicate_for_guardian(child_control.get());
  auto guard_input = duplicate_for_guardian(child_input.get());
  auto guard_output = duplicate_for_guardian(child_output.get());
  if (!options.pre_supervisor_exec_marker.empty()) {
    UniqueFd marker(::open(options.pre_supervisor_exec_marker.c_str(),
                           O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (marker.get() < 0) {
      throw GpuVideoProbeError("failed to create video probe supervisor pre-exec marker: " +
                               std::string(std::strerror(errno)));
    }
    const auto process = std::to_string(static_cast<std::uint64_t>(::getpid()));
    std::size_t offset = 0;
    while (offset < process.size()) {
      ssize_t written = -1;
      do {
        written = ::write(marker.get(), process.data() + offset, process.size() - offset);
      } while (written < 0 && errno == EINTR);
      if (written <= 0) {
        throw GpuVideoProbeError("failed to write video probe supervisor pre-exec marker");
      }
      offset += static_cast<std::size_t>(written);
    }
  }
  wait_for_worker_launch_delay(options.pre_worker_spawn_delay, deadline);
  auto guardian_launch = spawn_guardian_process(
      executable, pinned_executable.get(), guard_control.get(), guard_input.get(),
      guard_output.get(), options.pre_worker_report_delay, options.pre_guardian_exec_delay,
      options.pre_guardian_exec_marker, &reservation);
  pinned_executable.reset();
  GuardianProcess guardian(guardian_launch.supervisor_pid, guardian_launch.watchdog_pid,
                           std::move(parent_control), std::move(guardian_launch.caller_lifetime),
                           cleanup_deadline, std::move(reservation));
  child_input.reset();
  child_output.reset();
  child_control.reset();
  guard_control.reset();
  guard_input.reset();
  guard_output.reset();
  char lifecycle = '\0';
  const auto throw_guardian_startup_error = [&] {
    int launch_error = 0;
    read_exact(guardian.control(), reinterpret_cast<char*>(&launch_error), sizeof(launch_error),
               deadline);
    throw GpuVideoProbeError("failed to start video probe worker guardian: " +
                             std::string(std::strerror(launch_error)));
  };
  read_exact(guardian.control(), &lifecycle, 1, deadline);
  if (lifecycle == kGuardianLaunchFailed) {
    throw_guardian_startup_error();
  }
  if (lifecycle != kGuardianReady) {
    throw GpuVideoProbeError("video probe guardian failed its readiness handshake");
  }
  write_all(guardian.control(), std::string_view(&kGuardianLaunch, 1), deadline);
  read_exact(guardian.control(), &lifecycle, 1, deadline);
  if (lifecycle == kGuardianLaunchFailed) {
    throw_guardian_startup_error();
  }
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
  require_worker_launch_active(deadline);
  write_all(guardian.control(), std::string_view(&kGuardianRelease, 1), deadline);
  write_request(parent_input.get(), request, deadline);
  parent_input.reset();

  read_exact(guardian.control(), &lifecycle, 1, deadline);
  if (lifecycle != kGuardianExited) {
    throw GpuVideoProbeError("video probe guardian failed its worker exit handshake");
  }
  int status = 0;
  read_exact(guardian.control(), reinterpret_cast<char*>(&status), sizeof(status), deadline);
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
                                     const ProbeLaunchOptions& options) {
  require_probe_process_generation();
#if defined(_WIN32)
  (void)deferred_windows_job_reaper();
#else
  (void)deferred_process_reaper();
#endif
  auto reservation = std::make_shared<SupervisorSlot>();
  wait_for_worker_launch_delay(options.supervisor_start_delay, public_deadline);
#if defined(_WIN32)
  return run_probe_worker(worker_path, request, worker_deadline, public_deadline, options,
                          std::move(reservation));
#else
  return run_probe_worker(worker_path, request, worker_deadline, public_deadline, options,
                          std::move(reservation));
#endif
}

GpuVideoProbe probe_gpu_video_with_delays(const GpuFileDecodeConfig& config,
                                          const std::filesystem::path& worker_path,
                                          std::uint64_t timeout_ns,
                                          const ProbeLaunchOptions& options) {
  require_probe_process_generation();
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
      run_probe_worker_bounded(worker_path, request, worker_deadline, public_deadline, options));
}

} // namespace

#if !defined(_WIN32)
int detail::run_gpu_video_probe_supervisor(const char* executable,
                                           std::uint64_t pre_worker_report_delay_ns,
                                           std::uint64_t pre_guardian_exec_delay_ns,
                                           std::uint64_t caller_pid, bool has_marker) {
  sigset_t empty_mask{};
  if (sigemptyset(&empty_mask) != 0 || ::pthread_sigmask(SIG_SETMASK, &empty_mask, nullptr) != 0) {
    return 2;
  }
  if (executable == nullptr || executable[0] == '\0' || caller_pid == 0 ||
      caller_pid > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max()) ||
      pre_worker_report_delay_ns >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      pre_guardian_exec_delay_ns >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return 2;
  }
  const auto maximum_descriptor = descriptor_scan_limit();
  if (maximum_descriptor < kSupervisorFirstUnusedDescriptor) {
    return 2;
  }
  run_supervisor_child(executable, std::chrono::nanoseconds(pre_worker_report_delay_ns),
                       std::chrono::nanoseconds(pre_guardian_exec_delay_ns),
                       static_cast<pid_t>(caller_pid), has_marker, maximum_descriptor);
}

int detail::run_gpu_video_probe_guardian(const char* executable,
                                         std::uint64_t pre_worker_report_delay_ns) {
  sigset_t empty_mask{};
  if (sigemptyset(&empty_mask) != 0) {
    report_guardian_startup_error(errno);
  }
  const auto mask_error = ::pthread_sigmask(SIG_SETMASK, &empty_mask, nullptr);
  if (mask_error != 0) {
    report_guardian_startup_error(mask_error);
  }
  if (executable == nullptr || executable[0] == '\0' ||
      pre_worker_report_delay_ns >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    report_guardian_startup_error(EINVAL);
  }
#if defined(__APPLE__)
  if (apple_proc_listpids() == nullptr || apple_proc_pid_rusage() == nullptr) {
    report_guardian_startup_error(ENOSYS);
  }
#endif
  const auto maximum_descriptor = descriptor_scan_limit();
  if (maximum_descriptor < kGuardianFirstUnusedDescriptor) {
    report_guardian_startup_error(EMFILE);
  }
  run_guardian_child(executable, STDIN_FILENO, kGuardianWorkerInput, kGuardianWorkerOutput,
                     maximum_descriptor, std::chrono::nanoseconds(pre_worker_report_delay_ns));
}
#endif

GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config,
                              const std::filesystem::path& worker_path, std::uint64_t timeout_ns) {
  return probe_gpu_video_with_delays(config, worker_path, timeout_ns, {});
}

GpuVideoProbe detail::probe_gpu_video_with_supervisor_start_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t supervisor_start_delay_ns) {
  return probe_gpu_video_with_delays(
      config, worker_path, timeout_ns,
      ProbeLaunchOptions{.supervisor_start_delay =
                             std::chrono::nanoseconds(supervisor_start_delay_ns)});
}

GpuVideoProbe detail::probe_gpu_video_with_pre_worker_spawn_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_worker_spawn_delay_ns) {
  return probe_gpu_video_with_delays(
      config, worker_path, timeout_ns,
      ProbeLaunchOptions{.pre_worker_spawn_delay =
                             std::chrono::nanoseconds(pre_worker_spawn_delay_ns)});
}

GpuVideoProbe detail::probe_gpu_video_with_pre_supervisor_exec_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_supervisor_exec_delay_ns,
    const std::filesystem::path& marker_path) {
  return probe_gpu_video_with_delays(
      config, worker_path, timeout_ns,
      ProbeLaunchOptions{.pre_worker_spawn_delay =
                             std::chrono::nanoseconds(pre_supervisor_exec_delay_ns),
                         .pre_supervisor_exec_marker = marker_path});
}

GpuVideoProbe detail::probe_gpu_video_with_pre_worker_report_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_worker_report_delay_ns) {
  return probe_gpu_video_with_delays(
      config, worker_path, timeout_ns,
      ProbeLaunchOptions{.pre_worker_report_delay =
                             std::chrono::nanoseconds(pre_worker_report_delay_ns)});
}

std::uint64_t detail::maximum_probe_worker_address_space_bytes_for_test() {
  return kMaximumWorkerAddressSpaceBytes;
}

std::uint64_t detail::maximum_aggregate_probe_worker_address_space_bytes_for_test() {
  return kMaximumAggregateWorkerAddressSpaceBytes;
}

std::uint64_t detail::reserved_probe_worker_address_space_bytes_for_test() {
  require_probe_process_generation();
  auto& slots = supervisor_slots();
  std::lock_guard lock(slots.mutex);
  return slots.reserved_worker_address_space_bytes;
}

void detail::hold_probe_worker_memory_reservation_for_test(std::uint64_t hold_ns) {
  SupervisorSlot slot;
  std::this_thread::sleep_for(std::chrono::nanoseconds(hold_ns));
}

#if !defined(_WIN32)
void detail::hold_probe_worker_admission_lock_for_test(std::uint64_t hold_ns,
                                                       int ready_descriptor) {
  require_probe_process_generation();
  auto& slots = supervisor_slots();
  std::lock_guard lock(slots.mutex);
  const char ready = 'R';
  ssize_t written = -1;
  do {
    written = ::write(ready_descriptor, &ready, 1);
  } while (written < 0 && errno == EINTR);
  if (written != 1) {
    throw GpuVideoProbeError("failed to report the held video probe admission lock");
  }
  std::this_thread::sleep_for(std::chrono::nanoseconds(hold_ns));
}

GpuVideoProbe detail::probe_gpu_video_with_pre_guardian_exec_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_guardian_exec_delay_ns,
    const std::filesystem::path& marker_path) {
  return probe_gpu_video_with_delays(
      config, worker_path, timeout_ns,
      ProbeLaunchOptions{.pre_guardian_exec_delay =
                             std::chrono::nanoseconds(pre_guardian_exec_delay_ns),
                         .pre_guardian_exec_marker = marker_path});
}

bool detail::guardian_watchdog_exit_is_fatal_for_test(bool memory_termination_sent, int wait_result,
                                                      int wait_error, std::int64_t observed_pid,
                                                      std::int64_t watchdog_pid) {
  if (observed_pid < std::numeric_limits<pid_t>::min() ||
      observed_pid > std::numeric_limits<pid_t>::max() ||
      watchdog_pid < std::numeric_limits<pid_t>::min() ||
      watchdog_pid > std::numeric_limits<pid_t>::max()) {
    return true;
  }
  return guardian_watchdog_exit_is_fatal(memory_termination_sent, wait_result, wait_error,
                                         static_cast<pid_t>(observed_pid),
                                         static_cast<pid_t>(watchdog_pid));
}
#endif

#if defined(__linux__) && defined(__x86_64__)
bool detail::probe_worker_rejects_x32_syscalls_for_test() {
  const auto child = ::fork();
  if (child == 0) {
    constexpr rlimit no_core{.rlim_cur = 0, .rlim_max = 0};
    if (::setrlimit(RLIMIT_CORE, &no_core) != 0) {
      ::_exit(1);
    }
    if (!guardian_restrict_worker_process_creation()) {
      ::_exit(2);
    }
    constexpr auto hostile_system_call =
        static_cast<long>(static_cast<std::uint32_t>(SYS_getpid) | 0x40000000U);
    (void)::syscall(hostile_system_call);
    ::_exit(3);
  }
  if (child < 0) {
    return false;
  }
  int status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  return waited == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGSYS;
}
#endif

} // namespace reco::io
