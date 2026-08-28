#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "calibration_worker_internal.hpp"

#include "calibration_worker_protocol.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <csignal>
#include <dirent.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/magic.h>
#include <linux/memfd.h>
#include <linux/sched.h>
#include <linux/seccomp.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <unistd.h>

extern char** environ;
#endif

#ifndef __has_feature
#define __has_feature(value) 0
#endif

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) ||                               \
    __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define RECO_CALIBRATION_WIDE_ADDRESS_SANITIZER 1
#endif

#if defined(__SANITIZE_THREAD__) || __has_feature(thread_sanitizer)
#define RECO_CALIBRATION_THREAD_SANITIZER 1
extern "C" void __sanitizer_syscall_pre_impl_fork() noexcept;
extern "C" void __sanitizer_syscall_post_impl_fork(long result) noexcept;
#endif

namespace reco::calibrate::detail {

#if !defined(__linux__)

CalibrationResult run_gpu_calibration_supervised(const GpuCalibrationRequest&) {
#if defined(_WIN32)
  throw CalibrationExecutionError(
      "isolated GPU calibration is not implemented on Windows and fails closed");
#else
  throw CalibrationExecutionError(
      "isolated GPU calibration is not implemented on this POSIX platform and fails closed");
#endif
}

int run_calibration_guardian_fd(int, const char*, std::uint64_t) { return EXIT_FAILURE; }

bool install_calibration_worker_sandbox() noexcept { return false; }

#else
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kCleanupReserve = std::chrono::milliseconds(500);
constexpr std::uint64_t kAdmissionHeadroomBytes = 512ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaximumWorkerExecutableBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::string_view kAdmissionFileName = "reco-video-stitcher-calibration.lock";
constexpr std::string_view kCgroupPrefix = "reco-calibration-";
constexpr std::string_view kCleanupCgroupPrefix = "reco-calibration-cleanup-";
constexpr std::string_view kSandboxPrefix = "reco-calibration-sandbox-";
#if defined(__x86_64__)
constexpr std::uint32_t kSeccompArchitecture = AUDIT_ARCH_X86_64;
constexpr std::uint32_t kX32SyscallBit = 0x40000000U;
#elif defined(__aarch64__)
constexpr std::uint32_t kSeccompArchitecture = AUDIT_ARCH_AARCH64;
#endif

class UniqueFd {
public:
  UniqueFd() = default;
  explicit UniqueFd(int descriptor) : descriptor_(descriptor) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset();
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }
  ~UniqueFd() { reset(); }

  [[nodiscard]] int get() const { return descriptor_; }
  [[nodiscard]] explicit operator bool() const { return descriptor_ >= 0; }
  [[nodiscard]] int release() noexcept { return std::exchange(descriptor_, -1); }
  void reset(int descriptor = -1) noexcept {
    if (descriptor_ >= 0) {
      (void)::close(descriptor_);
    }
    descriptor_ = descriptor;
  }

private:
  int descriptor_ = -1;
};

class CgroupRemovalGuard {
public:
  CgroupRemovalGuard(int parent, const std::string& name) noexcept
      : parent_(parent), name_(&name) {}
  CgroupRemovalGuard(const CgroupRemovalGuard&) = delete;
  CgroupRemovalGuard& operator=(const CgroupRemovalGuard&) = delete;
  ~CgroupRemovalGuard() {
    if (!armed_) {
      return;
    }
    int removed = -1;
    do {
      removed = ::unlinkat(parent_, name_->c_str(), AT_REMOVEDIR);
    } while (removed < 0 && errno == EINTR);
  }

  void dismiss() noexcept { armed_ = false; }

private:
  int parent_ = -1;
  const std::string* name_ = nullptr;
  bool armed_ = true;
};

[[nodiscard]] std::string errno_message(std::string_view operation, int error = errno) {
  return std::string(operation) + ": " + std::strerror(error);
}

[[nodiscard]] bool inject_cgroup_setup_failure(std::string_view point) noexcept {
  const char* configured = std::getenv("RECO_FAKE_CALIBRATION_CGROUP_SETUP_FAILURE");
  return configured != nullptr && std::string_view(configured) == point;
}

[[nodiscard]] std::uint64_t time_point_nanoseconds(Clock::time_point value) {
  const auto count =
      std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count();
  if (count < 0) {
    throw CalibrationExecutionError("calibration deadline is out of range");
  }
  return static_cast<std::uint64_t>(count);
}

[[nodiscard]] Clock::time_point deadline_from_nanoseconds(std::uint64_t value) {
  using Duration = Clock::duration;
  const auto maximum = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Duration::max()).count());
  if (value > maximum) {
    throw CalibrationExecutionError("calibration deadline is out of range");
  }
  return Clock::time_point(std::chrono::duration_cast<Duration>(std::chrono::nanoseconds(value)));
}

[[nodiscard]] int deadline_timeout(Clock::time_point deadline) {
  if (Clock::now() >= deadline) {
    return 0;
  }
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
  return static_cast<int>(std::clamp<std::int64_t>(remaining, 1, 100));
}

void signal_pidfd_noexcept(int descriptor, int signal) noexcept {
#if defined(SYS_pidfd_send_signal)
  while (::syscall(SYS_pidfd_send_signal, descriptor, signal, nullptr, 0U) != 0 && errno == EINTR) {
  }
#else
  (void)descriptor;
  (void)signal;
#endif
}

struct ProcessExit {
  bool known = false;
  int code = 0;
  int status = 0;
};

class DeferredProcessReaper {
public:
  DeferredProcessReaper() {
    std::thread([this] { run(); }).detach();
  }
  DeferredProcessReaper(const DeferredProcessReaper&) = delete;
  DeferredProcessReaper& operator=(const DeferredProcessReaper&) = delete;

  [[nodiscard]] bool add(int descriptor) noexcept {
    std::lock_guard lock(mutex_);
    const auto available = std::find(descriptors_.begin(), descriptors_.end(), -1);
    if (available == descriptors_.end()) {
      return false;
    }
    *available = descriptor;
    ready_.notify_one();
    return true;
  }

private:
  void run() noexcept {
    std::unique_lock lock(mutex_);
    while (true) {
      ready_.wait(lock, [this] {
        return std::any_of(descriptors_.begin(), descriptors_.end(),
                           [](int descriptor) { return descriptor >= 0; });
      });
      bool pending = false;
      for (auto& descriptor : descriptors_) {
        if (descriptor < 0) {
          continue;
        }
        pollfd item{.fd = descriptor, .events = POLLIN, .revents = 0};
        const auto polled = ::poll(&item, 1, 0);
        if (polled <= 0) {
          pending = true;
          continue;
        }
        const auto completed = std::exchange(descriptor, -1);
        lock.unlock();
        siginfo_t information{};
        constexpr auto pidfd_id_type = static_cast<idtype_t>(3);
        while (::waitid(pidfd_id_type, static_cast<id_t>(completed), &information,
                        WEXITED | __WALL) != 0 &&
               errno == EINTR) {
        }
        (void)::close(completed);
        lock.lock();
      }
      if (pending) {
        ready_.wait_for(lock, std::chrono::milliseconds(10));
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::array<int, 128> descriptors_ = [] {
    std::array<int, 128> result{};
    result.fill(-1);
    return result;
  }();
};

DeferredProcessReaper& deferred_process_reaper() {
  static auto* reaper = new DeferredProcessReaper;
  return *reaper;
}

void defer_process_reap_noexcept(UniqueFd& pidfd) noexcept {
  if (!pidfd) {
    return;
  }
  try {
    if (deferred_process_reaper().add(pidfd.get())) {
      (void)pidfd.release();
    }
  } catch (...) {
  }
}

class OwnedProcess {
public:
  OwnedProcess() = default;
  OwnedProcess(pid_t pid, int pidfd, Clock::time_point cleanup_deadline = Clock::time_point::min())
      : pid_(pid), pidfd_(pidfd), cleanup_deadline_(cleanup_deadline) {}
  OwnedProcess(pid_t pid, int pidfd, std::thread launcher, int launcher_stop,
               Clock::time_point cleanup_deadline)
      : pid_(pid), pidfd_(pidfd), launcher_(std::move(launcher)), launcher_stop_(launcher_stop),
        cleanup_deadline_(cleanup_deadline) {}
  OwnedProcess(const OwnedProcess&) = delete;
  OwnedProcess& operator=(const OwnedProcess&) = delete;
  OwnedProcess(OwnedProcess&& other) noexcept
      : pid_(std::exchange(other.pid_, -1)), pidfd_(std::move(other.pidfd_)),
        reaped_(std::exchange(other.reaped_, true)), launcher_(std::move(other.launcher_)),
        launcher_stop_(std::move(other.launcher_stop_)),
        cleanup_deadline_(other.cleanup_deadline_) {}
  OwnedProcess& operator=(OwnedProcess&& other) noexcept {
    if (this != &other) {
      terminate_and_reap_noexcept();
      stop_launcher_noexcept();
      pid_ = std::exchange(other.pid_, -1);
      pidfd_ = std::move(other.pidfd_);
      reaped_ = std::exchange(other.reaped_, true);
      launcher_ = std::move(other.launcher_);
      launcher_stop_ = std::move(other.launcher_stop_);
      cleanup_deadline_ = other.cleanup_deadline_;
    }
    return *this;
  }
  ~OwnedProcess() {
    terminate_and_reap_noexcept();
    stop_launcher_noexcept();
  }

  [[nodiscard]] pid_t pid() const { return pid_; }
  [[nodiscard]] int pidfd() const { return pidfd_.get(); }
  [[nodiscard]] int release_pidfd() noexcept {
    reaped_ = true;
    return pidfd_.release();
  }
  void detach_noexcept() noexcept {
    defer_process_reap_noexcept(pidfd_);
    reaped_ = true;
    stop_launcher_noexcept();
  }

  [[nodiscard]] bool exited() const {
    pollfd item{.fd = pidfd_.get(), .events = POLLIN, .revents = 0};
    int result = -1;
    do {
      result = ::poll(&item, 1, 0);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      throw CalibrationExecutionError(errno_message("cannot inspect calibration process pidfd"));
    }
    return result > 0 && (item.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
  }

  void wait_until(Clock::time_point deadline) const {
    while (!exited()) {
      if (Clock::now() >= deadline) {
        throw CalibrationExecutionError("calibration worker exceeded its end-to-end deadline");
      }
      pollfd item{.fd = pidfd_.get(), .events = POLLIN, .revents = 0};
      const auto result = ::poll(&item, 1, deadline_timeout(deadline));
      if (result < 0 && errno != EINTR) {
        throw CalibrationExecutionError(errno_message("cannot wait for calibration process"));
      }
    }
  }

  [[nodiscard]] ProcessExit reap() noexcept {
    ProcessExit result;
    if (reaped_ || !pidfd_) {
      return result;
    }
    siginfo_t information{};
    constexpr auto pidfd_id_type = static_cast<idtype_t>(3);
    while (::waitid(pidfd_id_type, static_cast<id_t>(pidfd_.get()), &information,
                    WEXITED | __WALL) != 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == ECHILD) {
        reaped_ = true;
      }
      return result;
    }
    result = {.known = true, .code = information.si_code, .status = information.si_status};
    reaped_ = true;
    stop_launcher_noexcept();
    return result;
  }

private:
  void terminate_and_reap_noexcept() noexcept {
    if (reaped_ || !pidfd_) {
      return;
    }
    signal_pidfd_noexcept(pidfd_.get(), SIGKILL);
    pollfd item{.fd = pidfd_.get(), .events = POLLIN, .revents = 0};
    const auto reserve_deadline = Clock::now() + kCleanupReserve;
    const auto deadline = cleanup_deadline_ == Clock::time_point::min()
                              ? reserve_deadline
                              : std::min(reserve_deadline, cleanup_deadline_);
    while (Clock::now() < deadline) {
      const auto result = ::poll(&item, 1, deadline_timeout(deadline));
      if (result > 0) {
        (void)reap();
        return;
      }
      if (result < 0 && errno != EINTR) {
        break;
      }
    }
    defer_process_reap_noexcept(pidfd_);
    reaped_ = true;
  }

  void stop_launcher_noexcept() noexcept {
    if (launcher_stop_) {
      const char stop = 'S';
      ssize_t written = -1;
      do {
        written = ::send(launcher_stop_.get(), &stop, 1, MSG_NOSIGNAL);
      } while (written < 0 && errno == EINTR);
      (void)written;
      launcher_stop_.reset();
    }
    if (launcher_.joinable()) {
      launcher_.join();
    }
  }

  pid_t pid_ = -1;
  UniqueFd pidfd_;
  bool reaped_ = false;
  std::thread launcher_;
  UniqueFd launcher_stop_;
  Clock::time_point cleanup_deadline_ = Clock::time_point::min();
};

[[noreturn]] void cgroup_cleanup_exit(int status) noexcept {
  (void)::syscall(SYS_exit_group, status);
  __builtin_unreachable();
}

void close_cgroup_cleanup_descriptors(int parent, int lifetime, int caller_pidfd,
                                      int return_cgroup) noexcept {
#if defined(SYS_close_range)
  std::array<int, 4> preserved{parent, lifetime, caller_pidfd, return_cgroup};
  for (std::size_t index = 0; index < preserved.size(); ++index) {
    for (std::size_t next = index + 1; next < preserved.size(); ++next) {
      if (preserved[next] < preserved[index]) {
        std::swap(preserved[index], preserved[next]);
      }
    }
  }
  unsigned int first = 3;
  for (const auto descriptor : preserved) {
    if (descriptor < 3) {
      cgroup_cleanup_exit(EXIT_FAILURE);
    }
    const auto current = static_cast<unsigned int>(descriptor);
    if (first < current && ::syscall(SYS_close_range, first, current - 1U, 0U) != 0) {
      cgroup_cleanup_exit(EXIT_FAILURE);
    }
    first = current + 1U;
  }
  if (::syscall(SYS_close_range, first, std::numeric_limits<unsigned int>::max(), 0U) != 0) {
    cgroup_cleanup_exit(EXIT_FAILURE);
  }
#else
  (void)parent;
  (void)lifetime;
  (void)caller_pidfd;
  (void)return_cgroup;
  cgroup_cleanup_exit(EXIT_FAILURE);
#endif
}

[[noreturn]] void cgroup_cleanup_child(int parent, int lifetime, int caller_pidfd,
                                       int return_cgroup, const char* name,
                                       const char* cleanup_name) noexcept {
  close_cgroup_cleanup_descriptors(parent, lifetime, caller_pidfd, return_cgroup);

  std::array<pollfd, 2> lifetime_events{
      pollfd{.fd = lifetime, .events = POLLIN, .revents = 0},
      pollfd{.fd = caller_pidfd, .events = POLLIN, .revents = 0},
  };
  while (::poll(lifetime_events.data(), lifetime_events.size(), -1) < 0 && errno == EINTR) {
  }
  (void)::close(lifetime);
  (void)::close(caller_pidfd);

  while (true) {
    const auto directory = ::openat(parent, name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (directory < 0) {
      const auto status = errno == ENOENT ? EXIT_SUCCESS : EXIT_FAILURE;
      if (status != EXIT_SUCCESS) {
        (void)::close(parent);
        cgroup_cleanup_exit(status);
      }
      break;
    }
    const auto kill = ::openat(directory, "cgroup.kill", O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (kill >= 0) {
      const char value = '1';
      ssize_t written = -1;
      do {
        written = ::write(kill, &value, 1);
      } while (written < 0 && errno == EINTR);
      (void)written;
      (void)::close(kill);
    }
    const auto events = ::openat(directory, "cgroup.events", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (events >= 0) {
      std::array<char, 128> contents{};
      (void)::lseek(events, 0, SEEK_SET);
      const auto size = ::read(events, contents.data(), contents.size() - 1U);
      if (size > 0 &&
          std::string_view(contents.data(), static_cast<std::size_t>(size)).find("populated 0") !=
              std::string_view::npos) {
        (void)::close(events);
      } else {
        pollfd item{.fd = events, .events = POLLPRI, .revents = 0};
        (void)::poll(&item, 1, 10);
        (void)::close(events);
      }
    }
    (void)::close(directory);
    if (::unlinkat(parent, name, AT_REMOVEDIR) == 0 || errno == ENOENT) {
      break;
    }
    if (errno != EBUSY && errno != ENOTEMPTY) {
      (void)::close(parent);
      cgroup_cleanup_exit(EXIT_FAILURE);
    }
    pollfd delay{.fd = -1, .events = 0, .revents = 0};
    (void)::poll(&delay, 0, 10);
  }
  const auto processes = ::openat(return_cgroup, "cgroup.procs", O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
  if (processes < 0) {
    (void)::close(return_cgroup);
    (void)::close(parent);
    cgroup_cleanup_exit(EXIT_FAILURE);
  }
  const char self[] = "0";
  ssize_t migrated = -1;
  do {
    migrated = ::write(processes, self, sizeof(self) - 1U);
  } while (migrated < 0 && errno == EINTR);
  (void)::close(processes);
  (void)::close(return_cgroup);
  if (migrated != static_cast<ssize_t>(sizeof(self) - 1U)) {
    (void)::close(parent);
    cgroup_cleanup_exit(EXIT_FAILURE);
  }
  int removed = -1;
  do {
    removed = ::unlinkat(parent, cleanup_name, AT_REMOVEDIR);
  } while (removed < 0 && errno == EINTR);
  const auto status = removed == 0 || errno == ENOENT ? EXIT_SUCCESS : EXIT_FAILURE;
  (void)::close(parent);
  cgroup_cleanup_exit(status);
}

[[nodiscard]] OwnedProcess spawn_cgroup_cleanup_process(int parent, int cleanup_cgroup,
                                                        int return_cgroup, int lifetime,
                                                        int caller_pidfd, const char* name,
                                                        const char* cleanup_name) {
#if !defined(SYS_clone3)
  (void)parent;
  (void)lifetime;
  (void)caller_pidfd;
  (void)cleanup_cgroup;
  (void)return_cgroup;
  (void)name;
  (void)cleanup_name;
  throw CalibrationExecutionError(
      "Linux clone3 is unavailable; calibration cgroup cleanup fails closed");
#else
  int pidfd = -1;
  clone_args arguments{};
  arguments.flags = CLONE_PIDFD | CLONE_INTO_CGROUP;
  arguments.cgroup = static_cast<std::uint64_t>(cleanup_cgroup);
  arguments.pidfd = reinterpret_cast<std::uint64_t>(&pidfd);
  // A clone child cannot be reaped by an unrelated waitpid(-1) caller and is
  // unaffected by a process-wide SIGCHLD disposition.
  arguments.exit_signal = 0;
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
  __sanitizer_syscall_pre_impl_fork();
#endif
  const auto result = static_cast<pid_t>(::syscall(SYS_clone3, &arguments, sizeof(arguments)));
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
  __sanitizer_syscall_post_impl_fork(result);
#endif
  if (result == 0) {
    cgroup_cleanup_child(parent, lifetime, caller_pidfd, return_cgroup, name, cleanup_name);
  }
  if (result < 0 || pidfd < 0) {
    throw CalibrationExecutionError(
        errno_message("cannot create calibration cgroup cleanup authority"));
  }
  return OwnedProcess(result, pidfd);
#endif
}

class ExternalProcessAuthority {
public:
  ExternalProcessAuthority() = default;
  ExternalProcessAuthority(int descriptor, Clock::time_point cleanup_deadline)
      : pidfd_(descriptor), cleanup_deadline_(cleanup_deadline) {}
  ExternalProcessAuthority(const ExternalProcessAuthority&) = delete;
  ExternalProcessAuthority& operator=(const ExternalProcessAuthority&) = delete;
  ExternalProcessAuthority(ExternalProcessAuthority&&) noexcept = default;
  ExternalProcessAuthority& operator=(ExternalProcessAuthority&&) noexcept = default;
  ~ExternalProcessAuthority() {
    if (pidfd_) {
      signal_pidfd_noexcept(pidfd_.get(), SIGKILL);
      pollfd item{.fd = pidfd_.get(), .events = POLLIN, .revents = 0};
      while (Clock::now() < cleanup_deadline_) {
        const auto result = ::poll(&item, 1, deadline_timeout(cleanup_deadline_));
        if (result != 0) {
          break;
        }
      }
    }
  }

private:
  UniqueFd pidfd_;
  Clock::time_point cleanup_deadline_ = Clock::time_point::min();
};

class AdmissionLock {
public:
  AdmissionLock() {
    struct stat directory_status{};
    if (::lstat("/tmp", &directory_status) != 0 || !S_ISDIR(directory_status.st_mode)) {
      throw CalibrationExecutionError(
          "a root-owned sticky temporary directory is required for calibration admission");
    }
    const bool root_sticky =
        directory_status.st_uid == 0 && (directory_status.st_mode & S_ISVTX) != 0;
    if (!root_sticky && directory_status.st_uid != ::getuid()) {
      throw CalibrationExecutionError(
          "a root-owned sticky temporary directory is required for calibration admission");
    }
    const auto path = std::filesystem::path("/tmp") /
                      (std::string(kAdmissionFileName) + "-" + std::to_string(::getuid()));
    descriptor_.reset(
        ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR));
    if (!descriptor_) {
      throw CalibrationExecutionError(errno_message("cannot open calibration admission lock"));
    }
    struct stat status{};
    if (::fstat(descriptor_.get(), &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_uid != ::getuid()) {
      throw CalibrationExecutionError("calibration admission lock has unsafe ownership or type");
    }
    if (::flock(descriptor_.get(), LOCK_EX | LOCK_NB) != 0) {
      if (errno == EWOULDBLOCK) {
        throw CalibrationExecutionError(
            "another calibration holds the cross-process host-memory admission reservation");
      }
      throw CalibrationExecutionError(errno_message("cannot acquire calibration admission lock"));
    }
  }

  [[nodiscard]] int fd() const { return descriptor_.get(); }

private:
  UniqueFd descriptor_;
};

[[nodiscard]] std::optional<std::uint64_t> parse_memory_control(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string value;
  if (!(input >> value) || value == "max") {
    return std::nullopt;
  }
  std::uint64_t parsed = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  return error == std::errc{} && end == value.data() + value.size()
             ? std::optional<std::uint64_t>(parsed)
             : std::nullopt;
}

[[nodiscard]] std::optional<std::uint64_t> linux_mem_available() {
  std::ifstream input("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (input >> key >> value >> unit) {
    if (key == "MemAvailable:" && unit == "kB" &&
        value <= std::numeric_limits<std::uint64_t>::max() / 1024ULL) {
      return value * 1024ULL;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::uint64_t> cgroup_v2_available() {
  std::ifstream membership("/proc/self/cgroup");
  std::string line;
  while (std::getline(membership, line)) {
    if (line.rfind("0::", 0) != 0) {
      continue;
    }
    const auto root = std::filesystem::path("/sys/fs/cgroup");
    auto path = root / std::filesystem::path(line.substr(3)).relative_path();
    std::optional<std::uint64_t> available;
    while (true) {
      auto limit = parse_memory_control(path / "memory.max");
      if (const auto high = parse_memory_control(path / "memory.high"); high.has_value()) {
        limit = limit.has_value() ? std::min(*limit, *high) : high;
      }
      if (limit.has_value()) {
        const auto current = parse_memory_control(path / "memory.current");
        if (!current.has_value()) {
          throw CalibrationExecutionError("cannot determine effective cgroup memory usage");
        }
        const auto headroom = *current >= *limit ? 0 : *limit - *current;
        available = available.has_value() ? std::min(*available, headroom) : headroom;
      }
      if (path == root) {
        break;
      }
      const auto parent = path.parent_path();
      if (parent.empty() || parent == path || !parent.string().starts_with(root.string())) {
        throw CalibrationExecutionError("cannot determine effective cgroup memory ancestry");
      }
      path = parent;
    }
    return available;
  }
  return std::nullopt;
}

void check_admission_headroom(std::uint64_t requested) {
  auto available = linux_mem_available();
  if (!available.has_value()) {
    throw CalibrationExecutionError(
        "cannot determine available host memory; calibration fails closed");
  }
  if (const auto cgroup = cgroup_v2_available(); cgroup.has_value()) {
    available = std::min(*available, *cgroup);
  }
  if (requested > std::numeric_limits<std::uint64_t>::max() - kAdmissionHeadroomBytes ||
      *available < requested + kAdmissionHeadroomBytes) {
    throw CalibrationExecutionError(
        "insufficient host-memory headroom to admit the calibration worker");
  }
}

[[nodiscard]] bool control_contains_token(const std::filesystem::path& path,
                                          std::string_view token) {
  std::ifstream input(path);
  std::string value;
  while (input >> value) {
    if (value == token) {
      return true;
    }
  }
  return false;
}

void write_cgroup_control(int directory, const char* name, std::string_view value,
                          bool required = true) {
  UniqueFd descriptor(::openat(directory, name, O_WRONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!descriptor) {
    if (!required && errno == ENOENT) {
      return;
    }
    throw CalibrationExecutionError(errno_message(std::string("cannot open cgroup ") + name));
  }
  std::size_t offset = 0;
  while (offset < value.size()) {
    ssize_t written = -1;
    do {
      written = ::write(descriptor.get(), value.data() + offset, value.size() - offset);
    } while (written < 0 && errno == EINTR);
    if (written <= 0) {
      throw CalibrationExecutionError(
          errno_message(std::string("cannot configure cgroup ") + name));
    }
    offset += static_cast<std::size_t>(written);
  }
}

[[nodiscard]] std::string read_cgroup_control(int directory, const char* name) {
  UniqueFd descriptor(::openat(directory, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!descriptor) {
    throw CalibrationExecutionError(errno_message(std::string("cannot read cgroup ") + name));
  }
  std::array<char, 256> contents{};
  ssize_t size = -1;
  do {
    size = ::read(descriptor.get(), contents.data(), contents.size() - 1U);
  } while (size < 0 && errno == EINTR);
  if (size <= 0 || static_cast<std::size_t>(size) == contents.size()) {
    throw CalibrationExecutionError(std::string("cgroup ") + name + " is invalid");
  }
  return std::string(contents.data(), static_cast<std::size_t>(size));
}

void require_cgroup_control_value(int directory, const char* name, std::uint64_t expected) {
  const auto configured = read_cgroup_control(directory, name);
  std::uint64_t parsed = 0;
  const auto [end, error] =
      std::from_chars(configured.data(), configured.data() + configured.size(), parsed);
  if (error != std::errc{} || end == configured.data() || parsed != expected) {
    throw CalibrationExecutionError(std::string("calibration cgroup ") + name +
                                    " was not installed");
  }
}

[[nodiscard]] std::string random_cgroup_name() {
  std::array<unsigned char, 12> random{};
  std::size_t offset = 0;
  while (offset < random.size()) {
    const auto received = ::getrandom(random.data() + offset, random.size() - offset, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      throw CalibrationExecutionError(errno_message("cannot create calibration cgroup identity"));
    }
    offset += static_cast<std::size_t>(received);
  }
  constexpr char digits[] = "0123456789abcdef";
  std::string name(kCgroupPrefix);
  name.reserve(name.size() + random.size() * 2U);
  for (const auto byte : random) {
    name.push_back(digits[byte >> 4U]);
    name.push_back(digits[byte & 0x0fU]);
  }
  return name;
}

[[nodiscard]] bool cgroup_oom_killed_noexcept(int directory) noexcept {
  const auto descriptor = ::openat(directory, "memory.events", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return false;
  }
  std::array<char, 512> contents{};
  ssize_t size = -1;
  do {
    size = ::read(descriptor, contents.data(), contents.size() - 1U);
  } while (size < 0 && errno == EINTR);
  (void)::close(descriptor);
  if (size <= 0) {
    return false;
  }
  const std::string_view text(contents.data(), static_cast<std::size_t>(size));
  const auto marker = text.find("oom_kill ");
  if (marker == std::string_view::npos) {
    return false;
  }
  const auto begin = marker + std::string_view("oom_kill ").size();
  std::uint64_t count = 0;
  const auto [end, error] = std::from_chars(text.data() + begin, text.data() + text.size(), count);
  return error == std::errc{} && end != text.data() + begin && count != 0;
}

void scavenge_stale_calibration_cgroups(int parent) {
  const auto duplicate = ::fcntl(parent, F_DUPFD_CLOEXEC, 3);
  if (duplicate < 0) {
    throw CalibrationExecutionError(errno_message("cannot scan stale calibration cgroups"));
  }
  auto* stream = ::fdopendir(duplicate);
  if (stream == nullptr) {
    (void)::close(duplicate);
    throw CalibrationExecutionError(errno_message("cannot scan stale calibration cgroups"));
  }
  std::vector<std::string> names;
  errno = 0;
  while (const auto* entry = ::readdir(stream)) {
    const std::string_view name(entry->d_name);
    if (name.starts_with(kCgroupPrefix)) {
      names.emplace_back(name);
    }
    errno = 0;
  }
  const auto read_error = errno;
  (void)::closedir(stream);
  if (read_error != 0) {
    throw CalibrationExecutionError(
        errno_message("cannot scan stale calibration cgroups", read_error));
  }

  const auto deadline = Clock::now() + kCleanupReserve;
  for (const auto& name : names) {
    UniqueFd directory(
        ::openat(parent, name.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    struct stat status{};
    if (!directory || ::fstat(directory.get(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        status.st_uid != ::geteuid()) {
      continue;
    }
    UniqueFd kill(::openat(directory.get(), "cgroup.kill", O_WRONLY | O_CLOEXEC | O_NOFOLLOW));
    if (kill) {
      const char value = '1';
      ssize_t written = -1;
      do {
        written = ::write(kill.get(), &value, 1);
      } while (written < 0 && errno == EINTR);
    }
    kill.reset();
    directory.reset();
    while (::unlinkat(parent, name.c_str(), AT_REMOVEDIR) != 0) {
      if (errno == ENOENT) {
        break;
      }
      if ((errno != EBUSY && errno != ENOTEMPTY) || Clock::now() >= deadline) {
        throw CalibrationExecutionError("cannot scavenge a stale calibration cgroup");
      }
      pollfd delay{.fd = -1, .events = 0, .revents = 0};
      (void)::poll(&delay, 0, 10);
    }
  }
}

class CgroupMemoryBoundary {
public:
  CgroupMemoryBoundary(std::uint64_t memory_limit, Clock::time_point teardown_deadline)
      : teardown_deadline_(teardown_deadline) {
    std::ifstream membership("/proc/self/cgroup");
    std::string line;
    std::optional<std::filesystem::path> current;
    while (std::getline(membership, line)) {
      if (line.rfind("0::", 0) == 0) {
        current = std::filesystem::path("/sys/fs/cgroup") /
                  std::filesystem::path(line.substr(3)).relative_path();
        break;
      }
    }
    if (!current.has_value()) {
      throw CalibrationExecutionError(
          "a delegated cgroup-v2 memory controller is required for calibration");
    }
    UniqueFd return_cgroup(
        ::open(current->c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (!return_cgroup) {
      throw CalibrationExecutionError(errno_message("cannot retain the calibration caller cgroup"));
    }

    const std::filesystem::path root("/sys/fs/cgroup");
    for (auto candidate = *current;; candidate = candidate.parent_path()) {
      if (control_contains_token(candidate / "cgroup.subtree_control", "memory") &&
          control_contains_token(candidate / "cgroup.subtree_control", "pids") &&
          create_under(candidate, return_cgroup.get(), memory_limit)) {
        return;
      }
      if (candidate == root || candidate.empty()) {
        break;
      }
    }
    throw CalibrationExecutionError(
        "a delegated cgroup-v2 memory controller is required for calibration");
  }

  CgroupMemoryBoundary(const CgroupMemoryBoundary&) = delete;
  CgroupMemoryBoundary& operator=(const CgroupMemoryBoundary&) = delete;
  CgroupMemoryBoundary(CgroupMemoryBoundary&&) = delete;
  CgroupMemoryBoundary& operator=(CgroupMemoryBoundary&&) = delete;
  ~CgroupMemoryBoundary() {
    try {
      finish();
    } catch (...) {
      cleanup_process_.detach_noexcept();
    }
  }

  [[nodiscard]] int fd() const { return directory_.get(); }

  [[nodiscard]] bool oom_killed() const noexcept {
    return cgroup_oom_killed_noexcept(directory_.get());
  }

  void finish() {
    if (finished_) {
      return;
    }
    cleanup_lifetime_.reset();
    cleanup_process_.wait_until(teardown_deadline_);
    const auto status = cleanup_process_.reap();
    if (!status.known || status.code != CLD_EXITED || status.status != EXIT_SUCCESS) {
      throw CalibrationExecutionError("calibration cgroup cleanup authority failed");
    }
    finished_ = true;
  }

private:
  [[nodiscard]] bool create_under(const std::filesystem::path& path, int return_cgroup,
                                  std::uint64_t memory_limit) {
    UniqueFd parent(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    struct statfs filesystem{};
    if (!parent || ::fstatfs(parent.get(), &filesystem) != 0 ||
        static_cast<unsigned long>(filesystem.f_type) != CGROUP2_SUPER_MAGIC) {
      return false;
    }
    scavenge_stale_calibration_cgroups(parent.get());
    auto name = random_cgroup_name();
    const auto cleanup_name =
        std::string(kCleanupCgroupPrefix) + random_cgroup_name().substr(kCgroupPrefix.size());
    if (::mkdirat(parent.get(), cleanup_name.c_str(), S_IRWXU) != 0) {
      if (errno == EACCES || errno == EPERM || errno == EROFS) {
        return false;
      }
      throw CalibrationExecutionError(errno_message("cannot create calibration cleanup cgroup"));
    }
    CgroupRemovalGuard cleanup_removal(parent.get(), cleanup_name);
    UniqueFd cleanup_directory(::openat(parent.get(), cleanup_name.c_str(),
                                        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (!cleanup_directory) {
      throw CalibrationExecutionError(errno_message("cannot open calibration cleanup cgroup"));
    }
    constexpr std::uint64_t cleanup_memory = 32ULL * 1024ULL * 1024ULL;
    write_cgroup_control(cleanup_directory.get(), "memory.max", std::to_string(cleanup_memory));
    write_cgroup_control(cleanup_directory.get(), "memory.swap.max", "0");
    write_cgroup_control(cleanup_directory.get(), "memory.oom.group", "1");
    write_cgroup_control(cleanup_directory.get(), "pids.max", "1");
    require_cgroup_control_value(cleanup_directory.get(), "memory.max", cleanup_memory);
    require_cgroup_control_value(cleanup_directory.get(), "memory.swap.max", 0);
    require_cgroup_control_value(cleanup_directory.get(), "memory.oom.group", 1);
    require_cgroup_control_value(cleanup_directory.get(), "pids.max", 1);
    std::array<int, 2> lifetime{-1, -1};
    if (inject_cgroup_setup_failure("lifetime-pipe")) {
      throw CalibrationExecutionError(
          errno_message("cannot create calibration cgroup cleanup lifetime", EMFILE));
    }
    if (::pipe2(lifetime.data(), O_CLOEXEC) != 0) {
      throw CalibrationExecutionError(
          errno_message("cannot create calibration cgroup cleanup lifetime"));
    }
    UniqueFd lifetime_read(lifetime[0]);
    UniqueFd lifetime_write(lifetime[1]);
    int caller_pidfd_descriptor = -1;
    if (inject_cgroup_setup_failure("caller-pidfd")) {
      errno = EMFILE;
    } else {
#if defined(SYS_pidfd_open)
      caller_pidfd_descriptor = static_cast<int>(::syscall(SYS_pidfd_open, ::getpid(), 0U));
#else
      errno = ENOSYS;
#endif
    }
    UniqueFd caller_pidfd(caller_pidfd_descriptor);
    if (!caller_pidfd) {
      throw CalibrationExecutionError(
          errno_message("cannot retain calibration caller process authority"));
    }
    auto cleanup = spawn_cgroup_cleanup_process(
        parent.get(), cleanup_directory.get(), return_cgroup, lifetime_read.get(),
        caller_pidfd.get(), name.c_str(), cleanup_name.c_str());
    cleanup_removal.dismiss();
    lifetime_read.reset();
    caller_pidfd.reset();
    const auto stop_cleanup = [&]() noexcept {
      lifetime_write.reset();
      try {
        cleanup.wait_until(Clock::now() + kCleanupReserve);
        (void)cleanup.reap();
      } catch (...) {
        cleanup.detach_noexcept();
      }
    };
    if (::mkdirat(parent.get(), name.c_str(), S_IRWXU) != 0) {
      if (errno == EACCES || errno == EPERM || errno == EROFS) {
        stop_cleanup();
        return false;
      }
      stop_cleanup();
      throw CalibrationExecutionError(errno_message("cannot create calibration memory cgroup"));
    }
    UniqueFd directory(
        ::openat(parent.get(), name.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (!directory) {
      stop_cleanup();
      throw CalibrationExecutionError(errno_message("cannot open calibration memory cgroup"));
    }
    try {
      const auto limit = std::to_string(memory_limit);
      write_cgroup_control(directory.get(), "memory.max", limit);
      write_cgroup_control(directory.get(), "memory.swap.max", "0");
      write_cgroup_control(directory.get(), "memory.oom.group", "1");
      write_cgroup_control(directory.get(), "pids.max", "64");
      require_cgroup_control_value(directory.get(), "memory.max", memory_limit);
      require_cgroup_control_value(directory.get(), "memory.swap.max", 0);
      require_cgroup_control_value(directory.get(), "memory.oom.group", 1);
      require_cgroup_control_value(directory.get(), "pids.max", 64);

      cleanup_lifetime_ = std::move(lifetime_write);
      cleanup_process_ = std::move(cleanup);
    } catch (...) {
      directory.reset();
      stop_cleanup();
      throw;
    }
    parent_ = std::move(parent);
    directory_ = std::move(directory);
    name_ = std::move(name);
    return true;
  }

  UniqueFd parent_;
  UniqueFd directory_;
  std::string name_;
  UniqueFd cleanup_lifetime_;
  OwnedProcess cleanup_process_;
  Clock::time_point teardown_deadline_;
  bool finished_ = false;
};

[[noreturn]] void executable_snapshot_child(int source, int channel,
                                            const struct stat& expected) noexcept {
  close_cgroup_cleanup_descriptors(source, channel, source, channel);
#if !defined(SYS_memfd_create)
  cgroup_cleanup_exit(EXIT_FAILURE);
#else
  constexpr auto base_flags = static_cast<unsigned int>(MFD_CLOEXEC | MFD_ALLOW_SEALING);
  int snapshot = -1;
#if defined(MFD_EXEC)
  snapshot = static_cast<int>(
      ::syscall(SYS_memfd_create, "reco-calibration-worker", base_flags | MFD_EXEC));
  if (snapshot < 0 && errno == EINVAL) {
    snapshot = static_cast<int>(::syscall(SYS_memfd_create, "reco-calibration-worker", base_flags));
  }
#else
  snapshot = static_cast<int>(::syscall(SYS_memfd_create, "reco-calibration-worker", base_flags));
#endif
  if (snapshot < 0) {
    cgroup_cleanup_exit(EXIT_FAILURE);
  }
  std::array<char, 64U * 1024U> buffer{};
  std::uint64_t copied = 0;
  const auto expected_size = static_cast<std::uint64_t>(expected.st_size);
  while (copied < expected_size) {
    const auto read_size =
        static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), expected_size - copied));
    ssize_t received = -1;
    do {
      received = ::read(source, buffer.data(), read_size);
    } while (received < 0 && errno == EINTR);
    if (received <= 0 || static_cast<std::uint64_t>(received) > expected_size - copied) {
      (void)::close(snapshot);
      cgroup_cleanup_exit(EXIT_FAILURE);
    }
    std::size_t offset = 0;
    while (offset < static_cast<std::size_t>(received)) {
      ssize_t written = -1;
      do {
        written =
            ::write(snapshot, buffer.data() + offset, static_cast<std::size_t>(received) - offset);
      } while (written < 0 && errno == EINTR);
      if (written <= 0) {
        (void)::close(snapshot);
        cgroup_cleanup_exit(EXIT_FAILURE);
      }
      offset += static_cast<std::size_t>(written);
      copied += static_cast<std::uint64_t>(written);
    }
  }
  char trailing = '\0';
  ssize_t extra = -1;
  do {
    extra = ::read(source, &trailing, 1);
  } while (extra < 0 && errno == EINTR);
  struct stat after{};
  struct stat snapshot_status{};
  const bool stable = extra == 0 && ::fstat(source, &after) == 0 &&
                      expected.st_dev == after.st_dev && expected.st_ino == after.st_ino &&
                      expected.st_size == after.st_size &&
                      expected.st_mtim.tv_sec == after.st_mtim.tv_sec &&
                      expected.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
                      expected.st_ctim.tv_sec == after.st_ctim.tv_sec &&
                      expected.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
  constexpr int seals =
      F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_FUTURE_WRITE;
  if (!stable || ::fstat(snapshot, &snapshot_status) != 0 ||
      snapshot_status.st_size != expected.st_size || ::fchmod(snapshot, S_IRUSR | S_IXUSR) != 0 ||
      ::fcntl(snapshot, F_ADD_SEALS, seals) != 0 ||
      (::fcntl(snapshot, F_GET_SEALS) & seals) != seals) {
    (void)::close(snapshot);
    cgroup_cleanup_exit(EXIT_FAILURE);
  }

  const char marker = 'X';
  std::array<char, CMSG_SPACE(sizeof(int))> control{};
  iovec bytes{.iov_base = const_cast<char*>(&marker), .iov_len = 1};
  msghdr message{};
  message.msg_iov = &bytes;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  auto* header = CMSG_FIRSTHDR(&message);
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(header), &snapshot, sizeof(snapshot));
  ssize_t sent = -1;
  do {
    sent = ::sendmsg(channel, &message, MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  (void)::close(snapshot);
  cgroup_cleanup_exit(sent == 1 ? EXIT_SUCCESS : EXIT_FAILURE);
#endif
}

[[nodiscard]] UniqueFd receive_executable_snapshot(int channel) {
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
    received = ::recvmsg(channel, &message, MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);
  int snapshot = -1;
  bool invalid = received != 1 || marker != 'X' || (message.msg_flags & MSG_CTRUNC) != 0;
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
      if (snapshot < 0) {
        snapshot = descriptors[index];
      } else {
        invalid = true;
        (void)::close(descriptors[index]);
      }
    }
  }
  if (invalid || snapshot < 0) {
    if (snapshot >= 0) {
      (void)::close(snapshot);
    }
    throw CalibrationExecutionError("calibration executable snapshot helper returned invalid data");
  }
  return UniqueFd(snapshot);
}

class PinnedExecutable {
public:
  PinnedExecutable(const std::filesystem::path& path, Clock::time_point deadline)
      : display_path_(path.string()) {
    if (!path.is_absolute()) {
      throw CalibrationExecutionError(
          "GPU calibration worker path must name an executable regular file");
    }
    UniqueFd source(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    struct stat before{};
    if (!source || ::fstat(source.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
        (before.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0 ||
        (before.st_mode & (S_ISUID | S_ISGID)) != 0 || before.st_size <= 0 ||
        static_cast<std::uint64_t>(before.st_size) > kMaximumWorkerExecutableBytes) {
      throw CalibrationExecutionError(
          "GPU calibration worker path must name an unprivileged executable regular file");
    }
    errno = 0;
    const auto capability_size = ::fgetxattr(source.get(), "security.capability", nullptr, 0);
    if (capability_size >= 0 || (errno != ENODATA && errno != EOPNOTSUPP)) {
      throw CalibrationExecutionError(
          "GPU calibration worker executable must not carry file capabilities");
    }

#if !defined(SYS_clone3)
    throw CalibrationExecutionError("bounded executable snapshots require clone3");
#else
    std::array<int, 2> transport{-1, -1};
    if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, transport.data()) != 0) {
      throw CalibrationExecutionError(errno_message("cannot create executable snapshot transport"));
    }
    UniqueFd parent_transport(transport[0]);
    UniqueFd child_transport(transport[1]);
    int pidfd = -1;
    clone_args arguments{};
    arguments.flags = CLONE_PIDFD;
    arguments.pidfd = reinterpret_cast<std::uint64_t>(&pidfd);
    arguments.exit_signal = 0;
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
    __sanitizer_syscall_pre_impl_fork();
#endif
    const auto child = static_cast<pid_t>(::syscall(SYS_clone3, &arguments, sizeof(arguments)));
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
    __sanitizer_syscall_post_impl_fork(child);
#endif
    if (child == 0) {
      (void)::close(parent_transport.get());
      executable_snapshot_child(source.get(), child_transport.get(), before);
    }
    if (child < 0 || pidfd < 0) {
      throw CalibrationExecutionError(errno_message("cannot create executable snapshot helper"));
    }
    OwnedProcess helper(child, pidfd, deadline);
    child_transport.reset();
    while (true) {
      std::array<pollfd, 2> events{
          pollfd{.fd = parent_transport.get(), .events = POLLIN, .revents = 0},
          pollfd{.fd = helper.pidfd(), .events = POLLIN, .revents = 0},
      };
      const auto polled = ::poll(events.data(), events.size(), deadline_timeout(deadline));
      if (polled < 0 && errno == EINTR) {
        continue;
      }
      if (polled <= 0) {
        throw CalibrationExecutionError("calibration worker snapshot exceeded its deadline");
      }
      if ((events[0].revents & POLLIN) != 0) {
        descriptor_ = receive_executable_snapshot(parent_transport.get());
        break;
      }
      if ((events[0].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0 ||
          (events[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
        throw CalibrationExecutionError("calibration executable snapshot helper failed");
      }
    }
    helper.wait_until(deadline);
    const auto helper_status = helper.reap();
    struct stat after{};
    struct stat snapshot{};
    constexpr int seals =
        F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_FUTURE_WRITE;
    const bool source_stable =
        helper_status.known && helper_status.code == CLD_EXITED &&
        helper_status.status == EXIT_SUCCESS && ::fstat(source.get(), &after) == 0 &&
        before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
        before.st_size == after.st_size && before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
        before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
        before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
        before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
    if (!source_stable || ::fstat(descriptor_.get(), &snapshot) != 0 ||
        snapshot.st_size != before.st_size ||
        (::fcntl(descriptor_.get(), F_GET_SEALS) & seals) != seals) {
      throw CalibrationExecutionError("calibration worker changed while it was snapshotted");
    }
#endif
  }

  [[nodiscard]] int fd() const { return descriptor_.get(); }
  [[nodiscard]] const std::string& display_path() const { return display_path_; }

private:
  UniqueFd descriptor_;
  std::string display_path_;
};

struct UnixListener {
  UniqueFd descriptor;
  std::string address;
};

void bound_ipc_socket_buffers(int descriptor) {
  constexpr int buffer_bytes = 4096;
  if (::setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0 ||
      ::setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) {
    throw CalibrationExecutionError(errno_message("cannot bound calibration IPC buffers"));
  }
}

void enforce_resident_limit(const OwnedProcess& process, std::uint64_t limit);

[[nodiscard]] UnixListener create_listener() {
  std::array<unsigned char, 16> random{};
  std::size_t offset = 0;
  while (offset < random.size()) {
    const auto received = ::getrandom(random.data() + offset, random.size() - offset, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      throw CalibrationExecutionError(errno_message("cannot create calibration IPC identity"));
    }
    offset += static_cast<std::size_t>(received);
  }
  constexpr char digits[] = "0123456789abcdef";
  std::string name = "reco-calibration-";
  name.reserve(name.size() + random.size() * 2U);
  for (const auto byte : random) {
    name.push_back(digits[byte >> 4U]);
    name.push_back(digits[byte & 0x0fU]);
  }

  UniqueFd descriptor(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (!descriptor) {
    throw CalibrationExecutionError(errno_message("cannot create calibration IPC listener"));
  }
  bound_ipc_socket_buffers(descriptor.get());
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path + 1, name.data(), name.size());
  const auto size = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name.size());
  if (::bind(descriptor.get(), reinterpret_cast<const sockaddr*>(&address), size) != 0 ||
      ::listen(descriptor.get(), 4) != 0) {
    throw CalibrationExecutionError(errno_message("cannot bind calibration IPC listener"));
  }
  return {.descriptor = std::move(descriptor), .address = std::move(name)};
}

[[nodiscard]] UniqueFd accept_authenticated(const UnixListener& listener, OwnedProcess& process,
                                            Clock::time_point deadline,
                                            std::uint64_t resident_limit = 0) {
  while (Clock::now() < deadline) {
    enforce_resident_limit(process, resident_limit);
    std::array<pollfd, 2> items{
        pollfd{.fd = listener.descriptor.get(), .events = POLLIN, .revents = 0},
        pollfd{.fd = process.pidfd(), .events = POLLIN, .revents = 0},
    };
    const auto result = ::poll(items.data(), items.size(), deadline_timeout(deadline));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      throw CalibrationExecutionError(errno_message("cannot accept calibration IPC"));
    }
    if ((items[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
      const auto status = process.reap();
      throw CalibrationExecutionError(
          "calibration process exited before IPC authentication (code=" +
          std::to_string(status.code) + ", status=" + std::to_string(status.status) + ")");
    }
    if ((items[0].revents & POLLIN) == 0) {
      continue;
    }
    UniqueFd peer(
        ::accept4(listener.descriptor.get(), nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK));
    if (!peer) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      throw CalibrationExecutionError(errno_message("cannot accept calibration IPC peer"));
    }
    bound_ipc_socket_buffers(peer.get());
    ucred credentials{};
    socklen_t size = sizeof(credentials);
    if (::getsockopt(peer.get(), SOL_SOCKET, SO_PEERCRED, &credentials, &size) == 0 &&
        size == sizeof(credentials) && credentials.pid == process.pid() &&
        credentials.uid == ::geteuid() && !process.exited()) {
      return peer;
    }
  }
  throw CalibrationExecutionError("calibration process exceeded its IPC startup deadline");
}

void wait_for_process_monitored(const OwnedProcess& process, Clock::time_point deadline,
                                std::uint64_t resident_limit) {
  while (!process.exited()) {
    enforce_resident_limit(process, resident_limit);
    if (Clock::now() >= deadline) {
      throw CalibrationExecutionError("calibration worker exceeded its end-to-end deadline");
    }
    pollfd item{.fd = process.pidfd(), .events = POLLIN, .revents = 0};
    const auto timeout =
        resident_limit == 0 ? deadline_timeout(deadline) : std::min(deadline_timeout(deadline), 10);
    const auto result = ::poll(&item, 1, timeout);
    if (result < 0 && errno != EINTR) {
      throw CalibrationExecutionError(errno_message("cannot wait for calibration process"));
    }
  }
}

[[nodiscard]] std::optional<std::uint64_t> resident_bytes_noalloc(pid_t process) noexcept {
  char path[64]{};
  const auto path_size =
      std::snprintf(path, sizeof(path), "/proc/%ld/statm", static_cast<long>(process));
  if (path_size <= 0 || static_cast<std::size_t>(path_size) >= sizeof(path)) {
    return std::nullopt;
  }
  const auto descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return std::nullopt;
  }
  std::array<char, 128> contents{};
  ssize_t size = -1;
  do {
    size = ::read(descriptor, contents.data(), contents.size() - 1U);
  } while (size < 0 && errno == EINTR);
  (void)::close(descriptor);
  if (size <= 0) {
    return std::nullopt;
  }

  const char* cursor = contents.data();
  const char* end = contents.data() + size;
  while (cursor != end && *cursor >= '0' && *cursor <= '9') {
    ++cursor;
  }
  while (cursor != end && (*cursor == ' ' || *cursor == '\t')) {
    ++cursor;
  }
  std::uint64_t pages = 0;
  const auto [parsed_end, error] = std::from_chars(cursor, end, pages);
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  if (error != std::errc{} || parsed_end == cursor || page_size <= 0 ||
      pages > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(page_size)) {
    return std::nullopt;
  }
  return pages * static_cast<std::uint64_t>(page_size);
}

void enforce_resident_limit(const OwnedProcess& process, std::uint64_t limit) {
  if (limit == 0 || process.exited()) {
    return;
  }
  const auto resident = resident_bytes_noalloc(process.pid());
  if (!resident.has_value()) {
    if (!process.exited()) {
      throw CalibrationExecutionError("cannot enforce calibration resident-memory limit");
    }
    return;
  }
  if (*resident > limit) {
    throw CalibrationExecutionError("calibration worker exceeded its host memory limit");
  }
}

void wait_for_socket(int socket, const OwnedProcess& process, short events,
                     Clock::time_point deadline, int abort_socket = -1,
                     std::uint64_t resident_limit = 0) {
  while (Clock::now() < deadline) {
    enforce_resident_limit(process, resident_limit);
    std::array<pollfd, 3> items{
        pollfd{.fd = socket, .events = events, .revents = 0},
        pollfd{.fd = process.pidfd(), .events = POLLIN, .revents = 0},
        pollfd{.fd = abort_socket, .events = 0, .revents = 0},
    };
    const auto count = abort_socket >= 0 ? 3U : 2U;
    const auto timeout =
        resident_limit == 0 ? deadline_timeout(deadline) : std::min(deadline_timeout(deadline), 10);
    const auto result = ::poll(items.data(), count, timeout);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      throw CalibrationExecutionError(errno_message("cannot monitor calibration IPC"));
    }
    if ((items[0].revents & events) != 0) {
      return;
    }
    if (abort_socket >= 0 && (items[2].revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      throw CalibrationExecutionError("calibration caller closed its response channel");
    }
    if ((items[0].revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      throw CalibrationExecutionError("calibration process closed its response channel");
    }
    if ((items[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0 && events != POLLIN) {
      throw CalibrationExecutionError("calibration process closed its response channel");
    }
  }
  throw CalibrationExecutionError("calibration worker exceeded its end-to-end deadline");
}

void write_all(int socket, const OwnedProcess& process, std::string_view bytes,
               Clock::time_point deadline, int abort_socket = -1,
               std::uint64_t resident_limit = 0) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    wait_for_socket(socket, process, POLLOUT, deadline, abort_socket, resident_limit);
    const auto written = ::send(socket, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
    if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (written <= 0) {
      throw CalibrationExecutionError("calibration process closed its request channel");
    }
    offset += static_cast<std::size_t>(written);
  }
}

void read_exact(int socket, const OwnedProcess& process, char* destination, std::size_t size,
                Clock::time_point deadline, int abort_socket = -1,
                std::uint64_t resident_limit = 0) {
  std::size_t offset = 0;
  while (offset < size) {
    wait_for_socket(socket, process, POLLIN, deadline, abort_socket, resident_limit);
    const auto received = ::recv(socket, destination + offset, size - offset, 0);
    if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (received <= 0) {
      throw CalibrationExecutionError("calibration process closed its response channel");
    }
    offset += static_cast<std::size_t>(received);
  }
}

[[nodiscard]] std::string read_frame(int socket, const OwnedProcess& process,
                                     Clock::time_point deadline, int abort_socket = -1,
                                     std::uint64_t resident_limit = 0) {
  CalibrationWorkerFrameHeader header{};
  read_exact(socket, process, header.data(), header.size(), deadline, abort_socket, resident_limit);
  const auto decoded = decode_calibration_worker_header(header);
  std::string response(header.data(), header.size());
  const auto payload_offset = response.size();
  response.resize(payload_offset + decoded.payload_size);
  read_exact(socket, process, response.data() + payload_offset, decoded.payload_size, deadline,
             abort_socket, resident_limit);
  return response;
}

void send_file_fd(int socket, const OwnedProcess& process, char marker, int file_descriptor,
                  Clock::time_point deadline, int abort_socket = -1,
                  std::uint64_t resident_limit = 0) {
  std::array<char, CMSG_SPACE(sizeof(int))> control{};
  iovec bytes{.iov_base = &marker, .iov_len = 1};
  msghdr message{};
  message.msg_iov = &bytes;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  auto* header = CMSG_FIRSTHDR(&message);
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(header), &file_descriptor, sizeof(file_descriptor));

  while (true) {
    wait_for_socket(socket, process, POLLOUT, deadline, abort_socket, resident_limit);
    const auto sent = ::sendmsg(socket, &message, MSG_NOSIGNAL);
    if (sent < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (sent != 1) {
      throw CalibrationExecutionError("cannot transfer calibration admission ownership");
    }
    return;
  }
}

[[nodiscard]] ExternalProcessAuthority
receive_worker_authority(int socket, const OwnedProcess& guardian, Clock::time_point deadline) {
  wait_for_socket(socket, guardian, POLLIN, deadline);
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
    received = ::recvmsg(socket, &message, MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);
  if (received != 1 || (message.msg_flags & MSG_CTRUNC) != 0 || (marker != 'P' && marker != 'N')) {
    throw CalibrationExecutionError("calibration guardian returned invalid worker authority");
  }

  int authority = -1;
  bool invalid = false;
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
      if (authority < 0) {
        authority = descriptors[index];
      } else {
        invalid = true;
        (void)::close(descriptors[index]);
      }
    }
  }
  if ((marker == 'P') != (authority >= 0)) {
    invalid = true;
  }
  if (invalid) {
    if (authority >= 0) {
      (void)::close(authority);
    }
    throw CalibrationExecutionError("calibration guardian returned invalid worker authority");
  }
  return ExternalProcessAuthority(authority, deadline);
}

void write_plain_all(int descriptor, std::string_view bytes, Clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    pollfd item{.fd = descriptor, .events = POLLOUT, .revents = 0};
    const auto result = ::poll(&item, 1, deadline_timeout(deadline));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0 || (item.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      throw CalibrationExecutionError("calibration caller closed its response channel");
    }
    const auto written =
        ::send(descriptor, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
    if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (written <= 0) {
      throw CalibrationExecutionError("calibration caller closed its response channel");
    }
    offset += static_cast<std::size_t>(written);
  }
}

void read_plain_exact(int descriptor, char* destination, std::size_t size,
                      Clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < size) {
    pollfd item{.fd = descriptor, .events = POLLIN, .revents = 0};
    const auto result = ::poll(&item, 1, deadline_timeout(deadline));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0 || (item.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      throw CalibrationExecutionError("calibration caller closed its request channel");
    }
    const auto received = ::recv(descriptor, destination + offset, size - offset, 0);
    if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (received <= 0) {
      throw CalibrationExecutionError("calibration caller closed its request channel");
    }
    offset += static_cast<std::size_t>(received);
  }
}

[[nodiscard]] std::string read_plain_frame(int descriptor, Clock::time_point deadline) {
  CalibrationWorkerFrameHeader header{};
  read_plain_exact(descriptor, header.data(), header.size(), deadline);
  const auto decoded = decode_calibration_worker_header(header);
  std::string request(header.data(), header.size());
  const auto payload_offset = request.size();
  request.resize(payload_offset + decoded.payload_size);
  read_plain_exact(descriptor, request.data() + payload_offset, decoded.payload_size, deadline);
  return request;
}

[[noreturn]] void child_exit() noexcept {
  static constexpr char failure[] = "calibration pre-exec child failed\n";
  ssize_t written = -1;
  do {
    written = ::write(STDERR_FILENO, failure, sizeof(failure) - 1U);
  } while (written < 0 && errno == EINTR);
  (void)::syscall(SYS_exit_group, 127);
  __builtin_unreachable();
}

void reset_child_signals() noexcept {
  struct sigaction action{};
  action.sa_handler = SIG_DFL;
  (void)::sigemptyset(&action.sa_mask);
  for (const auto signal : {SIGCHLD, SIGTERM, SIGUSR1, SIGINT, SIGHUP}) {
    if (::sigaction(signal, &action, nullptr) != 0) {
      static constexpr char failure[] = "calibration child signal reset failed\n";
      ssize_t written = -1;
      do {
        written = ::write(STDERR_FILENO, failure, sizeof(failure) - 1U);
      } while (written < 0 && errno == EINTR);
      child_exit();
    }
  }
  sigset_t empty{};
  (void)::sigemptyset(&empty);
  if (::sigprocmask(SIG_SETMASK, &empty, nullptr) != 0) {
    static constexpr char failure[] = "calibration child signal mask reset failed\n";
    ssize_t written = -1;
    do {
      written = ::write(STDERR_FILENO, failure, sizeof(failure) - 1U);
    } while (written < 0 && errno == EINTR);
    child_exit();
  }
}

void close_child_descriptors_except(int preserved) noexcept {
#if defined(SYS_close_range)
  if (preserved < 3 ||
      (preserved > 3 &&
       ::syscall(SYS_close_range, 3U, static_cast<unsigned int>(preserved - 1), 0U) != 0) ||
      ::syscall(SYS_close_range, static_cast<unsigned int>(preserved + 1),
                std::numeric_limits<unsigned int>::max(), 0U) != 0) {
    static constexpr char failure[] = "calibration child descriptor close failed\n";
    ssize_t written = -1;
    do {
      written = ::write(STDERR_FILENO, failure, sizeof(failure) - 1U);
    } while (written < 0 && errno == EINTR);
    child_exit();
  }
#else
  child_exit();
#endif
}

[[maybe_unused, nodiscard]] bool install_initial_exec_filter(int executable) noexcept {
#if !defined(__x86_64__) && !defined(__aarch64__)
  (void)executable;
  return false;
#elif defined(SYS_execveat)
  const sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kSeccompArchitecture, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_execve, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_execveat, 0, 7),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, static_cast<std::uint32_t>(executable), 0, 4),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[4])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AT_EMPTY_PATH, 0, 2),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[4]) + sizeof(std::uint32_t)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  const sock_fprog program{.len = static_cast<unsigned short>(std::size(filter)),
                           .filter = const_cast<sock_filter*>(filter)};
  return ::prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) == 0 &&
         ::prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) == 0 &&
         ::syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0U, &program) == 0;
#else
  (void)executable;
  return false;
#endif
}

[[nodiscard]] bool install_worker_pre_main_filter() noexcept {
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
  return true;
#elif defined(__x86_64__) || defined(__aarch64__)
#define RECO_PRE_MAIN_DENY(syscall_number)                                                         \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, syscall_number, 0, 1),                                       \
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)
  const sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kSeccompArchitecture, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
#if defined(__x86_64__)
      BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, kX32SyscallBit, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      RECO_PRE_MAIN_DENY(SYS_fork),
      RECO_PRE_MAIN_DENY(SYS_vfork),
#endif
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone3, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone, 0, 5),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, CLONE_THREAD),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      RECO_PRE_MAIN_DENY(SYS_kill),
      RECO_PRE_MAIN_DENY(SYS_tkill),
      RECO_PRE_MAIN_DENY(SYS_tgkill),
      RECO_PRE_MAIN_DENY(SYS_rt_sigqueueinfo),
      RECO_PRE_MAIN_DENY(SYS_rt_tgsigqueueinfo),
      RECO_PRE_MAIN_DENY(SYS_pidfd_open),
      RECO_PRE_MAIN_DENY(SYS_pidfd_send_signal),
      RECO_PRE_MAIN_DENY(SYS_pidfd_getfd),
      RECO_PRE_MAIN_DENY(SYS_ptrace),
      RECO_PRE_MAIN_DENY(SYS_process_vm_readv),
      RECO_PRE_MAIN_DENY(SYS_process_vm_writev),
      RECO_PRE_MAIN_DENY(SYS_kcmp),
      RECO_PRE_MAIN_DENY(SYS_mount),
      RECO_PRE_MAIN_DENY(SYS_umount2),
      RECO_PRE_MAIN_DENY(SYS_pivot_root),
      RECO_PRE_MAIN_DENY(SYS_setns),
      RECO_PRE_MAIN_DENY(SYS_unshare),
#if defined(SYS_io_uring_setup)
      RECO_PRE_MAIN_DENY(SYS_io_uring_setup),
#endif
#if defined(SYS_io_uring_register)
      RECO_PRE_MAIN_DENY(SYS_io_uring_register),
#endif
#if defined(SYS_io_uring_enter)
      RECO_PRE_MAIN_DENY(SYS_io_uring_enter),
#endif
#if defined(SYS_flock)
      RECO_PRE_MAIN_DENY(SYS_flock),
#endif
#if defined(SYS_chmod)
      RECO_PRE_MAIN_DENY(SYS_chmod),
#endif
#if defined(SYS_fchmod)
      RECO_PRE_MAIN_DENY(SYS_fchmod),
#endif
#if defined(SYS_fchmodat)
      RECO_PRE_MAIN_DENY(SYS_fchmodat),
#endif
#if defined(SYS_fchmodat2)
      RECO_PRE_MAIN_DENY(SYS_fchmodat2),
#endif
#if defined(SYS_chown)
      RECO_PRE_MAIN_DENY(SYS_chown),
#endif
#if defined(SYS_fchown)
      RECO_PRE_MAIN_DENY(SYS_fchown),
#endif
#if defined(SYS_lchown)
      RECO_PRE_MAIN_DENY(SYS_lchown),
#endif
#if defined(SYS_fchownat)
      RECO_PRE_MAIN_DENY(SYS_fchownat),
#endif
#if defined(SYS_chown32) && (!defined(SYS_chown) || SYS_chown32 != SYS_chown)
      RECO_PRE_MAIN_DENY(SYS_chown32),
#endif
#if defined(SYS_fchown32) && (!defined(SYS_fchown) || SYS_fchown32 != SYS_fchown)
      RECO_PRE_MAIN_DENY(SYS_fchown32),
#endif
#if defined(SYS_lchown32) && (!defined(SYS_lchown) || SYS_lchown32 != SYS_lchown)
      RECO_PRE_MAIN_DENY(SYS_lchown32),
#endif
#if defined(SYS_utime)
      RECO_PRE_MAIN_DENY(SYS_utime),
#endif
#if defined(SYS_utimes)
      RECO_PRE_MAIN_DENY(SYS_utimes),
#endif
#if defined(SYS_futimesat)
      RECO_PRE_MAIN_DENY(SYS_futimesat),
#endif
#if defined(SYS_utimensat)
      RECO_PRE_MAIN_DENY(SYS_utimensat),
#endif
#if defined(SYS_utimensat_time64) &&                                                               \
    (!defined(SYS_utimensat) || SYS_utimensat_time64 != SYS_utimensat)
      RECO_PRE_MAIN_DENY(SYS_utimensat_time64),
#endif
#if defined(SYS_setxattr)
      RECO_PRE_MAIN_DENY(SYS_setxattr),
#endif
#if defined(SYS_lsetxattr)
      RECO_PRE_MAIN_DENY(SYS_lsetxattr),
#endif
#if defined(SYS_fsetxattr)
      RECO_PRE_MAIN_DENY(SYS_fsetxattr),
#endif
#if defined(SYS_setxattrat)
      RECO_PRE_MAIN_DENY(SYS_setxattrat),
#endif
#if defined(SYS_removexattr)
      RECO_PRE_MAIN_DENY(SYS_removexattr),
#endif
#if defined(SYS_lremovexattr)
      RECO_PRE_MAIN_DENY(SYS_lremovexattr),
#endif
#if defined(SYS_fremovexattr)
      RECO_PRE_MAIN_DENY(SYS_fremovexattr),
#endif
#if defined(SYS_removexattrat)
      RECO_PRE_MAIN_DENY(SYS_removexattrat),
#endif
      RECO_PRE_MAIN_DENY(SYS_open_by_handle_at),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socket, 0, 3),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EAFNOSUPPORT),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
#undef RECO_PRE_MAIN_DENY
  const sock_fprog program{.len = static_cast<unsigned short>(std::size(filter)),
                           .filter = const_cast<sock_filter*>(filter)};
  return ::prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) == 0 &&
         ::syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0U, &program) == 0;
#else
  return false;
#endif
}

[[nodiscard]] bool install_worker_filesystem_boundary(const char* scratch, int left_input,
                                                      int right_input) noexcept {
#if !defined(SYS_landlock_create_ruleset) || !defined(SYS_landlock_add_rule) ||                    \
    !defined(SYS_landlock_restrict_self)
  (void)scratch;
  (void)left_input;
  (void)right_input;
  return false;
#else
  if (scratch == nullptr || scratch[0] != '/') {
    return false;
  }
  const auto abi = static_cast<int>(
      ::syscall(SYS_landlock_create_ruleset, nullptr, 0U, LANDLOCK_CREATE_RULESET_VERSION));
  if (abi < 1) {
    return false;
  }
  std::uint64_t write_access = LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_REMOVE_DIR |
                               LANDLOCK_ACCESS_FS_REMOVE_FILE | LANDLOCK_ACCESS_FS_MAKE_CHAR |
                               LANDLOCK_ACCESS_FS_MAKE_DIR | LANDLOCK_ACCESS_FS_MAKE_REG |
                               LANDLOCK_ACCESS_FS_MAKE_SOCK | LANDLOCK_ACCESS_FS_MAKE_FIFO |
                               LANDLOCK_ACCESS_FS_MAKE_BLOCK | LANDLOCK_ACCESS_FS_MAKE_SYM;
#if defined(LANDLOCK_ACCESS_FS_REFER)
  if (abi >= 2) {
    write_access |= LANDLOCK_ACCESS_FS_REFER;
  }
#endif
#if defined(LANDLOCK_ACCESS_FS_TRUNCATE)
  if (abi >= 3) {
    write_access |= LANDLOCK_ACCESS_FS_TRUNCATE;
  }
#endif
  constexpr std::uint64_t read_access =
      LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR;
  const landlock_ruleset_attr ruleset_attributes{.handled_access_fs = write_access | read_access};
  UniqueFd ruleset(static_cast<int>(
      ::syscall(SYS_landlock_create_ruleset, &ruleset_attributes, sizeof(ruleset_attributes), 0U)));
  if (!ruleset) {
    return false;
  }
  const auto allow_path = [&](const char* path, std::uint64_t access) {
    UniqueFd parent(::open(path, O_PATH | O_CLOEXEC));
    if (!parent) {
      return false;
    }
    const landlock_path_beneath_attr rule{.allowed_access = access, .parent_fd = parent.get()};
    return ::syscall(SYS_landlock_add_rule, ruleset.get(), LANDLOCK_RULE_PATH_BENEATH, &rule, 0U) ==
           0;
  };
  const auto allow_input = [&](int descriptor) {
    if (descriptor < 0) {
      return true;
    }
    struct stat status{};
    if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
      return false;
    }
    const landlock_path_beneath_attr rule{.allowed_access = LANDLOCK_ACCESS_FS_READ_FILE,
                                          .parent_fd = descriptor};
    return ::syscall(SYS_landlock_add_rule, ruleset.get(), LANDLOCK_RULE_PATH_BENEATH, &rule, 0U) ==
           0;
  };
  const auto allow_optional_system_directory = [&](const char* path) {
    UniqueFd directory(::open(path, O_PATH | O_CLOEXEC));
    if (!directory) {
      return errno == ENOENT;
    }
    struct stat status{};
    if (::fstat(directory.get(), &status) != 0 || !S_ISDIR(status.st_mode) || status.st_uid != 0 ||
        (status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
      return false;
    }
    const landlock_path_beneath_attr rule{.allowed_access = read_access,
                                          .parent_fd = directory.get()};
    return ::syscall(SYS_landlock_add_rule, ruleset.get(), LANDLOCK_RULE_PATH_BENEATH, &rule, 0U) ==
           0;
  };
  const auto allow_optional_system_file = [&](const char* path) {
    UniqueFd file(::open(path, O_PATH | O_CLOEXEC));
    if (!file) {
      return errno == ENOENT;
    }
    struct stat status{};
    if (::fstat(file.get(), &status) != 0 || !S_ISREG(status.st_mode) || status.st_uid != 0 ||
        (status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
      return false;
    }
    const landlock_path_beneath_attr rule{.allowed_access = LANDLOCK_ACCESS_FS_EXECUTE |
                                                            LANDLOCK_ACCESS_FS_READ_FILE,
                                          .parent_fd = file.get()};
    return ::syscall(SYS_landlock_add_rule, ruleset.get(), LANDLOCK_RULE_PATH_BENEATH, &rule, 0U) ==
           0;
  };
  const auto allow_optional_device = [&](const char* path) {
    UniqueFd device(::open(path, O_PATH | O_CLOEXEC | O_NOFOLLOW));
    if (!device) {
      return errno == ENOENT;
    }
    struct stat status{};
    if (::fstat(device.get(), &status) != 0 || !S_ISCHR(status.st_mode)) {
      return false;
    }
    const landlock_path_beneath_attr rule{.allowed_access = LANDLOCK_ACCESS_FS_READ_FILE |
                                                            LANDLOCK_ACCESS_FS_WRITE_FILE,
                                          .parent_fd = device.get()};
    return ::syscall(SYS_landlock_add_rule, ruleset.get(), LANDLOCK_RULE_PATH_BENEATH, &rule, 0U) ==
           0;
  };
  const auto allow_optional_device_directory = [&](const char* path) {
    UniqueFd directory(::open(path, O_PATH | O_CLOEXEC | O_NOFOLLOW));
    if (!directory) {
      return errno == ENOENT;
    }
    struct stat status{};
    if (::fstat(directory.get(), &status) != 0 || !S_ISDIR(status.st_mode)) {
      return false;
    }
    const landlock_path_beneath_attr rule{.allowed_access = LANDLOCK_ACCESS_FS_READ_FILE |
                                                            LANDLOCK_ACCESS_FS_READ_DIR |
                                                            LANDLOCK_ACCESS_FS_WRITE_FILE,
                                          .parent_fd = directory.get()};
    return ::syscall(SYS_landlock_add_rule, ruleset.get(), LANDLOCK_RULE_PATH_BENEATH, &rule, 0U) ==
           0;
  };
  std::uint64_t proc_access = LANDLOCK_ACCESS_FS_WRITE_FILE;
#if defined(LANDLOCK_ACCESS_FS_TRUNCATE)
  if (abi >= 3) {
    proc_access |= LANDLOCK_ACCESS_FS_TRUNCATE;
  }
#endif
  proc_access |= read_access;
  char worker_proc[64]{};
  const auto worker_proc_size = std::snprintf(worker_proc, sizeof(worker_proc), "/proc/%lld",
                                              static_cast<long long>(::getpid()));
  const bool system_reads_allowed =
      allow_optional_system_directory("/usr") && allow_optional_system_directory("/lib") &&
      allow_optional_system_directory("/lib64") && allow_optional_system_directory("/opt/nvidia") &&
      allow_optional_system_directory("/sys") &&
      allow_optional_system_directory("/proc/driver/nvidia") &&
      allow_optional_system_file("/etc/ld.so.cache") &&
      allow_optional_system_file("/etc/localtime") && allow_optional_system_file("/proc/cpuinfo") &&
      allow_optional_system_file("/proc/devices") &&
      allow_optional_system_file("/proc/filesystems") &&
      allow_optional_system_file("/proc/meminfo") && allow_optional_system_file("/proc/version");
  bool devices_allowed =
      allow_optional_device("/dev/null") && allow_optional_device("/dev/urandom") &&
      allow_optional_device("/dev/zero") && allow_optional_device("/dev/nvidiactl") &&
      allow_optional_device("/dev/nvidia-uvm") && allow_optional_device("/dev/nvidia-uvm-tools") &&
      allow_optional_device("/dev/nvidia-modeset") &&
      allow_optional_device_directory("/dev/nvidia-caps");
  for (unsigned int index = 0; devices_allowed && index < 32U; ++index) {
    char path[32]{};
    const auto size = std::snprintf(path, sizeof(path), "/dev/nvidia%u", index);
    devices_allowed =
        size > 0 && static_cast<std::size_t>(size) < sizeof(path) && allow_optional_device(path);
  }
  if (!devices_allowed || !system_reads_allowed || worker_proc_size <= 0 ||
      static_cast<std::size_t>(worker_proc_size) >= sizeof(worker_proc) ||
      !allow_path(worker_proc, proc_access) || !allow_input(left_input) ||
      !allow_input(right_input) || !allow_path(scratch, write_access | read_access) ||
      ::prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) != 0 ||
      ::syscall(SYS_landlock_restrict_self, ruleset.get(), 0U) != 0) {
    return false;
  }
  return true;
#endif
}

[[nodiscard]] bool install_worker_syscall_filter() noexcept {
#define RECO_WORKER_DENY(syscall_number, error_number)                                             \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, syscall_number, 0, 1),                                       \
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | error_number)
#if defined(__x86_64__)
  constexpr std::uint32_t architecture = AUDIT_ARCH_X86_64;
  constexpr std::uint32_t x32_syscall_bit = 0x40000000U;
  static const sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, architecture, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, x32_syscall_bit, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      RECO_WORKER_DENY(SYS_execve, EPERM),
      RECO_WORKER_DENY(SYS_execveat, EPERM),
      RECO_WORKER_DENY(SYS_kill, EPERM),
      RECO_WORKER_DENY(SYS_tkill, EPERM),
      RECO_WORKER_DENY(SYS_tgkill, EPERM),
      RECO_WORKER_DENY(SYS_rt_sigqueueinfo, EPERM),
      RECO_WORKER_DENY(SYS_rt_tgsigqueueinfo, EPERM),
      RECO_WORKER_DENY(SYS_pidfd_open, EPERM),
      RECO_WORKER_DENY(SYS_pidfd_send_signal, EPERM),
      RECO_WORKER_DENY(SYS_pidfd_getfd, EPERM),
      RECO_WORKER_DENY(SYS_ptrace, EPERM),
      RECO_WORKER_DENY(SYS_process_vm_readv, EPERM),
      RECO_WORKER_DENY(SYS_process_vm_writev, EPERM),
      RECO_WORKER_DENY(SYS_kcmp, EPERM),
      RECO_WORKER_DENY(SYS_mount, EPERM),
      RECO_WORKER_DENY(SYS_umount2, EPERM),
      RECO_WORKER_DENY(SYS_pivot_root, EPERM),
      RECO_WORKER_DENY(SYS_setns, EPERM),
      RECO_WORKER_DENY(SYS_unshare, EPERM),
#if defined(SYS_io_uring_setup)
      RECO_WORKER_DENY(SYS_io_uring_setup, EPERM),
#endif
#if defined(SYS_io_uring_register)
      RECO_WORKER_DENY(SYS_io_uring_register, EPERM),
#endif
#if defined(SYS_io_uring_enter)
      RECO_WORKER_DENY(SYS_io_uring_enter, EPERM),
#endif
#if defined(SYS_flock)
      RECO_WORKER_DENY(SYS_flock, EPERM),
#endif
#if defined(SYS_chmod)
      RECO_WORKER_DENY(SYS_chmod, EPERM),
#endif
#if defined(SYS_fchmod)
      RECO_WORKER_DENY(SYS_fchmod, EPERM),
#endif
#if defined(SYS_fchmodat)
      RECO_WORKER_DENY(SYS_fchmodat, EPERM),
#endif
#if defined(SYS_fchmodat2)
      RECO_WORKER_DENY(SYS_fchmodat2, EPERM),
#endif
#if defined(SYS_chown)
      RECO_WORKER_DENY(SYS_chown, EPERM),
#endif
#if defined(SYS_fchown)
      RECO_WORKER_DENY(SYS_fchown, EPERM),
#endif
#if defined(SYS_lchown)
      RECO_WORKER_DENY(SYS_lchown, EPERM),
#endif
#if defined(SYS_fchownat)
      RECO_WORKER_DENY(SYS_fchownat, EPERM),
#endif
#if defined(SYS_chown32) && (!defined(SYS_chown) || SYS_chown32 != SYS_chown)
      RECO_WORKER_DENY(SYS_chown32, EPERM),
#endif
#if defined(SYS_fchown32) && (!defined(SYS_fchown) || SYS_fchown32 != SYS_fchown)
      RECO_WORKER_DENY(SYS_fchown32, EPERM),
#endif
#if defined(SYS_lchown32) && (!defined(SYS_lchown) || SYS_lchown32 != SYS_lchown)
      RECO_WORKER_DENY(SYS_lchown32, EPERM),
#endif
#if defined(SYS_utime)
      RECO_WORKER_DENY(SYS_utime, EPERM),
#endif
#if defined(SYS_utimes)
      RECO_WORKER_DENY(SYS_utimes, EPERM),
#endif
#if defined(SYS_futimesat)
      RECO_WORKER_DENY(SYS_futimesat, EPERM),
#endif
#if defined(SYS_utimensat)
      RECO_WORKER_DENY(SYS_utimensat, EPERM),
#endif
#if defined(SYS_utimensat_time64) &&                                                               \
    (!defined(SYS_utimensat) || SYS_utimensat_time64 != SYS_utimensat)
      RECO_WORKER_DENY(SYS_utimensat_time64, EPERM),
#endif
#if defined(SYS_setxattr)
      RECO_WORKER_DENY(SYS_setxattr, EPERM),
#endif
#if defined(SYS_lsetxattr)
      RECO_WORKER_DENY(SYS_lsetxattr, EPERM),
#endif
#if defined(SYS_fsetxattr)
      RECO_WORKER_DENY(SYS_fsetxattr, EPERM),
#endif
#if defined(SYS_setxattrat)
      RECO_WORKER_DENY(SYS_setxattrat, EPERM),
#endif
#if defined(SYS_removexattr)
      RECO_WORKER_DENY(SYS_removexattr, EPERM),
#endif
#if defined(SYS_lremovexattr)
      RECO_WORKER_DENY(SYS_lremovexattr, EPERM),
#endif
#if defined(SYS_fremovexattr)
      RECO_WORKER_DENY(SYS_fremovexattr, EPERM),
#endif
#if defined(SYS_removexattrat)
      RECO_WORKER_DENY(SYS_removexattrat, EPERM),
#endif
      RECO_WORKER_DENY(SYS_open_by_handle_at, EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socket, 0, 7),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 0, 4),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[1])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, 0xfU),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SOCK_SEQPACKET, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socketpair, 0, 8),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 0, 4),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[1])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, 0xfU),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SOCK_STREAM, 2, 0),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SOCK_SEQPACKET, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      RECO_WORKER_DENY(SYS_connect, ENOENT),
      RECO_WORKER_DENY(SYS_accept, EPERM),
      RECO_WORKER_DENY(SYS_accept4, EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_fork, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_vfork, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone3, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone, 0, 5),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, CLONE_THREAD),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_prctl, 0, 3),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, PR_SET_PDEATHSIG, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
#elif defined(__aarch64__)
  constexpr std::uint32_t architecture = AUDIT_ARCH_AARCH64;
  static const sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, architecture, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      RECO_WORKER_DENY(SYS_execve, EPERM),
      RECO_WORKER_DENY(SYS_execveat, EPERM),
      RECO_WORKER_DENY(SYS_kill, EPERM),
      RECO_WORKER_DENY(SYS_tkill, EPERM),
      RECO_WORKER_DENY(SYS_tgkill, EPERM),
      RECO_WORKER_DENY(SYS_rt_sigqueueinfo, EPERM),
      RECO_WORKER_DENY(SYS_rt_tgsigqueueinfo, EPERM),
      RECO_WORKER_DENY(SYS_pidfd_open, EPERM),
      RECO_WORKER_DENY(SYS_pidfd_send_signal, EPERM),
      RECO_WORKER_DENY(SYS_pidfd_getfd, EPERM),
      RECO_WORKER_DENY(SYS_ptrace, EPERM),
      RECO_WORKER_DENY(SYS_process_vm_readv, EPERM),
      RECO_WORKER_DENY(SYS_process_vm_writev, EPERM),
      RECO_WORKER_DENY(SYS_kcmp, EPERM),
      RECO_WORKER_DENY(SYS_mount, EPERM),
      RECO_WORKER_DENY(SYS_umount2, EPERM),
      RECO_WORKER_DENY(SYS_pivot_root, EPERM),
      RECO_WORKER_DENY(SYS_setns, EPERM),
      RECO_WORKER_DENY(SYS_unshare, EPERM),
#if defined(SYS_io_uring_setup)
      RECO_WORKER_DENY(SYS_io_uring_setup, EPERM),
#endif
#if defined(SYS_io_uring_register)
      RECO_WORKER_DENY(SYS_io_uring_register, EPERM),
#endif
#if defined(SYS_io_uring_enter)
      RECO_WORKER_DENY(SYS_io_uring_enter, EPERM),
#endif
#if defined(SYS_flock)
      RECO_WORKER_DENY(SYS_flock, EPERM),
#endif
#if defined(SYS_fchmod)
      RECO_WORKER_DENY(SYS_fchmod, EPERM),
#endif
#if defined(SYS_fchmodat)
      RECO_WORKER_DENY(SYS_fchmodat, EPERM),
#endif
#if defined(SYS_fchmodat2)
      RECO_WORKER_DENY(SYS_fchmodat2, EPERM),
#endif
#if defined(SYS_chown)
      RECO_WORKER_DENY(SYS_chown, EPERM),
#endif
#if defined(SYS_fchown)
      RECO_WORKER_DENY(SYS_fchown, EPERM),
#endif
#if defined(SYS_lchown)
      RECO_WORKER_DENY(SYS_lchown, EPERM),
#endif
#if defined(SYS_fchownat)
      RECO_WORKER_DENY(SYS_fchownat, EPERM),
#endif
#if defined(SYS_chown32) && (!defined(SYS_chown) || SYS_chown32 != SYS_chown)
      RECO_WORKER_DENY(SYS_chown32, EPERM),
#endif
#if defined(SYS_fchown32) && (!defined(SYS_fchown) || SYS_fchown32 != SYS_fchown)
      RECO_WORKER_DENY(SYS_fchown32, EPERM),
#endif
#if defined(SYS_lchown32) && (!defined(SYS_lchown) || SYS_lchown32 != SYS_lchown)
      RECO_WORKER_DENY(SYS_lchown32, EPERM),
#endif
#if defined(SYS_utime)
      RECO_WORKER_DENY(SYS_utime, EPERM),
#endif
#if defined(SYS_utimes)
      RECO_WORKER_DENY(SYS_utimes, EPERM),
#endif
#if defined(SYS_futimesat)
      RECO_WORKER_DENY(SYS_futimesat, EPERM),
#endif
#if defined(SYS_utimensat)
      RECO_WORKER_DENY(SYS_utimensat, EPERM),
#endif
#if defined(SYS_utimensat_time64) &&                                                               \
    (!defined(SYS_utimensat) || SYS_utimensat_time64 != SYS_utimensat)
      RECO_WORKER_DENY(SYS_utimensat_time64, EPERM),
#endif
#if defined(SYS_setxattr)
      RECO_WORKER_DENY(SYS_setxattr, EPERM),
#endif
#if defined(SYS_lsetxattr)
      RECO_WORKER_DENY(SYS_lsetxattr, EPERM),
#endif
#if defined(SYS_fsetxattr)
      RECO_WORKER_DENY(SYS_fsetxattr, EPERM),
#endif
#if defined(SYS_setxattrat)
      RECO_WORKER_DENY(SYS_setxattrat, EPERM),
#endif
#if defined(SYS_removexattr)
      RECO_WORKER_DENY(SYS_removexattr, EPERM),
#endif
#if defined(SYS_lremovexattr)
      RECO_WORKER_DENY(SYS_lremovexattr, EPERM),
#endif
#if defined(SYS_fremovexattr)
      RECO_WORKER_DENY(SYS_fremovexattr, EPERM),
#endif
#if defined(SYS_removexattrat)
      RECO_WORKER_DENY(SYS_removexattrat, EPERM),
#endif
      RECO_WORKER_DENY(SYS_open_by_handle_at, EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socket, 0, 7),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 0, 4),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[1])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, 0xfU),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SOCK_SEQPACKET, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socketpair, 0, 8),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 0, 4),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[1])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, 0xfU),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SOCK_STREAM, 2, 0),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SOCK_SEQPACKET, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      RECO_WORKER_DENY(SYS_connect, ENOENT),
      RECO_WORKER_DENY(SYS_accept, EPERM),
      RECO_WORKER_DENY(SYS_accept4, EPERM),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone3, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone, 0, 5),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, CLONE_THREAD),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_prctl, 0, 3),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, PR_SET_PDEATHSIG, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
#else
  return false;
#endif
#undef RECO_WORKER_DENY
  const sock_fprog program{.len = static_cast<unsigned short>(std::size(filter)),
                           .filter = const_cast<sock_filter*>(filter)};
  return ::prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) == 0 &&
         ::prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) == 0 &&
#if defined(SYS_seccomp)
         ::syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_TSYNC, &program) == 0;
#else
         false;
#endif
}

void apply_worker_limit(std::uint64_t bytes) noexcept {
  const struct rlimit core_limit{.rlim_cur = 0, .rlim_max = 0};
  if (::setrlimit(RLIMIT_CORE, &core_limit) != 0) {
    child_exit();
  }
#if defined(RECO_CALIBRATION_WIDE_ADDRESS_SANITIZER)
  (void)bytes;
#else
  if (bytes > std::numeric_limits<rlim_t>::max()) {
    child_exit();
  }
  const struct rlimit limit{.rlim_cur = static_cast<rlim_t>(bytes),
                            .rlim_max = static_cast<rlim_t>(bytes)};
  if (::setrlimit(RLIMIT_DATA, &limit) != 0) {
    child_exit();
  }
#endif
}

class WorkerScratchTree {
public:
  WorkerScratchTree() {
    const auto suffix = random_cgroup_name().substr(kCgroupPrefix.size());
    root_ = std::filesystem::path("/tmp") / (std::string(kSandboxPrefix) + suffix);
    if (::mkdir(root_.c_str(), S_IRWXU) != 0) {
      throw CalibrationExecutionError(errno_message("cannot create calibration worker scratch"));
    }
    try {
      scratch_ = root_ / "scratch";
      if (::mkdir(scratch_.c_str(), S_IRWXU) != 0) {
        throw CalibrationExecutionError(errno_message("cannot create calibration worker scratch"));
      }
      plugin_directory_ = scratch_ / "gst-plugins";
      if (::mkdir(plugin_directory_.c_str(), S_IRWXU) != 0) {
        throw CalibrationExecutionError(
            errno_message("cannot create calibration worker plugin directory"));
      }
      populate_plugin_directory();
      populate_runtime_overrides();
      if (::chmod(plugin_directory_.c_str(), S_IRUSR | S_IXUSR) != 0) {
        throw CalibrationExecutionError(
            errno_message("cannot seal calibration worker plugin directory"));
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  WorkerScratchTree(const WorkerScratchTree&) = delete;
  WorkerScratchTree& operator=(const WorkerScratchTree&) = delete;
  ~WorkerScratchTree() { cleanup(); }

  [[nodiscard]] const std::filesystem::path& scratch() const { return scratch_; }
  [[nodiscard]] const std::filesystem::path& plugin_directory() const { return plugin_directory_; }
  [[nodiscard]] const std::vector<std::pair<std::string, std::string>>& runtime_overrides() const {
    return runtime_overrides_;
  }

private:
  [[nodiscard]] static bool trusted_system_path(const std::filesystem::path& path) {
    if (!path.is_absolute()) {
      return false;
    }
    for (auto current = path; !current.empty(); current = current.parent_path()) {
      struct stat status{};
      if (::stat(current.c_str(), &status) != 0 || status.st_uid != 0 ||
          (status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        return false;
      }
      if (current == current.root_path()) {
        break;
      }
    }
    return true;
  }

  [[nodiscard]] static std::optional<std::filesystem::path>
  find_system_plugin(std::string_view filename) {
    static const std::array roots{
#if defined(__x86_64__)
        std::filesystem::path("/usr/lib/x86_64-linux-gnu/gstreamer-1.0"),
        std::filesystem::path("/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0"),
#elif defined(__aarch64__)
        std::filesystem::path("/usr/lib/aarch64-linux-gnu/gstreamer-1.0"),
        std::filesystem::path("/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0"),
#endif
        std::filesystem::path("/usr/lib64/gstreamer-1.0"),
        std::filesystem::path("/usr/lib/gstreamer-1.0"),
        std::filesystem::path("/usr/local/lib64/gstreamer-1.0"),
        std::filesystem::path("/usr/local/lib/gstreamer-1.0"),
        std::filesystem::path("/opt/nvidia/deepstream/deepstream/lib/gst-plugins"),
        std::filesystem::path("/opt/nvidia/deepstream/deepstream/lib/gstreamer-1.0"),
    };
    for (const auto& root : roots) {
      for (const auto& candidate : {root / filename, root / "deepstream" / filename}) {
        std::error_code error;
        const auto canonical = std::filesystem::canonical(candidate, error);
        if (error || !trusted_system_path(canonical)) {
          continue;
        }
        struct stat status{};
        if (::stat(canonical.c_str(), &status) == 0 && S_ISREG(status.st_mode)) {
          return canonical;
        }
      }
    }
    return std::nullopt;
  }

  void populate_plugin_directory() {
    static constexpr std::array plugin_files{
        "libgstcoreelements.so",
        "libgstisomp4.so",
        "libgstplayback.so",
        "libgstvideoparsersbad.so",
        "libgstapp.so",
        "libgstnvvideo4linux2.so",
        "libgstnvvideoconvert.so",
        "libgsttypefindfunctions.so",
    };
    for (const std::string_view filename : plugin_files) {
      const auto source = find_system_plugin(filename);
      if (!source.has_value()) {
        continue;
      }
      const auto destination = plugin_directory_ / filename;
      if (::symlink(source->c_str(), destination.c_str()) != 0) {
        throw CalibrationExecutionError(
            errno_message("cannot link trusted calibration worker plugin"));
      }
    }
  }

  void populate_runtime_overrides() {
    static constexpr std::array environment_names{
        "RECO_CUDA_DRIVER_DYLIB_PATH", "RECO_GSTREAMER_DYLIB_PATH", "RECO_GSTAPP_DYLIB_PATH",
        "RECO_GLIB_DYLIB_PATH",        "RECO_GOBJECT_DYLIB_PATH",   "RECO_NVBUFSURFACE_DYLIB_PATH",
        "RECO_NVDS_UTILS_DYLIB_PATH",
    };
    const auto runtime_directory = scratch_ / "runtime";
    if (::mkdir(runtime_directory.c_str(), S_IRWXU) != 0) {
      throw CalibrationExecutionError(
          errno_message("cannot create calibration worker runtime directory"));
    }
    constexpr std::uint64_t maximum_runtime_bytes = 256ULL * 1024ULL * 1024ULL;
    std::size_t runtime_index = 0;
    for (const char* name : environment_names) {
      const char* raw_path = std::getenv(name);
      if (raw_path == nullptr || raw_path[0] == '\0') {
        continue;
      }
      const std::filesystem::path source_path(raw_path);
      if (!source_path.is_absolute()) {
        throw CalibrationExecutionError("calibration worker runtime path must be absolute");
      }
      UniqueFd source(::open(source_path.c_str(), O_RDONLY | O_CLOEXEC));
      struct stat before{};
      if (!source || ::fstat(source.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
          before.st_size <= 0 ||
          static_cast<std::uint64_t>(before.st_size) > maximum_runtime_bytes) {
        throw CalibrationExecutionError("calibration worker runtime path must name a regular file");
      }
      const auto destination = runtime_directory / ("runtime-" + std::to_string(runtime_index++));
      UniqueFd output(::open(destination.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0500));
      if (!output) {
        throw CalibrationExecutionError(
            errno_message("cannot create calibration worker runtime snapshot"));
      }
      std::array<char, 64U * 1024U> buffer{};
      while (true) {
        ssize_t received = -1;
        do {
          received = ::read(source.get(), buffer.data(), buffer.size());
        } while (received < 0 && errno == EINTR);
        if (received < 0) {
          throw CalibrationExecutionError(
              errno_message("cannot read calibration worker runtime snapshot"));
        }
        if (received == 0) {
          break;
        }
        std::size_t offset = 0;
        while (offset < static_cast<std::size_t>(received)) {
          ssize_t written = -1;
          do {
            written = ::write(output.get(), buffer.data() + offset,
                              static_cast<std::size_t>(received) - offset);
          } while (written < 0 && errno == EINTR);
          if (written <= 0) {
            throw CalibrationExecutionError(
                errno_message("cannot write calibration worker runtime snapshot"));
          }
          offset += static_cast<std::size_t>(written);
        }
      }
      struct stat after{};
      struct stat snapshot{};
      if (::fstat(source.get(), &after) != 0 || ::fstat(output.get(), &snapshot) != 0 ||
          before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
          before.st_size != after.st_size || before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
          before.st_mtim.tv_nsec != after.st_mtim.tv_nsec || snapshot.st_size != before.st_size ||
          ::fchmod(output.get(), S_IRUSR | S_IXUSR) != 0) {
        throw CalibrationExecutionError("calibration worker runtime changed while snapshotted");
      }
      runtime_overrides_.emplace_back(name, destination.string());
    }
    if (::chmod(runtime_directory.c_str(), S_IRUSR | S_IXUSR) != 0) {
      throw CalibrationExecutionError(
          errno_message("cannot seal calibration worker runtime directory"));
    }
  }

  void cleanup() noexcept {
    try {
      if (!plugin_directory_.empty()) {
        (void)::chmod(plugin_directory_.c_str(), S_IRWXU);
      }
      if (!scratch_.empty()) {
        (void)::chmod((scratch_ / "runtime").c_str(), S_IRWXU);
      }
      std::error_code error;
      (void)std::filesystem::remove_all(root_, error);
    } catch (...) {
    }
  }

  std::filesystem::path root_;
  std::filesystem::path scratch_;
  std::filesystem::path plugin_directory_;
  std::vector<std::pair<std::string, std::string>> runtime_overrides_;
};

[[noreturn]] void exec_child(int executable, char* const* argv, char* const* environment) noexcept {
  reset_child_signals();
  constexpr int pinned_exec_descriptor = 3;
  if (executable != pinned_exec_descriptor) {
    if (::dup3(executable, pinned_exec_descriptor, O_CLOEXEC) != pinned_exec_descriptor) {
      child_exit();
    }
    (void)::close(executable);
    executable = pinned_exec_descriptor;
  }
  close_child_descriptors_except(executable);
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
  // TSan may self-reexec and cannot reliably recover an AT_EMPTY_PATH image.
  (void)::execve(argv[0], argv, environment);
#elif defined(SYS_execveat)
  if (!install_initial_exec_filter(executable)) {
    child_exit();
  }
  (void)::syscall(SYS_execveat, executable, "", argv, environment, AT_EMPTY_PATH);
#endif
  const auto error = errno;
  const char* failure = error == EMFILE   ? "calibration child exec failed: EMFILE\n"
                        : error == EPERM  ? "calibration child exec failed: EPERM\n"
                        : error == EACCES ? "calibration child exec failed: EACCES\n"
                        : error == ENOENT ? "calibration child exec failed: ENOENT\n"
                        : error == ENOMEM ? "calibration child exec failed: ENOMEM\n"
                        : error == EINVAL ? "calibration child exec failed: EINVAL\n"
                                          : "calibration child exec failed\n";
  ssize_t written = -1;
  do {
    written = ::write(STDERR_FILENO, failure, std::strlen(failure));
  } while (written < 0 && errno == EINTR);
  child_exit();
}

[[noreturn]] void guardian_child(int executable, char* const* argv, char* const* environment,
                                 int gate_read, int gate_write, pid_t expected_parent) noexcept {
  (void)::close(gate_write);
  if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || ::getppid() != expected_parent ||
      ::prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) != 0) {
    child_exit();
  }
  char ready = '\0';
  ssize_t received = -1;
  do {
    received = ::read(gate_read, &ready, 1);
  } while (received < 0 && errno == EINTR);
  if (received != 1 || ready != '1') {
    child_exit();
  }
  (void)::close(gate_read);
  exec_child(executable, argv, environment);
}

[[noreturn]] void worker_child(int executable, char* const* argv, char* const* environment,
                               std::uint64_t memory_limit, int child_gate, int parent_gate,
                               const std::filesystem::path& scratch, int left_input,
                               int right_input) noexcept {
  (void)::close(parent_gate);
  char mapped = '\0';
  ssize_t map_received = -1;
  do {
    map_received = ::recv(child_gate, &mapped, 1, 0);
  } while (map_received < 0 && errno == EINTR);
  if (map_received != 1 || mapped != 'S' || ::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
    child_exit();
  }
  if (!install_worker_filesystem_boundary(scratch.c_str(), left_input, right_input)) {
    child_exit();
  }
  apply_worker_limit(memory_limit);
  const char ready = 'R';
  ssize_t sent = -1;
  do {
    sent = ::send(child_gate, &ready, 1, MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  char release = '\0';
  ssize_t received = -1;
  do {
    received = ::recv(child_gate, &release, 1, 0);
  } while (received < 0 && errno == EINTR);
  if (sent != 1 || received != 1 || release != '1') {
    static constexpr char failure[] = "calibration worker gate failed\n";
    ssize_t written = -1;
    do {
      written = ::write(STDERR_FILENO, failure, sizeof(failure) - 1U);
    } while (written < 0 && errno == EINTR);
    child_exit();
  }
  (void)::close(child_gate);
  if (!install_worker_pre_main_filter()) {
    child_exit();
  }
  exec_child(executable, argv, environment);
}

template <typename Child>
[[nodiscard]] OwnedProcess
clone_process(std::uint64_t flags, Child&& child, int cgroup = -1,
              Clock::time_point cleanup_deadline = Clock::time_point::min()) {
#if !defined(SYS_clone3)
  (void)flags;
  (void)child;
  throw CalibrationExecutionError(
      "Linux clone3 is unavailable; calibration process isolation fails closed");
#else
  int pidfd = -1;
  clone_args arguments{};
  arguments.flags = flags | CLONE_PIDFD;
  if (cgroup >= 0) {
    arguments.flags |= CLONE_INTO_CGROUP;
    arguments.cgroup = static_cast<std::uint64_t>(cgroup);
  }
  arguments.pidfd = reinterpret_cast<std::uint64_t>(&pidfd);
  arguments.exit_signal = SIGCHLD;
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
  __sanitizer_syscall_pre_impl_fork();
#endif
  const auto result = static_cast<pid_t>(::syscall(SYS_clone3, &arguments, sizeof(arguments)));
#if defined(RECO_CALIBRATION_THREAD_SANITIZER)
  __sanitizer_syscall_post_impl_fork(result);
#endif
  if (result == 0) {
    child();
  }
  if (result < 0) {
    throw CalibrationExecutionError(
        errno_message("cannot create an atomically owned calibration process"));
  }
  if (pidfd < 0) {
    throw CalibrationExecutionError("clone3 did not return calibration process authority");
  }
  return OwnedProcess(result, pidfd, cleanup_deadline);
#endif
}

template <typename Child>
[[nodiscard]] OwnedProcess clone_process_with_stable_parent(std::uint64_t flags, Child&& child,
                                                            Clock::time_point cleanup_deadline) {
  std::array<int, 2> stop_pipe{-1, -1};
  if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, stop_pipe.data()) !=
      0) {
    throw CalibrationExecutionError(errno_message("cannot create calibration launcher lifetime"));
  }
  UniqueFd stop_read(stop_pipe[0]);
  UniqueFd stop_write(stop_pipe[1]);
  struct LaunchState {
    std::mutex mutex;
    std::condition_variable ready;
    pid_t pid = -1;
    int pidfd = -1;
    bool complete = false;
  } state;

  std::thread launcher([flags, child = std::forward<Child>(child), &state,
                        stop = std::move(stop_read), cleanup_deadline]() mutable {
    pid_t pid = -1;
    int authority = -1;
    int monitor = -1;
    try {
      auto process = clone_process(flags, std::move(child), -1, cleanup_deadline);
      pid = process.pid();
      authority = process.release_pidfd();
      monitor = ::fcntl(authority, F_DUPFD_CLOEXEC, 3);
      if (monitor < 0) {
        signal_pidfd_noexcept(authority, SIGKILL);
        (void)::close(authority);
        authority = -1;
        pid = -1;
      }
    } catch (...) {
      pid = -1;
      authority = -1;
    }
    {
      std::lock_guard lock(state.mutex);
      state.pid = pid;
      state.pidfd = authority;
      state.complete = true;
      state.ready.notify_one();
    }
    if (monitor < 0) {
      return;
    }
    UniqueFd monitor_fd(monitor);
    while (true) {
      std::array<pollfd, 2> items{
          pollfd{.fd = monitor_fd.get(), .events = POLLIN, .revents = 0},
          pollfd{.fd = stop.get(), .events = POLLIN, .revents = 0},
      };
      const auto result = ::poll(items.data(), items.size(), -1);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      return;
    }
  });

  {
    std::unique_lock lock(state.mutex);
    state.ready.wait(lock, [&state] { return state.complete; });
  }
  if (state.pid <= 0 || state.pidfd < 0) {
    const char stop = 'S';
    (void)::send(stop_write.get(), &stop, 1, MSG_NOSIGNAL);
    stop_write.reset();
    launcher.join();
    throw CalibrationExecutionError("cannot create stable calibration launcher process");
  }
  return OwnedProcess(state.pid, state.pidfd, std::move(launcher), stop_write.release(),
                      cleanup_deadline);
}

[[nodiscard]] std::vector<char*> make_argv(const std::string& executable,
                                           std::vector<std::string>& arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 2U);
  argv.push_back(const_cast<char*>(executable.c_str()));
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);
  return argv;
}

class SanitizedEnvironment {
public:
  explicit SanitizedEnvironment(bool worker, const WorkerScratchTree* worker_scratch = nullptr) {
    constexpr std::size_t maximum_entries = 4096;
    constexpr std::size_t maximum_bytes = 1024U * 1024U;
    storage_.reserve(128);
    std::size_t bytes = 0;
    for (auto** item = environ; item != nullptr && *item != nullptr; ++item) {
      const std::string_view entry(*item);
      const auto separator = entry.find('=');
      if (separator == std::string_view::npos || separator == 0) {
        continue;
      }
      const auto name = entry.substr(0, separator);
      if (unsafe_name(name) ||
          (worker && (overridden_worker_name(name) || !allowed_worker_name(name)))) {
        continue;
      }
      if (storage_.size() >= maximum_entries || entry.size() > maximum_bytes - bytes) {
        throw CalibrationExecutionError("calibration worker environment exceeds its bound");
      }
      bytes += entry.size();
      storage_.emplace_back(entry);
    }
    if (worker) {
      if (worker_scratch == nullptr || !worker_scratch->scratch().is_absolute() ||
          !worker_scratch->plugin_directory().is_absolute()) {
        throw CalibrationExecutionError("calibration worker scratch path is invalid");
      }
      const auto scratch = worker_scratch->scratch().string();
      const auto registry = scratch + "/gstreamer-registry.bin";
      storage_.emplace_back("GST_REGISTRY=" + registry);
      storage_.emplace_back("GST_REGISTRY_1_0=" + registry);
      storage_.emplace_back("GST_REGISTRY_FORK=no");
      storage_.emplace_back("GST_REGISTRY_UPDATE=no");
      storage_.emplace_back("GST_PLUGIN_SYSTEM_PATH_1_0=" +
                            worker_scratch->plugin_directory().string());
      storage_.emplace_back(
          "GST_PLUGIN_LOADING_WHITELIST=coreelements,isomp4,playback,videoparsersbad,app,"
          "nvvideo4linux2,nvvideoconvert,typefindfunctions");
      storage_.emplace_back("CUDA_CACHE_DISABLE=0");
      storage_.emplace_back("CUDA_CACHE_PATH=" + scratch + "/cuda-cache");
      storage_.emplace_back("__GL_SHADER_DISK_CACHE=0");
#if !defined(RECO_CALIBRATION_THREAD_SANITIZER)
      storage_.emplace_back("RECO_CALIBRATION_PRE_MAIN_RESTRICTED=1");
#endif
      storage_.emplace_back("HOME=" + scratch);
      storage_.emplace_back("XDG_CACHE_HOME=" + scratch);
      storage_.emplace_back("TMPDIR=" + scratch);
      storage_.emplace_back("TMP=" + scratch);
      storage_.emplace_back("TEMP=" + scratch);
      for (const auto& [name, path] : worker_scratch->runtime_overrides()) {
        storage_.emplace_back(name + "=" + path);
      }
    }
    pointers_.reserve(storage_.size() + 1U);
    for (auto& entry : storage_) {
      pointers_.push_back(entry.data());
    }
    pointers_.push_back(nullptr);
  }

  [[nodiscard]] char* const* data() { return pointers_.data(); }

private:
  [[nodiscard]] static bool unsafe_name(std::string_view name) {
    return name.starts_with("LD_") || name == "GLIBC_TUNABLES" || name == "GCONV_PATH" ||
           name == "LOCPATH" || name == "NLSPATH" || name == "MALLOC_TRACE" ||
           name == "HOSTALIASES" || name == "LOCALDOMAIN" || name == "RES_OPTIONS" ||
           name == "QT_PLUGIN_PATH" || name == "QML2_IMPORT_PATH" || name == "GST_PLUGIN_PATH" ||
           name == "GST_PLUGIN_PATH_1_0" || name == "GST_PLUGIN_SYSTEM_PATH" ||
           name == "GST_PLUGIN_SYSTEM_PATH_1_0" || name == "GST_PLUGIN_SCANNER" ||
           name == "GST_PRELOAD" || name == "GST_REGISTRY" || name == "GST_REGISTRY_1_0" ||
           name == "CUDA_INJECTION32_PATH" || name == "CUDA_INJECTION64_PATH";
  }

  [[nodiscard]] static bool overridden_worker_name(std::string_view name) {
    return name == "GST_REGISTRY" || name == "GST_REGISTRY_1_0" || name == "GST_REGISTRY_FORK" ||
           name == "GST_REGISTRY_UPDATE" || name == "GST_PLUGIN_LOADING_WHITELIST" ||
           name == "CUDA_CACHE_DISABLE" || name == "GST_PLUGIN_SYSTEM_PATH" ||
           name == "GST_PLUGIN_SYSTEM_PATH_1_0" || name == "__GL_SHADER_DISK_CACHE" ||
           name == "RECO_CALIBRATION_PRE_MAIN_RESTRICTED" || name == "CUDA_CACHE_PATH" ||
           name == "HOME" || name == "XDG_CACHE_HOME" || name == "TMPDIR" || name == "TMP" ||
           name == "TEMP" || name == "RECO_CUDA_DRIVER_DYLIB_PATH" ||
           name == "RECO_GSTREAMER_DYLIB_PATH" || name == "RECO_GSTAPP_DYLIB_PATH" ||
           name == "RECO_GLIB_DYLIB_PATH" || name == "RECO_GOBJECT_DYLIB_PATH" ||
           name == "RECO_NVBUFSURFACE_DYLIB_PATH" || name == "RECO_NVDS_UTILS_DYLIB_PATH";
  }

  [[nodiscard]] static bool allowed_worker_name(std::string_view name) {
    return name == "ASAN_OPTIONS" || name == "LSAN_OPTIONS" || name == "MSAN_OPTIONS" ||
           name == "TSAN_OPTIONS" || name == "UBSAN_OPTIONS" || name == "CUDA_DEVICE_ORDER" ||
           name == "CUDA_MODULE_LOADING" || name == "CUDA_VISIBLE_DEVICES" ||
           name == "NVIDIA_DRIVER_CAPABILITIES" || name == "NVIDIA_VISIBLE_DEVICES" ||
           name == "RECO_CUDA_DRIVER_DYLIB_PATH" || name == "RECO_NVBUFSURFACE_DYLIB_PATH" ||
           name == "RECO_NVDS_UTILS_DYLIB_PATH" || name == "RECO_GSTREAMER_DYLIB_PATH" ||
           name == "RECO_GSTAPP_DYLIB_PATH" || name == "RECO_GLIB_DYLIB_PATH" ||
           name == "RECO_GOBJECT_DYLIB_PATH" || name == "RECO_FAKE_GST_EVENT_PATH" ||
           name == "RECO_FAKE_GST_SCENARIO" || name == "RECO_FAKE_CALIBRATION_FORBIDDEN_FD" ||
           name == "RECO_FAKE_CALIBRATION_METADATA_TARGET" ||
           name == "RECO_FAKE_CALIBRATION_PRE_REQUEST_DELAY_MS" ||
           name == "RECO_FAKE_CALIBRATION_SIGNAL_TARGET_PID" ||
           name == "RECO_FAKE_CALIBRATION_WORKER_PID_PATH" ||
           name == "RECO_FAKE_CALIBRATION_WORKER_SCENARIO" ||
           name == "RECO_FAKE_CALIBRATION_WRITE_TARGET";
  }

  std::vector<std::string> storage_;
  std::vector<char*> pointers_;
};

[[nodiscard]] OwnedProcess spawn_guardian(const PinnedExecutable& executable,
                                          std::string_view address, Clock::time_point deadline) {
  const auto& executable_text = executable.display_path();
  std::vector<std::string> arguments{std::string(kCalibrationGuardianArgument),
                                     std::string(address),
                                     std::to_string(time_point_nanoseconds(deadline))};
  auto argv = make_argv(executable_text, arguments);
  SanitizedEnvironment environment(false);
  std::array<int, 2> gate{-1, -1};
  if (::pipe2(gate.data(), O_CLOEXEC) != 0) {
    throw CalibrationExecutionError(errno_message("cannot create calibration namespace gate"));
  }
  UniqueFd gate_read(gate[0]);
  UniqueFd gate_write(gate[1]);
  const auto expected_parent = ::getpid();
  auto process = clone_process_with_stable_parent(
      0,
      [&] {
        guardian_child(executable.fd(), argv.data(), environment.data(), gate_read.get(),
                       gate_write.get(), expected_parent);
      },
      deadline);
  gate_read.reset();
  const char ready = '1';
  ssize_t written = -1;
  do {
    written = ::write(gate_write.get(), &ready, 1);
  } while (written < 0 && errno == EINTR);
  if (written != 1) {
    throw CalibrationExecutionError(errno_message("cannot release calibration namespace gate"));
  }
  gate_write.reset();
  return process;
}

class GatedWorker {
public:
  GatedWorker(std::unique_ptr<WorkerScratchTree> sandbox, OwnedProcess process, int gate)
      : sandbox_(std::move(sandbox)), process_(std::move(process)), gate_(gate) {}
  GatedWorker(const GatedWorker&) = delete;
  GatedWorker& operator=(const GatedWorker&) = delete;
  GatedWorker(GatedWorker&&) noexcept = default;
  GatedWorker& operator=(GatedWorker&&) noexcept = default;

  [[nodiscard]] OwnedProcess& process() { return process_; }
  void release(Clock::time_point deadline) {
    wait_for_socket(gate_.get(), process_, POLLOUT, deadline);
    const char release = '1';
    ssize_t sent = -1;
    do {
      sent = ::send(gate_.get(), &release, 1, MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);
    if (sent != 1) {
      throw CalibrationExecutionError("cannot release the gated calibration worker");
    }
    gate_.reset();
  }

private:
  std::unique_ptr<WorkerScratchTree> sandbox_;
  OwnedProcess process_;
  UniqueFd gate_;
};

[[nodiscard]] GatedWorker spawn_worker(const std::string& executable, int executable_fd,
                                       int cgroup_fd, std::string_view address,
                                       Clock::time_point deadline,
                                       Clock::time_point cleanup_deadline,
                                       std::uint64_t memory_limit, int left_input,
                                       int right_input) {
  std::vector<std::string> arguments{std::string(kCalibrationWorkerIpcArgument),
                                     std::string(address),
                                     std::to_string(time_point_nanoseconds(deadline))};
  auto argv = make_argv(executable, arguments);
  auto scratch = std::make_unique<WorkerScratchTree>();
  SanitizedEnvironment environment(true, scratch.get());
  std::array<int, 2> gate{-1, -1};
  if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, gate.data()) != 0) {
    throw CalibrationExecutionError(errno_message("cannot create calibration worker gate"));
  }
  UniqueFd parent_gate(gate[0]);
  UniqueFd child_gate(gate[1]);
  auto process = clone_process(
      0,
      [&] {
        worker_child(executable_fd, argv.data(), environment.data(), memory_limit, child_gate.get(),
                     parent_gate.get(), scratch->scratch(), left_input, right_input);
      },
      cgroup_fd, cleanup_deadline);
  child_gate.reset();
  const char mapped = 'S';
  ssize_t map_sent = -1;
  do {
    map_sent = ::send(parent_gate.get(), &mapped, 1, MSG_NOSIGNAL);
  } while (map_sent < 0 && errno == EINTR);
  if (map_sent != 1) {
    throw CalibrationExecutionError("cannot release calibration worker setup gate");
  }
  wait_for_socket(parent_gate.get(), process, POLLIN, deadline);
  char ready = '\0';
  const auto received = ::recv(parent_gate.get(), &ready, 1, 0);
  if (received != 1 || ready != 'R') {
    throw CalibrationExecutionError("calibration worker did not reach its launch gate");
  }
  if (const char* marker = std::getenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH");
      marker != nullptr && marker[0] != '\0') {
    std::ofstream output(marker);
    output << process.pid();
  }
  return GatedWorker(std::move(scratch), std::move(process), parent_gate.release());
}

[[nodiscard]] CalibrationWorkerMessage response_message(std::string_view response) {
  if (response.size() < kCalibrationWorkerFrameHeaderBytes) {
    throw CalibrationExecutionError("calibration worker response is truncated");
  }
  CalibrationWorkerFrameHeader header{};
  std::copy_n(response.data(), header.size(), header.data());
  return decode_calibration_worker_header(header).message;
}

void certify_channel_eof(int descriptor) {
  char trailing = '\0';
  const auto received = ::recv(descriptor, &trailing, 1, MSG_DONTWAIT);
  if (received > 0) {
    throw CalibrationExecutionError("calibration worker returned trailing protocol bytes");
  }
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    throw CalibrationExecutionError(
        "calibration worker response channel remains open after worker exit");
  }
  if (received < 0) {
    throw CalibrationExecutionError(errno_message("cannot certify calibration worker response"));
  }
}

[[nodiscard]] std::string
supervise_worker(int caller_socket, const std::string& executable, int executable_fd, int cgroup_fd,
                 const std::string& encoded_request, const GpuCalibrationRequest& request,
                 Clock::time_point worker_deadline, Clock::time_point cleanup_deadline,
                 int left_input_fd, int right_input_fd, bool* authority_reported) {
  auto listener = create_listener();
  auto worker = spawn_worker(
      executable, executable_fd, cgroup_fd, listener.address, worker_deadline, cleanup_deadline,
      request.calibration_host_memory_limit_bytes, left_input_fd, right_input_fd);
  try {
    send_file_fd(caller_socket, worker.process(), 'P', worker.process().pidfd(), worker_deadline,
                 -1, request.calibration_host_memory_limit_bytes);
    *authority_reported = true;
    worker.release(worker_deadline);
    auto channel = accept_authenticated(listener, worker.process(), worker_deadline,
                                        request.calibration_host_memory_limit_bytes);
    write_all(channel.get(), worker.process(), encoded_request, worker_deadline, caller_socket,
              request.calibration_host_memory_limit_bytes);
    if (left_input_fd >= 0) {
      send_file_fd(channel.get(), worker.process(), 'I', left_input_fd, worker_deadline,
                   caller_socket, request.calibration_host_memory_limit_bytes);
    }
    if (right_input_fd >= 0) {
      send_file_fd(channel.get(), worker.process(), 'J', right_input_fd, worker_deadline,
                   caller_socket, request.calibration_host_memory_limit_bytes);
    }
    if (::shutdown(channel.get(), SHUT_WR) != 0) {
      throw CalibrationExecutionError("cannot finish the calibration worker request");
    }
    auto response = read_frame(channel.get(), worker.process(), worker_deadline, caller_socket,
                               request.calibration_host_memory_limit_bytes);
    wait_for_process_monitored(worker.process(), worker_deadline,
                               request.calibration_host_memory_limit_bytes);
    const auto status = worker.process().reap();
    certify_channel_eof(channel.get());

    const auto message = response_message(response);
    if (message == CalibrationWorkerMessage::Success &&
        (!status.known || status.code != CLD_EXITED || status.status != EXIT_SUCCESS)) {
      throw CalibrationExecutionError(
          "calibration worker did not exit successfully after returning a result");
    }
    if (status.known && status.code != CLD_EXITED) {
      throw CalibrationExecutionError("calibration worker terminated abnormally");
    }
    return response;
  } catch (...) {
    if (cgroup_oom_killed_noexcept(cgroup_fd)) {
      throw CalibrationExecutionError("calibration worker exceeded its cgroup host memory limit");
    }
    throw;
  }
}

} // namespace

bool install_calibration_worker_sandbox() noexcept { return install_worker_syscall_filter(); }

int run_calibration_guardian_fd(int descriptor, const char* executable,
                                std::uint64_t deadline_nanoseconds) {
  if (descriptor < 0 || executable == nullptr) {
    return EXIT_FAILURE;
  }
  const auto deadline = deadline_from_nanoseconds(deadline_nanoseconds);
  UniqueFd admission;
  UniqueFd executable_image;
  UniqueFd cgroup;
  bool authority_reported = false;
  try {
    admission.reset(receive_calibration_admission_fd(descriptor, deadline_nanoseconds));
    executable_image.reset(receive_calibration_executable_fd(descriptor, deadline_nanoseconds));
    cgroup.reset(receive_calibration_cgroup_fd(descriptor, deadline_nanoseconds));
    const auto encoded_request = read_plain_frame(descriptor, deadline);
    auto request = decode_calibration_worker_request(encoded_request);
    UniqueFd left_input;
    UniqueFd right_input;
    if (request.left.retained_path.has_value()) {
      left_input.reset(receive_calibration_left_input_fd(descriptor, deadline_nanoseconds));
    }
    if (request.right.retained_path.has_value()) {
      right_input.reset(receive_calibration_right_input_fd(descriptor, deadline_nanoseconds));
    }
    const auto now = Clock::now();
    if (now >= deadline || deadline - now <= kCleanupReserve) {
      throw CalibrationExecutionError("calibration deadline leaves no cleanup reserve");
    }
    const auto response =
        supervise_worker(descriptor, executable, executable_image.get(), cgroup.get(),
                         encoded_request, request, deadline - kCleanupReserve, deadline,
                         left_input.get(), right_input.get(), &authority_reported);
    write_plain_all(descriptor, response, deadline);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    try {
      if (!authority_reported) {
        write_plain_all(descriptor, "N", deadline);
      }
      write_plain_all(descriptor, encode_calibration_worker_failure(error.what()), deadline);
    } catch (...) {
    }
    return EXIT_FAILURE;
  } catch (...) {
    try {
      if (!authority_reported) {
        write_plain_all(descriptor, "N", deadline);
      }
      write_plain_all(descriptor,
                      encode_calibration_worker_failure("unknown calibration guardian failure"),
                      deadline);
    } catch (...) {
    }
    return EXIT_FAILURE;
  }
}

CalibrationResult run_gpu_calibration_supervised(const GpuCalibrationRequest& request) {
  const auto timeout = std::chrono::nanoseconds(request.calibration_timeout_ns);
  if (timeout <= kCleanupReserve || Clock::time_point::max() - Clock::now() < timeout) {
    throw CalibrationExecutionError("calibration timeout is outside the supervisor clock range");
  }
  const auto deadline = Clock::now() + timeout;
  AdmissionLock admission;
  check_admission_headroom(request.calibration_host_memory_limit_bytes);
  PinnedExecutable executable(std::filesystem::path(request.calibration_worker_path), deadline);
  const auto encoded_request = encode_calibration_worker_request(request);
  const auto open_retained_input = [](const std::optional<std::string>& path,
                                      std::string_view label) {
    UniqueFd descriptor;
    if (!path.has_value()) {
      return descriptor;
    }
    descriptor.reset(::open(path->c_str(), O_RDONLY | O_CLOEXEC));
    struct stat identity{};
    if (!descriptor || ::fstat(descriptor.get(), &identity) != 0 || !S_ISREG(identity.st_mode)) {
      throw CalibrationExecutionError("cannot retain " + std::string(label) +
                                      " calibration video identity");
    }
    return descriptor;
  };
  auto left_input = open_retained_input(request.left.retained_path, "left");
  auto right_input = open_retained_input(request.right.retained_path, "right");
  CgroupMemoryBoundary memory_boundary(request.calibration_host_memory_limit_bytes, deadline);
  try {
    auto listener = create_listener();
    auto guardian = spawn_guardian(executable, listener.address, deadline);
    auto channel = accept_authenticated(listener, guardian, deadline);
    send_file_fd(channel.get(), guardian, 'L', admission.fd(), deadline);
    send_file_fd(channel.get(), guardian, 'X', executable.fd(), deadline);
    send_file_fd(channel.get(), guardian, 'C', memory_boundary.fd(), deadline);
    write_all(channel.get(), guardian, encoded_request, deadline);
    if (left_input) {
      send_file_fd(channel.get(), guardian, 'I', left_input.get(), deadline);
    }
    if (right_input) {
      send_file_fd(channel.get(), guardian, 'J', right_input.get(), deadline);
    }
    if (::shutdown(channel.get(), SHUT_WR) != 0) {
      throw CalibrationExecutionError("cannot finish the calibration guardian request");
    }
    auto worker_authority = receive_worker_authority(channel.get(), guardian, deadline);
    (void)worker_authority;
    auto response = read_frame(channel.get(), guardian, deadline);
    guardian.wait_until(deadline);
    (void)guardian.reap();
    certify_channel_eof(channel.get());
    if (memory_boundary.oom_killed()) {
      throw CalibrationExecutionError("calibration worker exceeded its cgroup host memory limit");
    }
    auto result = decode_calibration_worker_response(response);
    memory_boundary.finish();
    return result;
  } catch (...) {
    if (memory_boundary.oom_killed()) {
      throw CalibrationExecutionError("calibration worker exceeded its cgroup host memory limit");
    }
    throw;
  }
}

#endif

} // namespace reco::calibrate::detail
