#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

namespace reco::io::detail {
int run_gpu_video_probe_worker();
#if defined(_WIN32)
int run_gpu_video_probe_guardian();
#else
int run_gpu_video_probe_supervisor(const char* executable, std::uint64_t pre_worker_report_delay_ns,
                                   std::uint64_t pre_guardian_exec_delay_ns,
                                   std::uint64_t caller_pid, bool has_marker);
int run_gpu_video_probe_guardian(const char* executable, std::uint64_t pre_worker_report_delay_ns);
#endif
} // namespace reco::io::detail

#if !defined(_WIN32)
namespace {

constexpr int kWorkerExecutable = 3;

#if defined(RECO_PROBE_TEST_WORKER_DESCRIPTOR_REUSE)
int reused_worker_descriptor = -1;

class WorkerDescriptorReuseFixture {
public:
  WorkerDescriptorReuseFixture() {
    const char* path = std::getenv("RECO_FAKE_PROBE_WORKER_DESCRIPTOR_REUSE_PATH");
    if (path != nullptr && path[0] != '\0') {
      reused_worker_descriptor =
          ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR);
    }
  }
};

WorkerDescriptorReuseFixture worker_descriptor_reuse_fixture;

bool audit_reused_worker_descriptor() {
  const char* path = std::getenv("RECO_FAKE_PROBE_WORKER_DESCRIPTOR_REUSE_PATH");
  if (path == nullptr || path[0] == '\0') {
    return true;
  }
  const char marker = '1';
  const bool valid = reused_worker_descriptor == kWorkerExecutable &&
                     ::write(reused_worker_descriptor, &marker, 1) == 1;
  if (reused_worker_descriptor >= 0) {
    (void)::close(reused_worker_descriptor);
    reused_worker_descriptor = -1;
  }
  return valid;
}
#endif

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
  return maximum;
}

bool close_unrelated_descriptors() {
#if defined(__APPLE__) || defined(__linux__)
#if defined(__APPLE__)
  constexpr const char* kDescriptorDirectory = "/dev/fd";
#else
  constexpr const char* kDescriptorDirectory = "/proc/self/fd";
#endif
  if (auto* directory = ::opendir(kDescriptorDirectory); directory != nullptr) {
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
#if defined(F_CLOSEM)
  if (::fcntl(3, F_CLOSEM, 0) == 0) {
    return true;
  }
#endif
  const auto maximum_descriptor = descriptor_scan_limit();
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

} // namespace
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
  if (argc == 2 && std::strcmp(argv[1], "--reco-video-probe-guardian") == 0) {
    return reco::io::detail::run_gpu_video_probe_guardian();
  }
#else
  if (argc == 6 && std::strcmp(argv[1], "--reco-video-probe-supervisor") == 0) {
    std::array<std::uint64_t, 3> values{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      const std::string_view value(argv[index + 2U]);
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), values[index]);
      if (error != std::errc{} || end != value.data() + value.size()) {
        return 2;
      }
    }
    if (argv[5][0] == '\0' || argv[5][1] != '\0' || (argv[5][0] != '0' && argv[5][0] != '1')) {
      return 2;
    }
    return reco::io::detail::run_gpu_video_probe_supervisor(argv[0], values[0], values[1],
                                                            values[2], argv[5][0] == '1');
  }
  if (argc == 3 && std::strcmp(argv[1], "--reco-video-probe-guardian") == 0) {
    const std::string_view delay_value(argv[2]);
    std::uint64_t delay_ns = 0;
    const auto [end, error] =
        std::from_chars(delay_value.data(), delay_value.data() + delay_value.size(), delay_ns);
    if (error != std::errc{} || end != delay_value.data() + delay_value.size()) {
      return 2;
    }
    return reco::io::detail::run_gpu_video_probe_guardian(argv[0], delay_ns);
  }
#endif
  if (argc < 2 || std::strcmp(argv[1], "--reco-video-probe-worker") != 0) {
    return 2;
  }
#if defined(_WIN32)
  if (argc != 2) {
    return 2;
  }
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
    return 2;
  }
  return reco::io::detail::run_gpu_video_probe_worker();
#else
  if (argc != 3) {
    return 2;
  }
  const std::string_view value(argv[2]);
  const auto separator = value.find(':');
  if (separator == std::string_view::npos || separator + 2 != value.size() ||
      (value[separator + 1] != '0' && value[separator + 1] != '1' && value[separator + 1] != '2')) {
    return 2;
  }
  // TSan re-executes a descriptor-backed PIE once during runtime startup. Only
  // that explicitly identified launch retains descriptor 3 through exec.
  if (value[separator + 1] == '2') {
    (void)::close(kWorkerExecutable);
  }
#if defined(RECO_PROBE_TEST_WORKER_DESCRIPTOR_REUSE)
  if (!audit_reused_worker_descriptor()) {
    return 2;
  }
#endif
  const auto pid_value = value.substr(0, separator);
  std::uint64_t expected_parent_pid = 0;
  const auto [end, error] =
      std::from_chars(pid_value.data(), pid_value.data() + pid_value.size(), expected_parent_pid);
  if (error != std::errc{} || end != pid_value.data() + pid_value.size() ||
      expected_parent_pid == 0 ||
      expected_parent_pid > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max()) ||
      static_cast<std::uint64_t>(::getppid()) != expected_parent_pid ||
      (value[separator + 1] == '0' && !close_unrelated_descriptors())) {
    return 2;
  }
  return reco::io::detail::run_gpu_video_probe_worker();
#endif
}
