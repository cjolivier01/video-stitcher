#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
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
#include <unistd.h>
#endif

namespace reco::io::detail {
int run_gpu_video_probe_worker();
#if defined(_WIN32)
int run_gpu_video_probe_guardian();
#endif
#if !defined(_WIN32)
int run_gpu_video_probe_guard();
#endif
} // namespace reco::io::detail

#if !defined(_WIN32)
namespace {

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

} // namespace
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
  if (argc == 2 && std::strcmp(argv[1], "--reco-video-probe-guardian") == 0) {
    return reco::io::detail::run_gpu_video_probe_guardian();
  }
#endif
#if !defined(_WIN32)
  if (argc == 2 && std::strcmp(argv[1], "--reco-video-probe-guard") == 0) {
    return reco::io::detail::run_gpu_video_probe_guard();
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
      (value[separator + 1] != '0' && value[separator + 1] != '1')) {
    return 2;
  }
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
