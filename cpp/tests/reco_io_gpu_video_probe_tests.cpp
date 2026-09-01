#include "reco/io/gpu_video_probe.hpp"

#include "gpu_video_probe_process_test.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/resource.h>
#if defined(__APPLE__)
#include <sys/xattr.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#include <sys/wait.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

using namespace reco::io;

namespace {

int failures = 0;
std::filesystem::path probe_worker_path;
std::filesystem::path fake_probe_worker_path;
#if !defined(_WIN32)
std::filesystem::path test_executable_path;

constexpr const char* kProbeCallerVideoPath = "RECO_FAKE_PROBE_CALLER_VIDEO_PATH";
constexpr const char* kProbeCallerWorkerPath = "RECO_FAKE_PROBE_CALLER_WORKER_PATH";
constexpr const char* kProbeCallerMode = "RECO_FAKE_PROBE_CALLER_MODE";
constexpr const char* kProbeCallerTimeout = "RECO_FAKE_PROBE_CALLER_TIMEOUT_NS";
constexpr const char* kProbeCallerDelay = "RECO_FAKE_PROBE_CALLER_DELAY_NS";
constexpr const char* kProbeCallerMarkerPath = "RECO_FAKE_PROBE_CALLER_MARKER_PATH";
constexpr const char* kProbeCallerInheritedChildPath =
    "RECO_FAKE_PROBE_CALLER_INHERITED_CHILD_PATH";
constexpr const char* kProbeCallerInheritedAuditPath =
    "RECO_FAKE_PROBE_CALLER_INHERITED_AUDIT_PATH";
volatile sig_atomic_t signal_fork_report_descriptor = -1;

void fork_from_signal_handler(int) {
  const auto saved_error = errno;
  const auto child = ::fork();
  if (child == 0) {
    ::_exit(EXIT_SUCCESS);
  }
  const auto descriptor = static_cast<int>(signal_fork_report_descriptor);
  if (child > 0 && descriptor >= 0) {
    (void)::write(descriptor, &child, sizeof(child));
  }
  errno = saved_error;
}
#endif

void expect_true(bool value, std::string_view message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_probe_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const GpuVideoProbeError& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_invalid_argument(Function&& function, std::string_view fragment,
                             std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

std::filesystem::path resolve_runfile(std::string path) {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
  const auto logical_path = std::string(workspace) + "/" + path;
  const auto resolved = std::filesystem::path(runfiles->Rlocation(logical_path));
  if (resolved.empty() || !std::filesystem::is_regular_file(resolved)) {
    throw std::runtime_error(path + " runfile not found");
  }
  return resolved;
}

std::filesystem::path executable_runfile(std::string path) {
#if defined(_WIN32)
  path += ".exe";
#endif
  return resolve_runfile(std::move(path));
}

void set_environment(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

std::optional<std::uint64_t>
wait_for_process_marker(const std::filesystem::path& path,
                        std::chrono::steady_clock::time_point deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream input(path);
    std::uint64_t process_id = 0;
    if (input >> process_id && process_id != 0) {
      return process_id;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

std::optional<std::pair<std::uint64_t, std::uint64_t>>
wait_for_process_pair_marker(const std::filesystem::path& path,
                             std::chrono::steady_clock::time_point deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream input(path);
    std::uint64_t first = 0;
    std::uint64_t second = 0;
    if (input >> first >> second && first != 0 && second != 0) {
      return std::pair(first, second);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

std::optional<char> wait_for_audit_marker(const std::filesystem::path& path,
                                          std::chrono::steady_clock::time_point deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream input(path);
    char value = '\0';
    if (input >> value && (value == '0' || value == '1')) {
      return value;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

#if defined(__APPLE__)
std::vector<std::filesystem::path> mac_probe_snapshot_directories() {
  std::vector<std::filesystem::path> result;
  std::error_code error;
  for (const auto& entry :
       std::filesystem::directory_iterator(std::filesystem::temp_directory_path(), error)) {
    if (error) {
      break;
    }
    const auto name = entry.path().filename().string();
    std::error_code status_error;
    if (name.rfind("reco-video-probe-", 0) == 0 && entry.is_directory(status_error) &&
        !status_error) {
      result.push_back(entry.path());
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}
#endif

#if defined(__linux__)
std::vector<pid_t> direct_children(pid_t parent) {
  std::vector<pid_t> result;
  const auto task_path = std::filesystem::path("/proc") / std::to_string(parent) / "task";
  std::error_code directory_error;
  for (const auto& task : std::filesystem::directory_iterator(task_path, directory_error)) {
    std::ifstream children(task.path() / "children");
    for (pid_t child = -1; children >> child;) {
      if (child > 0 && std::find(result.begin(), result.end(), child) == result.end()) {
        result.push_back(child);
      }
    }
  }
  return result;
}

std::string process_command_line(pid_t process) {
  std::ifstream input(std::filesystem::path("/proc") / std::to_string(process) / "cmdline",
                      std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::optional<pid_t> wait_for_direct_child(pid_t parent,
                                           std::chrono::steady_clock::time_point deadline,
                                           std::string_view command_fragment = {}) {
  while (std::chrono::steady_clock::now() < deadline) {
    for (const auto child : direct_children(parent)) {
      if (command_fragment.empty() ||
          process_command_line(child).find(command_fragment) != std::string::npos) {
        return child;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

std::optional<pid_t> wait_for_descendant(pid_t parent,
                                         std::chrono::steady_clock::time_point deadline,
                                         std::string_view command_fragment) {
  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<pid_t> pending{parent};
    std::vector<pid_t> visited;
    while (!pending.empty()) {
      const auto process = pending.back();
      pending.pop_back();
      if (std::find(visited.begin(), visited.end(), process) != visited.end()) {
        continue;
      }
      visited.push_back(process);
      for (const auto child : direct_children(process)) {
        if (process_command_line(child).find(command_fragment) != std::string::npos) {
          return child;
        }
        pending.push_back(child);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

bool process_has_linux_probe_memfd(pid_t process) {
  std::error_code directory_error;
  const auto descriptors = std::filesystem::path("/proc") / std::to_string(process) / "fd";
  for (const auto& entry : std::filesystem::directory_iterator(descriptors, directory_error)) {
    std::error_code link_error;
    const auto target = std::filesystem::read_symlink(entry.path(), link_error).string();
    if (!link_error && target.find("reco-video-probe") != std::string::npos) {
      return true;
    }
  }
  return false;
}
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
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define RECO_PROBE_THREAD_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define RECO_PROBE_THREAD_SANITIZER 1
#endif

#if defined(_WIN32)
class WindowsHandle {
public:
  WindowsHandle() = default;
  explicit WindowsHandle(HANDLE handle) : handle_(handle) {}
  WindowsHandle(const WindowsHandle&) = delete;
  WindowsHandle& operator=(const WindowsHandle&) = delete;
  WindowsHandle(WindowsHandle&&) = delete;
  WindowsHandle& operator=(WindowsHandle&&) = delete;
  ~WindowsHandle() {
    if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
  }

  [[nodiscard]] HANDLE get() const { return handle_; }
  [[nodiscard]] explicit operator bool() const {
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
  }

private:
  HANDLE handle_ = nullptr;
};

std::filesystem::path current_test_executable() {
  std::vector<wchar_t> path(32'768);
  const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (length == 0 || length >= path.size()) {
    throw std::runtime_error("failed to resolve the current test executable");
  }
  return std::filesystem::path(std::wstring(path.data(), length));
}
#endif

void set_scenario(std::string_view value) {
  set_environment("RECO_FAKE_GST_SCENARIO", std::string(value));
}

GpuFileDecodeConfig container_config(const std::filesystem::path& path) {
#if defined(_WIN32)
  const auto encoded = path.u8string();
  const std::string path_string(reinterpret_cast<const char*>(encoded.data()), encoded.size());
#else
  const auto path_string = path.string();
#endif
  return {.path = path_string,
          .codec = GpuDecodeCodec::H264,
          .elementary_stream = false,
          .container = GpuDecodeContainer::QuickTime,
          .max_buffers = 4,
          .drop = false};
}

GpuFileDecodeConfig elementary_config(const std::filesystem::path& path, GpuDecodeCodec codec) {
#if defined(_WIN32)
  const auto encoded = path.u8string();
  const std::string path_string(reinterpret_cast<const char*>(encoded.data()), encoded.size());
#else
  const auto path_string = path.string();
#endif
  return {.path = path_string,
          .codec = codec,
          .elementary_stream = true,
          .container = std::nullopt,
          .max_buffers = 4,
          .drop = false};
}

GpuVideoProbe probe_video(const GpuFileDecodeConfig& config, std::uint64_t timeout_ns) {
  try {
    return reco::io::probe_gpu_video(config, probe_worker_path, timeout_ns);
  } catch (const GpuVideoProbeError& error) {
    const char* scenario = std::getenv("RECO_FAKE_GST_SCENARIO");
    throw GpuVideoProbeError(std::string(scenario == nullptr ? "unknown" : scenario) + ": " +
                             error.what());
  }
}

#if !defined(RECO_PROBE_TEST_FORCE_GUARDIAN_FALLBACKS)
void exhaustive_calibration_probe_scans_to_eos(const std::filesystem::path& video_path) {
  constexpr std::uint64_t exhaustive_fixture_timeout_ns = 30'000'000'000ULL;
  set_scenario("probe-exhaustive-50001");
  const auto exact = reco::io::detail::probe_gpu_video_exhaustive_for_test(
      container_config(video_path), exhaustive_fixture_timeout_ns);
  expect_eq(exact.total_frames, 50'001ULL,
            "exhaustive calibration probe scans beyond the bounded AU ceiling");
  expect_true(!exact.total_frames_is_estimated && exact.selected_stream_caps_verified &&
                  exact.indexed_sampling_cadence_verified,
              "exhaustive calibration probe returns exact indexed metadata");

  set_scenario("probe-exhaustive-hidden-vfr");
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_exhaustive_for_test(container_config(video_path),
                                                                    exhaustive_fixture_timeout_ns);
      },
      "missing or duplicated", "exhaustive calibration probe catches an unsampled cadence gap");

  set_scenario("probe-exhaustive-endless");
  const auto started = std::chrono::steady_clock::now();
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_exhaustive_for_test(container_config(video_path),
                                                                    1'000'000'000ULL);
      },
      "timed out", "exhaustive calibration probe retains its end-to-end deadline");
  expect_true(std::chrono::steady_clock::now() - started < std::chrono::seconds(2),
              "exhaustive calibration deadline remains bounded");
  set_scenario("probe-ok");
}
#endif

#if !defined(_WIN32)
bool parse_probe_caller_duration(const char* value, std::uint64_t* duration_ns) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const auto text = std::string_view(value);
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), *duration_ns);
  return error == std::errc{} && end == text.data() + text.size();
}

int run_posix_probe_caller() {
  const char* video_path = std::getenv(kProbeCallerVideoPath);
  const char* worker_path = std::getenv(kProbeCallerWorkerPath);
  const char* mode_value = std::getenv(kProbeCallerMode);
  const char* marker_path = std::getenv(kProbeCallerMarkerPath);
  const char* inherited_child_path = std::getenv(kProbeCallerInheritedChildPath);
  const char* inherited_audit_path = std::getenv(kProbeCallerInheritedAuditPath);
  std::uint64_t timeout_ns = 0;
  std::uint64_t delay_ns = 0;
  if (video_path == nullptr || video_path[0] == '\0' || worker_path == nullptr ||
      worker_path[0] == '\0' || mode_value == nullptr || mode_value[0] == '\0' ||
      !parse_probe_caller_duration(std::getenv(kProbeCallerTimeout), &timeout_ns) ||
      !parse_probe_caller_duration(std::getenv(kProbeCallerDelay), &delay_ns)) {
    return EXIT_FAILURE;
  }

  const auto mode = std::string_view(mode_value);
  if (mode != "probe" && mode != "pre-worker-report-delay" && mode != "pre-supervisor-exec-delay" &&
      mode != "pre-supervisor-arm-delay" && mode != "pre-guardian-exec-delay") {
    return EXIT_FAILURE;
  }
  if ((mode == "pre-supervisor-exec-delay" || mode == "pre-supervisor-arm-delay" ||
       mode == "pre-guardian-exec-delay") &&
      (marker_path == nullptr || marker_path[0] == '\0')) {
    return EXIT_FAILURE;
  }
  if (inherited_child_path != nullptr && inherited_child_path[0] != '\0' &&
      (marker_path == nullptr || marker_path[0] == '\0')) {
    return EXIT_FAILURE;
  }

  std::atomic<bool> stop_inherited_child = false;
  std::atomic<pid_t> inherited_child = -1;
  std::thread inherited_child_thread;
  if (inherited_child_path != nullptr && inherited_child_path[0] != '\0') {
    const auto snapshot_marker = std::filesystem::path(marker_path);
    const auto child_marker = std::filesystem::path(inherited_child_path);
    const auto audit_marker = inherited_audit_path == nullptr
                                  ? std::filesystem::path{}
                                  : std::filesystem::path(inherited_audit_path);
    inherited_child_thread = std::thread([&, snapshot_marker, child_marker, audit_marker] {
      int snapshot_descriptor = -1;
      while (!stop_inherited_child.load(std::memory_order_acquire)) {
        std::ifstream marker(snapshot_marker);
        std::uint64_t process_id = 0;
        if (marker >> process_id && process_id != 0 &&
            (audit_marker.empty() || (marker >> snapshot_descriptor && snapshot_descriptor >= 0))) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      if (stop_inherited_child.load(std::memory_order_acquire)) {
        return;
      }
      const auto child = ::fork();
      if (child == 0) {
        if (!audit_marker.empty()) {
          errno = 0;
          const auto descriptor_status = ::fcntl(snapshot_descriptor, F_GETFD);
          const char retained = descriptor_status >= 0 || errno != EBADF ? '1' : '0';
          const int audit =
              ::open(audit_marker.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
          if (audit >= 0) {
            (void)::write(audit, &retained, 1);
            (void)::close(audit);
          }
        }
        while (true) {
          (void)::pause();
        }
      }
      if (child <= 0) {
        return;
      }
      inherited_child.store(child, std::memory_order_release);
      {
        std::ofstream marker(child_marker, std::ios::trunc);
        marker << static_cast<std::uint64_t>(child);
      }
      int status = 0;
      while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
      }
    });
  }

  int result = EXIT_SUCCESS;
  try {
    const auto config = container_config(std::filesystem::path(video_path));
    const auto worker = std::filesystem::path(worker_path);
    if (mode == "probe") {
      (void)reco::io::probe_gpu_video(config, worker, timeout_ns);
    } else if (mode == "pre-worker-report-delay") {
      (void)reco::io::detail::probe_gpu_video_with_pre_worker_report_delay_for_test(
          config, worker, timeout_ns, delay_ns);
      result = 2;
    } else if (mode == "pre-supervisor-exec-delay") {
      (void)reco::io::detail::probe_gpu_video_with_pre_supervisor_exec_delay_for_test(
          config, worker, timeout_ns, delay_ns, std::filesystem::path(marker_path));
    } else if (mode == "pre-supervisor-arm-delay") {
      (void)reco::io::detail::probe_gpu_video_with_pre_supervisor_arm_delay_for_test(
          config, worker, timeout_ns, delay_ns, std::filesystem::path(marker_path));
    } else {
      (void)reco::io::detail::probe_gpu_video_with_pre_guardian_exec_delay_for_test(
          config, worker, timeout_ns, delay_ns, std::filesystem::path(marker_path));
    }
  } catch (...) {
    result = EXIT_SUCCESS;
  }
  stop_inherited_child.store(true, std::memory_order_release);
  const auto child = inherited_child.load(std::memory_order_acquire);
  if (child > 0) {
    (void)::kill(child, SIGKILL);
  }
  if (inherited_child_thread.joinable()) {
    inherited_child_thread.join();
  }
  return result;
}

pid_t fork_exec_probe_caller(const std::filesystem::path& video_path,
                             const std::filesystem::path& worker_path, std::string_view mode,
                             std::uint64_t timeout_ns, std::uint64_t delay_ns = 0,
                             const std::filesystem::path& marker_path = {},
                             const std::filesystem::path& inherited_child_path = {},
                             const std::filesystem::path& inherited_audit_path = {},
                             bool new_process_group = false) {
  set_environment(kProbeCallerVideoPath, video_path.string());
  set_environment(kProbeCallerWorkerPath, worker_path.string());
  set_environment(kProbeCallerMode, std::string(mode));
  set_environment(kProbeCallerTimeout, std::to_string(timeout_ns));
  set_environment(kProbeCallerDelay, std::to_string(delay_ns));
  set_environment(kProbeCallerMarkerPath, marker_path.string());
  set_environment(kProbeCallerInheritedChildPath, inherited_child_path.string());
  set_environment(kProbeCallerInheritedAuditPath, inherited_audit_path.string());

  auto executable = test_executable_path.string();
  std::string helper_mode = "--reco-posix-probe-caller";
  std::array<char*, 3> arguments{executable.data(), helper_mode.data(), nullptr};
  const auto caller = ::fork();
  if (caller == 0) {
    if (new_process_group && ::setpgid(0, 0) != 0) {
      ::_exit(126);
    }
    ::execv(arguments[0], arguments.data());
    ::_exit(127);
  }

  set_environment(kProbeCallerVideoPath, "");
  set_environment(kProbeCallerWorkerPath, "");
  set_environment(kProbeCallerMode, "");
  set_environment(kProbeCallerTimeout, "");
  set_environment(kProbeCallerDelay, "");
  set_environment(kProbeCallerMarkerPath, "");
  set_environment(kProbeCallerInheritedChildPath, "");
  set_environment(kProbeCallerInheritedAuditPath, "");
  return caller;
}
#endif

#if defined(_WIN32)
int run_windows_parent_death_probe_caller() {
  const char* video_path = std::getenv("RECO_FAKE_PROBE_CALLER_VIDEO_PATH");
  const char* worker_path = std::getenv("RECO_FAKE_PROBE_CALLER_WORKER_PATH");
  if (video_path == nullptr || video_path[0] == '\0' || worker_path == nullptr ||
      worker_path[0] == '\0') {
    return EXIT_FAILURE;
  }
  try {
    (void)reco::io::probe_gpu_video(container_config(std::filesystem::path(video_path)),
                                    std::filesystem::path(worker_path), 30'000'000'000ULL);
  } catch (...) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#endif

std::vector<std::string> read_events(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::string> events;
  for (std::string line; std::getline(input, line);) {
    events.push_back(std::move(line));
  }
  return events;
}

bool has_event(const std::vector<std::string>& events, std::string_view value) {
  return std::find(events.begin(), events.end(), value) != events.end();
}

std::size_t count_event(const std::vector<std::string>& events, std::string_view value) {
  return static_cast<std::size_t>(std::count(events.begin(), events.end(), value));
}

void probe_contracts(const std::filesystem::path& video_path,
                     const std::filesystem::path& event_path) {
  constexpr std::uint64_t timeout_ns = 60'000'000'000ULL;
  set_scenario("probe-ok");
  std::filesystem::remove(event_path);
  const auto result = probe_video(container_config(video_path), timeout_ns);
  expect_eq(result.width, 3840U, "parser-visible width");
  expect_eq(result.height, 2160U, "parser-visible height");
  expect_eq(result.fps_numerator, 30'000U, "parser FPS numerator");
  expect_eq(result.fps_denominator, 1'001U, "parser FPS denominator");
  expect_true(std::abs(result.fps - 30'000.0 / 1'001.0) < 1e-12, "rational FPS");
  expect_eq(result.duration_ns, 10'000'000'000ULL, "queried duration");
  expect_eq(result.total_frames, 300ULL, "EOS scan preserves the exact access-unit count");
  expect_true(result.first_stream_time_ns == 0,
              "zero-origin stream preserves its first presentation stream time");
  expect_true(!result.duration_is_estimated, "known duration is not estimated");
  expect_true(!result.total_frames_is_estimated, "EOS-proven frame count is exact");
  expect_true(result.indexed_sampling_cadence_verified,
              "dense full-stream timing is safe for indexed calibration sampling");
  expect_true(result.selected_stream_caps_verified,
              "EOS scan verifies codec, geometry, and timing caps for every access unit");

  const auto events = read_events(event_path);
  expect_true(has_event(events, "parse-probe"), "probe constructs parser-only pipeline");
  expect_true(has_event(events, "probe-codec-filter"),
              "container probe filters out attached images and unsupported codecs");
  expect_true(has_event(events, "probe-parsebin"),
              "container probe uses the production parser selection topology");
  expect_true(has_event(events, "probe-container-info"),
              "container probe retains the pre-parser rational metadata caps");
  expect_true(has_event(events, "probe-decoder-caps"),
              "probe constrains parser output to the NVDEC sink contract");
  expect_true(has_event(events, "state-playing"),
              "probe runs only the backpressured compressed parser branch");
  expect_true(has_event(events, "state-null"), "probe resets pipeline before release");
  expect_true(!has_event(events, "decoder-element"), "probe does not construct a decoder");
  expect_true(!has_event(events, "pull"), "probe does not pull a decoded frame");
  expect_true(has_event(events, "pull-probe"),
              "probe seeks only compressed access-unit timestamps");
  expect_true(has_event(events, "sample-caps"),
              "probe reads caps from the retained compressed sample");
  expect_true(has_event(events, "pad-current-caps"),
              "probe reads negotiated pre-parser caps after compressed-sample preroll");
  expect_true(!has_event(events, "map"), "probe does not map frame memory");
  expect_true(!has_event(events, "raw-video-caps"), "probe never negotiates raw video caps");

  set_scenario("probe-duration-unknown");
  const auto unknown_duration = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_duration.duration_ns, 200'000'000'000ULL,
            "unknown duration covers the EOS-proven compressed AU count");
  expect_eq(unknown_duration.total_frames, 5'995ULL,
            "unknown container duration retains the full-stream exact count");
  expect_true(!unknown_duration.duration_is_estimated,
              "EOS timing supersedes missing container duration metadata");
  expect_true(!unknown_duration.total_frames_is_estimated,
              "dense EOS-proven frame count is exact despite unknown duration metadata");

  set_scenario("probe-duration-zero");
  expect_true(!probe_video(container_config(video_path), timeout_ns).duration_is_estimated,
              "EOS timing supersedes a zero container duration");

  set_scenario("probe-exact-frame-count");
  expect_eq(probe_video(container_config(video_path), timeout_ns).total_frames, 3ULL,
            "exact rational arithmetic avoids floating-point off-by-one");

  set_scenario("probe-integral-frame-count");
  expect_eq(probe_video(container_config(video_path), timeout_ns).total_frames, 30'000ULL,
            "integral rational frame count is not rounded down");

  set_scenario("probe-frame-count-overflow");
  const auto enormous_estimate = probe_video(container_config(video_path), timeout_ns);
  expect_true(enormous_estimate.total_frames_is_estimated,
              "impractically large streams retain explicitly estimated frame counts");
  expect_true(!enormous_estimate.indexed_sampling_cadence_verified,
              "impractically large streams do not claim an exhaustive cadence proof");

  set_scenario("probe-duration-mismatch");
  const auto selected_duration = probe_video(container_config(video_path), timeout_ns);
  expect_true(selected_duration.duration_ns > 900'000'000ULL &&
                  selected_duration.duration_ns < 1'100'000'000ULL,
              "selected stream duration excludes longer unrelated tracks");
  expect_eq(selected_duration.total_frames, 30ULL,
            "selected stream frame count excludes longer unrelated tracks");
  expect_true(!selected_duration.duration_is_estimated,
              "timestamp-correlated selected duration is not estimated");

  set_scenario("probe-long-video-shorter-than-container");
  const auto long_selected_stream = probe_video(container_config(video_path), timeout_ns);
  expect_true(long_selected_stream.duration_ns > 199'000'000'000ULL &&
                  long_selected_stream.duration_ns < 201'000'000'000ULL,
              "timing windows use the selected video duration instead of a longer container track");
  expect_eq(long_selected_stream.total_frames, 5'995ULL,
            "long selected-stream correlation excludes trailing unrelated tracks");

  set_scenario("probe-container-duration-underestimate");
  const auto underestimated_container = probe_video(container_config(video_path), timeout_ns);
  expect_true(underestimated_container.duration_ns > 199'000'000'000ULL &&
                  underestimated_container.duration_ns < 201'000'000'000ULL,
              "selected-stream search expands beyond an underestimated container duration");
  expect_eq(underestimated_container.total_frames, 6'000ULL,
            "expanded selected-stream search recovers frames past the container bound");

  set_scenario("probe-delayed-stream");
  const auto delayed_stream = probe_video(container_config(video_path), timeout_ns);
  expect_eq(delayed_stream.duration_ns, 1'000'000'000ULL,
            "selected stream duration excludes its delayed container start");
  expect_eq(delayed_stream.total_frames, 30ULL,
            "delayed selected stream reports only its own frame count");
  expect_true(!delayed_stream.duration_is_estimated,
              "delayed selected stream duration remains timestamp-correlated");

  set_scenario("probe-nonzero-origin");
  const auto nonzero_origin = probe_video(container_config(video_path), timeout_ns);
  expect_true(nonzero_origin.first_stream_time_ns == 766'666'666ULL,
              "nonzero presentation stream origin is preserved through worker IPC");
  expect_eq(nonzero_origin.duration_ns, 2'000'000'000ULL,
            "nonzero timeline origin does not shorten a duration span");
  expect_eq(nonzero_origin.total_frames, 60ULL,
            "short demux duration still searches the final access unit");

  set_scenario("probe-decode-order-origin");
  const auto decode_order_origin = probe_video(container_config(video_path), timeout_ns);
  expect_true(decode_order_origin.first_stream_time_ns == 0,
              "presentation-order minimum defines the indexed stream origin");
  expect_eq(decode_order_origin.duration_ns, 1'000'000'000ULL,
            "first decode-order access unit does not define the timeline origin");
  expect_eq(decode_order_origin.total_frames, 30ULL,
            "decode-ordered first PTS does not drop presentation frames");

  set_scenario("probe-caps-runahead");
  const auto sample_caps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(sample_caps.width, 854U, "metadata width remains correlated with the selected sample");
  expect_eq(sample_caps.height, 480U,
            "metadata height remains correlated with the selected sample");
  expect_eq(sample_caps.fps_numerator, 24U,
            "metadata frame rate remains correlated with the selected sample");

  set_scenario("probe-dynamic-resolution");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "changed codec, geometry, or timing caps",
                     "midstream parser geometry changes are rejected");

  set_scenario("probe-unknown-pts");
  const auto unknown_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_pts.duration_ns, 1'000'000'000ULL,
            "unknown-PTS short stream uses frame-count duration");
  expect_eq(unknown_pts.total_frames, 30ULL, "unknown-PTS access units remain valid frames");
  expect_true(unknown_pts.duration_is_estimated,
              "unknown-PTS frame-count duration is explicitly estimated");

  set_scenario("probe-one-frame-rounding");
  const auto one_frame = probe_video(container_config(video_path), timeout_ns);
  expect_eq(one_frame.total_frames, 1ULL, "nanosecond rounding cannot erase a proven frame");

  set_scenario("probe-inexact-caps-fps");
  const auto inferred_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(inferred_fps.fps_numerator, 30U,
            "constant presentation timing corrects inexact container caps");
  expect_eq(inferred_fps.fps_denominator, 1U, "inferred constant frame rate is reduced exactly");
  expect_eq(inferred_fps.total_frames, 60ULL, "inexact container caps do not lose a proven frame");

  set_scenario("probe-cadence-proof-rate-mismatch");
  const auto corrected_drifting_caps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(corrected_drifting_caps.fps_numerator, 30U,
            "full-stream phase validation replaces parser caps with cumulative drift");
  expect_eq(corrected_drifting_caps.fps_denominator, 1U,
            "the phase-verified inferred rate remains exact");
  expect_true(corrected_drifting_caps.indexed_sampling_cadence_verified,
              "the returned corrected rate carries its full-stream cadence proof");

  set_scenario("probe-short-quantized-exact-30");
  const auto short_exact_30 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_exact_30.fps_numerator, 30U,
            "short quantized timing preserves an exact caps numerator");
  expect_eq(short_exact_30.fps_denominator, 1U,
            "short quantized timing preserves an exact caps denominator");
  expect_eq(short_exact_30.total_frames, 3ULL, "short quantized timing retains the exact AU count");
  expect_true(short_exact_30.indexed_sampling_cadence_verified,
              "short quantized timing can certify the retained caps rate");

  set_scenario("probe-short-quantized-exact-5997");
  const auto short_exact_5997 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_exact_5997.fps_numerator, 5'997U,
            "short quantized near-canonical timing preserves the exact caps numerator");
  expect_eq(short_exact_5997.fps_denominator, 100U,
            "short quantized near-canonical timing preserves the exact caps denominator");
  expect_eq(short_exact_5997.total_frames, 3ULL,
            "short near-canonical timing retains the exact AU count");
  expect_true(short_exact_5997.indexed_sampling_cadence_verified,
              "short near-canonical timing can certify the retained caps rate");

  set_scenario("probe-long-unknown-pts");
  const auto long_unknown_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_unknown_pts.duration_ns, 3'000'000'000ULL,
            "long unknown-PTS stream uses its exact access-unit count");
  expect_eq(long_unknown_pts.total_frames, 90ULL,
            "unknown-PTS stream scanning continues beyond the timing window");
  expect_true(long_unknown_pts.duration_is_estimated,
              "long unknown-PTS duration remains explicitly estimated");
  expect_true(!long_unknown_pts.total_frames_is_estimated,
              "unknown-PTS EOS scan still proves the AU count");

  set_scenario("probe-mixed-prefix-pts");
  const auto mixed_prefix = probe_video(container_config(video_path), timeout_ns);
  expect_eq(mixed_prefix.duration_ns, 4'000'000'000ULL,
            "untimed prefix retains its nominal frame interval");
  expect_eq(mixed_prefix.total_frames, 120ULL, "untimed prefix does not shift the stream origin");
  expect_true(mixed_prefix.duration_is_estimated,
              "duration with an untimed prefix is explicitly estimated");

  set_scenario("probe-long-mixed-prefix-pts");
  const auto long_mixed_prefix = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_mixed_prefix.duration_ns, 20'000'000'000ULL,
            "long untimed prefix is included in correlated duration");
  expect_eq(long_mixed_prefix.total_frames, 600ULL,
            "long untimed prefix is included in correlated frame count");
  expect_true(long_mixed_prefix.first_stream_time_ns == 0,
              "long untimed prefix exports the inferred frame-zero origin");
  expect_true(long_mixed_prefix.duration_is_estimated,
              "long untimed-prefix duration remains explicitly estimated");
  expect_true(long_mixed_prefix.total_frames_is_estimated,
              "bounded long untimed-prefix count remains explicitly estimated");
  expect_true(!long_mixed_prefix.indexed_sampling_cadence_verified,
              "incomplete PTS evidence is not exposed as an indexed-sampling cadence proof");
  expect_true(!long_mixed_prefix.selected_stream_caps_verified,
              "bounded untimed timing does not claim full-stream caps stability");

  set_scenario("probe-reordered-untimed-prefix");
  const auto reordered_untimed_prefix = probe_video(container_config(video_path), timeout_ns);
  expect_eq(reordered_untimed_prefix.duration_ns, 20'000'000'000ULL,
            "reordered untimed prefix is counted once in correlated duration");
  expect_eq(reordered_untimed_prefix.total_frames, 600ULL,
            "reordered untimed prefix is counted once in correlated frame count");
  expect_true(reordered_untimed_prefix.first_stream_time_ns == 4'000'000'000ULL,
              "reordered untimed prefix exports its inferred frame-zero origin");

  set_scenario("probe-unset-fps-inferred");
  const auto unset_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unset_fps.fps_numerator, 30U, "unset caps frame rate is inferred from timestamps");
  expect_eq(unset_fps.fps_denominator, 1U, "inferred unset-caps frame rate is exact");
  expect_eq(unset_fps.total_frames, 120ULL, "unset caps frame rate remains seek-countable");

  const auto untimed_elementary =
      probe_video(elementary_config(video_path, GpuDecodeCodec::H264), timeout_ns);
  expect_eq(untimed_elementary.fps_numerator, 30U,
            "timing-less elementary stream uses the explicit frame-rate fallback");
  expect_eq(untimed_elementary.total_frames, 120ULL,
            "timing-less elementary stream is counted through EOS");
  expect_true(untimed_elementary.duration_is_estimated,
              "timing-less elementary duration remains explicitly estimated");

  set_scenario("probe-long-untimed-elementary");
  const auto long_untimed_elementary =
      probe_video(elementary_config(video_path, GpuDecodeCodec::H264), timeout_ns);
  expect_true(long_untimed_elementary.total_frames >= 513ULL,
              "long elementary estimate preserves the observed AU lower bound");
  expect_true(long_untimed_elementary.total_frames_is_estimated,
              "long elementary AU count is explicitly estimated");
  expect_true(long_untimed_elementary.duration_is_estimated,
              "untimed elementary duration query is not treated as authoritative");
  expect_true(!long_untimed_elementary.selected_stream_caps_verified,
              "bounded elementary probing leaves full-stream caps stability unverified");

  set_scenario("probe-vfr-unset-fps");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "dense nonconstant timing is rejected when caps do not provide a constant rate");

  set_scenario("probe-short-vfr");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "short EOS-complete nonconstant timing cannot retain plausible caps");

  set_scenario("probe-vfr-late-transition");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "dense nonconstant timing cannot silently retain plausible average caps");

  set_scenario("probe-retimed-constant-pts");
  const auto retimed_constant = probe_video(container_config(video_path), timeout_ns);
  expect_eq(retimed_constant.fps_numerator, 30U,
            "whole-stream constant PTS timing overrides stale plausible caps");
  expect_eq(retimed_constant.total_frames, 150ULL,
            "retimed constant stream retains its EOS-proven AU count");

  set_scenario("probe-eos-vui-duration-mismatch");
  const auto corrected_duration = probe_video(container_config(video_path), timeout_ns);
  expect_eq(corrected_duration.fps_numerator, 30U,
            "EOS timestamp cadence replaces contradictory VUI caps");
  expect_eq(corrected_duration.duration_ns, 5'000'000'000ULL,
            "corrected cadence does not trust contradictory parser buffer durations");
  expect_eq(corrected_duration.total_frames, 150ULL,
            "corrected EOS duration retains the exact AU count");
  expect_true(corrected_duration.duration_is_estimated,
              "timestamp-derived corrected duration is explicitly estimated");

  set_scenario("probe-duplicate-pts-pairs");
  const auto duplicate_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_pts.fps_numerator, 50U,
            "duplicate PTS pairs cannot halve a plausible caps frame rate");
  expect_eq(duplicate_pts.total_frames, 200ULL,
            "duplicate PTS pairs retain their EOS-proven AU count");
  expect_eq(duplicate_pts.duration_ns, 4'000'000'000ULL,
            "duplicate PTS pairs retain their observed terminal span");
  expect_eq(duplicate_pts.timestamp_multiplicity, 2U,
            "duplicate PTS pair multiplicity remains explicit");

  set_scenario("probe-short-duplicate-pts-pairs");
  const auto short_duplicate_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_duplicate_pts.total_frames, 6ULL,
            "short duplicate-PTS calibration clip retains every physical frame");
  expect_eq(short_duplicate_pts.timestamp_multiplicity, 2U,
            "short duplicate-PTS calibration clip proves pair multiplicity");
  expect_true(short_duplicate_pts.indexed_sampling_cadence_verified,
              "short duplicate-PTS calibration clip is safe for indexed sampling");
  auto short_duplicate_config = container_config(video_path);
  short_duplicate_config.indexed_fps_numerator = short_duplicate_pts.fps_numerator;
  short_duplicate_config.indexed_fps_denominator = short_duplicate_pts.fps_denominator;
  short_duplicate_config.indexed_timestamp_multiplicity =
      short_duplicate_pts.timestamp_multiplicity;
  short_duplicate_config.indexed_stream_time_origin_ns = short_duplicate_pts.first_stream_time_ns;
  short_duplicate_config.start_frame_index = 0U;
  auto short_duplicate_source =
      open_gstreamer_gpu_file_decode_source(short_duplicate_config, NvbufSurfaceAbi::DeepStream9_1);
  const auto short_duplicate_frame_zero = short_duplicate_source->read();
  expect_true(short_duplicate_frame_zero.frame.has_value(),
              "short duplicate-PTS calibration clip decodes frame zero");
  if (short_duplicate_frame_zero.frame.has_value()) {
    expect_eq(short_duplicate_frame_zero.frame->frame_index, 0ULL,
              "short duplicate-PTS calibration clip retains frame-zero index");
    expect_eq(short_duplicate_frame_zero.frame->nvmm.cuda_base_ptr, 0x10000000ULL,
              "frame-zero seek returns the first physical duplicate-PTS frame");
  }
  short_duplicate_source->seek_to_frame(0U);
  const auto repeated_short_duplicate_frame_zero = short_duplicate_source->read();
  expect_true(repeated_short_duplicate_frame_zero.frame.has_value(),
              "repeated duplicate-PTS frame-zero seek decodes a frame");
  if (repeated_short_duplicate_frame_zero.frame.has_value()) {
    expect_eq(repeated_short_duplicate_frame_zero.frame->nvmm.cuda_base_ptr, 0x10000000ULL,
              "repeated frame-zero seek rewinds to the first physical duplicate-PTS frame");
  }

  set_scenario("probe-duplicate-pts-pairs");
  auto duplicate_decode_config = container_config(video_path);
  duplicate_decode_config.indexed_fps_numerator = duplicate_pts.fps_numerator;
  duplicate_decode_config.indexed_fps_denominator = duplicate_pts.fps_denominator;
  duplicate_decode_config.indexed_timestamp_multiplicity = duplicate_pts.timestamp_multiplicity;
  duplicate_decode_config.indexed_stream_time_origin_ns = duplicate_pts.first_stream_time_ns;
  duplicate_decode_config.start_frame_index = 101U;
  auto duplicate_source = open_gstreamer_gpu_file_decode_source(duplicate_decode_config,
                                                                NvbufSurfaceAbi::DeepStream9_1);
  const auto group_start = duplicate_source->read();
  const auto odd_member = duplicate_source->read();
  expect_true(group_start.frame.has_value() && odd_member.frame.has_value(),
              "probe-configured duplicate PTS seek returns the target group");
  if (group_start.frame.has_value() && odd_member.frame.has_value()) {
    expect_eq(group_start.frame->frame_index, 100U,
              "odd indexed seek begins at its duplicate group base");
    expect_eq(odd_member.frame->frame_index, 101U,
              "duplicate group ordinal identifies the odd target frame");
  }
  duplicate_source->seek_to_frame(151U);
  const auto reused_group_start = duplicate_source->read();
  const auto reused_odd_member = duplicate_source->read();
  expect_true(reused_group_start.frame.has_value() && reused_odd_member.frame.has_value(),
              "one decoder supports a second duplicate PTS seek");
  if (reused_group_start.frame.has_value() && reused_odd_member.frame.has_value()) {
    expect_eq(reused_group_start.frame->frame_index, 150U,
              "reused seek resets duplicate group indexing");
    expect_eq(reused_odd_member.frame->frame_index, 151U,
              "reused seek preserves duplicate group ordinal");
  }

  set_scenario("probe-triplicate-pts");
  const auto triplicate_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(triplicate_pts.fps_numerator, 75U,
            "triplicate PTS groups retain their access-unit rate");
  expect_eq(triplicate_pts.total_frames, 300ULL,
            "triplicate PTS groups retain their exact access-unit count");
  expect_eq(triplicate_pts.duration_ns, 4'000'000'000ULL,
            "verified cadence closes the full triplicate terminal group");
  expect_true(!triplicate_pts.duration_is_estimated,
              "verified cadence and exact count provide an exact triplicate duration");
  expect_eq(triplicate_pts.timestamp_multiplicity, 3U,
            "triplicate PTS multiplicity remains explicit");

  set_scenario("probe-short-triplicate-pts");
  const auto short_triplicate_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_triplicate_pts.total_frames, 9ULL,
            "short triplicate-PTS calibration clip retains every physical frame");
  expect_eq(short_triplicate_pts.timestamp_multiplicity, 3U,
            "short triplicate-PTS calibration clip proves triplicate multiplicity");
  expect_true(short_triplicate_pts.indexed_sampling_cadence_verified,
              "short triplicate-PTS calibration clip is safe for indexed sampling");
  auto short_triplicate_config = container_config(video_path);
  short_triplicate_config.indexed_fps_numerator = short_triplicate_pts.fps_numerator;
  short_triplicate_config.indexed_fps_denominator = short_triplicate_pts.fps_denominator;
  short_triplicate_config.indexed_timestamp_multiplicity =
      short_triplicate_pts.timestamp_multiplicity;
  short_triplicate_config.indexed_stream_time_origin_ns = short_triplicate_pts.first_stream_time_ns;
  short_triplicate_config.start_frame_index = 0U;
  auto short_triplicate_source = open_gstreamer_gpu_file_decode_source(
      short_triplicate_config, NvbufSurfaceAbi::DeepStream9_1);
  const auto short_triplicate_frame_zero = short_triplicate_source->read();
  expect_true(short_triplicate_frame_zero.frame.has_value(),
              "short triplicate-PTS calibration clip decodes frame zero");
  if (short_triplicate_frame_zero.frame.has_value()) {
    expect_eq(short_triplicate_frame_zero.frame->frame_index, 0ULL,
              "short triplicate-PTS calibration clip retains frame-zero index");
    expect_eq(short_triplicate_frame_zero.frame->nvmm.cuda_base_ptr, 0x10000000ULL,
              "frame-zero seek returns the first physical triplicate-PTS frame");
  }
  short_triplicate_source->seek_to_frame(0U);
  const auto repeated_short_triplicate_frame_zero = short_triplicate_source->read();
  expect_true(repeated_short_triplicate_frame_zero.frame.has_value(),
              "repeated triplicate-PTS frame-zero seek decodes a frame");
  if (repeated_short_triplicate_frame_zero.frame.has_value()) {
    expect_eq(repeated_short_triplicate_frame_zero.frame->nvmm.cuda_base_ptr, 0x10000000ULL,
              "repeated frame-zero seek rewinds to the first physical triplicate-PTS frame");
  }

  set_scenario("probe-long-duplicate-pts-pairs");
  const auto long_duplicate_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_duplicate_pts.fps_numerator, 50U,
            "long duplicate PTS pairs retain their AU frame rate");
  expect_eq(long_duplicate_pts.duration_ns, 12'000'000'000ULL,
            "terminal duplicate PTS group retains its complete presentation span");
  expect_eq(long_duplicate_pts.total_frames, 600ULL,
            "bounded terminal correlation counts every duplicate-PTS AU");

  set_scenario("probe-duplicate-clustered-missing-groups");
  const auto duplicate_clustered_missing = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_clustered_missing.fps_numerator, 50U,
            "periodically untimed duplicate groups preserve the AU rate");
  expect_eq(duplicate_clustered_missing.fps_denominator, 1U,
            "periodically untimed duplicate groups preserve the rate denominator");
  expect_eq(duplicate_clustered_missing.total_frames, 200ULL,
            "periodically untimed duplicate groups retain their EOS-proven AU count");
  expect_eq(duplicate_clustered_missing.duration_ns, 4'000'000'000ULL,
            "periodically untimed duplicate groups retain caps-rate duration");

  set_scenario("probe-duplicate-pts-transition");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "dense duplicate PTS multiplicity transitions are rejected unconditionally");

  set_scenario("probe-duplicate-pts-reorder-cutoff");
  const auto duplicate_reorder_cutoff = probe_video(container_config(video_path), timeout_ns);
  expect_eq(duplicate_reorder_cutoff.fps_numerator, 50U,
            "reordered cutoff group does not invalidate uniform duplicate PTS timing");
  expect_eq(duplicate_reorder_cutoff.fps_denominator, 1U,
            "reordered duplicate PTS timing retains its inferred denominator");
  expect_eq(duplicate_reorder_cutoff.total_frames, 1'000ULL,
            "reordered duplicate PTS timing extends to an EOS-proven AU count");
  expect_true(!duplicate_reorder_cutoff.total_frames_is_estimated,
              "reordered duplicate PTS count is exact after reaching EOS");

  set_scenario("probe-duplicate-pts-transition-untimed-tail");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "an untimed tail cannot excuse a dense duplicate PTS cadence transition");

  set_scenario("probe-duplicate-pts-larger-reorder-suffix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "larger duplicate groups in the bounded suffix are rejected as a cadence transition");

  set_scenario("probe-paired-au-missing-pts");
  const auto paired_missing_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(paired_missing_pts.fps_numerator, 50U,
            "uniformly missing paired-AU PTS values preserve the AU rate");
  expect_eq(paired_missing_pts.total_frames, 200ULL,
            "uniformly missing paired-AU PTS values retain the exact AU count");

  set_scenario("probe-reordered-periodic-missing-pts");
  const auto reordered_periodic_missing = probe_video(container_config(video_path), timeout_ns);
  expect_eq(reordered_periodic_missing.fps_numerator, 30U,
            "periodic missing PTS survives B-frame presentation reordering");
  expect_eq(reordered_periodic_missing.fps_denominator, 1U,
            "reordered sparse timing retains the inferred denominator");
  expect_eq(reordered_periodic_missing.total_frames, 200ULL,
            "reordered sparse timing retains the EOS-proven AU count");
  expect_true(reordered_periodic_missing.duration_ns >= 6'666'666'666ULL &&
                  reordered_periodic_missing.duration_ns <= 6'666'666'667ULL,
              "reordered sparse timing derives duration from the corrected frame rate");

  set_scenario("probe-long-reordered-periodic-missing-pts");
  const auto long_reordered_periodic_missing =
      probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_reordered_periodic_missing.fps_numerator, 30U,
            "bounded reordered sparse timing corrects misleading caps");
  expect_eq(long_reordered_periodic_missing.total_frames, 600ULL,
            "conflicting reordered sparse timing extends to an exact AU count");
  expect_true(!long_reordered_periodic_missing.total_frames_is_estimated,
              "adaptive conflicting-metadata scan reaches EOS when bounded");

  set_scenario("probe-bounded-caps-underestimate");
  const auto bounded_caps_underestimate = probe_video(container_config(video_path), timeout_ns);
  expect_eq(bounded_caps_underestimate.fps_numerator, 30U,
            "bounded timing overrides caps that cannot cover observed access units");
  expect_eq(bounded_caps_underestimate.total_frames, 600ULL,
            "bounded timing and container duration recover the complete frame estimate");

  set_scenario("probe-container-rate-over-vui");
  const auto container_rate = probe_video(container_config(video_path), timeout_ns);
  expect_eq(container_rate.fps_numerator, 15U,
            "constant container timestamps override nearby bitstream VUI caps");
  expect_eq(container_rate.fps_denominator, 1U,
            "container timing correction remains an exact rational");
  expect_eq(container_rate.total_frames, 600ULL,
            "container frame rate preserves the 600-frame duration count");

  set_scenario("probe-bounded-stale-caps");
  const auto stale_caps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(stale_caps.fps_numerator, 15U,
            "bounded constant timing overrides substantially stale caps");
  expect_eq(stale_caps.fps_denominator, 1U, "substantial stale-caps correction remains exact");
  expect_eq(stale_caps.total_frames, 600ULL,
            "stale high-rate caps cannot double the calibration frame count");

  set_scenario("probe-sparse-exact-30-gaps");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "fully timestamped doubled intervals are not reinterpreted as sparse timing");

  set_scenario("probe-clustered-missing-pts");
  const auto clustered_missing_pts = probe_video(container_config(video_path), timeout_ns);
  expect_eq(clustered_missing_pts.fps_numerator, 30U,
            "clustered missing PTS values do not imply uniform AU multiplicity");
  expect_eq(clustered_missing_pts.fps_denominator, 1U,
            "clustered missing PTS values preserve the caps frame-rate denominator");
  expect_eq(clustered_missing_pts.total_frames, 120ULL,
            "clustered missing PTS values retain the exact AU count");
  expect_eq(clustered_missing_pts.duration_ns, 4'000'000'000ULL,
            "clustered missing PTS values retain the caps-rate duration");

  set_scenario("probe-exact-5997-fps");
  const auto exact_5997_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(exact_5997_fps.fps_numerator, 5'997U,
            "exact near-canonical caps are not snapped to an NTSC numerator");
  expect_eq(exact_5997_fps.fps_denominator, 100U,
            "exact near-canonical caps preserve their denominator");

  set_scenario("probe-container-exact-5997-parser-ntsc");
  auto exact_matroska_config = container_config(video_path);
  exact_matroska_config.container = GpuDecodeContainer::Matroska;
  const auto container_exact_5997 = probe_video(exact_matroska_config, timeout_ns);
  expect_eq(container_exact_5997.fps_numerator, 5'997U,
            "exact container cadence overrides normalized parser caps");
  expect_eq(container_exact_5997.fps_denominator, 100U,
            "exact container cadence preserves its noncanonical denominator");
  expect_eq(container_exact_5997.total_frames, 600ULL,
            "near-NTSC conflict extension retains the exact AU count");

  set_scenario("probe-matroska-duration-transition");
  expect_probe_error([&] { (void)probe_video(exact_matroska_config, timeout_ns); },
                     "compressed-buffer durations disagree",
                     "Matroska DefaultDuration changes cannot retain an exact indexed rate");

  set_scenario("probe-quantized-no-vui-5994");
  const auto quantized_no_vui = probe_video(container_config(video_path), timeout_ns);
  expect_eq(quantized_no_vui.fps_numerator, 60'000U,
            "90 kHz quantized timing snaps to the standard NTSC numerator");
  expect_eq(quantized_no_vui.fps_denominator, 1'001U,
            "90 kHz quantized timing snaps to the standard NTSC denominator");
  expect_eq(quantized_no_vui.total_frames, 240ULL,
            "quantized no-VUI stream retains its EOS-proven AU count");

  set_scenario("probe-short-unset-fps-15");
  const auto short_unset_15 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(short_unset_15.fps_numerator, 15U,
            "short EOS-proven timing infers an exact noncanonical frame rate");
  expect_eq(short_unset_15.fps_denominator, 1U,
            "short EOS-proven noncanonical timing retains its denominator");
  expect_eq(short_unset_15.total_frames, 15ULL,
            "short noncanonical inference retains its EOS-proven AU count");

  set_scenario("probe-long-unset-fps-15");
  const auto long_unset_15 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(long_unset_15.fps_numerator, 15U,
            "long unset-caps stream extends to EOS before selecting a noncanonical rate");
  expect_eq(long_unset_15.fps_denominator, 1U,
            "long unset-caps stream preserves its exact denominator");
  expect_eq(long_unset_15.total_frames, 600ULL,
            "long unset-caps stream retains its EOS-proven AU count");

  set_scenario("probe-durationless-unseekable-15");
  const auto bounded_unset_15 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(bounded_unset_15.fps_numerator, 15U,
            "bounded constant timing selects a noncanonical rate without duration or EOS");
  expect_eq(bounded_unset_15.fps_denominator, 1U,
            "bounded durationless inference preserves its exact denominator");
  expect_eq(bounded_unset_15.total_frames, 9'000ULL,
            "durationless inference retains the EOS-proven compressed-AU count");
  expect_eq(bounded_unset_15.duration_ns, 600'000'000'000ULL,
            "durationless inference covers its exact compressed-AU count");
  expect_true(!bounded_unset_15.total_frames_is_estimated,
              "durationless dense full-stream frame count is exact");

  std::filesystem::remove(event_path);
  set_scenario("probe-million-au-no-duration");
  const auto bounded_million = probe_video(container_config(video_path), timeout_ns);
  expect_true(bounded_million.total_frames_is_estimated,
              "million-AU durationless probe keeps its bounded count explicitly estimated");
  expect_true(!bounded_million.selected_stream_caps_verified,
              "bounded long probe does not claim complete selected-stream caps coverage");
  expect_true(!bounded_million.indexed_sampling_cadence_verified,
              "bounded long probe does not claim exhaustive cadence coverage");
  const auto bounded_events = read_events(event_path);
  expect_eq(count_event(bounded_events, "pull-probe"), 10'000U,
            "durationless long probe reads exactly the eager 10,000-AU ceiling");
  expect_eq(count_event(bounded_events, "seek-compressed"), 0U,
            "durationless long probe does not add random-access work");

  std::filesystem::remove(event_path);
  set_scenario("probe-long-gop-seek-budget");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "frame-seek work limit",
                     "long-GOP seek preroll fails closed at its compressed-AU budget");
  expect_true(count_event(read_events(event_path), "pull-probe") <= 5'600U,
              "long-GOP duration correlation cannot scan the recording from its first keyframe");

  set_scenario("probe-late-vfr-after-bounded-prefix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "terminal timing rejects a VFR transition after the bounded sequential prefix");

  set_scenario("probe-vfr-in-final-window");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "mixed cadence inside the terminal timing window is rejected");

  set_scenario("probe-low-amplitude-vfr");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "a 30-to-31 fps EOS transition fails cadence phase validation");

  set_scenario("probe-interior-vfr-recovery");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "a middle 30-to-15-to-30 cadence change is rejected by interior sampling");

  set_scenario("probe-vfr-between-windows");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "bounded sequential cadence validation catches the interior transition");

  set_scenario("probe-interior-seek-gap");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "samples beyond an interior window cannot stand in for its missing timing evidence");

  set_scenario("probe-prefix-vfr-tail-cfr");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "a constant terminal cadence cannot excuse a dense nonconstant prefix");

  set_scenario("probe-gap-before-reorder-suffix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "a dense gap immediately before the reorder suffix remains part of prefix analysis");

  set_scenario("probe-descending-pts");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "reorder depth",
      "descending PTS cannot grow unresolved cadence state beyond the reorder bound");

  set_scenario("probe-oldest-pts-refresh");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "reorder depth",
                     "refreshing the oldest pending PTS cannot grow cadence state without bound");

  std::filesystem::remove(event_path);
  set_scenario("probe-terminal-60-eos");
  const auto terminal_60 = probe_video(container_config(video_path), timeout_ns);
  expect_eq(terminal_60.fps_numerator, 60U,
            "60 fps terminal timing retains the exact caps numerator");
  expect_eq(terminal_60.total_frames, 1'200ULL,
            "60 fps terminal timing retains the correlated frame count");
  expect_true(has_event(read_events(event_path), "terminal-window-eos"),
              "the five-second 60 fps terminal window drains through EOS");

  set_scenario("probe-terminal-transition-after-256");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "a terminal cadence transition after 256 analyzed samples is detected before EOS");

  set_scenario("probe-terminal-duplicate-pts-transition");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "EOS-complete terminal timing rejects smaller duplicate-PTS groups in the final 32 AUs");

  set_scenario("probe-vfr-missing-durations");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "missing buffer durations do not hide dense nonconstant PTS timing");

  set_scenario("probe-dropped-frame-after-prefix");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); }, "variable frame-rate",
      "a fully timestamped missing access-unit interval is rejected as nonconstant cadence");

  set_scenario("probe-reduced-cadence-after-prefix");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "variable frame-rate",
                     "reduced cadence after the prefix is rejected as variable frame rate");

  set_scenario("probe-bframe-cutoff");
  const auto reordered_cutoff = probe_video(container_config(video_path), timeout_ns);
  expect_eq(reordered_cutoff.fps_numerator, 30U,
            "timing lookahead closes a B-frame group at the analysis boundary");
  expect_eq(reordered_cutoff.fps_denominator, 1U,
            "B-frame timing inference keeps the reduced rational rate");
  expect_eq(reordered_cutoff.total_frames, 120ULL,
            "B-frame cutoff does not lose the delayed presentation frame");

  set_scenario("probe-quantized-timestamps");
  const auto quantized = probe_video(container_config(video_path), timeout_ns);
  expect_eq(quantized.total_frames, 100ULL,
            "seek-proven count survives coarse timestamp quantization");
  expect_eq(quantized.duration_ns, 4'170'833'334ULL,
            "quantized duration is clamped to the proven frame-count boundary");
  expect_true(!quantized.duration_is_estimated,
              "EOS-correlated quantized duration remains authoritative");
  expect_true(!quantized.total_frames_is_estimated,
              "quantized EOS scan proves its compressed AU count");

  set_scenario("probe-bad-fps");
  const auto invalid_caps_fps = probe_video(container_config(video_path), timeout_ns);
  expect_eq(invalid_caps_fps.fps_numerator, 30U,
            "invalid caps rate is replaced by constant parser timing");
  expect_eq(invalid_caps_fps.fps_denominator, 1U,
            "invalid caps timing inference is reduced exactly");

  set_scenario("probe-estimated-count-lower-bound");
  const auto lower_bound = probe_video(container_config(video_path), timeout_ns);
  expect_eq(lower_bound.total_frames, 6'000ULL,
            "expanded selected-stream search agrees with the complete sequential count");
  expect_true(!lower_bound.total_frames_is_estimated,
              "full-stream cadence scan proves the recovered compressed-AU count");

  set_scenario("probe-seek-unsupported");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); },
      "could not verify constant frame timing",
      "an unseekable long container fails closed when terminal timing cannot be verified");

  set_scenario("probe-seek-preroll");
  const auto seek_preroll = probe_video(container_config(video_path), timeout_ns);
  expect_eq(seek_preroll.total_frames, 5'995ULL,
            "seek preroll outside the active segment is ignored");
  expect_true(seek_preroll.total_frames_is_estimated,
              "bounded seek-preroll frame count remains explicitly estimated");
  expect_true(seek_preroll.duration_is_estimated,
              "bounded terminal seek duration remains explicitly estimated");

  set_scenario("probe-seek-unknown-pts-preroll");
  const auto unknown_pts_preroll = probe_video(container_config(video_path), timeout_ns);
  expect_eq(unknown_pts_preroll.total_frames, 5'995ULL,
            "PTS-less seek preroll is skipped before duration correlation");
  expect_true(unknown_pts_preroll.total_frames_is_estimated,
              "bounded PTS-less seek-preroll frame count remains explicitly estimated");
  expect_true(unknown_pts_preroll.duration_is_estimated,
              "PTS-less seek-preroll duration remains explicitly estimated");

  set_scenario("probe-seek-untimestamped-tail");
  expect_probe_error(
      [&] { (void)probe_video(container_config(video_path), timeout_ns); },
      "could not verify constant frame timing",
      "an untimestamped terminal window cannot establish constant presentation timing");

  set_scenario("probe-seek-dts-reorder-tail");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "could not verify constant frame timing",
                     "DTS-only terminal evidence cannot prove constant presentation timing");

  set_scenario("probe-blocking-seek");
  const auto blocking_seek_started = std::chrono::steady_clock::now();
  expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                     "worker timed out after exceeding the configured timeout",
                     "blocking seek respects the probe deadline");
  const auto blocking_seek_elapsed = std::chrono::steady_clock::now() - blocking_seek_started;
  expect_true(blocking_seek_elapsed < std::chrono::milliseconds(1'800),
              "blocking seek returns before the synchronous runtime call completes");

  set_scenario("probe-blocking-duration-query");
  const auto blocking_query_started = std::chrono::steady_clock::now();
  expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                     "worker timed out after exceeding the configured timeout",
                     "blocking duration query respects the probe deadline");
  const auto blocking_query_elapsed = std::chrono::steady_clock::now() - blocking_query_started;
  expect_true(blocking_query_elapsed < std::chrono::milliseconds(1'800),
              "blocking duration query returns before the runtime call completes");

  set_scenario("probe-blocking-null-state");
  const auto blocking_teardown_started = std::chrono::steady_clock::now();
  expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                     "worker timed out after exceeding the configured timeout",
                     "blocking pipeline teardown respects the probe deadline");
  const auto blocking_teardown_elapsed =
      std::chrono::steady_clock::now() - blocking_teardown_started;
  expect_true(blocking_teardown_elapsed < std::chrono::milliseconds(1'800),
              "blocking pipeline teardown is reclaimed with the worker process");

  for (const auto scenario_name : {"probe-blocking-playing", "probe-blocking-pull"}) {
    set_scenario(scenario_name);
    const auto blocking_call_started = std::chrono::steady_clock::now();
    expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                       "worker timed out after exceeding the configured timeout", scenario_name);
    expect_true(std::chrono::steady_clock::now() - blocking_call_started <
                    std::chrono::milliseconds(1'800),
                std::string(scenario_name) + " is reclaimed with the worker process");
  }

  set_scenario("probe-ok");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      1'000'000'000ULL)
                .width,
            854U, "one-second timeout boundary accepted");
  expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      3'600'000'000'000ULL)
                .width,
            854U, "one-hour timeout boundary accepted");
  std::filesystem::remove(event_path);
  expect_eq(probe_video(elementary_config(video_path, GpuDecodeCodec::H264), timeout_ns).width,
            3840U, "H.264 elementary stream uses explicit parser");
  expect_true(has_event(read_events(event_path), "probe-h264-parser"),
              "H.264 elementary probe constructs only the requested parser");
  std::filesystem::remove(event_path);
  expect_eq(probe_video(elementary_config(video_path, GpuDecodeCodec::Hevc), timeout_ns).width,
            3840U, "HEVC elementary stream uses explicit parser");
  expect_true(has_event(read_events(event_path), "probe-h265-parser"),
              "HEVC elementary probe constructs only the requested parser");
}

void invalid_inputs_fail(const std::filesystem::path& video_path,
                         const std::filesystem::path& event_path) {
  constexpr std::uint64_t timeout_ns = 5'000'000'000ULL;
  expect_invalid_argument([&] { (void)probe_video({}, timeout_ns); }, "path is required",
                          "empty path rejected");
  expect_invalid_argument([&] { (void)probe_video(container_config(video_path), 999'999'999); },
                          "one second", "sub-second timeout rejected");
  expect_invalid_argument(
      [&] { (void)probe_video(container_config(video_path), 3'600'000'000'001ULL); }, "one hour",
      "over-one-hour timeout rejected");
  expect_invalid_argument(
      [&] { (void)reco::io::probe_gpu_video(container_config(video_path), {}, timeout_ns); },
      "worker path is required", "empty worker path rejected");
  expect_invalid_argument(
      [&] {
        (void)reco::io::probe_gpu_video(container_config(video_path), "relative-probe-worker",
                                        timeout_ns);
      },
      "must be absolute", "relative worker path rejected");
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(container_config(video_path),
                                        video_path.parent_path() / "missing-probe-worker",
                                        timeout_ns);
      },
      "failed to start video probe worker", "missing worker executable rejected");

  expect_probe_error(
      [&] {
        (void)probe_video(container_config(video_path.parent_path() / "missing.mp4"), timeout_ns);
      },
      "not a readable regular file", "missing input rejected before probing");

  for (const auto& [scenario_name, fragment] :
       std::array<std::pair<std::string_view, std::string_view>, 23>{
           {{"old-version", "1.10 or newer"},
            {"init-error", "fake initialization failure"},
            {"probe-parse-error", "fake parse failure"},
            {"probe-missing-info", "metadata identity"},
            {"probe-missing-pad", "metadata pad"},
            {"probe-missing-sink", "compressed-stream sink"},
            {"probe-missing-input-budget", "compressed-input budget"},
            {"probe-input-budget-missing-pad", "compressed-input budget pad"},
            {"probe-checkpoint-install-error", "access-unit budget checkpoint"},
            {"probe-input-budget-install-error", "compressed-input byte budget"},
            {"probe-missing-bus", "message bus"},
            {"probe-state-error", "playing state"},
            {"probe-stream-error", "playing state"},
            {"probe-async-error", "fake parser failure"},
            {"probe-buffered-async-error", "fake parser failure"},
            {"probe-no-supported-video", "H.264 or HEVC"},
            {"probe-missing-sample-caps", "H.264 or HEVC"},
            {"probe-missing-caps-structure", "no structure"},
            {"probe-wrong-codec-caps", "decoder-compatible"},
            {"probe-unparsed-caps", "decoder-compatible"},
            {"probe-avc-caps", "decoder-compatible"},
            {"probe-nal-caps", "decoder-compatible"},
            {"probe-bad-dimensions", "invalid visible"}}}) {
    set_scenario(scenario_name);
    expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                       fragment, scenario_name);
  }

  for (const auto& [scenario_name, expected_parser_inputs] :
       std::array<std::pair<std::string_view, std::size_t>, 3>{
           {{"probe-input-byte-budget", 0},
            {"probe-input-byte-budget-runahead", 8},
            {"probe-input-byte-budget-dequeue-race", 2}}}) {
    set_scenario(scenario_name);
    std::filesystem::remove(event_path);
    expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                       "bytes-per-access-unit", scenario_name);
    const auto events = read_events(event_path);
    expect_true(has_event(events, "add-input-budget-probe"),
                std::string(scenario_name) + " installs a synchronous input pad probe");
    expect_eq(count_event(events, "parser-input"), expected_parser_inputs,
              std::string(scenario_name) + " blocks the offending buffer before the parser");
    expect_eq(count_event(events, "input-budget-drop"), std::size_t{1},
              std::string(scenario_name) + " drops the first buffer over the byte budget");
    expect_true(has_event(events, "remove-input-budget-probe"),
                std::string(scenario_name) + " removes the input probe during teardown");
    expect_true(!has_event(events, "probe-leaked"),
                std::string(scenario_name) + " does not leak the input probe");
  }

  const auto dequeue_race_events = read_events(event_path);
  expect_true(has_event(dequeue_race_events, "post-dequeue-input-admitted"),
              "dequeue race admits the next access unit before the pull returns");
  expect_true(has_event(dequeue_race_events, "access-unit-checkpoint"),
              "parser output checkpoints the completed access unit in streaming order");

  for (const auto scenario_name :
       {"probe-input-buffer-list-budget", "probe-input-buffer-list-overflow"}) {
    set_scenario(scenario_name);
    std::filesystem::remove(event_path);
    expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                       "bytes-per-access-unit", scenario_name);
    const auto events = read_events(event_path);
    expect_true(has_event(events, "add-input-budget-probe"),
                std::string(scenario_name) + " installs the buffer-list input probe");
    expect_eq(count_event(events, "parser-input-list"), std::size_t{0},
              std::string(scenario_name) + " blocks the complete list before the parser");
    expect_eq(count_event(events, "input-budget-list-drop"), std::size_t{1},
              std::string(scenario_name) + " rejects the oversized aggregate");
    expect_true(has_event(events, "remove-input-budget-probe"),
                std::string(scenario_name) + " removes the input probe during teardown");
    expect_true(!has_event(events, "probe-leaked"),
                std::string(scenario_name) + " does not leak either pad probe");
  }

  set_scenario("probe-input-byte-budget-checkpoint");
  std::filesystem::remove(event_path);
  expect_eq(probe_video(container_config(video_path), timeout_ns).width, 3840U,
            "an exact-limit input snapshot is accepted and checkpointed atomically");
  const auto checkpoint_events = read_events(event_path);
  expect_true(count_event(checkpoint_events, "parser-input") > 1,
              "input following an exact-limit access unit continues through the parser");
  expect_true(!has_event(checkpoint_events, "input-budget-drop"),
              "checkpointing the exact-limit snapshot does not charge later input twice");

  for (const auto scenario_name : {"probe-timeout", "probe-pull-timeout"}) {
    set_scenario(scenario_name);
    expect_probe_error([&] { (void)probe_video(container_config(video_path), 1'000'000'000ULL); },
                       "timed out", scenario_name);
  }

  set_scenario("probe-odd-dimensions");
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "incompatible with NV12", "odd parser-visible dimensions rejected");
  set_scenario("probe-high-fps");
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_bounded_in_process_for_test(
            container_config(video_path), timeout_ns);
      },
      "implausible frame rate", "time-base artifact FPS rejected independently of worker startup");

  set_scenario("probe-parse-partial-error");
  std::filesystem::remove(event_path);
  expect_probe_error([&] { (void)probe_video(container_config(video_path), timeout_ns); },
                     "fake partial parse failure", "partial parse failure rejected");
  expect_true(has_event(read_events(event_path), "unref-pipeline"),
              "partially constructed pipeline is released before error propagation");
}

void worker_ipc_failures_are_bounded(const std::filesystem::path& video_path) {
  auto large_config = container_config(video_path);
  large_config.path.assign(250'000, 'a');
  large_config.path += ".mp4";

  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");
  const auto blocked_input_started = std::chrono::steady_clock::now();
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(large_config, fake_probe_worker_path, 1'000'000'000ULL);
      },
      "timed out", "worker that never reads its request respects the probe deadline");
  expect_true(std::chrono::steady_clock::now() - blocked_input_started <
                  std::chrono::milliseconds(1'800),
              "blocked worker request IPC is reclaimed before its native sleep completes");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "blocked worker cleanup is certified before returning");

  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "close-input");
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(large_config, fake_probe_worker_path, 5'000'000'000ULL);
      },
      "video probe worker", "closed worker input is an exception rather than process SIGPIPE");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "closed-input worker cleanup is certified before returning");

  const std::array<std::pair<std::string_view, std::string_view>, 11> invalid_workers{{
      {"crash", "exited abnormally"},
      {"malformed-response", "not valid CBOR"},
      {"deep-response", "nesting"},
      {"truncated-frame", "truncated IPC frame"},
      {"oversized-frame", "IPC frame length"},
      {"trailing-response", "trailing IPC bytes"},
      {"wrong-version", "unsupported protocol version"},
      {"wrapped-version", "out-of-range protocol_version"},
      {"invalid-metadata", "invalid metadata"},
      {"negative-metadata", "out-of-range width"},
      {"oversized-metadata", "out-of-range width"},
  }};
  for (const auto& [scenario, fragment] : invalid_workers) {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", std::string(scenario));
    expect_probe_error(
        [&] {
          (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                          5'000'000'000ULL);
        },
        fragment, std::string("fake worker response: ") + std::string(scenario));
    expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
              std::string("invalid worker cleanup is certified: ") + std::string(scenario));
  }
}

void aggregate_worker_memory_budget_is_enforced() {
  constexpr std::uint64_t kExpectedWorkerBytes = 512ULL * 1024ULL * 1024ULL;
  constexpr std::uint64_t kExpectedSnapshotBytes = 256ULL * 1024ULL * 1024ULL;
  constexpr std::uint64_t kExpectedAggregateBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
  const auto worker_bytes = reco::io::detail::maximum_probe_worker_address_space_bytes_for_test();
  const auto snapshot_bytes = reco::io::detail::maximum_probe_executable_snapshot_bytes_for_test();
  const auto reservation_bytes =
      reco::io::detail::maximum_per_probe_memory_reservation_bytes_for_test();
  const auto aggregate_bytes = reco::io::detail::maximum_aggregate_probe_memory_bytes_for_test();
  expect_eq(worker_bytes, kExpectedWorkerBytes,
            "probe worker memory reservation matches its limit");
  expect_eq(snapshot_bytes, kExpectedSnapshotBytes,
            "probe executable snapshot memory reservation matches its limit");
#if defined(__linux__)
  expect_eq(reservation_bytes, kExpectedWorkerBytes + kExpectedSnapshotBytes,
            "Linux probe admission includes worker and RAM-backed snapshot memory");
#else
  expect_eq(reservation_bytes, kExpectedWorkerBytes,
            "non-Linux probe admission includes its worker memory");
#endif
  expect_eq(aggregate_bytes, kExpectedAggregateBytes,
            "aggregate probe memory budget leaves caller headroom");
  if (worker_bytes == 0 || reservation_bytes == 0 || aggregate_bytes < reservation_bytes) {
    expect_true(false, "aggregate probe memory budget admits at least one whole reservation");
    return;
  }

  const auto reservation_count = static_cast<std::size_t>(aggregate_bytes / reservation_bytes);
  const auto expected_reserved_worker_bytes = reservation_count * worker_bytes;
  const auto expected_reserved_probe_bytes = reservation_count * reservation_bytes;
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> reservation_failed{false};
  std::vector<std::thread> reservations;
  try {
    reservations.reserve(reservation_count);
    for (std::size_t index = 0; index < reservation_count; ++index) {
      reservations.emplace_back([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        try {
          reco::io::detail::hold_probe_worker_memory_reservation_for_test(2'000'000'000ULL);
        } catch (...) {
          reservation_failed.store(true, std::memory_order_release);
        }
      });
    }
  } catch (...) {
    start.store(true, std::memory_order_release);
    for (auto& reservation : reservations) {
      if (reservation.joinable()) {
        reservation.join();
      }
    }
    expect_true(false, "aggregate probe worker reservation threads are created");
    return;
  }
  while (ready.load(std::memory_order_acquire) != reservation_count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  const auto saturation_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (reco::io::detail::reserved_probe_memory_bytes_for_test() !=
             expected_reserved_probe_bytes &&
         std::chrono::steady_clock::now() < saturation_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(),
            expected_reserved_worker_bytes,
            "all admitted probe worker address-space reservations are tracked");
  expect_eq(reco::io::detail::reserved_probe_memory_bytes_for_test(), expected_reserved_probe_bytes,
            "all admitted probe memory reservations are tracked");
  expect_true(aggregate_bytes - expected_reserved_probe_bytes < reservation_bytes,
              "aggregate probe memory leaves less than one reservation unallocated");
  expect_probe_error([] { reco::io::detail::hold_probe_worker_memory_reservation_for_test(0); },
                     "aggregate video probe memory budget",
                     "an additional probe is rejected at the aggregate memory boundary");

  for (auto& reservation : reservations) {
    reservation.join();
  }
  expect_true(!reservation_failed.load(std::memory_order_acquire),
              "every reservation within the aggregate worker memory budget succeeds");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "probe worker memory reservations are released after completion");
  expect_eq(reco::io::detail::reserved_probe_memory_bytes_for_test(), 0ULL,
            "aggregate probe memory reservations are released after completion");
}

void maximum_linux_snapshots_are_aggregate_bounded() {
#if defined(__linux__)
  const auto snapshot_bytes = reco::io::detail::maximum_probe_executable_snapshot_bytes_for_test();
  const auto reservation_bytes =
      reco::io::detail::maximum_per_probe_memory_reservation_bytes_for_test();
  const auto aggregate_bytes = reco::io::detail::maximum_aggregate_probe_memory_bytes_for_test();
  const auto reservation_count = static_cast<std::size_t>(aggregate_bytes / reservation_bytes);
  expect_eq(reservation_count, 2U,
            "Linux aggregate memory budget admits two maximum-snapshot probes");

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() /
                    ("reco_probe_maximum_snapshot_" + std::to_string(unique));
  const auto worker = root / "maximum-snapshot-worker";
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
  std::vector<std::array<int, 2>> ready(reservation_count, {-1, -1});
  std::vector<std::array<int, 2>> release(reservation_count, {-1, -1});
  std::vector<std::thread> snapshots;
  std::atomic<bool> snapshot_failed = false;
  try {
    std::filesystem::create_directory(root);
    std::filesystem::copy_file(fake_probe_worker_path, worker);
    std::filesystem::permissions(worker,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);
    const int worker_descriptor = ::open(worker.c_str(), O_WRONLY | O_CLOEXEC);
    if (worker_descriptor < 0 ||
        ::ftruncate(worker_descriptor, static_cast<off_t>(snapshot_bytes)) != 0) {
      const auto resize_error = errno;
      if (worker_descriptor >= 0) {
        (void)::close(worker_descriptor);
      }
      throw std::runtime_error("failed to resize maximum snapshot fixture: " +
                               std::string(std::strerror(resize_error)));
    }
    (void)::close(worker_descriptor);

    snapshots.reserve(reservation_count);
    for (std::size_t index = 0; index < reservation_count; ++index) {
      if (::pipe(ready[index].data()) != 0 || ::pipe(release[index].data()) != 0) {
        throw std::runtime_error("failed to create maximum snapshot synchronization pipes");
      }
      snapshots.emplace_back([&, index] {
        try {
          reco::io::detail::hold_linux_probe_executable_snapshot_for_test(worker, ready[index][1],
                                                                          release[index][0]);
        } catch (...) {
          snapshot_failed.store(true, std::memory_order_release);
        }
      });
    }

    bool every_snapshot_ready = true;
    for (std::size_t index = 0; index < reservation_count; ++index) {
      pollfd descriptor{.fd = ready[index][0], .events = POLLIN, .revents = 0};
      int poll_result = -1;
      do {
        poll_result = ::poll(&descriptor, 1, 30'000);
      } while (poll_result < 0 && errno == EINTR);
      char ready_byte = '\0';
      const auto ready_size = poll_result == 1 ? ::read(ready[index][0], &ready_byte, 1) : -1;
      every_snapshot_ready = every_snapshot_ready && ready_size == 1 && ready_byte == 'R';
    }
    expect_true(every_snapshot_ready,
                "concurrent maximum-size Linux executable snapshots are populated");
    expect_eq(reco::io::detail::reserved_probe_memory_bytes_for_test(),
              reservation_count * reservation_bytes,
              "maximum-size Linux snapshots remain inside aggregate admission");
    expect_probe_error(
        [&] { reco::io::detail::hold_linux_probe_executable_snapshot_for_test(worker, -1, -1); },
        "aggregate video probe memory budget",
        "another maximum Linux snapshot is rejected before allocating its memfd");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: maximum Linux snapshot fixture failed: " << error.what() << '\n';
    ++failures;
  }

  for (auto& pipe : release) {
    if (pipe[1] >= 0) {
      const char release_byte = 'R';
      (void)::write(pipe[1], &release_byte, 1);
    }
  }
  for (auto& snapshot : snapshots) {
    if (snapshot.joinable()) {
      snapshot.join();
    }
  }
  for (auto& pipe : ready) {
    (void)::close(pipe[0]);
    (void)::close(pipe[1]);
  }
  for (auto& pipe : release) {
    (void)::close(pipe[0]);
    (void)::close(pipe[1]);
  }
  expect_true(!snapshot_failed.load(std::memory_order_acquire),
              "maximum Linux snapshot holders complete successfully");
  expect_eq(reco::io::detail::reserved_probe_memory_bytes_for_test(), 0ULL,
            "maximum Linux snapshot reservations are released");
  std::filesystem::remove_all(root, cleanup_error);
#endif
}

void inherited_probe_state_is_rejected_after_fork(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  std::atomic<bool> reservation_failed = false;
  std::thread reservation([&] {
    try {
      reco::io::detail::hold_probe_worker_memory_reservation_for_test(2'000'000'000ULL);
    } catch (...) {
      reservation_failed.store(true, std::memory_order_release);
    }
  });
  const auto reservation_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (reco::io::detail::reserved_probe_worker_address_space_bytes_for_test() == 0 &&
         std::chrono::steady_clock::now() < reservation_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  int readiness[2] = {-1, -1};
  expect_true(::pipe(readiness) == 0, "post-fork admission-lock readiness pipe opens");
  std::atomic<bool> lock_failed = false;
  std::thread admission_lock;
  bool admission_lock_held = false;
  if (readiness[0] >= 0) {
    admission_lock = std::thread([&] {
      try {
        reco::io::detail::hold_probe_worker_admission_lock_for_test(2'000'000'000ULL, readiness[1]);
      } catch (...) {
        lock_failed.store(true, std::memory_order_release);
      }
    });
    pollfd readiness_poll{.fd = readiness[0], .events = POLLIN, .revents = 0};
    int poll_result = -1;
    do {
      poll_result = ::poll(&readiness_poll, 1, 1'000);
    } while (poll_result < 0 && errno == EINTR);
    char ready = '\0';
    const auto ready_size = poll_result > 0 ? ::read(readiness[0], &ready, 1) : -1;
    admission_lock_held = ready_size == 1 && ready == 'R';
    expect_true(admission_lock_held, "post-fork test observes the inherited admission lock");
  }

  const auto child = ::fork();
  if (child == 0) {
    (void)::alarm(3);
    try {
      (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      5'000'000'000ULL);
    } catch (const GpuVideoProbeError& error) {
      const auto message = std::string_view(error.what());
      ::_exit(message.find("after fork without exec") != std::string_view::npos ? 0 : 2);
    } catch (...) {
      ::_exit(3);
    }
    ::_exit(4);
  }
  expect_true(child > 0, "post-fork rejection child starts");
  int child_status = 0;
  if (child > 0) {
    while (::waitpid(child, &child_status, 0) < 0 && errno == EINTR) {
    }
    expect_true(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
                "post-fork probe use rejects before inherited admission or reaper state");
  }

  if (admission_lock.joinable()) {
    admission_lock.join();
  }
  if (readiness[0] >= 0) {
    (void)::close(readiness[0]);
    (void)::close(readiness[1]);
  }
  reservation.join();
  expect_true(!reservation_failed.load(std::memory_order_acquire),
              "post-fork reservation setup succeeds");
  expect_true(!lock_failed.load(std::memory_order_acquire),
              "post-fork admission-lock setup succeeds");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "fork rejection does not alter the parent's aggregate admission");
#else
  (void)video_path;
#endif
}

void deferred_cleanup_retains_worker_memory_reservation(const std::filesystem::path& video_path) {
#if defined(__linux__)
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".deferred-worker");
  std::filesystem::remove(worker_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");
  bool probe_failed = false;
  std::thread probe([&] {
    try {
      (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      1'000'000'000ULL);
    } catch (...) {
      probe_failed = true;
    }
  });

  const auto discovery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  const auto owner =
      wait_for_descendant(::getpid(), discovery_deadline, "--reco-video-probe-owner");
  const auto supervisor =
      owner.has_value() ? wait_for_direct_child(*owner, discovery_deadline) : std::nullopt;
  const auto guardian = supervisor.has_value()
                            ? wait_for_direct_child(*supervisor, discovery_deadline)
                            : std::nullopt;
  const auto worker = wait_for_process_marker(worker_marker, std::chrono::steady_clock::now() +
                                                                 std::chrono::seconds(2));
  expect_true(owner.has_value(), "deferred-cleanup probe owner starts");
  expect_true(supervisor.has_value(), "deferred-cleanup probe supervisor starts");
  expect_true(guardian.has_value(), "deferred-cleanup probe guardian starts");
  expect_true(worker.has_value(), "deferred-cleanup probe worker starts");
  const bool worker_group_stopped =
      worker.has_value() && ::kill(-static_cast<pid_t>(*worker), SIGSTOP) == 0;
  const bool guardian_stopped = guardian.has_value() && ::kill(*guardian, SIGSTOP) == 0;
  const bool supervisor_stopped = supervisor.has_value() && ::kill(*supervisor, SIGSTOP) == 0;
  const bool owner_stopped = owner.has_value() && ::kill(*owner, SIGSTOP) == 0;
  expect_true(worker_group_stopped, "deferred-cleanup worker group is frozen adversarially");
  expect_true(guardian_stopped, "deferred-cleanup guardian is frozen adversarially");
  expect_true(supervisor_stopped, "deferred-cleanup supervisor stops before timeout");
  expect_true(owner_stopped, "deferred-cleanup owner stops before timeout");

  probe.join();
  expect_true(probe_failed, "stopped supervisor causes a bounded probe failure");
  const bool process_tree_frozen =
      worker_group_stopped && guardian_stopped && supervisor_stopped && owner_stopped;
  if (process_tree_frozen) {
    expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(),
              reco::io::detail::maximum_probe_worker_address_space_bytes_for_test(),
              "deferred supervisor cleanup retains its worker memory reservation");
    (void)::kill(*owner, SIGCONT);
  } else {
    if (owner.has_value()) {
      (void)::kill(*owner, SIGCONT);
    }
    if (supervisor.has_value()) {
      (void)::kill(*supervisor, SIGCONT);
    }
    if (guardian.has_value()) {
      (void)::kill(*guardian, SIGCONT);
    }
    if (worker.has_value()) {
      (void)::kill(-static_cast<pid_t>(*worker), SIGCONT);
    }
  }

  const auto release_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (reco::io::detail::reserved_probe_worker_address_space_bytes_for_test() != 0 &&
         std::chrono::steady_clock::now() < release_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "deferred supervisor cleanup releases its reservation after process-tree exit");
  if (process_tree_frozen && worker.has_value()) {
    errno = 0;
    expect_true(::kill(static_cast<pid_t>(*worker), 0) != 0 && errno == ESRCH,
                "admission is released only after the frozen worker group is empty");
  }
  if (supervisor_stopped &&
      reco::io::detail::reserved_probe_worker_address_space_bytes_for_test() != 0) {
    (void)::kill(*supervisor, SIGKILL);
    (void)::kill(*supervisor, SIGCONT);
  }
  if (owner_stopped) {
    (void)::kill(*owner, SIGCONT);
  }
  if (guardian.has_value()) {
    (void)::kill(*guardian, SIGCONT);
  }
  if (worker.has_value()) {
    (void)::kill(-static_cast<pid_t>(*worker), SIGCONT);
    (void)::kill(-static_cast<pid_t>(*worker), SIGKILL);
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(worker_marker);
#else
  (void)video_path;
#endif
}

void killed_cleanup_authority_fails_closed(const std::filesystem::path& video_path) {
#if defined(__linux__)
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".killed-authority-worker");
  std::filesystem::remove(worker_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");
  bool probe_failed = false;
  std::thread probe([&] {
    try {
      (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      5'000'000'000ULL);
    } catch (...) {
      probe_failed = true;
    }
  });

  const auto owner =
      wait_for_direct_child(::getpid(), std::chrono::steady_clock::now() + std::chrono::seconds(2));
  const auto worker = wait_for_process_marker(worker_marker, std::chrono::steady_clock::now() +
                                                                 std::chrono::seconds(2));
  expect_true(owner.has_value(), "killed-authority probe owner starts");
  expect_true(worker.has_value(), "killed-authority probe worker starts");
  const bool authority_killed = owner.has_value() && ::kill(*owner, SIGKILL) == 0;
  expect_true(authority_killed, "cleanup authority is killed before certification");

  probe.join();
  expect_true(probe_failed, "killed cleanup authority fails the active probe");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(),
            reco::io::detail::maximum_probe_worker_address_space_bytes_for_test(),
            "uncertified supervisor death retains aggregate admission for process life");

  if (worker.has_value()) {
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < cleanup_deadline) {
      errno = 0;
      if (::kill(static_cast<pid_t>(*worker), 0) != 0 && errno == ESRCH) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    (void)::kill(-static_cast<pid_t>(*worker), SIGKILL);
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(worker_marker);
#else
  (void)video_path;
#endif
}

void delayed_supervision_cannot_launch_worker(const std::filesystem::path& video_path) {
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".delayed-worker-startup");
  std::filesystem::remove(marker_path);
  set_environment("RECO_FAKE_PROBE_STARTUP_MARKER_PATH", marker_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "startup-marker");
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_with_supervisor_start_delay_for_test(
            container_config(video_path), fake_probe_worker_path, 1'000'000'000ULL,
            1'250'000'000ULL);
      },
      "timed out", "delayed supervision returns at the public deadline");
  expect_true(!std::filesystem::exists(marker_path),
              "expired supervision delay never launches a worker");
  std::filesystem::remove(marker_path);
}

void launch_gate_prevents_post_timeout_process_start(const std::filesystem::path& video_path) {
  const auto lifecycle_path =
      video_path.parent_path() / (video_path.filename().string() + ".launch-lifecycle");
  std::filesystem::remove(lifecycle_path);
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", lifecycle_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  const auto started = std::chrono::steady_clock::now();
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_with_pre_worker_spawn_delay_for_test(
            container_config(video_path), fake_probe_worker_path, 1'000'000'000ULL,
            1'250'000'000ULL);
      },
      "timed out", "pre-worker-spawn delay reaches a bounded timeout");
  const auto elapsed = std::chrono::steady_clock::now() - started;
  expect_true(elapsed >= std::chrono::milliseconds(800) &&
                  elapsed < std::chrono::milliseconds(1'200),
              "pre-worker launch delay returns within the public timeout bound");
  const auto events = read_events(lifecycle_path);
  expect_true(!has_event(events, "worker"), "expired launch sequence does not spawn a worker");
  expect_true(!has_event(events, "guard"), "expired launch sequence does not spawn a guard");
  expect_true(!has_event(events, "request"), "expired launch sequence does not write a request");
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", "");
  std::filesystem::remove(lifecycle_path);
}

void guardian_death_before_pid_report_reclaims_worker(const std::filesystem::path& video_path) {
#if defined(__linux__)
  const auto caller =
      fork_exec_probe_caller(video_path, fake_probe_worker_path, "pre-worker-report-delay",
                             30'000'000'000ULL, 5'000'000'000ULL);
  expect_true(caller > 0, "pre-report guardian-death caller starts");
  if (caller <= 0) {
    return;
  }

  const auto discovery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  const auto supervisor =
      wait_for_descendant(caller, discovery_deadline, "--reco-video-probe-supervisor");
  const auto guardian = supervisor.has_value()
                            ? wait_for_direct_child(*supervisor, discovery_deadline)
                            : std::nullopt;
  const auto worker =
      guardian.has_value() ? wait_for_direct_child(*guardian, discovery_deadline) : std::nullopt;
  expect_true(supervisor.has_value(), "pre-exec guardian supervisor starts");
  expect_true(guardian.has_value(), "guardian starts before the delayed PID report");
  expect_true(worker.has_value(), "start-gated worker exists before its PID is reported");
  if (guardian.has_value()) {
    (void)::kill(*guardian, SIGKILL);
  }

  int caller_status = 0;
  while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
  }
  expect_true(WIFEXITED(caller_status) && WEXITSTATUS(caller_status) == 0,
              "guardian loss fails the blocked probe call");

  bool worker_exited = false;
  const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (worker.has_value() && std::chrono::steady_clock::now() < exit_deadline) {
    errno = 0;
    if (::kill(*worker, 0) != 0 && errno == ESRCH) {
      worker_exited = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  expect_true(worker_exited, "guardian death closes the unreleased worker start gate");
  if (worker.has_value() && !worker_exited) {
    (void)::kill(*worker, SIGKILL);
  }
#else
  (void)video_path;
#endif
}

void auto_reaped_workers_are_supported(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  struct sigaction ignore_action{};
  struct sigaction previous_action{};
  ignore_action.sa_handler = SIG_IGN;
  sigemptyset(&ignore_action.sa_mask);
  if (sigaction(SIGCHLD, &ignore_action, &previous_action) != 0) {
    expect_true(false, "SIGCHLD auto-reap policy installs");
    return;
  }

  bool probe_succeeded = false;
  try {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    probe_succeeded = reco::io::probe_gpu_video(container_config(video_path),
                                                fake_probe_worker_path, 5'000'000'000ULL)
                          .width == 854U;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: auto-reaped worker probe threw: " << error.what() << '\n';
    ++failures;
  }
  struct sigaction current_action{};
  const bool disposition_preserved =
      sigaction(SIGCHLD, nullptr, &current_action) == 0 && current_action.sa_handler == SIG_IGN;
  if (sigaction(SIGCHLD, &previous_action, nullptr) != 0) {
    expect_true(false, "SIGCHLD policy restores");
  }
  expect_true(probe_succeeded, "auto-reaped host returns its framed worker response");
  expect_true(disposition_preserved, "auto-reaped host policy remains installed");
#else
  (void)video_path;
#endif
}

void no_child_wait_workers_are_supported(const std::filesystem::path& video_path) {
#if !defined(_WIN32) && defined(SA_NOCLDWAIT)
  struct sigaction no_wait_action{};
  struct sigaction previous_action{};
  no_wait_action.sa_handler = SIG_DFL;
  no_wait_action.sa_flags = SA_NOCLDWAIT;
  sigemptyset(&no_wait_action.sa_mask);
  if (sigaction(SIGCHLD, &no_wait_action, &previous_action) != 0) {
    expect_true(false, "SIGCHLD no-wait policy installs");
    return;
  }

  bool probe_succeeded = false;
  try {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    probe_succeeded = reco::io::probe_gpu_video(container_config(video_path),
                                                fake_probe_worker_path, 5'000'000'000ULL)
                          .width == 854U;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: no-wait worker probe threw: " << error.what() << '\n';
    ++failures;
  }
  struct sigaction current_action{};
  const bool disposition_preserved = sigaction(SIGCHLD, nullptr, &current_action) == 0 &&
                                     (current_action.sa_flags & SA_NOCLDWAIT) != 0;
  if (sigaction(SIGCHLD, &previous_action, nullptr) != 0) {
    expect_true(false, "SIGCHLD no-wait policy restores");
  }
  expect_true(probe_succeeded, "no-wait host returns its framed worker response");
  expect_true(disposition_preserved, "no-wait host policy remains installed");
#else
  (void)video_path;
#endif
}

void post_admission_sigchld_change_is_supported(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto marker =
      video_path.parent_path() / (video_path.filename().string() + ".post-admission-sigchld");
  std::filesystem::remove(marker);
  std::optional<GpuVideoProbe> result;
  std::string failure;
  std::thread probe([&] {
    try {
      result = reco::io::detail::probe_gpu_video_with_pre_supervisor_exec_delay_for_test(
          container_config(video_path), fake_probe_worker_path, 5'000'000'000ULL, 500'000'000ULL,
          marker);
    } catch (const std::exception& error) {
      failure = error.what();
    }
  });

  const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while ((!std::filesystem::exists(marker) || std::filesystem::file_size(marker) == 0) &&
         std::chrono::steady_clock::now() < marker_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  const bool admitted = std::filesystem::exists(marker) && std::filesystem::file_size(marker) > 0;
  expect_true(admitted, "post-admission SIGCHLD probe reaches its owner-launch delay");

  struct sigaction ignored{};
  struct sigaction previous{};
  ignored.sa_handler = SIG_IGN;
  (void)sigemptyset(&ignored.sa_mask);
  const bool disposition_installed = admitted && ::sigaction(SIGCHLD, &ignored, &previous) == 0;
  expect_true(disposition_installed, "post-admission SIGCHLD auto-reap policy installs");
  probe.join();

  struct sigaction current{};
  const bool disposition_preserved = disposition_installed &&
                                     ::sigaction(SIGCHLD, nullptr, &current) == 0 &&
                                     current.sa_handler == SIG_IGN;
  if (disposition_installed) {
    (void)::sigaction(SIGCHLD, &previous, nullptr);
  }
  if (!failure.empty()) {
    std::cerr << "FAIL: post-admission SIGCHLD probe threw: " << failure << '\n';
    ++failures;
  }
  expect_true(result.has_value() && result->width == 854U,
              "post-admission SIGCHLD change preserves the probe result");
  expect_true(disposition_preserved, "post-admission SIGCHLD change remains owned by the host");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "post-admission SIGCHLD change releases aggregate admission");
  std::filesystem::remove(marker);
#else
  (void)video_path;
#endif
}

void owner_pre_main_startup_uses_probe_deadline(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto marker =
      video_path.parent_path() / (video_path.filename().string() + ".owner-pre-main-delay");
  std::filesystem::remove(marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "owner-pre-main-delay");
  std::optional<GpuVideoProbe> result;
  std::string failure;
  const auto started = std::chrono::steady_clock::now();
  try {
    result = reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                       5'000'000'000ULL);
  } catch (const std::exception& error) {
    failure = error.what();
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  if (!failure.empty()) {
    std::cerr << "FAIL: delayed owner startup probe threw: " << failure << '\n';
    ++failures;
  }
  expect_true(std::filesystem::exists(marker), "delayed owner reaches its pre-main initializer");
  expect_true(result.has_value() && result->width == 854U,
              "owner startup may use the configured probe deadline");
  expect_true(elapsed >= std::chrono::milliseconds(450) && elapsed < std::chrono::seconds(2),
              "delayed owner startup is not capped by the termination reserve");
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(marker);
#else
  (void)video_path;
#endif
}

void partial_owner_launch_report_respects_probe_deadline(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "partial-owner-launch-report");
  const auto started = std::chrono::steady_clock::now();
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        1'000'000'000ULL);
      },
      "failed to arm", "partial owner launch report cannot stall past the probe deadline");
  const auto elapsed = std::chrono::steady_clock::now() - started;
  expect_true(elapsed >= std::chrono::milliseconds(800) &&
                  elapsed < std::chrono::milliseconds(1'300),
              "partial owner launch report returns within the public timeout bound");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "partial owner launch report releases aggregate admission after certification");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#else
  (void)video_path;
#endif
}

void competing_waitpid_reaper_cannot_steal_cleanup_authority(
    const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  int owner_pid_pipe[2] = {-1, -1};
  int owner_release_pipe[2] = {-1, -1};
  if (::pipe(owner_pid_pipe) != 0 || ::pipe(owner_release_pipe) != 0) {
    (void)::close(owner_pid_pipe[0]);
    (void)::close(owner_pid_pipe[1]);
    (void)::close(owner_release_pipe[0]);
    (void)::close(owner_release_pipe[1]);
    expect_true(false, "competing waitpid reaper creates its owner barrier");
    return;
  }
  std::optional<GpuVideoProbe> result;
  std::string failure;
  std::thread probe([&] {
    try {
      result = reco::io::detail::probe_gpu_video_with_reaper_barrier_for_test(
          container_config(video_path), fake_probe_worker_path, 5'000'000'000ULL, owner_pid_pipe[1],
          owner_release_pipe[0], 500'000'000ULL);
    } catch (const std::exception& error) {
      failure = error.what();
    }
  });
  pollfd owner_ready{.fd = owner_pid_pipe[0], .events = POLLIN, .revents = 0};
  int owner_wait = -1;
  do {
    owner_wait = ::poll(&owner_ready, 1, 2'000);
  } while (owner_wait < 0 && errno == EINTR);
  pid_t owner = -1;
  ssize_t owner_size = -1;
  if (owner_wait > 0) {
    do {
      owner_size = ::read(owner_pid_pipe[0], &owner, sizeof(owner));
    } while (owner_size < 0 && errno == EINTR);
  }
  const bool owner_published = owner_size == static_cast<ssize_t>(sizeof(owner)) && owner > 0;
  expect_true(owner_published, "competing waitpid reaper receives the exact live probe owner");

  std::atomic<bool> stop_reaper{false};
  std::atomic<bool> reaper_armed{false};
  std::atomic<pid_t> reaped_owner{-1};
  std::thread reaper;
  if (owner_published) {
    reaper = std::thread([&] {
      while (!stop_reaper.load(std::memory_order_acquire)) {
        int status = 0;
        const auto child = ::waitpid(-1, &status, WNOHANG);
        reaper_armed.store(true, std::memory_order_release);
        if (child == owner) {
          reaped_owner.store(child, std::memory_order_release);
          return;
        }
        if (child == 0 || (child < 0 && errno == ECHILD)) {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      }
    });
  }
  const auto arm_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!reaper_armed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < arm_deadline) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  expect_true(reaper_armed.load(std::memory_order_acquire),
              "competing waitpid reaper is armed before owner release");
  ssize_t released = -1;
  do {
    released = ::write(owner_release_pipe[1], "G", 1);
  } while (released < 0 && errno == EINTR);
  expect_true(released == 1, "competing waitpid reaper releases the armed probe owner");
  (void)::close(owner_release_pipe[1]);
  owner_release_pipe[1] = -1;
  probe.join();
  stop_reaper.store(true, std::memory_order_release);
  if (reaper.joinable()) {
    reaper.join();
  }

  if (!failure.empty()) {
    std::cerr << "FAIL: competing waitpid reaper probe threw: " << failure << '\n';
    ++failures;
  }
  expect_true(result.has_value() && result->width == 854U,
              "competing waitpid reaper preserves the probe result");
  expect_true(owner_published && reaped_owner.load(std::memory_order_acquire) == owner,
              "competing waitpid reaper deterministically steals the exact probe owner");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "competing waitpid reaper cannot retain aggregate admission");
  (void)::close(owner_pid_pipe[0]);
  (void)::close(owner_pid_pipe[1]);
  (void)::close(owner_release_pipe[0]);
  (void)::close(owner_release_pipe[1]);
#else
  (void)video_path;
#endif
}

void windows_job_reclaims_worker_descendants(const std::filesystem::path& video_path) {
#if defined(_WIN32)
  const auto descendant_path =
      video_path.parent_path() / (video_path.filename().string() + ".probe-descendant");
  std::filesystem::remove(descendant_path);
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", descendant_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata-with-descendant");
  try {
    expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        5'000'000'000ULL)
                  .width,
              854U, "Windows worker with a descendant returns its framed response");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: Windows descendant probe threw: " << error.what() << '\n';
    ++failures;
  }

  const auto descendant = wait_for_process_marker(
      descendant_path, std::chrono::steady_clock::now() + std::chrono::seconds(2));
  expect_true(descendant.has_value(), "Windows worker records its descendant");
  if (descendant.has_value() && *descendant <= std::numeric_limits<DWORD>::max()) {
    SetLastError(ERROR_SUCCESS);
    WindowsHandle process(
        OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(*descendant)));
    const auto open_error = GetLastError();
    bool exited = !process && open_error == ERROR_INVALID_PARAMETER;
    if (process) {
      exited = WaitForSingleObject(process.get(), 2'000) == WAIT_OBJECT_0;
      if (!exited) {
        (void)TerminateProcess(process.get(), 1);
      }
    }
    expect_true(exited, "closing the Windows worker Job kills its descendant");
  }
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", "");
  std::filesystem::remove(descendant_path);
#else
  (void)video_path;
#endif
}

void caller_death_reclaims_worker_and_descendant(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".caller-worker");
  std::filesystem::remove(worker_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");
  const auto caller =
      fork_exec_probe_caller(video_path, fake_probe_worker_path, "probe", 30'000'000'000ULL);
  expect_true(caller > 0, "POSIX parent-death probe caller starts");
  if (caller > 0) {
    const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    const auto worker = wait_for_process_marker(worker_marker, marker_deadline);
    expect_true(worker.has_value(), "POSIX isolated worker starts before caller death");
    (void)::kill(caller, SIGKILL);
    int caller_status = 0;
    while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
    }
    const auto expect_process_exit = [](const std::optional<std::uint64_t>& process_id,
                                        std::string_view message) {
      if (!process_id.has_value() ||
          *process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
        return;
      }
      const auto pid = static_cast<pid_t>(*process_id);
      bool exited = false;
      const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (std::chrono::steady_clock::now() < exit_deadline) {
        errno = 0;
        if (::kill(pid, 0) != 0 && errno == ESRCH) {
          exited = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      expect_true(exited, message);
      if (!exited) {
        (void)::kill(pid, SIGKILL);
      }
    };
    expect_process_exit(worker, "POSIX worker exits when its caller dies");
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  std::filesystem::remove(worker_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#elif defined(_WIN32)
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".caller-worker");
  const auto descendant_marker =
      video_path.parent_path() / (video_path.filename().string() + ".caller-descendant");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(descendant_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", descendant_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input-with-descendant");
  set_environment("RECO_FAKE_PROBE_CALLER_VIDEO_PATH", video_path.string());
  set_environment("RECO_FAKE_PROBE_CALLER_WORKER_PATH", fake_probe_worker_path.string());

  const auto application = current_test_executable().native();
  auto command_line = L"\"" + application + L"\" --reco-parent-death-probe-caller";
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION caller_info{};
  const bool caller_started =
      CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                     CREATE_NO_WINDOW, nullptr, nullptr, &startup, &caller_info) != 0;
  expect_true(caller_started, "Windows parent-death probe caller starts with CreateProcessW");
  if (caller_started) {
    WindowsHandle caller_process(caller_info.hProcess);
    WindowsHandle caller_thread(caller_info.hThread);
    const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    const auto worker = wait_for_process_marker(worker_marker, marker_deadline);
    const auto descendant = wait_for_process_marker(descendant_marker, marker_deadline);
    expect_true(worker.has_value(), "Windows isolated worker starts before caller death");
    expect_true(descendant.has_value(), "Windows worker descendant starts before caller death");

    const auto open_process = [](const std::optional<std::uint64_t>& process_id) {
      if (!process_id.has_value() || *process_id > std::numeric_limits<DWORD>::max()) {
        return static_cast<HANDLE>(nullptr);
      }
      return OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(*process_id));
    };
    WindowsHandle worker_process(open_process(worker));
    WindowsHandle descendant_process(open_process(descendant));
    expect_true(worker_process && WaitForSingleObject(worker_process.get(), 0) == WAIT_TIMEOUT,
                "Windows isolated worker is live before caller death");
    expect_true(descendant_process &&
                    WaitForSingleObject(descendant_process.get(), 0) == WAIT_TIMEOUT,
                "Windows worker descendant is live before caller death");

    (void)TerminateProcess(caller_process.get(), 1);
    (void)WaitForSingleObject(caller_process.get(), 2'000);
    const bool worker_exited =
        worker_process && WaitForSingleObject(worker_process.get(), 2'000) == WAIT_OBJECT_0;
    expect_true(worker_exited, "Windows Job kill-on-close reclaims the worker after caller death");
    const bool descendant_exited =
        descendant_process && WaitForSingleObject(descendant_process.get(), 2'000) == WAIT_OBJECT_0;
    expect_true(descendant_exited,
                "Windows Job kill-on-close reclaims the descendant after caller death");
    if (worker_process && !worker_exited) {
      (void)TerminateProcess(worker_process.get(), 1);
    }
    if (descendant_process && !descendant_exited) {
      (void)TerminateProcess(descendant_process.get(), 1);
    }
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_DESCENDANT_PATH", "");
  set_environment("RECO_FAKE_PROBE_CALLER_VIDEO_PATH", "");
  set_environment("RECO_FAKE_PROBE_CALLER_WORKER_PATH", "");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(descendant_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#else
  (void)video_path;
#endif
}

void guardian_death_after_worker_release_reclaims_group(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto guardian_marker =
      video_path.parent_path() / (video_path.filename().string() + ".released-guardian");
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".released-worker");
  std::filesystem::remove(guardian_marker);
  std::filesystem::remove(worker_marker);
  set_environment("RECO_FAKE_PROBE_GUARDIAN_PID_PATH", guardian_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");

  bool probe_failed = false;
  std::thread probe([&] {
    try {
      (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      30'000'000'000ULL);
    } catch (...) {
      probe_failed = true;
    }
  });
  const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  const auto guardian = wait_for_process_marker(guardian_marker, marker_deadline);
  const auto worker = wait_for_process_marker(worker_marker, marker_deadline);
  expect_true(guardian.has_value(), "released worker guardian records its process ID");
  expect_true(worker.has_value(), "released worker records its process ID");
#if defined(__linux__) && defined(SYS_pidfd_open)
  const auto open_pidfd = [](const std::optional<std::uint64_t>& process_id) {
    if (!process_id.has_value() ||
        *process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
      return -1;
    }
    return static_cast<int>(::syscall(SYS_pidfd_open, static_cast<pid_t>(*process_id), 0U));
  };
  const int worker_pidfd = open_pidfd(worker);
#else
  constexpr int worker_pidfd = -1;
#endif
  if (guardian.has_value() &&
      *guardian <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    (void)::kill(static_cast<pid_t>(*guardian), SIGKILL);
  }
  probe.join();
  expect_true(probe_failed, "guardian loss fails the active probe call");

  const auto expect_exit = [](const std::optional<std::uint64_t>& process_id, int pidfd,
                              std::string_view message) {
    if (!process_id.has_value() ||
        *process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
      return;
    }
    const auto pid = static_cast<pid_t>(*process_id);
    bool exited = false;
#if defined(__linux__) && defined(SYS_pidfd_open)
    if (pidfd >= 0) {
      pollfd descriptor{.fd = pidfd, .events = POLLIN, .revents = 0};
      int poll_result = -1;
      do {
        poll_result = ::poll(&descriptor, 1, 5'000);
      } while (poll_result < 0 && errno == EINTR);
      exited = poll_result == 1 && (descriptor.revents & POLLIN) != 0;
      (void)::close(pidfd);
    }
#endif
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!exited && pidfd < 0 && std::chrono::steady_clock::now() < deadline) {
      errno = 0;
      if (::kill(pid, 0) != 0 && errno == ESRCH) {
        exited = true;
        break;
      }
#if defined(__linux__)
      std::ifstream status(std::filesystem::path("/proc") / std::to_string(pid) / "stat");
      std::string pid_text;
      std::string command;
      char state = '\0';
      if (status >> pid_text >> command >> state && state == 'Z') {
        exited = true;
        break;
      }
#endif
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect_true(exited, message);
    if (!exited) {
      (void)::kill(pid, SIGKILL);
    }
  };
  expect_exit(worker, worker_pidfd, "worker exits when its released guardian dies");

  set_environment("RECO_FAKE_PROBE_GUARDIAN_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(guardian_marker);
  std::filesystem::remove(worker_marker);
#else
  (void)video_path;
#endif
}

void caller_death_before_supervisor_main(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
#if defined(__APPLE__)
  const auto snapshots_before = mac_probe_snapshot_directories();
  std::vector<std::filesystem::path> caller_snapshots;
#endif
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-supervisor");
  const auto inherited_child_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-supervisor-child");
  std::filesystem::remove(marker_path);
  std::filesystem::remove(inherited_child_path);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", marker_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "supervisor-pre-main-block");
  const auto caller =
      fork_exec_probe_caller(video_path, fake_probe_worker_path, "probe", 30'000'000'000ULL, 0,
                             marker_path, inherited_child_path);
  expect_true(caller > 0, "POSIX pre-main supervisor caller starts");
  if (caller > 0) {
    const auto supervisor = wait_for_process_marker(marker_path, std::chrono::steady_clock::now() +
                                                                     std::chrono::seconds(2));
    expect_true(supervisor.has_value(), "supervisor reaches its pre-main test block");
    const auto inherited_child = wait_for_process_marker(
        inherited_child_path, std::chrono::steady_clock::now() + std::chrono::seconds(2));
    expect_true(inherited_child.has_value(), "fork-only caller child starts while probe is active");
    if (inherited_child.has_value()) {
      errno = 0;
      expect_true(::kill(static_cast<pid_t>(*inherited_child), 0) == 0,
                  "fork-only caller child remains live before caller death");
    }
#if defined(__APPLE__)
    const auto snapshots_during = mac_probe_snapshot_directories();
    for (const auto& snapshot : snapshots_during) {
      if (!std::binary_search(snapshots_before.begin(), snapshots_before.end(), snapshot)) {
        caller_snapshots.push_back(snapshot);
      }
    }
    expect_true(!caller_snapshots.empty(),
                "macOS caller owns a named executable snapshot before termination");
#endif
    (void)::kill(caller, SIGKILL);
    int caller_status = 0;
    while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
    }
    bool exited = false;
    const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (supervisor.has_value() && std::chrono::steady_clock::now() < exit_deadline) {
      errno = 0;
      if (::kill(static_cast<pid_t>(*supervisor), 0) != 0 && errno == ESRCH) {
        exited = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect_true(exited, "pre-main watchdog reclaims the supervisor when its caller dies");
    if (inherited_child.has_value()) {
      errno = 0;
      expect_true(::kill(static_cast<pid_t>(*inherited_child), 0) == 0,
                  "pre-main supervisor cleanup does not depend on killing the fork-only child");
      (void)::kill(static_cast<pid_t>(*inherited_child), SIGKILL);
    }
    if (supervisor.has_value() && !exited) {
      (void)::kill(static_cast<pid_t>(*supervisor), SIGKILL);
    }
#if defined(__APPLE__)
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    bool snapshots_removed = caller_snapshots.empty();
    while (!snapshots_removed && std::chrono::steady_clock::now() < cleanup_deadline) {
      snapshots_removed =
          std::all_of(caller_snapshots.begin(), caller_snapshots.end(), [](const auto& snapshot) {
            std::error_code exists_error;
            const bool exists = std::filesystem::exists(snapshot, exists_error);
            return !exists && !exists_error;
          });
      if (!snapshots_removed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    expect_true(snapshots_removed, "macOS caller death reclaims its executable snapshot");
#endif
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(marker_path);
  std::filesystem::remove(inherited_child_path);
#else
  (void)video_path;
#endif
}

void caller_process_group_death_before_supervisor_main(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-supervisor-group");
  std::filesystem::remove(marker_path);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", marker_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "supervisor-pre-main-block");
  const auto caller = fork_exec_probe_caller(video_path, fake_probe_worker_path, "probe",
                                             30'000'000'000ULL, 0, marker_path, {}, {}, true);
  expect_true(caller > 0, "POSIX process-group supervisor caller starts");
  if (caller > 0) {
    const auto supervisor = wait_for_process_marker(marker_path, std::chrono::steady_clock::now() +
                                                                     std::chrono::seconds(2));
    expect_true(supervisor.has_value(), "supervisor reaches its process-group pre-main block");
    (void)::kill(-caller, SIGKILL);
    int caller_status = 0;
    while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
    }

    bool exited = false;
    const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (supervisor.has_value() && std::chrono::steady_clock::now() < exit_deadline) {
      errno = 0;
      if (::kill(static_cast<pid_t>(*supervisor), 0) != 0 && errno == ESRCH) {
        exited = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect_true(exited,
                "detached pre-main watchdog reclaims the supervisor after caller group death");
    if (supervisor.has_value() && !exited) {
      (void)::kill(static_cast<pid_t>(*supervisor), SIGKILL);
    }
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(marker_path);
#else
  (void)video_path;
#endif
}

void mac_owner_reclaims_stalled_guardian_session(const std::filesystem::path& video_path) {
#if defined(__APPLE__)
  const auto marker =
      video_path.parent_path() / (video_path.filename().string() + ".stalled-guardian-session");
  std::filesystem::remove(marker);
  bool probe_failed = false;
  std::thread probe([&] {
    try {
      (void)reco::io::detail::probe_gpu_video_with_pre_guardian_exec_delay_for_test(
          container_config(video_path), fake_probe_worker_path, 2'000'000'000ULL, 30'000'000'000ULL,
          marker);
    } catch (...) {
      probe_failed = true;
    }
  });

  const auto discovery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  const auto guardian = wait_for_process_marker(marker, discovery_deadline);
  const auto owner =
      wait_for_descendant(::getpid(), discovery_deadline, "--reco-video-probe-owner");
  const auto supervisor =
      owner.has_value() ? wait_for_direct_child(*owner, discovery_deadline) : std::nullopt;
  expect_true(guardian.has_value(), "Darwin stalled guardian starts before owner cleanup");
  expect_true(owner.has_value(), "Darwin stalled guardian has an exec-backed session owner");
  expect_true(supervisor.has_value(), "Darwin stalled guardian has a supervisor");
  const bool supervisor_stopped = supervisor.has_value() && ::kill(*supervisor, SIGSTOP) == 0;
  expect_true(supervisor_stopped, "Darwin supervisor stalls before caller timeout");

  probe.join();
  expect_true(probe_failed, "Darwin stalled guardian causes a bounded probe failure");
  bool guardian_exited = false;
  const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (guardian.has_value() && std::chrono::steady_clock::now() < exit_deadline) {
    errno = 0;
    if (::kill(*guardian, 0) != 0 && errno == ESRCH) {
      guardian_exited = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  expect_true(guardian_exited, "Darwin owner drains its stalled guardian session");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "Darwin owner certifies only after its private session is empty");

  if (guardian.has_value() && !guardian_exited) {
    (void)::kill(*guardian, SIGKILL);
  }
  std::filesystem::remove(marker);
#else
  (void)video_path;
#endif
}

void caller_death_before_supervisor_arm(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-arm-supervisor");
  const auto inherited_child_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-arm-child");
  std::filesystem::remove(marker_path);
  std::filesystem::remove(inherited_child_path);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  const auto caller = fork_exec_probe_caller(video_path, fake_probe_worker_path,
                                             "pre-supervisor-arm-delay", 30'000'000'000ULL,
                                             30'000'000'000ULL, marker_path, inherited_child_path);
  expect_true(caller > 0, "POSIX pre-arm supervisor caller starts");
  if (caller <= 0) {
    return;
  }

  const auto processes = wait_for_process_pair_marker(
      marker_path, std::chrono::steady_clock::now() + std::chrono::seconds(2));
  const auto inherited_child = wait_for_process_marker(
      inherited_child_path, std::chrono::steady_clock::now() + std::chrono::seconds(2));
  expect_true(processes.has_value(), "supervisor and watchdog reach their pre-arm stall");
  expect_true(inherited_child.has_value(), "fork-only child starts before supervisor arming");

  (void)::kill(caller, SIGKILL);
  int caller_status = 0;
  while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
  }

  const auto expect_exit = [&](std::uint64_t process, std::string_view message) {
    bool exited = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (process <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max()) &&
           std::chrono::steady_clock::now() < deadline) {
      errno = 0;
      if (::kill(static_cast<pid_t>(process), 0) != 0 && errno == ESRCH) {
        exited = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect_true(exited, message);
    if (!exited && process <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
      (void)::kill(static_cast<pid_t>(process), SIGKILL);
    }
  };
  if (processes.has_value()) {
    expect_exit(processes->first,
                "caller death reclaims the supervisor before its launch gate is armed");
    expect_exit(processes->second,
                "caller death reclaims the watchdog before its target is delivered");
  }
  if (inherited_child.has_value()) {
    errno = 0;
    expect_true(::kill(static_cast<pid_t>(*inherited_child), 0) == 0,
                "pre-arm cleanup does not depend on killing the fork-only child");
    (void)::kill(static_cast<pid_t>(*inherited_child), SIGKILL);
  }
  std::filesystem::remove(marker_path);
  std::filesystem::remove(inherited_child_path);
#else
  (void)video_path;
#endif
}

void dead_watchdog_cannot_release_the_supervisor(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto reserved_before =
      reco::io::detail::reserved_probe_worker_address_space_bytes_for_test();
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".dead-pre-arm-watchdog");
  std::filesystem::remove(marker_path);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::atomic<bool> watchdog_killed = false;
  std::thread killer([&] {
    const auto processes = wait_for_process_pair_marker(
        marker_path, std::chrono::steady_clock::now() + std::chrono::seconds(2));
    if (processes.has_value() &&
        processes->second <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
      watchdog_killed.store(::kill(static_cast<pid_t>(processes->second), SIGKILL) == 0,
                            std::memory_order_release);
    }
  });
  expect_probe_error(
      [&] {
        (void)reco::io::detail::probe_gpu_video_with_pre_supervisor_arm_delay_for_test(
            container_config(video_path), fake_probe_worker_path, 5'000'000'000ULL, 250'000'000ULL,
            marker_path);
      },
      "", "a dead pre-main watchdog prevents supervisor release");
  killer.join();
  expect_true(watchdog_killed.load(std::memory_order_acquire),
              "pre-main watchdog is killed before the supervisor launch gate opens");
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(),
            reserved_before + reco::io::detail::maximum_probe_worker_address_space_bytes_for_test(),
            "uncertified pre-main owner death retains its admission for process life");
  std::filesystem::remove(marker_path);
#else
  (void)video_path;
#endif
}

void caller_death_before_guardian_main(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-guardian");
  const auto inherited_child_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-guardian-child");
  std::filesystem::remove(marker_path);
  std::filesystem::remove(inherited_child_path);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  const auto caller = fork_exec_probe_caller(video_path, fake_probe_worker_path,
                                             "pre-guardian-exec-delay", 30'000'000'000ULL,
                                             30'000'000'000ULL, marker_path, inherited_child_path);
  expect_true(caller > 0, "POSIX pre-main guardian caller starts");
  if (caller <= 0) {
    return;
  }

  const auto guardian = wait_for_process_marker(marker_path, std::chrono::steady_clock::now() +
                                                                 std::chrono::seconds(2));
  const auto inherited_child = wait_for_process_marker(
      inherited_child_path, std::chrono::steady_clock::now() + std::chrono::seconds(2));
  expect_true(guardian.has_value(), "POSIX guardian reaches its pre-main stall");
  expect_true(inherited_child.has_value(),
              "fork-only child starts after the guardian launch is committed");
  if (guardian.has_value() &&
      *guardian <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    errno = 0;
    expect_true(::kill(static_cast<pid_t>(*guardian), 0) == 0,
                "POSIX pre-main guardian is live before caller death");
  }

  (void)::kill(caller, SIGKILL);
  int caller_status = 0;
  while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
  }

  bool guardian_exited = false;
  const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (guardian.has_value() &&
         *guardian <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max()) &&
         std::chrono::steady_clock::now() < exit_deadline) {
    errno = 0;
    if (::kill(static_cast<pid_t>(*guardian), 0) != 0 && errno == ESRCH) {
      guardian_exited = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  expect_true(guardian_exited,
              "POSIX caller death reclaims the guardian before guardian main or loader completion");
  if (inherited_child.has_value() &&
      *inherited_child <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    expect_true(::kill(static_cast<pid_t>(*inherited_child), 0) == 0,
                "guardian cleanup does not depend on a fork-only host child exiting");
    (void)::kill(static_cast<pid_t>(*inherited_child), SIGKILL);
  }
  if (guardian.has_value() && !guardian_exited &&
      *guardian <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    (void)::kill(static_cast<pid_t>(*guardian), SIGKILL);
  }
  std::filesystem::remove(marker_path);
  std::filesystem::remove(inherited_child_path);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#else
  (void)video_path;
#endif
}

void executable_replacement_cannot_change_the_pinned_probe_image(
    const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() /
                    ("reco_probe_executable_race_" + std::to_string(unique));
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);

  try {
    std::filesystem::create_directory(root);
    const auto original = root / "original-probe-worker";
    const auto hostile = root / "replacement-probe-worker";
    std::filesystem::copy_file(fake_probe_worker_path, original);
    std::filesystem::permissions(original,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);
    {
      std::ofstream output(hostile, std::ios::binary | std::ios::trunc);
      output << "#!/bin/sh\nexit 97\n";
      if (!output) {
        throw std::runtime_error("failed to create hostile probe replacement");
      }
    }
    std::filesystem::permissions(hostile,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);

    const auto run_race = [&](std::string_view name, bool symlink_retarget, bool in_place,
                              bool before_supervisor) {
      const auto selected = root / (std::string(name) + "-selected");
      const auto staged = root / (std::string(name) + "-staged");
      const auto marker = root / (std::string(name) + "-marker");
      if (symlink_retarget) {
        std::filesystem::create_symlink(original.filename(), selected);
        std::filesystem::create_symlink(hostile.filename(), staged);
      } else {
        std::filesystem::copy_file(original, selected);
        std::filesystem::copy_file(hostile, staged);
      }

      std::optional<GpuVideoProbe> result;
      std::exception_ptr probe_error;
      std::thread probe([&] {
        try {
          if (before_supervisor) {
            result = reco::io::detail::probe_gpu_video_with_pre_supervisor_exec_delay_for_test(
                container_config(video_path), selected, 5'000'000'000ULL, 1'000'000'000ULL, marker);
          } else {
            result = reco::io::detail::probe_gpu_video_with_pre_guardian_exec_delay_for_test(
                container_config(video_path), selected, 5'000'000'000ULL, 1'000'000'000ULL, marker);
          }
        } catch (...) {
          probe_error = std::current_exception();
        }
      });

      const auto guardian = wait_for_process_marker(marker, std::chrono::steady_clock::now() +
                                                                std::chrono::seconds(2));
      expect_true(guardian.has_value(), std::string(name) + " reaches the pre-exec race hook");
      std::error_code replace_error;
      if (guardian.has_value()) {
        if (in_place) {
          std::ifstream input(hostile, std::ios::binary);
          std::ofstream output(selected, std::ios::binary | std::ios::trunc);
          output << input.rdbuf();
          if (!input || !output) {
            replace_error = std::make_error_code(std::errc::io_error);
          }
        } else {
          std::filesystem::rename(staged, selected, replace_error);
        }
      }
      expect_true(!replace_error, std::string(name) + " replacement completes");
      probe.join();

      if (probe_error != nullptr) {
        try {
          std::rethrow_exception(probe_error);
        } catch (const std::exception& error) {
          std::cerr << "FAIL: " << name << " changed the pinned executable: " << error.what()
                    << '\n';
          ++failures;
        } catch (...) {
          std::cerr << "FAIL: " << name << " changed the pinned executable with an unknown error\n";
          ++failures;
        }
      }
      expect_true(result.has_value() && result->width == 854U,
                  std::string(name) + " executes the originally pinned image");
    };

    set_scenario("probe-ok");
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    run_race("pre-supervisor regular executable replacement", false, false, true);
    run_race("pre-supervisor in-place executable overwrite", false, true, true);
    run_race("pre-supervisor symlink retarget", true, false, true);
    run_race("pre-guardian regular executable replacement", false, false, false);
    run_race("pre-guardian in-place executable overwrite", false, true, false);
    run_race("pre-guardian symlink retarget", true, false, false);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: pinned executable race fixture failed: " << error.what() << '\n';
    ++failures;
  }
  std::filesystem::remove_all(root, cleanup_error);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#else
  (void)video_path;
#endif
}

void linux_fork_child_does_not_retain_snapshot_memfd(const std::filesystem::path& video_path) {
#if defined(__linux__)
  const auto snapshot_marker =
      video_path.parent_path() / (video_path.filename().string() + ".linux-snapshot-ready");
  const auto child_marker =
      video_path.parent_path() / (video_path.filename().string() + ".linux-snapshot-child");
  const auto audit_marker =
      video_path.parent_path() / (video_path.filename().string() + ".linux-snapshot-audit");
  std::filesystem::remove(snapshot_marker);
  std::filesystem::remove(child_marker);
  std::filesystem::remove(audit_marker);
  std::optional<std::uint64_t> inherited_child;
  pid_t caller = -1;
  try {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    caller = fork_exec_probe_caller(video_path, fake_probe_worker_path, "pre-supervisor-exec-delay",
                                    30'000'000'000ULL, 30'000'000'000ULL, snapshot_marker,
                                    child_marker, audit_marker);
    expect_true(caller > 0, "Linux snapshot inheritance caller starts");
    const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const auto snapshot_ready = wait_for_process_marker(snapshot_marker, marker_deadline);
    inherited_child = wait_for_process_marker(child_marker, marker_deadline);
    const auto inherited_audit = wait_for_audit_marker(audit_marker, marker_deadline);
    expect_true(snapshot_ready.has_value(), "Linux snapshot inheritance fixture pins its memfd");
    expect_true(inherited_child.has_value(),
                "Linux snapshot inheritance fixture forks after memfd creation");
    expect_true(inherited_audit.has_value() && *inherited_audit == '0',
                "Linux at-fork child closes the exact pinned memfd descriptor");
    const auto child_pid =
        inherited_child.has_value() &&
                *inherited_child <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())
            ? static_cast<pid_t>(*inherited_child)
            : static_cast<pid_t>(-1);
    expect_true(child_pid > 0 && !process_has_linux_probe_memfd(child_pid),
                "Linux fork-only child retains no video probe memfd");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: Linux snapshot inheritance fixture failed: " << error.what() << '\n';
    ++failures;
  }
  if (caller > 0) {
    (void)::kill(caller, SIGKILL);
    int status = 0;
    while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (inherited_child.has_value() &&
      *inherited_child <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    (void)::kill(static_cast<pid_t>(*inherited_child), SIGKILL);
  }
  std::filesystem::remove(snapshot_marker);
  std::filesystem::remove(child_marker);
  std::filesystem::remove(audit_marker);
#else
  (void)video_path;
#endif
}

void mac_probe_snapshot_preserves_quarantine(const std::filesystem::path& video_path) {
#if defined(__APPLE__)
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root =
      std::filesystem::temp_directory_path() / ("reco_probe_quarantine_" + std::to_string(unique));
  const auto selected = root / "quarantined-probe-worker";
  const auto marker = root / "snapshot-ready";
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
  try {
    std::filesystem::create_directory(root);
    std::filesystem::copy_file(fake_probe_worker_path, selected);
    std::filesystem::permissions(selected,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);
    static constexpr std::string_view quarantine =
        "0081;65000000;Reco;00000000-0000-0000-0000-000000000000";
    if (::setxattr(selected.c_str(), "com.apple.quarantine", quarantine.data(), quarantine.size(),
                   0, 0) != 0) {
      throw std::runtime_error("failed to install quarantine metadata");
    }
    const auto snapshots_before = mac_probe_snapshot_directories();
    const auto caller = fork_exec_probe_caller(video_path, selected, "pre-supervisor-exec-delay",
                                               30'000'000'000ULL, 30'000'000'000ULL, marker);
    expect_true(caller > 0, "macOS quarantine snapshot caller starts");
    const auto reached_marker =
        wait_for_process_marker(marker, std::chrono::steady_clock::now() + std::chrono::seconds(2));
    expect_true(reached_marker.has_value(), "macOS quarantine snapshot reaches its race hook");
    bool preserved = false;
    std::vector<std::filesystem::path> caller_snapshots;
    for (const auto& snapshot : mac_probe_snapshot_directories()) {
      if (std::binary_search(snapshots_before.begin(), snapshots_before.end(), snapshot)) {
        continue;
      }
      caller_snapshots.push_back(snapshot);
      std::array<char, 256> value{};
      const auto value_size = ::getxattr((snapshot / "probe-worker").c_str(),
                                         "com.apple.quarantine", value.data(), value.size(), 0, 0);
      preserved =
          preserved ||
          (value_size == static_cast<ssize_t>(quarantine.size()) &&
           std::string_view(value.data(), static_cast<std::size_t>(value_size)) == quarantine);
    }
    expect_true(preserved, "macOS executable snapshot preserves quarantine metadata");
    if (caller > 0) {
      (void)::kill(caller, SIGKILL);
      int status = 0;
      while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
      }
    }
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    bool snapshots_removed = caller_snapshots.empty();
    while (!snapshots_removed && std::chrono::steady_clock::now() < cleanup_deadline) {
      snapshots_removed =
          std::all_of(caller_snapshots.begin(), caller_snapshots.end(), [](const auto& snapshot) {
            std::error_code exists_error;
            return !std::filesystem::exists(snapshot, exists_error) && !exists_error;
          });
      if (!snapshots_removed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    expect_true(snapshots_removed, "macOS quarantine snapshot cleanup is bounded");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: macOS quarantine snapshot fixture failed: " << error.what() << '\n';
    ++failures;
  }
  std::filesystem::remove_all(root, cleanup_error);
#else
  (void)video_path;
#endif
}

void mac_snapshot_normal_cleanup_ignores_fork_only_children(
    const std::filesystem::path& video_path) {
#if defined(__APPLE__)
  const auto snapshot_marker =
      video_path.parent_path() / (video_path.filename().string() + ".snapshot-success-ready");
  const auto child_marker =
      video_path.parent_path() / (video_path.filename().string() + ".snapshot-success-child");
  std::filesystem::remove(snapshot_marker);
  std::filesystem::remove(child_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  const auto started = std::chrono::steady_clock::now();
  const auto caller =
      fork_exec_probe_caller(video_path, fake_probe_worker_path, "pre-supervisor-exec-delay",
                             5'000'000'000ULL, 100'000'000ULL, snapshot_marker, child_marker);
  int status = 0;
  while (caller > 0 && ::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
  }
  expect_true(caller > 0 && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
              "successful macOS probe with a fork-only child exits cleanly");
  expect_true(std::filesystem::exists(child_marker),
              "successful macOS probe keeps a fork-only child alive through snapshot teardown");
  expect_true(std::chrono::steady_clock::now() - started < std::chrono::milliseconds(900),
              "fork-only child cannot force the one-second snapshot cleanup fallback");
  std::filesystem::remove(snapshot_marker);
  std::filesystem::remove(child_marker);
#else
  (void)video_path;
#endif
}

void mac_maximum_concurrent_probes_fit_descriptor_registry(
    const std::filesystem::path& video_path) {
#if defined(__APPLE__)
  constexpr std::size_t probe_count = 4;
  std::array<std::optional<GpuVideoProbe>, probe_count> results;
  std::array<std::exception_ptr, probe_count> errors;
  std::vector<std::thread> probes;
  probes.reserve(probe_count);
  int ready[2] = {-1, -1};
  int release[2] = {-1, -1};
  if (::pipe(ready) != 0 || ::pipe(release) != 0) {
    (void)::close(ready[0]);
    (void)::close(ready[1]);
    (void)::close(release[0]);
    (void)::close(release[1]);
    expect_true(false, "macOS descriptor-peak fixture creates its barriers");
    return;
  }
  const auto ready_flags = ::fcntl(ready[0], F_GETFL);
  if (ready_flags < 0 || ::fcntl(ready[0], F_SETFL, ready_flags | O_NONBLOCK) != 0) {
    (void)::close(ready[0]);
    (void)::close(ready[1]);
    (void)::close(release[0]);
    (void)::close(release[1]);
    expect_true(false, "macOS descriptor-peak readiness is nonblocking");
    return;
  }
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  try {
    for (std::size_t index = 0; index < probe_count; ++index) {
      probes.emplace_back([&, index] {
        try {
          results[index] = reco::io::detail::probe_gpu_video_with_pre_owner_fork_barrier_for_test(
              container_config(video_path), fake_probe_worker_path, 10'000'000'000ULL, ready[1],
              release[0]);
        } catch (...) {
          errors[index] = std::current_exception();
        }
      });
    }
  } catch (...) {
    for (std::size_t index = 0; index < probes.size(); ++index) {
      (void)::write(release[1], "G", 1);
    }
    for (auto& probe : probes) {
      probe.join();
    }
    (void)::close(ready[0]);
    (void)::close(ready[1]);
    (void)::close(release[0]);
    (void)::close(release[1]);
    expect_true(false, "four concurrent macOS probe threads are created");
    return;
  }

  std::size_t ready_count = 0;
  const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (ready_count < probe_count && std::chrono::steady_clock::now() < ready_deadline) {
    char value = '\0';
    const auto received = ::read(ready[0], &value, 1);
    if (received == 1 && value == 'R') {
      ++ready_count;
    } else if (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  expect_eq(ready_count, probe_count,
            "all four macOS probes hold the protected-descriptor allocation peak");
  std::size_t release_count = 0;
  for (std::size_t index = 0; index < ready_count; ++index) {
    ssize_t written = -1;
    do {
      written = ::write(release[1], "G", 1);
    } while (written < 0 && errno == EINTR);
    if (written == 1) {
      ++release_count;
    }
  }
  expect_eq(release_count, ready_count, "macOS descriptor-peak barriers are released");
  (void)::close(release[1]);
  release[1] = -1;
  for (auto& probe : probes) {
    probe.join();
  }
  (void)::close(ready[0]);
  (void)::close(ready[1]);
  (void)::close(release[0]);
  (void)::close(release[1]);
  for (std::size_t index = 0; index < probe_count; ++index) {
    expect_true(errors[index] == nullptr && results[index].has_value() &&
                    results[index]->width == 854U,
                "all four admitted macOS probes fit the protected descriptor registry");
  }
#else
  (void)video_path;
#endif
}

void mac_snapshot_helper_survives_owner_process_group_termination(
    const std::filesystem::path& video_path) {
#if defined(__APPLE__)
  const auto marker =
      video_path.parent_path() / (video_path.filename().string() + ".snapshot-group-ready");
  std::filesystem::remove(marker);
  const auto snapshots_before = mac_probe_snapshot_directories();
  const auto caller =
      fork_exec_probe_caller(video_path, fake_probe_worker_path, "pre-supervisor-exec-delay",
                             30'000'000'000ULL, 30'000'000'000ULL, marker, {}, {}, true);
  expect_true(caller > 0, "macOS process-group snapshot caller starts");
  const auto ready =
      wait_for_process_marker(marker, std::chrono::steady_clock::now() + std::chrono::seconds(3));
  expect_true(ready.has_value(), "macOS process-group fixture creates its snapshot");
  std::vector<std::filesystem::path> caller_snapshots;
  for (const auto& snapshot : mac_probe_snapshot_directories()) {
    if (!std::binary_search(snapshots_before.begin(), snapshots_before.end(), snapshot)) {
      caller_snapshots.push_back(snapshot);
    }
  }
  expect_true(!caller_snapshots.empty(), "macOS process-group fixture observes the snapshot");
  if (caller > 0) {
    (void)::kill(-caller, SIGTERM);
    int status = 0;
    while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
    }
  }
  bool removed = caller_snapshots.empty();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!removed && std::chrono::steady_clock::now() < deadline) {
    removed = std::all_of(caller_snapshots.begin(), caller_snapshots.end(), [](const auto& path) {
      std::error_code error;
      return !std::filesystem::exists(path, error) && !error;
    });
    if (!removed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  expect_true(removed, "macOS cleanup helper survives owner process-group termination");
  std::filesystem::remove(marker);
#else
  (void)video_path;
#endif
}

void mac_probe_snapshot_cleanup_tracks_owner_process(const std::filesystem::path& video_path) {
#if defined(__APPLE__)
  const auto snapshot_marker =
      video_path.parent_path() / (video_path.filename().string() + ".snapshot-owner-ready");
  const auto child_marker =
      video_path.parent_path() / (video_path.filename().string() + ".snapshot-inherited-child");
  const auto audit_marker =
      video_path.parent_path() / (video_path.filename().string() + ".snapshot-inherited-audit");
  std::filesystem::remove(snapshot_marker);
  std::filesystem::remove(child_marker);
  std::filesystem::remove(audit_marker);
  const auto snapshots_before = mac_probe_snapshot_directories();
  std::vector<std::filesystem::path> caller_snapshots;
  std::optional<std::uint64_t> inherited_child;
  pid_t caller = -1;
  try {
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    caller = fork_exec_probe_caller(video_path, fake_probe_worker_path, "pre-supervisor-exec-delay",
                                    30'000'000'000ULL, 30'000'000'000ULL, snapshot_marker,
                                    child_marker, audit_marker);
    expect_true(caller > 0, "macOS inherited-socket snapshot caller starts");
    const auto marker_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const auto snapshot_ready = wait_for_process_marker(snapshot_marker, marker_deadline);
    inherited_child = wait_for_process_marker(child_marker, marker_deadline);
    const auto inherited_audit = wait_for_audit_marker(audit_marker, marker_deadline);
    expect_true(snapshot_ready.has_value(),
                "macOS inherited-socket fixture creates its executable snapshot");
    expect_true(inherited_child.has_value(),
                "macOS inherited-socket fixture forks a child after snapshot creation");
    expect_true(inherited_audit.has_value() && *inherited_audit == '0',
                "macOS at-fork child closes the exact snapshot backing vnode descriptor");
    for (const auto& snapshot : mac_probe_snapshot_directories()) {
      if (!std::binary_search(snapshots_before.begin(), snapshots_before.end(), snapshot)) {
        caller_snapshots.push_back(snapshot);
      }
    }
    expect_true(!caller_snapshots.empty(),
                "macOS inherited-socket fixture observes the caller snapshot");

    if (caller > 0) {
      (void)::kill(caller, SIGKILL);
      int status = 0;
      while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
      }
      caller = -1;
    }
    const auto child_pid =
        inherited_child.has_value() &&
                *inherited_child <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())
            ? static_cast<pid_t>(*inherited_child)
            : static_cast<pid_t>(-1);
    expect_true(child_pid > 0 && ::kill(child_pid, 0) == 0,
                "fork-only child remains alive after its cleanup socket is closed at fork");

    bool snapshots_removed = caller_snapshots.empty();
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!snapshots_removed && std::chrono::steady_clock::now() < cleanup_deadline) {
      snapshots_removed =
          std::all_of(caller_snapshots.begin(), caller_snapshots.end(), [](const auto& snapshot) {
            std::error_code exists_error;
            return !std::filesystem::exists(snapshot, exists_error) && !exists_error;
          });
      if (!snapshots_removed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    expect_true(snapshots_removed,
                "macOS owner death removes snapshots despite an inherited socket endpoint");
    expect_true(child_pid > 0 && ::kill(child_pid, 0) == 0,
                "macOS snapshot cleanup does not depend on killing the fork-only child");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: macOS inherited-socket snapshot fixture failed: " << error.what() << '\n';
    ++failures;
  }
  if (caller > 0) {
    (void)::kill(caller, SIGKILL);
    int status = 0;
    while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (inherited_child.has_value() &&
      *inherited_child <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    (void)::kill(static_cast<pid_t>(*inherited_child), SIGKILL);
  }
  std::filesystem::remove(snapshot_marker);
  std::filesystem::remove(child_marker);
  std::filesystem::remove(audit_marker);
#else
  (void)video_path;
#endif
}

void caller_death_before_worker_main(const std::filesystem::path& video_path) {
#if defined(_WIN32)
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-worker");
  const auto lifecycle_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-lifecycle");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(lifecycle_path);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", lifecycle_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "pre-main-block");
  set_environment("RECO_FAKE_PROBE_CALLER_VIDEO_PATH", video_path.string());
  set_environment("RECO_FAKE_PROBE_CALLER_WORKER_PATH", fake_probe_worker_path.string());

  const auto application = current_test_executable().native();
  auto command_line = L"\"" + application + L"\" --reco-parent-death-probe-caller";
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION caller_info{};
  const bool caller_started =
      CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                     CREATE_NO_WINDOW, nullptr, nullptr, &startup, &caller_info) != 0;
  expect_true(caller_started, "Windows pre-main parent-death probe caller starts");
  if (caller_started) {
    WindowsHandle caller_process(caller_info.hProcess);
    WindowsHandle caller_thread(caller_info.hThread);
    const auto worker = wait_for_process_marker(worker_marker, std::chrono::steady_clock::now() +
                                                                   std::chrono::seconds(2));
    expect_true(worker.has_value(), "Windows worker reaches its pre-main stall");
    WindowsHandle worker_process(
        worker.has_value() && *worker <= std::numeric_limits<DWORD>::max()
            ? OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(*worker))
            : nullptr);
    expect_true(worker_process && WaitForSingleObject(worker_process.get(), 0) == WAIT_TIMEOUT,
                "Windows pre-main worker is live before caller death");

    (void)TerminateProcess(caller_process.get(), 1);
    (void)WaitForSingleObject(caller_process.get(), 2'000);
    const bool worker_exited =
        worker_process && WaitForSingleObject(worker_process.get(), 2'000) == WAIT_OBJECT_0;
    expect_true(worker_exited, "Windows Job kill-on-close reclaims the worker before worker main");
    expect_true(!has_event(read_events(lifecycle_path), "worker"),
                "Windows startup-death test terminates before worker main");
    if (worker_process && !worker_exited) {
      (void)TerminateProcess(worker_process.get(), 1);
    }
  }

  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  set_environment("RECO_FAKE_PROBE_CALLER_VIDEO_PATH", "");
  set_environment("RECO_FAKE_PROBE_CALLER_WORKER_PATH", "");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(lifecycle_path);
#else
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-worker");
  const auto lifecycle_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-lifecycle");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(lifecycle_path);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", lifecycle_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "pre-main-block");
  const auto caller =
      fork_exec_probe_caller(video_path, fake_probe_worker_path, "probe", 30'000'000'000ULL);
  expect_true(caller > 0, "POSIX pre-main parent-death probe caller starts");
  if (caller > 0) {
    const auto worker = wait_for_process_marker(worker_marker, std::chrono::steady_clock::now() +
                                                                   std::chrono::seconds(2));
    expect_true(worker.has_value(), "POSIX worker reaches its pre-main stall");
    (void)::kill(caller, SIGKILL);
    int caller_status = 0;
    while (::waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
    }
    bool worker_exited = false;
    const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.has_value() && std::chrono::steady_clock::now() < exit_deadline) {
      errno = 0;
      if (::kill(static_cast<pid_t>(*worker), 0) != 0 && errno == ESRCH) {
        worker_exited = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect_true(worker_exited,
                "POSIX guardian reclaims the worker before worker main after caller death");
    expect_true(!has_event(read_events(lifecycle_path), "worker"),
                "POSIX startup-death test terminates before worker main");
    if (worker.has_value() && !worker_exited) {
      (void)::kill(static_cast<pid_t>(*worker), SIGKILL);
    }
  }
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(worker_marker);
  std::filesystem::remove(lifecycle_path);
#endif
}

void pre_main_loader_stall_respects_timeout(const std::filesystem::path& video_path) {
  const auto worker_marker =
      video_path.parent_path() / (video_path.filename().string() + ".loader-stall-worker");
  std::filesystem::remove(worker_marker);
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", worker_marker.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "pre-main-block");
  const auto started = std::chrono::steady_clock::now();
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        1'000'000'000ULL);
      },
      "timed out", "pre-main worker loader stall reaches a bounded timeout");
  const auto elapsed = std::chrono::steady_clock::now() - started;
  expect_true(elapsed >= std::chrono::milliseconds(800) &&
                  elapsed < std::chrono::milliseconds(1'200),
              "pre-main worker loader stall returns within the public timeout bound");
  const auto worker = wait_for_process_marker(worker_marker, std::chrono::steady_clock::now() +
                                                                 std::chrono::milliseconds(100));
  expect_true(worker.has_value(), "pre-main timeout observes the contained process");
#if defined(_WIN32)
  WindowsHandle worker_process(
      worker.has_value() && *worker <= std::numeric_limits<DWORD>::max()
          ? OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(*worker))
          : nullptr);
  const bool worker_exited =
      !worker_process || WaitForSingleObject(worker_process.get(), 0) == WAIT_OBJECT_0;
#else
  errno = 0;
  const bool worker_exited =
      worker.has_value() && ::kill(static_cast<pid_t>(*worker), 0) != 0 && errno == ESRCH;
#endif
  expect_true(worker_exited, "pre-main timeout synchronously reclaims the contained process");
  set_environment("RECO_FAKE_PROBE_WORKER_PID_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(worker_marker);
}

void unrelated_descriptor_writer_does_not_delay_pipe_eof(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  int pipe_descriptors[2] = {-1, -1};
  if (::pipe(pipe_descriptors) != 0) {
    expect_true(false, "unrelated descriptor EOF pipe opens");
    return;
  }

  const auto lifecycle_path =
      video_path.parent_path() / (video_path.filename().string() + ".pipe-lifecycle");
  std::filesystem::remove(lifecycle_path);
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", lifecycle_path.string());
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "block-input");
  std::exception_ptr probe_failure;
  std::thread probe([&] {
    try {
      (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                      2'000'000'000ULL);
    } catch (...) {
      probe_failure = std::current_exception();
    }
  });

  bool worker_started = false;
  const auto launch_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (std::chrono::steady_clock::now() < launch_deadline && !worker_started) {
    const auto events = read_events(lifecycle_path);
    worker_started = has_event(events, "worker");
    if (!worker_started) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  expect_true(worker_started, "pipe EOF regression observes the blocked worker");

  (void)::close(pipe_descriptors[1]);
  pipe_descriptors[1] = -1;
  pollfd read_end{.fd = pipe_descriptors[0], .events = POLLIN, .revents = 0};
  int poll_result = -1;
  do {
    poll_result = ::poll(&read_end, 1, 500);
  } while (poll_result < 0 && errno == EINTR);
  char byte = '\0';
  const auto read_result = poll_result > 0 ? ::read(pipe_descriptors[0], &byte, 1) : -1;
  expect_true(read_result == 0,
              "unrelated worker and guard descriptors do not keep a caller pipe open");
  (void)::close(pipe_descriptors[0]);
  probe.join();

  bool timed_out = false;
  try {
    if (probe_failure != nullptr) {
      std::rethrow_exception(probe_failure);
    }
  } catch (const GpuVideoProbeError& error) {
    timed_out = std::string_view(error.what()).find("timed out") != std::string_view::npos;
  } catch (...) {
  }
  expect_true(timed_out, "pipe EOF regression's blocked worker reaches its bounded timeout");
  set_environment("RECO_FAKE_PROBE_LIFECYCLE_PATH", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  std::filesystem::remove(lifecycle_path);
#else
  (void)video_path;
#endif
}

void unrelated_descriptors_are_not_inherited(const std::filesystem::path& video_path) {
#if defined(__linux__)
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".forbidden-descriptor");
  {
    std::ofstream marker(marker_path);
    marker << "descriptor marker";
  }
  const auto marker_descriptor = ::open(marker_path.c_str(), O_RDONLY);
  expect_true(marker_descriptor >= 0, "descriptor-isolation marker opens");
  if (marker_descriptor >= 0) {
    set_environment("RECO_FAKE_PROBE_FORBIDDEN_PATH", marker_path.string());
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "descriptor-isolation");
    expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        5'000'000'000ULL)
                  .width,
              854U, "probe worker closes unrelated inherited descriptors");
    (void)::close(marker_descriptor);
  }
  std::filesystem::remove(marker_path);
#else
  (void)video_path;
#endif
}

int run_registry_owner_signal_fork_fixture(std::string_view mode,
                                           int descendant_lifetime_descriptor) {
#if defined(_WIN32)
  return 0;
#else
  std::array<int, 2> ready{-1, -1};
  std::array<int, 2> release{-1, -1};
  std::array<int, 2> report{-1, -1};
  if (::pipe(ready.data()) != 0 || ::pipe(release.data()) != 0 || ::pipe(report.data()) != 0) {
    return 10;
  }
  struct sigaction action{};
  struct sigaction previous{};
  sigset_t signal_set{};
  sigset_t previous_mask{};
  if (sigemptyset(&signal_set) != 0 || sigaddset(&signal_set, SIGUSR1) != 0 ||
      ::pthread_sigmask(SIG_UNBLOCK, &signal_set, &previous_mask) != 0) {
    return 11;
  }
  action.sa_handler = fork_from_signal_handler;
  (void)sigemptyset(&action.sa_mask);
  if (::sigaction(SIGUSR1, &action, &previous) != 0) {
    (void)::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr);
    return 12;
  }
  signal_fork_report_descriptor = report[1];
  std::exception_ptr holder_error;
  std::thread holder([&] {
    try {
      reco::io::detail::hold_probe_fork_descriptor_registry_for_test(ready[1], release[0]);
      const char unlocked = 'U';
      ssize_t written = -1;
      do {
        written = ::write(ready[1], &unlocked, 1);
      } while (written < 0 && errno == EINTR);
      if (written != 1) {
        throw std::runtime_error("failed to report the unlocked probe descriptor registry");
      }
      char finish = '\0';
      ssize_t received = -1;
      do {
        received = ::read(release[0], &finish, 1);
      } while (received < 0 && errno == EINTR);
      if (received != 1 || finish != 'F') {
        throw std::runtime_error("failed to retain the registry owner after signal delivery");
      }
    } catch (...) {
      holder_error = std::current_exception();
    }
  });
  const auto read_marker = [](int descriptor, char expected) {
    pollfd pending{.fd = descriptor, .events = POLLIN, .revents = 0};
    if (::poll(&pending, 1, 2000) <= 0) {
      return false;
    }
    char marker = '\0';
    return ::read(descriptor, &marker, 1) == 1 && marker == expected;
  };
  char ready_marker = '\0';
  if (::read(ready[0], &ready_marker, 1) != 1 || ready_marker != 'R') {
    ::_exit(13);
  }
  if (::pthread_kill(holder.native_handle(), SIGUSR1) != 0) {
    ::_exit(14);
  }
  const char release_marker = 'R';
  if (::write(release[1], &release_marker, 1) != 1 || !read_marker(ready[0], 'U')) {
    ::_exit(15);
  }
  pollfd pending{.fd = report[0], .events = POLLIN, .revents = 0};
  const auto reported = ::poll(&pending, 1, 2000);
  pid_t fork_child = -1;
  if (reported > 0) {
    (void)::read(report[0], &fork_child, sizeof(fork_child));
  }
  const char finish_marker = 'F';
  (void)::write(release[1], &finish_marker, 1);
  holder.join();
  if (holder_error != nullptr || fork_child <= 0) {
    return 16;
  }
  if (fork_child > 0) {
    int status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(fork_child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != fork_child) {
      return 17;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
      return 19;
    }
  }
  signal_fork_report_descriptor = -1;
  if (::sigaction(SIGUSR1, &previous, nullptr) != 0 ||
      ::pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr) != 0) {
    return 18;
  }
  for (const auto descriptor : {ready[0], ready[1], release[0], release[1], report[0], report[1]}) {
    (void)::close(descriptor);
  }
  if (mode == "--fail-with-retained-descendant") {
    const auto descendant = ::fork();
    if (descendant == 0) {
      for (;;) {
        (void)::pause();
      }
    }
    (void)::close(descendant_lifetime_descriptor);
    return descendant > 0 ? 20 : 21;
  }
  if (mode == "--fail-after-signal-fork") {
    (void)::close(descendant_lifetime_descriptor);
    return 20;
  }
  if (mode != "--success") {
    (void)::close(descendant_lifetime_descriptor);
    return 22;
  }
  (void)::close(descendant_lifetime_descriptor);
  return 0;
#endif
}

void signal_handler_fork_cannot_deadlock_the_probe_registry(std::string_view fixture_mode = {}) {
#if !defined(_WIN32)
  struct sigaction default_child_disposition{};
  default_child_disposition.sa_handler = SIG_DFL;
  (void)sigemptyset(&default_child_disposition.sa_mask);
  struct sigaction previous_child_disposition{};
  if (::sigaction(SIGCHLD, &default_child_disposition, &previous_child_disposition) != 0) {
    expect_true(false, "probe owner-thread signal-fork fixture makes children waitable");
    return;
  }
  struct SigchldRestorer {
    ~SigchldRestorer() { (void)::sigaction(SIGCHLD, &previous, nullptr); }
    struct sigaction previous{};
  } sigchld_restorer{previous_child_disposition};

  const auto executable = test_executable_path.string();
  std::array<int, 2> group_ready{-1, -1};
  std::array<int, 2> descendant_lifetime{-1, -1};
  if (::pipe(group_ready.data()) != 0 || ::pipe(descendant_lifetime.data()) != 0) {
    for (const auto descriptor :
         {group_ready[0], group_ready[1], descendant_lifetime[0], descendant_lifetime[1]}) {
      if (descriptor >= 0) {
        (void)::close(descriptor);
      }
    }
    expect_true(false, "probe owner-thread signal-fork fixture creates its containment pipes");
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  const auto selected_fixture_mode =
      fixture_mode.empty() ? std::string("--success") : std::string(fixture_mode);
  const auto lifetime_descriptor_argument = std::to_string(descendant_lifetime[1]);
  std::array<char*, 5> arguments{
      const_cast<char*>(executable.c_str()),
      const_cast<char*>("--reco-registry-owner-signal-fork-fixture"),
      const_cast<char*>(selected_fixture_mode.c_str()),
      const_cast<char*>(lifetime_descriptor_argument.c_str()),
      nullptr,
  };
  const auto fixture = ::fork();
  if (fixture == 0) {
    (void)::close(group_ready[0]);
    (void)::close(descendant_lifetime[0]);
    const int group_error = ::setpgid(0, 0) == 0 ? 0 : errno;
    ssize_t written = -1;
    do {
      written = ::write(group_ready[1], &group_error, sizeof(group_error));
    } while (written < 0 && errno == EINTR);
    (void)::close(group_ready[1]);
    if (group_error != 0 || written != static_cast<ssize_t>(sizeof(group_error))) {
      ::_exit(126);
    }
    ::execv(arguments[0], arguments.data());
    ::_exit(127);
  }
  (void)::close(group_ready[1]);
  (void)::close(descendant_lifetime[1]);
  expect_true(fixture > 0, "probe owner-thread signal-fork fixture starts");
  if (fixture <= 0) {
    (void)::close(group_ready[0]);
    (void)::close(descendant_lifetime[0]);
    return;
  }
  bool parent_established_group = ::setpgid(fixture, fixture) == 0;
  if (!parent_established_group && errno == EACCES) {
    parent_established_group = ::getpgid(fixture) == fixture;
  }
  const auto kill_fixture_group = [fixture] { (void)::kill(-fixture, SIGKILL); };
  const auto terminate_fixture = [fixture, &kill_fixture_group] {
    kill_fixture_group();
    (void)::kill(fixture, SIGKILL);
  };
  const auto reap_fixture = [fixture](int* status) {
    pid_t waited = -1;
    do {
      waited = ::waitpid(fixture, status, 0);
    } while (waited < 0 && errno == EINTR);
    return waited;
  };
  const auto wait_for_descendant_lifetime_eof = [&descendant_lifetime] {
    const auto eof_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < eof_deadline) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          eof_deadline - std::chrono::steady_clock::now());
      pollfd pending{.fd = descendant_lifetime[0], .events = POLLIN, .revents = 0};
      const auto polled = ::poll(&pending, 1, std::max(1, static_cast<int>(remaining.count())));
      if (polled < 0 && errno == EINTR) {
        continue;
      }
      if (polled <= 0) {
        return false;
      }
      char marker = '\0';
      ssize_t received = -1;
      do {
        received = ::read(descendant_lifetime[0], &marker, 1);
      } while (received < 0 && errno == EINTR);
      return received == 0;
    }
    return false;
  };
  int group_error = EIO;
  ssize_t received = -1;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    pollfd pending{.fd = group_ready[0], .events = POLLIN, .revents = 0};
    const auto polled = ::poll(&pending, 1, std::max(1, static_cast<int>(remaining.count())));
    if (polled < 0 && errno == EINTR) {
      continue;
    }
    if (polled <= 0) {
      break;
    }
    do {
      received = ::read(group_ready[0], &group_error, sizeof(group_error));
    } while (received < 0 && errno == EINTR);
    break;
  }
  (void)::close(group_ready[0]);
  const bool private_group_ready = parent_established_group &&
                                   received == static_cast<ssize_t>(sizeof(group_error)) &&
                                   group_error == 0;
  expect_true(private_group_ready,
              "probe owner-thread signal-fork fixture acknowledges its private process group");
  if (!private_group_ready) {
    terminate_fixture();
    (void)wait_for_descendant_lifetime_eof();
    (void)reap_fixture(nullptr);
    (void)::close(descendant_lifetime[0]);
    return;
  }
  siginfo_t observed_status{};
  bool fixture_observed = false;
  int observation_error = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    observed_status = {};
    if (::waitid(P_PID, static_cast<id_t>(fixture), &observed_status,
                 WEXITED | WNOHANG | WNOWAIT) == 0) {
      if (observed_status.si_pid == fixture) {
        fixture_observed = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    if (errno == EINTR) {
      continue;
    }
    observation_error = errno;
    break;
  }

  int status = 0;
  pid_t waited = -1;
  bool descendants_terminated = false;
  if (!fixture_observed && observation_error == 0) {
    terminate_fixture();
    descendants_terminated = wait_for_descendant_lifetime_eof();
    waited = reap_fixture(&status);
  } else if (!fixture_observed) {
    if (observation_error != ECHILD) {
      terminate_fixture();
      (void)wait_for_descendant_lifetime_eof();
      (void)reap_fixture(nullptr);
    }
    (void)::close(descendant_lifetime[0]);
    expect_true(false, "probe owner-thread signal-fork fixture remains waitable");
    return;
  } else {
    const bool observed_success =
        observed_status.si_code == CLD_EXITED && observed_status.si_status == EXIT_SUCCESS;
    if (!observed_success) {
      kill_fixture_group();
      descendants_terminated = wait_for_descendant_lifetime_eof();
    }
    waited = reap_fixture(&status);
  }
  (void)::close(descendant_lifetime[0]);
  if (waited != fixture) {
    expect_true(false, "probe owner-thread signal-fork fixture is reaped");
    return;
  }
  const bool fixture_succeeded = WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS;
  if (fixture_mode.empty()) {
    expect_true(fixture_succeeded,
                "probe registry owner defers signal-handler fork until after unlock");
    return;
  }
  const bool expected_failure = WIFEXITED(status) && WEXITSTATUS(status) == 20;
  expect_true(expected_failure, "probe signal-fork failure fixture reaches its forced outcome");
  expect_true(descendants_terminated,
              "probe signal-fork failure closes every inherited descendant lifetime");
#endif
}

void ignored_sigchld_cannot_auto_reap_the_signal_fork_fixture() {
#if !defined(_WIN32)
  struct sigaction ignored{};
  ignored.sa_handler = SIG_IGN;
  (void)sigemptyset(&ignored.sa_mask);
  struct sigaction previous{};
  if (::sigaction(SIGCHLD, &ignored, &previous) != 0) {
    expect_true(false, "probe signal-fork regression ignores SIGCHLD");
    return;
  }
  signal_handler_fork_cannot_deadlock_the_probe_registry();
  struct sigaction restored{};
  const bool disposition_restored =
      ::sigaction(SIGCHLD, nullptr, &restored) == 0 && restored.sa_handler == SIG_IGN;
  (void)::sigaction(SIGCHLD, &previous, nullptr);
  expect_true(disposition_restored,
              "probe signal-fork fixture restores an ignored SIGCHLD disposition");
#endif
}

void failed_signal_fork_fixtures_are_terminated_before_leader_reap() {
#if !defined(_WIN32)
  signal_handler_fork_cannot_deadlock_the_probe_registry("--fail-after-signal-fork");
  signal_handler_fork_cannot_deadlock_the_probe_registry("--fail-with-retained-descendant");
#endif
}

void worker_main_preserves_a_reused_executable_descriptor(const std::filesystem::path& video_path) {
#if defined(RECO_PROBE_TEST_FORCE_GUARDIAN_FALLBACKS) && !defined(RECO_PROBE_THREAD_SANITIZER) &&  \
    !defined(_WIN32)
  const auto audit_path =
      video_path.parent_path() / (video_path.filename().string() + ".worker-fd-reuse");
  std::filesystem::remove(audit_path);
  set_environment("RECO_FAKE_PROBE_WORKER_DESCRIPTOR_REUSE_PATH", audit_path.string());
  set_scenario("probe-ok");
  try {
    expect_eq(probe_video(container_config(video_path), 5'000'000'000ULL).width, 3840U,
              "non-TSan worker preserves a descriptor reused after exec");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: descriptor-reuse worker threw: " << error.what() << '\n';
    ++failures;
  }
  set_environment("RECO_FAKE_PROBE_WORKER_DESCRIPTOR_REUSE_PATH", "");
  std::ifstream audit(audit_path);
  char marker = '\0';
  expect_true(audit >> marker && marker == '1',
              "worker main does not close a constructor-owned descriptor 3");
  std::filesystem::remove(audit_path);
#else
  (void)video_path;
#endif
}

void guardian_initializers_cannot_observe_unrelated_descriptors(
    const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  const auto leak_path =
      video_path.parent_path() / (video_path.filename().string() + ".pre-main-descriptor-leak");
  std::filesystem::remove(leak_path);
  const auto base_descriptor = ::open("/dev/null", O_RDONLY);
  expect_true(base_descriptor >= 0, "pre-main descriptor-isolation source opens");
  const auto forbidden_descriptor =
      base_descriptor >= 0 ? ::fcntl(base_descriptor, F_DUPFD, 256) : -1;
  expect_true(forbidden_descriptor >= 256, "pre-main descriptor-isolation high descriptor opens");
  if (base_descriptor >= 0) {
    (void)::close(base_descriptor);
  }
  if (forbidden_descriptor >= 256) {
    set_environment("RECO_FAKE_PROBE_PRE_MAIN_FORBIDDEN_FD", std::to_string(forbidden_descriptor));
    set_environment("RECO_FAKE_PROBE_PRE_MAIN_DESCRIPTOR_LEAK_PATH", leak_path.string());
    set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
    try {
      expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                          5'000'000'000ULL)
                    .width,
                854U, "guardian starts with isolated descriptors");
    } catch (const std::exception& error) {
      std::cerr << "FAIL: pre-main guardian descriptor isolation threw: " << error.what() << '\n';
      ++failures;
    }
    expect_true(!std::filesystem::exists(leak_path),
                "guardian pre-main initializer cannot observe an unrelated descriptor");
    (void)::close(forbidden_descriptor);
  }
  set_environment("RECO_FAKE_PROBE_PRE_MAIN_FORBIDDEN_FD", "");
  set_environment("RECO_FAKE_PROBE_PRE_MAIN_DESCRIPTOR_LEAK_PATH", "");
  std::filesystem::remove(leak_path);
#else
  (void)video_path;
#endif
}

void lowered_file_limit_does_not_leak_high_descriptors(const std::filesystem::path& video_path) {
#if defined(__APPLE__)
  const auto marker_path =
      video_path.parent_path() / (video_path.filename().string() + ".high-descriptor");
  const auto marker = ::open(marker_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  expect_true(marker >= 0, "high-descriptor marker opens");
  if (marker < 0) {
    return;
  }
  const auto high_descriptor = ::fcntl(marker, F_DUPFD, 512);
  expect_true(high_descriptor >= 512, "high descriptor opens before lowering RLIMIT_NOFILE");
  struct rlimit original{};
  const bool have_limit = ::getrlimit(RLIMIT_NOFILE, &original) == 0;
  expect_true(have_limit, "RLIMIT_NOFILE is readable");
  if (high_descriptor >= 512 && have_limit) {
    const struct rlimit lowered{.rlim_cur = std::min<rlim_t>(64, original.rlim_max),
                                .rlim_max = original.rlim_max};
    const bool lowered_ok = ::setrlimit(RLIMIT_NOFILE, &lowered) == 0;
    expect_true(lowered_ok, "RLIMIT_NOFILE soft limit is lowered");
    if (lowered_ok) {
      set_environment("RECO_FAKE_PROBE_FORBIDDEN_FD", std::to_string(high_descriptor));
      set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "descriptor-isolation");
      try {
        expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                            5'000'000'000ULL)
                      .width,
                  854U, "macOS worker closes descriptors above the lowered file limit");
      } catch (const std::exception& error) {
        std::cerr << "FAIL: lowered-limit descriptor isolation threw: " << error.what() << '\n';
        ++failures;
      }
      (void)::setrlimit(RLIMIT_NOFILE, &original);
    }
  }
  set_environment("RECO_FAKE_PROBE_FORBIDDEN_FD", "");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
  if (high_descriptor >= 0) {
    (void)::close(high_descriptor);
  }
  (void)::close(marker);
  std::filesystem::remove(marker_path);
#else
  (void)video_path;
#endif
}

void worker_address_space_is_limited(const std::filesystem::path& video_path) {
#if defined(__APPLE__) && !defined(RECO_PROBE_WIDE_ADDRESS_SANITIZER)
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "memory-over-limit");
  expect_probe_error(
      [&] {
        (void)reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        5'000'000'000ULL);
      },
      "video probe worker exited abnormally",
      "macOS watchdog-first memory termination reports the worker status");
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#elif !defined(_WIN32) && !defined(RECO_PROBE_WIDE_ADDRESS_SANITIZER)
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "memory-limit");
  try {
    expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        5'000'000'000ULL)
                  .width,
              854U, "POSIX worker inherits the bounded address-space limit");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: address-space limit probe threw: " << error.what() << '\n';
    ++failures;
  }
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#else
  (void)video_path;
#endif
}

void worker_process_creation_is_confined(const std::filesystem::path& video_path) {
#if !defined(_WIN32)
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "process-spawn-denied");
  try {
    expect_eq(reco::io::probe_gpu_video(container_config(video_path), fake_probe_worker_path,
                                        5'000'000'000ULL)
                  .width,
              854U, "POSIX worker denies child processes while retaining thread support");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: process-confinement probe threw: " << error.what() << '\n';
    ++failures;
  }
  set_environment("RECO_FAKE_PROBE_WORKER_SCENARIO", "valid-metadata");
#else
  (void)video_path;
#endif
}

void worker_rejects_x32_syscall_namespace() {
#if defined(__linux__) && defined(__x86_64__)
  expect_true(reco::io::detail::probe_worker_rejects_x32_syscalls_for_test(),
              "x86-64 worker kills x32-tagged syscalls before syscall dispatch");
#endif
}

void watchdog_exit_after_memory_termination_is_expected() {
#if !defined(_WIN32)
  constexpr std::int64_t watchdog_pid = 73;
  expect_true(reco::io::detail::guardian_watchdog_exit_is_fatal_for_test(false, 0, 0, watchdog_pid,
                                                                         watchdog_pid),
              "watchdog exit before memory termination is fatal");
  expect_true(!reco::io::detail::guardian_watchdog_exit_is_fatal_for_test(true, 0, 0, watchdog_pid,
                                                                          watchdog_pid),
              "watchdog-first exit after memory termination is expected");
  expect_true(!reco::io::detail::guardian_watchdog_exit_is_fatal_for_test(true, -1, ECHILD, 0,
                                                                          watchdog_pid),
              "reaped watchdog after memory termination is expected");
#endif
}

void non_utf8_path_round_trips() {
#if defined(__linux__)
  auto filename = std::string("reco_gpu_probe_non_utf8_");
  filename.push_back(static_cast<char>(0xFF));
  filename += ".mp4";
  const auto path = std::filesystem::temp_directory_path() / filename;
  {
    std::ofstream video(path, std::ios::binary);
    video << "fake video container";
  }
  set_scenario("probe-ok");
  expect_eq(probe_video(container_config(path), 5'000'000'000ULL).width, 3840U,
            "non-UTF-8 POSIX path survives probe worker IPC");
  std::filesystem::remove(path);
#endif
}

void windows_unicode_path_round_trips() {
#if defined(_WIN32)
  const auto filename = std::u8string(u8"reco_gpu_probe_\u5f55\u50cf.mp4");
  const auto path = std::filesystem::temp_directory_path() / std::filesystem::path(filename);
  {
    std::ofstream video(path, std::ios::binary);
    video << "fake video container";
  }
  set_scenario("probe-ok");
  expect_eq(probe_video(container_config(path), 5'000'000'000ULL).width, 3840U,
            "UTF-8 Windows path survives probe worker validation and pipeline construction");
  std::filesystem::remove(path);
#endif
}

void windows_path_runtime_discovery(const std::filesystem::path& video_path,
                                    const std::filesystem::path& runtime) {
#if defined(_WIN32)
  const auto directory = std::filesystem::temp_directory_path() /
                         ("reco_gstreamer_path_" + std::to_string(GetCurrentProcessId()));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  for (const auto* name :
       {"gstreamer-1.0-0.dll", "gstapp-1.0-0.dll", "libglib-2.0-0.dll", "libgobject-2.0-0.dll"}) {
    std::filesystem::copy_file(runtime, directory / name,
                               std::filesystem::copy_options::overwrite_existing);
  }
  const char* current_path = std::getenv("PATH");
  const std::string original_path = current_path == nullptr ? "" : current_path;
  set_environment("PATH", directory.string() + ";" + original_path);
  set_environment("RECO_GSTREAMER_DYLIB_PATH", "");
  set_environment("RECO_GSTAPP_DYLIB_PATH", "");
  set_environment("RECO_GLIB_DYLIB_PATH", "");
  set_environment("RECO_GOBJECT_DYLIB_PATH", "");
  try {
    set_scenario("probe-ok");
    expect_eq(probe_video(container_config(video_path), 5'000'000'000ULL).width, 3840U,
              "Windows probe resolves conventional GStreamer DLLs from PATH securely");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: Windows PATH GStreamer discovery threw: " << error.what() << '\n';
    ++failures;
  }
  set_environment("PATH", original_path);
  set_environment("RECO_GSTREAMER_DYLIB_PATH", runtime.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", runtime.string());
  set_environment("RECO_GLIB_DYLIB_PATH", runtime.string());
  set_environment("RECO_GOBJECT_DYLIB_PATH", runtime.string());
  std::filesystem::remove_all(directory);
#else
  (void)video_path;
  (void)runtime;
#endif
}

void parent_death_reclaims_worker(const std::filesystem::path& video_path) {
#if defined(__linux__)
  set_scenario("probe-blocking-playing");
  const auto caller =
      fork_exec_probe_caller(video_path, probe_worker_path, "probe", 5'000'000'000ULL);
  expect_true(caller > 0, "parent-death probe caller starts");
  if (caller <= 0) {
    return;
  }

  const auto discovery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  const auto supervisor =
      wait_for_descendant(caller, discovery_deadline, "--reco-video-probe-supervisor");
  const auto guardian = supervisor.has_value()
                            ? wait_for_direct_child(*supervisor, discovery_deadline)
                            : std::nullopt;
  const auto worker =
      guardian.has_value() ? wait_for_direct_child(*guardian, discovery_deadline) : std::nullopt;
  expect_true(supervisor.has_value(), "parent-death probe supervisor starts");
  expect_true(guardian.has_value(), "parent-death probe guardian starts");
  expect_true(worker.has_value(), "isolated worker starts before parent-death test");
  (void)kill(caller, SIGKILL);
  int caller_status = 0;
  while (waitpid(caller, &caller_status, 0) < 0 && errno == EINTR) {
  }

  if (worker.has_value()) {
    const auto exit_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < exit_deadline &&
           (kill(*worker, 0) == 0 || errno != ESRCH)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect_true(kill(*worker, 0) != 0 && errno == ESRCH,
                "worker process group exits when its parent dies");
  }
  set_scenario("probe-ok");
#else
  (void)video_path;
#endif
}

void expect_no_unreaped_children() {
#if !defined(_WIN32)
  int status = 0;
  errno = 0;
  const auto child = waitpid(-1, &status, WNOHANG);
  expect_true(child == -1 && errno == ECHILD, "all probe workers are reaped before API return");
#endif
}

} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  if (argc == 2 && std::string_view(argv[1]) == "--reco-parent-death-probe-caller") {
    return run_windows_parent_death_probe_caller();
  }
#else
  try {
    test_executable_path = std::filesystem::absolute(argv[0]);
  } catch (...) {
    return EXIT_FAILURE;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--reco-posix-probe-caller") {
    return run_posix_probe_caller();
  }
  if (argc == 4 && std::string_view(argv[1]) == "--reco-registry-owner-signal-fork-fixture") {
    int lifetime_descriptor = -1;
    const std::string_view descriptor_argument(argv[3]);
    const auto [end, error] = std::from_chars(
        descriptor_argument.data(), descriptor_argument.data() + descriptor_argument.size(),
        lifetime_descriptor);
    if (error != std::errc{} || end != descriptor_argument.data() + descriptor_argument.size() ||
        lifetime_descriptor < 0) {
      return EXIT_FAILURE;
    }
    return run_registry_owner_signal_fork_fixture(argv[2], lifetime_descriptor);
  }
#endif
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  const auto runtime = resolve_runfile("cpp/tests/libfake_gstreamer_runtime.so");
#if defined(RECO_PROBE_TEST_FORCE_GUARDIAN_FALLBACKS)
  probe_worker_path = executable_runfile("cpp/reco_io/reco_video_probe_worker_fallback");
  fake_probe_worker_path = executable_runfile("cpp/tests/fake_video_probe_worker_fallback");
#else
  probe_worker_path = executable_runfile("cpp/reco_io/reco_video_probe_worker");
  fake_probe_worker_path = executable_runfile("cpp/tests/fake_video_probe_worker");
#endif
  set_environment("RECO_GSTREAMER_DYLIB_PATH", runtime.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", runtime.string());
  set_environment("RECO_GLIB_DYLIB_PATH", runtime.string());
  set_environment("RECO_GOBJECT_DYLIB_PATH", runtime.string());
#if defined(__linux__)
  const auto nvbufsurface = resolve_runfile("cpp/tests/libfake_nvbufsurface.so");
  set_environment("RECO_NVBUFSURFACE_DYLIB_PATH", nvbufsurface.string());
  set_environment("RECO_NVDS_UTILS_DYLIB_PATH", nvbufsurface.string());
#endif

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto video_path = std::filesystem::temp_directory_path() /
                          ("reco_gpu_probe_" + std::to_string(unique) + ".mp4");
  const auto event_path = std::filesystem::temp_directory_path() /
                          ("reco_gpu_probe_events_" + std::to_string(unique) + ".txt");
  {
    std::ofstream video(video_path, std::ios::binary);
    video << "fake video container";
  }
  set_environment("RECO_FAKE_GST_EVENT_PATH", event_path.string());

  probe_contracts(video_path, event_path);
#if !defined(RECO_PROBE_TEST_FORCE_GUARDIAN_FALLBACKS)
  exhaustive_calibration_probe_scans_to_eos(video_path);
#endif
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "probe contracts leave no aggregate admission behind");
  invalid_inputs_fail(video_path, event_path);
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "invalid inputs leave no aggregate admission behind");
  non_utf8_path_round_trips();
  windows_unicode_path_round_trips();
  windows_path_runtime_discovery(video_path, runtime);
  expect_eq(reco::io::detail::reserved_probe_worker_address_space_bytes_for_test(), 0ULL,
            "path tests leave no aggregate admission behind");
  worker_ipc_failures_are_bounded(video_path);
  aggregate_worker_memory_budget_is_enforced();
  maximum_linux_snapshots_are_aggregate_bounded();
  inherited_probe_state_is_rejected_after_fork(video_path);
  deferred_cleanup_retains_worker_memory_reservation(video_path);
  delayed_supervision_cannot_launch_worker(video_path);
  launch_gate_prevents_post_timeout_process_start(video_path);
  guardian_death_before_pid_report_reclaims_worker(video_path);
  auto_reaped_workers_are_supported(video_path);
  no_child_wait_workers_are_supported(video_path);
  post_admission_sigchld_change_is_supported(video_path);
  owner_pre_main_startup_uses_probe_deadline(video_path);
  partial_owner_launch_report_respects_probe_deadline(video_path);
  competing_waitpid_reaper_cannot_steal_cleanup_authority(video_path);
  windows_job_reclaims_worker_descendants(video_path);
  guardian_death_after_worker_release_reclaims_group(video_path);
  unrelated_descriptors_are_not_inherited(video_path);
  signal_handler_fork_cannot_deadlock_the_probe_registry();
  ignored_sigchld_cannot_auto_reap_the_signal_fork_fixture();
  failed_signal_fork_fixtures_are_terminated_before_leader_reap();
  worker_main_preserves_a_reused_executable_descriptor(video_path);
  guardian_initializers_cannot_observe_unrelated_descriptors(video_path);
  lowered_file_limit_does_not_leak_high_descriptors(video_path);
  worker_address_space_is_limited(video_path);
  worker_process_creation_is_confined(video_path);
  worker_rejects_x32_syscall_namespace();
  watchdog_exit_after_memory_termination_is_expected();
  unrelated_descriptor_writer_does_not_delay_pipe_eof(video_path);
  parent_death_reclaims_worker(video_path);
  caller_death_reclaims_worker_and_descendant(video_path);
  caller_death_before_supervisor_arm(video_path);
  caller_death_before_supervisor_main(video_path);
  caller_process_group_death_before_supervisor_main(video_path);
  mac_owner_reclaims_stalled_guardian_session(video_path);
  executable_replacement_cannot_change_the_pinned_probe_image(video_path);
  linux_fork_child_does_not_retain_snapshot_memfd(video_path);
  mac_probe_snapshot_preserves_quarantine(video_path);
  mac_snapshot_normal_cleanup_ignores_fork_only_children(video_path);
  mac_maximum_concurrent_probes_fit_descriptor_registry(video_path);
  mac_snapshot_helper_survives_owner_process_group_termination(video_path);
  mac_probe_snapshot_cleanup_tracks_owner_process(video_path);
  caller_death_before_guardian_main(video_path);
  caller_death_before_worker_main(video_path);
  pre_main_loader_stall_respects_timeout(video_path);
  killed_cleanup_authority_fails_closed(video_path);
  dead_watchdog_cannot_release_the_supervisor(video_path);
  expect_no_unreaped_children();

  std::filesystem::remove(video_path);
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
