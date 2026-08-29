#include "reco/calibrate/pipeline.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <unistd.h>
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
#endif

using namespace reco::calibrate;

namespace {

int failures = 0;
std::filesystem::path fake_worker;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
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
void expect_execution_error(Function&& function, std::string_view needle,
                            std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const CalibrationExecutionError& error) {
    if (std::string_view(error.what()).find(needle) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " error=" << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected error=" << error.what() << '\n';
    ++failures;
  }
}

template <typename Function> void run_case(std::string_view name, Function&& function) {
  try {
    function();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << name << " threw: " << error.what() << '\n';
    ++failures;
  } catch (...) {
    std::cerr << "FAIL: " << name << " threw an unknown exception\n";
    ++failures;
  }
}

std::filesystem::path executable_runfile(std::string path) {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
#if defined(_WIN32)
  path += ".exe";
#endif
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
  const auto resolved =
      std::filesystem::path(runfiles->Rlocation(std::string(workspace) + "/" + path));
  if (resolved.empty() || !std::filesystem::is_regular_file(resolved)) {
    throw std::runtime_error(path + " runfile not found");
  }
  return resolved;
}

GpuCalibrationRequest request_fixture() {
  GpuCalibrationRequest request;
  request.left.path = "left.mp4";
  request.left.retained_path = fake_worker.string();
  request.left.lens_profile = fake_worker.string();
  request.right.path = "right.mp4";
  request.right.retained_path = fake_worker.string();
  request.config.num_frames = 2;
  request.no_auto_imu = true;
  request.auto_sync = false;
  request.output = "match.json";
  request.probe_worker = "/workers/reco_video_probe_worker";
  request.probe_timeout_ns = 1'000'000'000ULL;
  request.calibration_worker_path = fake_worker.string();
  request.calibration_timeout_ns = 1'000'000'000ULL;
#if defined(RECO_CALIBRATION_WIDE_ADDRESS_SANITIZER)
  request.calibration_host_memory_limit_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
#else
  request.calibration_host_memory_limit_bytes = 256ULL * 1024ULL * 1024ULL;
#endif
  return request;
}

GpuCalibrationRequest lifecycle_request_fixture() {
  auto request = request_fixture();
  request.calibration_timeout_ns = 3'000'000'000ULL;
  return request;
}

CalibrationBackendStatus ready_backends() {
  return {.cuda = {.available = true, .detail = "cuda"},
          .gstreamer = {.available = true, .detail = "gstreamer"},
          .npp = {.available = true, .detail = "npp"},
          .nvbufsurface = {.available = true, .detail = "nvbufsurface"}};
}

#if defined(__linux__)
class Scenario {
public:
  explicit Scenario(std::string_view value) {
    if (::setenv("RECO_FAKE_CALIBRATION_WORKER_SCENARIO", std::string(value).c_str(), 1) != 0) {
      throw std::runtime_error("cannot set fake worker scenario");
    }
  }
  ~Scenario() { (void)::unsetenv("RECO_FAKE_CALIBRATION_WORKER_SCENARIO"); }
};

class EnvironmentValue {
public:
  EnvironmentValue(std::string name, std::string_view value) : name_(std::move(name)) {
    if (const char* previous = std::getenv(name_.c_str()); previous != nullptr) {
      previous_ = previous;
    }
    if (::setenv(name_.c_str(), std::string(value).c_str(), 1) != 0) {
      throw std::runtime_error("cannot set test environment value");
    }
  }
  EnvironmentValue(const EnvironmentValue&) = delete;
  EnvironmentValue& operator=(const EnvironmentValue&) = delete;
  ~EnvironmentValue() {
    if (previous_.has_value()) {
      (void)::setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      (void)::unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

std::filesystem::path temporary_path(std::string_view suffix) {
  return std::filesystem::temp_directory_path() /
         ("reco-calibration-supervisor-" + std::to_string(::getpid()) + "-" + std::string(suffix));
}

bool process_exists(pid_t process) {
  errno = 0;
  return ::kill(process, 0) == 0 || errno == EPERM;
}

std::set<std::filesystem::path> calibration_cgroups() {
  std::ifstream membership("/proc/self/cgroup");
  std::string line;
  std::filesystem::path current;
  while (std::getline(membership, line)) {
    if (line.rfind("0::", 0) == 0) {
      current = std::filesystem::path("/sys/fs/cgroup") /
                std::filesystem::path(line.substr(3)).relative_path();
      break;
    }
  }

  std::set<std::filesystem::path> result;
  const std::filesystem::path root("/sys/fs/cgroup");
  for (auto candidate = current; !candidate.empty(); candidate = candidate.parent_path()) {
    std::error_code error;
    for (std::filesystem::directory_iterator entries(candidate, error), end;
         !error && entries != end; entries.increment(error)) {
      if (entries->is_directory(error) &&
          entries->path().filename().string().rfind("reco-calibration-", 0) == 0) {
        result.insert(entries->path());
      }
    }
    if (candidate == root) {
      break;
    }
  }
  return result;
}

#if !defined(RECO_CALIBRATION_WIDE_ADDRESS_SANITIZER)
std::optional<std::filesystem::path> current_cgroup() {
  std::ifstream membership("/proc/self/cgroup");
  std::string line;
  while (std::getline(membership, line)) {
    if (line.rfind("0::", 0) == 0) {
      return std::filesystem::path("/sys/fs/cgroup") /
             std::filesystem::path(line.substr(3)).relative_path();
    }
  }
  return std::nullopt;
}
#endif

bool wait_for_cgroup_set(const std::set<std::filesystem::path>& expected,
                         std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (calibration_cgroups() == expected) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  } while (std::chrono::steady_clock::now() < deadline);
  return calibration_cgroups() == expected;
}

std::optional<pid_t> process_parent(pid_t process) {
  std::ifstream input(std::filesystem::path("/proc") / std::to_string(process) / "status");
  std::string key;
  while (input >> key) {
    if (key == "PPid:") {
      pid_t parent = -1;
      return input >> parent && parent > 1 ? std::optional<pid_t>(parent) : std::nullopt;
    }
    std::string ignored;
    std::getline(input, ignored);
  }
  return std::nullopt;
}

std::optional<pid_t> wait_for_pid_marker(const std::filesystem::path& path,
                                         std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream input(path);
    pid_t process = -1;
    if (input >> process && process > 1) {
      return process;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return std::nullopt;
}

void wait_for_process_removal(pid_t process, std::string_view message) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (process_exists(process) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  expect_true(!process_exists(process), message);
}

void success_returns_the_bounded_result() {
  Scenario scenario("success");
  const auto result = run_gpu_calibration(request_fixture(), ready_backends());
  expect_eq(result.total_matches, 12U, "supervisor transfers result totals");
  expect_eq(result.frames_used, 1U, "supervisor transfers used frame count");
  expect_eq(result.per_frame.size(), 1U, "supervisor transfers one frame summary");
  expect_eq(result.per_frame[0].points.size(), 12U,
            "supervisor transfers each reported correspondence");
}

void retained_input_descriptors_reach_the_sandboxed_worker() {
  const auto left = temporary_path("retained-left.mp4");
  const auto right = temporary_path("retained-right.mp4");
  const auto left_profile = temporary_path("retained-left-profile.json");
  const auto right_profile = temporary_path("retained-right-profile.json");
  {
    std::ofstream output(left, std::ios::binary | std::ios::trunc);
    output << "retained left input\n";
  }
  {
    std::ofstream output(right, std::ios::binary | std::ios::trunc);
    output << "retained right input\n";
  }
  {
    std::ofstream output(left_profile, std::ios::binary | std::ios::trunc);
    output << "retained left profile\n";
  }
  {
    std::ofstream output(right_profile, std::ios::binary | std::ios::trunc);
    output << "retained right profile\n";
  }
  auto request = request_fixture();
  request.left.retained_path = left.string();
  request.right.retained_path = right.string();
  request.left.lens_profile = left_profile.string();
  request.right.lens_profile = right_profile.string();
  {
    Scenario scenario("retained-inputs");
    expect_eq(run_gpu_calibration(request, ready_backends()).total_matches, 12U,
              "guardian transfers retained media descriptors to the sandboxed worker");
  }
  request.left.path = left.string();
  request.right.path = right.string();
  request.left.retained_path.reset();
  request.right.retained_path.reset();
  {
    Scenario scenario("retained-inputs");
    expect_eq(run_gpu_calibration(request, ready_backends()).total_matches, 12U,
              "public calibration API pins original media paths before sandboxing");
  }
  {
    auto metadata_request = request;
    metadata_request.right.lens_profile.reset();
    Scenario scenario("profile-metadata");
    const auto result = run_gpu_calibration(metadata_request, ready_backends());
    expect_true(result.left_lens_profile.has_value() &&
                    result.left_lens_profile->path == metadata_request.left.lens_profile,
                "public calibration API restores the selected left profile path");
    expect_true(result.right_lens_profile.has_value() &&
                    result.right_lens_profile->path == metadata_request.left.lens_profile,
                "public calibration API restores the fallback right profile path");
  }
  request.left.expected_identity = CalibrationFileIdentity{};
  expect_execution_error([&] { (void)run_gpu_calibration(request, ready_backends()); },
                         "left calibration video changed",
                         "public calibration API honors an expected original-path identity");
  std::filesystem::remove(left);
  std::filesystem::remove(right);
  std::filesystem::remove(left_profile);
  std::filesystem::remove(right_profile);
}

void input_mutation_is_rejected() {
  const auto left = temporary_path("mutated-left.mp4");
  const auto right = temporary_path("mutated-right.mp4");
  const auto profile = temporary_path("mutated-profile.json");
  const auto marker = temporary_path("mutation-guardian-ready");
  const auto write_fixture = [](const std::filesystem::path& path, std::string_view contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
  };
  write_fixture(left, "original left input\n");
  write_fixture(right, "original right input\n");
  write_fixture(profile, "original lens profile\n");

  const auto run_mutation = [&](const auto& mutate, std::string_view expected_label,
                                std::string_view expectation) {
    std::filesystem::remove(marker);
    auto request = lifecycle_request_fixture();
    request.left.path = left.string();
    request.right.path = right.string();
    request.left.retained_path.reset();
    request.right.retained_path.reset();
    request.left.lens_profile = profile.string();
    std::exception_ptr error;
    {
      EnvironmentValue marker_path("RECO_FAKE_CALIBRATION_GUARDIAN_READY_PATH", marker.string());
      EnvironmentValue guardian_delay("RECO_FAKE_CALIBRATION_GUARDIAN_DELAY_MS", "250");
      Scenario scenario("success");
      std::thread calibration([&] {
        try {
          (void)run_gpu_calibration(request, ready_backends());
        } catch (...) {
          error = std::current_exception();
        }
      });
      const auto guardian = wait_for_pid_marker(marker, std::chrono::seconds(1));
      expect_true(guardian.has_value(), "guardian reports readiness before input mutation");
      if (guardian.has_value()) {
        mutate();
      }
      calibration.join();
    }
    bool rejected = false;
    try {
      if (error != nullptr) {
        std::rethrow_exception(error);
      }
    } catch (const CalibrationExecutionError& failure) {
      rejected = std::string_view(failure.what()).find(expected_label) != std::string_view::npos;
    }
    expect_true(rejected, expectation);
  };

  run_mutation([&] { write_fixture(left, "mutated left input with a different identity\n"); },
               "left calibration video",
               "in-place media mutation is rejected after descriptor pinning");
  write_fixture(left, "restored left input\n");
  run_mutation([&] { write_fixture(profile, "mutated lens profile with a different identity\n"); },
               "left lens profile",
               "in-place profile mutation is rejected after descriptor pinning");
  write_fixture(profile, "restored lens profile\n");

  const auto moved_left = temporary_path("mutated-left-original.mp4");
  std::filesystem::remove(moved_left);
  run_mutation(
      [&] {
        std::filesystem::rename(left, moved_left);
        write_fixture(left, "replacement left input\n");
      },
      "left calibration video", "media path replacement is rejected after descriptor pinning");
  std::filesystem::remove(left);
  std::filesystem::rename(moved_left, left);

  const auto moved_profile = temporary_path("mutated-profile-original.json");
  std::filesystem::remove(moved_profile);
  run_mutation(
      [&] {
        std::filesystem::rename(profile, moved_profile);
        write_fixture(profile, "replacement lens profile\n");
      },
      "left lens profile", "profile path replacement is rejected after descriptor pinning");
  std::filesystem::remove(profile);
  std::filesystem::rename(moved_profile, profile);

  std::filesystem::remove(left);
  std::filesystem::remove(right);
  std::filesystem::remove(profile);
  std::filesystem::remove(marker);
}

void native_stdout_noise_does_not_corrupt_protocol() {
  Scenario scenario("stdout-noise");
  const auto result = run_gpu_calibration(request_fixture(), ready_backends());
  expect_eq(result.total_matches, 12U,
            "native stdout noise remains separate from the protocol descriptor");
}

void delayed_worker_request_io_obeys_the_deadline() {
  (void)::setenv("RECO_FAKE_CALIBRATION_PRE_REQUEST_DELAY_MS", "100", 1);
  {
    Scenario scenario("large-response");
    auto request = request_fixture();
    request.config.num_frames = 256;
    request.left.path = std::string(12U * 1024U - 4U, 'l') + ".mp4";
    request.right.path = std::string(12U * 1024U - 4U, 'r') + ".mp4";
    request.calibration_timeout_ns = 2'000'000'000ULL;
    const auto result = run_gpu_calibration(request, ready_backends());
    expect_eq(result.total_matches, 256U * 12U,
              "worker waits for a delayed maximum nonblocking request");
    expect_eq(result.per_frame.size(), 256U, "supervisor drains a backpressured maximum result");
  }
  (void)::unsetenv("RECO_FAKE_CALIBRATION_PRE_REQUEST_DELAY_MS");

  {
    Scenario scenario("large-response");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "more frame summaries than requested",
                           "compact excess frame summaries are rejected after decode");
  }
}

void worker_failures_crashes_and_bad_frames_are_contained() {
  {
    Scenario scenario("failure");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "synthetic worker failure", "structured worker failure propagated");
  }
  {
    Scenario scenario("crash");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "response", "worker crash cannot crash the caller");
  }
  {
    Scenario scenario("malformed");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "response", "malformed response rejected");
  }
  {
    Scenario scenario("oversized");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "payload size", "oversized response rejected before allocation");
  }
  {
    Scenario scenario("request-oversized");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "requested frame limit",
                           "request-specific oversized response rejected before allocation");
  }
}

void timeout_is_end_to_end_and_reaps_the_worker() {
  Scenario scenario("timeout");
  const auto started = std::chrono::steady_clock::now();
  expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                         "deadline", "blocked worker is terminated at the end-to-end deadline");
  const auto elapsed = std::chrono::steady_clock::now() - started;
  expect_true(elapsed < std::chrono::seconds(2), "timeout includes bounded teardown");
}

#if !defined(RECO_CALIBRATION_WIDE_ADDRESS_SANITIZER)
void aggregate_host_memory_is_monitored() {
  Scenario scenario("memory");
  auto request = request_fixture();
  request.calibration_host_memory_limit_bytes = 32ULL * 1024ULL * 1024ULL;
  const auto started = std::chrono::steady_clock::now();
  expect_execution_error([&] { (void)run_gpu_calibration(request, ready_backends()); }, "memory",
                         "resident-memory breach terminates the worker");
  expect_true(std::chrono::steady_clock::now() - started < std::chrono::seconds(2),
              "memory monitor reacts before the full deadline");
}

void shared_mapping_memory_is_monitored() {
  Scenario scenario("shared-memory");
  auto request = request_fixture();
  request.calibration_host_memory_limit_bytes = 32ULL * 1024ULL * 1024ULL;
  expect_execution_error([&] { (void)run_gpu_calibration(request, ready_backends()); }, "memory",
                         "resident-memory guard covers shared mappings outside RLIMIT_DATA");
}

#endif

void worker_allows_threads_but_blocks_process_descendants() {
  {
    Scenario scenario("process-spawn-blocked");
    expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
              "kernel policy allows worker threads and rejects process creation");
  }
  const auto target = ::fork();
  if (target == 0) {
    pollfd delay{.fd = -1, .events = 0, .revents = 0};
    (void)::poll(&delay, 0, 30'000);
    std::_Exit(EXIT_SUCCESS);
  }
  expect_true(target > 0, "sandbox policy creates a sacrificial same-UID target");
  if (target > 0) {
    const auto write_target = temporary_path("sandbox-forbidden-write");
    const auto metadata_target = temporary_path("sandbox-forbidden-metadata");
    std::filesystem::remove(write_target);
    std::filesystem::remove(metadata_target);
    {
      std::ofstream output(metadata_target, std::ios::binary | std::ios::trunc);
      output << "host metadata must remain unchanged";
    }
    static constexpr char original_xattr[] = "user.reco.original";
    static constexpr char new_xattr[] = "user.reco.worker";
    static constexpr char original_xattr_value[] = "unchanged";
    expect_true(::setxattr(metadata_target.c_str(), original_xattr, original_xattr_value,
                           sizeof(original_xattr_value) - 1U, 0U) == 0,
                "sandbox metadata fixture creates its original xattr");
    (void)::setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-worker", 1);
    const std::array<timespec, 2> original_times{
        timespec{.tv_sec = 946'684'800, .tv_nsec = 123'456'789},
        timespec{.tv_sec = 946'684'801, .tv_nsec = 987'654'321},
    };
    expect_true(::utimensat(AT_FDCWD, metadata_target.c_str(), original_times.data(), 0) == 0,
                "sandbox metadata fixture installs stable timestamps");
    struct stat metadata_before{};
    expect_true(::lstat(metadata_target.c_str(), &metadata_before) == 0,
                "sandbox metadata fixture is stat-able before worker launch");
    (void)::setenv("RECO_FAKE_CALIBRATION_SIGNAL_TARGET_PID", std::to_string(target).c_str(), 1);
    (void)::setenv("RECO_FAKE_CALIBRATION_WRITE_TARGET", write_target.c_str(), 1);
    (void)::setenv("RECO_FAKE_CALIBRATION_METADATA_TARGET", metadata_target.c_str(), 1);
    (void)::setenv("LD_AUDIT", "/definitely/missing/reco-audit.so", 1);
    (void)::setenv("GST_PLUGIN_PATH", "/definitely/untrusted/gstreamer", 1);
    (void)::setenv("GST_PLUGIN_SYSTEM_PATH_1_0", "/definitely/untrusted/system-gstreamer", 1);
    (void)::setenv("GST_PLUGIN_LOADING_WHITELIST", "untrusted", 1);
    (void)::setenv("GST_REGISTRY", "/definitely/untrusted/gstreamer-registry.bin", 1);
    (void)::setenv("GST_REGISTRY_1_0", "/definitely/untrusted/gstreamer-registry-1.0.bin", 1);
    {
      Scenario scenario("sandbox-policy");
      expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
                "worker blocks process, network, host filesystem, and metadata mutation");
    }
    (void)::unsetenv("AWS_SECRET_ACCESS_KEY");
    expect_true(process_exists(target), "worker cannot signal the same-UID target");
    expect_true(!std::filesystem::exists(write_target), "worker cannot create host files");
    struct stat metadata_after{};
    expect_true(::lstat(metadata_target.c_str(), &metadata_after) == 0,
                "sandbox metadata fixture remains after worker exit");
    expect_true(metadata_after.st_uid == metadata_before.st_uid &&
                    metadata_after.st_gid == metadata_before.st_gid &&
                    metadata_after.st_mode == metadata_before.st_mode,
                "worker cannot change host ownership or mode metadata");
    expect_true(metadata_after.st_atim.tv_sec == metadata_before.st_atim.tv_sec &&
                    metadata_after.st_atim.tv_nsec == metadata_before.st_atim.tv_nsec &&
                    metadata_after.st_mtim.tv_sec == metadata_before.st_mtim.tv_sec &&
                    metadata_after.st_mtim.tv_nsec == metadata_before.st_mtim.tv_nsec &&
                    metadata_after.st_ctim.tv_sec == metadata_before.st_ctim.tv_sec &&
                    metadata_after.st_ctim.tv_nsec == metadata_before.st_ctim.tv_nsec,
                "worker cannot change host timestamps");
    std::array<char, 64> xattr_value{};
    const auto xattr_size =
        ::getxattr(metadata_target.c_str(), original_xattr, xattr_value.data(), xattr_value.size());
    expect_true(xattr_size == static_cast<ssize_t>(sizeof(original_xattr_value) - 1U) &&
                    std::string_view(xattr_value.data(),
                                     xattr_size > 0 ? static_cast<std::size_t>(xattr_size) : 0U) ==
                        original_xattr_value,
                "worker cannot remove or replace host xattrs");
    errno = 0;
    expect_true(::getxattr(metadata_target.c_str(), new_xattr, xattr_value.data(),
                           xattr_value.size()) < 0 &&
                    errno == ENODATA,
                "worker cannot add host xattrs");
    (void)::unsetenv("RECO_FAKE_CALIBRATION_SIGNAL_TARGET_PID");
    (void)::unsetenv("RECO_FAKE_CALIBRATION_WRITE_TARGET");
    (void)::unsetenv("RECO_FAKE_CALIBRATION_METADATA_TARGET");
    (void)::unsetenv("LD_AUDIT");
    (void)::unsetenv("GST_PLUGIN_PATH");
    (void)::unsetenv("GST_PLUGIN_SYSTEM_PATH_1_0");
    (void)::unsetenv("GST_PLUGIN_LOADING_WHITELIST");
    (void)::unsetenv("GST_REGISTRY");
    (void)::unsetenv("GST_REGISTRY_1_0");
    (void)::kill(target, SIGKILL);
    int status = 0;
    while (::waitpid(target, &status, 0) < 0 && errno == EINTR) {
    }
    std::filesystem::remove(metadata_target);
  }
}

void worker_starts_inside_the_hard_memory_boundary() {
  Scenario scenario("cgroup-boundary");
  expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
            "worker starts inside its configured cgroup-v2 memory boundary");
}

void cgroup_setup_failures_remove_the_cleanup_boundary() {
  struct FaultCase {
    const char* point;
    const char* error;
    const char* description;
  };
  const std::array cases{
      FaultCase{.point = "lifetime-pipe",
                .error = "cleanup lifetime",
                .description = "lifetime pipe setup failure"},
      FaultCase{.point = "caller-pidfd",
                .error = "caller process authority",
                .description = "caller pidfd setup failure"},
  };
  for (const auto& test : cases) {
    const auto cgroups_before = calibration_cgroups();
    expect_true(::setenv("RECO_FAKE_CALIBRATION_CGROUP_SETUP_FAILURE", test.point, 1) == 0,
                "cgroup setup fault injection installs");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           test.error, test.description);
    (void)::unsetenv("RECO_FAKE_CALIBRATION_CGROUP_SETUP_FAILURE");
    expect_true(wait_for_cgroup_set(cgroups_before, std::chrono::milliseconds(500)),
                std::string(test.description) + " leaves no cleanup cgroup");
  }
}

std::set<std::filesystem::path> calibration_scratch_roots() {
  std::set<std::filesystem::path> roots;
  std::error_code error;
  for (std::filesystem::directory_iterator item("/tmp", error), end; item != end && !error;
       item.increment(error)) {
    if (item->path().filename().string().starts_with("reco-calibration-sandbox-")) {
      roots.insert(item->path());
    }
  }
  return roots;
}

void worker_scratch_is_removed_recursively() {
  const auto before = calibration_scratch_roots();
  {
    Scenario scenario("scratch-residue");
    expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
              "worker can populate its private scratch tree");
  }
  expect_true(calibration_scratch_roots() == before,
              "supervisor removes nested worker scratch contents");
}

void runtime_snapshots_have_an_aggregate_byte_limit() {
  const auto first = temporary_path("runtime-snapshot-first");
  const auto second = temporary_path("runtime-snapshot-second");
  std::filesystem::remove(first);
  std::filesystem::remove(second);
  const std::string contents(700U * 1024U, 'r');
  {
    std::ofstream output(first, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    expect_true(static_cast<bool>(output), "first runtime snapshot fixture is written");
  }
  {
    std::ofstream output(second, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    expect_true(static_cast<bool>(output), "second runtime snapshot fixture is written");
  }

  const auto before = calibration_scratch_roots();
  {
    EnvironmentValue aggregate_limit("RECO_FAKE_CALIBRATION_RUNTIME_SNAPSHOT_LIMIT_BYTES",
                                     "1048576");
    EnvironmentValue first_runtime("RECO_CUDA_DRIVER_DYLIB_PATH", first.string());
    EnvironmentValue second_runtime("RECO_GSTREAMER_DYLIB_PATH", second.string());
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "aggregate byte limit",
                           "runtime snapshots cannot exhaust aggregate scratch storage");
  }
  expect_true(calibration_scratch_roots() == before,
              "failed aggregate runtime snapshots leave no scratch tree");
  std::filesystem::remove(first);
  std::filesystem::remove(second);
}

void oversized_scratch_file_is_terminated_by_pidfd() {
  const auto before = calibration_scratch_roots();
  const auto started = std::chrono::steady_clock::now();
  {
    EnvironmentValue scratch_limit("RECO_FAKE_CALIBRATION_SCRATCH_LIMIT_BYTES", "1048576");
    Scenario scenario("scratch-oversized-file");
    expect_execution_error(
        [&] { (void)run_gpu_calibration(lifecycle_request_fixture(), ready_backends()); },
        "scratch disk quota", "oversized scratch allocation is terminated with a quota error");
  }
  expect_true(std::chrono::steady_clock::now() - started < std::chrono::seconds(2),
              "scratch disk monitor terminates the worker before its deadline");
  expect_true(calibration_scratch_roots() == before,
              "oversized scratch allocation leaves no scratch tree");
}

void excessive_scratch_entries_are_terminated_by_pidfd() {
  const auto before = calibration_scratch_roots();
  const auto started = std::chrono::steady_clock::now();
  {
    Scenario scenario("scratch-many-files");
    expect_execution_error(
        [&] { (void)run_gpu_calibration(lifecycle_request_fixture(), ready_backends()); },
        "scratch inode quota", "excessive scratch entries are terminated with a quota error");
  }
  expect_true(std::chrono::steady_clock::now() - started < std::chrono::seconds(2),
              "scratch inode monitor terminates the worker before its deadline");
  expect_true(calibration_scratch_roots() == before,
              "excessive scratch entries leave no scratch tree");
}

void caller_sigchld_policy_cannot_steal_worker_ownership() {
  struct sigaction original{};
  struct sigaction ignored{};
  ignored.sa_handler = SIG_IGN;
  (void)::sigemptyset(&ignored.sa_mask);
  expect_true(::sigaction(SIGCHLD, &ignored, &original) == 0, "SIGCHLD ignore policy installs");
  {
    Scenario scenario("success");
    try {
      expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
                "SIGCHLD ignore policy cannot break guardian observation");
    } catch (const std::exception& error) {
      std::cerr << "FAIL: SIGCHLD ignore policy threw: " << error.what() << '\n';
      ++failures;
    }
  }
  expect_true(::sigaction(SIGCHLD, &original, nullptr) == 0, "SIGCHLD policy restores");

  std::atomic<bool> stop{false};
  std::thread thief([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      int status = 0;
      (void)::waitpid(-1, &status, WNOHANG);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  {
    Scenario scenario("success");
    try {
      expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
                "competing waitpid cannot steal calibration worker ownership");
    } catch (const std::exception& error) {
      std::cerr << "FAIL: competing waitpid threw: " << error.what() << '\n';
      ++failures;
    }
  }
  stop.store(true, std::memory_order_relaxed);
  thief.join();
}

void worker_descriptors_are_isolated_across_exec() {
  const auto source = ::open("/dev/null", O_RDONLY);
  expect_true(source >= 0, "descriptor isolation source opens");
  const auto high_descriptor = source >= 0 ? ::fcntl(source, F_DUPFD, 512) : -1;
  expect_true(high_descriptor >= 512, "high descriptor opens before lowering RLIMIT_NOFILE");
  struct rlimit original{};
  const bool have_limit = ::getrlimit(RLIMIT_NOFILE, &original) == 0;
  expect_true(have_limit, "RLIMIT_NOFILE is readable");
  if (high_descriptor >= 512 && have_limit) {
    const struct rlimit lowered{.rlim_cur = std::min<rlim_t>(64, original.rlim_max),
                                .rlim_max = original.rlim_max};
    expect_true(::setrlimit(RLIMIT_NOFILE, &lowered) == 0, "RLIMIT_NOFILE soft limit lowers");
    (void)::setenv("RECO_FAKE_CALIBRATION_FORBIDDEN_FD", std::to_string(high_descriptor).c_str(),
                   1);
    {
      Scenario scenario("descriptor-isolation");
      try {
        expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
                  "posix_spawn closes descriptors above the lowered soft limit");
      } catch (const std::exception& error) {
        std::cerr << "FAIL: high descriptor isolation threw: " << error.what() << '\n';
        ++failures;
      }
    }
    (void)::unsetenv("RECO_FAKE_CALIBRATION_FORBIDDEN_FD");
    (void)::setrlimit(RLIMIT_NOFILE, &original);
  }
  if (high_descriptor >= 0) {
    (void)::close(high_descriptor);
  }
  if (source >= 0) {
    (void)::close(source);
  }

  Scenario scenario("process-spawn-blocked");
  expect_eq(run_gpu_calibration(request_fixture(), ready_backends()).total_matches, 12U,
            "post-exec IPC descriptors are close-on-exec");
}

#if !defined(RECO_CALIBRATION_THREAD_SANITIZER)
void executable_identity_is_pinned_across_both_execs() {
  const auto image = temporary_path("pinned-worker");
  const auto alias = temporary_path("pinned-worker-alias");
  const auto replacement = temporary_path("replacement-worker");
  const auto ready = temporary_path("guardian-ready.pid");
  std::filesystem::remove(image);
  std::filesystem::remove(alias);
  std::filesystem::remove(replacement);
  std::filesystem::remove(ready);
  std::filesystem::copy_file(fake_worker, image);
  std::filesystem::create_hard_link(image, alias);
  {
    std::ofstream output(replacement, std::ios::binary | std::ios::trunc);
    output << "replacement must never execute\n";
  }
  std::filesystem::permissions(replacement, std::filesystem::perms::owner_read |
                                                std::filesystem::perms::owner_write |
                                                std::filesystem::perms::owner_exec);
  (void)::setenv("RECO_FAKE_CALIBRATION_GUARDIAN_READY_PATH", ready.c_str(), 1);
  (void)::setenv("RECO_FAKE_CALIBRATION_GUARDIAN_DELAY_MS", "100", 1);
  std::exception_ptr error;
  CalibrationResult result;
  {
    Scenario scenario("success");
    std::thread call([&] {
      try {
        auto request = request_fixture();
        request.calibration_worker_path = image.string();
        result = run_gpu_calibration(request, ready_backends());
      } catch (...) {
        error = std::current_exception();
      }
    });
    const auto guardian = wait_for_pid_marker(ready, std::chrono::milliseconds(500));
    expect_true(guardian.has_value(), "pinned guardian reports its startup");
    std::filesystem::permissions(alias, std::filesystem::perms::owner_all);
    {
      std::ofstream output(alias, std::ios::binary | std::ios::trunc);
      output << "original inode changed after sealed snapshot\n";
    }
    std::error_code replace_error;
    std::filesystem::rename(replacement, image, replace_error);
    expect_true(!replace_error, "worker path is replaced after guardian exec");
    call.join();
  }
  expect_true(error == nullptr, "pinned image executes after its path is replaced");
  if (error == nullptr) {
    expect_eq(result.total_matches, 12U, "guardian and worker use the same pinned image");
  }
  (void)::unsetenv("RECO_FAKE_CALIBRATION_GUARDIAN_READY_PATH");
  (void)::unsetenv("RECO_FAKE_CALIBRATION_GUARDIAN_DELAY_MS");
  std::filesystem::remove(image);
  std::filesystem::remove(alias);
  std::filesystem::remove(replacement);
  std::filesystem::remove(ready);
}
#endif

void caller_death_kills_the_worker_boundary() {
  const auto cgroups_before = calibration_cgroups();
  const auto marker = temporary_path("caller-death-worker.pid");
  const auto holder_marker = temporary_path("caller-death-holder.pid");
  std::filesystem::remove(marker);
  std::filesystem::remove(holder_marker);
  (void)::setenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH", marker.c_str(), 1);
  Scenario scenario("timeout");
  const auto caller = ::fork();
  if (caller == 0) {
    std::thread calibration([] {
      try {
        (void)run_gpu_calibration(lifecycle_request_fixture(), ready_backends());
      } catch (...) {
      }
    });
#if !defined(RECO_CALIBRATION_THREAD_SANITIZER)
    if (wait_for_pid_marker(marker, std::chrono::milliseconds(1500)).has_value()) {
      const auto holder = ::fork();
      if (holder == 0) {
        pollfd delay{.fd = -1, .events = 0, .revents = 0};
        (void)::poll(&delay, 0, 30'000);
        std::_Exit(EXIT_SUCCESS);
      }
      if (holder > 0) {
        std::ofstream output(holder_marker);
        output << holder;
      }
    }
#endif
    calibration.join();
    std::_Exit(EXIT_SUCCESS);
  }
  expect_true(caller > 0, "caller-death child starts");
  const auto worker = wait_for_pid_marker(marker, std::chrono::milliseconds(1500));
  expect_true(worker.has_value(), "caller-death worker records its host PID");
#if !defined(RECO_CALIBRATION_THREAD_SANITIZER)
  const auto holder = wait_for_pid_marker(holder_marker, std::chrono::milliseconds(1500));
  expect_true(holder.has_value(), "caller-death sibling retains an inherited lifetime pipe");
#endif
  if (caller > 0) {
    (void)::kill(caller, SIGKILL);
    int status = 0;
    while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (worker.has_value()) {
    wait_for_process_removal(*worker, "caller death terminates the isolated worker");
  }
  expect_true(wait_for_cgroup_set(cgroups_before, std::chrono::milliseconds(1500)),
              "caller death removes its delegated cgroup boundary");
#if !defined(RECO_CALIBRATION_THREAD_SANITIZER)
  if (holder.has_value()) {
    (void)::kill(*holder, SIGKILL);
    wait_for_process_removal(*holder, "caller-death sibling is terminated after cleanup");
  }
#endif
  (void)::unsetenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH");
  std::filesystem::remove(marker);
  std::filesystem::remove(holder_marker);
}

void caller_death_before_worker_launch_removes_the_boundary() {
  const auto cgroups_before = calibration_cgroups();
  const auto marker = temporary_path("prelaunch-guardian.pid");
  std::filesystem::remove(marker);
  (void)::setenv("RECO_FAKE_CALIBRATION_GUARDIAN_READY_PATH", marker.c_str(), 1);
  (void)::setenv("RECO_FAKE_CALIBRATION_GUARDIAN_DELAY_MS", "1000", 1);
  Scenario scenario("timeout");
  const auto caller = ::fork();
  if (caller == 0) {
    try {
      (void)run_gpu_calibration(lifecycle_request_fixture(), ready_backends());
    } catch (...) {
    }
    std::_Exit(EXIT_SUCCESS);
  }
  expect_true(caller > 0, "prelaunch caller starts");
  const auto guardian = wait_for_pid_marker(marker, std::chrono::milliseconds(1500));
  expect_true(guardian.has_value(), "guardian reaches the pre-worker launch boundary");
  if (caller > 0) {
    (void)::kill(caller, SIGKILL);
    int status = 0;
    while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (guardian.has_value()) {
    wait_for_process_removal(*guardian, "caller death terminates the prelaunch guardian");
  }
  expect_true(wait_for_cgroup_set(cgroups_before, std::chrono::milliseconds(1500)),
              "caller death before worker creation removes both cleanup cgroups");
  (void)::unsetenv("RECO_FAKE_CALIBRATION_GUARDIAN_READY_PATH");
  (void)::unsetenv("RECO_FAKE_CALIBRATION_GUARDIAN_DELAY_MS");
  std::filesystem::remove(marker);
}

#if !defined(RECO_CALIBRATION_WIDE_ADDRESS_SANITIZER)
void caller_cgroup_oom_preserves_cleanup_authority() {
  const auto parent = current_cgroup();
  expect_true(parent.has_value(), "caller OOM test locates cgroup-v2 membership");
  if (!parent.has_value()) {
    return;
  }
  const auto boundary =
      parent->parent_path() / ("reco-test-caller-oom-" + std::to_string(::getpid()));
  std::error_code error;
  std::filesystem::create_directory(boundary, error);
  expect_true(!error, "caller OOM test creates a private cgroup");
  if (error) {
    return;
  }
  const auto write_control = [&](std::string_view name, std::string_view value) {
    std::ofstream output(boundary / name);
    output << value;
    output.flush();
    return static_cast<bool>(output);
  };
  expect_true(write_control("memory.max", std::to_string(768ULL * 1024ULL * 1024ULL)),
              "caller OOM test installs a memory maximum");
  expect_true(write_control("memory.swap.max", "0"), "caller OOM test disables swap growth");
  expect_true(write_control("memory.oom.group", "1"),
              "caller OOM test enables group OOM termination");

  const auto cgroups_before = calibration_cgroups();
  const auto marker = temporary_path("caller-oom-worker.pid");
  std::filesystem::remove(marker);
  (void)::setenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH", marker.c_str(), 1);
  std::array<int, 2> gate{-1, -1};
  expect_true(::pipe2(gate.data(), O_CLOEXEC) == 0, "caller OOM test creates its launch gate");
  const auto caller = ::fork();
  if (caller == 0) {
    (void)::close(gate[1]);
    char release = '\0';
    if (::read(gate[0], &release, 1) != 1 || release != '1') {
      std::_Exit(EXIT_FAILURE);
    }
    (void)::close(gate[0]);
    Scenario scenario("timeout");
    std::thread calibration([] {
      try {
        auto request = lifecycle_request_fixture();
        request.calibration_host_memory_limit_bytes = 64ULL * 1024ULL * 1024ULL;
        (void)run_gpu_calibration(request, ready_backends());
      } catch (...) {
      }
    });
    if (wait_for_pid_marker(marker, std::chrono::milliseconds(1500)).has_value()) {
      constexpr std::size_t allocation_bytes = 1024ULL * 1024ULL * 1024ULL;
      std::vector<std::uint8_t> allocation(allocation_bytes);
      volatile auto* bytes = allocation.data();
      for (std::size_t offset = 0; offset < allocation.size(); offset += 4096U) {
        bytes[offset] = static_cast<std::uint8_t>(offset);
      }
    }
    calibration.join();
    std::_Exit(EXIT_FAILURE);
  }
  (void)::close(gate[0]);
  expect_true(caller > 0, "caller OOM process starts");
  if (caller > 0) {
    expect_true(write_control("cgroup.procs", std::to_string(caller)),
                "caller OOM process enters its private cgroup");
    const char release = '1';
    ssize_t written = -1;
    do {
      written = ::write(gate[1], &release, 1);
    } while (written < 0 && errno == EINTR);
    expect_true(written == 1, "caller OOM process launch gate is released");
  }
  (void)::close(gate[1]);
  const auto worker = wait_for_pid_marker(marker, std::chrono::milliseconds(2000));
  expect_true(worker.has_value(), "caller OOM worker reaches its isolated cgroup");
  int status = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (caller > 0 && ::waitpid(caller, &status, WNOHANG) == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (caller > 0 && process_exists(caller)) {
    (void)::kill(caller, SIGKILL);
    while (::waitpid(caller, &status, 0) < 0 && errno == EINTR) {
    }
  }
  std::ifstream events(boundary / "memory.events");
  const std::string event_text((std::istreambuf_iterator<char>(events)),
                               std::istreambuf_iterator<char>());
  expect_true(event_text.find("oom_kill 0") == std::string::npos,
              "caller cgroup reaches a kernel OOM kill");
  if (worker.has_value()) {
    wait_for_process_removal(*worker, "caller group OOM terminates the isolated worker");
  }
  expect_true(wait_for_cgroup_set(cgroups_before, std::chrono::milliseconds(1500)),
              "cleanup authority survives caller group OOM");
  (void)::unsetenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH");
  std::filesystem::remove(marker);
  std::filesystem::remove(boundary, error);
  expect_true(!error, "caller OOM test removes its private cgroup");
}
#endif

void unexpected_guardian_death_kills_the_worker_boundary() {
  const auto marker = temporary_path("guardian-death-worker.pid");
  std::filesystem::remove(marker);
  (void)::setenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH", marker.c_str(), 1);
  std::exception_ptr first_error;
  {
    Scenario scenario("timeout");
    std::thread first([&] {
      try {
        (void)run_gpu_calibration(lifecycle_request_fixture(), ready_backends());
      } catch (...) {
        first_error = std::current_exception();
      }
    });
    const auto worker = wait_for_pid_marker(marker, std::chrono::milliseconds(1500));
    expect_true(worker.has_value(), "guardian-death worker records its host PID");
    const auto guardian = worker.has_value() ? process_parent(*worker) : std::nullopt;
    expect_true(guardian.has_value(), "worker exposes its outer guardian PID");
    if (guardian.has_value()) {
      expect_true(::kill(*guardian, SIGKILL) == 0, "test terminates the worker guardian");
    }
    first.join();
    expect_true(first_error != nullptr, "unexpected guardian death fails the calibration call");
    if (worker.has_value()) {
      wait_for_process_removal(*worker, "guardian death terminates the isolated worker");
    }
  }
  (void)::unsetenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH");
  std::filesystem::remove(marker);
}

void active_worker_retains_cross_process_admission() {
  const auto marker = temporary_path("active-worker.pid");
  std::filesystem::remove(marker);
  (void)::setenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH", marker.c_str(), 1);
  std::exception_ptr first_error;
  {
    Scenario scenario("timeout");
    std::thread first([&] {
      try {
        (void)run_gpu_calibration(lifecycle_request_fixture(), ready_backends());
      } catch (...) {
        first_error = std::current_exception();
      }
    });
    const auto worker = wait_for_pid_marker(marker, std::chrono::milliseconds(1500));
    expect_true(worker.has_value(), "active worker records its host PID");
    expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                           "admission", "active worker retains cross-process admission");
    first.join();
    expect_true(first_error != nullptr, "active worker reaches its configured deadline");
    if (worker.has_value()) {
      expect_true(!process_exists(*worker), "admission owner exits before its lock is released");
    }
  }
  (void)::unsetenv("RECO_FAKE_CALIBRATION_WORKER_PID_PATH");
  std::filesystem::remove(marker);
}
#endif

void invalid_worker_paths_fail_before_launch() {
  auto request = request_fixture();
  request.calibration_worker_path = "/definitely/missing/reco_calibration_worker";
#if defined(__linux__)
  expect_execution_error([&] { (void)run_gpu_calibration(request, ready_backends()); },
                         "executable regular file", "missing worker fails closed");
#else
  expect_execution_error([&] { (void)run_gpu_calibration(request, ready_backends()); },
                         "only supported on Linux", "unsupported platform fails closed");
#endif
}

} // namespace

int main() {
  fake_worker = executable_runfile("cpp/tests/fake_calibration_worker");
  invalid_worker_paths_fail_before_launch();
#if defined(__linux__)
  try {
    success_returns_the_bounded_result();
  } catch (const CalibrationExecutionError& error) {
    if (std::string_view(error.what()).find("delegated cgroup-v2") != std::string_view::npos) {
      const char* required = std::getenv("RECO_REQUIRE_CALIBRATION_CONTAINMENT_TEST");
      if (required != nullptr && std::string_view(required) == "1") {
        std::cerr << "FAIL: delegated cgroup-v2 memory controller is required but unavailable\n";
        return EXIT_FAILURE;
      }
      std::cout << "SKIP: delegated cgroup-v2 memory controller is unavailable\n";
      return EXIT_SUCCESS;
    }
    throw;
  }
  run_case("retained media descriptors", retained_input_descriptors_reach_the_sandboxed_worker);
  run_case("input mutation", input_mutation_is_rejected);
  run_case("native stdout isolation", native_stdout_noise_does_not_corrupt_protocol);
  run_case("delayed worker request", delayed_worker_request_io_obeys_the_deadline);
  run_case("worker failure containment", worker_failures_crashes_and_bad_frames_are_contained);
  run_case("end-to-end timeout", timeout_is_end_to_end_and_reaps_the_worker);
#if !defined(RECO_CALIBRATION_WIDE_ADDRESS_SANITIZER)
  run_case("aggregate memory monitoring", aggregate_host_memory_is_monitored);
  run_case("shared memory monitoring", shared_mapping_memory_is_monitored);
  run_case("caller cgroup OOM cleanup", caller_cgroup_oom_preserves_cleanup_authority);
#endif
  run_case("worker process policy", worker_allows_threads_but_blocks_process_descendants);
  run_case("hard memory boundary", worker_starts_inside_the_hard_memory_boundary);
  run_case("cgroup setup failure cleanup", cgroup_setup_failures_remove_the_cleanup_boundary);
  run_case("recursive scratch cleanup", worker_scratch_is_removed_recursively);
  run_case("runtime snapshot aggregate quota", runtime_snapshots_have_an_aggregate_byte_limit);
  run_case("scratch disk quota", oversized_scratch_file_is_terminated_by_pidfd);
  run_case("scratch inode quota", excessive_scratch_entries_are_terminated_by_pidfd);
  run_case("SIGCHLD ownership", caller_sigchld_policy_cannot_steal_worker_ownership);
  run_case("descriptor isolation", worker_descriptors_are_isolated_across_exec);
#if !defined(RECO_CALIBRATION_THREAD_SANITIZER)
  run_case("pinned executable identity", executable_identity_is_pinned_across_both_execs);
#endif
  run_case("caller death cleanup", caller_death_kills_the_worker_boundary);
  run_case("prelaunch caller death cleanup",
           caller_death_before_worker_launch_removes_the_boundary);
  run_case("guardian death cleanup", unexpected_guardian_death_kills_the_worker_boundary);
  run_case("cross-process admission", active_worker_retains_cross_process_admission);
#else
  expect_execution_error([&] { (void)run_gpu_calibration(request_fixture(), ready_backends()); },
                         "fails closed",
                         "unsupported platforms reject isolated calibration explicitly");
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
