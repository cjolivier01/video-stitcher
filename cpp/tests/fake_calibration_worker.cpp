#include "calibration_worker_internal.hpp"
#include "calibration_worker_protocol.hpp"
#include "reco/calibrate/types.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/io_uring.h>
#include <linux/memfd.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/xattr.h>
#include <unistd.h>
#include <utime.h>

extern char** environ;
#endif

using namespace reco::calibrate;
using namespace reco::calibrate::detail;

namespace {

#if defined(__linux__)
const bool ipc_closed_during_static_initialization = [] {
  errno = 0;
  return ::fcntl(3, F_GETFD) < 0 && errno == EBADF;
}();

const bool environment_sanitized_during_static_initialization = [] {
  return std::getenv("LD_AUDIT") == nullptr && std::getenv("LD_PRELOAD") == nullptr &&
         std::getenv("GST_PLUGIN_PATH") == nullptr &&
         std::getenv("RECO_CALIBRATION_TSAN_EXEC_FD") == nullptr;
}();

const bool worker_environment_sanitized_during_static_initialization = [] {
  const char* whitelist = std::getenv("GST_PLUGIN_LOADING_WHITELIST");
  const char* scratch = std::getenv("TMPDIR");
  const char* plugin_path = std::getenv("GST_PLUGIN_SYSTEM_PATH_1_0");
  const char* registry = std::getenv("GST_REGISTRY");
  const char* versioned_registry = std::getenv("GST_REGISTRY_1_0");
  return std::getenv("AWS_SECRET_ACCESS_KEY") == nullptr && whitelist != nullptr &&
         std::string_view(whitelist) ==
             "coreelements,isomp4,playback,videoparsersbad,app,nvvideo4linux2,nvvideoconvert,"
             "typefindfunctions" &&
         scratch != nullptr && plugin_path != nullptr && registry != nullptr &&
         versioned_registry != nullptr &&
         std::filesystem::path(plugin_path) == std::filesystem::path(scratch) / "gst-plugins" &&
         std::filesystem::path(registry) ==
             std::filesystem::path(scratch) / "gstreamer-registry.bin" &&
         std::filesystem::path(versioned_registry) == std::filesystem::path(registry) &&
         std::getenv("GST_PLUGIN_SYSTEM_PATH") == nullptr;
}();

const bool pre_main_worker_restrictions_active = [] {
  const char* restricted = std::getenv("RECO_CALIBRATION_PRE_MAIN_RESTRICTED");
  if (restricted == nullptr || std::string_view(restricted) != "1") {
    return true;
  }
  errno = 0;
  const auto child = ::fork();
  if (child == 0) {
    std::_Exit(EXIT_FAILURE);
  }
  if (child > 0) {
    (void)::waitpid(child, nullptr, 0);
    return false;
  }
  if (errno != EPERM) {
    return false;
  }
  errno = 0;
  const auto network = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (network >= 0) {
    (void)::close(network);
    return false;
  }
  return errno == EAFNOSUPPORT;
}();

bool read_exact(int descriptor, char* destination, std::size_t size, std::uint64_t deadline) {
  try {
    read_calibration_worker_bytes_fd(descriptor, std::span<char>(destination, size), deadline);
    return true;
  } catch (...) {
    return false;
  }
}

bool read_request(int descriptor, std::uint64_t deadline, GpuCalibrationRequest* request) {
  CalibrationWorkerFrameHeader header{};
  if (!read_exact(descriptor, header.data(), header.size(), deadline)) {
    return false;
  }
  try {
    const auto decoded = decode_calibration_worker_header(header);
    std::string frame(header.data(), header.size());
    const auto payload_offset = frame.size();
    frame.resize(payload_offset + decoded.payload_size);
    if (!read_exact(descriptor, frame.data() + payload_offset, decoded.payload_size, deadline)) {
      return false;
    }
    *request = decode_calibration_worker_request(frame);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}
#endif

reco::core::CameraParams camera() {
  return {.width = 1920,
          .height = 1080,
          .fx = 960.0,
          .fy = 960.0,
          .cx = 959.5,
          .cy = 539.5,
          .d = {-0.1, 0.02, 0.0, 0.0}};
}

CalibrationResult result() {
  FrameMatches frame{.keypoints_left = 80,
                     .keypoints_right = 75,
                     .min_descriptors = 75,
                     .post_ratio_test = 40,
                     .post_spatial_filter = 25,
                     .post_ransac = 12};
  frame.points.assign(frame.post_ransac, MatchedPoint::from_planes({0.1, 0.2}, {0.3, 0.4}));
  CalibrationResult value;
  value.calibration.left = camera();
  value.calibration.right = camera();
  value.calibration.layout = {
      .camera_axis_offset = 0.25, .intersect = 0.5, .x_ty = 0.1, .x_rz = 0.0, .z_rx = 0.0};
  value.total_matches = 12;
  value.frames_used = 1;
  value.residual_error = 0.2;
  value.confidence = 0.7;
  value.per_frame.push_back(frame);
  return value;
}

#if defined(__linux__)
void write_bytes(int descriptor, std::string_view bytes, std::uint64_t deadline) {
  try {
    write_calibration_worker_bytes_fd(descriptor, bytes, deadline);
  } catch (...) {
  }
}

void write_stderr(std::string_view message) noexcept {
  ssize_t written = -1;
  do {
    written = ::write(STDERR_FILENO, message.data(), message.size());
  } while (written < 0 && errno == EINTR);
}
#endif

#if defined(__linux__)
void write_environment_pid_marker(const char* variable, pid_t process) {
  const char* path = std::getenv(variable);
  if (path == nullptr || path[0] == '\0' || process <= 1) {
    return;
  }
  std::ofstream output(path);
  output << process;
}

pid_t host_process_id() {
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("NSpid:", 0) == 0) {
      std::istringstream values(line.substr(6));
      pid_t process = -1;
      if (values >> process && process > 1) {
        return process;
      }
    }
  }
  return ::getpid() > 1 ? ::getpid() : -1;
}

#endif

} // namespace

int main(int argc, char** argv) {
#if !defined(__linux__)
  (void)argc;
  (void)argv;
  return EXIT_FAILURE;
#else
  const bool is_worker = argc == 4 && argv != nullptr && argv[1] != nullptr &&
                         std::string_view(argv[1]) == kCalibrationWorkerIpcArgument;
  if (argc != 4 || argv == nullptr || argv[1] == nullptr || argv[2] == nullptr ||
      !ipc_closed_during_static_initialization ||
      !environment_sanitized_during_static_initialization ||
      (is_worker && !worker_environment_sanitized_during_static_initialization) ||
      !pre_main_worker_restrictions_active) {
    const std::string failure =
        "fake calibration worker rejected startup state: argc=" + std::to_string(argc) +
        " ipc_closed=" + std::to_string(ipc_closed_during_static_initialization) +
        " environment_sanitized=" +
        std::to_string(environment_sanitized_during_static_initialization) +
        " worker_environment_sanitized=" +
        std::to_string(worker_environment_sanitized_during_static_initialization) +
        " restrictions_active=" + std::to_string(pre_main_worker_restrictions_active) +
        " ld_audit=" + (std::getenv("LD_AUDIT") == nullptr ? "<unset>" : std::getenv("LD_AUDIT")) +
        " ld_preload=" +
        (std::getenv("LD_PRELOAD") == nullptr ? "<unset>" : std::getenv("LD_PRELOAD")) +
        " gst_plugin_path=" +
        (std::getenv("GST_PLUGIN_PATH") == nullptr ? "<unset>" : std::getenv("GST_PLUGIN_PATH")) +
        " whitelist=" +
        (std::getenv("GST_PLUGIN_LOADING_WHITELIST") == nullptr
             ? "<unset>"
             : std::getenv("GST_PLUGIN_LOADING_WHITELIST")) +
        "\n";
    write_stderr(failure);
    return EXIT_FAILURE;
  }
  const auto parse_unsigned = [](const char* value, std::uint64_t* result) {
    if (value == nullptr || result == nullptr) {
      return false;
    }
    const std::string_view text(value);
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), *result);
    return error == std::errc{} && end == text.data() + text.size();
  };
  std::uint64_t deadline = 0;
  if (!parse_unsigned(argv[3], &deadline)) {
    return EXIT_FAILURE;
  }
  const auto descriptor = connect_calibration_ipc(argv[2], deadline);
  if (descriptor < 0) {
    static constexpr char failure[] = "fake calibration worker could not connect IPC\n";
    write_stderr(failure);
    return EXIT_FAILURE;
  }
  if (is_worker && !install_calibration_worker_sandbox()) {
    (void)::close(descriptor);
    return EXIT_FAILURE;
  }
  if (std::string_view(argv[1]) == kCalibrationGuardianArgument) {
    write_environment_pid_marker("RECO_FAKE_CALIBRATION_GUARDIAN_READY_PATH", host_process_id());
    if (const char* raw_delay = std::getenv("RECO_FAKE_CALIBRATION_GUARDIAN_DELAY_MS");
        raw_delay != nullptr) {
      std::uint64_t delay = 0;
      if (!parse_unsigned(raw_delay, &delay) || delay > 1'000) {
        return EXIT_FAILURE;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
    return run_calibration_guardian_fd(descriptor, argv[0], deadline);
  }
  if (!is_worker) {
    return EXIT_FAILURE;
  }
  if (const char* raw_delay = std::getenv("RECO_FAKE_CALIBRATION_PRE_REQUEST_DELAY_MS");
      raw_delay != nullptr) {
    std::uint64_t delay = 0;
    if (!parse_unsigned(raw_delay, &delay) || delay > 1'000) {
      return EXIT_FAILURE;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  }
  GpuCalibrationRequest request;
  if (!read_request(descriptor, deadline, &request)) {
    static constexpr char failure[] = "fake calibration worker did not receive its request\n";
    write_stderr(failure);
    return EXIT_FAILURE;
  }
  const auto retained_path = [](int file_descriptor) {
    return (std::filesystem::path("/proc") / std::to_string(::getpid()) / "fd" /
            std::to_string(file_descriptor))
        .string();
  };
  if (request.left.retained_path.has_value()) {
    const int left_input = receive_calibration_left_input_fd(descriptor, deadline);
    request.left.retained_path = retained_path(left_input);
  }
  if (request.right.retained_path.has_value()) {
    const int right_input = receive_calibration_right_input_fd(descriptor, deadline);
    request.right.retained_path = retained_path(right_input);
  }
  if (request.left.lens_profile.has_value()) {
    const int left_profile = receive_calibration_left_profile_fd(descriptor, deadline);
    request.left.lens_profile = retained_path(left_profile);
  }
  if (request.right.lens_profile.has_value()) {
    const int right_profile = receive_calibration_right_profile_fd(descriptor, deadline);
    request.right.lens_profile = retained_path(right_profile);
  }
  write_environment_pid_marker("RECO_FAKE_CALIBRATION_WORKER_PID_PATH", host_process_id());
  const char* raw_scenario = std::getenv("RECO_FAKE_CALIBRATION_WORKER_SCENARIO");
  const std::string_view scenario = raw_scenario == nullptr ? "success" : raw_scenario;
  if (scenario == "retained-inputs") {
    if (!request.left.retained_path.has_value() || !request.right.retained_path.has_value() ||
        !request.left.lens_profile.has_value() || !request.right.lens_profile.has_value()) {
      return EXIT_FAILURE;
    }
    std::ifstream left(*request.left.retained_path, std::ios::binary);
    std::ifstream right(*request.right.retained_path, std::ios::binary);
    const std::string left_contents{std::istreambuf_iterator<char>(left),
                                    std::istreambuf_iterator<char>()};
    const std::string right_contents{std::istreambuf_iterator<char>(right),
                                     std::istreambuf_iterator<char>()};
    std::ifstream left_profile(*request.left.lens_profile, std::ios::binary);
    std::ifstream right_profile(*request.right.lens_profile, std::ios::binary);
    const std::string left_profile_contents{std::istreambuf_iterator<char>(left_profile),
                                            std::istreambuf_iterator<char>()};
    const std::string right_profile_contents{std::istreambuf_iterator<char>(right_profile),
                                             std::istreambuf_iterator<char>()};
    if (left_contents != "retained left input\n" || right_contents != "retained right input\n" ||
        left_profile_contents != "retained left profile\n" ||
        right_profile_contents != "retained right profile\n") {
      return EXIT_FAILURE;
    }
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "profile-metadata") {
    auto value = result();
    value.left_lens_profile = LensProfileInfo{.camera = "test camera",
                                              .lens = "left lens",
                                              .source = ProfileSource::File,
                                              .path = request.left.lens_profile};
    value.right_lens_profile =
        LensProfileInfo{.camera = "test camera",
                        .lens = "right lens",
                        .source = request.right.lens_profile.has_value() ? ProfileSource::File
                                                                         : ProfileSource::Fallback,
                        .path = request.right.lens_profile.has_value() ? request.right.lens_profile
                                                                       : request.left.lens_profile};
    write_bytes(descriptor, encode_calibration_worker_success(value), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "success") {
    std::atomic<bool> thread_ready{false};
    std::atomic<bool> release_thread{false};
    std::thread thread([&] {
      thread_ready.store(true, std::memory_order_release);
      while (!release_thread.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    });
    while (!thread_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    const auto named = ::pthread_setname_np(thread.native_handle(), "reco-gpu-worker") == 0;
    release_thread.store(true, std::memory_order_release);
    thread.join();
    if (!named) {
      return EXIT_FAILURE;
    }
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "delayed-success") {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "large-response") {
    auto large = result();
    large.per_frame.assign(kMaximumCalibrationWorkerResultFrames, large.per_frame.front());
    large.frames_used = large.per_frame.size();
    large.total_matches = large.frames_used * large.per_frame.front().post_ransac;
    write_bytes(descriptor, encode_calibration_worker_success(large), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "request-oversized") {
    std::array<char, kCalibrationWorkerFrameHeaderBytes> header{
        'R',
        'C',
        'A',
        'L',
        static_cast<char>(kCalibrationWorkerProtocolVersion >> 8U),
        static_cast<char>(kCalibrationWorkerProtocolVersion),
        0,
        2,
        0,
        0,
        0,
        0};
    const auto oversized_payload = static_cast<std::uint32_t>(
        maximum_calibration_worker_success_frame_bytes(request.config.num_frames,
                                                       request.config.akaze.max_keypoints) -
        kCalibrationWorkerFrameHeaderBytes + 1U);
    header[8] = static_cast<char>(oversized_payload >> 24U);
    header[9] = static_cast<char>(oversized_payload >> 16U);
    header[10] = static_cast<char>(oversized_payload >> 8U);
    header[11] = static_cast<char>(oversized_payload);
    write_bytes(descriptor, std::string_view(header.data(), header.size()), deadline);
    return EXIT_FAILURE;
  }
  if (scenario == "stdout-noise") {
    std::cout << "Opening in BLOCKING MODE\n";
    std::cout.flush();
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "failure") {
    write_bytes(descriptor, encode_calibration_worker_failure("synthetic worker failure"),
                deadline);
    return EXIT_FAILURE;
  }
  if (scenario == "timeout") {
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return EXIT_FAILURE;
  }
  if (scenario == "crash") {
    __builtin_trap();
  }
  if (scenario == "malformed") {
    write_bytes(descriptor, "not-a-frame", deadline);
    return EXIT_FAILURE;
  }
  if (scenario == "oversized") {
    std::array<char, kCalibrationWorkerFrameHeaderBytes> header{
        'R',
        'C',
        'A',
        'L',
        static_cast<char>(kCalibrationWorkerProtocolVersion >> 8U),
        static_cast<char>(kCalibrationWorkerProtocolVersion),
        0,
        2,
        0,
        0,
        0,
        0};
    constexpr auto oversized_payload =
        static_cast<std::uint32_t>(kMaximumCalibrationWorkerSuccessFrameBytes);
    header[8] = static_cast<char>(oversized_payload >> 24U);
    header[9] = static_cast<char>(oversized_payload >> 16U);
    header[10] = static_cast<char>(oversized_payload >> 8U);
    header[11] = static_cast<char>(oversized_payload);
    write_bytes(descriptor, std::string_view(header.data(), header.size()), deadline);
    return EXIT_FAILURE;
  }
  if (scenario == "memory") {
    struct rlimit limit{};
    if (::getrlimit(RLIMIT_DATA, &limit) != 0 || limit.rlim_cur == RLIM_INFINITY ||
        limit.rlim_cur != limit.rlim_max) {
      write_bytes(descriptor,
                  encode_calibration_worker_failure("kernel host-memory limit is absent"),
                  deadline);
      return EXIT_FAILURE;
    }
    const struct rlimit raised{.rlim_cur = limit.rlim_cur, .rlim_max = limit.rlim_max + 1U};
    errno = 0;
    if (::setrlimit(RLIMIT_DATA, &raised) == 0 || errno != EPERM) {
      write_bytes(descriptor,
                  encode_calibration_worker_failure("kernel host-memory limit can be raised"),
                  deadline);
      return EXIT_FAILURE;
    }
    try {
      constexpr std::size_t kAllocationBytes = 96ULL * 1024ULL * 1024ULL;
      std::vector<std::uint8_t> allocation(kAllocationBytes);
      volatile auto* bytes = allocation.data();
      for (std::size_t offset = 0; offset < allocation.size(); offset += 4096) {
        bytes[offset] = static_cast<std::uint8_t>(offset);
      }
      std::this_thread::sleep_for(std::chrono::seconds(30));
    } catch (const std::bad_alloc&) {
      write_bytes(descriptor,
                  encode_calibration_worker_failure("kernel host-memory limit enforced"), deadline);
    }
    return EXIT_FAILURE;
  }
  if (scenario == "shared-memory") {
    constexpr std::size_t kAllocationBytes = 96ULL * 1024ULL * 1024ULL;
    const auto backing =
        static_cast<int>(::syscall(SYS_memfd_create, "reco-shared-memory-test", MFD_CLOEXEC));
    if (backing < 0 || ::ftruncate(backing, static_cast<off_t>(kAllocationBytes)) != 0) {
      if (backing >= 0) {
        (void)::close(backing);
      }
      return EXIT_FAILURE;
    }
    auto* allocation = static_cast<std::uint8_t*>(
        ::mmap(nullptr, kAllocationBytes, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0));
    (void)::close(backing);
    if (allocation == MAP_FAILED) {
      return EXIT_FAILURE;
    }
    for (std::size_t offset = 0; offset < kAllocationBytes; offset += 4096) {
      allocation[offset] = static_cast<std::uint8_t>(offset);
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return EXIT_FAILURE;
  }
  if (scenario == "descriptor-isolation") {
    const char* raw_descriptor = std::getenv("RECO_FAKE_CALIBRATION_FORBIDDEN_FD");
    int forbidden_descriptor = -1;
    if (raw_descriptor == nullptr) {
      return EXIT_FAILURE;
    }
    const std::string_view text(raw_descriptor);
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), forbidden_descriptor);
    errno = 0;
    if (error != std::errc{} || end != text.data() + text.size() || forbidden_descriptor < 4 ||
        ::fcntl(forbidden_descriptor, F_GETFD) >= 0 || errno != EBADF) {
      return EXIT_FAILURE;
    }
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "process-spawn-blocked") {
    errno = 0;
    if (::prctl(PR_SET_PDEATHSIG, 0) == 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
    errno = 0;
    const auto child = ::fork();
    if (child >= 0 || errno != EPERM) {
      if (child == 0) {
        std::_Exit(EXIT_FAILURE);
      }
      return EXIT_FAILURE;
    }
    std::thread thread([] {});
    thread.join();
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "sandbox-policy") {
    const char* raw_target = std::getenv("RECO_FAKE_CALIBRATION_SIGNAL_TARGET_PID");
    std::uint64_t target = 0;
    if (!parse_unsigned(raw_target, &target) || target == 0 ||
        target > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
      return EXIT_FAILURE;
    }
    std::atomic<bool> thread_exec_blocked{false};
    std::thread exec_thread([&] {
      std::array<char*, 2> child_argv{const_cast<char*>("/bin/true"), nullptr};
      errno = 0;
      thread_exec_blocked.store(::execve(child_argv[0], child_argv.data(), environ) < 0 &&
                                    errno == EPERM,
                                std::memory_order_relaxed);
    });
    exec_thread.join();
    if (!thread_exec_blocked.load(std::memory_order_relaxed)) {
      return EXIT_FAILURE;
    }
    errno = 0;
    if (::kill(static_cast<pid_t>(target), 0) == 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
#if defined(SYS_tgkill)
    errno = 0;
    if (::syscall(SYS_tgkill, ::getpid(), ::syscall(SYS_gettid), 0) == 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
#endif
#if defined(SYS_pidfd_open)
    errno = 0;
    if (::syscall(SYS_pidfd_open, static_cast<pid_t>(target), 0U) >= 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
#endif
#if defined(SYS_pidfd_send_signal)
    errno = 0;
    if (::syscall(SYS_pidfd_send_signal, -1, 0, nullptr, 0U) == 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
#endif
#if defined(SYS_fchmodat2)
    errno = 0;
    if (::syscall(SYS_fchmodat2, AT_FDCWD, "/definitely/missing", 0600U, 0U) == 0 ||
        errno != EPERM) {
      return EXIT_FAILURE;
    }
#endif
#if defined(SYS_io_uring_setup)
    io_uring_params ring_parameters{};
    errno = 0;
    if (::syscall(SYS_io_uring_setup, 2U, &ring_parameters) >= 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
#endif
#if defined(SYS_io_uring_register)
    errno = 0;
    if (::syscall(SYS_io_uring_register, -1, 0U, nullptr, 0U) >= 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
#endif
#if defined(SYS_io_uring_enter)
    errno = 0;
    if (::syscall(SYS_io_uring_enter, -1, 0U, 0U, 0U, nullptr, 0U) >= 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
#endif
    const char* metadata_target = std::getenv("RECO_FAKE_CALIBRATION_METADATA_TARGET");
    if (metadata_target == nullptr || metadata_target[0] != '/') {
      return EXIT_FAILURE;
    }
    errno = 0;
    const auto forbidden_read = ::open(metadata_target, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (forbidden_read >= 0 || (errno != EACCES && errno != EPERM)) {
      if (forbidden_read >= 0) {
        (void)::close(forbidden_read);
      }
      return EXIT_FAILURE;
    }
    const auto target_mem = "/proc/" + std::to_string(target) + "/mem";
    errno = 0;
    const auto forbidden_process_memory = ::open(target_mem.c_str(), O_RDWR | O_CLOEXEC);
    if (forbidden_process_memory >= 0 || (errno != EACCES && errno != EPERM)) {
      if (forbidden_process_memory >= 0) {
        (void)::close(forbidden_process_memory);
      }
      return EXIT_FAILURE;
    }
    const auto modules = ::open("/proc/modules", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (modules < 0) {
      return EXIT_FAILURE;
    }
    char module_byte = '\0';
    ssize_t module_read = -1;
    do {
      module_read = ::read(modules, &module_byte, 1);
    } while (module_read < 0 && errno == EINTR);
    (void)::close(modules);
    if (module_read < 0) {
      return EXIT_FAILURE;
    }
    errno = 0;
    const auto neighboring_proc = ::open("/proc/uptime", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (neighboring_proc >= 0 || (errno != EACCES && errno != EPERM)) {
      if (neighboring_proc >= 0) {
        (void)::close(neighboring_proc);
      }
      return EXIT_FAILURE;
    }
    const char* scratch = std::getenv("TMPDIR");
    if (scratch == nullptr) {
      return EXIT_FAILURE;
    }
    const auto metadata_scratch_path = std::string(scratch) + "/worker-metadata";
    const auto metadata_descriptor = ::open(
        metadata_scratch_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (metadata_descriptor < 0) {
      return EXIT_FAILURE;
    }
    const auto metadata_failure = [&] {
      (void)::close(metadata_descriptor);
      (void)::unlink(metadata_scratch_path.c_str());
      return EXIT_FAILURE;
    };
#if defined(SYS_flock)
    errno = 0;
    if (::syscall(SYS_flock, metadata_descriptor, LOCK_UN) >= 0 || errno != EPERM) {
      return metadata_failure();
    }
#endif
    const struct utimbuf legacy_times{.actime = 1, .modtime = 2};
    const std::array<timeval, 2> microsecond_times{
        timeval{.tv_sec = 3, .tv_usec = 4},
        timeval{.tv_sec = 5, .tv_usec = 6},
    };
    const std::array<timespec, 2> nanosecond_times{
        timespec{.tv_sec = 7, .tv_nsec = 8},
        timespec{.tv_sec = 9, .tv_nsec = 10},
    };
    static constexpr char original_xattr[] = "user.reco.original";
    static constexpr char new_xattr[] = "user.reco.worker";
    static constexpr char xattr_value[] = "mutated";
#define RECO_REQUIRE_METADATA_DENIED(expression)                                                   \
  do {                                                                                             \
    errno = 0;                                                                                     \
    if ((expression) != -1 || errno != EPERM) {                                                    \
      return metadata_failure();                                                                   \
    }                                                                                              \
  } while (false)
#if defined(SYS_chown)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_chown, metadata_target, ::getuid(), ::getgid()));
#endif
#if defined(SYS_fchown)
    RECO_REQUIRE_METADATA_DENIED(
        ::syscall(SYS_fchown, metadata_descriptor, ::getuid(), ::getgid()));
#endif
#if defined(SYS_lchown)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_lchown, metadata_target, ::getuid(), ::getgid()));
#endif
#if defined(SYS_fchownat)
    RECO_REQUIRE_METADATA_DENIED(
        ::syscall(SYS_fchownat, AT_FDCWD, metadata_target, ::getuid(), ::getgid(), 0U));
#endif
#if defined(SYS_chown32) && (!defined(SYS_chown) || SYS_chown32 != SYS_chown)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_chown32, metadata_target, ::getuid(), ::getgid()));
#endif
#if defined(SYS_fchown32) && (!defined(SYS_fchown) || SYS_fchown32 != SYS_fchown)
    RECO_REQUIRE_METADATA_DENIED(
        ::syscall(SYS_fchown32, metadata_descriptor, ::getuid(), ::getgid()));
#endif
#if defined(SYS_lchown32) && (!defined(SYS_lchown) || SYS_lchown32 != SYS_lchown)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_lchown32, metadata_target, ::getuid(), ::getgid()));
#endif
#if defined(SYS_utime)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_utime, metadata_target, &legacy_times));
#endif
#if defined(SYS_utimes)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_utimes, metadata_target, microsecond_times.data()));
#endif
#if defined(SYS_futimesat)
    RECO_REQUIRE_METADATA_DENIED(
        ::syscall(SYS_futimesat, AT_FDCWD, metadata_target, microsecond_times.data()));
#endif
#if defined(SYS_utimensat)
    RECO_REQUIRE_METADATA_DENIED(
        ::syscall(SYS_utimensat, AT_FDCWD, metadata_target, nanosecond_times.data(), 0U));
#endif
#if defined(SYS_utimensat_time64) &&                                                               \
    (!defined(SYS_utimensat) || SYS_utimensat_time64 != SYS_utimensat)
    RECO_REQUIRE_METADATA_DENIED(
        ::syscall(SYS_utimensat_time64, AT_FDCWD, metadata_target, nanosecond_times.data(), 0U));
#endif
#if defined(SYS_setxattr)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_setxattr, metadata_target, new_xattr, xattr_value,
                                           sizeof(xattr_value) - 1U, 0U));
#endif
#if defined(SYS_lsetxattr)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_lsetxattr, metadata_target, new_xattr, xattr_value,
                                           sizeof(xattr_value) - 1U, 0U));
#endif
#if defined(SYS_fsetxattr)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_fsetxattr, metadata_descriptor, new_xattr,
                                           xattr_value, sizeof(xattr_value) - 1U, 0U));
#endif
#if defined(SYS_removexattr)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_removexattr, metadata_target, original_xattr));
#endif
#if defined(SYS_lremovexattr)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_lremovexattr, metadata_target, original_xattr));
#endif
#if defined(SYS_fremovexattr)
    RECO_REQUIRE_METADATA_DENIED(::syscall(SYS_fremovexattr, metadata_descriptor, original_xattr));
#endif
#undef RECO_REQUIRE_METADATA_DENIED
    (void)::close(metadata_descriptor);
    errno = 0;
    if (::unlink(metadata_scratch_path.c_str()) == 0 || (errno != EPERM && errno != EACCES)) {
      return EXIT_FAILURE;
    }
    errno = 0;
    const auto network = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (network >= 0 || errno != EPERM) {
      if (network >= 0) {
        (void)::close(network);
      }
      return EXIT_FAILURE;
    }
    std::array<int, 2> local_pair{-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, local_pair.data()) != 0) {
      return EXIT_FAILURE;
    }
    (void)::close(local_pair[0]);
    (void)::close(local_pair[1]);
    local_pair = {-1, -1};
    errno = 0;
    if (::socketpair(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0, local_pair.data()) == 0 ||
        errno != EPERM) {
      if (local_pair[0] >= 0) {
        (void)::close(local_pair[0]);
      }
      if (local_pair[1] >= 0) {
        (void)::close(local_pair[1]);
      }
      return EXIT_FAILURE;
    }
    const auto local = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (local < 0) {
      return EXIT_FAILURE;
    }
    sockaddr_un local_address{};
    local_address.sun_family = AF_UNIX;
    static constexpr char local_path[] = "/tmp/reco-forbidden-worker-ipc";
    std::memcpy(local_address.sun_path, local_path, sizeof(local_path));
    errno = 0;
    const auto connected =
        ::connect(local, reinterpret_cast<const sockaddr*>(&local_address), sizeof(local_address));
    const auto connect_error = errno;
    (void)::close(local);
    if (connected >= 0 || connect_error != ENOENT) {
      return EXIT_FAILURE;
    }
    const auto local_listener = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (local_listener < 0) {
      return EXIT_FAILURE;
    }
    sockaddr_un listener_address{};
    listener_address.sun_family = AF_UNIX;
    const auto listener_size =
        std::snprintf(listener_address.sun_path + 1, sizeof(listener_address.sun_path) - 1U,
                      "reco-worker-%ld", static_cast<long>(::getpid()));
    if (listener_size <= 0 ||
        static_cast<std::size_t>(listener_size) >= sizeof(listener_address.sun_path) - 1U) {
      (void)::close(local_listener);
      return EXIT_FAILURE;
    }
    const auto address_size = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1U +
                                                     static_cast<std::size_t>(listener_size));
    const bool listening =
        ::bind(local_listener, reinterpret_cast<const sockaddr*>(&listener_address),
               address_size) == 0 &&
        ::listen(local_listener, 1) == 0;
    errno = 0;
    const auto accepted = ::accept4(local_listener, nullptr, nullptr, SOCK_CLOEXEC);
    const auto accept_error = errno;
    (void)::close(local_listener);
    if (!listening || accepted >= 0 || accept_error != EPERM) {
      if (accepted >= 0) {
        (void)::close(accepted);
      }
      return EXIT_FAILURE;
    }
    const char* write_target = std::getenv("RECO_FAKE_CALIBRATION_WRITE_TARGET");
    if (write_target == nullptr || write_target[0] != '/') {
      return EXIT_FAILURE;
    }
    errno = 0;
    if (::truncate(metadata_target, 0) == 0 ||
        (errno != EPERM && errno != EACCES && errno != EROFS)) {
      return EXIT_FAILURE;
    }
    errno = 0;
    const auto forbidden_truncate = ::open(metadata_target, O_RDONLY | O_TRUNC | O_CLOEXEC);
    if (forbidden_truncate >= 0 || (errno != EPERM && errno != EACCES && errno != EROFS)) {
      if (forbidden_truncate >= 0) {
        (void)::close(forbidden_truncate);
      }
      return EXIT_FAILURE;
    }
    errno = 0;
    const auto forbidden_write =
        ::open(write_target, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (forbidden_write >= 0 || (errno != EACCES && errno != EROFS)) {
      if (forbidden_write >= 0) {
        (void)::close(forbidden_write);
        (void)::unlink(write_target);
      }
      return EXIT_FAILURE;
    }
    const auto scratch_file = std::string(scratch) + "/worker-write";
    const auto allowed_write =
        ::open(scratch_file.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (allowed_write < 0) {
      return EXIT_FAILURE;
    }
    (void)::close(allowed_write);
    const auto renamed_scratch_file = std::string(scratch) + "/worker-write-renamed";
    errno = 0;
    if (::rename(scratch_file.c_str(), renamed_scratch_file.c_str()) == 0 ||
        (errno != EPERM && errno != EACCES)) {
      return EXIT_FAILURE;
    }
    errno = 0;
    if (::unlink("") == 0 || errno != ENOENT) {
      return EXIT_FAILURE;
    }
#if defined(SYS_renameat2)
    errno = 0;
    if (::syscall(SYS_renameat2, AT_FDCWD, scratch_file.c_str(), AT_FDCWD,
                  renamed_scratch_file.c_str(), 0U) == 0 ||
        (errno != EPERM && errno != EACCES)) {
      return EXIT_FAILURE;
    }
#endif
#if defined(O_TMPFILE)
    errno = 0;
    const auto anonymous = ::open(scratch, O_RDWR | O_TMPFILE | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (anonymous >= 0 || (errno != EPERM && errno != EACCES)) {
      if (anonymous >= 0) {
        (void)::close(anonymous);
      }
      return EXIT_FAILURE;
    }
#endif
    errno = 0;
    if (::unlink(scratch_file.c_str()) == 0 || (errno != EPERM && errno != EACCES)) {
      return EXIT_FAILURE;
    }
    struct rlimit core{};
    if (::getrlimit(RLIMIT_CORE, &core) != 0 || core.rlim_cur != 0 || core.rlim_max != 0) {
      return EXIT_FAILURE;
    }
    const struct rlimit raised_core{.rlim_cur = 1, .rlim_max = 1};
    errno = 0;
    if (::setrlimit(RLIMIT_CORE, &raised_core) == 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
    struct rlimit file_size{};
    if (::getrlimit(RLIMIT_FSIZE, &file_size) != 0 || file_size.rlim_cur == RLIM_INFINITY ||
        file_size.rlim_cur == 0 || file_size.rlim_cur != file_size.rlim_max) {
      return EXIT_FAILURE;
    }
    const struct rlimit raised_file_size{.rlim_cur = file_size.rlim_cur,
                                         .rlim_max = file_size.rlim_max + 1U};
    errno = 0;
    if (::setrlimit(RLIMIT_FSIZE, &raised_file_size) == 0 || errno != EPERM) {
      return EXIT_FAILURE;
    }
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "scratch-oversized-file" || scenario == "scratch-unlink-oversized-file") {
    const char* scratch = std::getenv("TMPDIR");
    if (scratch == nullptr) {
      return EXIT_FAILURE;
    }
    const auto path = std::filesystem::path(scratch) / "oversized-worker-cache";
    const auto output =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (output < 0) {
      return EXIT_FAILURE;
    }
    if (scenario == "scratch-unlink-oversized-file") {
      errno = 0;
      if (::unlink(path.c_str()) == 0 || (errno != EPERM && errno != EACCES)) {
        (void)::close(output);
        return EXIT_FAILURE;
      }
    }
    std::array<char, 64U * 1024U> block{};
    constexpr std::size_t attack_bytes = 4U * 1024U * 1024U;
    std::size_t written_bytes = 0;
    while (written_bytes < attack_bytes) {
      const auto written = ::write(output, block.data(), block.size());
      if (written <= 0) {
        (void)::close(output);
        return EXIT_FAILURE;
      }
      written_bytes += static_cast<std::size_t>(written);
    }
    if (::fsync(output) != 0) {
      (void)::close(output);
      return EXIT_FAILURE;
    }
    (void)::close(output);
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return EXIT_FAILURE;
  }
  if (scenario == "scratch-many-files") {
    const char* scratch = std::getenv("TMPDIR");
    if (scratch == nullptr) {
      return EXIT_FAILURE;
    }
    constexpr std::size_t attack_entries = 5'000U;
    for (std::size_t index = 0; index < attack_entries; ++index) {
      const auto path = std::filesystem::path(scratch) / ("worker-entry-" + std::to_string(index));
      const auto entry =
          ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
      if (entry < 0) {
        return EXIT_FAILURE;
      }
      (void)::close(entry);
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return EXIT_FAILURE;
  }
  if (scenario == "scratch-residue") {
    const char* scratch = std::getenv("TMPDIR");
    if (scratch == nullptr) {
      return EXIT_FAILURE;
    }
    const auto nested = std::filesystem::path(scratch) / "cache" / "nested";
    std::error_code error;
    if (!std::filesystem::create_directories(nested, error) || error) {
      return EXIT_FAILURE;
    }
    std::ofstream residue(nested / "entry");
    residue << "cache";
    residue.close();
    if (!residue) {
      return EXIT_FAILURE;
    }
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  if (scenario == "cgroup-boundary") {
    std::ifstream membership("/proc/self/cgroup");
    std::string line;
    std::string relative;
    while (std::getline(membership, line)) {
      if (line.rfind("0::", 0) == 0) {
        relative = line.substr(3);
        break;
      }
    }
    if (relative.empty() || relative == "/") {
      return EXIT_FAILURE;
    }
    const auto cgroup = std::filesystem::path("/sys/fs/cgroup") / relative.substr(1);
    std::ifstream limit(cgroup / "memory.max");
    std::uint64_t configured = 0;
    if (!(limit >> configured) || configured != request.calibration_host_memory_limit_bytes) {
      return EXIT_FAILURE;
    }
    errno = 0;
    const auto migration = ::open((cgroup / "cgroup.procs").c_str(), O_WRONLY | O_CLOEXEC);
    if (migration >= 0 || (errno != EACCES && errno != EROFS)) {
      if (migration >= 0) {
        (void)::close(migration);
      }
      return EXIT_FAILURE;
    }
    write_bytes(descriptor, encode_calibration_worker_success(result()), deadline);
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
#endif
}
