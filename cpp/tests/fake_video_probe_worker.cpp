#include <array>
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
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace reco::io::detail {
#if !defined(_WIN32)
int run_gpu_video_probe_supervisor(const char* executable, std::uint64_t pre_worker_report_delay_ns,
                                   std::uint64_t pre_guardian_exec_delay_ns,
                                   std::uint64_t caller_pid, bool has_marker);
int run_gpu_video_probe_guardian(const char* executable, std::uint64_t pre_worker_report_delay_ns);
#endif
} // namespace reco::io::detail

namespace {

class PreMainWorkerBlock {
public:
  PreMainWorkerBlock() {
#if !defined(_WIN32)
    const char* guardian_process = std::getenv("RECO_VIDEO_PROBE_GUARDIAN_PROCESS");
    const char* forbidden_descriptor = std::getenv("RECO_FAKE_PROBE_PRE_MAIN_FORBIDDEN_FD");
    const char* leak_marker = std::getenv("RECO_FAKE_PROBE_PRE_MAIN_DESCRIPTOR_LEAK_PATH");
    if (guardian_process != nullptr && std::strcmp(guardian_process, "1") == 0 &&
        forbidden_descriptor != nullptr && forbidden_descriptor[0] != '\0' &&
        leak_marker != nullptr && leak_marker[0] != '\0') {
      char* end = nullptr;
      errno = 0;
      const auto descriptor = std::strtol(forbidden_descriptor, &end, 10);
      if (errno == 0 && end != forbidden_descriptor && end != nullptr && *end == '\0' &&
          descriptor >= 0 && descriptor <= std::numeric_limits<int>::max() &&
          ::fcntl(static_cast<int>(descriptor), F_GETFD) >= 0) {
        const auto marker = ::open(leak_marker, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (marker >= 0) {
          constexpr std::string_view leaked = "leaked";
          (void)::write(marker, leaked.data(), leaked.size());
          (void)::close(marker);
        }
      }
    }
#endif
    const char* scenario = std::getenv("RECO_FAKE_PROBE_WORKER_SCENARIO");
    const char* marker_path = std::getenv("RECO_FAKE_PROBE_WORKER_PID_PATH");
#if defined(_WIN32)
    const char* guardian_process = std::getenv("RECO_VIDEO_PROBE_GUARDIAN_PROCESS");
#endif
    const bool worker_block =
        scenario != nullptr && std::strcmp(scenario, "pre-main-block") == 0 &&
        (guardian_process == nullptr || std::strcmp(guardian_process, "1") != 0);
#if !defined(_WIN32)
    const char* process_role = std::getenv("RECO_VIDEO_PROBE_PROCESS_ROLE");
    const bool supervisor_block =
        scenario != nullptr && std::strcmp(scenario, "supervisor-pre-main-block") == 0 &&
        process_role != nullptr && std::strcmp(process_role, "supervisor") == 0;
#else
    constexpr bool supervisor_block = false;
#endif
    if ((!worker_block && !supervisor_block) || marker_path == nullptr || marker_path[0] == '\0') {
      return;
    }
#if defined(_WIN32)
    const auto marker = CreateFileA(marker_path, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                                    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (marker == INVALID_HANDLE_VALUE) {
      return;
    }
#else
    const auto marker = ::open(marker_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (marker < 0) {
      return;
    }
#endif
    std::array<char, 32> process_id{};
    const auto [end, error] =
        std::to_chars(process_id.data(), process_id.data() + process_id.size(),
                      static_cast<std::uint64_t>(
#if defined(_WIN32)
                          GetCurrentProcessId()
#else
                          ::getpid()
#endif
                              ));
#if defined(_WIN32)
    DWORD written = 0;
    if (error == std::errc{}) {
      (void)WriteFile(marker, process_id.data(), static_cast<DWORD>(end - process_id.data()),
                      &written, nullptr);
    }
    (void)CloseHandle(marker);
#else
    if (error == std::errc{}) {
      (void)::write(marker, process_id.data(), static_cast<std::size_t>(end - process_id.data()));
    }
    (void)::close(marker);
#endif
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }
};

PreMainWorkerBlock pre_main_worker_block;

constexpr std::size_t kMaximumFrameBytes = 256U * 1024U;
constexpr std::size_t kFrameHeaderBytes = sizeof(std::uint32_t);
using FrameHeader = std::array<char, kFrameHeaderBytes>;

FrameHeader frame_header(std::size_t size) {
  const auto value = static_cast<std::uint32_t>(size);
  return {static_cast<char>((value >> 24U) & 0xFFU), static_cast<char>((value >> 16U) & 0xFFU),
          static_cast<char>((value >> 8U) & 0xFFU), static_cast<char>(value & 0xFFU)};
}

bool read_exact(char* destination, std::size_t size) {
  std::cin.read(destination, static_cast<std::streamsize>(size));
  return std::cin.gcount() == static_cast<std::streamsize>(size);
}

bool read_request() {
  FrameHeader header{};
  if (!read_exact(header.data(), header.size())) {
    return false;
  }
  const auto byte = [](char value) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(value));
  };
  const auto size = (byte(header[0]) << 24U) | (byte(header[1]) << 16U) | (byte(header[2]) << 8U) |
                    byte(header[3]);
  if (size == 0 || size > kMaximumFrameBytes) {
    return false;
  }
  std::string request(size, '\0');
  return read_exact(request.data(), request.size());
}

void write_frame(std::string_view payload) {
  const auto header = frame_header(payload.size());
  std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::cout.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  std::cout.flush();
  if (!std::cout) {
    throw std::runtime_error("failed to write video probe worker response");
  }
}

bool write_process_marker(const char* environment_name, std::uint64_t process_id) {
  const char* marker_path = std::getenv(environment_name);
  if (marker_path == nullptr || marker_path[0] == '\0') {
    return true;
  }
  std::ofstream marker(marker_path);
  marker << process_id;
  return static_cast<bool>(marker);
}

bool write_lifecycle_event(std::string_view event) {
  const char* path = std::getenv("RECO_FAKE_PROBE_LIFECYCLE_PATH");
  if (path == nullptr || path[0] == '\0') {
    return true;
  }
  std::ofstream output(path, std::ios::app);
  output << event << '\n';
  return static_cast<bool>(output);
}

std::uint64_t spawn_sleeping_descendant() {
#if defined(_WIN32)
  std::vector<wchar_t> executable(32'768);
  const auto length =
      GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  if (length == 0 || length >= executable.size()) {
    return 0;
  }
  const std::wstring application(executable.data(), length);
  auto command_line = L"\"" + application + L"\" --reco-fake-probe-sleeper";
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                     CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process) == 0) {
    return 0;
  }
  const auto process_id = static_cast<std::uint64_t>(process.dwProcessId);
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  return process_id;
#else
  const auto process_id = ::fork();
  if (process_id == 0) {
    std::this_thread::sleep_for(std::chrono::seconds(30));
    std::_Exit(EXIT_SUCCESS);
  }
  return process_id < 0 ? 0 : static_cast<std::uint64_t>(process_id);
#endif
}

#if !defined(_WIN32)
void terminate_and_reap(pid_t process_id) {
  if (process_id <= 0) {
    return;
  }
  (void)::kill(process_id, SIGKILL);
  int status = 0;
  while (::waitpid(process_id, &status, 0) < 0 && errno == EINTR) {
  }
}

bool write_pipe_byte(int descriptor, char value) {
  ssize_t written = -1;
  do {
    written = ::write(descriptor, &value, 1);
  } while (written < 0 && errno == EINTR);
  return written == 1;
}

bool read_pipe_byte(int descriptor, char* value) {
  ssize_t received = -1;
  do {
    received = ::read(descriptor, value, 1);
  } while (received < 0 && errno == EINTR);
  return received == 1;
}

int run_aggregate_memory_with_descendant() {
  constexpr std::size_t kAllocationBytes = 280ULL * 1024ULL * 1024ULL;
  int readiness[2] = {-1, -1};
  if (::pipe(readiness) != 0) {
    return EXIT_FAILURE;
  }

  const auto descendant = ::fork();
  if (descendant == 0) {
    (void)::close(readiness[0]);
    try {
      std::vector<std::uint8_t> allocation(kAllocationBytes);
      volatile auto* bytes = allocation.data();
      for (std::size_t offset = 0; offset < allocation.size(); offset += 4'096U) {
        bytes[offset] = static_cast<std::uint8_t>(offset);
      }
      if (!write_lifecycle_event("descendant-memory-ready") ||
          !write_pipe_byte(readiness[1], 'R')) {
        (void)::close(readiness[1]);
        std::_Exit(EXIT_FAILURE);
      }
      (void)::close(readiness[1]);
      std::this_thread::sleep_for(std::chrono::seconds(30));
    } catch (...) {
      (void)::close(readiness[1]);
      std::_Exit(EXIT_FAILURE);
    }
    std::_Exit(EXIT_SUCCESS);
  }

  (void)::close(readiness[1]);
  if (descendant < 0) {
    (void)::close(readiness[0]);
    return EXIT_FAILURE;
  }
  if (!write_process_marker("RECO_FAKE_PROBE_DESCENDANT_PATH",
                            static_cast<std::uint64_t>(descendant))) {
    (void)::close(readiness[0]);
    terminate_and_reap(descendant);
    return EXIT_FAILURE;
  }

  char ready = '\0';
  if (!read_pipe_byte(readiness[0], &ready) || ready != 'R') {
    (void)::close(readiness[0]);
    terminate_and_reap(descendant);
    return EXIT_FAILURE;
  }
  (void)::close(readiness[0]);
  if (!write_lifecycle_event("worker-memory-allocation-started")) {
    terminate_and_reap(descendant);
    return EXIT_FAILURE;
  }

  try {
    std::vector<std::uint8_t> allocation(kAllocationBytes);
    volatile auto* bytes = allocation.data();
    for (std::size_t offset = 0; offset < allocation.size(); offset += 4'096U) {
      bytes[offset] = static_cast<std::uint8_t>(offset);
    }
    if (!write_lifecycle_event("worker-memory-ready")) {
      terminate_and_reap(descendant);
      return EXIT_FAILURE;
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));
  } catch (...) {
    (void)write_lifecycle_event("worker-memory-allocation-failed");
    terminate_and_reap(descendant);
    return EXIT_FAILURE;
  }
  terminate_and_reap(descendant);
  return EXIT_FAILURE;
}

bool close_unrelated_descriptors() {
#if defined(__linux__)
  std::error_code directory_error;
  std::vector<int> descriptors;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd", directory_error)) {
    const auto name = entry.path().filename().string();
    errno = 0;
    char* end = nullptr;
    const auto descriptor = std::strtol(name.c_str(), &end, 10);
    if (errno == 0 && end != name.c_str() && end != nullptr && *end == '\0' && descriptor > 2 &&
        descriptor <= std::numeric_limits<int>::max()) {
      descriptors.push_back(static_cast<int>(descriptor));
    }
  }
  if (!directory_error) {
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

[[noreturn]] void kill_guarded_process_group() {
  (void)::kill(0, SIGKILL);
  std::_Exit(EXIT_FAILURE);
}

int run_process_group_guard() {
  if (!close_unrelated_descriptors() || !write_lifecycle_event("guard")) {
    kill_guarded_process_group();
  }
  constexpr char kReady = 'R';
#if defined(SO_NOSIGPIPE) && !defined(MSG_NOSIGNAL)
  const int suppress_sigpipe = 1;
  if (::setsockopt(STDOUT_FILENO, SOL_SOCKET, SO_NOSIGPIPE, &suppress_sigpipe,
                   sizeof(suppress_sigpipe)) != 0) {
    kill_guarded_process_group();
  }
#endif
  ssize_t written = -1;
  do {
    written = ::send(STDOUT_FILENO, &kReady, 1,
#if defined(MSG_NOSIGNAL)
                     MSG_NOSIGNAL
#else
                     0
#endif
    );
  } while (written < 0 && errno == EINTR);
  if (written != 1) {
    kill_guarded_process_group();
  }
  char value = '\0';
  while (true) {
    const auto received = ::read(STDIN_FILENO, &value, 1);
    if (received > 0 || (received < 0 && errno == EINTR)) {
      continue;
    }
    kill_guarded_process_group();
  }
}

#else
int run_windows_guardian() {
  std::vector<wchar_t> executable(32'768);
  const auto length =
      GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  if (length == 0 || length >= executable.size()) {
    return EXIT_FAILURE;
  }
  const std::wstring application(executable.data(), length);
  auto command_line = L"\"" + application + L"\" --reco-video-probe-worker";
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  PROCESS_INFORMATION process{};
  if (CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup,
                     &process) == 0) {
    return EXIT_FAILURE;
  }
  BOOL worker_in_job = FALSE;
  const bool contained =
      IsProcessInJob(process.hProcess, nullptr, &worker_in_job) != 0 && worker_in_job != FALSE;
  if (!contained || ResumeThread(process.hThread) == std::numeric_limits<DWORD>::max()) {
    (void)TerminateProcess(process.hProcess, EXIT_FAILURE);
    (void)CloseHandle(process.hThread);
    (void)CloseHandle(process.hProcess);
    return EXIT_FAILURE;
  }
  (void)CloseHandle(process.hThread);
  const auto wait_result = WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = EXIT_FAILURE;
  if (wait_result != WAIT_OBJECT_0 || GetExitCodeProcess(process.hProcess, &exit_code) == 0) {
    exit_code = EXIT_FAILURE;
  }
  (void)CloseHandle(process.hProcess);
  return exit_code <= static_cast<DWORD>(std::numeric_limits<int>::max())
             ? static_cast<int>(exit_code)
             : EXIT_FAILURE;
}
#endif

} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--reco-fake-probe-sleeper") == 0) {
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return EXIT_SUCCESS;
  }
#if !defined(_WIN32)
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
    if (error != std::errc{} || end != delay_value.data() + delay_value.size() ||
        !write_process_marker("RECO_FAKE_PROBE_GUARDIAN_PID_PATH",
                              static_cast<std::uint64_t>(::getpid()))) {
      return EXIT_FAILURE;
    }
    return reco::io::detail::run_gpu_video_probe_guardian(argv[0], delay_ns);
  }
  if (argc == 2 && std::strcmp(argv[1], "--reco-video-probe-guard") == 0) {
    return run_process_group_guard();
  }
#else
  if (argc == 2 && std::strcmp(argv[1], "--reco-video-probe-guardian") == 0) {
    return run_windows_guardian();
  }
#endif
#if defined(_WIN32)
  constexpr int kExpectedArguments = 2;
#else
  constexpr int kExpectedArguments = 3;
#endif
  if (argc != kExpectedArguments || std::strcmp(argv[1], "--reco-video-probe-worker") != 0) {
    return EXIT_FAILURE;
  }
#if !defined(_WIN32)
  const auto parent_argument = std::string_view(argv[2]);
  const auto separator = parent_argument.find(':');
  std::uint64_t expected_parent = 0;
  const auto [parent_end, parent_error] =
      std::from_chars(parent_argument.data(),
                      separator == std::string_view::npos ? parent_argument.data()
                                                          : parent_argument.data() + separator,
                      expected_parent);
  const bool valid_parent =
      separator != std::string_view::npos && separator + 2 == parent_argument.size() &&
      (parent_argument[separator + 1] == '0' || parent_argument[separator + 1] == '1' ||
       parent_argument[separator + 1] == '2') &&
      parent_error == std::errc{} && parent_end == parent_argument.data() + separator &&
      expected_parent == static_cast<std::uint64_t>(::getppid());
  if (!valid_parent || (parent_argument[separator + 1] == '0' && !close_unrelated_descriptors())) {
    return EXIT_FAILURE;
  }
  if (parent_argument[separator + 1] == '2') {
    (void)::close(3);
  }
#endif
#if defined(_WIN32)
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
    return EXIT_FAILURE;
  }
#endif
  const char* scenario = std::getenv("RECO_FAKE_PROBE_WORKER_SCENARIO");
#if !defined(_WIN32)
#if !defined(__APPLE__)
  if (scenario != nullptr && std::strcmp(scenario, "memory-limit") == 0) {
    struct rlimit address_space{};
    constexpr rlim_t kExpectedMaximum = 512ULL * 1024ULL * 1024ULL;
    if (::getrlimit(RLIMIT_AS, &address_space) != 0 || address_space.rlim_cur == RLIM_INFINITY ||
        address_space.rlim_cur > kExpectedMaximum ||
        address_space.rlim_max != address_space.rlim_cur) {
      return 5;
    }
    const struct rlimit raised{.rlim_cur = address_space.rlim_cur,
                               .rlim_max = address_space.rlim_max + 1U};
    errno = 0;
    if (::setrlimit(RLIMIT_AS, &raised) == 0 || errno != EPERM) {
      return 6;
    }
  }
#else
  if (scenario != nullptr && std::strcmp(scenario, "memory-over-limit") == 0) {
    constexpr std::size_t kAllocationBytes = 640ULL * 1024ULL * 1024ULL;
    std::vector<std::uint8_t> allocation(kAllocationBytes);
    volatile auto* bytes = allocation.data();
    for (std::size_t offset = 0; offset < allocation.size(); offset += 4'096U) {
      bytes[offset] = static_cast<std::uint8_t>(offset);
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return 6;
  }
#endif
  if (scenario != nullptr && std::strcmp(scenario, "descriptor-isolation") == 0) {
    const char* forbidden_descriptor = std::getenv("RECO_FAKE_PROBE_FORBIDDEN_FD");
    if (forbidden_descriptor != nullptr && forbidden_descriptor[0] != '\0') {
      char* end = nullptr;
      errno = 0;
      const auto descriptor = std::strtol(forbidden_descriptor, &end, 10);
      if (errno != 0 || end == forbidden_descriptor || end == nullptr || *end != '\0' ||
          descriptor < 0 || descriptor > std::numeric_limits<int>::max()) {
        return 5;
      }
      errno = 0;
      if (::fcntl(static_cast<int>(descriptor), F_GETFD) >= 0 || errno != EBADF) {
        return 4;
      }
    }
  }
#endif
  if (!write_lifecycle_event("worker")) {
    return EXIT_FAILURE;
  }
  const auto current_process_id =
#if defined(_WIN32)
      static_cast<std::uint64_t>(GetCurrentProcessId());
#else
      static_cast<std::uint64_t>(::getpid());
#endif
  if (!write_process_marker("RECO_FAKE_PROBE_WORKER_PID_PATH", current_process_id)) {
    return EXIT_FAILURE;
  }
#if !defined(_WIN32)
  if (scenario != nullptr && std::strcmp(scenario, "process-spawn-denied") == 0) {
    const auto descendant = ::fork();
    if (descendant == 0) {
      std::_Exit(EXIT_SUCCESS);
    }
    if (descendant > 0) {
      terminate_and_reap(descendant);
      return 8;
    }
#if defined(__APPLE__)
    if (errno != EAGAIN) {
#else
    if (errno != EPERM) {
#endif
      return 9;
    }
    bool thread_ran = false;
    std::thread thread([&thread_ran] { thread_ran = true; });
    thread.join();
    if (!thread_ran) {
      return 10;
    }
  }
  if (scenario != nullptr && std::strcmp(scenario, "aggregate-memory-with-descendant") == 0) {
    return run_aggregate_memory_with_descendant();
  }
#endif
  if (scenario != nullptr && std::strcmp(scenario, "startup-marker") == 0) {
    const char* marker_path = std::getenv("RECO_FAKE_PROBE_STARTUP_MARKER_PATH");
    if (marker_path == nullptr || marker_path[0] == '\0') {
      return EXIT_FAILURE;
    }
    std::ofstream marker(marker_path);
    marker << "started";
    if (!marker) {
      return EXIT_FAILURE;
    }
  }
#if defined(__linux__)
  if (scenario != nullptr && std::strcmp(scenario, "descriptor-isolation") == 0) {
    const char* forbidden_path = std::getenv("RECO_FAKE_PROBE_FORBIDDEN_PATH");
    if (forbidden_path == nullptr || forbidden_path[0] == '\0') {
      return EXIT_FAILURE;
    }
    std::error_code directory_error;
    for (const auto& descriptor :
         std::filesystem::directory_iterator("/proc/self/fd", directory_error)) {
      std::error_code link_error;
      if (std::filesystem::read_symlink(descriptor.path(), link_error) == forbidden_path) {
        return 4;
      }
    }
  }
#endif
  if (scenario != nullptr && std::strcmp(scenario, "close-input") == 0) {
#if defined(_WIN32)
    CloseHandle(GetStdHandle(STD_INPUT_HANDLE));
#else
    close(STDIN_FILENO);
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return EXIT_SUCCESS;
  }
  if (scenario != nullptr && (std::strcmp(scenario, "valid-metadata-with-descendant") == 0 ||
                              std::strcmp(scenario, "block-input-with-descendant") == 0)) {
    const char* descendant_path = std::getenv("RECO_FAKE_PROBE_DESCENDANT_PATH");
    if (descendant_path == nullptr || descendant_path[0] == '\0') {
      return EXIT_FAILURE;
    }
    const auto descendant = spawn_sleeping_descendant();
    if (descendant == 0) {
      return EXIT_FAILURE;
    }
    if (!write_process_marker("RECO_FAKE_PROBE_DESCENDANT_PATH", descendant)) {
      return EXIT_FAILURE;
    }
  }
  if (scenario != nullptr && std::strcmp(scenario, "block-input") != 0 &&
      std::strcmp(scenario, "block-input-with-descendant") != 0) {
    if (!read_request()) {
      return EXIT_FAILURE;
    }
    if (!write_lifecycle_event("request")) {
      return EXIT_FAILURE;
    }
    if (std::strcmp(scenario, "crash") == 0) {
      return 3;
    }
    if (std::strcmp(scenario, "malformed-response") == 0) {
      write_frame("not CBOR");
      return EXIT_SUCCESS;
    }
    if (std::strcmp(scenario, "deep-response") == 0) {
      std::string nested(4'096, static_cast<char>(0x81));
      nested.push_back(static_cast<char>(0xF6));
      write_frame(nested);
      return EXIT_SUCCESS;
    }
    if (std::strcmp(scenario, "truncated-frame") == 0) {
      const auto header = frame_header(16);
      std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
      std::cout << "short";
      return EXIT_SUCCESS;
    }
    if (std::strcmp(scenario, "oversized-frame") == 0) {
      const auto header = frame_header(kMaximumFrameBytes + 1U);
      std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
      return EXIT_SUCCESS;
    }
    nlohmann::json response;
    if (std::strcmp(scenario, "wrong-version") == 0) {
      response = {{"protocol_version", 1}, {"ok", false}};
    } else if (std::strcmp(scenario, "wrapped-version") == 0) {
      response = {{"protocol_version", std::numeric_limits<std::uint64_t>::max()}, {"ok", false}};
    } else if (std::strcmp(scenario, "valid-metadata") == 0 ||
               std::strcmp(scenario, "valid-metadata-with-descendant") == 0 ||
#if !defined(_WIN32)
               std::strcmp(scenario, "descriptor-isolation") == 0 ||
               std::strcmp(scenario, "process-spawn-denied") == 0 ||
#if !defined(__APPLE__)
               std::strcmp(scenario, "memory-limit") == 0 ||
#endif
#endif
               std::strcmp(scenario, "invalid-metadata") == 0 ||
               std::strcmp(scenario, "negative-metadata") == 0 ||
               std::strcmp(scenario, "oversized-metadata") == 0) {
      const bool negative = std::strcmp(scenario, "negative-metadata") == 0;
      const bool oversized = std::strcmp(scenario, "oversized-metadata") == 0;
      response = {{"protocol_version", 5},
                  {"ok", true},
                  {"width",
                   negative ? nlohmann::json(-2)
                   : oversized
                       ? nlohmann::json(std::numeric_limits<std::uint64_t>::max())
                       : nlohmann::json(
                             std::strcmp(scenario, "valid-metadata") == 0 ||
                                     std::strcmp(scenario, "valid-metadata-with-descendant") == 0 ||
#if !defined(_WIN32)
                                     std::strcmp(scenario, "descriptor-isolation") == 0 ||
                                     std::strcmp(scenario, "process-spawn-denied") == 0 ||
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
                                     std::strcmp(scenario, "memory-limit") == 0 || false
#else
                                     false
#endif
                                 ? 854
                                 : 853)},
                  {"height", 480},
                  {"fps_numerator", 30},
                  {"fps_denominator", 1},
                  {"duration_ns", negative ? nlohmann::json(-1) : nlohmann::json(1'000'000'000)},
                  {"total_frames", negative ? nlohmann::json(-1) : nlohmann::json(30)},
                  {"first_stream_time_ns", 0},
                  {"timestamp_multiplicity", 1},
                  {"duration_is_estimated", false},
                  {"total_frames_is_estimated", false},
                  {"selected_stream_caps_verified", true},
                  {"indexed_sampling_cadence_verified", true}};
    }
    const auto encoded = nlohmann::json::to_cbor(response);
    write_frame({reinterpret_cast<const char*>(encoded.data()), encoded.size()});
    if (std::strcmp(scenario, "trailing-response") == 0) {
      std::cout << "trailing";
    }
    return EXIT_SUCCESS;
  }
  std::this_thread::sleep_for(std::chrono::seconds(30));
  return EXIT_SUCCESS;
}
