#include "reco/cli/cli.hpp"
#include "reco/cli/interrupt.hpp"

#include "reco/calibrate/pipeline.hpp"
#include "reco/core/path.hpp"
#include "reco/io/stable_media_file.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <variant>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef __has_feature
#define __has_feature(value) 0
#endif

using namespace reco::cli;

#if defined(__SANITIZE_ADDRESS__) || __has_feature(address_sanitizer)
extern "C" const char* __lsan_default_suppressions() { return "leak:libcuda.so.1\n"; }
#endif

namespace {

int failures = 0;

template <typename Test> void run_test_case(std::string_view name, Test&& test) {
#if defined(_WIN32)
  std::cerr << "RUN: " << name << std::endl;
#endif
  try {
    std::forward<Test>(test)();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << name << " threw: " << error.what() << '\n';
    ++failures;
  } catch (...) {
    std::cerr << "FAIL: " << name << " threw a non-standard exception\n";
    ++failures;
  }
}

void set_environment(std::string_view name, const std::optional<std::string>& value) {
#if defined(_WIN32)
  _putenv_s(std::string(name).c_str(), value.value_or("").c_str());
#else
  if (value.has_value()) {
    setenv(std::string(name).c_str(), value->c_str(), 1);
  } else {
    unsetenv(std::string(name).c_str());
  }
#endif
}

class ScopedEnvironment {
public:
  ScopedEnvironment(std::string name, std::optional<std::string> value) : name_(std::move(name)) {
    if (const char* existing = std::getenv(name_.c_str()); existing != nullptr) {
      previous_ = std::string(existing);
    }
    set_environment(name_, value);
  }

  ~ScopedEnvironment() { set_environment(name_, previous_); }

  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
  std::string name_;
  std::optional<std::string> previous_;
};

#if defined(_WIN32)
class ScopedWideEnvironment {
public:
  ScopedWideEnvironment(std::wstring name, const std::filesystem::path& value)
      : name_(std::move(name)) {
    const auto required = GetEnvironmentVariableW(name_.c_str(), nullptr, 0);
    if (required != 0) {
      previous_.resize(required);
      const auto written = GetEnvironmentVariableW(name_.c_str(), previous_.data(), required);
      if (written < required) {
        previous_.resize(written);
      } else {
        previous_.clear();
      }
    }
    if (SetEnvironmentVariableW(name_.c_str(), value.c_str()) == 0) {
      throw std::runtime_error("cannot set native Windows environment path");
    }
  }

  ~ScopedWideEnvironment() {
    (void)SetEnvironmentVariableW(name_.c_str(), previous_.empty() ? nullptr : previous_.c_str());
  }

  ScopedWideEnvironment(const ScopedWideEnvironment&) = delete;
  ScopedWideEnvironment& operator=(const ScopedWideEnvironment&) = delete;

private:
  std::wstring name_;
  std::wstring previous_;
};
#endif

class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const std::filesystem::path& path)
      : previous_(std::filesystem::current_path()) {
    std::filesystem::current_path(path);
  }

  ~ScopedCurrentPath() {
    std::error_code ignored;
    std::filesystem::current_path(previous_, ignored);
  }

  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
  std::filesystem::path previous_;
};

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::random_device random;
    const auto base = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 128; ++attempt) {
      path_ =
          base / ("reco_cli_stage27_" + std::to_string(random()) + "_" + std::to_string(random()));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error && error != std::errc::file_exists) {
        throw std::filesystem::filesystem_error("cannot create test directory", path_, error);
      }
    }
    throw std::runtime_error("cannot create unique CLI test directory");
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

private:
  std::filesystem::path path_;
};

void write_text_file(const std::filesystem::path& path, std::string_view contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create test file " + path.string());
  }
  output << contents;
  output.close();
  if (!output) {
    throw std::runtime_error("cannot write test file " + path.string());
  }
}

void write_text_descriptor(int descriptor, std::string_view contents) {
  std::size_t offset = 0;
  while (offset < contents.size()) {
#if defined(_WIN32)
    const auto chunk_size = static_cast<unsigned int>(
        std::min<std::size_t>(contents.size() - offset, std::numeric_limits<unsigned int>::max()));
    const int written = _write(descriptor, contents.data() + offset, chunk_size);
#else
    const auto written = ::write(descriptor, contents.data() + offset, contents.size() - offset);
#endif
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      throw std::system_error(errno == 0 ? EIO : errno, std::generic_category(),
                              "cannot write test output descriptor");
    }
    offset += static_cast<std::size_t>(written);
  }
#if defined(_WIN32)
  if (_commit(descriptor) != 0) {
    throw std::system_error(errno, std::generic_category(), "cannot flush test output descriptor");
  }
#endif
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open test file " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

#if defined(_WIN32)
std::string wide_to_utf8(std::wstring_view value) {
  if (value.empty()) {
    return {};
  }
  if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("Windows argument is too long to convert to UTF-8");
  }
  const auto input_size = static_cast<int>(value.size());
  const int output_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                                              input_size, nullptr, 0, nullptr, nullptr);
  if (output_size <= 0) {
    throw std::runtime_error("cannot convert Windows argument to UTF-8");
  }
  std::string result(static_cast<std::size_t>(output_size), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), input_size, result.data(),
                          output_size, nullptr, nullptr) != output_size) {
    throw std::runtime_error("cannot convert Windows argument to UTF-8");
  }
  return result;
}

std::wstring quote_windows_argument(std::wstring_view argument) {
  std::wstring quoted(1, L'"');
  std::size_t backslashes = 0;
  for (const wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'"') {
      quoted.append(backslashes * 2 + 1, L'\\');
      quoted.push_back(character);
      backslashes = 0;
      continue;
    }
    quoted.append(backslashes, L'\\');
    backslashes = 0;
    quoted.push_back(character);
  }
  quoted.append(backslashes * 2, L'\\');
  quoted.push_back(L'"');
  return quoted;
}

HANDLE start_windows_publication_writer(const std::filesystem::path& destination,
                                        const std::filesystem::path& left_input,
                                        const std::filesystem::path& right_input,
                                        std::wstring_view ready_event,
                                        std::wstring_view release_event,
                                        std::wstring_view contention_event,
                                        std::wstring_view payload) {
  std::wstring executable(32768, L'\0');
  const DWORD executable_size =
      GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  if (executable_size == 0 || executable_size >= executable.size()) {
    return nullptr;
  }
  executable.resize(executable_size);
  std::wstring command = quote_windows_argument(executable);
  for (const auto argument :
       {std::wstring_view(L"--reco-publication-writer-child"),
        std::wstring_view(destination.native()), std::wstring_view(left_input.native()),
        std::wstring_view(right_input.native()), ready_event, release_event, contention_event,
        payload}) {
    command.push_back(L' ');
    command += quote_windows_argument(argument);
  }
  command.push_back(L'\0');
  STARTUPINFOW startup{.cb = sizeof(startup)};
  PROCESS_INFORMATION process{};
  if (CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE, 0, nullptr,
                     nullptr, &startup, &process) == 0) {
    return nullptr;
  }
  (void)CloseHandle(process.hThread);
  return process.hProcess;
}

int run_windows_publication_writer_child(int argc, wchar_t** argv) {
  if (argc != 9) {
    return 90;
  }
  const HANDLE ready_event = OpenEventW(EVENT_MODIFY_STATE, FALSE, argv[5]);
  const HANDLE release_event = OpenEventW(SYNCHRONIZE, FALSE, argv[6]);
  const HANDLE contention_event =
      std::wstring_view(argv[7]) == L"-" ? nullptr : OpenEventW(EVENT_MODIFY_STATE, FALSE, argv[7]);
  if (ready_event == nullptr || release_event == nullptr) {
    if (ready_event != nullptr) {
      (void)CloseHandle(ready_event);
    }
    if (release_event != nullptr) {
      (void)CloseHandle(release_event);
    }
    if (contention_event != nullptr) {
      (void)CloseHandle(contention_event);
    }
    return 91;
  }
  try {
    const std::string payload = wide_to_utf8(argv[8]);
    detail::write_calibration_json_atomically(
        payload, std::filesystem::path(argv[2]), std::filesystem::path(argv[3]),
        std::filesystem::path(argv[4]), {}, {},
        [&] {
          if (SetEvent(ready_event) == 0 ||
              WaitForSingleObject(release_event, 10000) != WAIT_OBJECT_0) {
            throw std::runtime_error("publication writer synchronization failed");
          }
        },
        false, {},
        [&] {
          if (contention_event != nullptr && SetEvent(contention_event) == 0) {
            throw std::runtime_error("cannot signal publication lock contention");
          }
        });
  } catch (...) {
    (void)CloseHandle(ready_event);
    (void)CloseHandle(release_event);
    if (contention_event != nullptr) {
      (void)CloseHandle(contention_event);
    }
    return 92;
  }
  (void)CloseHandle(ready_event);
  (void)CloseHandle(release_event);
  if (contention_event != nullptr) {
    (void)CloseHandle(contention_event);
  }
  return 0;
}

bool wait_windows_process_success(HANDLE process, DWORD timeout_ms) {
  if (process == nullptr) {
    return false;
  }
  if (WaitForSingleObject(process, timeout_ms) != WAIT_OBJECT_0) {
    (void)TerminateProcess(process, 93);
    (void)WaitForSingleObject(process, 5000);
    return false;
  }
  DWORD exit_code = 1;
  return GetExitCodeProcess(process, &exit_code) != 0 && exit_code == 0;
}

std::optional<std::filesystem::path> find_windows_reco_runfile() {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir != nullptr && *test_srcdir != '\0') {
    std::error_code error;
    const std::filesystem::recursive_directory_iterator entries(
        test_srcdir, std::filesystem::directory_options::skip_permission_denied, error);
    if (!error) {
      for (const auto& entry : entries) {
        if (entry.is_regular_file(error) && !error &&
            entry.path().filename().native() == L"reco.exe") {
          return entry.path();
        }
        error.clear();
      }
    }
  }
  const char* manifest_path = std::getenv("RUNFILES_MANIFEST_FILE");
  if (manifest_path == nullptr || *manifest_path == '\0') {
    return std::nullopt;
  }
  std::ifstream manifest(manifest_path, std::ios::binary);
  std::string line;
  constexpr std::string_view suffix = "cpp/apps/reco_cli/reco.exe";
  while (std::getline(manifest, line)) {
    const auto separator = line.find(' ');
    if (separator != std::string::npos &&
        std::string_view(line.data(), separator).ends_with(suffix)) {
      return std::filesystem::path(line.substr(separator + 1));
    }
  }
  return std::nullopt;
}

struct WindowsProcessResult {
  bool started = false;
  DWORD exit_code = 1;
  std::string output;
};

WindowsProcessResult run_windows_process_capture(const std::filesystem::path& executable,
                                                 const std::vector<std::wstring>& arguments,
                                                 const std::filesystem::path& output_path) {
  SECURITY_ATTRIBUTES inheritable{
      .nLength = sizeof(SECURITY_ATTRIBUTES),
      .lpSecurityDescriptor = nullptr,
      .bInheritHandle = TRUE,
  };
  const HANDLE output =
      CreateFileW(output_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE,
                  &inheritable, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  const HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   &inheritable, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (output == INVALID_HANDLE_VALUE || input == INVALID_HANDLE_VALUE) {
    if (output != INVALID_HANDLE_VALUE) {
      (void)CloseHandle(output);
    }
    if (input != INVALID_HANDLE_VALUE) {
      (void)CloseHandle(input);
    }
    return {};
  }
  std::wstring command = quote_windows_argument(executable.native());
  for (const auto& argument : arguments) {
    command.push_back(L' ');
    command += quote_windows_argument(argument);
  }
  command.push_back(L'\0');
  STARTUPINFOW startup{
      .cb = sizeof(STARTUPINFOW),
      .dwFlags = STARTF_USESTDHANDLES,
      .hStdInput = input,
      .hStdOutput = output,
      .hStdError = output,
  };
  PROCESS_INFORMATION process{};
  WindowsProcessResult result;
  result.started = CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
                                  CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process) != 0;
  (void)CloseHandle(input);
  if (!result.started) {
    (void)CloseHandle(output);
    return result;
  }
  (void)CloseHandle(process.hThread);
  if (WaitForSingleObject(process.hProcess, 30000) != WAIT_OBJECT_0) {
    (void)TerminateProcess(process.hProcess, 94);
    (void)WaitForSingleObject(process.hProcess, 5000);
  }
  (void)GetExitCodeProcess(process.hProcess, &result.exit_code);
  (void)CloseHandle(process.hProcess);
  (void)CloseHandle(output);
  result.output = read_text_file(output_path);
  return result;
}
#else
bool read_byte_with_timeout(int descriptor, char& value, std::chrono::milliseconds timeout) {
  pollfd event{.fd = descriptor, .events = POLLIN};
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining <= std::chrono::milliseconds::zero()) {
      return false;
    }
    const int result = ::poll(&event, 1, static_cast<int>(remaining.count()));
    if (result > 0) {
      return (event.revents & POLLIN) != 0 && ::read(descriptor, &value, 1) == 1;
    }
    if (result == 0) {
      return false;
    }
    if (errno != EINTR) {
      return false;
    }
  }
}

bool wait_child_success_with_timeout(pid_t process, std::chrono::milliseconds timeout) {
  if (process <= 0) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t result = ::waitpid(process, &status, WNOHANG);
    if (result == process) {
      return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    if (result < 0 && errno != EINTR) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  (void)::kill(process, SIGKILL);
  while (::waitpid(process, &status, 0) < 0 && errno == EINTR) {
  }
  return false;
}
#endif

enum class AtomicReadStatus { Success, RetryableFailure, Missing, UnexpectedFailure };

struct AtomicReadResult {
  AtomicReadStatus status = AtomicReadStatus::UnexpectedFailure;
  std::string contents;
};

AtomicReadResult read_atomic_output(const std::filesystem::path& path) {
#if defined(_WIN32)
  const HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    const auto error = GetLastError();
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) {
      return {.status = AtomicReadStatus::Missing};
    }
    if (error == ERROR_SHARING_VIOLATION || error == ERROR_LOCK_VIOLATION ||
        error == ERROR_ACCESS_DENIED) {
      return {.status = AtomicReadStatus::RetryableFailure};
    }
    return {.status = AtomicReadStatus::UnexpectedFailure};
  }
  AtomicReadResult result{.status = AtomicReadStatus::Success};
  std::array<char, 4096> buffer{};
  for (;;) {
    DWORD bytes_read = 0;
    if (ReadFile(handle, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_read, nullptr) ==
        0) {
      result.status = AtomicReadStatus::UnexpectedFailure;
      break;
    }
    result.contents.append(buffer.data(), bytes_read);
    if (bytes_read == 0) {
      break;
    }
  }
  if (CloseHandle(handle) == 0) {
    result.status = AtomicReadStatus::UnexpectedFailure;
  }
  return result;
#else
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return {.status =
                errno == ENOENT ? AtomicReadStatus::Missing : AtomicReadStatus::UnexpectedFailure};
  }
  AtomicReadResult result{.status = AtomicReadStatus::Success};
  std::array<char, 4096> buffer{};
  for (;;) {
    const auto bytes_read = ::read(descriptor, buffer.data(), buffer.size());
    if (bytes_read > 0) {
      result.contents.append(buffer.data(), static_cast<std::size_t>(bytes_read));
      continue;
    }
    if (bytes_read == 0) {
      break;
    }
    if (errno == EINTR) {
      continue;
    }
    result.status = AtomicReadStatus::UnexpectedFailure;
    break;
  }
  if (::close(descriptor) != 0) {
    result.status = AtomicReadStatus::UnexpectedFailure;
  }
  return result;
#endif
}

std::size_t retained_io_resource_count() {
#if defined(_WIN32)
  DWORD count = 0;
  if (GetProcessHandleCount(GetCurrentProcess(), &count) == 0) {
    throw std::runtime_error("cannot count process handles");
  }
  return count;
#else
#if defined(__APPLE__)
  constexpr const char* descriptor_directory = "/dev/fd";
#else
  constexpr const char* descriptor_directory = "/proc/self/fd";
#endif
  return static_cast<std::size_t>(
      std::distance(std::filesystem::directory_iterator(descriptor_directory),
                    std::filesystem::directory_iterator{}));
#endif
}

void make_executable(const std::filesystem::path& path) {
#if !defined(_WIN32)
  std::error_code error;
  std::filesystem::permissions(
      path,
      std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
          std::filesystem::perms::owner_exec | std::filesystem::perms::group_read |
          std::filesystem::perms::group_exec | std::filesystem::perms::others_read |
          std::filesystem::perms::others_exec,
      std::filesystem::perm_options::replace, error);
  if (error) {
    throw std::filesystem::filesystem_error("cannot make test file executable", path, error);
  }
#else
  (void)path;
#endif
}

std::filesystem::path canonical_path(const std::filesystem::path& path) {
  std::error_code error;
  auto canonical = std::filesystem::weakly_canonical(path, error);
  return error ? std::filesystem::absolute(path) : canonical;
}

#if defined(__linux__)
std::filesystem::path find_shared_library_runfile(std::string_view needle) {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir == nullptr || test_srcdir[0] == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(test_srcdir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (filename.find(needle) != std::string::npos &&
        (entry.path().extension() == ".so" || entry.path().extension() == ".dylib" ||
         entry.path().extension() == ".dll")) {
      return entry.path();
    }
  }
  throw std::runtime_error("shared-library test runfile not found: " + std::string(needle));
}
#endif

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_no_stitch_output_artifacts(const std::filesystem::path& directory,
                                       std::string_view context) {
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const auto filename = entry.path().filename().string();
    const bool is_staging = filename.starts_with(".reco-stitch-");
    const bool is_legacy_temporary = filename.find(".tmp.") != std::string::npos;
    const bool is_legacy_publication = filename.find(".publish.") != std::string::npos;
    expect_true(!is_staging && !is_legacy_temporary && !is_legacy_publication,
                std::string(context) + " leaves no stitch output staging artifact");
  }
}

void interrupt_request_unwinds_stitch_output_staging() {
  TemporaryDirectory root;
  const auto destination = root.path() / "cancelled.mp4";
  write_text_file(destination, "existing destination\n");

#if defined(_WIN32)
  {
    detail::InterruptMonitor interrupts;
    {
      detail::AtomicOutputFile output(destination);
      write_text_descriptor(output.descriptor(), "partial output\n");
      expect_true(std::raise(SIGINT) == 0, "Windows CRT interrupt is delivered");
      expect_true(interrupts.requested(), "Windows interrupt monitor records cancellation");
    }
  }
#else
  int ready[2]{};
  if (::pipe(ready) != 0) {
    throw std::system_error(errno, std::generic_category(), "cannot create interrupt test pipe");
  }
  const pid_t child = ::fork();
  if (child == 0) {
    (void)::close(ready[0]);
    int status = 91;
    try {
      detail::InterruptMonitor interrupts;
      {
        detail::AtomicOutputFile output(destination);
        write_text_descriptor(output.descriptor(), "partial output\n");
        const char value = 1;
        if (::write(ready[1], &value, 1) != 1) {
          throw std::runtime_error("cannot signal interrupt child readiness");
        }
        while (!interrupts.requested()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
      status = kCancelledExitCode;
    } catch (...) {
      status = 92;
    }
    (void)::close(ready[1]);
    ::_exit(status);
  }
  (void)::close(ready[1]);
  char value = 0;
  const bool started =
      child > 0 && read_byte_with_timeout(ready[0], value, std::chrono::seconds(5));
  (void)::close(ready[0]);
  expect_true(started, "interrupt child creates its retained staging output");
  if (started) {
    expect_true(::kill(child, SIGINT) == 0, "SIGINT is delivered to the stitch child");
  }

  int child_status = 0;
  bool exited = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (child > 0 && std::chrono::steady_clock::now() < deadline) {
    const auto result = ::waitpid(child, &child_status, WNOHANG);
    if (result == child) {
      exited = true;
      break;
    }
    if (result < 0 && errno != EINTR) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!exited && child > 0) {
    (void)::kill(child, SIGKILL);
    while (::waitpid(child, &child_status, 0) < 0 && errno == EINTR) {
    }
  }
  expect_true(exited, "SIGINT child exits within the cancellation bound");
  expect_true(exited && WIFEXITED(child_status) && WEXITSTATUS(child_status) == kCancelledExitCode,
              "SIGINT child unwinds with the cancellation status");
#endif

  expect_true(read_text_file(destination) == "existing destination\n",
              "cancellation preserves the existing destination");
  expect_no_stitch_output_artifacts(root.path(), "cancellation");
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

Command expect_command(std::variant<Command, ParseError> parsed, std::string_view message) {
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    std::cerr << "FAIL: " << message << " unexpected error=" << error->message << '\n';
    ++failures;
    return HelpCommand{};
  }
  return std::get<Command>(std::move(parsed));
}

std::string valid_calibration_json() {
  return R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
}

std::filesystem::path write_valid_calibration_file() {
  const auto dir = std::filesystem::temp_directory_path() / "reco_cli_tests";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto path = dir / "match.json";
  std::ofstream output(path, std::ios::binary);
  output << valid_calibration_json();
  return path;
}

void production_windows_cli_preserves_unicode_paths() {
#if defined(_WIN32)
  TemporaryDirectory root;
  const auto calibration = root.path() / std::filesystem::path(L"match-Ω-例.json");
  write_text_file(calibration, valid_calibration_json());
  const auto left = root.path() / std::filesystem::path(L"left-Ω-例.mp4");
  const auto right = root.path() / std::filesystem::path(L"right-例-Ω.mp4");
  const auto output = root.path() / std::filesystem::path(L"output-Ω-例.mp4");
  const auto executable = find_windows_reco_runfile();
  expect_true(executable.has_value(), "production Windows reco.exe runfile is available");
  if (!executable.has_value()) {
    return;
  }
  const auto result = run_windows_process_capture(*executable,
                                                  {L"stitch", left.native(), right.native(), L"-c",
                                                   calibration.native(), L"-o", output.native()},
                                                  root.path() / "reco-output.txt");
  expect_true(result.started, "production Windows reco.exe starts through CreateProcessW");
  expect_eq(result.exit_code, static_cast<DWORD>(2),
            "production Windows stitch plan remains intentionally blocked");
  expect_true(result.output.find(wide_to_utf8(left.filename().native())) != std::string::npos,
              "production Windows CLI preserves the Unicode left path");
  expect_true(result.output.find(wide_to_utf8(right.filename().native())) != std::string::npos,
              "production Windows CLI preserves the Unicode right path");
  expect_true(result.output.find(wide_to_utf8(output.native())) != std::string::npos,
              "production Windows CLI preserves the Unicode output path");
#endif
}

void expect_error(const std::variant<Command, ParseError>& parsed, std::string_view message) {
  if (!std::holds_alternative<ParseError>(parsed)) {
    std::cerr << "FAIL: " << message << " expected parse error\n";
    ++failures;
  }
}

void validators_match_rust() {
  const auto blend = parse_blend("0.25");
  expect_true(std::holds_alternative<float>(blend), "blend parses");
  expect_near(std::get<float>(blend), 0.25F, 1.0e-6F, "blend value");
  expect_true(std::holds_alternative<float>(parse_blend("0.0")), "blend lower inclusive");
  expect_true(std::holds_alternative<float>(parse_blend("1.0")), "blend upper inclusive");
  expect_true(std::holds_alternative<ParseError>(parse_blend("-0.1")), "blend below rejected");
  expect_true(std::holds_alternative<ParseError>(parse_blend("1.1")), "blend above rejected");
  expect_true(std::holds_alternative<ParseError>(parse_blend("NaN")), "blend nan rejected");
  expect_true(std::holds_alternative<ParseError>(parse_blend("0.5x")), "blend suffix rejected");

  const auto wxh = parse_wxh("1280x720");
  expect_true(std::holds_alternative<WxH>(wxh), "wxh parses");
  expect_eq(std::get<WxH>(wxh).width, 1280U, "wxh width");
  expect_eq(std::get<WxH>(wxh).height, 720U, "wxh height");
  expect_true(std::holds_alternative<WxH>(parse_wxh("856X480")), "wxh uppercase separator");
  expect_true(std::holds_alternative<ParseError>(parse_wxh("0x720")), "zero width rejected");
  expect_true(std::holds_alternative<ParseError>(parse_wxh("854x480")), "width alignment rejected");
  expect_true(std::holds_alternative<ParseError>(parse_wxh("1280x721")), "height parity rejected");
}

void stitch_parse_matches_rust_defaults() {
  const auto command =
      expect_command(parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json",
                                 "--sync-offset", "-3", "--replay-scale", "1280x720",
                                 "--quality-value", "80", "--allow-no-tracking", "--no-zero-copy"}),
                     "stitch parse");
  const auto* stitch = std::get_if<StitchCommand>(&command);
  expect_true(stitch != nullptr, "stitch variant");
  if (stitch == nullptr)
    return;
  expect_eq(stitch->left, std::string("left.mp4"), "stitch left");
  expect_eq(stitch->right, std::string("right.mp4"), "stitch right");
  expect_eq(stitch->calibration, std::string("match.json"), "stitch calibration");
  expect_eq(stitch->output, std::string("output.mp4"), "stitch output default");
  expect_eq(stitch->width, 1920U, "stitch width default");
  expect_eq(stitch->height, 1080U, "stitch height default");
  expect_near(stitch->blend, 0.15F, 1.0e-6F, "stitch blend default");
  expect_eq(stitch->sync_offset, -3, "stitch negative sync");
  expect_eq(stitch->codec, std::string("h264"), "stitch codec default");
  expect_eq(stitch->quality, std::string("balanced"), "stitch quality default");
  expect_eq(stitch->tracking, std::string("field"), "stitch tracking default");
  expect_true(stitch->replay_scale.has_value(), "stitch replay scale");
  expect_eq(stitch->replay_scale->width, 1280U, "stitch replay width");
  expect_eq(stitch->quality_value.value_or(0), 80U, "stitch quality value");
  expect_true(stitch->allow_no_tracking, "stitch allow no tracking");
  expect_true(stitch->no_zero_copy, "stitch no zero copy");
}

void stitch_frame_timing_preserves_source_gaps_and_rejects_overflow() {
  const auto first = detail::derive_stitch_frame_timing(100, 100, 30, 1);
  const auto after_gap = detail::derive_stitch_frame_timing(102, 100, 30, 1);
  expect_eq(first.pts_ns, 0ULL, "first aligned source frame starts at output zero");
  expect_eq(first.duration_ns, 33'333'333ULL, "frame duration uses exact rational cadence");
  expect_eq(after_gap.pts_ns, 66'666'666ULL,
            "missing aligned source frame remains a timestamp gap");
  expect_true(after_gap.pts_ns > first.pts_ns + first.duration_ns,
              "source gap is not compressed to emitted-frame count");

  bool backwards_rejected = false;
  try {
    (void)detail::derive_stitch_frame_timing(99, 100, 30, 1);
  } catch (const std::runtime_error&) {
    backwards_rejected = true;
  }
  expect_true(backwards_rejected, "backwards source frame index is rejected");

  bool overflow_rejected = false;
  try {
    (void)detail::derive_stitch_frame_timing(std::numeric_limits<std::uint64_t>::max(), 0, 30, 1);
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  expect_true(overflow_rejected, "terminal source frame index overflow is rejected");
}

void stitch_audio_is_clipped_by_presentation_time_at_the_video_boundary() {
  const auto crossing = detail::clip_stitch_audio_duration(90U, 80U, 20U, 100U);
  expect_eq(crossing.value_or(0), 10ULL,
            "terminal audio duration is clipped from its presentation timestamp");

  const auto outside = detail::clip_stitch_audio_duration(100U, 90U, 20U, 100U);
  expect_true(!outside.has_value(),
              "decode timestamp does not admit audio presented at the video boundary");

  const auto dts_fallback = detail::clip_stitch_audio_duration(std::nullopt, 95U, 10U, 100U);
  expect_eq(dts_fallback.value_or(0), 5ULL,
            "decode timestamp is used only when presentation time is unavailable");

  const auto contained = detail::clip_stitch_audio_duration(40U, 30U, 20U, 100U);
  expect_eq(contained.value_or(0), 20ULL, "fully contained audio duration is unchanged");

  bool unknown_duration_rejected = false;
  try {
    (void)detail::clip_stitch_audio_duration(40U, 30U, 0U, 100U);
  } catch (const std::runtime_error& error) {
    unknown_duration_rejected =
        std::string_view(error.what()).find("duration is unknown") != std::string_view::npos;
  }
  expect_true(unknown_duration_rejected,
              "unknown compressed duration fails closed at the final boundary");
}

void stitch_second_conversion_supports_the_unsigned_gstreamer_range() {
  expect_eq(detail::nanoseconds_from_seconds(0.000'000'000'5, "timestamp"), 1ULL,
            "sub-nanosecond stitch time rounds to the nearest nanosecond");
  expect_eq(detail::nanoseconds_from_seconds(10'000'000'000.0, "timestamp"),
            10'000'000'000'000'000'000ULL,
            "stitch time supports timestamps above the signed 64-bit range");

  constexpr double kExclusiveUint64Seconds = 18'446'744'073.709'551'616;
  const auto largest_representable_seconds = std::nextafter(kExclusiveUint64Seconds, 0.0);
  expect_true(detail::nanoseconds_from_seconds(largest_representable_seconds, "timestamp") >
                  static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
              "largest representable stitch time remains in the unsigned range");
  bool overflow_rejected = false;
  try {
    (void)detail::nanoseconds_from_seconds(kExclusiveUint64Seconds, "timestamp");
  } catch (const std::runtime_error& error) {
    overflow_rejected =
        std::string_view(error.what()).find("GStreamer time range") != std::string_view::npos;
  }
  expect_true(overflow_rejected, "stitch time rejects the exclusive unsigned boundary");
}

void stitch_frame_window_uses_one_rounded_timeline() {
  const auto window = detail::derive_stitch_frame_window(0.051, 0.151, std::nullopt, 30, 1);
  expect_eq(window.start_frame, 2ULL, "stitch start time rounds to the nearest source frame");
  expect_eq(window.start_time_ns, 66'666'666ULL,
            "audio trim uses the same rounded frame boundary as video");
  expect_eq(window.frame_limit.value_or(0), 3ULL,
            "stitch end duration matches Rust nearest-frame rounding");

  const auto bounded = detail::derive_stitch_frame_window(0.0, 1.0, 7U, 30, 1);
  expect_eq(bounded.frame_limit.value_or(0), 7ULL, "max frames bounds the rounded time window");
  bool empty_window_rejected = false;
  try {
    (void)detail::derive_stitch_frame_window(0.0, 0.01, std::nullopt, 30, 1);
  } catch (const std::runtime_error&) {
    empty_window_rejected = true;
  }
  expect_true(empty_window_rejected, "sub-frame end window is rejected explicitly");

  expect_eq(detail::stitch_timeline_duration_ns(3, 30'000, 1'001), 100'100'000ULL,
            "audio segment duration follows the exact rational frame boundary");
  expect_eq(detail::stitch_timeline_duration_ns(1'001, 30'000, 1'001), 33'400'033'333ULL,
            "chained segment duration does not inherit container duration drift");
}

void stitch_descriptor_budget_is_checked_before_input_acquisition() {
  const auto required = detail::stitch_descriptor_requirement(8192);
  expect_eq(required, std::size_t{8192 + 1 + detail::kStitchTransientDescriptorReserve},
            "stitch descriptor estimate retains one authority per segment");
  expect_true(detail::stitch_descriptor_budget_fits(7, required + 7, 8192),
              "stitch descriptor budget accepts the exact available boundary");
  expect_true(!detail::stitch_descriptor_budget_fits(7, required + 6, 8192),
              "stitch descriptor budget rejects one descriptor below the boundary");

  bool overflow_rejected = false;
  try {
    (void)detail::stitch_descriptor_requirement(std::numeric_limits<std::size_t>::max());
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  expect_true(overflow_rejected, "stitch descriptor estimate rejects arithmetic overflow");

#if !defined(_WIN32)
  struct rlimit original{};
  if (::getrlimit(RLIMIT_NOFILE, &original) != 0) {
    throw std::runtime_error("cannot inspect the test descriptor limit");
  }
  struct RestoreLimit {
    struct rlimit value{};
    ~RestoreLimit() { (void)::setrlimit(RLIMIT_NOFILE, &value); }
  } restore{original};
  struct rlimit constrained = original;
  constrained.rlim_cur = std::min<rlim_t>(original.rlim_cur, 32);
  if (::setrlimit(RLIMIT_NOFILE, &constrained) != 0) {
    throw std::runtime_error("cannot constrain the test descriptor limit");
  }
  const auto descriptor_count = [&] {
    std::size_t count = 0;
    for (int descriptor = 0; descriptor < static_cast<int>(constrained.rlim_cur); ++descriptor) {
      if (::fcntl(descriptor, F_GETFD) >= 0 || errno != EBADF) {
        ++count;
      }
    }
    return count;
  };
  const auto before = descriptor_count();

  for (int attempt = 0; attempt < 16; ++attempt) {
    bool rejected = false;
    try {
      detail::require_stitch_descriptor_budget(2);
    } catch (const std::runtime_error& error) {
      rejected = std::string_view(error.what()).find("raise the process descriptor limit") !=
                 std::string_view::npos;
    }
    expect_true(rejected, "low descriptor limit is rejected without partial acquisition");
  }
  expect_eq(descriptor_count(), before,
            "repeated low-limit admission failures do not leak descriptors");
#endif
}

void stitch_output_transaction_is_descriptor_pinned_and_atomic() {
  TemporaryDirectory root;
  const auto destination = root.path() / "stitched.mp4";

  const auto descriptor_budget_input = root.path() / "descriptor-budget-input.mp4";
  write_text_file(descriptor_budget_input, "descriptor budget input\n");
  const auto retained_input = reco::io::StableMediaFile::open(descriptor_budget_input);
  std::vector<detail::AtomicOutputProtectedPath> repeated_protected_paths;
  repeated_protected_paths.reserve(128);
  for (int index = 0; index < 128; ++index) {
    repeated_protected_paths.push_back({.path = descriptor_budget_input,
                                        .label = "the repeated protected input",
                                        .stable_source = retained_input});
  }
  const auto retained_resources_before = retained_io_resource_count();
  {
    detail::AtomicOutputFile output(root.path() / "descriptor-budget-output.mp4", {}, {},
                                    repeated_protected_paths);
    expect_true(retained_io_resource_count() <= retained_resources_before + 8,
                "publication protection reuses retained input authorities");
    write_text_descriptor(output.descriptor(), "descriptor budget output\n");
  }
  expect_true(retained_io_resource_count() <= retained_resources_before + 1,
              "publication protection cleanup releases transaction resources");

  {
    detail::AtomicOutputFile output(destination);
    write_text_descriptor(output.descriptor(), "first encoded output\n");
    expect_true(output.descriptor() >= 0, "stitch output exposes a retained descriptor");
    const auto verification = output.verification_source();
    expect_eq(verification->read_all(4096), std::string("first encoded output\n"),
              "verification stream reads independently from the write descriptor");
#if defined(_WIN32)
    const HANDLE ordinary_reader = CreateFileW(output.temporary_path().c_str(), GENERIC_READ,
                                               FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    expect_true(ordinary_reader != INVALID_HANDLE_VALUE,
                "Windows verification does not require readers to share delete access");
    if (ordinary_reader != INVALID_HANDLE_VALUE) {
      (void)CloseHandle(ordinary_reader);
    }
#endif
    output.commit();
    output.commit();
  }
  expect_eq(read_text_file(destination), std::string("first encoded output\n"),
            "stitch output transaction publishes a new file and repeated commit is a no-op");
  expect_no_stitch_output_artifacts(root.path(), "new stitch output publication");
  expect_eq(std::filesystem::hard_link_count(destination), std::uintmax_t{1},
            "new stitch output has exactly one directory entry");

#if defined(_WIN32)
  const auto integrity_destination = root.path() / "write-locked-stitch.mp4";
  bool publication_hook_ran = false;
  bool publication_writer_denied = false;
  {
    detail::AtomicOutputFile output(
        integrity_destination, [&](const std::filesystem::path& publication) {
          publication_hook_ran = true;
          SetLastError(ERROR_SUCCESS);
          const HANDLE competing_writer =
              CreateFileW(publication.c_str(), GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
          const DWORD writer_error = GetLastError();
          publication_writer_denied =
              competing_writer == INVALID_HANDLE_VALUE && writer_error == ERROR_SHARING_VIOLATION;
          if (competing_writer != INVALID_HANDLE_VALUE) {
            constexpr char attacker_contents[] = "mutated after verification\n";
            DWORD written = 0;
            (void)WriteFile(competing_writer, attacker_contents,
                            static_cast<DWORD>(sizeof(attacker_contents) - 1U), &written, nullptr);
            (void)CloseHandle(competing_writer);
          }
        });
    write_text_descriptor(output.descriptor(), "verified encoded output\n");
    const auto verification = output.verification_source();
    expect_eq(verification->read_all(4096), std::string("verified encoded output\n"),
              "Windows verification observes the descriptor-bound bytes");

    SetLastError(ERROR_SUCCESS);
    const HANDLE competing_writer =
        CreateFileW(output.temporary_path().c_str(), GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr);
    const DWORD writer_error = GetLastError();
    expect_true(competing_writer == INVALID_HANDLE_VALUE && writer_error == ERROR_SHARING_VIOLATION,
                "Windows temporary rejects a second writer after verification");
    if (competing_writer != INVALID_HANDLE_VALUE) {
      (void)CloseHandle(competing_writer);
    }
    output.commit();
  }
  expect_true(publication_hook_ran,
              "Windows write-lock fixture reaches the final publication window");
  expect_true(publication_writer_denied,
              "Windows temporary rejects mutation at the publication boundary");
  expect_eq(read_text_file(integrity_destination), std::string("verified encoded output\n"),
            "Windows publication preserves the bytes that were verified");
  expect_no_stitch_output_artifacts(root.path(), "write-locked Windows stitch publication");
#endif

  {
    detail::AtomicOutputFile output(destination);
    write_text_descriptor(output.descriptor(), "replacement encoded output\n");
    output.commit();
  }
  expect_eq(read_text_file(destination), std::string("replacement encoded output\n"),
            "stitch output transaction atomically replaces an existing file");
  expect_no_stitch_output_artifacts(root.path(), "replacement stitch output publication");
  expect_eq(std::filesystem::hard_link_count(destination), std::uintmax_t{1},
            "replacement stitch output does not retain a temporary hard link");

  bool injected_failure_reported = false;
  try {
    detail::AtomicOutputFile output(
        destination, {}, [] { throw std::runtime_error("injected stitch publication failure"); });
    write_text_descriptor(output.descriptor(), "output that must roll back\n");
    output.commit();
  } catch (const std::runtime_error& error) {
    injected_failure_reported =
        std::string_view(error.what()).find("injected stitch publication failure") !=
        std::string_view::npos;
  }
  expect_true(injected_failure_reported, "stitch publication reports an injected failure");
  expect_eq(read_text_file(destination), std::string("replacement encoded output\n"),
            "failed stitch replacement restores the previous output");
  expect_no_stitch_output_artifacts(root.path(), "failed stitch output publication");
  expect_eq(std::filesystem::hard_link_count(destination), std::uintmax_t{1},
            "failed stitch replacement leaves one destination entry");

  const auto protected_input = root.path() / "protected-input.mp4";
  const auto aliased_output = root.path() / "aliased-output.mp4";
  write_text_file(protected_input, "protected input\n");
  std::filesystem::create_hard_link(protected_input, aliased_output);
  bool protected_alias_rejected = false;
  try {
    const std::array protected_paths{
        detail::AtomicOutputProtectedPath{protected_input, "the protected input"}};
    detail::AtomicOutputFile output(aliased_output, {}, {}, protected_paths);
  } catch (const std::runtime_error& error) {
    protected_alias_rejected = std::string_view(error.what()).find("aliases the protected input") !=
                               std::string_view::npos;
  }
  expect_true(protected_alias_rejected,
              "stitch output rejects a retained protected-input hard link");
  expect_eq(read_text_file(protected_input), std::string("protected input\n"),
            "rejected stitch output alias preserves its protected input");
  std::filesystem::remove(aliased_output);
  expect_no_stitch_output_artifacts(root.path(), "rejected protected stitch output alias");

#if !defined(_WIN32)
  const auto moved_protected_input = root.path() / "moved-protected-input.mp4";
  const auto raced_output = root.path() / "protected-race-output.mp4";
  bool protected_mutation_rejected = false;
  try {
    const std::array protected_paths{
        detail::AtomicOutputProtectedPath{protected_input, "the protected input"}};
    detail::AtomicOutputFile output(
        raced_output, {},
        [&] {
          std::filesystem::rename(protected_input, moved_protected_input);
          write_text_file(protected_input, "replacement input\n");
        },
        protected_paths);
    write_text_descriptor(output.descriptor(), "raced output\n");
    output.commit();
  } catch (const std::runtime_error& error) {
    protected_mutation_rejected =
        std::string_view(error.what()).find("changed before") != std::string_view::npos;
  }
  expect_true(protected_mutation_rejected,
              "stitch publication rejects a protected path changed at commit");
  expect_true(!std::filesystem::exists(raced_output),
              "protected path mutation rolls back a newly published stitch output");
  expect_no_stitch_output_artifacts(root.path(), "protected path mutation rollback");
  std::filesystem::remove(protected_input);
  std::filesystem::rename(moved_protected_input, protected_input);
#endif

  const auto retained_output_parent = root.path() / "retained-stitch-parent";
  const auto redirected_output_parent = root.path() / "redirected-stitch-parent";
  const auto active_output_parent = root.path() / "active-stitch-parent";
  std::filesystem::create_directory(retained_output_parent);
  std::filesystem::create_directory(redirected_output_parent);
  const auto redirected_victim = redirected_output_parent / "parent-race.mp4";
  write_text_file(redirected_victim, "redirected victim\n");
  std::error_code output_parent_symlink_error;
  std::filesystem::create_directory_symlink(retained_output_parent, active_output_parent,
                                            output_parent_symlink_error);
  expect_true(!output_parent_symlink_error,
              "stitch output parent symlink race fixture is available");
  if (!output_parent_symlink_error) {
    const std::array protected_paths{
        detail::AtomicOutputProtectedPath{redirected_victim, "the redirected victim"}};
    detail::AtomicOutputFile output(active_output_parent / "parent-race.mp4", {}, {},
                                    protected_paths);
    write_text_descriptor(output.descriptor(), "retained parent output\n");
    std::filesystem::remove(active_output_parent);
    std::filesystem::create_directory_symlink(redirected_output_parent, active_output_parent);
    const auto verification = output.verification_source();
    expect_eq(verification->read_all(4096), std::string("retained parent output\n"),
              "verification follows the retained output parent after path retargeting");
    output.commit();
    expect_eq(read_text_file(retained_output_parent / "parent-race.mp4"),
              std::string("retained parent output\n"),
              "stitch publication stays in its descriptor-pinned parent");
    expect_eq(read_text_file(redirected_victim), std::string("redirected victim\n"),
              "redirected output parent cannot replace a protected file");
  }

#if !defined(_WIN32)
  const auto staging_destination = root.path() / "staging-race.mp4";
  const auto retained_staging = root.path() / "retained-staging";
  std::filesystem::path substituted_staging;
  {
    detail::AtomicOutputFile output(staging_destination);
    write_text_descriptor(output.descriptor(), "retained staging output\n");
    substituted_staging = output.temporary_path().parent_path();
    std::filesystem::rename(substituted_staging, retained_staging);
    std::filesystem::create_directory(substituted_staging);
    write_text_file(substituted_staging / output.temporary_path().filename(),
                    "staging pathname substitute\n");
    const auto verification = output.verification_source();
    expect_eq(verification->read_all(4096), std::string("retained staging output\n"),
              "verification opens the retained staging directory identity");
    output.commit();
  }
  expect_eq(read_text_file(staging_destination), std::string("retained staging output\n"),
            "publication moves the verified retained staging identity");
  expect_eq(read_text_file(substituted_staging / "output"),
            std::string("staging pathname substitute\n"),
            "retained publication leaves the substituted staging tree untouched");
  std::filesystem::remove_all(substituted_staging);
  std::filesystem::remove_all(retained_staging);
#endif

  std::vector<std::filesystem::path> retained_windows_temporaries;
  const auto remove_transaction_temporary = [&](const std::filesystem::path& publication) {
#if defined(_WIN32)
    auto retained = publication;
    retained += ".retained";
    std::filesystem::rename(publication, retained);
    retained_windows_temporaries.push_back(std::move(retained));
#else
    std::filesystem::remove(publication);
#endif
  };
  const auto expect_no_retained_windows_temporary = [&] {
#if defined(_WIN32)
    expect_true(!retained_windows_temporaries.empty() &&
                    !std::filesystem::exists(retained_windows_temporaries.back()),
                "Windows stitch cleanup removes the renamed descriptor-bound temporary");
#endif
  };

  std::filesystem::path attacker_entry;
  bool substitution_rejected = false;
  {
    detail::AtomicOutputFile output(destination, [&](const std::filesystem::path& publication) {
      attacker_entry = publication;
      remove_transaction_temporary(publication);
      write_text_file(publication, "attacker replacement\n");
    });
    write_text_file(output.temporary_path(), "descriptor-bound encoded output\n");
    try {
      output.commit();
    } catch (const std::runtime_error& error) {
      substitution_rejected =
          std::string_view(error.what()).find("publication boundary") != std::string_view::npos;
    }
  }
  expect_true(substitution_rejected,
              "stitch publication rejects a substituted temporary directory entry");
  expect_eq(read_text_file(destination), std::string("replacement encoded output\n"),
            "rejected stitch publication preserves the existing destination");
  expect_true(!std::filesystem::exists(attacker_entry),
              "stitch cleanup removes a substituted temporary file");
  expect_no_retained_windows_temporary();
  expect_no_stitch_output_artifacts(root.path(), "substituted stitch temporary file cleanup");

  std::filesystem::path attacker_directory;
  bool directory_substitution_rejected = false;
  {
    detail::AtomicOutputFile output(destination, [&](const std::filesystem::path& publication) {
      attacker_directory = publication;
      remove_transaction_temporary(publication);
      std::filesystem::create_directory(publication);
    });
    write_text_descriptor(output.descriptor(), "descriptor-bound directory output\n");
    try {
      output.commit();
    } catch (const std::runtime_error& error) {
      directory_substitution_rejected =
          std::string_view(error.what()).find("temporary") != std::string_view::npos;
    }
  }
  expect_true(directory_substitution_rejected,
              "stitch publication rejects a substituted temporary directory");
  expect_true(!std::filesystem::exists(attacker_directory),
              "stitch cleanup removes a substituted empty directory");
  expect_no_retained_windows_temporary();
  expect_eq(read_text_file(destination), std::string("replacement encoded output\n"),
            "directory substitution preserves the existing destination");
  expect_no_stitch_output_artifacts(root.path(), "substituted stitch directory cleanup");

  const auto symlink_victim = root.path() / "stitch-cleanup-victim.mp4";
  const auto symlink_probe = root.path() / "stitch-cleanup-symlink-probe";
  write_text_file(symlink_victim, "protected victim content\n");
  std::error_code symlink_error;
  std::filesystem::create_symlink(symlink_victim, symlink_probe, symlink_error);
  if (!symlink_error) {
    std::filesystem::remove(symlink_probe);
    std::filesystem::path attacker_symlink;
    bool symlink_substitution_rejected = false;
    {
      const std::array protected_paths{
          detail::AtomicOutputProtectedPath{symlink_victim, "the cleanup victim"}};
      detail::AtomicOutputFile output(
          destination,
          [&](const std::filesystem::path& publication) {
            attacker_symlink = publication;
            remove_transaction_temporary(publication);
            std::filesystem::create_symlink(symlink_victim, publication);
          },
          {}, protected_paths);
      write_text_descriptor(output.descriptor(), "descriptor-bound symlink output\n");
      try {
        output.commit();
      } catch (const std::runtime_error& error) {
        symlink_substitution_rejected =
            std::string_view(error.what()).find("temporary") != std::string_view::npos;
      }
    }
    expect_true(symlink_substitution_rejected,
                "stitch publication rejects a substituted temporary symlink");
    expect_true(!std::filesystem::exists(attacker_symlink),
                "stitch cleanup removes a substituted symlink without following it");
    expect_no_retained_windows_temporary();
    expect_eq(read_text_file(symlink_victim), std::string("protected victim content\n"),
              "stitch cleanup preserves a substituted symlink victim");
    expect_eq(read_text_file(destination), std::string("replacement encoded output\n"),
              "symlink substitution preserves the existing destination");
    expect_no_stitch_output_artifacts(root.path(), "substituted stitch symlink cleanup");
  }

#if defined(__linux__)
  const auto descriptor_count = [] {
    return static_cast<std::size_t>(
        std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                      std::filesystem::directory_iterator{}));
  };
  const auto before = descriptor_count();
  const std::string oversized_name(8'192, 'x');
  for (int attempt = 0; attempt < 32; ++attempt) {
    try {
      detail::AtomicOutputFile output(root.path() / oversized_name);
      expect_true(false, "oversized stitch temporary unexpectedly opened");
    } catch (const std::system_error&) {
    }
  }
  const auto after = descriptor_count();
  expect_true(after <= before + 1U,
              "failed stitch output construction does not leak its directory descriptor");
#elif defined(_WIN32)
  const auto retained_parent = root.path() / "retained-output-parent";
  const auto redirected_parent = root.path() / "redirected-output-parent";
  const auto active_parent = root.path() / "active-output-parent";
  std::filesystem::create_directory(retained_parent);
  std::filesystem::create_directory(redirected_parent);
  std::error_code parent_symlink_error;
  std::filesystem::create_directory_symlink(retained_parent, active_parent, parent_symlink_error);
  expect_true(!parent_symlink_error, "Windows stitch output parent symlink fixture is available");
  if (!parent_symlink_error) {
    detail::AtomicOutputFile output(active_parent / "stitched.mp4");
    write_text_descriptor(output.descriptor(), "retained parent encoded output\n");
    std::filesystem::remove(active_parent);
    std::filesystem::create_directory_symlink(redirected_parent, active_parent);
    const auto verification = output.verification_source();
    expect_eq(verification->read_all(4096), std::string("retained parent encoded output\n"),
              "Windows verification follows the retained output directory");
    output.commit();
    expect_eq(read_text_file(retained_parent / "stitched.mp4"),
              std::string("retained parent encoded output\n"),
              "Windows stitch publication stays in the retained output directory");
    expect_true(!std::filesystem::exists(redirected_parent / "stitched.mp4"),
                "Windows redirected output parent receives no publication");
  }
#endif
}

void preview_and_calibrate_parse_matches_rust_defaults() {
  const auto preview_command =
      expect_command(parse_args({"preview", "l.mp4", "r.mp4", "--calibration", "match.json",
                                 "--rig-tilt", "-2.5"}),
                     "preview parse");
  const auto* preview = std::get_if<PreviewCommand>(&preview_command);
  expect_true(preview != nullptr, "preview variant");
  if (preview != nullptr) {
    expect_eq(preview->width, 1280U, "preview width default");
    expect_eq(preview->height, 720U, "preview height default");
    expect_near(preview->blend, 0.15F, 1.0e-6F, "preview blend default");
    expect_near(preview->rig_tilt, -2.5F, 1.0e-6F, "preview negative rig tilt");
  }

  const auto preview_unbounded_blend = expect_command(
      parse_args({"preview", "l.mp4", "r.mp4", "-c", "match.json", "--blend", "1.1"}),
      "preview unbounded blend parse");
  const auto* unbounded_preview = std::get_if<PreviewCommand>(&preview_unbounded_blend);
  expect_true(unbounded_preview != nullptr, "preview unbounded blend variant");
  if (unbounded_preview != nullptr) {
    expect_near(unbounded_preview->blend, 1.1F, 1.0e-6F, "preview blend is raw f32");
  }

  expect_error(parse_args({"preview", "l.mp4", "r.mp4", "-c", "match.json", "--blend", "NaN"}),
               "preview blend rejects nan");

  const auto calibrate_command =
      expect_command(parse_args({"calibrate", "l.mp4", "r.mp4", "--frames", "8", "--no-auto-imu",
                                 "--no-auto-sync", "--output", "out.json"}),
                     "calibrate parse");
  const auto* calibrate = std::get_if<CalibrateCommand>(&calibrate_command);
  expect_true(calibrate != nullptr, "calibrate variant");
  if (calibrate != nullptr) {
    expect_eq(calibrate->frames, 8U, "calibrate frames");
    expect_true(calibrate->no_auto_imu, "calibrate no auto imu");
    expect_true(!calibrate->auto_sync, "calibrate no auto sync");
    expect_eq(calibrate->output, std::string("out.json"), "calibrate output");
    expect_near(static_cast<float>(calibrate->akaze_threshold), 0.0001F, 1.0e-8F,
                "calibrate akaze default");
    expect_near(static_cast<float>(calibrate->lowe_ratio), 0.75F, 1.0e-6F,
                "calibrate lowe default");
  }
}

void live_command_parse_matches_rust_defaults() {
  const auto camera_command = expect_command(
      parse_args({"camera", "--left-device", "0", "--right-device", "1", "-c", "match.json", "-o",
                  "out.mp4", "--stream-url", "rtmp://example/live", "--replay-scale", "1280x720",
                  "--live-calibrate", "--left-lens-profile", "lens.json"}),
      "camera parse");
  const auto* camera = std::get_if<CameraCommand>(&camera_command);
  expect_true(camera != nullptr, "camera variant");
  if (camera != nullptr) {
    expect_eq(camera->left_device, std::string("0"), "camera left device");
    expect_eq(camera->right_device, std::string("1"), "camera right device");
    expect_eq(camera->capture_width, 3840U, "camera capture width default");
    expect_eq(camera->capture_height, 2160U, "camera capture height default");
    expect_eq(camera->capture_fps, 30U, "camera fps default");
    expect_eq(camera->width, 1920U, "camera output width default");
    expect_eq(camera->height, 1080U, "camera output height default");
    expect_eq(camera->quality, std::string("fast"), "camera quality default");
    expect_eq(camera->tracking, std::string("field"), "camera tracking default");
    expect_true(camera->stream_url.has_value(), "camera stream url");
    expect_true(camera->replay_scale.has_value(), "camera replay scale");
    expect_true(camera->live_calibrate, "camera live calibrate");
    expect_eq(camera->calibrate_frames, 8U, "camera calibrate frames default");
    expect_eq(camera->exposure, 780U, "camera exposure default");
    expect_eq(camera->sensor_gain, 16U, "camera sensor gain default");
    expect_eq(camera->left_lens_profile.value_or(""), std::string("lens.json"),
              "camera lens profile");
  }

  const auto libcamera_command = expect_command(
      parse_args({"libcamera", "-c", "match.json", "-o", "out.mp4", "--left-camera", "2",
                  "--right-camera", "3", "--quality-value", "70", "--preset", "fast"}),
      "libcamera parse");
  const auto* libcamera = std::get_if<LibcameraCommand>(&libcamera_command);
  expect_true(libcamera != nullptr, "libcamera variant");
  if (libcamera != nullptr) {
    expect_eq(libcamera->left_camera, 2U, "libcamera left camera");
    expect_eq(libcamera->right_camera, 3U, "libcamera right camera");
    expect_eq(libcamera->capture_width, 1920U, "libcamera capture width default");
    expect_eq(libcamera->capture_height, 1080U, "libcamera capture height default");
    expect_eq(libcamera->quality, std::string("fast"), "libcamera quality default");
    expect_eq(libcamera->quality_value.value_or(0), 70U, "libcamera quality value");
    expect_eq(libcamera->preset.value_or(""), std::string("fast"), "libcamera preset");
  }

  const auto gopro_command = expect_command(
      parse_args({"gopro", "--serial", "123", "--start", "--sports-preset"}), "gopro parse");
  const auto* gopro = std::get_if<GoproCommand>(&gopro_command);
  expect_true(gopro != nullptr, "gopro variant");
  if (gopro != nullptr) {
    expect_eq(gopro->serial.value_or(""), std::string("123"), "gopro serial");
    expect_true(gopro->start, "gopro start");
    expect_true(!gopro->stop, "gopro stop default");
    expect_true(gopro->sports_preset, "gopro sports preset");
  }
}

void parse_errors_are_reported() {
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4"}), "missing calibration");
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4", "-c"}), "missing option value");
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4", "-c", "--unknown"}),
               "string value cannot be another option");
  expect_error(parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json", "--output",
                           "--codec", "h264"}),
               "string value cannot consume valid option");
  expect_error(
      parse_args({"preview", "left.mp4", "right.mp4", "-c", "match.json", "--width", "1920x"}),
      "numeric suffix rejected");
  expect_error(
      parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json", "--quality-value", "256"}),
      "quality value u8 range rejected");
  expect_error(
      parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json", "--lookahead", "-1"}),
      "lookahead does not allow hyphen value");
  expect_error(
      parse_args({"stitch", "left.mp4", "right.mp4", "-c", "match.json", "--max-frames", "0"}),
      "zero-frame stitch is rejected before GPU setup");
  expect_error(parse_args({"calibrate", "left.mp4", "right.mp4", "--skip-start", "-1"}),
               "skip start does not allow hyphen value");
  expect_error(parse_args({"calibrate", "left.mp4", "right.mp4", "--akaze-threshold", "NaN"}),
               "calibrate rejects non-finite double");
  expect_error(parse_args({"info", "--verbose"}), "info rejects options");
  expect_error(parse_args({"camera", "--right-device", "1", "-c", "match.json", "-o", "out.mp4"}),
               "camera requires left device");
  expect_error(parse_args({"libcamera", "-c", "match.json"}), "libcamera requires output");
  expect_error(parse_args({"gopro", "--bogus"}), "gopro rejects unknown option");

  const auto help = expect_command(parse_args({"--help"}), "help parse");
  expect_true(std::holds_alternative<HelpCommand>(help), "help variant");
  const auto stitch_help = expect_command(parse_args({"stitch", "--help"}), "stitch help parse");
  expect_true(std::holds_alternative<HelpCommand>(stitch_help), "stitch help variant");
  const auto partial_stitch_help =
      expect_command(parse_args({"stitch", "left.mp4", "--help"}), "partial stitch help parse");
  expect_true(std::holds_alternative<HelpCommand>(partial_stitch_help),
              "partial stitch help variant");
  const auto camera_help =
      expect_command(parse_args({"camera", "--left-device", "0", "--help"}), "camera help parse");
  expect_true(std::holds_alternative<HelpCommand>(camera_help), "camera help variant");
  expect_true(
      help_text().find("calibrate LEFT RIGHT --left-profile PROFILE --no-auto-sync --no-auto-imu "
                       "[options]") != std::string::npos,
      "help advertises the currently executable calibration contract");
}

void probe_worker_discovery_handles_path_and_bzlmod_runfiles() {
#if defined(_WIN32)
  constexpr std::string_view cli_name = "reco.exe";
  constexpr std::string_view worker_name = "reco_video_probe_worker.exe";
  constexpr std::string_view calibration_name = "reco_calibration_worker.exe";
#else
  constexpr std::string_view cli_name = "reco";
  constexpr std::string_view worker_name = "reco_video_probe_worker";
  constexpr std::string_view calibration_name = "reco_calibration_worker";
#endif

  ScopedEnvironment no_override("RECO_VIDEO_PROBE_WORKER", std::nullopt);
  ScopedEnvironment no_calibration_override("RECO_CALIBRATION_WORKER", std::nullopt);

  const auto bazel_worker = detail::resolve_video_probe_worker("stage27-missing-reco");
  expect_true(bazel_worker.has_value(), "bzlmod runfiles resolve probe worker");
  if (bazel_worker.has_value()) {
    expect_eq(bazel_worker->filename().string(), std::string(worker_name),
              "bzlmod runfiles select probe worker target");
    expect_true(std::filesystem::is_regular_file(*bazel_worker), "bzlmod runfile worker exists");
  }
  const auto bazel_calibration = detail::resolve_calibration_worker("stage27-missing-reco");
  expect_true(bazel_calibration.has_value(), "bzlmod runfiles resolve calibration worker");
  if (bazel_calibration.has_value()) {
    expect_eq(bazel_calibration->filename().string(), std::string(calibration_name),
              "bzlmod runfiles select calibration worker target");
    expect_true(std::filesystem::is_regular_file(*bazel_calibration),
                "bzlmod calibration worker exists");
  }

  TemporaryDirectory runfiles_root;
  const auto mapped_worker = runfiles_root.path() / worker_name;
  const auto repo_mapping = runfiles_root.path() / "repo_mapping";
  const auto manifest = runfiles_root.path() / "MANIFEST";
  write_text_file(mapped_worker, "synthetic worker");
  make_executable(mapped_worker);
  write_text_file(repo_mapping, ",reco_video_stitcher,stage27_canonical\n");
  write_text_file(manifest, "stage27_canonical/cpp/reco_io/" + std::string(worker_name) + " " +
                                mapped_worker.string() + "\n_repo_mapping " +
                                repo_mapping.string() + "\n");
  {
    ScopedEnvironment use_manifest("RUNFILES_MANIFEST_FILE", manifest.string());
    ScopedEnvironment no_runfiles_directory("RUNFILES_DIR", std::nullopt);
    ScopedEnvironment empty_path("PATH", std::string{});
    const auto remapped = detail::resolve_video_probe_worker("stage27-missing-reco");
    expect_true(remapped.has_value(), "synthetic bzlmod repository mapping resolves worker");
    if (remapped.has_value()) {
      expect_eq(canonical_path(*remapped).string(), canonical_path(mapped_worker).string(),
                "runfiles lookup honors canonical repository mapping");
    }
  }

  TemporaryDirectory path_root;
  const auto bin = path_root.path() / "bin";
  const auto hostile_working_directory = path_root.path() / "working";
  std::filesystem::create_directories(bin);
  std::filesystem::create_directories(hostile_working_directory);
  const auto path_cli = bin / cli_name;
  const auto path_worker = bin / worker_name;
  const auto hostile_worker = hostile_working_directory / worker_name;
  write_text_file(path_cli, "PATH CLI");
  write_text_file(path_worker, "PATH worker");
  write_text_file(hostile_worker, "wrong working-directory worker");
  make_executable(path_cli);
  make_executable(path_worker);
  make_executable(hostile_worker);
  {
    ScopedEnvironment no_manifest("RUNFILES_MANIFEST_FILE", std::nullopt);
    ScopedEnvironment no_runfiles_directory("RUNFILES_DIR", std::nullopt);
    ScopedEnvironment no_test_srcdir("TEST_SRCDIR", std::nullopt);
    ScopedEnvironment path("PATH", bin.string());
    ScopedCurrentPath working_directory(hostile_working_directory);
    const auto resolved = detail::resolve_video_probe_worker(std::filesystem::path(cli_name));
    expect_true(resolved.has_value(), "ordinary PATH invocation resolves worker");
    if (resolved.has_value()) {
      expect_eq(canonical_path(*resolved).string(), canonical_path(path_worker).string(),
                "PATH executable directory wins over working directory");
    }
  }

#if defined(_WIN32)
  const auto unicode_bin =
      path_root.path() / reco::core::path_from_utf8("bin-\xCE\xA9-\xE4\xBE\x8B");
  std::filesystem::create_directories(unicode_bin);
  const auto unicode_worker = unicode_bin / worker_name;
  write_text_file(unicode_worker, "Unicode override worker");
  {
    ScopedWideEnvironment override_worker(L"RECO_VIDEO_PROBE_WORKER", unicode_worker);
    const auto resolved = detail::resolve_video_probe_worker("stage27-missing-reco");
    expect_true(resolved.has_value(), "native Unicode worker override resolves");
    if (resolved.has_value()) {
      expect_true(canonical_path(*resolved).native() == canonical_path(unicode_worker).native(),
                  "worker override preserves native Unicode path components");
    }
  }
#endif
}

void calibration_output_replacement_is_exclusive_and_atomic() {
  TemporaryDirectory root;
  const auto destination = root.path() / "match.json";
  const auto left_input = root.path() / "left.mp4";
  const auto right_input = root.path() / "right.mp4";
  const auto victim = root.path() / "victim.json";
  auto predictable_temporary = destination;
  predictable_temporary += ".tmp.0";
  write_text_file(left_input, "left video must not change\n");
  write_text_file(right_input, "right video must not change\n");
  write_text_file(victim, "victim must not change\n");

  const auto unicode_destination =
      root.path() / reco::core::path_from_utf8("match-\xCE\xA9-\xE4\xBE\x8B.json");
  const auto unicode_left = root.path() / reco::core::path_from_utf8("left-\xCE\xA9.mp4");
  const auto unicode_right = root.path() / reco::core::path_from_utf8("right-\xE4\xBE\x8B.mp4");
  write_text_file(unicode_left, "Unicode left video\n");
  write_text_file(unicode_right, "Unicode right video\n");
  detail::write_calibration_json_atomically(R"json({"writer":"unicode-path"})json",
                                            unicode_destination, unicode_left, unicode_right);
  expect_eq(read_text_file(unicode_destination), std::string("{\"writer\":\"unicode-path\"}\n"),
            "calibration publication uses native Unicode filesystem paths");

  std::error_code output_symlink_error;
  std::filesystem::create_symlink(victim, destination, output_symlink_error);
  if (output_symlink_error) {
    write_text_file(destination, "old output\n");
  }
  std::error_code temporary_symlink_error;
  std::filesystem::create_symlink(victim, predictable_temporary, temporary_symlink_error);
  if (temporary_symlink_error) {
    write_text_file(predictable_temporary, "predictable temporary guard\n");
  }
#if !defined(_WIN32)
  expect_true(!output_symlink_error, "destination symlink fixture is available");
  expect_true(!temporary_symlink_error, "temporary symlink fixture is available");
#endif

  try {
    detail::write_calibration_json_atomically(R"json({"writer":"symlink-test"})json", destination,
                                              left_input, right_input);
  } catch (const std::exception& error) {
#if defined(_WIN32)
    throw std::runtime_error(
        std::string(output_symlink_error ? "regular-file fallback: " : "destination symlink: ") +
        error.what());
#else
    throw;
#endif
  }
  expect_eq(read_text_file(destination), std::string("{\"writer\":\"symlink-test\"}\n"),
            "calibration output replaces destination atomically");
  expect_eq(read_text_file(victim), std::string("victim must not change\n"),
            "calibration output does not follow destination or temporary symlink");
  if (!output_symlink_error) {
    expect_true(!std::filesystem::is_symlink(destination),
                "calibration replacement replaces destination symlink itself");

    const auto rollback_destination = root.path() / "symlink-rollback.json";
    std::filesystem::create_symlink(victim, rollback_destination);
    bool rollback_hook_failed = false;
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"symlink-rollback"})json", rollback_destination, left_input, right_input,
          {}, {}, {}, false, [] { throw std::runtime_error("synthetic symlink rollback"); });
    } catch (const std::exception& error) {
      rollback_hook_failed = std::string_view(error.what()).find("synthetic symlink rollback") !=
                             std::string_view::npos;
      if (!rollback_hook_failed) {
        std::cerr << "symlink rollback failure: " << error.what() << '\n';
      }
    }
    expect_true(rollback_hook_failed, "symlink publication fault is reported");
    expect_true(std::filesystem::is_symlink(rollback_destination),
                "symlink publication fault preserves the destination link");
    expect_true(std::filesystem::read_symlink(rollback_destination) == victim,
                "symlink publication fault preserves the original target");
    expect_eq(read_text_file(victim), std::string("victim must not change\n"),
              "symlink publication fault does not modify the original target");
  }
  if (!temporary_symlink_error) {
    expect_true(std::filesystem::is_symlink(predictable_temporary),
                "predictable temporary symlink remains untouched");
  } else {
    expect_eq(read_text_file(predictable_temporary), std::string("predictable temporary guard\n"),
              "predictable temporary filename remains untouched");
  }

  std::filesystem::remove(predictable_temporary);
  std::vector<std::string> payloads;
  for (int writer = 0; writer < 8; ++writer) {
    payloads.push_back("{\"writer\":" + std::to_string(writer) + "}");
  }
  detail::write_calibration_json_atomically(payloads.front(), destination, left_input, right_input);

  std::atomic<bool> running{true};
  std::atomic<bool> observed_partial_output{false};
  std::atomic<bool> writer_failed{false};
  std::atomic<std::uint64_t> successful_reads{0};
  AtomicReadResult failed_read;
  std::thread reader([&] {
    while (running.load(std::memory_order_acquire)) {
      const auto result = read_atomic_output(destination);
      if (result.status == AtomicReadStatus::RetryableFailure) {
        std::this_thread::yield();
        continue;
      }
      if (result.status != AtomicReadStatus::Success) {
        failed_read = result;
        observed_partial_output.store(true, std::memory_order_release);
        return;
      }
      successful_reads.fetch_add(1, std::memory_order_relaxed);
      const auto complete = std::any_of(payloads.begin(), payloads.end(), [&](const auto& payload) {
        return result.contents == payload + '\n';
      });
      if (!complete) {
        failed_read = result;
        observed_partial_output.store(true, std::memory_order_release);
        return;
      }
      std::this_thread::yield();
    }
  });
  for (int attempt = 0; attempt < 100 && successful_reads.load(std::memory_order_relaxed) == 0 &&
                        !observed_partial_output.load(std::memory_order_acquire);
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  expect_true(successful_reads.load(std::memory_order_relaxed) > 0,
              "concurrent reader opens the initial complete calibration output");

  std::vector<std::thread> writers;
  for (std::size_t writer = 0; writer < payloads.size(); ++writer) {
    writers.emplace_back([&, writer] {
      try {
        for (int iteration = 0; iteration < 3; ++iteration) {
          detail::write_calibration_json_atomically(payloads[writer], destination, left_input,
                                                    right_input);
        }
      } catch (...) {
        writer_failed.store(true, std::memory_order_release);
      }
    });
  }
  for (auto& writer : writers) {
    writer.join();
  }
  running.store(false, std::memory_order_release);
  reader.join();

  if (observed_partial_output.load(std::memory_order_acquire)) {
    std::cerr << "atomic reader failure status=" << static_cast<int>(failed_read.status)
              << " bytes=" << failed_read.contents.size() << " contents=" << failed_read.contents
              << '\n';
  }

  expect_true(!writer_failed.load(std::memory_order_acquire),
              "concurrent calibration writers all succeed");
  expect_true(!observed_partial_output.load(std::memory_order_acquire),
              "concurrent reader never observes partial calibration output");
  expect_true(successful_reads.load(std::memory_order_relaxed) > 0,
              "concurrent reader observes at least one complete calibration output");
  const auto final_contents = read_text_file(destination);
  expect_true(std::any_of(payloads.begin(), payloads.end(),
                          [&](const auto& payload) { return final_contents == payload + '\n'; }),
              "concurrent calibration output is one complete writer payload");

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  const auto timeout_destination = root.path() / "lock-timeout.json";
  std::atomic<bool> timeout_holder_entered{false};
  std::atomic<bool> release_timeout_holder{false};
  std::exception_ptr timeout_holder_error;
  std::thread timeout_holder([&] {
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"timeout-holder"})json", timeout_destination, left_input, right_input,
          {}, {}, [&] {
            timeout_holder_entered.store(true, std::memory_order_release);
            while (!release_timeout_holder.load(std::memory_order_acquire)) {
              std::this_thread::yield();
            }
          });
    } catch (...) {
      timeout_holder_error = std::current_exception();
    }
  });
  for (int attempt = 0; attempt < 100 && !timeout_holder_entered.load(std::memory_order_acquire);
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  expect_true(timeout_holder_entered.load(std::memory_order_acquire),
              "lock timeout holder reaches the publication boundary");
  bool lock_contention_reported = false;
  bool lock_timeout_reported = false;
  if (timeout_holder_entered.load(std::memory_order_acquire)) {
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"timeout-waiter"})json", timeout_destination, left_input, right_input,
          {}, {}, {}, false, {}, [&] { lock_contention_reported = true; },
          std::chrono::milliseconds(50));
    } catch (const std::exception& error) {
      lock_timeout_reported =
          std::string_view(error.what()).find("timed out locking") != std::string_view::npos;
    }
  }
  expect_true(lock_contention_reported, "lock timeout waiter reports contention");
  expect_true(lock_timeout_reported, "lock timeout waiter fails closed at its deadline");
  bool slow_callback_timeout_reported = false;
  if (timeout_holder_entered.load(std::memory_order_acquire)) {
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"slow-timeout-waiter"})json", timeout_destination, left_input,
          right_input, {}, {}, {}, false, {},
          [&] {
            release_timeout_holder.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(75));
          },
          std::chrono::milliseconds(25));
    } catch (const std::exception& error) {
      slow_callback_timeout_reported =
          std::string_view(error.what()).find("timed out locking") != std::string_view::npos;
    }
  }
  expect_true(slow_callback_timeout_reported,
              "lock waiter cannot acquire after a slow callback exhausts its deadline");
  release_timeout_holder.store(true, std::memory_order_release);
  timeout_holder.join();
  expect_true(timeout_holder_error == nullptr, "lock timeout holder completes after release");
#endif

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  std::vector<bool> throwing_hook_fallbacks{false};
#if defined(__linux__)
  throwing_hook_fallbacks.push_back(true);
#endif
  for (const bool force_fallback : throwing_hook_fallbacks) {
    const auto throwing_hook_destination =
        root.path() /
        (force_fallback ? "throwing-fallback-hook.json" : "throwing-exchange-hook.json");
    write_text_file(throwing_hook_destination, "throwing hook original output\n");
    bool throwing_hook_propagated = false;
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"throwing-hook"})json", throwing_hook_destination, left_input,
          right_input, {}, {}, {}, force_fallback,
          [] { throw std::runtime_error("synthetic after-publication hook failure"); });
    } catch (const std::exception& error) {
      throwing_hook_propagated =
          std::string_view(error.what()).find("synthetic after-publication hook failure") !=
          std::string_view::npos;
    }
    expect_true(throwing_hook_propagated, force_fallback
                                              ? "throwing fallback publication hook propagates"
                                              : "throwing publication hook propagates");
    expect_eq(read_text_file(throwing_hook_destination),
              std::string("throwing hook original output\n"),
              force_fallback ? "throwing fallback hook preserves the original output"
                             : "throwing hook preserves the original output");

    const auto missing_destination =
        root.path() / (force_fallback ? "throwing-fallback-new.json" : "throwing-primary-new.json");
    std::filesystem::remove(missing_destination);
    throwing_hook_propagated = false;
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"throwing-new-hook"})json", missing_destination, left_input, right_input,
          {}, {}, {}, force_fallback,
          [] { throw std::runtime_error("synthetic after-publication hook failure"); });
    } catch (const std::exception& error) {
      throwing_hook_propagated =
          std::string_view(error.what()).find("synthetic after-publication hook failure") !=
          std::string_view::npos;
    }
    expect_true(throwing_hook_propagated, "throwing new-output publication hook propagates");
    expect_true(!std::filesystem::exists(missing_destination),
                "throwing new-output hook leaves no output");
  }
#endif

#if defined(__linux__)
  const auto serialized_destination = root.path() / "serialized.json";
  write_text_file(serialized_destination, "serialized initial output\n");
  std::atomic<bool> first_commit_entered{false};
  std::atomic<bool> release_first_commit{false};
  std::atomic<bool> second_writer_entered{false};
  std::exception_ptr first_writer_error;
  std::exception_ptr second_writer_error;
  std::thread first_writer([&] {
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"serialized-first"})json", serialized_destination, left_input,
          right_input, {}, {}, [&] {
            first_commit_entered.store(true, std::memory_order_release);
            while (!release_first_commit.load(std::memory_order_acquire)) {
              std::this_thread::yield();
            }
          });
    } catch (...) {
      first_writer_error = std::current_exception();
    }
  });
  for (int attempt = 0; attempt < 100 && !first_commit_entered.load(std::memory_order_acquire);
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  expect_true(first_commit_entered.load(std::memory_order_acquire),
              "first writer reaches the locked commit boundary");
  std::thread second_writer([&] {
    try {
      detail::write_calibration_json_atomically(
          R"json({"writer":"serialized-second"})json", serialized_destination, left_input,
          right_input, [&] { second_writer_entered.store(true, std::memory_order_release); });
    } catch (...) {
      second_writer_error = std::current_exception();
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  expect_true(!second_writer_entered.load(std::memory_order_acquire),
              "second writer blocks before publication while the lock is held");
  release_first_commit.store(true, std::memory_order_release);
  first_writer.join();
  second_writer.join();
  expect_true(first_writer_error == nullptr && second_writer_error == nullptr,
              "serialized writers both complete successfully");
  expect_true(second_writer_entered.load(std::memory_order_acquire),
              "second writer proceeds after lock release");
  expect_eq(read_text_file(serialized_destination),
            std::string("{\"writer\":\"serialized-second\"}\n"),
            "lock serialization preserves commit order");
#endif

#if !defined(_WIN32)
  const auto process_destination = root.path() / "process-serialized.json";
  write_text_file(process_destination, "process serialization initial output\n");
  int first_ready[2]{};
  int release_first[2]{};
  int second_contended[2]{};
  int second_entered[2]{};
  const bool pipes_ready = ::pipe(first_ready) == 0 && ::pipe(release_first) == 0 &&
                           ::pipe(second_contended) == 0 && ::pipe(second_entered) == 0;
  expect_true(pipes_ready, "interprocess serialization pipes are available");
  if (pipes_ready) {
    const pid_t first_writer = ::fork();
    if (first_writer == 0) {
      (void)::close(first_ready[0]);
      (void)::close(release_first[1]);
      (void)::close(second_contended[0]);
      (void)::close(second_contended[1]);
      (void)::close(second_entered[0]);
      (void)::close(second_entered[1]);
      try {
        detail::write_calibration_json_atomically(
            R"json({"writer":"process-first"})json", process_destination, left_input, right_input,
            {}, {}, [&] {
              const char ready = 'r';
              if (::write(first_ready[1], &ready, 1) != 1) {
                throw std::runtime_error("cannot signal first writer readiness");
              }
              char release = 0;
              if (::read(release_first[0], &release, 1) != 1) {
                throw std::runtime_error("cannot release first writer");
              }
            });
        ::_exit(0);
      } catch (...) {
        ::_exit(2);
      }
    }
    expect_true(first_writer > 0, "first interprocess writer starts");
    (void)::close(first_ready[1]);
    char ready = 0;
    expect_true(first_writer > 0 &&
                    read_byte_with_timeout(first_ready[0], ready, std::chrono::seconds(5)),
                "first interprocess writer reaches the locked boundary");

    const pid_t second_writer = ::fork();
    if (second_writer == 0) {
      (void)::close(first_ready[0]);
      (void)::close(first_ready[1]);
      (void)::close(release_first[0]);
      (void)::close(release_first[1]);
      (void)::close(second_contended[0]);
      (void)::close(second_entered[0]);
      try {
        detail::write_calibration_json_atomically(
            R"json({"writer":"process-second"})json", process_destination, left_input, right_input,
            [&] {
              const char entered = 'e';
              if (::write(second_entered[1], &entered, 1) != 1) {
                throw std::runtime_error("cannot signal second writer entry");
              }
            },
            {}, {}, false, {},
            [&] {
              const char contended = 'c';
              if (::write(second_contended[1], &contended, 1) != 1) {
                throw std::runtime_error("cannot signal directory-lock contention");
              }
            });
        ::_exit(0);
      } catch (...) {
        ::_exit(3);
      }
    }
    expect_true(second_writer > 0, "second interprocess writer starts");
    (void)::close(second_contended[1]);
    (void)::close(second_entered[1]);
    char contended = 0;
    expect_true(second_writer > 0 &&
                    read_byte_with_timeout(second_contended[0], contended, std::chrono::seconds(5)),
                "second process observes directory-lock contention");
    const char release = 'x';
    expect_true(::write(release_first[1], &release, 1) == 1,
                "first interprocess writer is released");
    expect_true(wait_child_success_with_timeout(first_writer, std::chrono::seconds(5)),
                "first interprocess writer succeeds");
    char entered = 0;
    expect_true(read_byte_with_timeout(second_entered[0], entered, std::chrono::seconds(5)),
                "second process enters after directory-lock release");
    expect_true(wait_child_success_with_timeout(second_writer, std::chrono::seconds(5)),
                "second interprocess writer succeeds");
    expect_eq(read_text_file(process_destination), std::string("{\"writer\":\"process-second\"}\n"),
              "interprocess lock preserves writer commit order");
    (void)::close(first_ready[0]);
    (void)::close(release_first[0]);
    (void)::close(release_first[1]);
    (void)::close(second_contended[0]);
    (void)::close(second_entered[0]);
  }
#endif

  const auto blocked_destination = root.path() / "blocked.json";
  std::filesystem::create_directory(blocked_destination);
  write_text_file(blocked_destination / "keep", "keep");
  bool replacement_failed = false;
  try {
    detail::write_calibration_json_atomically(R"json({"writer":"failure"})json",
                                              blocked_destination, left_input, right_input);
  } catch (const std::exception&) {
    replacement_failed = true;
  }
  expect_true(replacement_failed, "failed calibration replacement reports an error");
  expect_true(std::filesystem::is_directory(blocked_destination),
              "failed calibration replacement preserves the destination directory");
  expect_eq(read_text_file(blocked_destination / "keep"), std::string("keep"),
            "failed calibration replacement preserves destination contents");

  const auto raced_destination = root.path() / "raced-match.json";
  const auto initial_identity_error = reco::calibrate::validate_calibration_output_identity(
      left_input, right_input, raced_destination);
  expect_true(!initial_identity_error.has_value(), "absent output passes initial identity check");
  std::filesystem::create_hard_link(left_input, raced_destination);
  bool raced_alias_rejected = false;
  try {
    detail::write_calibration_json_atomically(R"json({"writer":"raced"})json", raced_destination,
                                              left_input, right_input);
  } catch (const std::exception& error) {
    raced_alias_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(raced_alias_rejected, "publication recheck rejects a newly introduced input alias");
  expect_eq(read_text_file(left_input), std::string("left video must not change\n"),
            "publication recheck preserves aliased input contents");

#if defined(__linux__)
  const auto legacy_lock_name = root.path() / ".reco-calibration-output.lock";
  detail::write_calibration_json_atomically(R"json({"writer":"directory-lock"})json",
                                            legacy_lock_name, left_input, right_input);
  expect_eq(read_text_file(legacy_lock_name), std::string("{\"writer\":\"directory-lock\"}\n"),
            "directory-inode locking does not reserve a replaceable lock-file name");

  const auto mutable_input = root.path() / "mutable-left.mp4";
  const auto mutable_output = root.path() / "mutable-match.json";
  write_text_file(mutable_input, "calibrated media identity\n");
  bool mutable_input_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"mutable-input"})json", mutable_output, mutable_input, right_input,
        [&] { write_text_file(mutable_input, "mutated media identity after calibration\n"); });
  } catch (const std::exception& error) {
    mutable_input_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(mutable_input_rejected,
              "publication rejects in-place media mutation after calibration");
  expect_true(!std::filesystem::exists(mutable_output), "rejected media mutation is not published");

  const auto mutable_profile = root.path() / "mutable-profile.json";
  const auto mutable_profile_output = root.path() / "mutable-profile-match.json";
  write_text_file(mutable_profile, "calibrated profile identity\n");
  const std::array<std::filesystem::path, 1> mutable_profiles{mutable_profile};
  bool mutable_profile_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"mutable-profile"})json", mutable_profile_output, left_input, right_input,
        [&] { write_text_file(mutable_profile, "mutated profile identity after calibration\n"); },
        mutable_profiles);
  } catch (const std::exception& error) {
    mutable_profile_rejected =
        std::string_view(error.what()).find("left lens profile") != std::string_view::npos;
  }
  expect_true(mutable_profile_rejected,
              "publication rejects in-place profile mutation after calibration");
  expect_true(!std::filesystem::exists(mutable_profile_output),
              "rejected profile mutation is not published");

  const auto commit_mutable_input = root.path() / "commit-mutable-left.mp4";
  const auto commit_mutable_output = root.path() / "commit-mutable-match.json";
  write_text_file(commit_mutable_input, "commit media identity\n");
  bool commit_mutable_input_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"commit-mutable-input"})json", commit_mutable_output, commit_mutable_input,
        right_input, {}, {},
        [&] { write_text_file(commit_mutable_input, "changed media identity\n"); });
  } catch (const std::exception& error) {
    commit_mutable_input_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(commit_mutable_input_rejected,
              "commit-boundary revalidation rejects an in-place media mutation");
  expect_true(!std::filesystem::exists(commit_mutable_output),
              "commit-boundary media mutation is not published");

  const auto commit_mutable_profile = root.path() / "commit-mutable-profile.json";
  const auto commit_mutable_profile_output = root.path() / "commit-mutable-profile-match.json";
  write_text_file(commit_mutable_profile, "commit profile identity\n");
  const std::array<std::filesystem::path, 1> commit_mutable_profiles{commit_mutable_profile};
  bool commit_mutable_profile_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"commit-mutable-profile"})json", commit_mutable_profile_output, left_input,
        right_input, {}, commit_mutable_profiles,
        [&] { write_text_file(commit_mutable_profile, "changed profile identity\n"); });
  } catch (const std::exception& error) {
    commit_mutable_profile_rejected =
        std::string_view(error.what()).find("left lens profile") != std::string_view::npos;
  }
  expect_true(commit_mutable_profile_rejected,
              "commit-boundary revalidation rejects an in-place lens-profile mutation");
  expect_true(!std::filesystem::exists(commit_mutable_profile_output),
              "commit-boundary profile mutation is not published");

  const auto symlink_alias = root.path() / "symlink-alias.json";
  std::filesystem::create_symlink(left_input, symlink_alias);
  bool symlink_alias_rejected = false;
  try {
    detail::write_calibration_json_atomically(R"json({"writer":"symlink-alias"})json",
                                              symlink_alias, left_input, right_input);
  } catch (const std::exception& error) {
    symlink_alias_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(symlink_alias_rejected, "publication rejects an existing symlink to a video input");
  expect_true(std::filesystem::is_symlink(symlink_alias),
              "rejected input symlink remains in place");

  const auto original_input = root.path() / "original-left.mp4";
  const auto replacement_input = root.path() / "replacement-left.mp4";
  const auto input_symlink = root.path() / "retargeted-left.mp4";
  const auto pinned_alias = root.path() / "pinned-alias.json";
  write_text_file(original_input, "original pinned input\n");
  write_text_file(replacement_input, "replacement input\n");
  std::filesystem::create_symlink(original_input, input_symlink);
  std::filesystem::create_hard_link(original_input, pinned_alias);
  bool retarget_hook_ran = false;
  bool retargeted_input_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"retarget-input"})json", pinned_alias, input_symlink, right_input, [&] {
          std::filesystem::remove(input_symlink);
          std::filesystem::create_symlink(replacement_input, input_symlink);
          retarget_hook_ran = true;
        });
  } catch (const std::exception& error) {
    retargeted_input_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(retarget_hook_ran, "input retarget happens at the publication boundary");
  expect_true(retargeted_input_rejected,
              "pinned input identity rejects a symlink retarget publication race");
  expect_true(std::filesystem::equivalent(original_input, pinned_alias),
              "rejected publication preserves the pinned input hardlink");

  const auto calibrated_input = root.path() / "calibrated-left.mp4";
  const auto post_calibration_input = root.path() / "post-calibration-left.mp4";
  const auto user_input_path = root.path() / "user-left.mp4";
  const auto calibrated_alias = root.path() / "calibrated-alias.json";
  write_text_file(calibrated_input, "calibrated input identity\n");
  write_text_file(post_calibration_input, "post-calibration input identity\n");
  std::filesystem::create_symlink(calibrated_input, user_input_path);
  const int calibrated_descriptor = ::open(user_input_path.c_str(), O_RDONLY | O_CLOEXEC);
  expect_true(calibrated_descriptor >= 0, "calibrated input descriptor is retained");
  std::filesystem::remove(user_input_path);
  std::filesystem::create_symlink(post_calibration_input, user_input_path);
  std::filesystem::create_hard_link(calibrated_input, calibrated_alias);
  bool calibrated_alias_rejected = false;
  if (calibrated_descriptor >= 0) {
    const auto retained_input = std::filesystem::path("/proc") / std::to_string(::getpid()) / "fd" /
                                std::to_string(calibrated_descriptor);
    try {
      detail::write_calibration_json_atomically(R"json({"writer":"post-calibration-retarget"})json",
                                                calibrated_alias, retained_input, right_input);
    } catch (const std::exception& error) {
      calibrated_alias_rejected =
          std::string_view(error.what()).find("left video input") != std::string_view::npos;
    }
    (void)::close(calibrated_descriptor);
  }
  expect_true(calibrated_alias_rejected,
              "retained calibrated identity survives a pre-publication input retarget");
  expect_true(std::filesystem::equivalent(calibrated_input, calibrated_alias),
              "pre-publication retarget cannot hide the calibrated input alias");

  const auto selected_profile = root.path() / "selected-lens.json";
  const auto moved_profile = root.path() / "selected-lens-original.json";
  const auto profile_alias = root.path() / "profile-alias.json";
  write_text_file(selected_profile, "selected lens profile\n");
  const std::array<std::filesystem::path, 1> lens_profiles{selected_profile};
  bool profile_swap_hook_ran = false;
  bool profile_alias_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"profile-retarget"})json", profile_alias, left_input, right_input,
        [&] {
          std::filesystem::rename(selected_profile, moved_profile);
          write_text_file(selected_profile, "replacement lens profile\n");
          std::filesystem::create_hard_link(moved_profile, profile_alias);
          profile_swap_hook_ran = true;
        },
        lens_profiles);
  } catch (const std::exception& error) {
    profile_alias_rejected =
        std::string_view(error.what()).find("left lens profile") != std::string_view::npos;
  }
  expect_true(profile_swap_hook_ran, "lens-profile retarget happens at publication boundary");
  expect_true(profile_alias_rejected,
              "pinned lens-profile identity rejects a publication alias after path replacement");
  expect_true(std::filesystem::equivalent(moved_profile, profile_alias),
              "rejected publication preserves the selected lens-profile identity");
  expect_eq(read_text_file(selected_profile), std::string("replacement lens profile\n"),
            "profile replacement cannot hide the selected identity from publication checks");

  const auto temporary_race_destination = root.path() / "temporary-race.json";
  std::filesystem::path replacement_temporary;
  bool temporary_identity_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"temporary-race"})json", temporary_race_destination, left_input,
        right_input, [&] {
          for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
            if (entry.path().filename().string().starts_with("temporary-race.json.tmp.")) {
              replacement_temporary = entry.path();
              std::filesystem::remove(replacement_temporary);
              write_text_file(replacement_temporary, "attacker replacement\n");
              return;
            }
          }
          throw std::runtime_error("temporary publication fixture was not found");
        });
  } catch (const std::exception& error) {
    temporary_identity_rejected =
        std::string_view(error.what()).find("temporary output identity changed") !=
        std::string_view::npos;
  }
  expect_true(temporary_identity_rejected,
              "publication rejects a replaced temporary output directory entry");
  expect_true(!std::filesystem::exists(temporary_race_destination),
              "replaced temporary output is not published");
  expect_eq(read_text_file(replacement_temporary), std::string("attacker replacement\n"),
            "cleanup does not unlink a replacement temporary entry");
  std::filesystem::remove(replacement_temporary);

  const auto postcheck_race_destination = root.path() / "postcheck-race.json";
  std::filesystem::path postcheck_replacement;
  bool postcheck_hook_ran = false;
  detail::write_calibration_json_atomically(
      R"json({"writer":"descriptor-bound"})json", postcheck_race_destination, left_input,
      right_input, {}, {}, [&] {
        for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
          if (entry.path().filename().string().starts_with("postcheck-race.json.tmp.")) {
            postcheck_replacement = entry.path();
            std::filesystem::remove(postcheck_replacement);
            write_text_file(postcheck_replacement, "post-check attacker replacement\n");
            postcheck_hook_ran = true;
            return;
          }
        }
        throw std::runtime_error("post-check publication fixture was not found");
      });
  expect_true(postcheck_hook_ran, "temporary replacement runs after identity validation");
  expect_eq(read_text_file(postcheck_race_destination),
            std::string("{\"writer\":\"descriptor-bound\"}\n"),
            "publication remains bound to the opened temporary file");
  expect_eq(read_text_file(postcheck_replacement), std::string("post-check attacker replacement\n"),
            "descriptor-bound publication does not rename or unlink the replacement path");
  std::filesystem::remove(postcheck_replacement);

  const auto publication_race_destination = root.path() / "publication-race.json";
  write_text_file(publication_race_destination, "original publication target\n");
  std::filesystem::path publication_replacement;
  bool publication_identity_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"publication-race"})json", publication_race_destination, left_input,
        right_input, {}, {}, [&] {
          for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
            if (entry.path().filename().string().starts_with("publication-race.json.publish.")) {
              publication_replacement = entry.path();
              std::filesystem::remove(publication_replacement);
              write_text_file(publication_replacement, "publication attacker replacement\n");
              return;
            }
          }
          throw std::runtime_error("descriptor publication fixture was not found");
        });
  } catch (const std::exception& error) {
    publication_identity_rejected =
        std::string_view(error.what()).find("descriptor-bound output identity changed") !=
        std::string_view::npos;
  }
  expect_true(publication_identity_rejected,
              "atomic exchange rejects a replaced descriptor publication entry");
  expect_eq(read_text_file(publication_race_destination),
            std::string("original publication target\n"),
            "failed descriptor publication exchange restores the original output");
  expect_eq(read_text_file(publication_replacement),
            std::string("publication attacker replacement\n"),
            "failed descriptor publication does not unlink the replacement entry");
  std::filesystem::remove(publication_replacement);

  const auto post_exchange_destination = root.path() / "post-exchange-race.json";
  write_text_file(post_exchange_destination, "post-exchange original output\n");
  std::filesystem::path post_exchange_replacement;
  bool post_exchange_rejected = false;
  bool post_exchange_hook_ran = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"post-exchange-race"})json", post_exchange_destination, left_input,
        right_input, {}, {}, {}, false, [&] {
          post_exchange_hook_ran = true;
          for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
            if (entry.path() != post_exchange_destination &&
                std::filesystem::is_regular_file(entry.path()) &&
                read_text_file(entry.path()) == "post-exchange original output\n") {
              post_exchange_replacement = entry.path();
              std::filesystem::remove(post_exchange_replacement);
              write_text_file(post_exchange_replacement, "post-exchange attacker replacement\n");
              return;
            }
          }
          throw std::runtime_error("post-exchange publication fixture was not found");
        });
  } catch (const std::exception& error) {
    post_exchange_rejected =
        std::string_view(error.what()).find("rollback refused") != std::string_view::npos;
  }
  expect_true(post_exchange_hook_ran, "post-exchange displaced-entry race hook runs");
  expect_true(post_exchange_rejected,
              "post-exchange displaced-entry substitution refuses unsafe cleanup");
  expect_eq(read_text_file(post_exchange_replacement),
            std::string("post-exchange attacker replacement\n"),
            "post-exchange cleanup preserves a substituted displaced entry");
  expect_eq(read_text_file(post_exchange_destination),
            std::string("{\"writer\":\"post-exchange-race\"}\n"),
            "post-exchange cleanup failure does not move an unverified entry over the output");
  std::filesystem::remove(post_exchange_replacement);

  const auto fallback_post_exchange_destination = root.path() / "fallback-post-exchange-race.json";
  write_text_file(fallback_post_exchange_destination, "fallback post-exchange original output\n");
  std::filesystem::path fallback_displaced_output;
  bool fallback_post_exchange_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"fallback-post-exchange-race"})json", fallback_post_exchange_destination,
        left_input, right_input, {}, {}, {}, true, [&] {
          for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
            if (entry.path().filename().string().starts_with(
                    "fallback-post-exchange-race.json.tmp.")) {
              fallback_displaced_output = entry.path();
              break;
            }
          }
          if (fallback_displaced_output.empty()) {
            throw std::runtime_error("fallback post-exchange fixture was not found");
          }
          std::filesystem::remove(fallback_post_exchange_destination);
          std::filesystem::create_hard_link(left_input, fallback_post_exchange_destination);
        });
  } catch (const std::exception& error) {
    fallback_post_exchange_rejected =
        std::string_view(error.what()).find("rollback refused") != std::string_view::npos;
  }
  expect_true(fallback_post_exchange_rejected,
              "fallback post-exchange destination substitution refuses unsafe rollback");
  expect_true(std::filesystem::equivalent(left_input, fallback_post_exchange_destination),
              "unsafe fallback rollback does not move an input alias");
  expect_eq(read_text_file(fallback_displaced_output),
            std::string("fallback post-exchange original output\n"),
            "unsafe fallback rollback preserves the displaced output");
  std::filesystem::remove(fallback_post_exchange_destination);
  std::filesystem::remove(fallback_displaced_output);

  const auto fallback_destination = root.path() / "fallback.json";
  detail::write_calibration_json_atomically(R"json({"writer":"fallback-new"})json",
                                            fallback_destination, left_input, right_input, {}, {},
                                            {}, true);
  expect_eq(read_text_file(fallback_destination), std::string("{\"writer\":\"fallback-new\"}\n"),
            "hard-link-disabled fallback publishes a new output");
  detail::write_calibration_json_atomically(R"json({"writer":"fallback-replace"})json",
                                            fallback_destination, left_input, right_input, {}, {},
                                            {}, true);
  expect_eq(read_text_file(fallback_destination),
            std::string("{\"writer\":\"fallback-replace\"}\n"),
            "hard-link-disabled fallback atomically replaces an output");

  std::filesystem::path fallback_replacement;
  bool fallback_identity_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"fallback-race"})json", fallback_destination, left_input, right_input, {},
        {},
        [&] {
          for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
            if (entry.path().filename().string().starts_with("fallback.json.tmp.")) {
              fallback_replacement = entry.path();
              std::filesystem::remove(fallback_replacement);
              write_text_file(fallback_replacement, "fallback attacker replacement\n");
              return;
            }
          }
          throw std::runtime_error("fallback publication fixture was not found");
        },
        true);
  } catch (const std::exception& error) {
    fallback_identity_rejected =
        std::string_view(error.what()).find("fallback output identity changed") !=
        std::string_view::npos;
  }
  expect_true(fallback_identity_rejected,
              "hard-link-disabled fallback reports substituted content as failure");
  expect_eq(read_text_file(fallback_destination),
            std::string("{\"writer\":\"fallback-replace\"}\n"),
            "failed hard-link-disabled exchange restores the original output");
  expect_eq(read_text_file(fallback_replacement), std::string("fallback attacker replacement\n"),
            "failed fallback exchange does not unlink the replacement entry");
  std::filesystem::remove(fallback_replacement);

  const auto fallback_absent_destination = root.path() / "fallback-absent.json";
  std::filesystem::path fallback_absent_replacement;
  bool fallback_absent_identity_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"fallback-absent-race"})json", fallback_absent_destination, left_input,
        right_input, {}, {},
        [&] {
          for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
            if (entry.path().filename().string().starts_with("fallback-absent.json.tmp.")) {
              fallback_absent_replacement = entry.path();
              std::filesystem::remove(fallback_absent_replacement);
              write_text_file(fallback_absent_replacement,
                              "fallback absent attacker replacement\n");
              return;
            }
          }
          throw std::runtime_error("absent fallback publication fixture was not found");
        },
        true);
  } catch (const std::exception& error) {
    fallback_absent_identity_rejected =
        std::string_view(error.what()).find("fallback output identity changed") !=
        std::string_view::npos;
  }
  expect_true(fallback_absent_identity_rejected,
              "hard-link-disabled fallback rejects a substituted new output");
  expect_true(!std::filesystem::exists(fallback_absent_destination),
              "failed hard-link-disabled new-output publication restores absence");
  expect_eq(read_text_file(fallback_absent_replacement),
            std::string("fallback absent attacker replacement\n"),
            "failed new-output fallback restores the substituted temporary entry");
  std::filesystem::remove(fallback_absent_replacement);

  const auto moved_input = root.path() / "commit-moved-left.mp4";
  const auto moved_input_destination = root.path() / "commit-moved-input-output.json";
  write_text_file(moved_input, "commit-bound media must survive\n");
  bool moved_input_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"moved-input"})json", moved_input_destination, moved_input, right_input,
        {}, {}, [&] { std::filesystem::rename(moved_input, moved_input_destination); });
  } catch (const std::exception& error) {
    moved_input_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(moved_input_rejected,
              "commit-bound destination substitution with an input is rejected");
  expect_eq(read_text_file(moved_input_destination),
            std::string("commit-bound media must survive\n"),
            "rollback preserves media moved onto the destination");
  std::filesystem::rename(moved_input_destination, moved_input);

  const auto moved_commit_profile = root.path() / "commit-moved-profile.json";
  const auto moved_profile_destination = root.path() / "commit-moved-profile-output.json";
  write_text_file(moved_commit_profile, "commit-bound profile must survive\n");
  const std::array<std::filesystem::path, 1> moved_commit_profiles{moved_commit_profile};
  bool moved_commit_profile_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"moved-profile"})json", moved_profile_destination, left_input, right_input,
        {}, moved_commit_profiles,
        [&] { std::filesystem::rename(moved_commit_profile, moved_profile_destination); });
  } catch (const std::exception& error) {
    moved_commit_profile_rejected =
        std::string_view(error.what()).find("left lens profile") != std::string_view::npos;
  }
  expect_true(moved_commit_profile_rejected,
              "commit-bound destination substitution with a lens profile is rejected");
  expect_eq(read_text_file(moved_profile_destination),
            std::string("commit-bound profile must survive\n"),
            "rollback preserves a lens profile moved onto the destination");
  std::filesystem::rename(moved_profile_destination, moved_commit_profile);

  const auto moved_symlink_target = root.path() / "commit-moved-symlink-target.mp4";
  const auto moved_symlink_input = root.path() / "commit-moved-symlink-left.mp4";
  const auto moved_symlink_destination = root.path() / "commit-moved-symlink-output.json";
  write_text_file(moved_symlink_target, "commit-bound symlink media must survive\n");
  std::filesystem::create_symlink(moved_symlink_target, moved_symlink_input);
  bool moved_symlink_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"moved-symlink-input"})json", moved_symlink_destination,
        moved_symlink_input, right_input, {}, {},
        [&] { std::filesystem::rename(moved_symlink_input, moved_symlink_destination); });
  } catch (const std::exception& error) {
    moved_symlink_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(moved_symlink_rejected, "descriptor-link publication rejects a moved input symlink");
  expect_true(std::filesystem::is_symlink(moved_symlink_destination),
              "descriptor-link rollback preserves the moved input symlink");
  expect_eq(read_text_file(moved_symlink_destination),
            std::string("commit-bound symlink media must survive\n"),
            "descriptor-link rollback preserves the moved symlink target");
  std::filesystem::rename(moved_symlink_destination, moved_symlink_input);

  const auto moved_profile_target = root.path() / "commit-moved-symlink-profile-target.json";
  const auto moved_symlink_profile = root.path() / "commit-moved-symlink-profile.json";
  const auto moved_symlink_profile_destination =
      root.path() / "commit-moved-symlink-profile-output.json";
  write_text_file(moved_profile_target, "commit-bound symlink profile must survive\n");
  std::filesystem::create_symlink(moved_profile_target, moved_symlink_profile);
  const std::array<std::filesystem::path, 1> moved_symlink_profiles{moved_symlink_profile};
  bool moved_symlink_profile_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"moved-symlink-profile"})json", moved_symlink_profile_destination,
        left_input, right_input, {}, moved_symlink_profiles,
        [&] { std::filesystem::rename(moved_symlink_profile, moved_symlink_profile_destination); },
        true);
  } catch (const std::exception& error) {
    moved_symlink_profile_rejected =
        std::string_view(error.what()).find("left lens profile") != std::string_view::npos;
  }
  expect_true(moved_symlink_profile_rejected,
              "forced-fallback publication rejects a moved profile symlink");
  expect_true(std::filesystem::is_symlink(moved_symlink_profile_destination),
              "forced-fallback rollback preserves the moved profile symlink");
  expect_eq(read_text_file(moved_symlink_profile_destination),
            std::string("commit-bound symlink profile must survive\n"),
            "forced-fallback rollback preserves the moved profile target");
  std::filesystem::rename(moved_symlink_profile_destination, moved_symlink_profile);

  const auto original_parent = root.path() / "original-parent";
  const auto redirected_parent = root.path() / "redirected-parent";
  const auto parent_symlink = root.path() / "output-parent";
  std::filesystem::create_directory(original_parent);
  std::filesystem::create_directory(redirected_parent);
  std::filesystem::create_symlink(original_parent, parent_symlink);
  const auto redirected_destination = parent_symlink / "match.json";
  bool parent_retarget_hook_ran = false;
  detail::write_calibration_json_atomically(R"json({"writer":"pinned-parent"})json",
                                            redirected_destination, left_input, right_input, [&] {
                                              std::filesystem::remove(parent_symlink);
                                              std::filesystem::create_symlink(redirected_parent,
                                                                              parent_symlink);
                                              parent_retarget_hook_ran = true;
                                            });
  expect_true(parent_retarget_hook_ran,
              "output parent retarget happens at the publication boundary");
  expect_eq(read_text_file(original_parent / "match.json"),
            std::string("{\"writer\":\"pinned-parent\"}\n"),
            "publication stays in the pinned output directory");
  expect_true(!std::filesystem::exists(redirected_parent / "match.json"),
              "retargeted output parent cannot redirect publication");
#else
#if defined(__APPLE__)
  const auto timestamp_mutable_input = root.path() / "timestamp-mutable-left.mp4";
  const auto timestamp_mutable_output = root.path() / "timestamp-mutable-match.json";
  write_text_file(timestamp_mutable_input, "aaaaaaaa\n");
  bool timestamp_mutation_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"timestamp-mutation"})json", timestamp_mutable_output,
        timestamp_mutable_input, right_input,
        [&] { write_text_file(timestamp_mutable_input, "bbbbbbbb\n"); });
  } catch (const std::exception& error) {
    timestamp_mutation_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(timestamp_mutation_rejected,
              "Darwin publication rejects same-size in-place media mutation");
  expect_true(!std::filesystem::exists(timestamp_mutable_output),
              "Darwin timestamp mutation is not published");

  const auto timestamp_mutable_profile = root.path() / "timestamp-mutable-profile.json";
  const auto timestamp_mutable_profile_output =
      root.path() / "timestamp-mutable-profile-match.json";
  write_text_file(timestamp_mutable_profile, "cccccccc\n");
  const std::array<std::filesystem::path, 1> timestamp_profiles{timestamp_mutable_profile};
  bool timestamp_profile_mutation_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"timestamp-profile-mutation"})json", timestamp_mutable_profile_output,
        left_input, right_input, [&] { write_text_file(timestamp_mutable_profile, "dddddddd\n"); },
        timestamp_profiles);
  } catch (const std::exception& error) {
    timestamp_profile_mutation_rejected =
        std::string_view(error.what()).find("left lens profile") != std::string_view::npos;
  }
  expect_true(timestamp_profile_mutation_rejected,
              "Darwin publication rejects same-size in-place lens-profile mutation");
  expect_true(!std::filesystem::exists(timestamp_mutable_profile_output),
              "Darwin profile timestamp mutation is not published");

  const auto darwin_original_parent = root.path() / "darwin-original-parent";
  const auto darwin_redirected_parent = root.path() / "darwin-redirected-parent";
  const auto darwin_parent_symlink = root.path() / "darwin-output-parent";
  std::filesystem::create_directory(darwin_original_parent);
  std::filesystem::create_directory(darwin_redirected_parent);
  std::filesystem::create_symlink(darwin_original_parent, darwin_parent_symlink);
  const auto darwin_redirected_destination = darwin_parent_symlink / "match.json";
  detail::write_calibration_json_atomically(
      R"json({"writer":"darwin-pinned-parent"})json", darwin_redirected_destination, left_input,
      right_input, [&] {
        std::filesystem::remove(darwin_parent_symlink);
        std::filesystem::create_symlink(darwin_redirected_parent, darwin_parent_symlink);
      });
  expect_eq(read_text_file(darwin_original_parent / "match.json"),
            std::string("{\"writer\":\"darwin-pinned-parent\"}\n"),
            "Darwin publication stays in the pinned output directory");
  expect_true(!std::filesystem::exists(darwin_redirected_parent / "match.json"),
              "Darwin retargeted parent cannot redirect publication");

  const auto darwin_cleanup_destination = root.path() / "darwin-cleanup-race.json";
  write_text_file(darwin_cleanup_destination, "Darwin cleanup original output\n");
  std::filesystem::path darwin_cleanup_replacement;
  bool darwin_cleanup_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"darwin-cleanup-race"})json", darwin_cleanup_destination, left_input,
        right_input, {}, {}, {}, false, [&] {
          for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
            if (entry.path().filename().string().starts_with("darwin-cleanup-race.json.tmp.")) {
              darwin_cleanup_replacement = entry.path();
              std::filesystem::remove(darwin_cleanup_replacement);
              write_text_file(darwin_cleanup_replacement, "Darwin cleanup attacker replacement\n");
              return;
            }
          }
          throw std::runtime_error("Darwin cleanup race fixture was not found");
        });
  } catch (const std::exception& error) {
    darwin_cleanup_rejected =
        std::string_view(error.what()).find("rollback refused") != std::string_view::npos;
  }
  expect_true(darwin_cleanup_rejected,
              "Darwin displaced-entry substitution refuses cleanup and rollback");
  expect_eq(read_text_file(darwin_cleanup_replacement),
            std::string("Darwin cleanup attacker replacement\n"),
            "Darwin cleanup preserves a substituted displaced entry");
  expect_eq(read_text_file(darwin_cleanup_destination),
            std::string("{\"writer\":\"darwin-cleanup-race\"}\n"),
            "Darwin unsafe cleanup does not move an unverified entry over the output");
  std::filesystem::remove(darwin_cleanup_replacement);
#endif

  const auto moved_input = root.path() / "commit-moved-left.mp4";
  const auto moved_input_destination = root.path() / "commit-moved-input-output.json";
  write_text_file(moved_input, "commit-bound media must survive\n");
  bool move_succeeded = false;
  bool moved_input_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"moved-input"})json", moved_input_destination, moved_input, right_input,
        {}, {}, [&] {
          std::error_code move_error;
          std::filesystem::rename(moved_input, moved_input_destination, move_error);
          move_succeeded = !move_error;
          if (move_error) {
            throw std::filesystem::filesystem_error("input move blocked by retained handle",
                                                    moved_input, moved_input_destination,
                                                    move_error);
          }
        });
  } catch (const std::exception&) {
    moved_input_rejected = true;
  }
  expect_true(moved_input_rejected,
              "commit-bound destination substitution with an input is rejected");
  const auto& surviving_input = move_succeeded ? moved_input_destination : moved_input;
  expect_eq(read_text_file(surviving_input), std::string("commit-bound media must survive\n"),
            "cross-platform publication preserves a moved or retained input");

#if defined(_WIN32)
  const auto unicode_process_parent =
      root.path() / std::filesystem::path(L"windows-publication-\u03a9-\u4f8b");
  std::filesystem::create_directory(unicode_process_parent);
  const auto process_destination = unicode_process_parent / "windows-process-serialized.json";
  write_text_file(process_destination, "windows process serialization initial output\n");
  const auto event_prefix = L"Local\\RecoCliPublicationTest-" +
                            std::to_wstring(GetCurrentProcessId()) + L"-" +
                            std::to_wstring(std::random_device{}());
  const auto first_ready_name = event_prefix + L"-first-ready";
  const auto first_release_name = event_prefix + L"-first-release";
  const auto second_ready_name = event_prefix + L"-second-ready";
  const auto second_release_name = event_prefix + L"-second-release";
  const auto second_contention_name = event_prefix + L"-second-contention";
  const HANDLE first_ready_event = CreateEventW(nullptr, TRUE, FALSE, first_ready_name.c_str());
  const HANDLE first_release_event = CreateEventW(nullptr, TRUE, FALSE, first_release_name.c_str());
  const HANDLE second_ready_event = CreateEventW(nullptr, TRUE, FALSE, second_ready_name.c_str());
  const HANDLE second_release_event =
      CreateEventW(nullptr, TRUE, FALSE, second_release_name.c_str());
  const HANDLE second_contention_event =
      CreateEventW(nullptr, TRUE, FALSE, second_contention_name.c_str());
  const bool events_ready = first_ready_event != nullptr && first_release_event != nullptr &&
                            second_ready_event != nullptr && second_release_event != nullptr &&
                            second_contention_event != nullptr;
  expect_true(events_ready, "Windows interprocess publication events are available");
  if (events_ready) {
    const HANDLE first_writer = start_windows_publication_writer(
        process_destination, left_input, right_input, first_ready_name, first_release_name, L"-",
        LR"json({"writer":"windows-process-first"})json");
    expect_true(first_writer != nullptr, "first Windows interprocess writer starts");
    expect_true(first_writer != nullptr &&
                    WaitForSingleObject(first_ready_event, 10000) == WAIT_OBJECT_0,
                "first Windows writer reaches the locked commit boundary");
    const HANDLE second_writer = start_windows_publication_writer(
        process_destination, left_input, right_input, second_ready_name, second_release_name,
        second_contention_name, LR"json({"writer":"windows-process-second"})json");
    expect_true(second_writer != nullptr, "second Windows interprocess writer starts");
    expect_true(second_writer != nullptr &&
                    WaitForSingleObject(second_contention_event, 10000) == WAIT_OBJECT_0,
                "second Windows process observes directory-lock contention");
    expect_true(SetEvent(first_release_event) != 0, "first Windows writer is released");
    expect_true(wait_windows_process_success(first_writer, 10000), "first Windows writer succeeds");
    expect_true(second_writer != nullptr &&
                    WaitForSingleObject(second_ready_event, 10000) == WAIT_OBJECT_0,
                "second Windows process enters after directory-lock release");
    expect_true(SetEvent(second_release_event) != 0, "second Windows writer is released");
    expect_true(wait_windows_process_success(second_writer, 10000),
                "second Windows writer succeeds");
    expect_eq(read_text_file(process_destination),
              std::string("{\"writer\":\"windows-process-second\"}\n"),
              "Windows interprocess lock preserves writer commit order");
    if (first_writer != nullptr) {
      (void)CloseHandle(first_writer);
    }
    if (second_writer != nullptr) {
      (void)CloseHandle(second_writer);
    }
  }
  for (const HANDLE event : {first_ready_event, first_release_event, second_ready_event,
                             second_release_event, second_contention_event}) {
    if (event != nullptr) {
      (void)CloseHandle(event);
    }
  }

  const auto active_parent = root.path() / "active-output-parent";
  const auto retained_parent = root.path() / "retained-output-parent";
  std::filesystem::create_directory(retained_parent);
  std::error_code parent_symlink_error;
  std::filesystem::create_directory_symlink(retained_parent, active_parent, parent_symlink_error);
  expect_true(!parent_symlink_error, "Windows output parent symlink fixture is available");
  if (!parent_symlink_error) {
    const auto parent_race_destination = active_parent / "match.json";
    bool parent_race_hook_ran = false;
    detail::write_calibration_json_atomically(R"json({"writer":"windows-pinned-parent"})json",
                                              parent_race_destination, left_input, right_input,
                                              [&] {
                                                std::filesystem::remove(active_parent);
                                                std::filesystem::create_directory(active_parent);
                                                parent_race_hook_ran = true;
                                              });
    expect_true(parent_race_hook_ran,
                "Windows output parent replacement happens at the publication boundary");
    expect_eq(read_text_file(retained_parent / "match.json"),
              std::string("{\"writer\":\"windows-pinned-parent\"}\n"),
              "Windows publication stays relative to the retained directory handle");
    expect_true(!std::filesystem::exists(active_parent / "match.json"),
                "Windows replacement parent cannot redirect publication");
  }

  const auto windows_directory_race_parent = root.path() / "windows-directory-race-parent";
  const auto windows_directory_race_moved = root.path() / "windows-directory-race-moved";
  const auto windows_directory_race_destination = windows_directory_race_parent / "match.json";
  std::filesystem::create_directory(windows_directory_race_parent);
  bool windows_directory_race_blocked = false;
  detail::write_calibration_json_atomically(
      R"json({"writer":"windows-directory-race"})json", windows_directory_race_destination,
      left_input, right_input, {}, {}, {}, false, [&] {
        std::error_code rename_error;
        std::filesystem::rename(windows_directory_race_parent, windows_directory_race_moved,
                                rename_error);
        windows_directory_race_blocked = static_cast<bool>(rename_error);
      });
  expect_true(windows_directory_race_blocked,
              "Windows final commit gate blocks output-directory replacement");
  expect_eq(read_text_file(windows_directory_race_destination),
            std::string("{\"writer\":\"windows-directory-race\"}\n"),
            "Windows directory race publishes into the pinned directory");
  expect_true(!std::filesystem::exists(windows_directory_race_moved),
              "blocked Windows directory race moves no directory");

  const auto windows_reader_destination = root.path() / "windows-reader.json";
  write_text_file(windows_reader_destination, "Windows retained reader output\n");
  const HANDLE windows_reader = CreateFileW(windows_reader_destination.c_str(), GENERIC_READ,
                                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  expect_true(windows_reader != INVALID_HANDLE_VALUE,
              "Windows retained destination reader fixture opens");
  if (windows_reader != INVALID_HANDLE_VALUE) {
    detail::write_calibration_json_atomically(R"json({"writer":"windows-reader"})json",
                                              windows_reader_destination, left_input, right_input);
    std::array<char, 128> old_contents{};
    DWORD old_contents_size = 0;
    expect_true(ReadFile(windows_reader, old_contents.data(),
                         static_cast<DWORD>(old_contents.size()), &old_contents_size, nullptr) != 0,
                "Windows retained destination reader remains readable after replacement");
    expect_eq(std::string(old_contents.data(), old_contents_size),
              std::string("Windows retained reader output\n"),
              "Windows retained destination reader observes the displaced file");
    expect_eq(read_text_file(windows_reader_destination),
              std::string("{\"writer\":\"windows-reader\"}\n"),
              "Windows destination path observes the committed replacement");
    (void)CloseHandle(windows_reader);
  }

  const auto windows_legacy_create = root.path() / "windows-legacy-create.json";
  detail::write_calibration_json_atomically(R"json({"writer":"windows-legacy-create"})json",
                                            windows_legacy_create, left_input, right_input, {}, {},
                                            {}, true);
  expect_eq(read_text_file(windows_legacy_create),
            std::string("{\"writer\":\"windows-legacy-create\"}\n"),
            "Windows class-10 fallback can publish a new output without replacement");
  const auto windows_legacy_replace = root.path() / "windows-legacy-replace.json";
  write_text_file(windows_legacy_replace, "Windows legacy replacement original\n");
  bool windows_legacy_replace_rejected = false;
  try {
    detail::write_calibration_json_atomically(R"json({"writer":"windows-legacy-replace"})json",
                                              windows_legacy_replace, left_input, right_input, {},
                                              {}, {}, true);
  } catch (const std::system_error& error) {
    windows_legacy_replace_rejected = error.code().value() == ERROR_NOT_SUPPORTED;
  }
  expect_true(windows_legacy_replace_rejected,
              "Windows class-10 fallback rejects existing-output replacement");
  expect_eq(read_text_file(windows_legacy_replace),
            std::string("Windows legacy replacement original\n"),
            "rejected Windows class-10 replacement preserves the original output");

  const auto windows_post_publish_destination = root.path() / "windows-post-publish-race.json";
  const auto windows_post_publish_retained =
      root.path() / "windows-post-publish-race-retained.json";
  write_text_file(windows_post_publish_destination, "Windows pre-commit original output\n");
  bool windows_post_publish_hook_ran = false;
  bool windows_post_publish_substitution_blocked = false;
  bool windows_post_publish_substitution_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"windows-post-publish-race"})json", windows_post_publish_destination,
        left_input, right_input, {}, {}, {}, false, [&] {
          windows_post_publish_hook_ran = true;
          std::error_code rename_error;
          std::filesystem::rename(windows_post_publish_destination, windows_post_publish_retained,
                                  rename_error);
          windows_post_publish_substitution_blocked = static_cast<bool>(rename_error);
          if (!rename_error) {
            write_text_file(windows_post_publish_destination, "Windows pre-commit replacement\n");
          }
        });
  } catch (const std::exception& error) {
    windows_post_publish_substitution_rejected =
        std::string_view(error.what()).find("destination output changed at the Windows commit") !=
        std::string_view::npos;
  }
  expect_true(windows_post_publish_hook_ran, "Windows final commit gate hook runs");
  expect_true(windows_post_publish_substitution_blocked ||
                  windows_post_publish_substitution_rejected,
              "Windows final commit gate blocks or rejects destination substitution");
  if (windows_post_publish_substitution_blocked) {
    expect_eq(read_text_file(windows_post_publish_destination),
              std::string("{\"writer\":\"windows-post-publish-race\"}\n"),
              "blocked Windows final commit race publishes the intended output");
    expect_true(!std::filesystem::exists(windows_post_publish_retained),
                "blocked Windows final commit race creates no retained substitute");
  } else {
    expect_eq(read_text_file(windows_post_publish_destination),
              std::string("Windows pre-commit replacement\n"),
              "rejected Windows final commit race preserves the concurrent replacement");
    expect_eq(read_text_file(windows_post_publish_retained),
              std::string("Windows pre-commit original output\n"),
              "rejected Windows final commit race preserves the moved original output");
  }

  const auto windows_replace_race_destination = root.path() / "windows-replace-race.json";
  const auto windows_replace_race_retained = root.path() / "windows-replace-race-retained.json";
  write_text_file(windows_replace_race_destination, "Windows replace original output\n");
  bool windows_replace_race_hook_ran = false;
  bool windows_replace_race_blocked = false;
  bool windows_replace_race_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"windows-replace-race"})json", windows_replace_race_destination,
        left_input, right_input, {}, {}, {}, false, {}, {}, std::chrono::seconds(2),
        [&](const std::filesystem::path&) {
          windows_replace_race_hook_ran = true;
          std::error_code rename_error;
          std::filesystem::rename(windows_replace_race_destination, windows_replace_race_retained,
                                  rename_error);
          windows_replace_race_blocked = static_cast<bool>(rename_error);
          if (!rename_error) {
            write_text_file(windows_replace_race_destination, "Windows concurrent replacement\n");
          }
        });
  } catch (const std::exception& error) {
    windows_replace_race_rejected =
        std::string_view(error.what()).find("destination output changed before Windows") !=
        std::string_view::npos;
  }
  expect_true(windows_replace_race_hook_ran, "Windows exact-replacement race hook runs");
  expect_true(windows_replace_race_blocked || windows_replace_race_rejected,
              "Windows exact-replacement race is blocked or rejected");
  if (windows_replace_race_blocked) {
    expect_eq(read_text_file(windows_replace_race_destination),
              std::string("{\"writer\":\"windows-replace-race\"}\n"),
              "blocked Windows exact-replacement race publishes the intended output");
    expect_true(!std::filesystem::exists(windows_replace_race_retained),
                "blocked Windows exact-replacement race creates no retained substitute");
  } else {
    expect_eq(read_text_file(windows_replace_race_destination),
              std::string("Windows concurrent replacement\n"),
              "rejected Windows exact-replacement race preserves the concurrent output");
    expect_eq(read_text_file(windows_replace_race_retained),
              std::string("Windows replace original output\n"),
              "rejected Windows exact-replacement race preserves the moved original output");
  }

  const auto windows_removed_destination = root.path() / "windows-removed-destination.json";
  const auto windows_removed_destination_retained =
      root.path() / "windows-removed-destination-retained.json";
  write_text_file(windows_removed_destination, "Windows removed destination original output\n");
  bool windows_removed_destination_blocked = false;
  bool windows_removed_destination_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"windows-removed-destination"})json", windows_removed_destination,
        left_input, right_input, {}, {}, {}, false, {}, {}, std::chrono::seconds(2),
        [&](const std::filesystem::path&) {
          std::error_code rename_error;
          std::filesystem::rename(windows_removed_destination, windows_removed_destination_retained,
                                  rename_error);
          windows_removed_destination_blocked = static_cast<bool>(rename_error);
        });
  } catch (const std::exception& error) {
    windows_removed_destination_rejected =
        std::string_view(error.what()).find("destination output changed before Windows") !=
        std::string_view::npos;
  }
  expect_true(windows_removed_destination_blocked || windows_removed_destination_rejected,
              "Windows destination removal is blocked or rejected");
  if (windows_removed_destination_blocked) {
    expect_eq(read_text_file(windows_removed_destination),
              std::string("{\"writer\":\"windows-removed-destination\"}\n"),
              "blocked Windows destination removal publishes the intended output");
    expect_true(!std::filesystem::exists(windows_removed_destination_retained),
                "blocked Windows destination removal creates no retained entry");
  } else {
    expect_true(!std::filesystem::exists(windows_removed_destination),
                "rejected Windows destination removal does not recreate the output");
    expect_eq(read_text_file(windows_removed_destination_retained),
              std::string("Windows removed destination original output\n"),
              "rejected Windows destination removal preserves the moved original output");
  }

  const auto windows_source_race_destination = root.path() / "windows-source-race.json";
  const auto windows_source_race_retained = root.path() / "windows-source-race-retained.json";
  write_text_file(windows_source_race_destination, "Windows source original output\n");
  std::filesystem::path windows_source_race_substitute;
  bool windows_source_race_blocked = false;
  detail::write_calibration_json_atomically(
      R"json({"writer":"windows-source-race"})json", windows_source_race_destination, left_input,
      right_input, {}, {}, {}, false, {}, {}, std::chrono::seconds(2),
      [&](const std::filesystem::path& temporary) {
        windows_source_race_substitute = temporary;
        std::error_code rename_error;
        std::filesystem::rename(temporary, windows_source_race_retained, rename_error);
        windows_source_race_blocked = static_cast<bool>(rename_error);
        if (!rename_error) {
          write_text_file(windows_source_race_substitute,
                          "Windows source concurrent replacement\n");
        }
      });
  expect_true(windows_source_race_blocked,
              "Windows replacement-source race is blocked by the retained temporary handle");
  expect_eq(read_text_file(windows_source_race_destination),
            std::string("{\"writer\":\"windows-source-race\"}\n"),
            "Windows replacement-source race publishes the intended output");
  expect_true(!std::filesystem::exists(windows_source_race_retained),
              "blocked Windows replacement-source race moves no output");
  expect_true(!std::filesystem::exists(windows_source_race_substitute),
              "published Windows temporary name no longer exists");

#endif

  const auto commit_alias_destination = root.path() / "commit-input-alias-output.json";
  bool commit_alias_rejected = false;
  try {
    detail::write_calibration_json_atomically(
        R"json({"writer":"input-alias"})json", commit_alias_destination, left_input, right_input,
        {}, {}, [&] { std::filesystem::create_hard_link(left_input, commit_alias_destination); });
  } catch (const std::exception& error) {
    commit_alias_rejected =
        std::string_view(error.what()).find("left video input") != std::string_view::npos;
  }
  expect_true(commit_alias_rejected, "commit-bound destination alias of an input is rejected");
  expect_true(std::filesystem::equivalent(left_input, commit_alias_destination),
              "rejected commit-bound alias preserves the input identity");
  std::filesystem::remove(commit_alias_destination);
#endif

  bool orphaned_temporary = false;
  for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
    const auto encoded_filename = entry.path().filename().u8string();
    const std::string filename(reinterpret_cast<const char*>(encoded_filename.data()),
                               encoded_filename.size());
    if (filename.starts_with("match.json.tmp.") || filename.starts_with("blocked.json.tmp.") ||
        filename.starts_with("raced-match.json.tmp.") ||
        filename.find(".publish.") != std::string::npos ||
        filename.find(".quarantine.") != std::string::npos ||
        filename.find(".rollback.") != std::string::npos) {
      orphaned_temporary = true;
    }
  }
  expect_true(!orphaned_temporary, "calibration replacement leaves no temporary files");
}

void command_execution_dispatches_available_stages() {
  const auto calibration_path = write_valid_calibration_file();
  std::ostringstream out;
  std::ostringstream err;
  const auto help_status = run_command(HelpCommand{}, out, err);
  expect_eq(help_status, 0, "help exits success");
  expect_true(out.str().find("Usage:") != std::string::npos, "help writes usage");
  expect_true(err.str().empty(), "help writes no stderr");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  const auto info_status = run_command(InfoCommand{}, out, err);
  expect_eq(info_status, 0, "info exits success");
  expect_true(out.str().find("Reco C++ capability report") != std::string::npos,
              "info writes report heading");
  expect_true(out.str().find("CUDA:") != std::string::npos, "info writes CUDA probe");
  expect_true(out.str().find("GStreamer:") != std::string::npos, "info writes GStreamer probe");
  expect_true(out.str().find("AI providers:") != std::string::npos, "info writes AI probe");
  expect_true(err.str().empty(), "info writes no stderr");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  StitchCommand stitch{
      .left = "left.mp4", .right = "right.mp4", .calibration = calibration_path.string()};
  const auto stitch_status = run_command(Command{stitch}, out, err);
  expect_eq(stitch_status, 2, "stitch exits blocked");
  expect_true(out.str().find("C++ reco stitch runtime plan") != std::string::npos,
              "blocked stitch writes runtime plan");
  expect_true(out.str().find("nvv4l2decoder") != std::string::npos,
              "blocked stitch describes GPU decode contract");
  expect_true(out.str().find("qtdemux ! capsfilter "
                             "caps=\"video/x-h264;video/x-h265;video/x-av1\" ! "
                             "parsebin ! identity name=display_info silent=true ! "
                             "nvv4l2decoder") != std::string::npos,
              "blocked stitch selects a supported video pad for containers");
  expect_true(out.str().find("video/x-raw(memory:NVMM),format=NV12") != std::string::npos,
              "blocked stitch preserves NVMM decode caps");
  expect_true(err.str().find("error:") != std::string::npos, "blocked stitch writes stderr");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  const auto cancelled_stitch_status =
      run_command(Command{stitch}, out, err, {}, [] { return true; });
  expect_eq(cancelled_stitch_status, kCancelledExitCode,
            "pre-cancelled stitch uses the cancellation exit status");
  expect_true(out.str().empty(), "pre-cancelled stitch starts no runtime plan");
  expect_eq(err.str(), std::string("cancelled\n"), "pre-cancelled stitch reports cancellation");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  stitch.no_zero_copy = true;
  const auto cpu_stitch_status = run_command(Command{stitch}, out, err);
  expect_eq(cpu_stitch_status, 2, "stitch no-zero-copy exits blocked");
  expect_true(err.str().find("force a CPU path") != std::string::npos,
              "stitch rejects CPU decode fallback");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  CalibrateCommand calibrate{.left = "left.mp4", .right = "right.mp4"};
#if defined(__linux__)
  const auto fake_nvbufsurface = find_shared_library_runfile("fake_nvbufsurface");
  ScopedEnvironment nvbufsurface_runtime("RECO_NVBUFSURFACE_DYLIB_PATH",
                                         fake_nvbufsurface.string());
  ScopedEnvironment nvds_utils_runtime("RECO_NVDS_UTILS_DYLIB_PATH", fake_nvbufsurface.string());
  ScopedEnvironment deepstream_version("RECO_FAKE_DEEPSTREAM_VERSION", "9.1");
  {
    ScopedEnvironment unsupported_version("RECO_FAKE_DEEPSTREAM_VERSION", "8.0");
    const auto unsupported_abi_status = run_command(Command{calibrate}, out, err);
    expect_eq(unsupported_abi_status, 2, "calibrate rejects an unsupported NvBufSurface ABI");
    expect_true(err.str().find("cannot discover the installed NvBufSurface ABI") !=
                    std::string::npos,
                "calibrate reports NvBufSurface ABI discovery failure");
  }
  out.str("");
  out.clear();
  err.str("");
  err.clear();
#endif
  const auto calibrate_status = run_command(Command{calibrate}, out, err);
#if defined(__linux__)
  expect_eq(calibrate_status, 2, "calibrate exits blocked when required GPU backends are absent");
  expect_true(out.str().find("GPU calibration plan") != std::string::npos,
              "calibrate writes GPU plan");
  expect_true(out.str().find("FeatureMatching") != std::string::npos,
              "calibrate writes AKAZE stage");
#else
  expect_eq(calibrate_status, 2, "calibrate exits blocked without Linux NvBufSurface discovery");
  expect_true(err.str().find("cannot discover the installed NvBufSurface ABI") != std::string::npos,
              "calibrate reports unsupported platform ABI discovery");
#endif
  expect_true(err.str().find("error:") != std::string::npos, "calibrate writes stderr error");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  PreviewCommand preview{
      .left = "left.mp4", .right = "right.mp4", .calibration = calibration_path.string()};
  const auto preview_status = run_command(Command{preview}, out, err);
  expect_eq(preview_status, 2, "preview exits blocked");
  expect_true(out.str().find("C++ reco preview runtime plan") != std::string::npos,
              "preview writes runtime plan");
  expect_true(out.str().find("left GPU decode") != std::string::npos,
              "preview writes decode pipeline");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  PreviewCommand hevc_preview{
      .left = "left.hevc", .right = "right.h265", .calibration = calibration_path.string()};
  const auto hevc_preview_status = run_command(Command{hevc_preview}, out, err);
  expect_eq(hevc_preview_status, 2, "HEVC preview exits blocked");
  expect_true(out.str().find("h265parse ! identity name=display_info silent=true ! "
                             "nvv4l2decoder") != std::string::npos,
              "HEVC preview plan selects HEVC parser");
  expect_true(out.str().find("qtdemux ! h265parse") == std::string::npos,
              "HEVC preview raw stream bypasses qtdemux");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  PreviewCommand invalid_preview{.left = "left.mp4 ! fakesink",
                                 .right = "right.mp4",
                                 .calibration = calibration_path.string()};
  const auto invalid_preview_status = run_command(Command{invalid_preview}, out, err);
  expect_eq(invalid_preview_status, 2, "invalid preview decode path exits blocked");
  expect_true(err.str().find("metacharacters") != std::string::npos,
              "invalid preview decode path is reported");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  PreviewCommand unsupported_preview{
      .left = "left.avi", .right = "right.mp4", .calibration = calibration_path.string()};
  const auto unsupported_preview_status = run_command(Command{unsupported_preview}, out, err);
  expect_eq(unsupported_preview_status, 2, "unsupported preview container exits blocked");
  expect_true(err.str().find("container is unsupported") != std::string::npos,
              "unsupported preview container is reported");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  PreviewCommand missing_calibration_preview{
      .left = "left.mp4", .right = "right.mp4", .calibration = "missing-match.json"};
  const auto missing_calibration_status =
      run_command(Command{missing_calibration_preview}, out, err);
  expect_eq(missing_calibration_status, 2, "missing preview calibration exits blocked");
  expect_true(err.str().find("cannot read calibration file") != std::string::npos,
              "missing preview calibration is reported");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  GoproCommand gopro{.start = true};
  const auto gopro_status = run_command(Command{gopro}, out, err);
  expect_eq(gopro_status, 2, "gopro exits blocked");
  expect_true(out.str().find("C++ reco gopro runtime plan") != std::string::npos,
              "gopro writes runtime plan");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  CameraCommand camera{.left_device = "/dev/video0",
                       .right_device = "/dev/video1",
                       .calibration = calibration_path.string(),
                       .output = "out.mp4",
                       .v4l2_direct = true};
  const auto camera_status = run_command(Command{camera}, out, err);
  expect_eq(camera_status, 2, "camera exits blocked");
  expect_true(out.str().find("V4L2 devices") != std::string::npos,
              "camera writes V4L2-direct plan");
  expect_true(err.str().find("CPU fallback") != std::string::npos ||
                  err.str().find("CUDA is required") != std::string::npos ||
                  err.str().find("NPP is required") != std::string::npos,
              "camera keeps V4L2-direct GPU gated");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  CameraCommand missing_calibration_camera{.left_device = "/dev/video0",
                                           .right_device = "/dev/video1",
                                           .calibration = "missing-match.json",
                                           .output = "out.mp4",
                                           .v4l2_direct = true};
  const auto missing_camera_status = run_command(Command{missing_calibration_camera}, out, err);
  expect_eq(missing_camera_status, 2, "missing camera calibration exits blocked");
  expect_true(err.str().find("cannot read calibration file") != std::string::npos,
              "missing camera calibration is reported");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  LibcameraCommand libcamera{.calibration = calibration_path.string(), .output = "out.mp4"};
  const auto libcamera_status = run_command(Command{libcamera}, out, err);
  expect_eq(libcamera_status, 2, "libcamera exits blocked");
  expect_true(out.str().find("rpicam-vid") != std::string::npos, "libcamera writes rpicam plan");
  expect_true(err.str().find("CPU YUV420P") != std::string::npos,
              "libcamera refuses CPU-resident path");

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  LibcameraCommand missing_calibration_libcamera{.calibration = "missing-match.json",
                                                 .output = "out.mp4"};
  const auto missing_libcamera_status =
      run_command(Command{missing_calibration_libcamera}, out, err);
  expect_eq(missing_libcamera_status, 2, "missing libcamera calibration exits blocked");
  expect_true(err.str().find("cannot read calibration file") != std::string::npos,
              "missing libcamera calibration is reported");
}

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
  if (argc >= 2 && std::wstring_view(argv[1]) == L"--reco-publication-writer-child") {
    return run_windows_publication_writer_child(argc, argv);
  }
#else
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
#endif
  run_test_case("validators_match_rust", validators_match_rust);
  run_test_case("stitch_parse_matches_rust_defaults", stitch_parse_matches_rust_defaults);
  run_test_case("stitch_frame_timing_preserves_source_gaps_and_rejects_overflow",
                stitch_frame_timing_preserves_source_gaps_and_rejects_overflow);
  run_test_case("stitch_audio_is_clipped_by_presentation_time_at_the_video_boundary",
                stitch_audio_is_clipped_by_presentation_time_at_the_video_boundary);
  run_test_case("stitch_second_conversion_supports_the_unsigned_gstreamer_range",
                stitch_second_conversion_supports_the_unsigned_gstreamer_range);
  run_test_case("stitch_frame_window_uses_one_rounded_timeline",
                stitch_frame_window_uses_one_rounded_timeline);
  run_test_case("stitch_descriptor_budget_is_checked_before_input_acquisition",
                stitch_descriptor_budget_is_checked_before_input_acquisition);
  run_test_case("stitch_output_transaction_is_descriptor_pinned_and_atomic",
                stitch_output_transaction_is_descriptor_pinned_and_atomic);
  run_test_case("interrupt_request_unwinds_stitch_output_staging",
                interrupt_request_unwinds_stitch_output_staging);
  run_test_case("preview_and_calibrate_parse_matches_rust_defaults",
                preview_and_calibrate_parse_matches_rust_defaults);
  run_test_case("live_command_parse_matches_rust_defaults",
                live_command_parse_matches_rust_defaults);
  run_test_case("parse_errors_are_reported", parse_errors_are_reported);
  run_test_case("production_windows_cli_preserves_unicode_paths",
                production_windows_cli_preserves_unicode_paths);
  run_test_case("probe_worker_discovery_handles_path_and_bzlmod_runfiles",
                probe_worker_discovery_handles_path_and_bzlmod_runfiles);
  run_test_case("calibration_output_replacement_is_exclusive_and_atomic",
                calibration_output_replacement_is_exclusive_and_atomic);
  run_test_case("command_execution_dispatches_available_stages",
                command_execution_dispatches_available_stages);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
