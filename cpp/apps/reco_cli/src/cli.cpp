#include "reco/cli/cli.hpp"

#include "reco/calibrate/pipeline.hpp"
#include "reco/core/calibration.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/detect/coreml_session.hpp"
#include "reco/detect/npp_interop.hpp"
#include "reco/detect/ort_session.hpp"
#include "reco/detect/probe.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gstreamer.hpp"
#include "rules_cc/cc/runfiles/runfiles.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

namespace reco::cli {
namespace {

using rules_cc::cc::runfiles::Runfiles;

#if defined(_WIN32)
constexpr std::string_view probe_worker_name = "reco_video_probe_worker.exe";
constexpr std::string_view calibration_worker_name = "reco_calibration_worker.exe";
constexpr char path_separator = ';';
#else
constexpr std::string_view probe_worker_name = "reco_video_probe_worker";
constexpr std::string_view calibration_worker_name = "reco_calibration_worker";
constexpr char path_separator = ':';
#endif

std::optional<std::filesystem::path>
existing_absolute_executable(const std::filesystem::path& path) {
  std::error_code error;
  if (path.empty() || !std::filesystem::is_regular_file(path, error) || error) {
    return std::nullopt;
  }
#if !defined(_WIN32)
  if (::access(path.c_str(), X_OK) != 0) {
    return std::nullopt;
  }
#endif
  auto absolute = std::filesystem::absolute(path, error);
  if (error) {
    return std::nullopt;
  }
  return absolute.lexically_normal();
}

std::optional<std::filesystem::path>
resolve_path_invocation(const std::filesystem::path& executable_path) {
  if (executable_path.empty()) {
    return std::nullopt;
  }
  if (executable_path.is_absolute() || executable_path.has_parent_path()) {
    return existing_absolute_executable(executable_path);
  }

  const char* path_value = std::getenv("PATH");
  if (path_value == nullptr) {
    return std::nullopt;
  }
  const std::string path(path_value);
  std::size_t begin = 0;
  while (begin <= path.size()) {
    const auto end = path.find(path_separator, begin);
    const auto component = path.substr(begin, end - begin);
    const auto directory =
        component.empty() ? std::filesystem::path(".") : std::filesystem::path(component);
    if (auto resolved = existing_absolute_executable(directory / executable_path);
        resolved.has_value()) {
      return resolved;
    }
#if defined(_WIN32)
    if (!executable_path.has_extension()) {
      auto with_extension = executable_path;
      with_extension += ".exe";
      if (auto resolved = existing_absolute_executable(directory / with_extension);
          resolved.has_value()) {
        return resolved;
      }
    }
#endif
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return std::nullopt;
}

std::optional<std::filesystem::path>
resolve_worker_impl(const std::filesystem::path& executable_path, const char* environment_name,
                    std::string_view name, std::string_view runfiles_directory,
                    std::string_view output_directory) {
  if (const char* configured = std::getenv(environment_name);
      configured != nullptr && configured[0] != '\0') {
    return existing_absolute_executable(configured);
  }

  const auto resolved_executable = resolve_path_invocation(executable_path);
  if (resolved_executable.has_value()) {
    if (auto sibling = existing_absolute_executable(resolved_executable->parent_path() / name);
        sibling.has_value()) {
      return sibling;
    }
  }

  std::string runfiles_error;
  const auto runfiles_argv0 = resolved_executable.value_or(executable_path).string();
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(runfiles_argv0, BAZEL_CURRENT_REPOSITORY, &runfiles_error));
  if (runfiles != nullptr) {
    const auto logical_path = std::string("reco_video_stitcher/") +
                              std::string(runfiles_directory) + "/" + std::string(name);
    if (auto resolved = existing_absolute_executable(runfiles->Rlocation(logical_path));
        resolved.has_value()) {
      return resolved;
    }
  }

  if (resolved_executable.has_value()) {
    if (auto output_tree_worker = existing_absolute_executable(
            resolved_executable->parent_path() / ".." / ".." / output_directory / name);
        output_tree_worker.has_value()) {
      return output_tree_worker;
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path>
resolve_video_probe_worker_impl(const std::filesystem::path& executable_path) {
  return resolve_worker_impl(executable_path, "RECO_VIDEO_PROBE_WORKER", probe_worker_name,
                             "cpp/reco_io", "reco_io");
}

std::optional<std::filesystem::path>
resolve_calibration_worker_impl(const std::filesystem::path& executable_path) {
  return resolve_worker_impl(executable_path, "RECO_CALIBRATION_WORKER", calibration_worker_name,
                             "cpp/reco_calibrate", "reco_calibrate");
}

[[noreturn]] void throw_file_error(std::string_view operation, const std::filesystem::path& path,
                                   int error) {
  throw std::system_error(error, std::system_category(),
                          std::string(operation) + " " + path.string());
}

#if defined(_WIN32)
std::filesystem::path create_exclusive_temporary(const std::filesystem::path& destination,
                                                 HANDLE& handle) {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  for (int attempt = 0; attempt < 128; ++attempt) {
    std::array<char, 32> token{};
    for (auto& digit : token) {
      digit = hex[random() & 0x0fU];
    }
    auto filename = destination.filename();
    filename += ".tmp." + std::string(token.begin(), token.end());
    auto temporary = destination.parent_path() / filename;
    handle = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle != INVALID_HANDLE_VALUE) {
      return temporary;
    }
    const auto error = GetLastError();
    if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS) {
      throw_file_error("cannot create temporary calibration output", temporary,
                       static_cast<int>(error));
    }
  }
  throw std::runtime_error("cannot create unique temporary calibration output for " +
                           destination.string());
}

void write_all(HANDLE handle, std::string_view contents, const std::filesystem::path& temporary) {
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const auto remaining = contents.size() - offset;
    const auto chunk =
        static_cast<DWORD>(std::min<std::size_t>(remaining, std::numeric_limits<DWORD>::max()));
    DWORD written = 0;
    if (WriteFile(handle, contents.data() + offset, chunk, &written, nullptr) == 0 ||
        written == 0) {
      throw_file_error("failed to write temporary calibration output", temporary,
                       static_cast<int>(GetLastError()));
    }
    offset += written;
  }
}
#elif defined(__linux__)
class UniqueFileDescriptor {
public:
  UniqueFileDescriptor() = default;
  explicit UniqueFileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ~UniqueFileDescriptor() {
    if (descriptor_ >= 0) {
      (void)::close(descriptor_);
    }
  }

  UniqueFileDescriptor(const UniqueFileDescriptor&) = delete;
  UniqueFileDescriptor& operator=(const UniqueFileDescriptor&) = delete;
  UniqueFileDescriptor(UniqueFileDescriptor&& other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}
  UniqueFileDescriptor& operator=(UniqueFileDescriptor&& other) noexcept {
    if (this != &other) {
      if (descriptor_ >= 0) {
        (void)::close(descriptor_);
      }
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }

  [[nodiscard]] int get() const { return descriptor_; }

  void close_checked(const std::filesystem::path& path) {
    const int descriptor = std::exchange(descriptor_, -1);
    if (descriptor >= 0 && ::close(descriptor) != 0) {
      throw_file_error("failed to close temporary calibration output", path, errno);
    }
  }

private:
  int descriptor_ = -1;
};

struct PinnedFileIdentity {
  UniqueFileDescriptor descriptor;
  struct stat identity{};

  [[nodiscard]] std::filesystem::path retained_path() const {
    return std::filesystem::path("/proc") / std::to_string(::getpid()) / "fd" /
           std::to_string(descriptor.get());
  }
};

[[nodiscard]] PinnedFileIdentity pin_input_identity(const std::filesystem::path& path,
                                                    std::string_view label) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    throw_file_error("cannot open " + std::string(label) + " video input", path, errno);
  }
  PinnedFileIdentity pinned{.descriptor = UniqueFileDescriptor(descriptor)};
  if (::fstat(descriptor, &pinned.identity) != 0) {
    throw_file_error("cannot inspect " + std::string(label) + " video input", path, errno);
  }
  if (!S_ISREG(pinned.identity.st_mode)) {
    throw std::runtime_error(std::string(label) +
                             " video input must be a regular file: " + path.string());
  }
  return pinned;
}

[[nodiscard]] bool same_file_identity(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

[[nodiscard]] bool temporary_name_identifies_descriptor(int directory_descriptor,
                                                        std::string_view name,
                                                        int temporary_descriptor) {
  struct stat descriptor_identity{};
  struct stat name_identity{};
  const std::string filename(name);
  return ::fstat(temporary_descriptor, &descriptor_identity) == 0 &&
         ::fstatat(directory_descriptor, filename.c_str(), &name_identity, AT_SYMLINK_NOFOLLOW) ==
             0 &&
         S_ISREG(name_identity.st_mode) && same_file_identity(descriptor_identity, name_identity);
}

[[nodiscard]] std::optional<std::string>
validate_pinned_output_identity(int directory_descriptor, std::string_view destination_name,
                                const PinnedFileIdentity& left_input,
                                const PinnedFileIdentity& right_input) {
  struct stat destination_identity{};
  const std::string name(destination_name);
  if (::fstatat(directory_descriptor, name.c_str(), &destination_identity, AT_SYMLINK_NOFOLLOW) !=
      0) {
    if (errno == ENOENT) {
      return std::nullopt;
    }
    throw std::system_error(errno, std::system_category(),
                            "cannot inspect output calibration path identity");
  }

  const auto alias_error = [](std::string_view label) {
    return "output calibration path identifies the " + std::string(label) + " video input";
  };
  if (same_file_identity(destination_identity, left_input.identity)) {
    return alias_error("left");
  }
  if (same_file_identity(destination_identity, right_input.identity)) {
    return alias_error("right");
  }

  if (S_ISLNK(destination_identity.st_mode)) {
    struct stat followed_identity{};
    if (::fstatat(directory_descriptor, name.c_str(), &followed_identity, 0) != 0) {
      if (errno == ENOENT || errno == ELOOP) {
        return std::nullopt;
      }
      throw std::system_error(errno, std::system_category(),
                              "cannot inspect output calibration symlink target identity");
    }
    if (same_file_identity(followed_identity, left_input.identity)) {
      return alias_error("left");
    }
    if (same_file_identity(followed_identity, right_input.identity)) {
      return alias_error("right");
    }
  }
  return std::nullopt;
}

struct TemporaryOutput {
  std::string name;
  UniqueFileDescriptor descriptor;
};

[[nodiscard]] TemporaryOutput
create_exclusive_temporary_at(int directory_descriptor, const std::filesystem::path& destination) {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  for (int attempt = 0; attempt < 128; ++attempt) {
    std::array<char, 32> token{};
    for (auto& digit : token) {
      digit = hex[random() & 0x0fU];
    }
    const auto name =
        destination.filename().string() + ".tmp." + std::string(token.begin(), token.end());
    const int descriptor = ::openat(directory_descriptor, name.c_str(),
                                    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor >= 0) {
      return TemporaryOutput{.name = name, .descriptor = UniqueFileDescriptor(descriptor)};
    }
    if (errno != EEXIST) {
      throw_file_error("cannot create temporary calibration output", destination, errno);
    }
  }
  throw std::runtime_error("cannot create unique temporary calibration output for " +
                           destination.string());
}

void write_all(int descriptor, std::string_view contents, const std::filesystem::path& temporary) {
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const auto written = ::write(descriptor, contents.data() + offset, contents.size() - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw_file_error("failed to write temporary calibration output", temporary, errno);
    }
    if (written == 0) {
      throw_file_error("failed to write temporary calibration output", temporary, EIO);
    }
    offset += static_cast<std::size_t>(written);
  }
}
#else
std::filesystem::path create_exclusive_temporary(const std::filesystem::path& destination,
                                                 int& descriptor) {
  auto temporary = destination.parent_path() / (destination.filename().string() + ".tmp.XXXXXX");
  auto mutable_path = temporary.string();
  std::vector<char> buffer(mutable_path.begin(), mutable_path.end());
  buffer.push_back('\0');
  descriptor = ::mkstemp(buffer.data());
  if (descriptor < 0) {
    throw_file_error("cannot create temporary calibration output", temporary, errno);
  }
  return std::filesystem::path(buffer.data());
}

void write_all(int descriptor, std::string_view contents, const std::filesystem::path& temporary) {
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const auto written = ::write(descriptor, contents.data() + offset, contents.size() - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw_file_error("failed to write temporary calibration output", temporary, errno);
    }
    if (written == 0) {
      throw_file_error("failed to write temporary calibration output", temporary, EIO);
    }
    offset += static_cast<std::size_t>(written);
  }
}
#endif

void write_calibration_json_atomically_impl(std::string_view json,
                                            const std::filesystem::path& destination,
                                            const std::filesystem::path& left_input,
                                            const std::filesystem::path& right_input,
                                            const std::function<void()>& before_publish) {
  std::string contents(json);
  contents.push_back('\n');
#if defined(_WIN32)
  HANDLE handle = INVALID_HANDLE_VALUE;
  std::filesystem::path temporary;
  bool temporary_exists = false;
  try {
    temporary = create_exclusive_temporary(destination, handle);
    temporary_exists = true;
    write_all(handle, contents, temporary);
    if (FlushFileBuffers(handle) == 0) {
      throw_file_error("failed to flush temporary calibration output", temporary,
                       static_cast<int>(GetLastError()));
    }
    if (CloseHandle(handle) == 0) {
      handle = INVALID_HANDLE_VALUE;
      throw_file_error("failed to close temporary calibration output", temporary,
                       static_cast<int>(GetLastError()));
    }
    handle = INVALID_HANDLE_VALUE;
    if (before_publish) {
      before_publish();
    }
    if (const auto error = reco::calibrate::validate_calibration_output_identity(
            left_input, right_input, destination);
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }
    if (MoveFileExW(temporary.c_str(), destination.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
      throw_file_error("failed to replace calibration output", destination,
                       static_cast<int>(GetLastError()));
    }
    temporary_exists = false;
  } catch (...) {
    if (handle != INVALID_HANDLE_VALUE) {
      CloseHandle(handle);
    }
    if (temporary_exists) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
    throw;
  }
#elif defined(__linux__)
  const auto left_identity = pin_input_identity(left_input, "left");
  const auto right_identity = pin_input_identity(right_input, "right");
  const auto parent =
      destination.has_parent_path() ? destination.parent_path() : std::filesystem::path(".");
  const auto destination_name = destination.filename().string();
  if (destination_name.empty() || destination_name == "." || destination_name == "..") {
    throw std::runtime_error("calibration output path must name a file: " + destination.string());
  }
  const int raw_directory_descriptor = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (raw_directory_descriptor < 0) {
    throw_file_error("cannot open calibration output directory", parent, errno);
  }
  UniqueFileDescriptor directory_descriptor(raw_directory_descriptor);
  struct stat directory_identity{};
  if (::fstat(directory_descriptor.get(), &directory_identity) != 0) {
    throw_file_error("cannot inspect calibration output directory", parent, errno);
  }
  if (!S_ISDIR(directory_identity.st_mode)) {
    throw std::runtime_error("calibration output parent is not a directory: " + parent.string());
  }

  TemporaryOutput temporary;
  bool temporary_exists = false;
  try {
    temporary = create_exclusive_temporary_at(directory_descriptor.get(), destination);
    temporary_exists = true;
    const auto temporary_path = parent / temporary.name;
    write_all(temporary.descriptor.get(), contents, temporary_path);
    if (::fsync(temporary.descriptor.get()) != 0) {
      throw_file_error("failed to flush temporary calibration output", temporary_path, errno);
    }
    if (before_publish) {
      before_publish();
    }
    if (!temporary_name_identifies_descriptor(directory_descriptor.get(), temporary.name,
                                              temporary.descriptor.get())) {
      temporary_exists = false;
      throw std::runtime_error(
          "refusing to publish calibration output: temporary output identity changed");
    }
    if (const auto error = validate_pinned_output_identity(
            directory_descriptor.get(), destination_name, left_identity, right_identity);
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }
    if (::renameat(directory_descriptor.get(), temporary.name.c_str(), directory_descriptor.get(),
                   destination_name.c_str()) != 0) {
      throw_file_error("failed to replace calibration output", destination, errno);
    }
    temporary_exists = false;
    temporary.descriptor.close_checked(destination);
    if (::fsync(directory_descriptor.get()) != 0) {
      throw_file_error("failed to flush calibration output directory", parent, errno);
    }
  } catch (...) {
    if (temporary_exists &&
        temporary_name_identifies_descriptor(directory_descriptor.get(), temporary.name,
                                             temporary.descriptor.get())) {
      (void)::unlinkat(directory_descriptor.get(), temporary.name.c_str(), 0);
    }
    throw;
  }
#else
  int descriptor = -1;
  std::filesystem::path temporary;
  bool temporary_exists = false;
  try {
    temporary = create_exclusive_temporary(destination, descriptor);
    temporary_exists = true;
    write_all(descriptor, contents, temporary);
    if (::fsync(descriptor) != 0) {
      throw_file_error("failed to flush temporary calibration output", temporary, errno);
    }
    const auto close_result = ::close(descriptor);
    descriptor = -1;
    if (close_result != 0) {
      throw_file_error("failed to close temporary calibration output", temporary, errno);
    }
    if (before_publish) {
      before_publish();
    }
    if (const auto error = reco::calibrate::validate_calibration_output_identity(
            left_input, right_input, destination);
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }
    if (::rename(temporary.c_str(), destination.c_str()) != 0) {
      throw_file_error("failed to replace calibration output", destination, errno);
    }
    temporary_exists = false;
  } catch (...) {
    if (descriptor >= 0) {
      ::close(descriptor);
    }
    if (temporary_exists) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
    throw;
  }
#endif
}

void write_calibration_result(const reco::calibrate::CalibrationResult& result,
                              const reco::calibrate::GpuCalibrationRequest& request) {
  const auto json = reco::core::calibration_to_json(result.calibration);
  const auto reparsed = reco::core::parse_match_calibration_json(json);
  if (!reparsed.has_value() || !reparsed->validate().empty()) {
    throw std::runtime_error("calibration result failed serialization validation");
  }

  const auto& left_path =
      request.left.retained_path.has_value() ? *request.left.retained_path : request.left.path;
  const auto& right_path =
      request.right.retained_path.has_value() ? *request.right.retained_path : request.right.path;
  detail::write_calibration_json_atomically(json, request.output, left_path, right_path);
}

void write_calibration_result_summary(const reco::calibrate::CalibrationResult& result,
                                      const std::string& output_path, std::ostream& out) {
  out << "Calibration result:\n"
      << "  output: " << output_path << '\n'
      << "  frames used: " << result.frames_used << '\n'
      << "  total matches: " << result.total_matches << '\n'
      << "  confidence: " << result.confidence * 100.0 << "%\n"
      << "  residual error: " << result.residual_error << '\n';
  if (result.quality.has_value()) {
    out << "  mean reprojection error: " << result.quality->mean_reprojection_error << '\n'
        << "  trimmed reprojection error: " << result.quality->trimmed_reprojection_error << '\n'
        << "  angular error: " << result.quality->angular_error << '\n';
  }
  for (std::size_t index = 0; index < result.per_frame.size(); ++index) {
    const auto& frame = result.per_frame[index];
    out << "  frame " << index << ": keypoints " << frame.keypoints_left << "/"
        << frame.keypoints_right << ", matches " << frame.post_ratio_test << " -> "
        << frame.post_spatial_filter << " -> " << frame.post_ransac << '\n';
  }
}

template <typename T>
std::variant<T, ParseError> parse_integral(std::string_view value, std::string_view name) {
  if (value.empty()) {
    return ParseError{std::string(name) + " requires a value"};
  }
  T parsed{};
  const char* begin = value.data();
  const char* end = value.data() + value.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end) {
    return ParseError{"invalid " + std::string(name) + " " + std::string(value)};
  }
  return parsed;
}

std::variant<double, ParseError> parse_double(std::string_view value, std::string_view name) {
  std::string text(value);
  char* end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (text.empty() || end != text.c_str() + text.size()) {
    return ParseError{"invalid " + std::string(name) + " " + text};
  }
  if (!std::isfinite(parsed)) {
    return ParseError{"invalid " + std::string(name) + " " + text};
  }
  return parsed;
}

std::variant<float, ParseError> parse_float(std::string_view value, std::string_view name) {
  std::string text(value);
  char* end = nullptr;
  const float parsed = std::strtof(text.c_str(), &end);
  if (text.empty() || end != text.c_str() + text.size()) {
    return ParseError{"invalid " + std::string(name) + " " + text};
  }
  if (!std::isfinite(parsed)) {
    return ParseError{"invalid " + std::string(name) + " " + text};
  }
  return parsed;
}

class Cursor {
public:
  explicit Cursor(const std::vector<std::string>& args) : args_(args) {}

  [[nodiscard]] bool empty() const { return index_ >= args_.size(); }
  [[nodiscard]] const std::string& peek() const { return args_[index_]; }
  [[nodiscard]] std::string take() { return args_[index_++]; }

  [[nodiscard]] std::variant<std::string, ParseError> value(std::string_view option,
                                                            bool allow_hyphen_value = false) {
    if (empty()) {
      return ParseError{"missing value for " + std::string(option)};
    }
    if (!allow_hyphen_value && !peek().empty() && peek().front() == '-') {
      return ParseError{"missing value for " + std::string(option)};
    }
    return take();
  }

private:
  const std::vector<std::string>& args_;
  std::size_t index_ = 0;
};

template <typename T>
bool assign_or_error(std::variant<T, ParseError>&& parsed, T& out, std::optional<ParseError>& err) {
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    err = *error;
    return false;
  }
  out = std::get<T>(std::move(parsed));
  return true;
}

std::optional<std::string> take_value_or_error(Cursor& cursor, std::string_view option,
                                               std::optional<ParseError>& err,
                                               bool allow_hyphen_value = false) {
  auto value = cursor.value(option, allow_hyphen_value);
  if (const auto* error = std::get_if<ParseError>(&value)) {
    err = *error;
    return std::nullopt;
  }
  return std::get<std::string>(std::move(value));
}

void write_probe(std::ostream& out, std::string_view label, bool available,
                 std::string_view detail) {
  out << "  " << label << ": " << (available ? "available" : "unavailable");
  if (!detail.empty()) {
    out << " (" << detail << ")";
  }
  out << '\n';
}

struct RuntimePlan {
  std::string command;
  std::vector<std::string> steps;
  std::optional<std::string> blocked_reason;
};

void write_runtime_plan(std::ostream& out, const RuntimePlan& plan) {
  out << "C++ reco " << plan.command << " runtime plan:\n";
  for (const auto& step : plan.steps) {
    out << "  - " << step << '\n';
  }
  if (plan.blocked_reason.has_value()) {
    out << "blocked: " << *plan.blocked_reason << '\n';
  }
}

std::optional<std::string>
require_gpu_video_backend(const reco::calibrate::CalibrationBackendStatus& backends,
                          bool require_nvbufsurface) {
  if (!backends.cuda.available) {
    return "CUDA is required for the C++ GPU video path: " + backends.cuda.detail;
  }
  if (!backends.gstreamer.available) {
    return "GStreamer is required for the C++ GPU video path: " + backends.gstreamer.detail;
  }
  if (!backends.npp.available) {
    return "NPP is required for GPU resize/color interop: " + backends.npp.detail;
  }
  if (require_nvbufsurface && !backends.nvbufsurface.available) {
    return "NvBufSurface is required for zero-copy camera ingest: " + backends.nvbufsurface.detail;
  }
  return std::nullopt;
}

std::optional<std::string>
require_cuda_npp_backend(const reco::calibrate::CalibrationBackendStatus& backends) {
  if (!backends.cuda.available) {
    return "CUDA is required for the C++ GPU video path: " + backends.cuda.detail;
  }
  if (!backends.npp.available) {
    return "NPP is required for GPU resize/color interop: " + backends.npp.detail;
  }
  return std::nullopt;
}

reco::calibrate::CalibrationBackendStatus probe_cuda_npp_backends() {
  reco::calibrate::CalibrationBackendStatus status;
  status.cuda.available = reco::core::CudaBackend::is_available();
  status.cuda.detail = status.cuda.available ? "CUDA driver/runtime available"
                                             : reco::core::CudaBackend::availability_error();
  status.npp.available = reco::detect::is_npp_available();
  status.npp.detail = status.npp.available ? "NPP image primitives available"
                                           : reco::detect::npp_availability_error();
  return status;
}

std::optional<std::string> add_gpu_decode_pipeline_step(RuntimePlan& plan, std::string_view label,
                                                        const std::string& path) {
  const reco::io::GpuFileDecodeConfig config{
      .path = path,
      .codec = reco::io::gpu_decode_codec_for_path(path),
      .elementary_stream = reco::io::gpu_decode_path_is_elementary_stream(path),
      .container = reco::io::gpu_decode_container_for_path(path)};
  if (const auto error = reco::io::validate_gpu_file_decode_config(config); error.has_value()) {
    return std::string(label) + " " + *error;
  }
  plan.steps.push_back(std::string(label) + " GPU decode: " +
                       reco::io::build_gstreamer_gpu_file_decode_pipeline(config));
  return std::nullopt;
}

std::optional<std::string> validate_calibration_file_for_plan(const std::string& path) {
  std::string error;
  if (!reco::core::load_match_calibration_file(path, &error).has_value()) {
    return error.empty() ? "invalid calibration JSON" : error;
  }
  return std::nullopt;
}

RuntimePlan build_stitch_plan(const StitchCommand& command,
                              const reco::calibrate::CalibrationBackendStatus& backends) {
  RuntimePlan plan{.command = "stitch",
                   .steps = {
                       "load and validate v1 calibration JSON",
                   }};
  if (const auto error = validate_calibration_file_for_plan(command.calibration);
      error.has_value()) {
    plan.blocked_reason = *error;
    return plan;
  }
  if (const auto error = add_gpu_decode_pipeline_step(plan, "left", command.left);
      error.has_value()) {
    plan.blocked_reason = *error;
    return plan;
  }
  if (const auto error = add_gpu_decode_pipeline_step(plan, "right", command.right);
      error.has_value()) {
    plan.blocked_reason = *error;
    return plan;
  }
  plan.steps.push_back("run stitch renderer without CPU frame readback");
  plan.steps.push_back("encode the stitched output through the GPU-capable video backend");
  if (command.no_zero_copy) {
    plan.blocked_reason =
        "C++ stitch does not support --no-zero-copy because it would force a CPU path";
  } else if (auto error = require_gpu_video_backend(backends, false); error.has_value()) {
    plan.blocked_reason = *error;
  } else {
    plan.blocked_reason =
        "C++ GPU stitch renderer and encode execution are not ported yet; refusing CPU fallback";
  }
  return plan;
}

RuntimePlan build_preview_plan(const PreviewCommand& command,
                               const reco::calibrate::CalibrationBackendStatus& backends) {
  RuntimePlan plan{.command = "preview",
                   .steps = {
                       "load and validate v1 calibration JSON",
                   }};
  if (const auto error = validate_calibration_file_for_plan(command.calibration);
      error.has_value()) {
    plan.blocked_reason = *error;
    return plan;
  }
  if (const auto error = add_gpu_decode_pipeline_step(plan, "left", command.left);
      error.has_value()) {
    plan.blocked_reason = *error;
    return plan;
  }
  if (const auto error = add_gpu_decode_pipeline_step(plan, "right", command.right);
      error.has_value()) {
    plan.blocked_reason = *error;
    return plan;
  }
  plan.steps.push_back("render stitched preview frames directly to the GPU presentation surface");
  if (auto error = require_gpu_video_backend(backends, false); error.has_value()) {
    plan.blocked_reason = *error;
  } else {
    plan.blocked_reason = "C++ GPU preview presentation is not ported yet; refusing CPU fallback";
  }
  return plan;
}

RuntimePlan build_camera_plan(const CameraCommand& command,
                              const reco::calibrate::CalibrationBackendStatus& backends) {
  RuntimePlan plan{.command = "camera"};
  if (command.v4l2_direct) {
    plan.steps = {
        "open paired V4L2 devices through the direct capture backend",
        "demosaic raw sensor frames on the GPU",
        "optionally run live GPU calibration on sampled frames",
        "stitch and encode the live output without CPU frame readback",
    };
    if (const auto error = validate_calibration_file_for_plan(command.calibration);
        error.has_value()) {
      plan.blocked_reason = *error;
    } else if (const auto error = reco::io::validate_capture_device(
                   command.left_device, reco::io::CapturePlatform::LinuxV4l2);
               error.has_value()) {
      plan.blocked_reason = "left V4L2 device is invalid: " + *error;
    } else if (const auto error = reco::io::validate_capture_device(
                   command.right_device, reco::io::CapturePlatform::LinuxV4l2);
               error.has_value()) {
      plan.blocked_reason = "right V4L2 device is invalid: " + *error;
    } else if (auto error = require_cuda_npp_backend(backends); error.has_value()) {
      plan.blocked_reason = *error;
    } else {
      plan.blocked_reason =
          "C++ V4L2-direct GPU capture/demosaic execution is not ported yet; refusing CPU fallback";
    }
    return plan;
  }

  plan.steps = {
      "open paired live capture sources as GPU/NVMM surfaces",
      "optionally run live GPU calibration on sampled frames",
      "stitch and encode the live output without CPU frame readback",
  };
  const auto platform = reco::io::detect_capture_platform();
  if (const auto error = validate_calibration_file_for_plan(command.calibration);
      error.has_value()) {
    plan.blocked_reason = *error;
  } else if (const auto error = reco::io::validate_capture_device(command.left_device, platform);
             error.has_value()) {
    plan.blocked_reason = "left camera device is invalid for this platform: " + *error;
  } else if (const auto error = reco::io::validate_capture_device(command.right_device, platform);
             error.has_value()) {
    plan.blocked_reason = "right camera device is invalid for this platform: " + *error;
  } else if (platform != reco::io::CapturePlatform::Jetson) {
    plan.blocked_reason =
        "C++ generic GStreamer camera ingest is CPU-resident; use a GPU/NVMM path or --v4l2-direct";
  } else if (auto error = require_gpu_video_backend(backends, true); error.has_value()) {
    plan.blocked_reason = *error;
  } else {
    plan.blocked_reason =
        "C++ live camera ingest/stitch/encode execution is not ported yet; refusing CPU fallback";
  }
  return plan;
}

RuntimePlan build_libcamera_plan(const LibcameraCommand& command) {
  RuntimePlan plan{
      .command = "libcamera",
      .steps =
          {
              "open paired libcamera sensors through rpicam-vid",
              "bridge captured frames into a GPU-resident stitch input",
              "stitch and encode the live output without CPU frame readback",
          },
  };
  if (const auto error = validate_calibration_file_for_plan(command.calibration);
      error.has_value()) {
    plan.blocked_reason = *error;
  } else {
    plan.blocked_reason =
        "C++ libcamera capture is not GPU-resident yet; refusing the CPU YUV420P path";
  }
  return plan;
}

RuntimePlan build_gopro_plan(const GoproCommand&) {
  return {
      .command = "gopro",
      .steps =
          {
              "discover or address a GoPro control endpoint",
              "send start/stop/preset commands with explicit HTTP error handling",
          },
      .blocked_reason = "C++ GoPro control execution is not ported yet",
  };
}

template <typename T, typename Parser>
bool assign_next_or_error(Cursor& cursor, std::string_view option, Parser parser, T& out,
                          std::optional<ParseError>& err, bool allow_hyphen_value = false) {
  auto value = take_value_or_error(cursor, option, err, allow_hyphen_value);
  if (!value.has_value()) {
    return false;
  }
  return assign_or_error(parser(*value), out, err);
}

template <typename T>
bool assign_optional_or_error(std::variant<T, ParseError>&& parsed, std::optional<T>& out,
                              std::optional<ParseError>& err) {
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    err = *error;
    return false;
  }
  out = std::get<T>(std::move(parsed));
  return true;
}

template <typename T, typename Parser>
bool assign_next_optional_or_error(Cursor& cursor, std::string_view option, Parser parser,
                                   std::optional<T>& out, std::optional<ParseError>& err,
                                   bool allow_hyphen_value = false) {
  auto value = take_value_or_error(cursor, option, err, allow_hyphen_value);
  if (!value.has_value()) {
    return false;
  }
  return assign_optional_or_error(parser(*value), out, err);
}

std::variant<StitchCommand, ParseError> parse_stitch(Cursor& cursor) {
  StitchCommand command;
  std::vector<std::string> positionals;
  std::optional<ParseError> err;

  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--start-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.start_time, err))
        return *err;
    } else if (arg == "--end-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.end_time, err))
        return *err;
    } else if (arg == "--max-frames") {
      if (!assign_next_optional_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.max_frames, err))
        return *err;
    } else if (arg == "--encoder") {
      if (!assign_optional_or_error(next(arg), command.encoder, err))
        return *err;
    } else if (arg == "--codec") {
      if (!assign_or_error(next(arg), command.codec, err))
        return *err;
    } else if (arg == "--quality") {
      if (!assign_or_error(next(arg), command.quality, err))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(cursor, arg, parse_blend, command.blend, err)) {
        return *err;
      }
    } else if (arg == "--sync-offset") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::int64_t>(v, arg); },
              command.sync_offset, err, true))
        return *err;
    } else if (arg == "--model") {
      if (!assign_optional_or_error(next(arg), command.model, err))
        return *err;
    } else if (arg == "--detection-interval") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.detection_interval, err))
        return *err;
    } else if (arg == "--lookahead") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.lookahead, err))
        return *err;
    } else if (arg == "--tracking") {
      if (!assign_or_error(next(arg), command.tracking, err))
        return *err;
    } else if (arg == "--quality-value") {
      std::uint32_t value = 0;
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); }, value,
              err))
        return *err;
      if (value > std::numeric_limits<std::uint8_t>::max()) {
        return ParseError{"invalid --quality-value " + std::to_string(value)};
      }
      command.quality_value = static_cast<std::uint8_t>(value);
    } else if (arg == "--preset") {
      if (!assign_optional_or_error(next(arg), command.preset, err))
        return *err;
    } else if (arg == "--container") {
      if (!assign_optional_or_error(next(arg), command.container, err))
        return *err;
    } else if (arg == "--replay") {
      if (!assign_optional_or_error(next(arg), command.replay, err))
        return *err;
    } else if (arg == "--replay-scale") {
      if (!assign_next_optional_or_error(cursor, arg, parse_wxh, command.replay_scale, err)) {
        return *err;
      }
    } else if (arg == "--allow-no-tracking") {
      command.allow_no_tracking = true;
    } else if (arg == "--no-zero-copy") {
      command.no_zero_copy = true;
    } else if (arg == "--events") {
      if (!assign_optional_or_error(next(arg), command.events, err))
        return *err;
    } else if (arg == "--trajectory") {
      if (!assign_optional_or_error(next(arg), command.trajectory, err))
        return *err;
    } else if (arg == "--panner-config") {
      if (!assign_optional_or_error(next(arg), command.panner_config, err))
        return *err;
    } else if (arg == "--panner-preset") {
      if (!assign_optional_or_error(next(arg), command.panner_preset, err))
        return *err;
    } else if (!arg.empty() && arg.front() == '-') {
      return ParseError{"unknown stitch option " + arg};
    } else {
      positionals.push_back(arg);
    }
  }

  if (positionals.size() != 2) {
    return ParseError{"stitch requires LEFT and RIGHT inputs"};
  }
  if (command.calibration.empty()) {
    return ParseError{"stitch requires -c/--calibration"};
  }
  command.left = positionals[0];
  command.right = positionals[1];
  return command;
}

std::variant<PreviewCommand, ParseError> parse_preview(Cursor& cursor) {
  PreviewCommand command;
  std::vector<std::string> positionals;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--sync-offset") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::int64_t>(v, arg); },
              command.sync_offset, err, true))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_float(v, arg); }, command.blend,
              err))
        return *err;
    } else if (arg == "--rig-tilt") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_float(v, arg); },
              command.rig_tilt, err, true))
        return *err;
    } else if (!arg.empty() && arg.front() == '-') {
      return ParseError{"unknown preview option " + arg};
    } else {
      positionals.push_back(arg);
    }
  }
  if (positionals.size() != 2) {
    return ParseError{"preview requires LEFT and RIGHT inputs"};
  }
  if (command.calibration.empty()) {
    return ParseError{"preview requires -c/--calibration"};
  }
  command.left = positionals[0];
  command.right = positionals[1];
  return command;
}

std::variant<CameraCommand, ParseError> parse_camera(Cursor& cursor) {
  CameraCommand command;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--left-device") {
      if (!assign_or_error(next(arg), command.left_device, err))
        return *err;
    } else if (arg == "--right-device") {
      if (!assign_or_error(next(arg), command.right_device, err))
        return *err;
    } else if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (arg == "--capture-width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_width, err))
        return *err;
    } else if (arg == "--capture-height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_height, err))
        return *err;
    } else if (arg == "--capture-fps") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_fps, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--encoder") {
      if (!assign_optional_or_error(next(arg), command.encoder, err))
        return *err;
    } else if (arg == "--codec") {
      if (!assign_or_error(next(arg), command.codec, err))
        return *err;
    } else if (arg == "--quality") {
      if (!assign_or_error(next(arg), command.quality, err))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(cursor, arg, parse_blend, command.blend, err)) {
        return *err;
      }
    } else if (arg == "--max-frames") {
      if (!assign_next_optional_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.max_frames, err))
        return *err;
    } else if (arg == "--end-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.end_time, err))
        return *err;
    } else if (arg == "--model") {
      if (!assign_optional_or_error(next(arg), command.model, err))
        return *err;
    } else if (arg == "--detection-interval") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.detection_interval, err))
        return *err;
    } else if (arg == "--quality-value") {
      std::uint32_t value = 0;
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); }, value,
              err))
        return *err;
      if (value > std::numeric_limits<std::uint8_t>::max()) {
        return ParseError{"invalid --quality-value " + std::to_string(value)};
      }
      command.quality_value = static_cast<std::uint8_t>(value);
    } else if (arg == "--preset") {
      if (!assign_optional_or_error(next(arg), command.preset, err))
        return *err;
    } else if (arg == "--container") {
      if (!assign_optional_or_error(next(arg), command.container, err))
        return *err;
    } else if (arg == "--stream-url") {
      if (!assign_optional_or_error(next(arg), command.stream_url, err))
        return *err;
    } else if (arg == "--tracking") {
      if (!assign_or_error(next(arg), command.tracking, err))
        return *err;
    } else if (arg == "--unconstrained") {
      command.unconstrained = true;
    } else if (arg == "--replay") {
      if (!assign_optional_or_error(next(arg), command.replay, err))
        return *err;
    } else if (arg == "--replay-scale") {
      if (!assign_next_optional_or_error(cursor, arg, parse_wxh, command.replay_scale, err)) {
        return *err;
      }
    } else if (arg == "--v4l2-direct") {
      command.v4l2_direct = true;
    } else if (arg == "--exposure") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.exposure, err))
        return *err;
    } else if (arg == "--sensor-gain") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.sensor_gain, err))
        return *err;
    } else if (arg == "--live-calibrate") {
      command.live_calibrate = true;
    } else if (arg == "--calibrate-frames") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::size_t>(v, arg); },
              command.calibrate_frames, err))
        return *err;
    } else if (arg == "--left-lens-profile") {
      if (!assign_optional_or_error(next(arg), command.left_lens_profile, err))
        return *err;
    } else {
      return ParseError{"unknown camera option " + arg};
    }
  }
  if (command.left_device.empty()) {
    return ParseError{"camera requires --left-device"};
  }
  if (command.right_device.empty()) {
    return ParseError{"camera requires --right-device"};
  }
  if (command.calibration.empty()) {
    return ParseError{"camera requires -c/--calibration"};
  }
  if (command.output.empty()) {
    return ParseError{"camera requires -o/--output"};
  }
  return command;
}

std::variant<LibcameraCommand, ParseError> parse_libcamera(Cursor& cursor) {
  LibcameraCommand command;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--left-camera") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.left_camera, err))
        return *err;
    } else if (arg == "--right-camera") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.right_camera, err))
        return *err;
    } else if (arg == "-c" || arg == "--calibration") {
      if (!assign_or_error(next(arg), command.calibration, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (arg == "--capture-width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_width, err))
        return *err;
    } else if (arg == "--capture-height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_height, err))
        return *err;
    } else if (arg == "--capture-fps") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.capture_fps, err))
        return *err;
    } else if (arg == "--width") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.width, err))
        return *err;
    } else if (arg == "--height") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); },
              command.height, err))
        return *err;
    } else if (arg == "--encoder") {
      if (!assign_optional_or_error(next(arg), command.encoder, err))
        return *err;
    } else if (arg == "--codec") {
      if (!assign_or_error(next(arg), command.codec, err))
        return *err;
    } else if (arg == "--quality") {
      if (!assign_or_error(next(arg), command.quality, err))
        return *err;
    } else if (arg == "--blend") {
      if (!assign_next_or_error(cursor, arg, parse_blend, command.blend, err)) {
        return *err;
      }
    } else if (arg == "--max-frames") {
      if (!assign_next_optional_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.max_frames, err))
        return *err;
    } else if (arg == "--end-time") {
      if (!assign_next_optional_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.end_time, err))
        return *err;
    } else if (arg == "--model") {
      if (!assign_optional_or_error(next(arg), command.model, err))
        return *err;
    } else if (arg == "--detection-interval") {
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint64_t>(v, arg); },
              command.detection_interval, err))
        return *err;
    } else if (arg == "--quality-value") {
      std::uint32_t value = 0;
      if (!assign_next_or_error(
              cursor, arg,
              [&](std::string_view v) { return parse_integral<std::uint32_t>(v, arg); }, value,
              err))
        return *err;
      if (value > std::numeric_limits<std::uint8_t>::max()) {
        return ParseError{"invalid --quality-value " + std::to_string(value)};
      }
      command.quality_value = static_cast<std::uint8_t>(value);
    } else if (arg == "--preset") {
      if (!assign_optional_or_error(next(arg), command.preset, err))
        return *err;
    } else {
      return ParseError{"unknown libcamera option " + arg};
    }
  }
  if (command.calibration.empty()) {
    return ParseError{"libcamera requires -c/--calibration"};
  }
  if (command.output.empty()) {
    return ParseError{"libcamera requires -o/--output"};
  }
  return command;
}

std::variant<GoproCommand, ParseError> parse_gopro(Cursor& cursor) {
  GoproCommand command;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--serial") {
      if (!assign_optional_or_error(next(arg), command.serial, err))
        return *err;
    } else if (arg == "--url") {
      if (!assign_optional_or_error(next(arg), command.url, err))
        return *err;
    } else if (arg == "--start") {
      command.start = true;
    } else if (arg == "--stop") {
      command.stop = true;
    } else if (arg == "--sports-preset") {
      command.sports_preset = true;
    } else {
      return ParseError{"unknown gopro option " + arg};
    }
  }
  return command;
}

std::variant<CalibrateCommand, ParseError> parse_calibrate(Cursor& cursor) {
  CalibrateCommand command;
  std::vector<std::string> positionals;
  std::optional<ParseError> err;
  while (!cursor.empty()) {
    const std::string arg = cursor.take();
    auto next = [&](std::string_view option) { return cursor.value(option); };
    if (arg == "--left-profile") {
      if (!assign_optional_or_error(next(arg), command.left_profile, err))
        return *err;
    } else if (arg == "--right-profile") {
      if (!assign_optional_or_error(next(arg), command.right_profile, err))
        return *err;
    } else if (arg == "--frames") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::size_t>(v, arg); },
              command.frames, err))
        return *err;
    } else if (arg == "--no-auto-imu") {
      command.no_auto_imu = true;
    } else if (arg == "--no-auto-sync") {
      command.auto_sync = false;
    } else if (arg == "--auto-sync") {
      std::string value;
      if (!assign_or_error(next(arg), value, err))
        return *err;
      if (value == "true") {
        command.auto_sync = true;
      } else if (value == "false") {
        command.auto_sync = false;
      } else {
        return ParseError{"--auto-sync expects true or false"};
      }
    } else if (arg == "--sync-offset") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_integral<std::int64_t>(v, arg); },
              command.sync_offset, err, true))
        return *err;
    } else if (arg == "--skip-start") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.skip_start, err))
        return *err;
    } else if (arg == "--skip-end") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.skip_end, err))
        return *err;
    } else if (arg == "--akaze-threshold") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.akaze_threshold, err))
        return *err;
    } else if (arg == "--lowe-ratio") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.lowe_ratio, err))
        return *err;
    } else if (arg == "--detect-x") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.detect_x, err))
        return *err;
    } else if (arg == "--detect-y-min") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.detect_y_min, err))
        return *err;
    } else if (arg == "--detect-y-max") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.detect_y_max, err))
        return *err;
    } else if (arg == "--lock-cam-d") {
      command.lock_cam_d = true;
    } else if (arg == "--lock-z-rx") {
      command.lock_z_rx = true;
    } else if (arg == "--trim") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); }, command.trim,
              err)) {
        return *err;
      }
    } else if (arg == "--seam-sigma") {
      if (!assign_next_or_error(
              cursor, arg, [&](std::string_view v) { return parse_double(v, arg); },
              command.seam_sigma, err))
        return *err;
    } else if (arg == "--debug-dir") {
      if (!assign_optional_or_error(next(arg), command.debug_dir, err))
        return *err;
    } else if (arg == "-o" || arg == "--output") {
      if (!assign_or_error(next(arg), command.output, err))
        return *err;
    } else if (!arg.empty() && arg.front() == '-') {
      return ParseError{"unknown calibrate option " + arg};
    } else {
      positionals.push_back(arg);
    }
  }
  if (positionals.size() != 2) {
    return ParseError{"calibrate requires LEFT and RIGHT inputs"};
  }
  command.left = positionals[0];
  command.right = positionals[1];
  return command;
}

} // namespace

namespace detail {

std::optional<std::filesystem::path>
resolve_video_probe_worker(const std::filesystem::path& executable_path) {
  return resolve_video_probe_worker_impl(executable_path);
}

std::optional<std::filesystem::path>
resolve_calibration_worker(const std::filesystem::path& executable_path) {
  return resolve_calibration_worker_impl(executable_path);
}

void write_calibration_json_atomically(std::string_view json,
                                       const std::filesystem::path& destination,
                                       const std::filesystem::path& left_input,
                                       const std::filesystem::path& right_input,
                                       const std::function<void()>& before_publish) {
  write_calibration_json_atomically_impl(json, destination, left_input, right_input,
                                         before_publish);
}

} // namespace detail

std::variant<float, ParseError> parse_blend(std::string_view value) {
  auto parsed = parse_float(value, "blend");
  if (const auto* error = std::get_if<ParseError>(&parsed)) {
    return *error;
  }
  const float blend = std::get<float>(parsed);
  if (!std::isfinite(blend) || blend < 0.0F || blend > 1.0F) {
    std::ostringstream out;
    out << blend << " is not in 0.0..=1.0";
    return ParseError{out.str()};
  }
  return blend;
}

std::variant<WxH, ParseError> parse_wxh(std::string_view value) {
  const auto split = value.find_first_of("xX");
  if (split == std::string_view::npos) {
    return ParseError{"expected WIDTHxHEIGHT, got " + std::string(value)};
  }
  auto width = parse_integral<std::uint32_t>(value.substr(0, split), "width");
  if (const auto* error = std::get_if<ParseError>(&width)) {
    return *error;
  }
  auto height = parse_integral<std::uint32_t>(value.substr(split + 1), "height");
  if (const auto* error = std::get_if<ParseError>(&height)) {
    return *error;
  }
  const WxH parsed{.width = std::get<std::uint32_t>(width),
                   .height = std::get<std::uint32_t>(height)};
  if (parsed.width == 0 || parsed.height == 0) {
    return ParseError{"dimensions must be > 0, got " + std::to_string(parsed.width) + "x" +
                      std::to_string(parsed.height)};
  }
  if (parsed.width % 4 != 0) {
    return ParseError{"width must be divisible by 4 (pack shader packs 4 pixels per u32 write), "
                      "got " +
                      std::to_string(parsed.width)};
  }
  if (parsed.height % 2 != 0) {
    return ParseError{"height must be even (YUV420P chroma subsampling), got " +
                      std::to_string(parsed.height)};
  }
  return parsed;
}

std::variant<Command, ParseError> parse_args(const std::vector<std::string>& args) {
  if (args.empty() || args[0] == "--help" || args[0] == "-h") {
    return Command{HelpCommand{}};
  }
  Cursor cursor(args);
  const std::string subcommand = cursor.take();
  if (subcommand == "stitch" || subcommand == "preview" || subcommand == "camera" ||
      subcommand == "libcamera" || subcommand == "calibrate" || subcommand == "gopro" ||
      subcommand == "info") {
    if (std::find(args.begin() + 1, args.end(), "--help") != args.end() ||
        std::find(args.begin() + 1, args.end(), "-h") != args.end()) {
      return Command{HelpCommand{}};
    }
  }
  if (subcommand == "stitch") {
    auto parsed = parse_stitch(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<StitchCommand>(std::move(parsed))};
  }
  if (subcommand == "preview") {
    auto parsed = parse_preview(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<PreviewCommand>(std::move(parsed))};
  }
  if (subcommand == "camera") {
    auto parsed = parse_camera(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<CameraCommand>(std::move(parsed))};
  }
  if (subcommand == "libcamera") {
    auto parsed = parse_libcamera(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<LibcameraCommand>(std::move(parsed))};
  }
  if (subcommand == "calibrate") {
    auto parsed = parse_calibrate(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<CalibrateCommand>(std::move(parsed))};
  }
  if (subcommand == "gopro") {
    auto parsed = parse_gopro(cursor);
    if (const auto* error = std::get_if<ParseError>(&parsed))
      return *error;
    return Command{std::get<GoproCommand>(std::move(parsed))};
  }
  if (subcommand == "info") {
    if (!cursor.empty()) {
      return ParseError{"info does not accept positional arguments or options"};
    }
    return Command{InfoCommand{}};
  }
  return ParseError{"unknown command " + subcommand};
}

std::string_view command_name(const Command& command) {
  return std::visit(
      [](const auto& value) -> std::string_view {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, StitchCommand>)
          return "stitch";
        if constexpr (std::is_same_v<T, PreviewCommand>)
          return "preview";
        if constexpr (std::is_same_v<T, CalibrateCommand>)
          return "calibrate";
        if constexpr (std::is_same_v<T, CameraCommand>)
          return "camera";
        if constexpr (std::is_same_v<T, LibcameraCommand>)
          return "libcamera";
        if constexpr (std::is_same_v<T, GoproCommand>)
          return "gopro";
        if constexpr (std::is_same_v<T, InfoCommand>)
          return "info";
        return "help";
      },
      command);
}

std::string help_text() {
  return "Reco C++ CLI port\n\n"
         "Usage:\n"
         "  reco stitch LEFT RIGHT -c CALIBRATION [options]\n"
         "  reco preview LEFT RIGHT -c CALIBRATION [options]\n"
         "  reco camera --left-device DEV --right-device DEV -c CALIBRATION -o OUTPUT [options]\n"
         "  reco libcamera -c CALIBRATION -o OUTPUT [options]\n"
         "  reco calibrate LEFT RIGHT --left-profile PROFILE --no-auto-sync --no-auto-imu "
         "[options]\n"
         "  reco gopro [options]\n"
         "  reco info\n\n"
         "Runtime command execution is staged behind the remaining GPU/backend ports.";
}

int run_command(const Command& command, std::ostream& out, std::ostream& err,
                const std::filesystem::path& executable_path) {
  if (std::holds_alternative<HelpCommand>(command)) {
    out << help_text() << '\n';
    return 0;
  }

  if (std::holds_alternative<InfoCommand>(command)) {
    out << "Reco C++ capability report\n";
    const bool cuda_available = reco::core::CudaBackend::is_available();
    if (cuda_available) {
      auto backend = reco::core::CudaBackend::create();
      const int devices = backend.device_count();
      out << "CUDA: available (" << devices << " device";
      if (devices != 1) {
        out << 's';
      }
      out << ")\n";
      for (int ordinal = 0; ordinal < devices; ++ordinal) {
        const auto device = backend.device_info(ordinal);
        out << "  cuda[" << ordinal << "]: " << device.name << '\n';
      }
    } else {
      out << "CUDA: unavailable (" << reco::core::CudaBackend::availability_error() << ")\n";
    }

    write_probe(out, "NPP", reco::detect::is_npp_available(),
                reco::detect::npp_availability_error());

    const auto gst = reco::io::probe_gstreamer_runtime();
    write_probe(out, "GStreamer", gst.available, gst.available ? gst.library : gst.error);
    const auto deepstream = reco::io::probe_deepstream_runtime();
    write_probe(out, "DeepStream", deepstream.available,
                deepstream.available ? deepstream.library : deepstream.error);
    const auto nvbuf = reco::io::probe_nvbufsurface_runtime();
    write_probe(out, "NvBufSurface", nvbuf.available,
                nvbuf.available ? nvbuf.library : nvbuf.error);

    const auto ort = reco::detect::probe_ort_runtime();
    write_probe(out, "ONNX Runtime", ort.available, ort.available ? ort.version : ort.error);
    const auto coreml = reco::detect::probe_coreml_runtime();
    write_probe(out, "CoreML", coreml.available, coreml.available ? coreml.provider : coreml.error);
    const auto ai = reco::detect::probe_execution_providers();
    out << "AI providers:";
    if (ai.providers.empty()) {
      out << " none";
    } else {
      for (const auto& provider : ai.providers) {
        out << ' ' << provider;
      }
    }
    out << '\n';
    out << "GPU frame inference: " << (ai.can_run_on_gpu_frames ? "available" : "unavailable")
        << '\n';
    return 0;
  }

  if (const auto* calibrate = std::get_if<CalibrateCommand>(&command)) {
    reco::calibrate::GpuCalibrationRequest request;
    request.left.path = calibrate->left;
    request.left.lens_profile = calibrate->left_profile;
    request.right.path = calibrate->right;
    request.right.lens_profile = calibrate->right_profile;
    request.config.num_frames = calibrate->frames;
    request.config.skip_start_secs = calibrate->skip_start;
    request.config.skip_end_secs = calibrate->skip_end;
    request.config.akaze.threshold = calibrate->akaze_threshold;
    request.config.akaze.detect_y_min = calibrate->detect_y_min;
    request.config.akaze.detect_y_max = calibrate->detect_y_max;
    request.config.matching.lowe_ratio = calibrate->lowe_ratio;
    request.config.matching.spatial_x_threshold = calibrate->detect_x;
    request.config.optimizer.lock_cam_d = calibrate->lock_cam_d;
    request.config.optimizer.lock_z_rx = calibrate->lock_z_rx;
    request.config.optimizer.trim_fraction = calibrate->trim;
    request.config.optimizer.seam_sigma = calibrate->seam_sigma;
    request.no_auto_imu = calibrate->no_auto_imu;
    request.auto_sync = calibrate->auto_sync;
    request.manual_sync_offset = calibrate->sync_offset;
    request.debug_dir = calibrate->debug_dir;
    request.output = calibrate->output;
    if (const auto probe_worker = detail::resolve_video_probe_worker(executable_path);
        probe_worker.has_value()) {
      request.probe_worker = probe_worker->string();
    }
    if (const auto calibration_worker = detail::resolve_calibration_worker(executable_path);
        calibration_worker.has_value()) {
      request.calibration_worker_path = calibration_worker->string();
    }

    try {
      request.nvbufsurface_abi = reco::io::discover_nvbufsurface_abi();
    } catch (const std::exception& error) {
      err << "error: cannot discover the installed NvBufSurface ABI: " << error.what() << '\n';
      return 2;
    }

    const auto backends = reco::calibrate::probe_calibration_backends();
    const auto plan = reco::calibrate::build_gpu_calibration_plan(request, backends);
    out << reco::calibrate::describe_calibration_plan(plan);
    if (!plan.ready) {
      err << "error: " << plan.blocked_reason.value_or("C++ GPU calibration is unavailable")
          << '\n';
      return 2;
    }

    try {
#if defined(__linux__)
      const auto left_identity = pin_input_identity(request.left.path, "left");
      const auto right_identity = pin_input_identity(request.right.path, "right");
      auto pinned_request = request;
      pinned_request.left.retained_path = left_identity.retained_path().string();
      pinned_request.right.retained_path = right_identity.retained_path().string();
      const auto result = reco::calibrate::run_gpu_calibration(pinned_request, backends);
      write_calibration_result(result, pinned_request);
#else
      const auto result = reco::calibrate::run_gpu_calibration(request, backends);
      write_calibration_result(result, request);
#endif
      write_calibration_result_summary(result, request.output, out);
    } catch (const std::exception& error) {
      err << "error: " << error.what() << '\n';
      return 2;
    }
    return 0;
  }

  RuntimePlan runtime_plan;
  if (const auto* stitch = std::get_if<StitchCommand>(&command)) {
    const auto backends = reco::calibrate::probe_calibration_backends();
    runtime_plan = build_stitch_plan(*stitch, backends);
  } else if (const auto* preview = std::get_if<PreviewCommand>(&command)) {
    const auto backends = reco::calibrate::probe_calibration_backends();
    runtime_plan = build_preview_plan(*preview, backends);
  } else if (const auto* camera = std::get_if<CameraCommand>(&command)) {
    const auto backends = camera->v4l2_direct ? probe_cuda_npp_backends()
                                              : reco::calibrate::probe_calibration_backends();
    runtime_plan = build_camera_plan(*camera, backends);
  } else if (const auto* libcamera = std::get_if<LibcameraCommand>(&command)) {
    runtime_plan = build_libcamera_plan(*libcamera);
  } else if (const auto* gopro = std::get_if<GoproCommand>(&command)) {
    runtime_plan = build_gopro_plan(*gopro);
  }

  if (!runtime_plan.command.empty()) {
    write_runtime_plan(out, runtime_plan);
    err << "error: " << runtime_plan.blocked_reason.value_or("C++ runtime execution is unavailable")
        << '\n';
    return 2;
  }

  err << "error: C++ reco " << command_name(command)
      << " execution is not ported yet; GPU/runtime backend stages remain authoritative in Rust.\n";
  return 2;
}

} // namespace reco::cli
