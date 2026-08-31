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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <stdio.h>
#endif
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
class UniqueWindowsHandle {
public:
  UniqueWindowsHandle() = default;
  explicit UniqueWindowsHandle(HANDLE handle) : handle_(handle) {}
  ~UniqueWindowsHandle() {
    if (handle_ != INVALID_HANDLE_VALUE) {
      (void)CloseHandle(handle_);
    }
  }

  UniqueWindowsHandle(const UniqueWindowsHandle&) = delete;
  UniqueWindowsHandle& operator=(const UniqueWindowsHandle&) = delete;
  UniqueWindowsHandle(UniqueWindowsHandle&& other) noexcept
      : handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)) {}
  UniqueWindowsHandle& operator=(UniqueWindowsHandle&& other) noexcept {
    if (this != &other) {
      if (handle_ != INVALID_HANDLE_VALUE) {
        (void)CloseHandle(handle_);
      }
      handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
    }
    return *this;
  }

  [[nodiscard]] HANDLE get() const noexcept { return handle_; }
  [[nodiscard]] HANDLE release() noexcept { return std::exchange(handle_, INVALID_HANDLE_VALUE); }

private:
  HANDLE handle_ = INVALID_HANDLE_VALUE;
};

struct PinnedWindowsPath {
  UniqueWindowsHandle target;
  UniqueWindowsHandle directory_entry;

  void verify_unchanged(const std::filesystem::path& path, std::string_view label) const;
};

struct PinnedWindowsDirectory {
  std::filesystem::path path;
  UniqueWindowsHandle handle;
};

using NtCreateFileFunction = NTSTATUS(NTAPI*)(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES,
                                              PIO_STATUS_BLOCK, PLARGE_INTEGER, ULONG, ULONG, ULONG,
                                              ULONG, PVOID, ULONG);
using NtSetInformationFileFunction = NTSTATUS(NTAPI*)(HANDLE, PIO_STATUS_BLOCK, PVOID, ULONG,
                                                      ULONG);
using RtlNtStatusToDosErrorFunction = ULONG(WINAPI*)(NTSTATUS);

struct NativeWindowsFunctions {
  NtCreateFileFunction create_file = nullptr;
  NtSetInformationFileFunction set_information_file = nullptr;
  RtlNtStatusToDosErrorFunction status_to_error = nullptr;
};

[[nodiscard]] const NativeWindowsFunctions& native_windows_functions() {
  static const NativeWindowsFunctions functions = [] {
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll == nullptr) {
      throw std::runtime_error("cannot load Windows native file APIs");
    }
    NativeWindowsFunctions loaded{
        .create_file =
            reinterpret_cast<NtCreateFileFunction>(GetProcAddress(ntdll, "NtCreateFile")),
        .set_information_file = reinterpret_cast<NtSetInformationFileFunction>(
            GetProcAddress(ntdll, "NtSetInformationFile")),
        .status_to_error = reinterpret_cast<RtlNtStatusToDosErrorFunction>(
            GetProcAddress(ntdll, "RtlNtStatusToDosError")),
    };
    if (loaded.create_file == nullptr || loaded.set_information_file == nullptr ||
        loaded.status_to_error == nullptr) {
      throw std::runtime_error("Windows native relative file APIs are unavailable");
    }
    return loaded;
  }();
  return functions;
}

[[nodiscard]] HANDLE open_windows_file_relative(HANDLE directory, std::wstring_view name,
                                                ACCESS_MASK access, ULONG share_access,
                                                ULONG disposition, ULONG options,
                                                DWORD& windows_error) {
  if (name.size() > std::numeric_limits<USHORT>::max() / sizeof(wchar_t)) {
    windows_error = ERROR_FILENAME_EXCED_RANGE;
    return INVALID_HANDLE_VALUE;
  }
  UNICODE_STRING unicode_name{
      .Length = static_cast<USHORT>(name.size() * sizeof(wchar_t)),
      .MaximumLength = static_cast<USHORT>(name.size() * sizeof(wchar_t)),
      .Buffer = const_cast<PWSTR>(name.data()),
  };
  OBJECT_ATTRIBUTES attributes{};
  InitializeObjectAttributes(&attributes, &unicode_name, OBJ_CASE_INSENSITIVE, directory, nullptr);
  IO_STATUS_BLOCK io_status{};
  HANDLE handle = INVALID_HANDLE_VALUE;
  const auto& functions = native_windows_functions();
  const NTSTATUS status =
      functions.create_file(&handle, access, &attributes, &io_status, nullptr,
                            FILE_ATTRIBUTE_NORMAL, share_access, disposition, options, nullptr, 0);
  if (status < 0) {
    windows_error = functions.status_to_error(status);
    return INVALID_HANDLE_VALUE;
  }
  windows_error = ERROR_SUCCESS;
  return handle;
}

[[nodiscard]] DWORD set_windows_file_information(HANDLE handle, void* information,
                                                 ULONG information_size, ULONG information_class) {
  IO_STATUS_BLOCK io_status{};
  const auto& functions = native_windows_functions();
  NTSTATUS status = functions.set_information_file(handle, &io_status, information,
                                                   information_size, information_class);
  constexpr NTSTATUS kStatusPending = 0x00000103L;
  if (status == kStatusPending) {
    if (WaitForSingleObject(handle, INFINITE) != WAIT_OBJECT_0) {
      return GetLastError();
    }
    status = io_status.Status;
  }
  return status < 0 ? functions.status_to_error(status) : ERROR_SUCCESS;
}

class WindowsPublicationLock {
public:
  WindowsPublicationLock(HANDLE mutex, bool owned) : mutex_(mutex), owned_(owned) {}
  ~WindowsPublicationLock() {
    if (owned_) {
      (void)ReleaseMutex(mutex_.get());
    }
  }

  WindowsPublicationLock(const WindowsPublicationLock&) = delete;
  WindowsPublicationLock& operator=(const WindowsPublicationLock&) = delete;
  WindowsPublicationLock(WindowsPublicationLock&& other) noexcept
      : mutex_(std::move(other.mutex_)), owned_(std::exchange(other.owned_, false)) {}

private:
  UniqueWindowsHandle mutex_;
  bool owned_ = false;
};

[[nodiscard]] WindowsPublicationLock
lock_windows_output_directory(HANDLE directory, const std::filesystem::path& path,
                              const std::function<void()>& on_contention) {
  BY_HANDLE_FILE_INFORMATION identity{};
  if (GetFileInformationByHandle(directory, &identity) == 0) {
    throw_file_error("cannot inspect calibration output directory lock", path,
                     static_cast<int>(GetLastError()));
  }
  const auto name = L"Local\\RecoCalibrationPublication-v1-" +
                    std::to_wstring(identity.dwVolumeSerialNumber) + L"-" +
                    std::to_wstring(identity.nFileIndexHigh) + L"-" +
                    std::to_wstring(identity.nFileIndexLow);
  const HANDLE mutex = CreateMutexW(nullptr, FALSE, name.c_str());
  if (mutex == nullptr) {
    throw_file_error("cannot create calibration output directory lock", path,
                     static_cast<int>(GetLastError()));
  }
  UniqueWindowsHandle retained_mutex(mutex);
  DWORD wait_result = WaitForSingleObject(mutex, 0);
  if (wait_result == WAIT_TIMEOUT) {
    if (on_contention) {
      on_contention();
    }
    constexpr DWORD kPublicationLockTimeoutMs = 30000;
    wait_result = WaitForSingleObject(mutex, kPublicationLockTimeoutMs);
  }
  if (wait_result == WAIT_TIMEOUT) {
    throw_file_error("timed out locking calibration output directory", path, ERROR_TIMEOUT);
  }
  if (wait_result != WAIT_OBJECT_0 && wait_result != WAIT_ABANDONED) {
    throw_file_error("cannot lock calibration output directory", path,
                     static_cast<int>(GetLastError()));
  }
  return WindowsPublicationLock(retained_mutex.release(), true);
}

[[nodiscard]] PinnedWindowsDirectory
pin_windows_output_directory(const std::filesystem::path& destination) {
  const auto parent =
      destination.has_parent_path() ? destination.parent_path() : std::filesystem::path(".");
  const HANDLE directory =
      CreateFileW(parent.c_str(), FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                  FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (directory == INVALID_HANDLE_VALUE) {
    throw_file_error("cannot retain calibration output directory", parent,
                     static_cast<int>(GetLastError()));
  }
  return {.path = parent, .handle = UniqueWindowsHandle(directory)};
}

[[nodiscard]] bool path_identifies_windows_handle(const std::filesystem::path& path, HANDLE source,
                                                  bool open_reparse_point) {
  DWORD flags = FILE_FLAG_BACKUP_SEMANTICS;
  if (open_reparse_point) {
    flags |= FILE_FLAG_OPEN_REPARSE_POINT;
  }
  const HANDLE current = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                     nullptr, OPEN_EXISTING, flags, nullptr);
  if (current == INVALID_HANDLE_VALUE) {
    return false;
  }
  UniqueWindowsHandle retained_current(current);
  BY_HANDLE_FILE_INFORMATION source_identity{};
  BY_HANDLE_FILE_INFORMATION current_identity{};
  return GetFileInformationByHandle(source, &source_identity) != 0 &&
         GetFileInformationByHandle(current, &current_identity) != 0 &&
         source_identity.dwVolumeSerialNumber == current_identity.dwVolumeSerialNumber &&
         source_identity.nFileIndexHigh == current_identity.nFileIndexHigh &&
         source_identity.nFileIndexLow == current_identity.nFileIndexLow;
}

void PinnedWindowsPath::verify_unchanged(const std::filesystem::path& path,
                                         std::string_view label) const {
  if (!path_identifies_windows_handle(path, directory_entry.get(), true) ||
      !path_identifies_windows_handle(path, target.get(), false)) {
    throw std::runtime_error(std::string(label) + " path identity changed before publication");
  }
}

[[nodiscard]] PinnedWindowsPath pin_windows_path_for_publication(const std::filesystem::path& path,
                                                                 std::string_view label) {
  constexpr DWORD kReadOnlySharing = FILE_SHARE_READ;
  const HANDLE directory_entry =
      CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, kReadOnlySharing, nullptr, OPEN_EXISTING,
                  FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (directory_entry == INVALID_HANDLE_VALUE) {
    throw_file_error("cannot retain " + std::string(label) + " path", path,
                     static_cast<int>(GetLastError()));
  }
  PinnedWindowsPath pinned{.directory_entry = UniqueWindowsHandle(directory_entry)};
  const HANDLE target =
      CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, kReadOnlySharing, nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (target == INVALID_HANDLE_VALUE) {
    throw_file_error("cannot retain " + std::string(label), path, static_cast<int>(GetLastError()));
  }
  pinned.target = UniqueWindowsHandle(target);
  return pinned;
}

std::filesystem::path create_exclusive_temporary(const std::filesystem::path& destination,
                                                 HANDLE directory, HANDLE& handle) {
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
    DWORD error = ERROR_SUCCESS;
    handle = open_windows_file_relative(
        directory, filename.wstring(), GENERIC_WRITE | DELETE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        FILE_SHARE_READ | FILE_SHARE_DELETE, FILE_CREATE,
        FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT, error);
    if (handle != INVALID_HANDLE_VALUE) {
      return temporary;
    }
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

class WindowsPublicationIdentityError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] bool published_path_identifies_handle(HANDLE directory,
                                                    std::wstring_view destination_name,
                                                    HANDLE source) {
  DWORD error = ERROR_SUCCESS;
  const HANDLE destination_handle = open_windows_file_relative(
      directory, destination_name, FILE_READ_ATTRIBUTES | SYNCHRONIZE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_OPEN,
      FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT, error);
  if (destination_handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  UniqueWindowsHandle retained_destination(destination_handle);
  FILE_ATTRIBUTE_TAG_INFO attributes{};
  BY_HANDLE_FILE_INFORMATION source_identity{};
  BY_HANDLE_FILE_INFORMATION destination_identity{};
  return GetFileInformationByHandleEx(destination_handle, FileAttributeTagInfo, &attributes,
                                      sizeof(attributes)) != 0 &&
         (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0 &&
         GetFileInformationByHandle(source, &source_identity) != 0 &&
         GetFileInformationByHandle(destination_handle, &destination_identity) != 0 &&
         source_identity.dwVolumeSerialNumber == destination_identity.dwVolumeSerialNumber &&
         source_identity.nFileIndexHigh == destination_identity.nFileIndexHigh &&
         source_identity.nFileIndexLow == destination_identity.nFileIndexLow;
}

void rename_open_file(HANDLE handle, HANDLE directory, std::wstring_view destination_name,
                      const std::filesystem::path& destination) {
  const auto filename_bytes = destination_name.size() * sizeof(wchar_t);
  struct ExtendedRenameInfo {
    DWORD flags;
    HANDLE root_directory;
    DWORD filename_length;
    wchar_t filename[1];
  };
  static_assert(offsetof(ExtendedRenameInfo, root_directory) ==
                offsetof(FILE_RENAME_INFO, RootDirectory));
  static_assert(offsetof(ExtendedRenameInfo, filename_length) ==
                offsetof(FILE_RENAME_INFO, FileNameLength));
  static_assert(offsetof(ExtendedRenameInfo, filename) == offsetof(FILE_RENAME_INFO, FileName));
  constexpr DWORD kReplaceIfExists = 0x00000001;
  constexpr DWORD kPosixSemantics = 0x00000002;
  constexpr ULONG kFileRenameInformation = 10;
  constexpr ULONG kFileRenameInformationEx = 65;
  // SetFileInformationByHandle documents a complete FILE_RENAME_INFO followed by the
  // variable-length name. Copying through filename[1] can trigger MSVC's object-size guard.
  const auto info_bytes = sizeof(FILE_RENAME_INFO) + filename_bytes;
  std::vector<std::max_align_t> storage(
      (info_bytes + sizeof(std::max_align_t) - 1U) / sizeof(std::max_align_t), std::max_align_t{});
  auto* raw = reinterpret_cast<std::byte*>(storage.data());
  const DWORD flags = kReplaceIfExists | kPosixSemantics;
  const HANDLE root_directory = directory;
  const auto filename_length = static_cast<DWORD>(filename_bytes);
  std::memcpy(raw + offsetof(ExtendedRenameInfo, flags), &flags, sizeof(flags));
  std::memcpy(raw + offsetof(ExtendedRenameInfo, root_directory), &root_directory,
              sizeof(root_directory));
  std::memcpy(raw + offsetof(ExtendedRenameInfo, filename_length), &filename_length,
              sizeof(filename_length));
  std::memcpy(raw + offsetof(ExtendedRenameInfo, filename), destination_name.data(),
              filename_bytes);

  constexpr DWORD retry_delay_ms = 10;
  constexpr int maximum_replace_attempts = 200;
  DWORD replace_error = ERROR_SUCCESS;
  bool extended_rename_unsupported = false;
  for (int attempt = 0; attempt < maximum_replace_attempts; ++attempt) {
    replace_error = set_windows_file_information(
        handle, storage.data(), static_cast<ULONG>(info_bytes), kFileRenameInformationEx);
    if (replace_error == ERROR_SUCCESS) {
      if (!published_path_identifies_handle(directory, destination_name, handle)) {
        throw WindowsPublicationIdentityError(
            "published calibration output does not identify the temporary file");
      }
      return;
    }
    if (replace_error == ERROR_INVALID_FUNCTION || replace_error == ERROR_NOT_SUPPORTED ||
        replace_error == ERROR_INVALID_PARAMETER) {
      extended_rename_unsupported = true;
      break;
    }
    if (replace_error != ERROR_SHARING_VIOLATION && replace_error != ERROR_ACCESS_DENIED) {
      throw_file_error("failed to replace calibration output", destination,
                       static_cast<int>(replace_error));
    }
    if (attempt + 1 < maximum_replace_attempts) {
      Sleep(retry_delay_ms);
    } else {
      throw_file_error("failed to replace calibration output", destination,
                       static_cast<int>(replace_error));
    }
  }

  DWORD destination_error = ERROR_SUCCESS;
  const HANDLE destination_handle = open_windows_file_relative(
      directory, destination_name, FILE_READ_ATTRIBUTES | SYNCHRONIZE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_OPEN,
      FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT,
      destination_error);
  UniqueWindowsHandle retained_destination(destination_handle);
  FILE_ATTRIBUTE_TAG_INFO destination_attributes{};
  const bool destination_is_reparse_point =
      destination_handle != INVALID_HANDLE_VALUE &&
      GetFileInformationByHandleEx(destination_handle, FileAttributeTagInfo,
                                   &destination_attributes, sizeof(destination_attributes)) != 0 &&
      (destination_attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
  if (!extended_rename_unsupported || destination_is_reparse_point) {
    throw_file_error(
        "failed to replace calibration output", destination,
        static_cast<int>(replace_error == ERROR_SUCCESS ? ERROR_NOT_SUPPORTED : replace_error));
  }
  const BOOLEAN replace_if_exists = TRUE;
  std::memcpy(raw + offsetof(FILE_RENAME_INFO, ReplaceIfExists), &replace_if_exists,
              sizeof(replace_if_exists));
  for (int attempt = 0; attempt < maximum_replace_attempts; ++attempt) {
    replace_error = set_windows_file_information(
        handle, storage.data(), static_cast<ULONG>(info_bytes), kFileRenameInformation);
    if (replace_error == ERROR_SUCCESS) {
      if (!published_path_identifies_handle(directory, destination_name, handle)) {
        throw WindowsPublicationIdentityError(
            "published calibration output does not identify the temporary file");
      }
      return;
    }
    if (replace_error != ERROR_SHARING_VIOLATION && replace_error != ERROR_ACCESS_DENIED) {
      break;
    }
    if (attempt + 1 < maximum_replace_attempts) {
      Sleep(retry_delay_ms);
    }
  }
  throw_file_error("failed to replace calibration output", destination,
                   static_cast<int>(replace_error));
}

void discard_open_file(HANDLE handle) noexcept {
  FILE_DISPOSITION_INFO disposition{.DeleteFile = TRUE};
  (void)SetFileInformationByHandle(handle, FileDispositionInfo, &disposition, sizeof(disposition));
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
  std::string label;
  std::filesystem::path path;
  UniqueFileDescriptor descriptor;
  struct stat identity{};
  struct stat path_identity{};

  [[nodiscard]] std::filesystem::path retained_path() const {
    return std::filesystem::path("/proc") / std::to_string(::getpid()) / "fd" /
           std::to_string(descriptor.get());
  }

  [[nodiscard]] reco::calibrate::CalibrationFileIdentity portable_identity() const {
    return {
        .device = static_cast<std::uint64_t>(identity.st_dev),
        .inode = static_cast<std::uint64_t>(identity.st_ino),
        .size = static_cast<std::uint64_t>(identity.st_size),
        .mode = static_cast<std::uint32_t>(identity.st_mode),
        .modified_seconds = static_cast<std::int64_t>(identity.st_mtim.tv_sec),
        .modified_nanoseconds = static_cast<std::int64_t>(identity.st_mtim.tv_nsec),
        .changed_seconds = static_cast<std::int64_t>(identity.st_ctim.tv_sec),
        .changed_nanoseconds = static_cast<std::int64_t>(identity.st_ctim.tv_nsec),
    };
  }

  void verify_unchanged() const {
    struct stat descriptor_identity{};
    struct stat named_identity{};
    struct stat followed_identity{};
    if (::fstat(descriptor.get(), &descriptor_identity) != 0 ||
        ::lstat(path.c_str(), &named_identity) != 0 ||
        ::stat(path.c_str(), &followed_identity) != 0) {
      throw std::runtime_error("cannot re-inspect " + label + " before publication");
    }
    const auto portable = [](const struct stat& value) {
      return reco::calibrate::CalibrationFileIdentity{
          .device = static_cast<std::uint64_t>(value.st_dev),
          .inode = static_cast<std::uint64_t>(value.st_ino),
          .size = static_cast<std::uint64_t>(value.st_size),
          .mode = static_cast<std::uint32_t>(value.st_mode),
          .modified_seconds = static_cast<std::int64_t>(value.st_mtim.tv_sec),
          .modified_nanoseconds = static_cast<std::int64_t>(value.st_mtim.tv_nsec),
          .changed_seconds = static_cast<std::int64_t>(value.st_ctim.tv_sec),
          .changed_nanoseconds = static_cast<std::int64_t>(value.st_ctim.tv_nsec),
      };
    };
    if (portable(descriptor_identity) != portable_identity() ||
        portable(followed_identity) != portable_identity() ||
        named_identity.st_dev != path_identity.st_dev ||
        named_identity.st_ino != path_identity.st_ino ||
        named_identity.st_mode != path_identity.st_mode) {
      throw std::runtime_error(label + " changed before calibration output publication");
    }
  }
};

[[nodiscard]] PinnedFileIdentity pin_input_identity(const std::filesystem::path& path,
                                                    std::string_view label) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    throw_file_error("cannot open " + std::string(label), path, errno);
  }
  PinnedFileIdentity pinned{
      .label = std::string(label), .path = path, .descriptor = UniqueFileDescriptor(descriptor)};
  if (::fstat(descriptor, &pinned.identity) != 0) {
    throw_file_error("cannot inspect " + std::string(label), path, errno);
  }
  if (::lstat(path.c_str(), &pinned.path_identity) != 0) {
    throw_file_error("cannot inspect " + std::string(label) + " path", path, errno);
  }
  if (!S_ISREG(pinned.identity.st_mode) || pinned.identity.st_size <= 0) {
    throw std::runtime_error(std::string(label) + " must be a regular file: " + path.string());
  }
  return pinned;
}

[[nodiscard]] bool same_file_identity(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

[[nodiscard]] std::optional<std::string>
pinned_identity_alias_error(const struct stat& identity, const PinnedFileIdentity& left_input,
                            const PinnedFileIdentity& right_input,
                            const std::vector<PinnedFileIdentity>& lens_profiles) {
  const auto alias_error = [](std::string_view label) {
    return "output calibration path identifies the " + std::string(label);
  };
  if (same_file_identity(identity, left_input.identity)) {
    return alias_error(left_input.label);
  }
  if (same_file_identity(identity, left_input.path_identity)) {
    return alias_error(left_input.label);
  }
  if (same_file_identity(identity, right_input.identity)) {
    return alias_error(right_input.label);
  }
  if (same_file_identity(identity, right_input.path_identity)) {
    return alias_error(right_input.label);
  }
  for (const auto& profile : lens_profiles) {
    if (same_file_identity(identity, profile.identity) ||
        same_file_identity(identity, profile.path_identity)) {
      return alias_error(profile.label);
    }
  }
  return std::nullopt;
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

struct DirectoryEntrySnapshot {
  UniqueFileDescriptor descriptor;
  struct stat identity{};
};

[[nodiscard]] DirectoryEntrySnapshot capture_directory_entry_snapshot(int directory_descriptor,
                                                                      std::string_view name) {
  DirectoryEntrySnapshot snapshot;
  const std::string filename(name);
  const int descriptor =
      ::openat(directory_descriptor, filename.c_str(), O_PATH | O_NOFOLLOW | O_CLOEXEC);
  if (descriptor < 0) {
    throw std::system_error(errno, std::system_category(),
                            "cannot retain calibration output entry identity");
  }
  snapshot.descriptor = UniqueFileDescriptor(descriptor);
  if (::fstat(snapshot.descriptor.get(), &snapshot.identity) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "cannot inspect retained calibration output entry identity");
  }
  return snapshot;
}

[[nodiscard]] bool directory_entry_matches_snapshot(int directory_descriptor, std::string_view name,
                                                    const DirectoryEntrySnapshot& snapshot) {
  struct stat current{};
  struct stat retained{};
  const std::string filename(name);
  return ::fstat(snapshot.descriptor.get(), &retained) == 0 &&
         ::fstatat(directory_descriptor, filename.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
         same_file_identity(retained, snapshot.identity) && same_file_identity(current, retained) &&
         current.st_mode == retained.st_mode;
}

[[nodiscard]] bool unlink_directory_entry_if_unchanged(int directory_descriptor,
                                                       std::string_view name,
                                                       const DirectoryEntrySnapshot& snapshot) {
  if (!directory_entry_matches_snapshot(directory_descriptor, name, snapshot)) {
    return false;
  }
  const std::string filename(name);
  return ::unlinkat(directory_descriptor, filename.c_str(), 0) == 0;
}

[[nodiscard]] std::optional<std::string>
validate_pinned_output_identity(int directory_descriptor, std::string_view destination_name,
                                const PinnedFileIdentity& left_input,
                                const PinnedFileIdentity& right_input,
                                const std::vector<PinnedFileIdentity>& lens_profiles) {
  left_input.verify_unchanged();
  right_input.verify_unchanged();
  for (const auto& profile : lens_profiles) {
    profile.verify_unchanged();
  }
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

  if (const auto error =
          pinned_identity_alias_error(destination_identity, left_input, right_input, lens_profiles);
      error.has_value()) {
    return error;
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
    return pinned_identity_alias_error(followed_identity, left_input, right_input, lens_profiles);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string> displaced_output_removal_error(
    int directory_descriptor, std::string_view name, const PinnedFileIdentity& left_input,
    const PinnedFileIdentity& right_input, const std::vector<PinnedFileIdentity>& lens_profiles) {
  struct stat identity{};
  const std::string filename(name);
  if (::fstatat(directory_descriptor, filename.c_str(), &identity, AT_SYMLINK_NOFOLLOW) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "cannot inspect displaced calibration output");
  }
  if (const auto error =
          pinned_identity_alias_error(identity, left_input, right_input, lens_profiles);
      error.has_value()) {
    return error;
  }
  if (S_ISDIR(identity.st_mode)) {
    return "output calibration path identifies a directory";
  }
  if (S_ISLNK(identity.st_mode)) {
    struct stat followed_identity{};
    if (::fstatat(directory_descriptor, filename.c_str(), &followed_identity, 0) != 0) {
      if (errno == ENOENT || errno == ELOOP) {
        return std::nullopt;
      }
      throw std::system_error(errno, std::system_category(),
                              "cannot inspect displaced calibration symlink target");
    }
    return pinned_identity_alias_error(followed_identity, left_input, right_input, lens_profiles);
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

enum class DescriptorLinkStatus { Linked, AlreadyExists, Unsupported };

[[nodiscard]] bool descriptor_link_is_unsupported(int error) {
  return error == EPERM || error == EACCES || error == EOPNOTSUPP || error == ENOSYS ||
         error == EXDEV || error == EMLINK || error == EINVAL || error == ENOENT;
}

[[nodiscard]] DescriptorLinkStatus link_descriptor_at(int directory_descriptor,
                                                      std::string_view name,
                                                      int temporary_descriptor,
                                                      bool force_rename_fallback) {
  if (force_rename_fallback) {
    return DescriptorLinkStatus::Unsupported;
  }
  const auto descriptor_path = std::filesystem::path("/proc") / std::to_string(::getpid()) / "fd" /
                               std::to_string(temporary_descriptor);
  const std::string filename(name);
  if (::linkat(AT_FDCWD, descriptor_path.c_str(), directory_descriptor, filename.c_str(),
               AT_SYMLINK_FOLLOW) == 0) {
    return DescriptorLinkStatus::Linked;
  }
  if (errno == EEXIST) {
    return DescriptorLinkStatus::AlreadyExists;
  }
  if (descriptor_link_is_unsupported(errno)) {
    return DescriptorLinkStatus::Unsupported;
  }
  throw std::system_error(errno, std::system_category(),
                          "cannot bind temporary calibration output");
}

[[nodiscard]] std::optional<std::string>
create_descriptor_publication_link_at(int directory_descriptor,
                                      const std::filesystem::path& destination,
                                      int temporary_descriptor, bool force_rename_fallback) {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  for (int attempt = 0; attempt < 128; ++attempt) {
    std::array<char, 32> token{};
    for (auto& digit : token) {
      digit = hex[random() & 0x0fU];
    }
    const auto name =
        destination.filename().string() + ".publish." + std::string(token.begin(), token.end());
    const auto status =
        link_descriptor_at(directory_descriptor, name, temporary_descriptor, force_rename_fallback);
    if (status == DescriptorLinkStatus::Linked) {
      return std::optional<std::string>(name);
    }
    if (status == DescriptorLinkStatus::Unsupported) {
      return std::nullopt;
    }
  }
  throw std::runtime_error("cannot bind unique temporary calibration output for " +
                           destination.string());
}

[[nodiscard]] bool directory_entry_exists_at(int directory_descriptor, std::string_view name) {
  struct stat identity{};
  const std::string filename(name);
  if (::fstatat(directory_descriptor, filename.c_str(), &identity, AT_SYMLINK_NOFOLLOW) == 0) {
    return true;
  }
  if (errno == ENOENT) {
    return false;
  }
  throw std::system_error(errno, std::system_category(), "cannot inspect calibration output entry");
}

[[nodiscard]] UniqueFileDescriptor
lock_output_directory(int directory_descriptor, const std::filesystem::path& directory,
                      const std::function<void()>& on_contention) {
  constexpr auto lock_timeout = std::chrono::seconds(2);
  constexpr auto retry_delay = std::chrono::milliseconds(10);
  const int raw_lock =
      ::openat(directory_descriptor, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (raw_lock < 0) {
    throw_file_error("cannot open calibration output directory lock", directory, errno);
  }
  UniqueFileDescriptor lock(raw_lock);
  struct stat identity{};
  if (::fstat(lock.get(), &identity) != 0) {
    throw_file_error("cannot inspect calibration output directory lock", directory, errno);
  }
  if (!S_ISDIR(identity.st_mode)) {
    throw std::runtime_error("calibration output lock is not a directory: " + directory.string());
  }
  const auto deadline = std::chrono::steady_clock::now() + lock_timeout;
  bool contention_reported = false;
  while (::flock(lock.get(), LOCK_EX | LOCK_NB) != 0) {
    if (errno == EINTR) {
      continue;
    }
    if ((errno == EWOULDBLOCK || errno == EAGAIN) && std::chrono::steady_clock::now() < deadline) {
      if (!contention_reported && on_contention) {
        on_contention();
        contention_reported = true;
      }
      std::this_thread::sleep_for(retry_delay);
      continue;
    }
    if (errno == EWOULDBLOCK || errno == EAGAIN) {
      throw std::runtime_error("timed out locking calibration output directory: " +
                               directory.string());
    }
    throw_file_error("cannot lock calibration output directory", directory, errno);
  }
  return lock;
}

void exchange_directory_entries_at(int directory_descriptor, std::string_view left,
                                   std::string_view right,
                                   const std::filesystem::path& destination) {
  const std::string left_name(left);
  const std::string right_name(right);
  if (::syscall(SYS_renameat2, directory_descriptor, left_name.c_str(), directory_descriptor,
                right_name.c_str(), RENAME_EXCHANGE) != 0) {
    throw_file_error("failed to exchange calibration output", destination, errno);
  }
}

[[nodiscard]] bool exchange_directory_entries_noexcept(int directory_descriptor,
                                                       const std::string& left,
                                                       const std::string& right) noexcept {
  return ::syscall(SYS_renameat2, directory_descriptor, left.c_str(), directory_descriptor,
                   right.c_str(), RENAME_EXCHANGE) == 0;
}

[[nodiscard]] bool rename_directory_entry_noreplace_at(int directory_descriptor,
                                                       std::string_view source,
                                                       std::string_view destination_name,
                                                       const std::filesystem::path& destination) {
  const std::string source_name(source);
  const std::string target_name(destination_name);
  if (::syscall(SYS_renameat2, directory_descriptor, source_name.c_str(), directory_descriptor,
                target_name.c_str(), RENAME_NOREPLACE) == 0) {
    return true;
  }
  if (errno == EEXIST) {
    return false;
  }
  throw_file_error("failed to publish calibration output", destination, errno);
}

[[nodiscard]] bool
rename_directory_entry_noreplace_noexcept(int directory_descriptor, const std::string& source,
                                          const std::string& destination_name) noexcept {
  return ::syscall(SYS_renameat2, directory_descriptor, source.c_str(), directory_descriptor,
                   destination_name.c_str(), RENAME_NOREPLACE) == 0;
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
[[nodiscard]] bool same_file_identity(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

#if defined(__APPLE__)
[[nodiscard]] bool same_timestamp(const timespec& left, const timespec& right) {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

[[nodiscard]] bool same_posix_snapshot(const struct stat& left, const struct stat& right) {
  return same_file_identity(left, right) && left.st_size == right.st_size &&
         left.st_mode == right.st_mode && same_timestamp(left.st_mtimespec, right.st_mtimespec) &&
         same_timestamp(left.st_ctimespec, right.st_ctimespec);
}
#endif

struct PinnedPosixPath {
  std::string label;
  std::filesystem::path path;
  int descriptor = -1;
  struct stat identity{};
  struct stat path_identity{};

  ~PinnedPosixPath() {
    if (descriptor >= 0) {
      (void)::close(descriptor);
    }
  }

  PinnedPosixPath() = default;
  PinnedPosixPath(const PinnedPosixPath&) = delete;
  PinnedPosixPath& operator=(const PinnedPosixPath&) = delete;
  PinnedPosixPath(PinnedPosixPath&& other) noexcept
      : label(std::move(other.label)), path(std::move(other.path)),
        descriptor(std::exchange(other.descriptor, -1)), identity(other.identity),
        path_identity(other.path_identity) {}
  PinnedPosixPath& operator=(PinnedPosixPath&& other) noexcept {
    if (this != &other) {
      if (descriptor >= 0) {
        (void)::close(descriptor);
      }
      label = std::move(other.label);
      path = std::move(other.path);
      descriptor = std::exchange(other.descriptor, -1);
      identity = other.identity;
      path_identity = other.path_identity;
    }
    return *this;
  }

  void verify_unchanged() const {
    struct stat descriptor_identity{};
    struct stat named_identity{};
    struct stat followed_identity{};
    if (::fstat(descriptor, &descriptor_identity) != 0 ||
        ::lstat(path.c_str(), &named_identity) != 0 ||
        ::stat(path.c_str(), &followed_identity) != 0) {
      throw std::runtime_error("cannot re-inspect " + label + " before publication");
    }
#if defined(__APPLE__)
    if (!same_posix_snapshot(descriptor_identity, identity) ||
        !same_posix_snapshot(followed_identity, identity) ||
        !same_posix_snapshot(named_identity, path_identity)) {
#else
    if (!same_file_identity(descriptor_identity, identity) ||
        descriptor_identity.st_size != identity.st_size ||
        descriptor_identity.st_mode != identity.st_mode ||
        !same_file_identity(followed_identity, identity) ||
        !same_file_identity(named_identity, path_identity) ||
        named_identity.st_mode != path_identity.st_mode) {
#endif
      throw std::runtime_error(label + " changed before calibration output publication");
    }
  }
};

#if defined(__APPLE__)
class PosixDirectoryLock {
public:
  explicit PosixDirectoryLock(int descriptor) : descriptor_(descriptor) {}
  ~PosixDirectoryLock() {
    if (descriptor_ >= 0) {
      (void)::close(descriptor_);
    }
  }

  PosixDirectoryLock(const PosixDirectoryLock&) = delete;
  PosixDirectoryLock& operator=(const PosixDirectoryLock&) = delete;
  PosixDirectoryLock(PosixDirectoryLock&& other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}

  [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
  int descriptor_ = -1;
};

[[nodiscard]] PosixDirectoryLock
lock_posix_output_directory(const std::filesystem::path& destination,
                            const std::function<void()>& on_contention) {
  const auto parent =
      destination.has_parent_path() ? destination.parent_path() : std::filesystem::path(".");
  const int descriptor = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (descriptor < 0) {
    throw_file_error("cannot open calibration output directory lock", parent, errno);
  }
  struct stat identity{};
  if (::fstat(descriptor, &identity) != 0) {
    const int error = errno;
    (void)::close(descriptor);
    throw_file_error("cannot inspect calibration output directory lock", parent, error);
  }
  if (!S_ISDIR(identity.st_mode)) {
    (void)::close(descriptor);
    throw_file_error("cannot inspect calibration output directory lock", parent, ENOTDIR);
  }
  bool contention_reported = false;
  for (;;) {
    if (::flock(descriptor, LOCK_EX | LOCK_NB) == 0) {
      break;
    }
    const int nonblocking_error = errno;
    if (nonblocking_error == EINTR) {
      continue;
    }
    if (nonblocking_error == EWOULDBLOCK || nonblocking_error == EAGAIN) {
      if (!contention_reported && on_contention) {
        try {
          on_contention();
        } catch (...) {
          (void)::close(descriptor);
          throw;
        }
        contention_reported = true;
      }
      for (;;) {
        if (::flock(descriptor, LOCK_EX) == 0) {
          return PosixDirectoryLock(descriptor);
        }
        if (errno != EINTR) {
          const int error = errno;
          (void)::close(descriptor);
          throw_file_error("cannot lock calibration output directory", parent, error);
        }
      }
    }
    (void)::close(descriptor);
    throw_file_error("cannot lock calibration output directory", parent, nonblocking_error);
  }
  return PosixDirectoryLock(descriptor);
}
#endif

[[nodiscard]] PinnedPosixPath pin_posix_path_for_publication(const std::filesystem::path& path,
                                                             std::string_view label) {
  PinnedPosixPath pinned;
  pinned.label = std::string(label);
  pinned.path = path;
  pinned.descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (pinned.descriptor < 0) {
    throw_file_error("cannot open " + pinned.label, path, errno);
  }
  if (::fstat(pinned.descriptor, &pinned.identity) != 0) {
    throw_file_error("cannot inspect " + pinned.label, path, errno);
  }
  if (::lstat(path.c_str(), &pinned.path_identity) != 0) {
    throw_file_error("cannot inspect " + pinned.label + " path", path, errno);
  }
  if (!S_ISREG(pinned.identity.st_mode) || pinned.identity.st_size <= 0) {
    throw std::runtime_error(pinned.label + " must be a regular file: " + path.string());
  }
  return pinned;
}

[[nodiscard]] std::optional<std::string>
pinned_posix_alias_error(const struct stat& identity, const PinnedPosixPath& left_input,
                         const PinnedPosixPath& right_input,
                         const std::vector<PinnedPosixPath>& lens_profiles) {
  const auto matches = [&](const PinnedPosixPath& input) {
    return same_file_identity(identity, input.identity) ||
           same_file_identity(identity, input.path_identity);
  };
  if (matches(left_input)) {
    return "output calibration path identifies the " + left_input.label;
  }
  if (matches(right_input)) {
    return "output calibration path identifies the " + right_input.label;
  }
  for (const auto& profile : lens_profiles) {
    if (matches(profile)) {
      return "output calibration path identifies the " + profile.label;
    }
  }
  return std::nullopt;
}

#if defined(__APPLE__)
[[nodiscard]] std::optional<std::string> validate_pinned_posix_output_identity_at(
    int directory_descriptor, std::string_view destination_name, const PinnedPosixPath& left_input,
    const PinnedPosixPath& right_input, const std::vector<PinnedPosixPath>& lens_profiles) {
  left_input.verify_unchanged();
  right_input.verify_unchanged();
  for (const auto& profile : lens_profiles) {
    profile.verify_unchanged();
  }
  const std::string name(destination_name);
  struct stat identity{};
  if (::fstatat(directory_descriptor, name.c_str(), &identity, AT_SYMLINK_NOFOLLOW) != 0) {
    if (errno == ENOENT) {
      return std::nullopt;
    }
    throw std::system_error(errno, std::system_category(),
                            "cannot inspect calibration output identity");
  }
  if (const auto error = pinned_posix_alias_error(identity, left_input, right_input, lens_profiles);
      error.has_value()) {
    return error;
  }
  if (S_ISLNK(identity.st_mode)) {
    struct stat followed_identity{};
    if (::fstatat(directory_descriptor, name.c_str(), &followed_identity, 0) != 0) {
      if (errno == ENOENT || errno == ELOOP) {
        return std::nullopt;
      }
      throw std::system_error(errno, std::system_category(),
                              "cannot inspect calibration output symlink target");
    }
    return pinned_posix_alias_error(followed_identity, left_input, right_input, lens_profiles);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string> displaced_posix_output_removal_error_at(
    int directory_descriptor, std::string_view name, const PinnedPosixPath& left_input,
    const PinnedPosixPath& right_input, const std::vector<PinnedPosixPath>& lens_profiles) {
  const std::string filename(name);
  struct stat identity{};
  if (::fstatat(directory_descriptor, filename.c_str(), &identity, AT_SYMLINK_NOFOLLOW) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "cannot inspect displaced calibration output");
  }
  if (const auto error = pinned_posix_alias_error(identity, left_input, right_input, lens_profiles);
      error.has_value()) {
    return error;
  }
  if (S_ISDIR(identity.st_mode)) {
    return "output calibration path identifies a directory";
  }
  if (S_ISLNK(identity.st_mode)) {
    struct stat followed_identity{};
    if (::fstatat(directory_descriptor, filename.c_str(), &followed_identity, 0) != 0) {
      if (errno == ENOENT || errno == ELOOP) {
        return std::nullopt;
      }
      throw std::system_error(errno, std::system_category(),
                              "cannot inspect displaced calibration symlink target");
    }
    return pinned_posix_alias_error(followed_identity, left_input, right_input, lens_profiles);
  }
  return std::nullopt;
}

[[nodiscard]] bool path_identifies_descriptor_at(int directory_descriptor, std::string_view name,
                                                 int descriptor) {
  const std::string filename(name);
  struct stat descriptor_identity{};
  struct stat path_identity{};
  return ::fstat(descriptor, &descriptor_identity) == 0 &&
         ::fstatat(directory_descriptor, filename.c_str(), &path_identity, AT_SYMLINK_NOFOLLOW) ==
             0 &&
         S_ISREG(path_identity.st_mode) && same_file_identity(descriptor_identity, path_identity);
}

struct PosixDirectoryEntrySnapshot {
  int descriptor = -1;
  struct stat identity{};

  ~PosixDirectoryEntrySnapshot() {
    if (descriptor >= 0) {
      (void)::close(descriptor);
    }
  }
  PosixDirectoryEntrySnapshot() = default;
  PosixDirectoryEntrySnapshot(const PosixDirectoryEntrySnapshot&) = delete;
  PosixDirectoryEntrySnapshot& operator=(const PosixDirectoryEntrySnapshot&) = delete;
  PosixDirectoryEntrySnapshot(PosixDirectoryEntrySnapshot&& other) noexcept
      : descriptor(std::exchange(other.descriptor, -1)), identity(other.identity) {}
  PosixDirectoryEntrySnapshot& operator=(PosixDirectoryEntrySnapshot&& other) noexcept {
    if (this != &other) {
      if (descriptor >= 0) {
        (void)::close(descriptor);
      }
      descriptor = std::exchange(other.descriptor, -1);
      identity = other.identity;
    }
    return *this;
  }
};

[[nodiscard]] PosixDirectoryEntrySnapshot
capture_posix_directory_entry_snapshot(int directory_descriptor, std::string_view name) {
  PosixDirectoryEntrySnapshot snapshot;
  const std::string filename(name);
  snapshot.descriptor =
      ::openat(directory_descriptor, filename.c_str(), O_EVTONLY | O_CLOEXEC | O_SYMLINK);
  if (snapshot.descriptor < 0) {
    throw std::system_error(errno, std::system_category(),
                            "cannot retain calibration output entry identity");
  }
  if (::fstat(snapshot.descriptor, &snapshot.identity) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "cannot inspect retained calibration output entry identity");
  }
  return snapshot;
}

[[nodiscard]] bool
posix_directory_entry_matches_snapshot(int directory_descriptor, std::string_view name,
                                       const PosixDirectoryEntrySnapshot& snapshot) {
  struct stat current{};
  struct stat retained{};
  const std::string filename(name);
  return ::fstat(snapshot.descriptor, &retained) == 0 &&
         ::fstatat(directory_descriptor, filename.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
         same_file_identity(retained, snapshot.identity) && same_file_identity(current, retained) &&
         current.st_mode == retained.st_mode;
}

[[nodiscard]] bool
unlink_posix_directory_entry_if_unchanged(int directory_descriptor, std::string_view name,
                                          const PosixDirectoryEntrySnapshot& snapshot) {
  if (!posix_directory_entry_matches_snapshot(directory_descriptor, name, snapshot)) {
    return false;
  }
  const std::string filename(name);
  return ::unlinkat(directory_descriptor, filename.c_str(), 0) == 0;
}

[[nodiscard]] std::string create_exclusive_temporary_at(int directory_descriptor,
                                                        const std::filesystem::path& destination,
                                                        int& descriptor) {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  for (int attempt = 0; attempt < 128; ++attempt) {
    std::array<char, 32> token{};
    for (auto& digit : token) {
      digit = hex[random() & 0x0fU];
    }
    const auto name =
        destination.filename().string() + ".tmp." + std::string(token.begin(), token.end());
    descriptor = ::openat(directory_descriptor, name.c_str(),
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor >= 0) {
      return name;
    }
    if (errno != EEXIST) {
      throw_file_error("cannot create temporary calibration output", destination, errno);
    }
  }
  throw std::runtime_error("cannot create unique temporary calibration output for " +
                           destination.string());
}
#endif

#if !defined(__APPLE__)
[[nodiscard]] std::optional<std::string> displaced_posix_output_removal_error(
    const std::filesystem::path& path, const PinnedPosixPath& left_input,
    const PinnedPosixPath& right_input, const std::vector<PinnedPosixPath>& lens_profiles) {
  struct stat identity{};
  if (::lstat(path.c_str(), &identity) != 0) {
    throw_file_error("cannot inspect displaced calibration output", path, errno);
  }
  if (const auto error = pinned_posix_alias_error(identity, left_input, right_input, lens_profiles);
      error.has_value()) {
    return error;
  }
  if (S_ISDIR(identity.st_mode)) {
    return "output calibration path identifies a directory";
  }
  if (S_ISLNK(identity.st_mode)) {
    struct stat followed_identity{};
    if (::stat(path.c_str(), &followed_identity) != 0) {
      if (errno == ENOENT || errno == ELOOP) {
        return std::nullopt;
      }
      throw_file_error("cannot inspect displaced calibration symlink target", path, errno);
    }
    return pinned_posix_alias_error(followed_identity, left_input, right_input, lens_profiles);
  }
  return std::nullopt;
}

[[nodiscard]] bool path_identifies_descriptor(const std::filesystem::path& path, int descriptor) {
  struct stat descriptor_identity{};
  struct stat path_identity{};
  return ::fstat(descriptor, &descriptor_identity) == 0 &&
         ::lstat(path.c_str(), &path_identity) == 0 && S_ISREG(path_identity.st_mode) &&
         descriptor_identity.st_dev == path_identity.st_dev &&
         descriptor_identity.st_ino == path_identity.st_ino;
}

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
#endif

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

void write_calibration_json_atomically_impl(
    std::string_view json, const std::filesystem::path& destination,
    const std::filesystem::path& left_input, const std::filesystem::path& right_input,
    const std::function<void()>& before_publish,
    std::span<const std::filesystem::path> lens_profiles,
    const std::function<void()>& before_commit, bool force_rename_fallback,
    const std::function<void()>& after_publish, const std::function<void()>& on_lock_contention) {
  std::string contents(json);
  contents.push_back('\n');
#if defined(_WIN32)
  (void)force_rename_fallback;
  const auto destination_name = destination.filename().wstring();
  if (destination_name.empty() || destination_name == L"." || destination_name == L"..") {
    throw std::runtime_error("calibration output path must name a file: " + destination.string());
  }
  const auto output_directory = pin_windows_output_directory(destination);
  const auto left_reservation = pin_windows_path_for_publication(left_input, "left video input");
  const auto right_reservation = pin_windows_path_for_publication(right_input, "right video input");
  std::vector<PinnedWindowsPath> profile_reservations;
  profile_reservations.reserve(lens_profiles.size());
  for (std::size_t index = 0; index < lens_profiles.size(); ++index) {
    profile_reservations.push_back(pin_windows_path_for_publication(
        lens_profiles[index], index == 0 ? "left lens profile" : "right lens profile"));
  }
  static std::mutex publication_mutex;
  std::lock_guard publication_lock(publication_mutex);
  [[maybe_unused]] auto interprocess_publication_lock = lock_windows_output_directory(
      output_directory.handle.get(), output_directory.path, on_lock_contention);
  HANDLE handle = INVALID_HANDLE_VALUE;
  std::filesystem::path temporary;
  bool temporary_exists = false;
  try {
    temporary = create_exclusive_temporary(destination, output_directory.handle.get(), handle);
    temporary_exists = true;
    write_all(handle, contents, temporary);
    if (FlushFileBuffers(handle) == 0) {
      throw_file_error("failed to flush temporary calibration output", temporary,
                       static_cast<int>(GetLastError()));
    }
    if (before_publish) {
      before_publish();
    }
    if (const auto error = reco::calibrate::validate_calibration_output_identity(
            left_input, right_input, destination, lens_profiles);
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }
    if (before_commit) {
      before_commit();
    }
    left_reservation.verify_unchanged(left_input, "left video input");
    right_reservation.verify_unchanged(right_input, "right video input");
    for (std::size_t index = 0; index < profile_reservations.size(); ++index) {
      profile_reservations[index].verify_unchanged(
          lens_profiles[index], index == 0 ? "left lens profile" : "right lens profile");
    }
    if (const auto error = reco::calibrate::validate_calibration_output_identity(
            left_input, right_input, destination, lens_profiles);
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }
    rename_open_file(handle, output_directory.handle.get(), destination_name, destination);
    temporary_exists = false;
    if (after_publish) {
      after_publish();
    }
    if (!published_path_identifies_handle(output_directory.handle.get(), destination_name,
                                          handle)) {
      throw WindowsPublicationIdentityError(
          "published calibration output identity changed after publication");
    }
    if (CloseHandle(handle) == 0) {
      handle = INVALID_HANDLE_VALUE;
      throw_file_error("failed to close calibration output", destination,
                       static_cast<int>(GetLastError()));
    }
    handle = INVALID_HANDLE_VALUE;
  } catch (const WindowsPublicationIdentityError&) {
    if (handle != INVALID_HANDLE_VALUE) {
      (void)CloseHandle(handle);
      handle = INVALID_HANDLE_VALUE;
    }
    temporary_exists = false;
    throw;
  } catch (...) {
    if (handle != INVALID_HANDLE_VALUE) {
      discard_open_file(handle);
      CloseHandle(handle);
      temporary_exists = false;
    } else if (temporary_exists) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
    throw;
  }
#elif defined(__linux__)
  const auto left_identity = pin_input_identity(left_input, "left video input");
  const auto right_identity = pin_input_identity(right_input, "right video input");
  std::vector<PinnedFileIdentity> profile_identities;
  profile_identities.reserve(lens_profiles.size());
  for (std::size_t index = 0; index < lens_profiles.size(); ++index) {
    profile_identities.push_back(pin_input_identity(
        lens_profiles[index], index == 0 ? "left lens profile" : "right lens profile"));
  }
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
  auto publication_lock =
      lock_output_directory(directory_descriptor.get(), parent, on_lock_contention);

  TemporaryOutput temporary;
  bool temporary_exists = false;
  std::string publication_name;
  bool publication_exists = false;
  bool commit_hook_ran = false;
  bool after_publish_hook_ran = false;
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
    if (const auto error =
            validate_pinned_output_identity(directory_descriptor.get(), destination_name,
                                            left_identity, right_identity, profile_identities);
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }

    const auto run_commit_hook = [&] {
      if (!commit_hook_ran && before_commit) {
        before_commit();
      }
      commit_hook_ran = true;
      if (const auto error =
              validate_pinned_output_identity(directory_descriptor.get(), destination_name,
                                              left_identity, right_identity, profile_identities);
          error.has_value()) {
        throw std::runtime_error("refusing to publish calibration output: " + *error);
      }
    };
    const auto validate_published_or_rollback = [&](auto&& rollback) {
      try {
        if (const auto error =
                validate_pinned_output_identity(directory_descriptor.get(), destination_name,
                                                left_identity, right_identity, profile_identities);
            error.has_value()) {
          throw std::runtime_error("refusing to publish calibration output: " + *error);
        }
      } catch (...) {
        if (!std::forward<decltype(rollback)>(rollback)()) {
          throw std::runtime_error(
              "refusing to publish calibration output: post-publication validation failed and "
              "rollback failed");
        }
        throw;
      }
    };
    const auto run_after_publish_hook = [&] {
      if (!after_publish_hook_ran && after_publish) {
        after_publish();
      }
      after_publish_hook_ran = true;
    };
    bool published = false;
    bool use_rename_fallback = force_rename_fallback;
    bool destination_exists =
        directory_entry_exists_at(directory_descriptor.get(), destination_name);

    if (!destination_exists && !use_rename_fallback) {
      const auto retention = create_descriptor_publication_link_at(
          directory_descriptor.get(), destination, temporary.descriptor.get(), false);
      if (!retention.has_value()) {
        use_rename_fallback = true;
      } else {
        publication_name = *retention;
        publication_exists = true;
        run_commit_hook();
        const auto status = link_descriptor_at(directory_descriptor.get(), destination_name,
                                               temporary.descriptor.get(), false);
        if (status == DescriptorLinkStatus::Linked) {
          published = temporary_name_identifies_descriptor(
              directory_descriptor.get(), destination_name, temporary.descriptor.get());
          if (!published) {
            throw std::runtime_error(
                "refusing to publish calibration output: published output identity changed");
          }
          run_after_publish_hook();
          validate_published_or_rollback([&] {
            return temporary_name_identifies_descriptor(
                       directory_descriptor.get(), destination_name, temporary.descriptor.get()) &&
                   ::unlinkat(directory_descriptor.get(), destination_name.c_str(), 0) == 0;
          });
          if (temporary_name_identifies_descriptor(directory_descriptor.get(), publication_name,
                                                   temporary.descriptor.get())) {
            if (::unlinkat(directory_descriptor.get(), publication_name.c_str(), 0) != 0) {
              throw_file_error("failed to remove calibration output retention link", destination,
                               errno);
            }
          }
          publication_exists = false;
        } else if (status == DescriptorLinkStatus::AlreadyExists) {
          destination_exists = true;
        } else {
          use_rename_fallback = true;
        }
      }
    }

    if (!published && !use_rename_fallback) {
      if (publication_name.empty()) {
        const auto publication = create_descriptor_publication_link_at(
            directory_descriptor.get(), destination, temporary.descriptor.get(), false);
        if (!publication.has_value()) {
          use_rename_fallback = true;
        } else {
          publication_name = *publication;
          publication_exists = true;
        }
      }
      if (!use_rename_fallback) {
        run_commit_hook();
        if (!temporary_name_identifies_descriptor(directory_descriptor.get(), publication_name,
                                                  temporary.descriptor.get())) {
          publication_exists = false;
          throw std::runtime_error(
              "refusing to publish calibration output: descriptor-bound output identity changed "
              "before exchange");
        }
        const auto displaced_snapshot =
            capture_directory_entry_snapshot(directory_descriptor.get(), destination_name);
        exchange_directory_entries_at(directory_descriptor.get(), publication_name,
                                      destination_name, destination);
        run_after_publish_hook();
        const auto rollback_exchange_if_unchanged = [&] {
          if (!temporary_name_identifies_descriptor(directory_descriptor.get(), destination_name,
                                                    temporary.descriptor.get()) ||
              !directory_entry_matches_snapshot(directory_descriptor.get(), publication_name,
                                                displaced_snapshot)) {
            return false;
          }
          const bool rolled_back = exchange_directory_entries_noexcept(
              directory_descriptor.get(), publication_name, destination_name);
          publication_exists = temporary_name_identifies_descriptor(
              directory_descriptor.get(), publication_name, temporary.descriptor.get());
          return rolled_back;
        };
        if (!temporary_name_identifies_descriptor(directory_descriptor.get(), destination_name,
                                                  temporary.descriptor.get()) ||
            !directory_entry_matches_snapshot(directory_descriptor.get(), publication_name,
                                              displaced_snapshot)) {
          publication_exists = temporary_name_identifies_descriptor(
              directory_descriptor.get(), publication_name, temporary.descriptor.get());
          throw std::runtime_error(
              "refusing to publish calibration output: exchanged output identity changed; "
              "rollback refused");
        }
        if (const auto error =
                displaced_output_removal_error(directory_descriptor.get(), publication_name,
                                               left_identity, right_identity, profile_identities);
            error.has_value()) {
          const bool rolled_back = rollback_exchange_if_unchanged();
          if (!rolled_back) {
            throw std::runtime_error("refusing to publish calibration output: " + *error +
                                     " and rollback failed");
          }
          throw std::runtime_error("refusing to publish calibration output: " + *error);
        }
        validate_published_or_rollback([&] { return rollback_exchange_if_unchanged(); });
        if (!unlink_directory_entry_if_unchanged(directory_descriptor.get(), publication_name,
                                                 displaced_snapshot)) {
          const bool rolled_back = rollback_exchange_if_unchanged();
          throw std::runtime_error(
              rolled_back
                  ? "failed to remove unchanged displaced calibration output"
                  : "displaced calibration output changed before removal; rollback refused");
        }
        publication_exists = false;
        published = true;
      }
    }

    if (!published && use_rename_fallback) {
      run_commit_hook();
      if (!temporary_name_identifies_descriptor(directory_descriptor.get(), temporary.name,
                                                temporary.descriptor.get())) {
        temporary_exists = false;
        throw std::runtime_error(
            "refusing to publish calibration output: fallback output identity changed before "
            "exchange");
      }
      bool exchanged = destination_exists;
      std::optional<DirectoryEntrySnapshot> displaced_snapshot;
      if (exchanged) {
        displaced_snapshot =
            capture_directory_entry_snapshot(directory_descriptor.get(), destination_name);
        exchange_directory_entries_at(directory_descriptor.get(), temporary.name, destination_name,
                                      destination);
      } else if (!rename_directory_entry_noreplace_at(directory_descriptor.get(), temporary.name,
                                                      destination_name, destination)) {
        exchanged = true;
        displaced_snapshot =
            capture_directory_entry_snapshot(directory_descriptor.get(), destination_name);
        exchange_directory_entries_at(directory_descriptor.get(), temporary.name, destination_name,
                                      destination);
      }
      run_after_publish_hook();
      const auto rollback_fallback_if_unchanged = [&] {
        if (!temporary_name_identifies_descriptor(directory_descriptor.get(), destination_name,
                                                  temporary.descriptor.get())) {
          return false;
        }
        bool rolled_back = false;
        if (exchanged) {
          if (!displaced_snapshot.has_value() ||
              !directory_entry_matches_snapshot(directory_descriptor.get(), temporary.name,
                                                *displaced_snapshot)) {
            return false;
          }
          rolled_back = exchange_directory_entries_noexcept(directory_descriptor.get(),
                                                            temporary.name, destination_name);
        } else {
          rolled_back = rename_directory_entry_noreplace_noexcept(directory_descriptor.get(),
                                                                  destination_name, temporary.name);
        }
        temporary_exists = temporary_name_identifies_descriptor(
            directory_descriptor.get(), temporary.name, temporary.descriptor.get());
        return rolled_back;
      };
      if (!temporary_name_identifies_descriptor(directory_descriptor.get(), destination_name,
                                                temporary.descriptor.get()) ||
          (exchanged && (!displaced_snapshot.has_value() ||
                         !directory_entry_matches_snapshot(directory_descriptor.get(),
                                                           temporary.name, *displaced_snapshot)))) {
        throw std::runtime_error(
            "refusing to publish calibration output: fallback output identity changed; rollback "
            "refused");
      }
      if (exchanged) {
        if (const auto error =
                displaced_output_removal_error(directory_descriptor.get(), temporary.name,
                                               left_identity, right_identity, profile_identities);
            error.has_value()) {
          const bool rolled_back = rollback_fallback_if_unchanged();
          if (!rolled_back) {
            throw std::runtime_error("refusing to publish calibration output: " + *error +
                                     " and rollback failed");
          }
          throw std::runtime_error("refusing to publish calibration output: " + *error);
        }
      }
      validate_published_or_rollback([&] { return rollback_fallback_if_unchanged(); });
      if (exchanged && !unlink_directory_entry_if_unchanged(directory_descriptor.get(),
                                                            temporary.name, *displaced_snapshot)) {
        const bool rolled_back = rollback_fallback_if_unchanged();
        throw std::runtime_error(
            rolled_back ? "failed to remove unchanged displaced calibration output"
                        : "displaced calibration output changed before removal; rollback refused");
      }
      temporary_exists = false;
      published = true;
    }

    if (temporary_exists &&
        temporary_name_identifies_descriptor(directory_descriptor.get(), temporary.name,
                                             temporary.descriptor.get())) {
      if (::unlinkat(directory_descriptor.get(), temporary.name.c_str(), 0) != 0) {
        throw_file_error("failed to remove temporary calibration output", temporary_path, errno);
      }
    }
    temporary_exists = false;
    temporary.descriptor.close_checked(destination);
    if (::fsync(directory_descriptor.get()) != 0) {
      throw_file_error("failed to flush calibration output directory", parent, errno);
    }
  } catch (...) {
    if (publication_exists &&
        temporary_name_identifies_descriptor(directory_descriptor.get(), publication_name,
                                             temporary.descriptor.get())) {
      (void)::unlinkat(directory_descriptor.get(), publication_name.c_str(), 0);
    }
    if (temporary_exists &&
        temporary_name_identifies_descriptor(directory_descriptor.get(), temporary.name,
                                             temporary.descriptor.get())) {
      (void)::unlinkat(directory_descriptor.get(), temporary.name.c_str(), 0);
    }
    throw;
  }
#else
  (void)force_rename_fallback;
#if !defined(__APPLE__)
  (void)after_publish;
  (void)on_lock_contention;
#endif
  const auto left_identity = pin_posix_path_for_publication(left_input, "left video input");
  const auto right_identity = pin_posix_path_for_publication(right_input, "right video input");
  std::vector<PinnedPosixPath> profile_identities;
  profile_identities.reserve(lens_profiles.size());
  for (std::size_t index = 0; index < lens_profiles.size(); ++index) {
    profile_identities.push_back(pin_posix_path_for_publication(
        lens_profiles[index], index == 0 ? "left lens profile" : "right lens profile"));
  }
  static std::mutex publication_mutex;
  std::lock_guard publication_lock(publication_mutex);
#if defined(__APPLE__)
  const auto parent =
      destination.has_parent_path() ? destination.parent_path() : std::filesystem::path(".");
  const auto destination_name = destination.filename().string();
  if (destination_name.empty() || destination_name == "." || destination_name == "..") {
    throw std::runtime_error("calibration output path must name a file: " + destination.string());
  }
  auto output_directory = lock_posix_output_directory(destination, on_lock_contention);
  std::string temporary_name;
#endif
  int descriptor = -1;
  std::filesystem::path temporary;
  bool temporary_exists = false;
#if defined(__APPLE__)
  bool destination_exchanged = false;
  std::optional<PosixDirectoryEntrySnapshot> displaced_output_snapshot;
#else
  bool destination_link_exists = false;
#endif
  try {
#if defined(__APPLE__)
    temporary_name = create_exclusive_temporary_at(output_directory.get(), destination, descriptor);
    temporary = parent / temporary_name;
#else
    temporary = create_exclusive_temporary(destination, descriptor);
#endif
    temporary_exists = true;
    write_all(descriptor, contents, temporary);
    if (::fsync(descriptor) != 0) {
      throw_file_error("failed to flush temporary calibration output", temporary, errno);
    }
    if (before_publish) {
      before_publish();
    }
#if defined(__APPLE__)
    if (const auto error = validate_pinned_posix_output_identity_at(
            output_directory.get(), destination_name, left_identity, right_identity,
            profile_identities);
#else
    if (const auto error = reco::calibrate::validate_calibration_output_identity(
            left_input, right_input, destination, lens_profiles);
#endif
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }
    if (before_commit) {
      before_commit();
    }
    left_identity.verify_unchanged();
    right_identity.verify_unchanged();
    for (const auto& profile : profile_identities) {
      profile.verify_unchanged();
    }
#if defined(__APPLE__)
    if (const auto error = validate_pinned_posix_output_identity_at(
            output_directory.get(), destination_name, left_identity, right_identity,
            profile_identities);
        error.has_value()) {
      throw std::runtime_error("refusing to publish calibration output: " + *error);
    }
    if (!path_identifies_descriptor_at(output_directory.get(), temporary_name, descriptor)) {
#else
    if (!path_identifies_descriptor(temporary, descriptor)) {
#endif
      temporary_exists = false;
      throw std::runtime_error(
          "refusing to publish calibration output: temporary output identity changed");
    }
#if defined(__APPLE__)
    if (::renameatx_np(output_directory.get(), temporary_name.c_str(), output_directory.get(),
                       destination_name.c_str(), RENAME_EXCL) == 0) {
      temporary_exists = false;
      if (after_publish) {
        after_publish();
      }
      const auto rollback_new_output = [&] {
        if (!path_identifies_descriptor_at(output_directory.get(), destination_name, descriptor)) {
          return false;
        }
        const bool rolled_back =
            ::renameatx_np(output_directory.get(), destination_name.c_str(), output_directory.get(),
                           temporary_name.c_str(), RENAME_EXCL) == 0;
        temporary_exists = rolled_back && path_identifies_descriptor_at(output_directory.get(),
                                                                        temporary_name, descriptor);
        return rolled_back;
      };
      if (!path_identifies_descriptor_at(output_directory.get(), destination_name, descriptor)) {
        const bool rolled_back = rollback_new_output();
        throw std::runtime_error(
            rolled_back
                ? "refusing to publish calibration output: published output identity changed"
                : "refusing to publish calibration output: published output identity changed and "
                  "rollback failed");
      }
      try {
        if (const auto error = validate_pinned_posix_output_identity_at(
                output_directory.get(), destination_name, left_identity, right_identity,
                profile_identities);
            error.has_value()) {
          throw std::runtime_error("refusing to publish calibration output: " + *error);
        }
      } catch (...) {
        if (!rollback_new_output()) {
          throw std::runtime_error(
              "refusing to publish calibration output: post-publication validation failed and "
              "rollback failed");
        }
        throw;
      }
    } else if (errno == EEXIST) {
      displaced_output_snapshot =
          capture_posix_directory_entry_snapshot(output_directory.get(), destination_name);
      if (::renameatx_np(output_directory.get(), temporary_name.c_str(), output_directory.get(),
                         destination_name.c_str(), RENAME_SWAP) != 0) {
        throw_file_error("failed to exchange calibration output", destination, errno);
      }
      destination_exchanged = true;
      if (after_publish) {
        after_publish();
      }
      const auto rollback_exchange_if_unchanged = [&] {
        if (!path_identifies_descriptor_at(output_directory.get(), destination_name, descriptor) ||
            !posix_directory_entry_matches_snapshot(output_directory.get(), temporary_name,
                                                    *displaced_output_snapshot)) {
          return false;
        }
        const bool rolled_back =
            ::renameatx_np(output_directory.get(), temporary_name.c_str(), output_directory.get(),
                           destination_name.c_str(), RENAME_SWAP) == 0;
        destination_exchanged = !rolled_back;
        temporary_exists = rolled_back && path_identifies_descriptor_at(output_directory.get(),
                                                                        temporary_name, descriptor);
        return rolled_back;
      };
      if (!path_identifies_descriptor_at(output_directory.get(), destination_name, descriptor) ||
          !posix_directory_entry_matches_snapshot(output_directory.get(), temporary_name,
                                                  *displaced_output_snapshot)) {
        throw std::runtime_error(
            "refusing to publish calibration output: exchanged output identity changed; rollback "
            "refused");
      }
      if (const auto error = displaced_posix_output_removal_error_at(
              output_directory.get(), temporary_name, left_identity, right_identity,
              profile_identities);
          error.has_value()) {
        const bool rolled_back = rollback_exchange_if_unchanged();
        throw std::runtime_error("refusing to publish calibration output: " + *error +
                                 (rolled_back ? "" : " and rollback failed"));
      }
      try {
        if (const auto error = validate_pinned_posix_output_identity_at(
                output_directory.get(), destination_name, left_identity, right_identity,
                profile_identities);
            error.has_value()) {
          throw std::runtime_error("refusing to publish calibration output: " + *error);
        }
      } catch (...) {
        if (!rollback_exchange_if_unchanged()) {
          throw std::runtime_error(
              "refusing to publish calibration output: post-publication validation failed and "
              "rollback failed");
        }
        throw;
      }
      if (!unlink_posix_directory_entry_if_unchanged(output_directory.get(), temporary_name,
                                                     *displaced_output_snapshot)) {
        const bool rolled_back = rollback_exchange_if_unchanged();
        throw std::runtime_error(
            rolled_back ? "failed to remove unchanged displaced calibration output"
                        : "displaced calibration output changed before removal; rollback refused");
      }
      temporary_exists = false;
      destination_exchanged = false;
      displaced_output_snapshot.reset();
    } else {
      throw_file_error("failed to publish calibration output", destination, errno);
    }
#else
    if (::link(temporary.c_str(), destination.c_str()) != 0) {
      if (errno == EEXIST) {
        throw std::runtime_error(
            "atomic calibration output replacement is unavailable on this POSIX platform");
      }
      throw_file_error("failed to publish calibration output", destination, errno);
    }
    destination_link_exists = true;
    if (!path_identifies_descriptor(destination, descriptor)) {
      throw std::runtime_error(
          "refusing to publish calibration output: published output identity changed");
    }
    if (::unlink(temporary.c_str()) != 0) {
      throw_file_error("failed to remove temporary calibration output", temporary, errno);
    }
    temporary_exists = false;
    destination_link_exists = false;
#endif
    const auto close_result = ::close(descriptor);
    descriptor = -1;
    if (close_result != 0) {
      throw_file_error("failed to close calibration output", destination, errno);
    }
  } catch (...) {
#if defined(__APPLE__)
    if (destination_exchanged) {
      if (displaced_output_snapshot.has_value() &&
          path_identifies_descriptor_at(output_directory.get(), destination_name, descriptor) &&
          posix_directory_entry_matches_snapshot(output_directory.get(), temporary_name,
                                                 *displaced_output_snapshot) &&
          ::renameatx_np(output_directory.get(), temporary_name.c_str(), output_directory.get(),
                         destination_name.c_str(), RENAME_SWAP) == 0) {
        destination_exchanged = false;
        temporary_exists =
            path_identifies_descriptor_at(output_directory.get(), temporary_name, descriptor);
      }
    }
    if (temporary_exists &&
        path_identifies_descriptor_at(output_directory.get(), temporary_name, descriptor)) {
      (void)::unlinkat(output_directory.get(), temporary_name.c_str(), 0);
    }
#else
    if (destination_link_exists && path_identifies_descriptor(destination, descriptor)) {
      (void)::unlink(destination.c_str());
    }
    if (temporary_exists && path_identifies_descriptor(temporary, descriptor)) {
      (void)::unlink(temporary.c_str());
    }
#endif
    if (descriptor >= 0) {
      ::close(descriptor);
    }
    throw;
  }
#endif
}

void write_calibration_result(const reco::calibrate::CalibrationResult& result,
                              const reco::calibrate::GpuCalibrationRequest& request,
                              const std::function<void()>& before_publish = {}) {
  const auto json = reco::core::calibration_to_json(result.calibration);
  const auto reparsed = reco::core::parse_match_calibration_json(json);
  if (!reparsed.has_value() || !reparsed->validate().empty()) {
    throw std::runtime_error("calibration result failed serialization validation");
  }

  const auto& left_path =
      request.left.retained_path.has_value() ? *request.left.retained_path : request.left.path;
  const auto& right_path =
      request.right.retained_path.has_value() ? *request.right.retained_path : request.right.path;
  std::vector<std::filesystem::path> profiles;
  if (request.left.lens_profile.has_value()) {
    profiles.emplace_back(*request.left.lens_profile);
  }
  if (request.right.lens_profile.has_value()) {
    profiles.emplace_back(*request.right.lens_profile);
  }
  detail::write_calibration_json_atomically(json, request.output, left_path, right_path,
                                            before_publish, profiles);
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

void write_calibration_json_atomically(
    std::string_view json, const std::filesystem::path& destination,
    const std::filesystem::path& left_input, const std::filesystem::path& right_input,
    const std::function<void()>& before_publish,
    std::span<const std::filesystem::path> lens_profiles,
    const std::function<void()>& before_commit, bool force_rename_fallback,
    const std::function<void()>& after_publish, const std::function<void()>& on_lock_contention) {
  write_calibration_json_atomically_impl(json, destination, left_input, right_input, before_publish,
                                         lens_profiles, before_commit, force_rename_fallback,
                                         after_publish, on_lock_contention);
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
      const auto left_identity = pin_input_identity(request.left.path, "left video input");
      const auto right_identity = pin_input_identity(request.right.path, "right video input");
      std::optional<PinnedFileIdentity> left_profile_identity;
      std::optional<PinnedFileIdentity> right_profile_identity;
      if (request.left.lens_profile.has_value()) {
        left_profile_identity.emplace(
            pin_input_identity(*request.left.lens_profile, "left lens profile"));
      }
      if (request.right.lens_profile.has_value()) {
        right_profile_identity.emplace(
            pin_input_identity(*request.right.lens_profile, "right lens profile"));
      }
      auto pinned_request = request;
      pinned_request.left.retained_path = left_identity.retained_path().string();
      pinned_request.right.retained_path = right_identity.retained_path().string();
      pinned_request.left.expected_identity = left_identity.portable_identity();
      pinned_request.right.expected_identity = right_identity.portable_identity();
      if (left_profile_identity.has_value()) {
        pinned_request.left.lens_profile = left_profile_identity->retained_path().string();
        pinned_request.left.lens_profile_expected_identity =
            left_profile_identity->portable_identity();
      }
      if (right_profile_identity.has_value()) {
        pinned_request.right.lens_profile = right_profile_identity->retained_path().string();
        pinned_request.right.lens_profile_expected_identity =
            right_profile_identity->portable_identity();
      }
      auto result = reco::calibrate::run_gpu_calibration(pinned_request, backends);
      if (result.left_lens_profile.has_value()) {
        result.left_lens_profile->path = request.left.lens_profile;
      }
      if (result.right_lens_profile.has_value()) {
        result.right_lens_profile->path = request.right.lens_profile.has_value()
                                              ? request.right.lens_profile
                                              : request.left.lens_profile;
      }
      const auto verify_pinned_inputs = [&] {
        left_identity.verify_unchanged();
        right_identity.verify_unchanged();
        if (left_profile_identity.has_value()) {
          left_profile_identity->verify_unchanged();
        }
        if (right_profile_identity.has_value()) {
          right_profile_identity->verify_unchanged();
        }
      };
      verify_pinned_inputs();
      // Publish against the original user-visible entries while the descriptors used by the
      // worker remain pinned. This preserves both symlink and target identities through commit.
      write_calibration_result(result, request, verify_pinned_inputs);
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
