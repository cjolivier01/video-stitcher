#include "reco/io/stable_media_file.hpp"

#include "reco/core/path.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace reco::io {
namespace {

[[noreturn]] void throw_file_error(std::string_view operation, const std::filesystem::path& path,
                                   int error) {
  throw std::system_error(error, std::system_category(),
                          std::string(operation) + " " + core::path_to_utf8(path));
}

#if defined(_WIN32)

class UniqueHandle final {
public:
  explicit UniqueHandle(HANDLE value = INVALID_HANDLE_VALUE) : value_(value) {}
  ~UniqueHandle() {
    if (value_ != INVALID_HANDLE_VALUE) {
      (void)CloseHandle(value_);
    }
  }
  UniqueHandle(const UniqueHandle&) = delete;
  UniqueHandle& operator=(const UniqueHandle&) = delete;
  UniqueHandle(UniqueHandle&& other) noexcept
      : value_(std::exchange(other.value_, INVALID_HANDLE_VALUE)) {}
  UniqueHandle& operator=(UniqueHandle&& other) noexcept {
    if (this != &other) {
      if (value_ != INVALID_HANDLE_VALUE) {
        (void)CloseHandle(value_);
      }
      value_ = std::exchange(other.value_, INVALID_HANDLE_VALUE);
    }
    return *this;
  }
  [[nodiscard]] HANDLE get() const noexcept { return value_; }
  [[nodiscard]] HANDLE release() noexcept { return std::exchange(value_, INVALID_HANDLE_VALUE); }

private:
  HANDLE value_;
};

struct WindowsFileSnapshot {
  BY_HANDLE_FILE_INFORMATION identity{};
  FILE_BASIC_INFO basic{};
};

bool same_windows_object(const WindowsFileSnapshot& left, const WindowsFileSnapshot& right) {
  return left.identity.dwVolumeSerialNumber == right.identity.dwVolumeSerialNumber &&
         left.identity.nFileIndexHigh == right.identity.nFileIndexHigh &&
         left.identity.nFileIndexLow == right.identity.nFileIndexLow;
}

bool same_windows_snapshot(const WindowsFileSnapshot& left, const WindowsFileSnapshot& right) {
  return same_windows_object(left, right) &&
         left.identity.dwFileAttributes == right.identity.dwFileAttributes &&
         left.identity.nFileSizeHigh == right.identity.nFileSizeHigh &&
         left.identity.nFileSizeLow == right.identity.nFileSizeLow &&
         left.basic.LastWriteTime.QuadPart == right.basic.LastWriteTime.QuadPart &&
         left.basic.ChangeTime.QuadPart == right.basic.ChangeTime.QuadPart;
}

WindowsFileSnapshot inspect_windows_handle(HANDLE handle, const std::filesystem::path& path,
                                           std::string_view operation) {
  WindowsFileSnapshot snapshot;
  if (GetFileInformationByHandle(handle, &snapshot.identity) == 0 ||
      GetFileInformationByHandleEx(handle, FileBasicInfo, &snapshot.basic,
                                   sizeof(snapshot.basic)) == 0) {
    throw_file_error(operation, path, static_cast<int>(GetLastError()));
  }
  return snapshot;
}

UniqueHandle open_windows_target(const std::filesystem::path& path) {
  const HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ | FILE_READ_ATTRIBUTES,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    throw_file_error("cannot open stable media input", path, static_cast<int>(GetLastError()));
  }
  return UniqueHandle(handle);
}

UniqueHandle open_windows_entry(const std::filesystem::path& path) {
  const HANDLE handle = CreateFileW(
      path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    throw_file_error("cannot retain stable media pathname", path, static_cast<int>(GetLastError()));
  }
  return UniqueHandle(handle);
}

int descriptor_from_windows_handle(UniqueHandle handle, const std::filesystem::path& path) {
  const auto raw = reinterpret_cast<std::intptr_t>(handle.release());
  const int descriptor = _open_osfhandle(raw, _O_RDONLY | _O_BINARY);
  if (descriptor < 0) {
    const int saved_error = errno;
    (void)CloseHandle(reinterpret_cast<HANDLE>(raw));
    throw_file_error("cannot create stable media descriptor", path, saved_error);
  }
  return descriptor;
}

#else

struct PosixTimestamp {
  std::int64_t seconds = 0;
  std::int64_t nanoseconds = 0;
};

bool operator==(const PosixTimestamp& left, const PosixTimestamp& right) {
  return left.seconds == right.seconds && left.nanoseconds == right.nanoseconds;
}

PosixTimestamp modified_time(const struct stat& value) {
#if defined(__APPLE__)
  return {.seconds = value.st_mtimespec.tv_sec, .nanoseconds = value.st_mtimespec.tv_nsec};
#else
  return {.seconds = value.st_mtim.tv_sec, .nanoseconds = value.st_mtim.tv_nsec};
#endif
}

PosixTimestamp changed_time(const struct stat& value) {
#if defined(__APPLE__)
  return {.seconds = value.st_ctimespec.tv_sec, .nanoseconds = value.st_ctimespec.tv_nsec};
#else
  return {.seconds = value.st_ctim.tv_sec, .nanoseconds = value.st_ctim.tv_nsec};
#endif
}

bool same_posix_object(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

bool same_posix_snapshot(const struct stat& left, const struct stat& right) {
  return same_posix_object(left, right) && left.st_mode == right.st_mode &&
         left.st_size == right.st_size && modified_time(left) == modified_time(right) &&
         changed_time(left) == changed_time(right);
}

int open_posix_target(const std::filesystem::path& path) {
  int descriptor = -1;
  do {
    descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  } while (descriptor < 0 && errno == EINTR);
  if (descriptor < 0) {
    throw_file_error("cannot open stable media input", path, errno);
  }
  return descriptor;
}

#endif

} // namespace

struct StableMediaFile::Impl {
  std::filesystem::path path;
  bool verify_named_path = false;
  int descriptor = -1;
#if defined(_WIN32)
  UniqueHandle entry;
  WindowsFileSnapshot target_identity;
  WindowsFileSnapshot entry_identity;
#else
  struct stat target_identity{};
  struct stat entry_identity{};
#endif

  ~Impl() {
    if (descriptor >= 0) {
#if defined(_WIN32)
      (void)_close(descriptor);
#else
      (void)::close(descriptor);
#endif
    }
  }
};

StableMediaFile::StableMediaFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

StableMediaFile::~StableMediaFile() = default;

std::shared_ptr<const StableMediaFile> StableMediaFile::open(const std::filesystem::path& path) {
  if (path.empty()) {
    throw std::invalid_argument("stable media input path is required");
  }
  auto impl = std::make_unique<Impl>();
  impl->path = path;
  impl->verify_named_path = true;
#if defined(_WIN32)
  auto target = open_windows_target(path);
  impl->target_identity = inspect_windows_handle(target.get(), path, "cannot inspect media input");
  if ((impl->target_identity.identity.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
      (impl->target_identity.identity.nFileSizeHigh == 0 &&
       impl->target_identity.identity.nFileSizeLow == 0)) {
    throw std::runtime_error("stable media input must be a non-empty regular file: " +
                             core::path_to_utf8(path));
  }
  impl->entry = open_windows_entry(path);
  impl->entry_identity =
      inspect_windows_handle(impl->entry.get(), path, "cannot inspect media pathname");
  impl->descriptor = descriptor_from_windows_handle(std::move(target), path);
#else
  impl->descriptor = open_posix_target(path);
  if (::fstat(impl->descriptor, &impl->target_identity) != 0) {
    throw_file_error("cannot inspect stable media input", path, errno);
  }
  if (::lstat(path.c_str(), &impl->entry_identity) != 0) {
    throw_file_error("cannot inspect stable media pathname", path, errno);
  }
  if (!S_ISREG(impl->target_identity.st_mode) || impl->target_identity.st_size <= 0) {
    throw std::runtime_error("stable media input must be a non-empty regular file: " +
                             core::path_to_utf8(path));
  }
#endif
  return std::shared_ptr<const StableMediaFile>(new StableMediaFile(std::move(impl)));
}

std::shared_ptr<const StableMediaFile> StableMediaFile::open_cursor() const {
  if (!impl_ || impl_->descriptor < 0) {
    throw std::runtime_error("stable media input is closed");
  }
  auto cursor = std::make_unique<Impl>();
  cursor->path = impl_->path;
#if defined(_WIN32)
  auto target = open_windows_target(impl_->path);
  cursor->target_identity =
      inspect_windows_handle(target.get(), impl_->path, "cannot inspect media cursor");
  if (!same_windows_snapshot(cursor->target_identity, impl_->target_identity)) {
    throw std::runtime_error("stable media input pathname selected a different file: " +
                             core::path_to_utf8(impl_->path));
  }
  cursor->descriptor = descriptor_from_windows_handle(std::move(target), impl_->path);
#else
  cursor->descriptor = open_posix_target(impl_->path);
  if (::fstat(cursor->descriptor, &cursor->target_identity) != 0) {
    throw_file_error("cannot inspect stable media cursor", impl_->path, errno);
  }
  if (!same_posix_snapshot(cursor->target_identity, impl_->target_identity)) {
    throw std::runtime_error("stable media input pathname selected a different file: " +
                             core::path_to_utf8(impl_->path));
  }
#endif
  return std::shared_ptr<const StableMediaFile>(new StableMediaFile(std::move(cursor)));
}

const std::filesystem::path& StableMediaFile::display_path() const noexcept { return impl_->path; }

int StableMediaFile::descriptor() const {
  if (!impl_ || impl_->descriptor < 0) {
    throw std::runtime_error("stable media input descriptor is closed");
  }
  return impl_->descriptor;
}

std::string StableMediaFile::read_all(std::size_t maximum_bytes) const {
  if (maximum_bytes == 0) {
    throw std::invalid_argument("stable media read limit must be non-zero");
  }
  const int source = descriptor();
#if defined(_WIN32)
  const auto length = _filelengthi64(source);
  if (length <= 0 || static_cast<std::uint64_t>(length) > maximum_bytes) {
    throw std::runtime_error("stable media input exceeds its read limit: " +
                             core::path_to_utf8(impl_->path));
  }
  if (_lseeki64(source, 0, SEEK_SET) < 0) {
    throw_file_error("cannot seek stable media input", impl_->path, errno);
  }
  std::string contents(static_cast<std::size_t>(length), '\0');
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const auto amount = static_cast<unsigned int>(std::min<std::size_t>(
        contents.size() - offset, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int read = _read(source, contents.data() + offset, amount);
    if (read <= 0) {
      throw_file_error("cannot read stable media input", impl_->path, read == 0 ? EIO : errno);
    }
    offset += static_cast<std::size_t>(read);
  }
#else
  struct stat identity{};
  if (::fstat(source, &identity) != 0) {
    throw_file_error("cannot inspect stable media input", impl_->path, errno);
  }
  if (identity.st_size <= 0 || static_cast<std::uint64_t>(identity.st_size) > maximum_bytes) {
    throw std::runtime_error("stable media input exceeds its read limit: " +
                             core::path_to_utf8(impl_->path));
  }
  std::string contents(static_cast<std::size_t>(identity.st_size), '\0');
  std::size_t offset = 0;
  while (offset < contents.size()) {
    ssize_t read = -1;
    do {
      read = ::pread(source, contents.data() + offset, contents.size() - offset,
                     static_cast<off_t>(offset));
    } while (read < 0 && errno == EINTR);
    if (read <= 0) {
      throw_file_error("cannot read stable media input", impl_->path, read == 0 ? EIO : errno);
    }
    offset += static_cast<std::size_t>(read);
  }
#endif
  verify_unchanged();
  return contents;
}

void StableMediaFile::rewind() const {
  const int source = descriptor();
#if defined(_WIN32)
  if (_lseeki64(source, 0, SEEK_SET) < 0) {
    throw_file_error("cannot rewind stable media input", impl_->path, errno);
  }
#else
  off_t result = -1;
  do {
    result = ::lseek(source, 0, SEEK_SET);
  } while (result < 0 && errno == EINTR);
  if (result < 0) {
    throw_file_error("cannot rewind stable media input", impl_->path, errno);
  }
#endif
}

void StableMediaFile::verify_unchanged() const {
  const int source = descriptor();
#if defined(_WIN32)
  const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(source));
  const auto descriptor_identity =
      inspect_windows_handle(handle, impl_->path, "cannot re-inspect media input");
  if (!same_windows_snapshot(descriptor_identity, impl_->target_identity)) {
    throw std::runtime_error("stable media input changed while it was retained: " +
                             core::path_to_utf8(impl_->path));
  }
  if (!impl_->verify_named_path) {
    return;
  }
  auto current_target = open_windows_target(impl_->path);
  auto current_entry = open_windows_entry(impl_->path);
  const auto target_identity =
      inspect_windows_handle(current_target.get(), impl_->path, "cannot re-inspect media path");
  const auto entry_identity =
      inspect_windows_handle(current_entry.get(), impl_->path, "cannot re-inspect media pathname");
  if (!same_windows_snapshot(target_identity, impl_->target_identity) ||
      !same_windows_snapshot(entry_identity, impl_->entry_identity)) {
    throw std::runtime_error("stable media input pathname changed while it was retained: " +
                             core::path_to_utf8(impl_->path));
  }
#else
  struct stat descriptor_identity{};
  if (::fstat(source, &descriptor_identity) != 0) {
    throw_file_error("cannot re-inspect stable media input", impl_->path, errno);
  }
  if (!same_posix_snapshot(descriptor_identity, impl_->target_identity)) {
    throw std::runtime_error("stable media input changed while it was retained: " +
                             core::path_to_utf8(impl_->path));
  }
  if (!impl_->verify_named_path) {
    return;
  }
  struct stat current_target{};
  struct stat current_entry{};
  if (::stat(impl_->path.c_str(), &current_target) != 0 ||
      ::lstat(impl_->path.c_str(), &current_entry) != 0) {
    throw_file_error("cannot re-inspect stable media pathname", impl_->path, errno);
  }
  if (!same_posix_snapshot(current_target, impl_->target_identity) ||
      !same_posix_snapshot(current_entry, impl_->entry_identity)) {
    throw std::runtime_error("stable media input pathname changed while it was retained: " +
                             core::path_to_utf8(impl_->path));
  }
#endif
}

std::shared_ptr<const StableMediaFile>
detail::adopt_stable_media_worker_descriptor(int descriptor, std::string display_path) {
  if (descriptor < 0) {
    throw std::invalid_argument("stable media worker descriptor is invalid");
  }
  auto impl = std::make_unique<StableMediaFile::Impl>();
  impl->path = core::path_from_utf8(display_path);
  impl->descriptor = descriptor;
#if defined(_WIN32)
  const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(descriptor));
  if (handle == INVALID_HANDLE_VALUE) {
    throw_file_error("cannot inspect inherited stable media input", impl->path, errno);
  }
  impl->target_identity =
      inspect_windows_handle(handle, impl->path, "cannot inspect inherited media input");
  if ((impl->target_identity.identity.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
      (impl->target_identity.identity.nFileSizeHigh == 0 &&
       impl->target_identity.identity.nFileSizeLow == 0)) {
    throw std::runtime_error("inherited stable media input is not a non-empty regular file");
  }
#else
  if (::fstat(descriptor, &impl->target_identity) != 0) {
    throw_file_error("cannot inspect inherited stable media input", impl->path, errno);
  }
  if (!S_ISREG(impl->target_identity.st_mode) || impl->target_identity.st_size <= 0) {
    throw std::runtime_error("inherited stable media input is not a non-empty regular file");
  }
#endif
  return std::shared_ptr<const StableMediaFile>(new StableMediaFile(std::move(impl)));
}

std::intptr_t detail::stable_media_native_handle(const StableMediaFile& file) {
#if defined(_WIN32)
  const auto handle = _get_osfhandle(file.descriptor());
  if (handle == -1) {
    throw_file_error("cannot access stable media native handle", file.display_path(), errno);
  }
  return handle;
#else
  return file.descriptor();
#endif
}

} // namespace reco::io
