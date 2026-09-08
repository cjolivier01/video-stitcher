#include "stable_media_file.hpp"

#include "reco/calibrate/pipeline.hpp"

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace reco::calibrate::detail {

#if defined(__linux__)
namespace {

[[noreturn]] void throw_file_error(const std::filesystem::path& path, const char* operation,
                                   const char* kind) {
  throw CalibrationExecutionError(std::string(operation) + " " + kind + " " + path.string() + ": " +
                                  std::strerror(errno));
}

bool same_timestamp(const timespec& left, const timespec& right) {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

CalibrationFileIdentity portable_identity(const struct stat& value) {
  return {
      .device = static_cast<std::uint64_t>(value.st_dev),
      .inode = static_cast<std::uint64_t>(value.st_ino),
      .size = static_cast<std::uint64_t>(value.st_size),
      .mode = static_cast<std::uint32_t>(value.st_mode),
      .modified_seconds = static_cast<std::int64_t>(value.st_mtim.tv_sec),
      .modified_nanoseconds = static_cast<std::int64_t>(value.st_mtim.tv_nsec),
      .changed_seconds = static_cast<std::int64_t>(value.st_ctim.tv_sec),
      .changed_nanoseconds = static_cast<std::int64_t>(value.st_ctim.tv_nsec),
  };
}

} // namespace

struct StableFileState {
  StableFileState(const std::filesystem::path& path, const char* file_kind,
                  bool require_original_path_identity,
                  const std::optional<CalibrationFileIdentity>& expected_identity)
      : original_path(path), kind(file_kind), verify_path_identity(require_original_path_identity) {
    descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      throw_file_error(path, "cannot open", kind);
    }
    if (::fstat(descriptor, &identity) != 0) {
      const auto saved_error = errno;
      ::close(descriptor);
      descriptor = -1;
      errno = saved_error;
      throw_file_error(path, "cannot inspect", kind);
    }
    if (!S_ISREG(identity.st_mode)) {
      ::close(descriptor);
      descriptor = -1;
      throw CalibrationExecutionError(std::string(kind) +
                                      " must be a regular file: " + path.string());
    }
    if (identity.st_size <= 0) {
      ::close(descriptor);
      descriptor = -1;
      throw CalibrationExecutionError(std::string(kind) + " must not be empty: " + path.string());
    }
    if (expected_identity.has_value() && portable_identity(identity) != *expected_identity) {
      ::close(descriptor);
      descriptor = -1;
      throw CalibrationExecutionError(
          std::string(kind) + " changed before isolated processing began: " + path.string());
    }
    // Isolated workers receive descriptor-pinned identities and deny flock in seccomp. The
    // supervisor and CLI revalidate those identities across processing and publication.
    if (!expected_identity.has_value() && ::flock(descriptor, LOCK_SH | LOCK_NB) != 0) {
      const auto saved_error = errno;
      ::close(descriptor);
      descriptor = -1;
      errno = saved_error;
      throw_file_error(path, "cannot acquire a shared lock for", kind);
    }
    retained_path = std::filesystem::path("/proc") / std::to_string(::getpid()) / "fd" /
                    std::to_string(descriptor);
  }

  ~StableFileState() {
    if (descriptor >= 0) {
      (void)::close(descriptor);
    }
  }

  void verify_unchanged() const {
    struct stat current{};
    if (::fstat(descriptor, &current) != 0) {
      throw_file_error(original_path, "cannot re-inspect", kind);
    }
    if (current.st_dev != identity.st_dev || current.st_ino != identity.st_ino ||
        current.st_mode != identity.st_mode || current.st_size != identity.st_size ||
        !same_timestamp(current.st_mtim, identity.st_mtim) ||
        !same_timestamp(current.st_ctim, identity.st_ctim)) {
      throw CalibrationExecutionError(
          std::string(kind) + " changed while it was being processed: " + original_path.string());
    }
    if (verify_path_identity) {
      struct stat named{};
      if (::stat(original_path.c_str(), &named) != 0 || named.st_dev != identity.st_dev ||
          named.st_ino != identity.st_ino) {
        throw CalibrationExecutionError(
            std::string(kind) +
            " path changed while it was being processed: " + original_path.string());
      }
    }
  }

  std::filesystem::path original_path;
  std::filesystem::path retained_path;
  const char* kind;
  bool verify_path_identity;
  int descriptor = -1;
  struct stat identity{};
};

struct StableMediaFile::Impl : StableFileState {
  Impl(const std::filesystem::path& path,
       const std::optional<CalibrationFileIdentity>& expected_identity)
      : StableFileState(path, "calibration video", false, expected_identity) {}
};

struct StableLensProfileFile::Impl : StableFileState {
  Impl(const std::filesystem::path& path,
       const std::optional<CalibrationFileIdentity>& expected_identity)
      : StableFileState(path, "calibration lens profile", true, expected_identity) {}
};

#else

struct StableMediaFile::Impl {
  Impl(const std::filesystem::path&, const std::optional<CalibrationFileIdentity>&) {
    throw CalibrationExecutionError(
        "stable file-backed GPU calibration is currently supported only on Linux");
  }
  void verify_unchanged() const {}
  std::filesystem::path retained_path;
};

struct StableLensProfileFile::Impl {
  Impl(const std::filesystem::path&, const std::optional<CalibrationFileIdentity>&) {
    throw CalibrationExecutionError(
        "stable file-backed GPU calibration is currently supported only on Linux");
  }
  void verify_unchanged() const {}
  std::filesystem::path retained_path;
};

#endif

StableMediaFile::StableMediaFile(const std::filesystem::path& path,
                                 std::optional<CalibrationFileIdentity> expected_identity)
    : impl_(std::make_unique<Impl>(path, expected_identity)) {}

StableMediaFile::~StableMediaFile() = default;
StableMediaFile::StableMediaFile(StableMediaFile&&) noexcept = default;
StableMediaFile& StableMediaFile::operator=(StableMediaFile&&) noexcept = default;

const std::filesystem::path& StableMediaFile::decode_path() const {
  if (impl_ == nullptr) {
    throw CalibrationExecutionError("cannot use a moved-from stable calibration video");
  }
  return impl_->retained_path;
}

void StableMediaFile::verify_unchanged() const {
  if (impl_ == nullptr) {
    throw CalibrationExecutionError("cannot use a moved-from stable calibration video");
  }
  impl_->verify_unchanged();
}

StableLensProfileFile::StableLensProfileFile(
    const std::filesystem::path& path, std::optional<CalibrationFileIdentity> expected_identity)
    : impl_(std::make_unique<Impl>(path, expected_identity)) {}

StableLensProfileFile::~StableLensProfileFile() = default;
StableLensProfileFile::StableLensProfileFile(StableLensProfileFile&&) noexcept = default;
StableLensProfileFile& StableLensProfileFile::operator=(StableLensProfileFile&&) noexcept = default;

const std::filesystem::path& StableLensProfileFile::retained_path() const {
  if (impl_ == nullptr) {
    throw CalibrationExecutionError("cannot use a moved-from stable calibration lens profile");
  }
  return impl_->retained_path;
}

void StableLensProfileFile::verify_unchanged() const {
  if (impl_ == nullptr) {
    throw CalibrationExecutionError("cannot use a moved-from stable calibration lens profile");
  }
  impl_->verify_unchanged();
}

} // namespace reco::calibrate::detail
