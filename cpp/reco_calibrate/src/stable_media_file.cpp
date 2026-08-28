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

} // namespace

struct StableFileState {
  StableFileState(const std::filesystem::path& path, const char* file_kind,
                  bool require_original_path_identity)
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
    if (::flock(descriptor, LOCK_SH | LOCK_NB) != 0) {
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
  explicit Impl(const std::filesystem::path& path)
      : StableFileState(path, "calibration video", false) {}
};

struct StableLensProfileFile::Impl : StableFileState {
  explicit Impl(const std::filesystem::path& path)
      : StableFileState(path, "calibration lens profile", true) {}
};

#else

struct StableMediaFile::Impl {
  explicit Impl(const std::filesystem::path&) {
    throw CalibrationExecutionError(
        "stable file-backed GPU calibration is currently supported only on Linux");
  }
  void verify_unchanged() const {}
  std::filesystem::path retained_path;
};

struct StableLensProfileFile::Impl {
  explicit Impl(const std::filesystem::path&) {
    throw CalibrationExecutionError(
        "stable file-backed GPU calibration is currently supported only on Linux");
  }
  void verify_unchanged() const {}
  std::filesystem::path retained_path;
};

#endif

StableMediaFile::StableMediaFile(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}

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

StableLensProfileFile::StableLensProfileFile(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}

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
