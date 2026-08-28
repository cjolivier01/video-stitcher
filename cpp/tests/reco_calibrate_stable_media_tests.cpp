#include "stable_media_file.hpp"

#include "reco/calibrate/lens_database.hpp"
#include "reco/calibrate/pipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using reco::calibrate::CalibrationExecutionError;
using reco::calibrate::detail::StableLensProfileFile;
using reco::calibrate::detail::StableMediaFile;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void write_file(const std::filesystem::path& path, std::string_view contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) {
    throw std::runtime_error("failed to write stable-media test fixture");
  }
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() / ("reco_stable_media_" + std::to_string(nonce));
    if (!std::filesystem::create_directory(path_)) {
      throw std::runtime_error("failed to create stable-media test directory");
    }
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

#if defined(__linux__)
reco::calibrate::CalibrationFileIdentity file_identity(const std::filesystem::path& path) {
  struct stat metadata{};
  if (::stat(path.c_str(), &metadata) != 0) {
    throw std::runtime_error("failed to inspect stable-media identity fixture");
  }
  return {
      .device = static_cast<std::uint64_t>(metadata.st_dev),
      .inode = static_cast<std::uint64_t>(metadata.st_ino),
      .size = static_cast<std::uint64_t>(metadata.st_size),
      .mode = static_cast<std::uint32_t>(metadata.st_mode),
      .modified_seconds = static_cast<std::int64_t>(metadata.st_mtim.tv_sec),
      .modified_nanoseconds = static_cast<std::int64_t>(metadata.st_mtim.tv_nsec),
      .changed_seconds = static_cast<std::int64_t>(metadata.st_ctim.tv_sec),
      .changed_nanoseconds = static_cast<std::int64_t>(metadata.st_ctim.tv_nsec),
  };
}
#endif

void retained_handle_survives_path_replacement() {
#if defined(__linux__)
  TemporaryDirectory directory;
  const auto input = directory.path() / "input.mp4";
  const auto moved = directory.path() / "original.mp4";
  write_file(input, "original media bytes");
  StableMediaFile stable(input);

  std::filesystem::rename(input, moved);
  write_file(input, "replacement media bytes");
  expect_true(read_file(stable.decode_path()) == "original media bytes",
              "retained decode path remains bound to the opened inode");
  bool replacement_rejected = false;
  try {
    stable.verify_unchanged();
  } catch (const CalibrationExecutionError& error) {
    replacement_rejected = std::string_view(error.what()).find("changed") != std::string_view::npos;
  }
  expect_true(replacement_rejected, "path replacement changes ctime and fails closed");
#endif
}

void same_size_mtime_restored_mutation_is_rejected() {
#if defined(__linux__)
  TemporaryDirectory directory;
  const auto input = directory.path() / "input.mp4";
  constexpr std::string_view original = "original media bytes";
  constexpr std::string_view mutation = "mutated! media bytes";
  static_assert(original.size() == mutation.size());
  write_file(input, original);
  const auto original_mtime = std::filesystem::last_write_time(input);
  StableMediaFile stable(input);

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  write_file(input, mutation);
  std::filesystem::last_write_time(input, original_mtime);

  expect_true(read_file(stable.decode_path()) == mutation,
              "same-size fixture mutates the retained inode");
  bool mutation_rejected = false;
  try {
    stable.verify_unchanged();
  } catch (const CalibrationExecutionError& error) {
    mutation_rejected = std::string_view(error.what()).find("changed") != std::string_view::npos;
  }
  expect_true(mutation_rejected, "ctime detects a same-size mutation with restored mtime");
#endif
}

void invalid_inputs_fail_before_gstreamer() {
#if defined(__linux__)
  TemporaryDirectory directory;
  bool directory_rejected = false;
  try {
    StableMediaFile stable(directory.path());
  } catch (const CalibrationExecutionError& error) {
    directory_rejected =
        std::string_view(error.what()).find("regular file") != std::string_view::npos;
  }
  expect_true(directory_rejected, "directory input is rejected");

  const auto empty = directory.path() / "empty.mp4";
  write_file(empty, "");
  bool empty_rejected = false;
  try {
    StableMediaFile stable(empty);
  } catch (const CalibrationExecutionError& error) {
    empty_rejected =
        std::string_view(error.what()).find("must not be empty") != std::string_view::npos;
  }
  expect_true(empty_rejected, "empty media input is rejected");
#endif
}

void expected_identity_mismatch_fails_before_processing() {
#if defined(__linux__)
  TemporaryDirectory directory;
  const auto input = directory.path() / "input.mp4";
  write_file(input, "pinned calibration media");
  auto expected = file_identity(input);
  ++expected.inode;
  bool mismatch_rejected = false;
  try {
    StableMediaFile stable(input, expected);
  } catch (const CalibrationExecutionError& error) {
    mismatch_rejected = std::string_view(error.what()).find("changed before isolated processing") !=
                        std::string_view::npos;
  }
  expect_true(mismatch_rejected, "initial media identity mismatch fails before processing");
#endif
}

void descriptor_pinned_identity_does_not_require_worker_flock() {
#if defined(__linux__)
  TemporaryDirectory directory;
  const auto input = directory.path() / "input.mp4";
  write_file(input, "descriptor-pinned calibration media");
  const int lock_descriptor = ::open(input.c_str(), O_RDONLY | O_CLOEXEC);
  expect_true(lock_descriptor >= 0, "exclusive lock fixture opens");
  if (lock_descriptor < 0) {
    return;
  }
  expect_true(::flock(lock_descriptor, LOCK_EX | LOCK_NB) == 0,
              "exclusive lock fixture acquires its lock");
  try {
    StableMediaFile stable(input, file_identity(input));
    expect_true(read_file(stable.decode_path()) == "descriptor-pinned calibration media",
                "descriptor-pinned media opens without a worker flock syscall");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: descriptor-pinned media rejected by advisory lock: " << error.what()
              << '\n';
    ++failures;
  }
  (void)::close(lock_descriptor);
#endif
}

void retained_lens_profile_survives_path_replacement() {
#if defined(__linux__)
  TemporaryDirectory directory;
  const auto profile = directory.path() / "selected-profile.json";
  const auto selected = directory.path() / "selected-profile-original.json";
  write_file(profile, R"json({
    "width": 1920,
    "height": 1080,
    "fx": 700.0,
    "fy": 701.0,
    "cx": 960.0,
    "cy": 540.0,
    "d": [0.01, -0.02, 0.003, -0.004]
  })json");
  StableLensProfileFile stable(profile);

  std::filesystem::rename(profile, selected);
  write_file(profile, R"json({
    "width": 1920,
    "height": 1080,
    "fx": 1700.0,
    "fy": 1701.0,
    "cx": 960.0,
    "cy": 540.0,
    "d": [0.01, -0.02, 0.003, -0.004]
  })json");

  const auto retained = reco::calibrate::load_lens_from_file(stable.retained_path().string());
  expect_true(retained.fx == 700.0,
              "retained lens profile remains bound to the selected profile identity");
  bool replacement_rejected = false;
  try {
    stable.verify_unchanged();
  } catch (const CalibrationExecutionError& error) {
    replacement_rejected = std::string_view(error.what()).find("changed") != std::string_view::npos;
  }
  expect_true(replacement_rejected, "lens profile path replacement fails closed");
#endif
}

} // namespace

int main() {
  retained_handle_survives_path_replacement();
  same_size_mtime_restored_mutation_is_rejected();
  invalid_inputs_fail_before_gstreamer();
  expected_identity_mismatch_fails_before_processing();
  descriptor_pinned_identity_does_not_require_worker_flock();
  retained_lens_profile_survives_path_replacement();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
