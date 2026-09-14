#include "reco/core/calibration.hpp"
#include "reco/io/stable_media_file.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <system_error>
#endif

namespace {

int failures = 0;

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    const auto token = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    path_ = std::filesystem::temp_directory_path() / ("reco-stable-media-" + token);
    std::filesystem::create_directory(path_);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, std::string_view contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) {
    throw std::runtime_error("failed to write stable-media fixture");
  }
}

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_failure(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  }
}

template <typename Function>
void expect_value_or_fail_closed(Function&& function, std::string_view expected,
                                 std::string_view message) {
  try {
    expect(function() == expected, message);
  } catch (const std::exception& error) {
    expect(std::string_view(error.what()).find("changed while it was retained") !=
               std::string_view::npos,
           std::string(message) + " fails closed after metadata detects the swap");
  }
}

template <typename Function>
void expect_verified_or_fail_closed(Function&& function, std::string_view message) {
  try {
    function();
  } catch (const std::exception& error) {
    expect(std::string_view(error.what()).find("changed while it was retained") !=
               std::string_view::npos,
           std::string(message) + " reports only the retained-file mutation");
  }
}

void retained_cursor_survives_swap_and_restore() {
  TemporaryDirectory root;
  const auto selected = root.path() / "selected.mp4";
  const auto substitute = root.path() / "substitute.mp4";
  const auto moved = root.path() / "selected-retained.mp4";
  write_file(selected, "pinned-media-bytes");
  write_file(substitute, "substitute-content");

  const auto retained = reco::io::StableMediaFile::open(selected);
  const auto processing_cursor = retained->open_cursor();
  std::filesystem::rename(selected, moved);
  std::filesystem::rename(substitute, selected);

  expect_value_or_fail_closed(
      [&] { return processing_cursor->read_all(1024); }, "pinned-media-bytes",
      "processing cursor reads only the identity pinned before pathname substitution");
  expect_value_or_fail_closed(
      [&] { return retained->open_cursor()->read_all(1024); }, "pinned-media-bytes",
      "cursor acquisition uses retained authority during pathname substitution");
  expect_failure([&] { retained->verify_unchanged(); }, "changed while it was retained",
                 "retained owner detects a substituted pathname");

  std::filesystem::rename(selected, substitute);
  std::filesystem::rename(moved, selected);
  expect_verified_or_fail_closed([&] { retained->verify_unchanged(); },
                                 "restored pathname validation");
  expect_value_or_fail_closed([&] { return retained->read_all(1024); }, "pinned-media-bytes",
                              "restored pathname cannot select substitute contents");
}

void chained_cursors_keep_segment_identities() {
  TemporaryDirectory root;
  const auto first_path = root.path() / "first.mp4";
  const auto second_path = root.path() / "second.mp4";
  const auto first_moved = root.path() / "first-retained.mp4";
  write_file(first_path, "segment-one");
  write_file(second_path, "segment-two");

  const auto first = reco::io::StableMediaFile::open(first_path);
  const auto second = reco::io::StableMediaFile::open(second_path);
  const auto first_cursor = first->open_cursor();
  const auto second_cursor = second->open_cursor();

  std::filesystem::rename(first_path, first_moved);
  std::filesystem::rename(second_path, first_path);
  std::filesystem::rename(first_moved, second_path);
  expect_value_or_fail_closed([&] { return first_cursor->read_all(1024); }, "segment-one",
                              "first chained cursor retains the first segment identity");
  expect_value_or_fail_closed([&] { return second_cursor->read_all(1024); }, "segment-two",
                              "second chained cursor retains the second segment identity");

  std::filesystem::rename(second_path, first_moved);
  std::filesystem::rename(first_path, second_path);
  std::filesystem::rename(first_moved, first_path);
  expect_verified_or_fail_closed([&] { first->verify_unchanged(); },
                                 "first restored chained segment");
  expect_verified_or_fail_closed([&] { second->verify_unchanged(); },
                                 "second restored chained segment");
}

void calibration_parser_consumes_pinned_bytes() {
  TemporaryDirectory root;
  const auto path = root.path() / "match.json";
  const auto moved = root.path() / "match-retained.json";
  const auto substitute = root.path() / "match-substitute.json";
  constexpr std::string_view calibration =
      R"json({"left_uniforms":{"width":1920,"height":1080,"fx":1000,"fy":1000,"cx":960,"cy":540,"d":[0,0,0,0]},"right_uniforms":{"width":1920,"height":1080,"fx":1000,"fy":1000,"cx":960,"cy":540,"d":[0,0,0,0]},"params":{"cameraAxisOffset":1,"intersect":0.5,"xTy":0,"xRz":0,"zRx":0,"xRx":0,"zRz":0}})json";
  write_file(path, calibration);
  write_file(substitute, "{}");
  const auto retained = reco::io::StableMediaFile::open(path);
  const auto cursor = retained->open_cursor();
  std::filesystem::rename(path, moved);
  std::filesystem::rename(substitute, path);
  try {
    const auto parsed = reco::core::parse_match_calibration_json(
        cursor->read_all(reco::core::kMaxCalibrationFileSize));
    expect(parsed.has_value(),
           "calibration parsing uses retained bytes while its pathname is swapped");
  } catch (const std::exception& error) {
    expect(std::string_view(error.what()).find("changed while it was retained") !=
               std::string_view::npos,
           "calibration parsing fails closed when ctime exposes the pathname swap");
  }
  std::filesystem::rename(path, substitute);
  std::filesystem::rename(moved, path);
  expect_verified_or_fail_closed([&] { retained->verify_unchanged(); },
                                 "restored calibration pathname");
}

#if !defined(_WIN32)

bool same_modified_time(const struct stat& left, const struct stat& right) {
#if defined(__APPLE__)
  return left.st_mtimespec.tv_sec == right.st_mtimespec.tv_sec &&
         left.st_mtimespec.tv_nsec == right.st_mtimespec.tv_nsec;
#else
  return left.st_mtim.tv_sec == right.st_mtim.tv_sec &&
         left.st_mtim.tv_nsec == right.st_mtim.tv_nsec;
#endif
}

bool same_changed_time(const struct stat& left, const struct stat& right) {
#if defined(__APPLE__)
  return left.st_ctimespec.tv_sec == right.st_ctimespec.tv_sec &&
         left.st_ctimespec.tv_nsec == right.st_ctimespec.tv_nsec;
#else
  return left.st_ctim.tv_sec == right.st_ctim.tv_sec &&
         left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
#endif
}

void restore_access_and_modified_times(const std::filesystem::path& path,
                                       const struct stat& identity) {
#if defined(__APPLE__)
  const struct timespec times[] = {identity.st_atimespec, identity.st_mtimespec};
#else
  const struct timespec times[] = {identity.st_atim, identity.st_mtim};
#endif
  if (::utimensat(AT_FDCWD, path.c_str(), times, 0) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "failed to restore stable-media fixture timestamps");
  }
}

struct stat inspect_file(const std::filesystem::path& path) {
  struct stat identity{};
  if (::stat(path.c_str(), &identity) != 0) {
    throw std::system_error(errno, std::system_category(),
                            "failed to inspect stable-media fixture");
  }
  return identity;
}

void same_size_mutation_with_restored_mtime_fails_closed() {
  TemporaryDirectory root;
  const auto path = root.path() / "same-size.mp4";
  constexpr std::string_view original = "original-payload";
  constexpr std::string_view mutated = "mutated-payload!";
  static_assert(original.size() == mutated.size());
  write_file(path, original);

  const auto retained = reco::io::StableMediaFile::open(path);
  const auto initial = inspect_file(path);
  auto changed = initial;
  for (int attempt = 0; attempt < 200 && same_changed_time(initial, changed); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    write_file(path, mutated);
    restore_access_and_modified_times(path, initial);
    changed = inspect_file(path);
  }

  expect(changed.st_size == initial.st_size, "same-size mutation preserves the retained size");
  expect(same_modified_time(initial, changed),
         "same-size mutation test restores the retained mtime");
  expect(!same_changed_time(initial, changed), "same-size mutation test observes a changed ctime");
  expect_failure([&] { retained->verify_unchanged(); }, "changed while it was retained",
                 "ctime detects a same-size rewrite with restored mtime");
}

#endif

} // namespace

int main() {
  retained_cursor_survives_swap_and_restore();
  chained_cursors_keep_segment_identities();
  calibration_parser_consumes_pinned_bytes();
#if !defined(_WIN32)
  same_size_mutation_with_restored_mtime_fails_closed();
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
