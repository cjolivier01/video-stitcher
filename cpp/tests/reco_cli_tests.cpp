#include "reco/cli/cli.hpp"

#include "reco/calibrate/pipeline.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

#if defined(__linux__)
#include <fcntl.h>
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

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
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

  detail::write_calibration_json_atomically(R"json({"writer":"symlink-test"})json", destination,
                                            left_input, right_input);
  expect_eq(read_text_file(destination), std::string("{\"writer\":\"symlink-test\"}\n"),
            "calibration output replaces destination atomically");
  expect_eq(read_text_file(victim), std::string("victim must not change\n"),
            "calibration output does not follow destination or temporary symlink");
  if (!output_symlink_error) {
    expect_true(!std::filesystem::is_symlink(destination),
                "calibration replacement replaces destination symlink itself");
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
  std::thread reader([&] {
    while (running.load(std::memory_order_acquire)) {
      const auto contents = read_text_file(destination);
      const auto complete = std::any_of(payloads.begin(), payloads.end(), [&](const auto& payload) {
        return contents == payload + '\n';
      });
      if (!complete) {
        observed_partial_output.store(true, std::memory_order_release);
        return;
      }
      std::this_thread::yield();
    }
  });

  std::vector<std::thread> writers;
  for (std::size_t writer = 0; writer < payloads.size(); ++writer) {
    writers.emplace_back([&, writer] {
      try {
        for (int iteration = 0; iteration < 12; ++iteration) {
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

  expect_true(!writer_failed.load(std::memory_order_acquire),
              "concurrent calibration writers all succeed");
  expect_true(!observed_partial_output.load(std::memory_order_acquire),
              "concurrent reader never observes partial calibration output");
  const auto final_contents = read_text_file(destination);
  expect_true(std::any_of(payloads.begin(), payloads.end(),
                          [&](const auto& payload) { return final_contents == payload + '\n'; }),
              "concurrent calibration output is one complete writer payload");

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
#endif

  bool orphaned_temporary = false;
  for (const auto& entry : std::filesystem::directory_iterator(root.path())) {
    const auto filename = entry.path().filename().string();
    if (filename.starts_with("match.json.tmp.") || filename.starts_with("blocked.json.tmp.") ||
        filename.starts_with("raced-match.json.tmp.")) {
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
  expect_true(out.str().find("qtdemux ! capsfilter caps=\"video/x-h264;video/x-h265\" ! "
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

int main() {
  validators_match_rust();
  stitch_parse_matches_rust_defaults();
  preview_and_calibrate_parse_matches_rust_defaults();
  live_command_parse_matches_rust_defaults();
  parse_errors_are_reported();
  probe_worker_discovery_handles_path_and_bzlmod_runfiles();
  calibration_output_replacement_is_exclusive_and_atomic();
  command_execution_dispatches_available_stages();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
