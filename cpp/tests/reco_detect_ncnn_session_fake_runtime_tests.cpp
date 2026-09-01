#include "reco/detect/detectors.hpp"
#include "reco/detect/ncnn_session.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace reco::detect;

namespace {

int failures = 0;

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_true(bool value, std::string_view message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_ncnn_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const NcnnError&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_detector_error(Fn&& fn, DetectorErrorKind kind,
                                                  std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const DetectorError& error) {
    if (error.kind() != kind) {
      std::cerr << "FAIL: " << message << " unexpected kind\n";
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

void set_env(const char* name, const std::filesystem::path& value) {
#if defined(_WIN32)
  std::wstring wide_name;
  for (const unsigned char character : std::string_view(name)) {
    wide_name.push_back(static_cast<wchar_t>(character));
  }
  if (SetEnvironmentVariableW(wide_name.c_str(), value.c_str()) == 0) {
    throw NcnnError("failed to set native NCNN runtime path");
  }
#else
  setenv(name, value.c_str(), 1);
#endif
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

std::filesystem::path find_fake_runtime_runfile() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || *runfiles == '\0') {
    throw NcnnError("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    const auto filename = entry.path().filename().string();
    if (filename.find("fake_ncnn") != std::string::npos &&
        (ends_with(filename, ".so") || ends_with(filename, ".dylib") ||
         ends_with(filename, ".dll"))) {
      return entry.path();
    }
  }
  throw NcnnError("fake NCNN runtime runfile not found");
}

std::filesystem::path native_fake_runtime_path() {
  const auto source = find_fake_runtime_runfile();
#if defined(_WIN32)
  const auto directory = std::filesystem::temp_directory_path() /
                         std::filesystem::path(std::u8string(u8"reco-ncnn-\u5f55\u50cf"));
  std::filesystem::create_directories(directory);
  const auto destination = directory / source.filename();
  std::filesystem::copy_file(source, destination,
                             std::filesystem::copy_options::overwrite_existing);
  return destination;
#else
  return source;
#endif
}

std::filesystem::path write_model_dir(std::string_view marker) {
  const auto dir = std::filesystem::temp_directory_path() / ("reco-fake-ncnn-" + std::string(marker));
  std::filesystem::create_directories(dir);
  {
    std::ofstream param(dir / "model.ncnn.param");
    param << "param";
  }
  {
    std::ofstream bin(dir / "model.ncnn.bin", std::ios::binary);
    bin << "bin";
  }
  return dir;
}

void fake_runtime_session_contract() {
  set_env("NCNN_DYLIB_PATH", native_fake_runtime_path());
  expect_true(ncnn_runtime_available(), "NCNN fake runtime available");
  expect_eq(ncnn_runtime_error(), std::string(), "NCNN fake runtime has no error");

  NcnnSession session(NcnnSessionConfig{.model_dir = write_model_dir("ok")});
  std::vector<float> input(3 * 7 * 7, 0.25F);
  std::fill(input.begin() + 49, input.begin() + 98, 0.50F);
  std::fill(input.begin() + 98, input.end(), 0.75F);
  const auto output = session.run_preprocessed_chw(input, 7);
  expect_eq(output.width, 2, "NCNN output width");
  expect_eq(output.height, 5, "NCNN output height");
  expect_eq(output.data.size(), 10U, "NCNN output copied");
  expect_eq(output.data[0], 0.25F, "NCNN output first value");
  expect_eq(output.data[1], 0.50F, "NCNN output second channel value");
  expect_eq(output.data[2], 0.75F, "NCNN output third channel value");

  expect_ncnn_error([&] { (void)session.run_preprocessed_chw({}, 7); },
                    "input size mismatch");
  expect_ncnn_error([] { NcnnSession(NcnnSessionConfig{.model_dir = "/does/not/exist"}); },
                    "missing model dir");
  expect_ncnn_error([] {
    NcnnSession(NcnnSessionConfig{.model_dir = write_model_dir("bad_threads"),
                                  .num_threads = 0});
  }, "bad thread count");
  expect_ncnn_error([] {
    NcnnSession(NcnnSessionConfig{.model_dir = write_model_dir("bad_input_name"),
                                  .input_name = std::string("bad\0name", 8)});
  }, "input name rejects nul");

  set_env("RECO_FAKE_NCNN_PARAM_FAIL", "1");
  expect_ncnn_error([] { NcnnSession(NcnnSessionConfig{.model_dir = write_model_dir("param_fail")}); },
                    "param load failure");
  unset_env("RECO_FAKE_NCNN_PARAM_FAIL");

  set_env("RECO_FAKE_NCNN_EXTRACT_FAIL", "1");
  expect_ncnn_error([&] { (void)session.run_preprocessed_chw(input, 7); }, "extract failure");
  unset_env("RECO_FAKE_NCNN_EXTRACT_FAIL");

  set_env("RECO_FAKE_NCNN_NULL_OUTPUT", "1");
  expect_ncnn_error([&] { (void)session.run_preprocessed_chw(input, 7); }, "null output");
  unset_env("RECO_FAKE_NCNN_NULL_OUTPUT");

  NcnnYoloDetector detector(write_model_dir("detector"), 7, 7, 7, 0.10F, {"ball"});
  DetectorFrame preprocessed(PreprocessedChwFrame{
      .data = input,
      .input_size = 7,
      .src_width = 7,
      .src_height = 7,
  });
  const auto detections = detector.detect(CameraId::Right, preprocessed);
  expect_eq(detector.name(), std::string_view("ncnn"), "NCNN detector name");
  expect_eq(detector.input_size(), 7U, "NCNN detector input size");
  expect_eq(detector.class_names()->size(), 1U, "NCNN detector labels");
  expect_eq(detections.size(), 2U, "NCNN detector transposed detections");
  expect_true(detections[0].camera == CameraId::Right, "NCNN detector camera preserved");
  expect_eq(detections[0].class_id, 0U, "NCNN detector class id");
  expect_eq(detections[0].confidence, 0.90F, "NCNN detector confidence sorted first");

  DetectorFrame wrong_size(PreprocessedChwFrame{
      .data = input,
      .input_size = 8,
      .src_width = 7,
      .src_height = 7,
  });
  expect_detector_error([&] { (void)detector.detect(CameraId::Left, wrong_size); },
                        DetectorErrorKind::InferenceFailed, "NCNN detector input size mismatch");
  DetectorFrame raw(RawFrame{});
  expect_detector_error([&] { (void)detector.detect(CameraId::Left, raw); },
                        DetectorErrorKind::UnsupportedFrameKind, "NCNN detector rejects raw frame");
}

} // namespace

int main() {
  fake_runtime_session_contract();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
