#include "reco/detect/ncnn_session.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

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

void set_env(const char* name, const std::filesystem::path& value) {
#if defined(_WIN32)
  _putenv_s(name, value.string().c_str());
#else
  setenv(name, value.string().c_str(), 1);
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
  set_env("NCNN_DYLIB_PATH", find_fake_runtime_runfile());
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
}

} // namespace

int main() {
  fake_runtime_session_contract();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
