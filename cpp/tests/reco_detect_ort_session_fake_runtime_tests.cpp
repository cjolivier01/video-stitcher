#include "reco/detect/ort_session.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace reco::detect;

namespace {

int failures = 0;

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

template <typename Fn> void expect_runtime_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::runtime_error&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

std::filesystem::path find_fake_runtime_runfile() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || *runfiles == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    const auto filename = entry.path().filename().string();
    if (filename.find("fake_onnxruntime") != std::string::npos &&
        (ends_with(filename, ".so") || ends_with(filename, ".dylib") ||
         ends_with(filename, ".dll"))) {
      return entry.path();
    }
  }
  throw std::runtime_error("fake ONNX Runtime runfile not found");
}

std::filesystem::path write_marker_model() {
  const auto path = std::filesystem::temp_directory_path() / "reco_fake_ort_model.onnx";
  std::ofstream file(path);
  file << "fake";
  return path;
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

void fake_runtime_session_contract() {
  const auto fake_runtime = find_fake_runtime_runfile();
  set_env("ORT_DYLIB_PATH", fake_runtime);

  const auto probe = probe_ort_runtime();
  expect_true(probe.available, "fake ORT runtime available");
  expect_eq(probe.version, std::string("1.23.2"), "fake ORT runtime version");

  const auto model = write_marker_model();
  OrtSession session(OrtSessionConfig{
      .model_path = model,
      .fallback_labels = {},
      .providers = {OrtExecutionProvider::Cpu},
  });
  expect_eq(session.metadata().input_size, 8U, "metadata input size");
  expect_eq(session.metadata().input_names[0], std::string("images"), "metadata input name");
  expect_eq(session.metadata().output_names[0], std::string("detections"), "metadata output name");
  expect_eq(session.metadata().labels.size(), 3U, "metadata labels fill gap");
  expect_eq(session.metadata().labels[0], std::string("ball"), "metadata first label");
  expect_eq(session.metadata().labels[1], std::string("class_1"), "metadata gap label");
  expect_eq(session.metadata().labels[2], std::string("player"), "metadata third label");

  const std::vector<float> input(1 * 3 * 8 * 8, 0.5F);
  const std::vector<std::int64_t> shape{1, 3, 8, 8};
  const auto outputs = session.run_cpu_f32(input, shape);
  expect_eq(outputs.size(), 1U, "run output count");
  expect_eq(outputs[0].shape.size(), 3U, "run output rank");
  expect_eq(outputs[0].data.size(), 12U, "run output float count");
  expect_eq(outputs[0].data[4], 0.9F, "run output data copied");

  set_env("RECO_FAKE_ORT_INT_OUTPUT", "1");
  expect_runtime_error([&] { (void)session.run_cpu_f32(input, shape); },
                       "non-f32 ORT output rejected");
  unset_env("RECO_FAKE_ORT_INT_OUTPUT");

  set_env("RECO_FAKE_ORT_EMPTY_OUTPUT", "1");
  const auto empty_outputs = session.run_cpu_f32(input, shape);
  expect_eq(empty_outputs[0].data.size(), 0U, "zero-sized ORT output copied as empty");
  unset_env("RECO_FAKE_ORT_EMPTY_OUTPUT");

  set_env("RECO_FAKE_ORT_METADATA_FAIL", "1");
  OrtSession fallback_session(OrtSessionConfig{
      .model_path = model,
      .fallback_labels = {},
      .providers = {OrtExecutionProvider::Cpu},
  });
  expect_eq(fallback_session.metadata().labels.size(), 1U, "metadata failure fallback count");
  expect_eq(fallback_session.metadata().labels[0], std::string("ball"),
            "metadata failure fallback label");
  unset_env("RECO_FAKE_ORT_METADATA_FAIL");
}

} // namespace

int main() {
  fake_runtime_session_contract();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
