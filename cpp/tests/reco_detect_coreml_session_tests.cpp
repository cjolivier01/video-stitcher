#include "reco/detect/coreml_session.hpp"

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

template <typename Fn> void expect_coreml_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const CoreMlError&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

std::filesystem::path write_model_bundle(std::string_view marker) {
  const auto dir =
      std::filesystem::temp_directory_path() / ("reco-fake-coreml-" + std::string(marker) + ".mlmodelc");
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "Manifest.json") << "{}";
  return dir;
}

void coreml_contract_matches_rust_boundaries() {
  const auto probe = probe_coreml_runtime();
#if defined(__APPLE__)
  expect_true(probe.available, "CoreML available on Apple platforms");
  expect_eq(probe.provider, std::string("CoreML"), "CoreML provider name");
#else
  expect_true(!probe.available, "CoreML unavailable on non-Apple platforms");
  expect_true(!probe.error.empty(), "CoreML unavailable error populated");
  expect_true(!coreml_runtime_available(), "CoreML runtime helper unavailable");
  expect_true(!coreml_runtime_error().empty(), "CoreML runtime helper error");
#endif

  expect_coreml_error([] { (void)CoreMlSession(CoreMlSessionConfig{}); },
                      "empty CoreML path rejected");
  expect_coreml_error([] {
    CoreMlSession(CoreMlSessionConfig{.model_path = "/does/not/exist.mlmodelc"});
  }, "missing CoreML path rejected");

  const auto bundle = write_model_bundle("ok");
  CoreMlSession session(CoreMlSessionConfig{
      .model_path = bundle,
      .input_shape = {1, 3, 7, 7},
  });
  expect_eq(session.config().input_name, std::string("images"), "CoreML default input name");
  expect_eq(session.config().output_name, std::string("output0"), "CoreML default output name");

  std::vector<float> input(3 * 7 * 7, 0.0F);
  expect_coreml_error([&] { (void)session.predict_shared_chw(input); },
                      "CoreML prediction reports platform bridge state");
  expect_coreml_error([&] {
    std::vector<float> short_input(1, 0.0F);
    (void)session.predict_shared_chw(short_input);
  }, "CoreML input length mismatch");

  expect_coreml_error([&] {
    CoreMlSession(CoreMlSessionConfig{
        .model_path = bundle,
        .input_name = std::string("bad\0name", 8),
        .input_shape = {1, 3, 7, 7},
    });
  }, "CoreML input name rejects nul");
  expect_coreml_error([&] {
    CoreMlSession(CoreMlSessionConfig{
        .model_path = bundle,
        .input_shape = {1, 1, 7, 7},
    });
  }, "CoreML rejects non-RGB shape");
}

} // namespace

int main() {
  coreml_contract_matches_rust_boundaries();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
