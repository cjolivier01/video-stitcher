#include "reco/core/cuda_backend.hpp"
#include "reco/detect/detectors.hpp"
#include "reco/detect/npp_interop.hpp"
#include "reco/detect/trt_engine.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <utility>
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

template <typename Fn> void expect_trt_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const TrtError&) {
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
    throw TrtError("failed to set native TensorRT runtime path");
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
    throw TrtError("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    const auto filename = entry.path().filename().string();
    if (filename.find("fake_tensorrt") != std::string::npos &&
        (ends_with(filename, ".so") || ends_with(filename, ".dylib") ||
         ends_with(filename, ".dll"))) {
      return entry.path();
    }
  }
  throw TrtError("fake TensorRT runtime runfile not found");
}

std::filesystem::path native_fake_runtime_path() {
  const auto source = find_fake_runtime_runfile();
#if defined(_WIN32)
  const auto directory = std::filesystem::temp_directory_path() /
                         std::filesystem::path(std::u8string(u8"reco-trt-\u5f55\u50cf"));
  std::filesystem::create_directories(directory);
  const auto destination = directory / source.filename();
  std::filesystem::copy_file(source, destination,
                             std::filesystem::copy_options::overwrite_existing);
  return destination;
#else
  return source;
#endif
}

std::filesystem::path write_engine(std::string_view marker) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("reco-fake-trt-" + std::string(marker) + ".engine");
  std::ofstream file(path, std::ios::binary);
  file << marker;
  return path;
}

void fake_runtime_engine_contract() {
  const auto fake_runtime = native_fake_runtime_path();
  set_env("TRT_DYLIB_PATH", fake_runtime);

  expect_true(trt_runtime_available(), "TensorRT fake runtime is available");
  expect_eq(trt_runtime_error(), std::string(), "TensorRT fake runtime has no error");
  expect_eq(trt_data_type_byte_size(TrtDataType::Float), 4U, "float byte size");
  expect_eq(trt_data_type_byte_size(TrtDataType::Half), 2U, "half byte size");
  expect_eq(trt_data_type_byte_size(TrtDataType::Int8), 1U, "int8 byte size");
  expect_eq(trt_data_type_byte_size(TrtDataType::Int32), 4U, "int32 byte size");

  TrtEngine engine(write_engine("ok"));
  const auto bindings = engine.bindings();
  expect_eq(bindings.size(), 3U, "binding count");
  expect_eq(bindings[0].name, std::string("images"), "input binding name");
  expect_true(bindings[0].is_input, "input binding kind");
  expect_eq(bindings[0].byte_size, 1U * 3U * 8U * 8U * 4U, "input byte size");
  expect_eq(bindings[1].name, std::string("detections"), "output binding name");
  expect_true(!bindings[1].is_input, "output binding kind");
  expect_eq(bindings[1].byte_size, 1U * 300U * 6U * 4U, "output byte size");
  expect_eq(bindings[2].byte_size, 1U * 16U * 2U, "dynamic dim treated as one");

  auto context = engine.create_context();
  std::vector<void*> binding_ptrs = {
      reinterpret_cast<void*>(0x12340000U),
      reinterpret_cast<void*>(0x56780000U),
      nullptr,
  };
  expect_trt_error([&] {
    std::vector<void*> short_bindings = {reinterpret_cast<void*>(0x12340000U)};
    context.enqueue(short_bindings, nullptr);
  }, "enqueue rejects short binding array");
  set_env("RECO_FAKE_TRT_VALIDATE_BINDINGS", "1");
  context.enqueue(binding_ptrs, reinterpret_cast<void*>(0x90120000U));
  unset_env("RECO_FAKE_TRT_VALIDATE_BINDINGS");

  std::vector<void*> retained_binding_ptrs = {
      reinterpret_cast<void*>(0x12340000U),
      reinterpret_cast<void*>(0x56780000U),
      nullptr,
  };
  auto retained_context = [&] {
    TrtEngine scoped_engine(write_engine("ok"));
    return scoped_engine.create_context();
  }();
  retained_context.enqueue(retained_binding_ptrs, nullptr);
  auto moved_context = std::move(retained_context);
  moved_context.enqueue(retained_binding_ptrs, nullptr);

  set_env("RECO_FAKE_TRT_ENQUEUE_FAIL", "1");
  expect_trt_error([&] { context.enqueue(binding_ptrs, nullptr); }, "enqueue failure");
  unset_env("RECO_FAKE_TRT_ENQUEUE_FAIL");

  expect_trt_error([] { TrtEngine(std::filesystem::path("/does/not/exist.engine")); },
                   "missing engine file");
  expect_trt_error([] { (void)TrtEngine(write_engine("bad_dtype")).bindings(); }, "bad dtype");
  expect_trt_error([] { (void)TrtEngine(write_engine("bad_rank")).bindings(); }, "bad rank");
  expect_trt_error([] { (void)TrtEngine(write_engine("overflow")).bindings(); }, "overflow dims");
  set_env("RECO_FAKE_TRT_CONTEXT_FAIL", "1");
  expect_trt_error([&] { (void)engine.create_context(); }, "context create failure");
  unset_env("RECO_FAKE_TRT_CONTEXT_FAIL");
}

void fake_runtime_detector_contract() {
  if (!reco::core::CudaBackend::is_available() || !is_npp_available()) {
    if (std::getenv("RECO_REQUIRE_CUDA_TEST") != nullptr ||
        std::getenv("RECO_REQUIRE_NPP_TEST") != nullptr) {
      std::cerr << "FAIL: CUDA/NPP required but unavailable: "
                << reco::core::CudaBackend::availability_error() << ' '
                << npp_availability_error() << '\n';
      ++failures;
    }
    return;
  }

  TrtGpuDetector detector(write_engine("ok"), 4, 4, 0.10F, {"ball"}, false);
  expect_eq(std::string_view(detector.name()), std::string_view("tensorrt-native"),
            "trt detector name");
  expect_eq(detector.input_size(), 8U, "trt detector input size");

  const auto backend = reco::core::CudaBackend::create();
  const std::vector<std::uint8_t> y{
      16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136,
  };
  const std::vector<std::uint8_t> uv{
      128, 128, 128, 128, 128, 128, 128, 128,
  };
  auto y_device = backend.allocate(y.size());
  auto uv_device = backend.allocate(uv.size());
  backend.copy_host_to_device_2d(reco::core::CudaHostToDevice2DCopy{
      .src = y.data(),
      .src_pitch = 4,
      .dst = y_device.ptr(),
      .dst_pitch = 4,
      .width_bytes = 4,
      .height = 4,
  });
  backend.copy_host_to_device_2d(reco::core::CudaHostToDevice2DCopy{
      .src = uv.data(),
      .src_pitch = 4,
      .dst = uv_device.ptr(),
      .dst_pitch = 4,
      .width_bytes = 4,
      .height = 2,
  });

  set_env("RECO_FAKE_TRT_VALIDATE_DETECTOR_BINDINGS", "1");
  const auto detections = detector.detect(
      CameraId::Left, DetectorFrame(GpuNv12Frame{
                          .y_ptr = y_device.ptr(),
                          .uv_ptr = uv_device.ptr(),
                          .y_pitch = 4,
                          .uv_pitch = 4,
                          .width = 4,
                          .height = 4,
                      }));
  unset_env("RECO_FAKE_TRT_VALIDATE_DETECTOR_BINDINGS");
  expect_eq(detections.size(), 0U, "fake TensorRT detector has zeroed output");

  set_env("RECO_FAKE_TRT_VALIDATE_DETECTOR_BINDINGS", "1");
  const auto colorimetry_detections = detector.detect(
      CameraId::Left, DetectorFrame(GpuNv12Frame{
                          .y_ptr = y_device.ptr(),
                          .uv_ptr = uv_device.ptr(),
                          .y_pitch = 4,
                          .uv_pitch = 4,
                          .width = 4,
                          .height = 4,
                          .color_matrix = reco::core::YuvColorMatrix::Bt601,
                          .color_range = reco::core::YuvColorRange::Limited,
                      }));
  unset_env("RECO_FAKE_TRT_VALIDATE_DETECTOR_BINDINGS");
  expect_eq(colorimetry_detections.size(), 0U, "TensorRT detector accepts colorimetry");

  try {
    (void)detector.detect(CameraId::Left, DetectorFrame(GpuNv12Frame{
                                            .y_ptr = y_device.ptr(),
                                            .uv_ptr = uv_device.ptr(),
                                            .y_pitch = 4,
                                            .uv_pitch = 4,
                                            .width = 4,
                                            .height = 4,
                                            .color_range = reco::core::YuvColorRange::Full,
                                        }));
    std::cerr << "FAIL: TensorRT detector accepted partial colorimetry\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::InferenceFailed,
                "TensorRT detector rejects partial colorimetry");
  }

  set_env("RECO_FAKE_TRT_ENQUEUE_FAIL", "1");
  try {
    (void)detector.detect(CameraId::Left, DetectorFrame(GpuNv12Frame{
                                            .y_ptr = y_device.ptr(),
                                            .uv_ptr = uv_device.ptr(),
                                            .y_pitch = 4,
                                            .uv_pitch = 4,
                                            .width = 4,
                                            .height = 4,
                                        }));
    std::cerr << "FAIL: TensorRT detector enqueue failure was not mapped\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::InferenceFailed,
                "TensorRT detector maps enqueue failure to DetectorError");
  }
  unset_env("RECO_FAKE_TRT_ENQUEUE_FAIL");

  try {
    (void)TrtGpuDetector(write_engine("batch2"), 4, 4, 0.10F, {"ball"}, false);
    std::cerr << "FAIL: TensorRT detector accepted batch > 1 input\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::InferenceFailed,
                "TensorRT detector rejects batch > 1 input");
  }

  try {
    const std::vector<float> chw(1 * 3 * 8 * 8, 0.5F);
    (void)detector.detect(CameraId::Left, DetectorFrame(PreprocessedChwFrame{
                                            .data = chw,
                                            .input_size = 8,
                                            .src_width = 4,
                                            .src_height = 4,
                                        }));
    std::cerr << "FAIL: TensorRT detector accepted CPU preprocessed frame\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::UnsupportedFrameKind,
                "TensorRT detector rejects CPU frames");
  }
}

} // namespace

int main() {
  fake_runtime_engine_contract();
  fake_runtime_detector_contract();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
