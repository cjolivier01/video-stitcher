#include "reco/detect/detectors.hpp"
#include "reco/detect/ort_session.hpp"
#include "reco/core/cuda_backend.hpp"

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

  set_env("RECO_FAKE_ORT_REQUIRE_CUDA_NO_CPU_FALLBACK", "1");
  OrtSession cuda_session(OrtSessionConfig{
      .model_path = model,
      .fallback_labels = {"ball"},
      .providers = {OrtExecutionProvider::Cuda},
  });
  unset_env("RECO_FAKE_ORT_REQUIRE_CUDA_NO_CPU_FALLBACK");
  set_env("RECO_FAKE_ORT_VALIDATE_CUDA_INPUT", "1");
  const auto cuda_outputs =
      cuda_session.run_cuda_f32(0x12340000U, input.size() * sizeof(float), shape);
  unset_env("RECO_FAKE_ORT_VALIDATE_CUDA_INPUT");
  expect_eq(cuda_outputs[0].data.size(), 12U, "cuda session output copied");
  try {
    (void)cuda_session.run_cpu_f32(input, shape);
    std::cerr << "FAIL: cpu run on cuda session did not throw\n";
    ++failures;
  } catch (const std::invalid_argument&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: cpu run rejected on cuda session threw unexpected exception: "
              << error.what() << '\n';
    ++failures;
  }

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

void fake_runtime_cpu_detector_contract() {
  const auto model = write_marker_model();
  CpuYoloDetector detector(model, 0.10F, {});
  expect_eq(std::string_view(detector.name()), std::string_view("ort-cpu"), "cpu detector name");
  expect_eq(detector.input_size(), 8U, "cpu detector input size");
  const auto labels = detector.class_names();
  expect_true(labels.has_value(), "cpu detector labels available");
  expect_eq((*labels)[2], std::string("player"), "cpu detector metadata labels");

  const std::vector<std::uint8_t> y{80, 90, 100, 110};
  const std::vector<std::uint8_t> uv{128, 128};
  const DetectorFrame raw_frame(RawFrame{
      .y = y,
      .chroma = Nv12Chroma{.uv = uv},
      .width = 2,
      .height = 2,
  });
  set_env("RECO_FAKE_ORT_VALIDATE_NV12_2X2", "1");
  const auto raw_detections = detector.detect(CameraId::Left, raw_frame);
  unset_env("RECO_FAKE_ORT_VALIDATE_NV12_2X2");
  expect_eq(raw_detections.size(), 2U, "cpu detector raw detections");
  expect_eq(raw_detections[0].class_id, 0U, "cpu detector first class");
  expect_eq(raw_detections[1].class_id, 2U, "cpu detector second class");
  expect_true(raw_detections[0].camera == CameraId::Left, "cpu detector raw camera");

  const std::vector<float> chw(1 * 3 * 8 * 8, 0.5F);
  const DetectorFrame preprocessed_frame(PreprocessedChwFrame{
      .data = chw,
      .input_size = 8,
      .src_width = 2,
      .src_height = 2,
  });
  const auto preprocessed_detections = detector.detect(CameraId::Right, preprocessed_frame);
  expect_eq(preprocessed_detections.size(), 2U, "cpu detector preprocessed detections");
  expect_true(preprocessed_detections[0].camera == CameraId::Right,
              "cpu detector preprocessed camera");

  const std::vector<std::uint8_t> y_4x2{10, 20, 30, 40, 50, 60, 70, 80};
  const std::vector<std::uint8_t> u_4x2{128, 128};
  const std::vector<std::uint8_t> v_4x2{128, 128};
  const DetectorFrame yuv420p_frame(RawFrame{
      .y = y_4x2,
      .chroma = Yuv420pChroma{.u = u_4x2, .v = v_4x2},
      .width = 4,
      .height = 2,
  });
  set_env("RECO_FAKE_ORT_VALIDATE_YUV420P_4X2", "1");
  const auto yuv420p_detections = detector.detect(CameraId::Left, yuv420p_frame);
  unset_env("RECO_FAKE_ORT_VALIDATE_YUV420P_4X2");
  expect_eq(yuv420p_detections.size(), 1U, "cpu detector letterboxed yuv420p detections");

  set_env("RECO_FAKE_ORT_EMPTY_OUTPUT", "1");
  expect_runtime_error([&] { (void)detector.detect(CameraId::Left, preprocessed_frame); },
                       "cpu detector rejects malformed ORT output shape");
  unset_env("RECO_FAKE_ORT_EMPTY_OUTPUT");

  try {
    (void)detector.detect(CameraId::Left,
                          DetectorFrame(GpuNv12Frame{
                              .y_ptr = 1,
                              .uv_ptr = 2,
                              .y_pitch = 2,
                              .uv_pitch = 2,
                              .width = 2,
                              .height = 2,
                          }));
    std::cerr << "FAIL: cpu detector accepted CUDA frame\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::UnsupportedFrameKind,
                "cpu detector rejects CUDA frames");
  }
}

void fake_runtime_cuda_detector_contract() {
  if (!reco::core::CudaBackend::is_available()) {
    if (std::getenv("RECO_REQUIRE_CUDA_TEST") != nullptr) {
      std::cerr << "FAIL: CUDA required but unavailable: "
                << reco::core::CudaBackend::availability_error() << '\n';
      ++failures;
    }
    return;
  }

  const auto model = write_marker_model();
  OrtCudaYoloDetector detector(model, 2, 2, 0.10F, {}, false);
  expect_eq(std::string_view(detector.name()), std::string_view("ort-cuda"), "cuda detector name");
  expect_eq(detector.input_size(), 8U, "cuda detector input size");

  const auto backend = reco::core::CudaBackend::create();
  const std::vector<std::uint8_t> y{80, 90, 100, 110};
  const std::vector<std::uint8_t> uv{128, 128};
  auto y_device = backend.allocate(y.size());
  auto uv_device = backend.allocate(uv.size());
  backend.copy_host_to_device_2d(reco::core::CudaHostToDevice2DCopy{
      .src = y.data(),
      .src_pitch = 2,
      .dst = y_device.ptr(),
      .dst_pitch = 2,
      .width_bytes = 2,
      .height = 2,
  });
  backend.copy_host_to_device_2d(reco::core::CudaHostToDevice2DCopy{
      .src = uv.data(),
      .src_pitch = 2,
      .dst = uv_device.ptr(),
      .dst_pitch = 2,
      .width_bytes = 2,
      .height = 1,
  });

  set_env("RECO_FAKE_ORT_VALIDATE_CUDA_INPUT_ANY", "1");
  const auto detections = detector.detect(
      CameraId::Left, DetectorFrame(GpuNv12Frame{
                          .y_ptr = y_device.ptr(),
                          .uv_ptr = uv_device.ptr(),
                          .y_pitch = 2,
                          .uv_pitch = 2,
                          .width = 2,
                          .height = 2,
                      }));
  unset_env("RECO_FAKE_ORT_VALIDATE_CUDA_INPUT_ANY");
  expect_eq(detections.size(), 2U, "cuda detector detections");

  set_env("RECO_FAKE_ORT_VALIDATE_CUDA_INPUT_ANY", "1");
  const auto colorimetry_detections = detector.detect(
      CameraId::Left, DetectorFrame(GpuNv12Frame{
                          .y_ptr = y_device.ptr(),
                          .uv_ptr = uv_device.ptr(),
                          .y_pitch = 2,
                          .uv_pitch = 2,
                          .width = 2,
                          .height = 2,
                          .color_matrix = reco::core::YuvColorMatrix::Bt709,
                          .color_range = reco::core::YuvColorRange::Limited,
                      }));
  unset_env("RECO_FAKE_ORT_VALIDATE_CUDA_INPUT_ANY");
  expect_eq(colorimetry_detections.size(), 2U, "cuda detector accepts colorimetry");

  try {
    (void)detector.detect(CameraId::Left, DetectorFrame(GpuNv12Frame{
                                            .y_ptr = y_device.ptr(),
                                            .uv_ptr = uv_device.ptr(),
                                            .y_pitch = 2,
                                            .uv_pitch = 2,
                                            .width = 2,
                                            .height = 2,
                                            .color_matrix = reco::core::YuvColorMatrix::Bt709,
                                        }));
    std::cerr << "FAIL: CUDA detector accepted partial colorimetry\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::InferenceFailed,
                "cuda detector rejects partial colorimetry");
  }

  try {
    const std::vector<float> chw(1 * 3 * 8 * 8, 0.5F);
    (void)detector.detect(CameraId::Left, DetectorFrame(PreprocessedChwFrame{
                                            .data = chw,
                                            .input_size = 8,
                                            .src_width = 2,
                                            .src_height = 2,
                                        }));
    std::cerr << "FAIL: cuda detector accepted CPU frame\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::UnsupportedFrameKind,
                "cuda detector rejects CPU frames");
  }
}

} // namespace

int main() {
  fake_runtime_session_contract();
  fake_runtime_cpu_detector_contract();
  fake_runtime_cuda_detector_contract();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
