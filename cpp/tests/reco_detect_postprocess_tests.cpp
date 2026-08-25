#include "reco/detect/detectors.hpp"
#include "reco/detect/ort_session.hpp"
#include "reco/detect/probe.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
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

template <typename Fn> void expect_invalid_argument(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
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

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

std::vector<float> tensor(std::initializer_list<std::array<float, 6>> rows) {
  std::vector<float> out;
  for (const auto& row : rows) {
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

constexpr std::uint32_t kFrameW = 1920;
constexpr std::uint32_t kFrameH = 1080;
constexpr float kModelSize = 640.0F;

float scale() {
  return std::min(kModelSize / static_cast<float>(kFrameW),
                  kModelSize / static_cast<float>(kFrameH));
}
float pad_x() { return (kModelSize - (static_cast<float>(kFrameW) * scale())) / 2.0F; }
float pad_y() { return (kModelSize - (static_cast<float>(kFrameH) * scale())) / 2.0F; }

void stock_yolo_postprocess_matches_rust() {
  const auto valid =
      postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.85F, 0.0F}}), 1, CameraId::Left, 0.10F,
                  scale(), pad_x(), pad_y(), kFrameW, kFrameH);
  expect_eq(valid.size(), 1U, "valid detection count");
  expect_eq(valid[0].class_id, 0U, "valid class id");
  expect_near(valid[0].confidence, 0.85F, 1.0e-6F, "valid confidence");
  expect_true(valid[0].center_x >= 0.0F && valid[0].center_x <= 1.0F, "valid center x");
  expect_true(valid[0].width > 0.0F, "valid width");

  const auto multiple =
      postprocess(tensor({{100.0F, 200.0F, 120.0F, 220.0F, 0.80F, 0.0F},
                          {400.0F, 300.0F, 500.0F, 400.0F, 0.70F, 1.0F}}),
                  2, CameraId::Right, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH);
  expect_eq(multiple.size(), 2U, "multiple detection count");
  expect_eq(multiple[1].class_id, 1U, "multiple class id");
  expect_true(multiple[0].camera == CameraId::Right, "camera preserved");

  expect_true(postprocess({}, 0, CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "zero detections");
  expect_true(postprocess({1.0F, 2.0F, 3.0F}, 1, CameraId::Left, 0.10F, scale(), pad_x(), pad_y(),
                          kFrameW, kFrameH)
                  .empty(),
              "short buffer");
  expect_true(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.05F, 0.0F}}), 1,
                          CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "low confidence filtered");
  expect_eq(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.10F, 0.0F}}), 1, CameraId::Left,
                        0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                .size(),
            1U, "exact threshold passes");
  expect_true(postprocess(tensor({{-500.0F, -500.0F, -400.0F, -400.0F, 0.90F, 0.0F}}), 1,
                          CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "out of bounds filtered");
}

void nan_and_class_guards_match_rust() {
  expect_true(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F,
                                   std::numeric_limits<float>::quiet_NaN(), 0.0F}}),
                          1, CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "nan confidence filtered");
  expect_true(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F,
                                   std::numeric_limits<float>::infinity(), 0.0F}}),
                          1, CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "infinite confidence filtered");
  for (std::size_t bad = 0; bad < 4; ++bad) {
    std::array<float, 6> row{300.0F, 300.0F, 340.0F, 340.0F, 0.80F, 0.0F};
    row[bad] = std::numeric_limits<float>::quiet_NaN();
    expect_true(postprocess(tensor({row}), 1, CameraId::Left, 0.10F, scale(), pad_x(), pad_y(),
                            kFrameW, kFrameH)
                    .empty(),
                "nan bbox filtered");
  }
  expect_true(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.80F,
                                   std::numeric_limits<float>::quiet_NaN()}}),
                          1, CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "nan class filtered");
  expect_true(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.80F, 1.0e9F}}), 1,
                          CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "class range filtered");
  expect_true(postprocess(tensor({{0.0F, 0.0F, 0.0F, 0.0F, 0.80F, 0.0F}}), 1, CameraId::Left, 0.10F,
                          0.0F, 0.0F, 0.0F, 1000, 1000)
                  .empty(),
              "derived nan stock coords filtered");
}

void ball_detector_adapter_matches_rust() {
  const auto decoded = postprocess_balldet({500.0F, 250.0F, 40.0F, 40.0F, 1.0F, 0.90F}, 1,
                                           CameraId::Left, 0.25F, 1.0F, 0.0F, 0.0F, 1000, 1000);
  expect_eq(decoded.size(), 1U, "ball decoded count");
  expect_eq(decoded[0].class_id, 0U, "ball class id");
  expect_near(decoded[0].center_x, 0.5F, 1.0e-3F, "ball center x");
  expect_near(decoded[0].width, 0.04F, 1.0e-3F, "ball width");

  const std::vector<float> data{500.0F, 250.0F, 40.0F,  40.0F,  1.0F,   0.10F,  500.0F, 250.0F,
                                40.0F,  40.0F,  1.0F,   0.90F,  505.0F, 252.0F, 40.0F,  40.0F,
                                1.0F,   0.80F,  100.0F, 100.0F, 30.0F,  30.0F,  1.0F,   0.70F};
  const auto nms =
      postprocess_balldet(data, 4, CameraId::Left, 0.25F, 1.0F, 0.0F, 0.0F, 1000, 1000);
  expect_eq(nms.size(), 2U, "ball nms count");
  expect_near(nms[0].confidence, 0.90F, 1.0e-6F, "ball nms highest first");

  expect_true(postprocess_balldet({0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.90F}, 1, CameraId::Left, 0.25F,
                                  0.0F, 0.0F, 0.0F, 1000, 1000)
                  .empty(),
              "derived nan ball coords filtered");

  const auto equal_conf = postprocess_balldet(
      {500.0F, 250.0F, 40.0F, 40.0F, 1.0F, 0.90F, 505.0F, 252.0F, 40.0F, 40.0F, 1.0F, 0.90F}, 2,
      CameraId::Left, 0.25F, 1.0F, 0.0F, 0.0F, 1000, 1000);
  expect_eq(equal_conf.size(), 1U, "equal confidence nms keeps one");
  expect_near(equal_conf[0].center_x, 0.5F, 1.0e-6F, "equal confidence keeps first input");
}

void labels_and_probe_match_rust() {
  const std::string path = "/tmp/reco_detect_labels_test.labels";
  {
    std::ofstream file(path);
    file << " ball \n\n player\t\n";
  }
  const auto labels = read_labels_file(path);
  expect_eq(labels.size(), 2U, "labels count");
  expect_eq(labels[0], std::string("ball"), "first label");
  expect_true(read_labels_file("/tmp/reco_detect_missing.labels").empty(), "missing labels empty");

  const AiProbeResult result{.providers = {"TensorRT", "CPU"}, .can_run_on_gpu_frames = true};
  expect_eq(result.best_provider(), std::string("TensorRT"), "best provider");
  expect_true(result.is_available(), "probe available");
  const AiProbeResult empty{
      .providers = {}, .can_run_on_gpu_frames = false, .errors = {"ORT init"}};
  expect_eq(empty.best_provider(), std::string("unavailable"), "empty best provider");
  expect_true(!empty.is_available(), "probe unavailable");

  const auto runtime_first = probe_ort_runtime();
  const auto runtime_second = probe_ort_runtime();
  expect_eq(runtime_first.available, runtime_second.available, "ort runtime probe cached verdict");
  expect_eq(runtime_first.version, runtime_second.version, "ort runtime probe cached version");

  const auto probed = probe_execution_providers();
  if (runtime_first.available) {
    expect_true(!probed.is_available(), "provider probe stays conservative until session port");
    expect_true(!probed.errors.empty(), "provider probe explains missing session port");
  } else {
    expect_true(!probed.is_available(), "runtime unavailable has no provider");
    expect_true(!probed.errors.empty(), "runtime unavailable reports error");
  }

  const auto cache = reco_cache_dir("ort-session-test");
  expect_true(cache.filename() == "ort-session-test", "cache dir subdir");
  expect_true(cache.parent_path().filename() == "reco", "cache dir reco parent");
}

void ort_session_contract_matches_rust() {
  expect_invalid_argument([] { (void)OrtSession(OrtSessionConfig{}); },
                          "empty ORT model path rejected before runtime load");

  const auto missing_model =
      std::filesystem::temp_directory_path() / "reco_missing_model_for_ort_session_test.onnx";
  expect_runtime_error(
      [&] {
        (void)OrtSession(OrtSessionConfig{
            .model_path = missing_model,
            .fallback_labels = {},
            .providers = {OrtExecutionProvider::Cpu},
        });
      },
      "missing ORT model rejected before runtime load");

  const std::string marker_path =
      (std::filesystem::temp_directory_path() / "reco_not_an_onnx_model.txt").string();
  {
    std::ofstream file(marker_path);
    file << "not onnx";
  }
  expect_runtime_error(
      [&] {
        (void)OrtSession(OrtSessionConfig{
            .model_path = marker_path,
            .fallback_labels = {"ball"},
            .providers = {OrtExecutionProvider::Cuda},
        });
      },
      "gpu-only ORT provider rejected until provider registration is ported");
  expect_runtime_error(
      [&] {
        (void)OrtSession(OrtSessionConfig{
            .model_path = marker_path,
            .fallback_labels = {"ball"},
            .providers = {OrtExecutionProvider::Cuda, OrtExecutionProvider::Cpu},
        });
      },
      "mixed GPU/CPU ORT provider config rejected until GPU provider registration is ported");

  const char* model_env = std::getenv("RECO_TEST_ONNX_MODEL_PATH");
  if (model_env == nullptr || *model_env == '\0') {
    return;
  }

  OrtSession session(OrtSessionConfig{
      .model_path = model_env,
      .fallback_labels = {"ball"},
      .providers = {OrtExecutionProvider::Cpu},
  });
  expect_true(!session.metadata().input_names.empty(), "ort session input names");
  expect_true(!session.metadata().output_names.empty(), "ort session output names");
  expect_true(!session.metadata().labels.empty(), "ort session labels");
  expect_true(session.metadata().input_size > 0, "ort session input size");
  const std::vector<float> one_input{0.0F};
  const std::vector<std::int64_t> dynamic_shape{1, 3, 0, 0};
  const std::vector<std::int64_t> mismatched_shape{1, 3, 2, 2};
  expect_invalid_argument(
      [&] { (void)session.run_cpu_f32({}, dynamic_shape); },
      "ort session rejects unresolved input shape");
  expect_invalid_argument(
      [&] { (void)session.run_cpu_f32(one_input, mismatched_shape); },
      "ort session rejects input length mismatch");
}

class FakeDetector final : public UnifiedDetector {
public:
  explicit FakeDetector(std::vector<std::string> labels) : labels_(std::move(labels)) {}

  [[nodiscard]] const char* name() const override { return "fake-cpu-only"; }

  [[nodiscard]] std::vector<Detection> detect(CameraId camera,
                                              const DetectorFrame& frame) override {
    if (!std::holds_alternative<RawFrame>(frame.variant())) {
      throw DetectorError::unsupported_frame_kind();
    }
    return {{.camera = camera,
             .class_id = 0,
             .confidence = 0.9F,
             .center_x = 0.5F,
             .center_y = 0.5F,
             .width = 0.1F,
             .height = 0.1F}};
  }

  [[nodiscard]] std::optional<std::span<const std::string>> class_names() const override {
    return std::span<const std::string>(labels_);
  }

private:
  std::vector<std::string> labels_;
};

void unified_detector_contract_matches_rust() {
  static_assert(!std::is_copy_constructible_v<DetectorFrame>);
  static_assert(std::is_move_constructible_v<DetectorFrame>);
  static_assert(!std::is_constructible_v<DetectorError, DetectorErrorKind, std::string>);

  const std::vector<std::uint8_t> y{0, 1, 2, 3};
  const std::vector<std::uint8_t> u{128};
  const std::vector<std::uint8_t> v{128};
  const DetectorFrame cpu_frame(RawFrame{
      .y = y,
      .chroma = Yuv420pChroma{.u = u, .v = v},
      .width = 2,
      .height = 2,
  });
  expect_eq(cpu_frame.variant_name(), std::string_view("Cpu"), "cpu variant name");

  const DetectorFrame nv12_frame(RawFrame{
      .y = y,
      .chroma = Nv12Chroma{.uv = u},
      .width = 2,
      .height = 2,
  });
  expect_eq(nv12_frame.variant_name(), std::string_view("Cpu"), "nv12 cpu variant name");

  const std::vector<float> chw(12, 0.0F);
  expect_eq(DetectorFrame(
                PreprocessedChwFrame{.data = chw, .input_size = 2, .src_width = 2, .src_height = 2})
                .variant_name(),
            std::string_view("PreprocessedChw"), "preprocessed variant name");
  expect_eq(DetectorFrame(RgbaFrame{.data = y, .width = 1, .height = 1}).variant_name(),
            std::string_view("Rgba"), "rgba variant name");
  expect_eq(DetectorFrame(
                GpuNv12Frame{
                    .y_ptr = 1, .uv_ptr = 2, .y_pitch = 2, .uv_pitch = 2, .width = 2, .height = 2})
                .variant_name(),
            std::string_view("Cuda"), "cuda variant name");
  expect_eq(
      DetectorFrame(CudaRgbaFrame{.ptr = 3, .pitch = 4, .width = 1, .height = 1}).variant_name(),
      std::string_view("CudaRgba"), "cuda rgba variant name");
  expect_eq(DetectorFrame(CudaRgbaLetterboxedFrame{.ptr = 5, .src_width = 4, .src_height = 4})
                .variant_name(),
            std::string_view("CudaRgbaLetterboxed"), "cuda rgba letterboxed variant name");
  expect_eq(
      DetectorFrame(MetalFrame{.cv_pixel_buffer = nullptr, .width = 1, .height = 1}).variant_name(),
      std::string_view("Metal"), "metal variant name");
  expect_eq(
      DetectorFrame(WgpuNv12Frame{.y_view = nullptr, .uv_view = nullptr, .width = 2, .height = 2})
          .variant_name(),
      std::string_view("WgpuNv12"), "wgpu variant name");

  FakeDetector detector({"ball"});
  expect_eq(std::string_view(detector.name()), std::string_view("fake-cpu-only"), "detector name");
  const auto detections = detector.detect(CameraId::Left, cpu_frame);
  expect_eq(detections.size(), 1U, "fake detector detection count");
  expect_true(detections[0].camera == CameraId::Left, "fake detector camera");
  const auto labels = detector.class_names();
  expect_true(labels.has_value(), "fake detector labels");
  expect_eq((*labels)[0], std::string("ball"), "fake detector label");

  try {
    (void)detector.detect(CameraId::Left,
                          DetectorFrame(RgbaFrame{.data = y, .width = 1, .height = 1}));
    std::cerr << "FAIL: unsupported frame did not throw\n";
    ++failures;
  } catch (const DetectorError& error) {
    expect_true(error.kind() == DetectorErrorKind::UnsupportedFrameKind, "unsupported error kind");
  }

  const auto inference_error = DetectorError::inference_failed("engine");
  expect_true(inference_error.kind() == DetectorErrorKind::InferenceFailed, "inference error kind");
  expect_eq(inference_error.detail(), std::string("engine"), "inference error detail");
  const auto timeout_error = DetectorError::timeout(std::chrono::microseconds(5));
  expect_true(timeout_error.kind() == DetectorErrorKind::Timeout, "timeout error kind");
  expect_true(timeout_error.after() == std::chrono::microseconds(5), "timeout error duration");
  expect_invalid_argument([] { (void)DetectorError::timeout(-std::chrono::nanoseconds(1)); },
                          "negative timeout rejected");
  const auto transport_error = DetectorError::transport("grpc");
  expect_true(transport_error.kind() == DetectorErrorKind::Transport, "transport error kind");
  expect_eq(transport_error.detail(), std::string("grpc"), "transport error detail");
  expect_eq(DetectorError::canceled().what(), std::string("detection canceled"),
            "canceled error message");
}

void onnx_names_parser_matches_rust() {
  const auto happy = parse_names_dict_string("{0: 'person', 1: 'bicycle', 2: 'car'}");
  expect_true(happy.has_value(), "happy path parses");
  expect_eq(happy->size(), 3U, "happy path count");
  expect_eq((*happy)[0], std::string("person"), "happy first label");
  expect_eq((*happy)[1], std::string("bicycle"), "happy second label");
  expect_eq((*happy)[2], std::string("car"), "happy third label");

  const auto gaps = parse_names_dict_string("{0: 'ball', 3: 'goal'}");
  expect_true(gaps.has_value(), "gaps parse");
  expect_eq(gaps->size(), 4U, "gaps count");
  expect_eq((*gaps)[0], std::string("ball"), "gaps first label");
  expect_eq((*gaps)[1], std::string("class_1"), "gap class 1");
  expect_eq((*gaps)[2], std::string("class_2"), "gap class 2");
  expect_eq((*gaps)[3], std::string("goal"), "gaps final label");

  expect_true(!parse_names_dict_string("{9999999999: 'ball'}").has_value(),
              "huge class index rejected");
  expect_true(!parse_names_dict_string("{10000: 'class'}").has_value(),
              "class cap boundary rejected");

  const auto just_below = parse_names_dict_string("{0: 'a', 9999: 'b'}");
  expect_true(just_below.has_value(), "class below cap accepted");
  expect_eq(just_below->size(), 10000U, "class below cap count");
  expect_eq((*just_below)[0], std::string("a"), "class below cap first");
  expect_eq((*just_below)[9999], std::string("b"), "class below cap last");

  expect_true(!parse_names_dict_string("{}").has_value(), "empty dict rejected");
  expect_true(!parse_names_dict_string("[0: 'x']").has_value(), "non dict rejected");
  expect_true(!parse_names_dict_string("random garbage").has_value(), "garbage rejected");

  const auto mixed = parse_names_dict_string("{bad: 'skip', 2: \"two\"}");
  expect_true(mixed.has_value(), "malformed entries skipped");
  expect_eq(mixed->size(), 3U, "malformed skip count");
  expect_eq((*mixed)[0], std::string("class_0"), "malformed skip gap");
  expect_eq((*mixed)[2], std::string("two"), "double quote trimmed");

  const auto duplicate = parse_names_dict_string("{0: 'first', 0: 'shadow', 2: 'blocked'}");
  expect_true(duplicate.has_value(), "duplicate index parses");
  expect_eq(duplicate->size(), 3U, "duplicate index count");
  expect_eq((*duplicate)[0], std::string("first"), "duplicate index keeps first");
  expect_eq((*duplicate)[1], std::string("class_1"), "duplicate index blocks iterator");
  expect_eq((*duplicate)[2], std::string("class_2"), "duplicate index shadows later label");
}

} // namespace

int main() {
  stock_yolo_postprocess_matches_rust();
  nan_and_class_guards_match_rust();
  ball_detector_adapter_matches_rust();
  labels_and_probe_match_rust();
  ort_session_contract_matches_rust();
  unified_detector_contract_matches_rust();
  onnx_names_parser_matches_rust();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
