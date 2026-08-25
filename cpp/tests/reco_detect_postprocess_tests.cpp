#include "reco/detect/detectors.hpp"
#include "reco/detect/probe.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
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

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance) {
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
  const auto valid = postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.85F, 0.0F}}), 1,
                                 CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW,
                                 kFrameH);
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
  expect_true(postprocess({1.0F, 2.0F, 3.0F}, 1, CameraId::Left, 0.10F, scale(), pad_x(),
                          pad_y(), kFrameW, kFrameH)
                  .empty(),
              "short buffer");
  expect_true(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.05F, 0.0F}}), 1,
                          CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
                  .empty(),
              "low confidence filtered");
  expect_eq(postprocess(tensor({{300.0F, 300.0F, 340.0F, 340.0F, 0.10F, 0.0F}}), 1,
                        CameraId::Left, 0.10F, scale(), pad_x(), pad_y(), kFrameW, kFrameH)
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
  expect_true(postprocess(tensor({{0.0F, 0.0F, 0.0F, 0.0F, 0.80F, 0.0F}}), 1,
                          CameraId::Left, 0.10F, 0.0F, 0.0F, 0.0F, 1000, 1000)
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

  const std::vector<float> data{
      500.0F, 250.0F, 40.0F, 40.0F, 1.0F, 0.10F,
      500.0F, 250.0F, 40.0F, 40.0F, 1.0F, 0.90F,
      505.0F, 252.0F, 40.0F, 40.0F, 1.0F, 0.80F,
      100.0F, 100.0F, 30.0F, 30.0F, 1.0F, 0.70F};
  const auto nms = postprocess_balldet(data, 4, CameraId::Left, 0.25F, 1.0F, 0.0F, 0.0F, 1000,
                                       1000);
  expect_eq(nms.size(), 2U, "ball nms count");
  expect_near(nms[0].confidence, 0.90F, 1.0e-6F, "ball nms highest first");

  expect_true(postprocess_balldet({0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.90F}, 1,
                                  CameraId::Left, 0.25F, 0.0F, 0.0F, 0.0F, 1000, 1000)
                  .empty(),
              "derived nan ball coords filtered");

  const auto equal_conf = postprocess_balldet(
      {500.0F, 250.0F, 40.0F, 40.0F, 1.0F, 0.90F, 505.0F, 252.0F, 40.0F, 40.0F, 1.0F,
       0.90F},
      2, CameraId::Left, 0.25F, 1.0F, 0.0F, 0.0F, 1000, 1000);
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
  const AiProbeResult empty{.providers = {}, .can_run_on_gpu_frames = false, .errors = {"ORT init"}};
  expect_eq(empty.best_provider(), std::string("unavailable"), "empty best provider");
  expect_true(!empty.is_available(), "probe unavailable");
}

} // namespace

int main() {
  stock_yolo_postprocess_matches_rust();
  nan_and_class_guards_match_rust();
  ball_detector_adapter_matches_rust();
  labels_and_probe_match_rust();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
