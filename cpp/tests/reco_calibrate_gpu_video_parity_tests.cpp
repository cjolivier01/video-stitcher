#include "reco/calibrate/pipeline.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gstreamer.hpp"
#include "reco/io/nvmm.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace reco::calibrate;
using namespace reco::core;
using namespace reco::io;

namespace {

using Json = nlohmann::json;

constexpr std::array<std::uint64_t, 3> kFrameIndices{1, 3, 5};
constexpr std::array<std::uint64_t, 3> kFramePtsNs{100'000'000, 300'000'000, 500'000'000};
constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 360;

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

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual
              << " tolerance=" << tolerance << '\n';
    ++failures;
  }
}

bool require_gpu() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

std::filesystem::path runfile(std::string path) {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
  const auto resolved =
      std::filesystem::path(runfiles->Rlocation(std::string(workspace) + "/" + path));
  if (resolved.empty() || !std::filesystem::is_regular_file(resolved)) {
    throw std::runtime_error(path + " runfile not found");
  }
  return resolved;
}

Json load_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open " + path.string());
  }
  Json value;
  input >> value;
  return value;
}

GpuFileDecodeConfig decode_config(const std::filesystem::path& path) {
  const auto text = path.string();
  return {.path = text,
          .codec = gpu_decode_codec_for_path(text),
          .elementary_stream = gpu_decode_path_is_elementary_stream(text),
          .container = gpu_decode_container_for_path(text),
          .max_buffers = 2,
          .drop = false,
          .read_timeout_ns = 30'000'000'000ULL,
          .indexed_fps_numerator = 10,
          .indexed_fps_denominator = 1,
          .indexed_timestamp_multiplicity = 1,
          .indexed_stream_time_origin_ns = 0,
          .start_frame_index = kFrameIndices.front()};
}

CameraParams camera() {
  return {.width = kWidth,
          .height = kHeight,
          .fx = 1.0e10,
          .fy = 1.0e10,
          .cx = 320.0,
          .cy = 180.0,
          .d = {0.0, 0.0, 0.0, 0.0}};
}

CalibrationConfig config() {
  CalibrationConfig result;
  result.num_frames = kFrameIndices.size();
  result.akaze.threshold = 0.0001;
  result.akaze.max_keypoints = 128;
  result.akaze.detect_y_min = 0.15;
  result.akaze.detect_y_max = 0.85;
  result.matching.lowe_ratio = 1.0;
  result.matching.min_matches = 8;
  result.matching.spatial_x_threshold = 0.5;
  result.matching.spatial_x_inner = 0.0;
  result.matching.spatial_y_low = 0.15;
  result.matching.spatial_y_high = 0.85;
  result.matching.max_y_disparity = 0.02;
  result.matching.ransac_threshold = 1.0;
  result.optimizer.max_iters = 5'000;
  return result;
}

void verify_frame_identity(const std::filesystem::path& path, NvbufSurfaceAbi abi,
                           std::string_view side) {
  auto source = open_gstreamer_gpu_file_decode_source(decode_config(path), abi);
  expect_true(source->gpu_resident(), std::string(side) + " decoder reports GPU residency");
  expect_true(source->pipeline().find("nvv4l2decoder") != std::string_view::npos,
              std::string(side) + " decoder uses NVIDIA hardware decode");
  expect_true(source->pipeline().find("video/x-raw(memory:NVMM)") != std::string_view::npos,
              std::string(side) + " decoder retains NVMM surfaces");
  for (std::size_t index = 0; index < kFrameIndices.size(); ++index) {
    if (index != 0U) {
      source->seek_to_frame(kFrameIndices[index]);
    }
    auto decoded = source->read();
    expect_true(decoded.status == GpuDecodeFrameStatus::Frame && decoded.frame.has_value(),
                std::string(side) + " selected frame exists");
    if (!decoded.frame.has_value()) {
      continue;
    }
    expect_eq(decoded.frame->frame_index, kFrameIndices[index],
              std::string(side) + " absolute frame index");
    expect_eq(decoded.frame->pts_ns.value_or(std::numeric_limits<std::uint64_t>::max()),
              kFramePtsNs[index], std::string(side) + " presentation timestamp");
    expect_eq(decoded.frame->visible_width, kWidth, std::string(side) + " visible width");
    expect_eq(decoded.frame->visible_height, kHeight, std::string(side) + " visible height");
    const auto mapped = map_gpu_decoded_frame_to_cuda(*decoded.frame);
    expect_true(mapped.y_ptr != 0 && mapped.uv_ptr != 0,
                std::string(side) + " frame maps to CUDA without host staging");
    expect_eq(mapped.width, kWidth, std::string(side) + " CUDA frame width");
    expect_eq(mapped.height, kHeight, std::string(side) + " CUDA frame height");
  }
}

struct CopyShape {
  std::size_t width_bytes = 0;
  std::size_t height = 0;
};

struct DeviceToHostTrace final : CudaBackendTraceSink {
  void device_to_host_copy_submitted(std::size_t width_bytes,
                                     std::size_t height) noexcept override {
    if (count == copies.size()) {
      overflow = true;
      return;
    }
    copies[count++] = {.width_bytes = width_bytes, .height = height};
  }

  std::array<CopyShape, 256> copies{};
  std::size_t count = 0;
  bool overflow = false;
};

Json point_json(const MatchedPoint& point) {
  return {{"left", point.left},
          {"right", point.right},
          {"left_pixel_nx", point.left_pixel_nx},
          {"right_pixel_nx", point.right_pixel_nx}};
}

Json result_json(const CalibrationResult& result) {
  Json per_frame = Json::array();
  for (const auto& frame : result.per_frame) {
    Json points = Json::array();
    for (const auto& point : frame.points) {
      points.push_back(point_json(point));
    }
    per_frame.push_back({{"points", std::move(points)},
                         {"keypoints_left", frame.keypoints_left},
                         {"keypoints_right", frame.keypoints_right},
                         {"min_descriptors", frame.min_descriptors},
                         {"post_ratio_test", frame.post_ratio_test},
                         {"post_spatial_filter", frame.post_spatial_filter},
                         {"post_ransac", frame.post_ransac}});
  }
  const auto& calibration = result.calibration;
  const auto& layout = calibration.layout;
  Json quality = nullptr;
  if (result.quality.has_value()) {
    quality = {{"mean_reprojection_error", result.quality->mean_reprojection_error},
               {"trimmed_reprojection_error", result.quality->trimmed_reprojection_error},
               {"angular_error", result.quality->angular_error}};
  }
  const auto camera_json = [](const CameraParams& value) {
    return Json{{"width", value.width}, {"height", value.height}, {"fx", value.fx},
                {"fy", value.fy},       {"cx", value.cx},         {"cy", value.cy},
                {"d", value.d}};
  };
  return {{"calibration",
           {{"left_uniforms", camera_json(calibration.left)},
            {"right_uniforms", camera_json(calibration.right)},
            {"params",
             {{"cameraAxisOffset", layout.camera_axis_offset},
              {"intersect", layout.intersect},
              {"xTy", layout.x_ty},
              {"xRz", layout.x_rz},
              {"zRx", layout.z_rx},
              {"xRx", layout.x_rx},
              {"zRz", layout.z_rz}}},
            {"rig_tilt", calibration.rig_tilt},
            {"rig_roll", calibration.rig_roll},
            {"sync_offset", calibration.sync_offset},
            {"field_roi", nullptr},
            {"lens_correction_amount", calibration.lens_correction_amount},
            {"blend_width", calibration.blend_width}}},
          {"total_matches", result.total_matches},
          {"frames_used", result.frames_used},
          {"residual_error", result.residual_error},
          {"confidence", result.confidence},
          {"per_frame", std::move(per_frame)},
          {"left_lens_profile", nullptr},
          {"right_lens_profile", nullptr},
          {"quality", std::move(quality)}};
}

void expect_count_near(std::size_t actual, std::size_t expected, std::size_t tolerance,
                       const std::string& message) {
  const auto delta = actual > expected ? actual - expected : expected - actual;
  if (delta > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual
              << " tolerance=" << tolerance << '\n';
    ++failures;
  }
}

bool point_matches(const MatchedPoint& actual, const Json& expected) {
  constexpr double kPointTolerance = 5.0e-4;
  const auto near = [](double lhs, double rhs) {
    return std::isfinite(lhs) && std::isfinite(rhs) && std::abs(lhs - rhs) <= kPointTolerance;
  };
  return near(actual.left[0], expected.at("left").at(0).get<double>()) &&
         near(actual.left[1], expected.at("left").at(1).get<double>()) &&
         near(actual.right[0], expected.at("right").at(0).get<double>()) &&
         near(actual.right[1], expected.at("right").at(1).get<double>()) &&
         near(actual.left_pixel_nx, expected.at("left_pixel_nx").get<double>()) &&
         near(actual.right_pixel_nx, expected.at("right_pixel_nx").get<double>());
}

void compare_frame(const FrameMatches& actual, const Json& expected, std::size_t frame_index) {
  const auto label = "frame " + std::to_string(frame_index);
  expect_eq(actual.keypoints_left, expected.at("keypoints_left").get<std::size_t>(),
            label + " left keypoints");
  expect_eq(actual.keypoints_right, expected.at("keypoints_right").get<std::size_t>(),
            label + " right keypoints");
  expect_eq(actual.min_descriptors, expected.at("min_descriptors").get<std::size_t>(),
            label + " minimum descriptors");
  expect_count_near(actual.post_ratio_test, expected.at("post_ratio_test").get<std::size_t>(), 2,
                    label + " ratio-test matches");
  expect_count_near(actual.post_spatial_filter,
                    expected.at("post_spatial_filter").get<std::size_t>(), 2,
                    label + " spatial matches");
  expect_count_near(actual.post_ransac, expected.at("post_ransac").get<std::size_t>(), 2,
                    label + " RANSAC inliers");

  const auto& expected_points = expected.at("points");
  expect_count_near(actual.points.size(), expected_points.size(), 2, label + " point count");
  std::vector<bool> used(actual.points.size(), false);
  std::size_t matched = 0;
  for (const auto& expected_point : expected_points) {
    for (std::size_t index = 0; index < actual.points.size(); ++index) {
      if (!used[index] && point_matches(actual.points[index], expected_point)) {
        used[index] = true;
        ++matched;
        break;
      }
    }
  }
  const auto required = std::min(actual.points.size(), expected_points.size());
  expect_count_near(matched, required, 2, label + " Rust correspondence overlap");
}

void compare_result(const CalibrationResult& actual, const Json& expected) {
  constexpr double kLayoutTolerance = 2.0e-3;
  constexpr double kQualityTolerance = 1.0e-2;
  const auto& expected_calibration = expected.at("calibration");
  const auto& expected_layout = expected_calibration.at("params");
  const auto compare_camera = [](const CameraParams& camera, const Json& golden,
                                 std::string_view side) {
    expect_eq(camera.width, golden.at("width").get<std::uint32_t>(),
              std::string(side) + " Rust camera width");
    expect_eq(camera.height, golden.at("height").get<std::uint32_t>(),
              std::string(side) + " Rust camera height");
    expect_near(camera.fx, golden.at("fx").get<double>(), 1.0e-12,
                std::string(side) + " Rust camera fx");
    expect_near(camera.fy, golden.at("fy").get<double>(), 1.0e-12,
                std::string(side) + " Rust camera fy");
    expect_near(camera.cx, golden.at("cx").get<double>(), 1.0e-12,
                std::string(side) + " Rust camera cx");
    expect_near(camera.cy, golden.at("cy").get<double>(), 1.0e-12,
                std::string(side) + " Rust camera cy");
    for (std::size_t index = 0; index < camera.d.size(); ++index) {
      expect_near(camera.d[index], golden.at("d").at(index).get<double>(), 1.0e-12,
                  std::string(side) + " Rust camera distortion");
    }
  };
  compare_camera(actual.calibration.left, expected_calibration.at("left_uniforms"), "left");
  compare_camera(actual.calibration.right, expected_calibration.at("right_uniforms"), "right");
  const auto& layout = actual.calibration.layout;
  expect_near(layout.camera_axis_offset, expected_layout.at("cameraAxisOffset").get<double>(),
              kLayoutTolerance, "Rust layout camera axis offset");
  expect_near(layout.intersect, expected_layout.at("intersect").get<double>(), kLayoutTolerance,
              "Rust layout intersection");
  expect_near(layout.x_ty, expected_layout.at("xTy").get<double>(), kLayoutTolerance,
              "Rust layout xTy");
  expect_near(layout.x_rz, expected_layout.at("xRz").get<double>(), kLayoutTolerance,
              "Rust layout xRz");
  expect_near(layout.z_rx, expected_layout.at("zRx").get<double>(), kLayoutTolerance,
              "Rust layout zRx");
  expect_near(layout.x_rx, expected_layout.at("xRx").get<double>(), kLayoutTolerance,
              "Rust layout xRx");
  expect_near(layout.z_rz, expected_layout.at("zRz").get<double>(), kLayoutTolerance,
              "Rust layout zRz");
  expect_near(actual.calibration.rig_tilt, expected_calibration.at("rig_tilt").get<double>(),
              1.0e-12, "Rust calibration rig tilt");
  expect_near(actual.calibration.rig_roll, expected_calibration.at("rig_roll").get<double>(),
              1.0e-12, "Rust calibration rig roll");
  expect_eq(actual.calibration.sync_offset,
            expected_calibration.at("sync_offset").get<std::int64_t>(),
            "Rust calibration sync offset");
  expect_true(!actual.calibration.field_roi.has_value(), "parity calibration has no field ROI");
  expect_near(actual.calibration.lens_correction_amount,
              expected_calibration.at("lens_correction_amount").get<double>(), 1.0e-7,
              "Rust lens correction amount");
  expect_near(actual.calibration.blend_width, expected_calibration.at("blend_width").get<double>(),
              1.0e-7, "Rust blend width");
  expect_eq(actual.frames_used, expected.at("frames_used").get<std::size_t>(),
            "Rust usable frame count");
  expect_count_near(actual.total_matches, expected.at("total_matches").get<std::size_t>(), 6,
                    "Rust total correspondence count");
  expect_near(actual.residual_error, expected.at("residual_error").get<double>(), kQualityTolerance,
              "Rust optimizer residual");
  expect_near(actual.confidence, expected.at("confidence").get<double>(), 1.0e-12,
              "Rust confidence");
  expect_true(!actual.left_lens_profile.has_value() && !actual.right_lens_profile.has_value(),
              "direct parity calibration has no profile metadata");
  expect_true(actual.quality.has_value(), "parity calibration reports quality metrics");
  if (actual.quality.has_value()) {
    const auto& quality = expected.at("quality");
    expect_near(actual.quality->mean_reprojection_error,
                quality.at("mean_reprojection_error").get<double>(), kQualityTolerance,
                "Rust mean reprojection error");
    expect_near(actual.quality->trimmed_reprojection_error,
                quality.at("trimmed_reprojection_error").get<double>(), kQualityTolerance,
                "Rust trimmed reprojection error");
    expect_near(actual.quality->angular_error, quality.at("angular_error").get<double>(),
                kQualityTolerance, "Rust angular error");
  }
  const auto& expected_frames = expected.at("per_frame");
  expect_eq(actual.per_frame.size(), expected_frames.size(), "Rust per-frame result count");
  for (std::size_t index = 0; index < std::min(actual.per_frame.size(), expected_frames.size());
       ++index) {
    compare_frame(actual.per_frame[index], expected_frames[index], index);
  }
}

CalibrationResult calibrate_once(CudaBackend& backend, const std::filesystem::path& left,
                                 const std::filesystem::path& right, NvbufSurfaceAbi abi,
                                 std::shared_ptr<DeviceToHostTrace> trace) {
  auto left_source = open_gstreamer_gpu_file_decode_source(decode_config(left), abi);
  auto right_source = open_gstreamer_gpu_file_decode_source(decode_config(right), abi);
  auto observed = backend.with_trace_sink(std::move(trace));
  return run_gpu_calibration_sources(observed, *left_source, *right_source, kFrameIndices,
                                     kFrameIndices, camera(), camera(), config());
}

void decoded_video_matches_rust_golden(const std::filesystem::path& left,
                                       const std::filesystem::path& right,
                                       const std::filesystem::path& golden_path,
                                       NvbufSurfaceAbi abi) {
  const auto golden = load_json(golden_path);
  expect_eq(golden.at("schema_version").get<unsigned>(), 1U, "golden schema version");
  expect_true(golden.at("frame_indices") == kFrameIndices, "golden records selected frame indices");
  expect_true(golden.at("frame_pts_ns") == kFramePtsNs,
              "golden records selected presentation timestamps");

  verify_frame_identity(left, abi, "left");
  verify_frame_identity(right, abi, "right");

  auto backend = CudaBackend::create();
  auto first_trace = std::make_shared<DeviceToHostTrace>();
  const auto first = calibrate_once(backend, left, right, abi, first_trace);
  auto second_trace = std::make_shared<DeviceToHostTrace>();
  const auto second = calibrate_once(backend, left, right, abi, second_trace);
  const auto first_json = result_json(first);
  const auto second_json = result_json(second);
  expect_true(first_json == second_json, "repeated CUDA calibration is byte-deterministic");
  expect_true(!first_trace->overflow, "calibration compact readback trace stays bounded");
  expect_true(first_trace->count != 0U, "calibration records bounded compact readback");
  for (std::size_t index = 0; index < first_trace->count; ++index) {
    const auto copy = first_trace->copies[index];
    expect_eq(copy.height, 1U, "calibration readback is one-dimensional compact data");
    expect_true(copy.width_bytes < static_cast<std::size_t>(kWidth) * kHeight,
                "calibration never reads a decoded or undistorted frame to the host");
  }
  expect_true(!second_trace->overflow, "repeated calibration readback trace stays bounded");
  expect_true(first_trace->count == second_trace->count,
              "repeated calibration has deterministic transfer count");
  compare_result(first, golden.at("result"));
}

} // namespace

int main() {
#if !defined(__linux__)
  std::cerr << "SKIP: decoded NVMM calibration parity requires Linux\n";
  return EXIT_SUCCESS;
#else
  try {
    const auto gstreamer = probe_gstreamer_runtime();
    const bool available =
        CudaBackend::is_available() && gstreamer.available && is_nvmm_cuda_interop_available();
    if (!available) {
      const std::string detail = !CudaBackend::is_available() ? CudaBackend::availability_error()
                                 : !gstreamer.available       ? gstreamer.error
                                                        : nvmm_cuda_interop_availability_error();
      if (require_gpu()) {
        std::cerr << "FAIL: GPU decoded-video parity unavailable: " << detail << '\n';
        return EXIT_FAILURE;
      }
      std::cerr << "SKIP: GPU decoded-video parity unavailable: " << detail << '\n';
      return EXIT_SUCCESS;
    }
    const auto abi = discover_nvbufsurface_abi();
    decoded_video_matches_rust_golden(
        runfile("cpp/tests/fixtures/calibration_video_parity/left.mp4"),
        runfile("cpp/tests/fixtures/calibration_video_parity/right.mp4"),
        runfile("cpp/tests/fixtures/calibration_video_parity/rust_golden.json"), abi);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
