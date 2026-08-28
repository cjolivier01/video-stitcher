#include "calibration_worker_internal.hpp"
#include "calibration_worker_protocol.hpp"
#include "reco/calibrate/pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace reco::calibrate;
using namespace reco::calibrate::detail;

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

template <typename Function>
void expect_error(Function&& function, std::string_view needle, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find(needle) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " error=" << error.what() << '\n';
      ++failures;
    }
  }
}

GpuCalibrationRequest request_fixture() {
  GpuCalibrationRequest request;
  request.left.path = "left.mp4";
  request.left.lens_profile = "/profiles/left.json";
  request.left.retained_path = "/proc/100/fd/10";
  request.right.path = "right.h265";
  request.right.lens_profile = "/profiles/right.json";
  request.right.retained_path = "/proc/100/fd/11";
  request.config.num_frames = 11;
  request.config.skip_start_secs = 1.25;
  request.config.skip_end_secs = 2.5;
  request.config.use_imu_rotation_seeds = true;
  request.config.imu_xrz_seed = 0.1;
  request.config.imu_zrx_seed = -0.2;
  request.config.akaze.max_keypoints = 4096;
  request.config.matching.min_matches = 12;
  request.config.optimizer.lock_cam_d = true;
  request.config.optimizer.enable_x_rx = true;
  request.config.optimizer.max_iters = 777;
  request.no_auto_imu = true;
  request.auto_sync = false;
  request.manual_sync_offset = -7;
  request.output = "/output/match.json";
  request.probe_worker = "/workers/reco_video_probe_worker";
  request.probe_timeout_ns = 12'000'000'000ULL;
  request.calibration_worker_path = "/workers/reco_calibration_worker";
  request.calibration_timeout_ns = 90'000'000'000ULL;
  request.calibration_host_memory_limit_bytes = 768ULL * 1024ULL * 1024ULL;
  request.nvbufsurface_abi = reco::io::NvbufSurfaceAbi::DeepStream9_1;
  return request;
}

reco::core::CameraParams camera_fixture() {
  return {.width = 1920,
          .height = 1080,
          .fx = 960.0,
          .fy = 961.0,
          .cx = 959.5,
          .cy = 539.5,
          .d = {-0.1, 0.02, 0.001, -0.001}};
}

CalibrationResult result_fixture() {
  FrameMatches frame;
  frame.points.push_back(MatchedPoint::from_planes({0.1, 0.2}, {0.3, 0.4}));
  frame.keypoints_left = 100;
  frame.keypoints_right = 90;
  frame.min_descriptors = 90;
  frame.post_ratio_test = 50;
  frame.post_spatial_filter = 30;
  frame.post_ransac = 20;
  CalibrationResult result;
  result.calibration.left = camera_fixture();
  result.calibration.right = camera_fixture();
  result.calibration.layout = {.camera_axis_offset = 0.25,
                               .intersect = 0.5,
                               .x_ty = 0.1,
                               .x_rz = -0.02,
                               .z_rx = 0.03,
                               .x_rx = 0.01,
                               .z_rz = -0.01};
  result.calibration.sync_offset = -7;
  result.calibration.field_roi =
      reco::core::FieldRoi{.left = {{0.0, 0.0}, {1.0, 1.0}}, .right = {{0.1, 0.2}, {0.9, 0.8}}};
  result.total_matches = 20;
  result.frames_used = 1;
  result.residual_error = 0.125;
  result.confidence = 0.8;
  result.per_frame.push_back(std::move(frame));
  result.left_lens_profile = LensProfileInfo{.camera = "GoPro",
                                             .lens = "Wide",
                                             .source = ProfileSource::File,
                                             .path = "/profiles/left.json"};
  result.quality = CalibrationQuality{
      .mean_reprojection_error = 0.2, .trimmed_reprojection_error = 0.1, .angular_error = 0.05};
  return result;
}

std::array<char, sizeof(std::uint64_t)> encoded_u64(std::uint64_t value) {
  std::array<char, sizeof(value)> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<char>(value >> ((bytes.size() - index - 1U) * 8U));
  }
  return bytes;
}

std::size_t replace_u64(std::string& bytes, std::uint64_t from, std::uint64_t to,
                        std::optional<std::size_t> occurrence = std::nullopt) {
  const auto source = encoded_u64(from);
  const auto replacement = encoded_u64(to);
  std::size_t found = 0;
  for (std::size_t offset = kCalibrationWorkerFrameHeaderBytes;
       offset + source.size() <= bytes.size(); ++offset) {
    if (!std::equal(source.begin(), source.end(), bytes.begin() + offset)) {
      continue;
    }
    if (!occurrence.has_value() || found == *occurrence) {
      std::copy(replacement.begin(), replacement.end(), bytes.begin() + offset);
      if (occurrence.has_value()) {
        return 1;
      }
    }
    ++found;
  }
  return occurrence.has_value() ? 0 : found;
}

CalibrationResult compact_result_fixture() {
  auto result = result_fixture();
  auto& summary = result.per_frame.front();
  summary.keypoints_left = 101;
  summary.keypoints_right = 97;
  summary.min_descriptors = 97;
  summary.post_ratio_test = 53;
  summary.post_spatial_filter = 31;
  summary.post_ransac = 17;
  result.total_matches = 17;
  result.calibration.field_roi.reset();
  result.left_lens_profile.reset();
  result.right_lens_profile.reset();
  result.quality.reset();
  return result;
}

void request_round_trip_is_exact_and_bounded() {
  const auto expected = request_fixture();
  const auto encoded = encode_calibration_worker_request(expected);
  expect_true(encoded.size() <= kMaximumCalibrationWorkerFrameBytes,
              "request frame respects hard byte ceiling");
  const auto decoded = decode_calibration_worker_request(encoded);
  expect_eq(decoded.left.path, expected.left.path, "left path round trip");
  expect_eq(*decoded.left.retained_path, *expected.left.retained_path,
            "left retained path round trip");
  expect_eq(*decoded.right.lens_profile, *expected.right.lens_profile,
            "right lens path round trip");
  expect_eq(*decoded.right.retained_path, *expected.right.retained_path,
            "right retained path round trip");
  expect_eq(decoded.config.num_frames, expected.config.num_frames, "frame count round trip");
  expect_eq(decoded.config.optimizer.max_iters, expected.config.optimizer.max_iters,
            "optimizer iterations round trip");
  expect_eq(decoded.manual_sync_offset, expected.manual_sync_offset, "sync offset round trip");
  expect_eq(decoded.calibration_timeout_ns, expected.calibration_timeout_ns, "deadline round trip");
  expect_eq(decoded.calibration_host_memory_limit_bytes,
            expected.calibration_host_memory_limit_bytes, "memory limit round trip");
  expect_true(decoded.nvbufsurface_abi == reco::io::NvbufSurfaceAbi::DeepStream9_1,
              "NvBufSurface ABI round trip");
}

void result_round_trip_transfers_only_compact_summaries() {
  const auto expected = result_fixture();
  const auto encoded = encode_calibration_worker_success(expected);
  expect_true(encoded.size() <= kMaximumCalibrationWorkerFrameBytes,
              "response frame respects hard byte ceiling");
  const auto decoded = decode_calibration_worker_response(encoded);
  expect_eq(decoded.total_matches, expected.total_matches, "total matches round trip");
  expect_eq(decoded.frames_used, expected.frames_used, "used frames round trip");
  expect_eq(decoded.per_frame.size(), 1U, "one frame summary transferred");
  expect_eq(decoded.per_frame[0].post_ransac, 20U, "frame count summary round trip");
  expect_true(decoded.per_frame[0].points.empty(),
              "raw correspondences do not cross the worker boundary");
  expect_eq(decoded.calibration.field_roi->left.size(), 2U, "bounded field ROI round trip");
  expect_eq(decoded.left_lens_profile->camera, std::string("GoPro"), "lens metadata round trip");
}

void malformed_frames_fail_before_unbounded_allocation() {
  auto request = encode_calibration_worker_request(request_fixture());

  auto bad_version = request;
  bad_version[4] = 0;
  bad_version[5] = 3;
  expect_error([&] { (void)decode_calibration_worker_request(bad_version); }, "version",
               "unknown protocol version rejected");

  auto oversized = request;
  constexpr auto impossible = static_cast<std::uint32_t>(kMaximumCalibrationWorkerFrameBytes);
  oversized[8] = static_cast<char>(impossible >> 24U);
  oversized[9] = static_cast<char>(impossible >> 16U);
  oversized[10] = static_cast<char>(impossible >> 8U);
  oversized[11] = static_cast<char>(impossible);
  expect_error([&] { (void)decode_calibration_worker_request(oversized); }, "payload size",
               "oversized declared payload rejected from its header");

  auto truncated = request;
  truncated.pop_back();
  expect_error([&] { (void)decode_calibration_worker_request(truncated); }, "length",
               "truncated request rejected");

  auto trailing = request;
  trailing.push_back('x');
  expect_error([&] { (void)decode_calibration_worker_request(trailing); }, "length",
               "trailing request bytes rejected");

  auto bad_abi = request;
  std::fill_n(bad_abi.end() - 4, 4, '\0');
  expect_error([&] { (void)decode_calibration_worker_request(bad_abi); }, "NvBufSurface ABI",
               "invalid ABI rejected");

  auto too_many_frames = result_fixture();
  too_many_frames.per_frame.resize(kMaximumCalibrationWorkerResultFrames + 1);
  expect_error([&] { (void)encode_calibration_worker_success(too_many_frames); }, "too many frame",
               "oversized result summary rejected");
}

void inconsistent_result_counts_are_rejected_on_encode_and_decode() {
  using Mutation = std::function<void(CalibrationResult&)>;
  const std::array<std::pair<std::string_view, Mutation>, 6> cases{{
      {"frame summary count", [](CalibrationResult& value) { value.frames_used = 2; }},
      {"minimum descriptor count",
       [](CalibrationResult& value) { value.per_frame[0].min_descriptors = 96; }},
      {"ratio count", [](CalibrationResult& value) { value.per_frame[0].post_ratio_test = 98; }},
      {"spatial count",
       [](CalibrationResult& value) { value.per_frame[0].post_spatial_filter = 54; }},
      {"RANSAC count", [](CalibrationResult& value) { value.per_frame[0].post_ransac = 32; }},
      {"total count", [](CalibrationResult& value) { value.total_matches = 16; }},
  }};
  for (const auto& [name, mutate] : cases) {
    auto result = compact_result_fixture();
    mutate(result);
    expect_error([&] { (void)encode_calibration_worker_success(result); }, "inconsistent",
                 std::string(name) + " is rejected by the encoder");
  }

  const auto expect_mutated_decode_error = [](std::string encoded, std::uint64_t from,
                                              std::uint64_t to, std::size_t occurrence,
                                              std::string_view name) {
    expect_eq(replace_u64(encoded, from, to, occurrence), 1U,
              std::string(name) + " decoder fixture mutation");
    expect_error([&] { (void)decode_calibration_worker_response(encoded); }, "inconsistent",
                 std::string(name) + " is rejected by the decoder");
  };
  const auto encoded = encode_calibration_worker_success(compact_result_fixture());
  expect_mutated_decode_error(encoded, 1, 2, 0, "frame summary count");
  expect_mutated_decode_error(encoded, 97, 96, 1, "minimum descriptor count");
  expect_mutated_decode_error(encoded, 53, 98, 0, "ratio count");
  expect_mutated_decode_error(encoded, 31, 54, 0, "spatial count");
  expect_mutated_decode_error(encoded, 17, 32, 1, "RANSAC count");
  expect_mutated_decode_error(encoded, 17, 16, 0, "total count");
}

void result_match_summation_overflow_is_rejected_on_encode_and_decode() {
  auto result = compact_result_fixture();
  auto first = result.per_frame.front();
  first.keypoints_left = 9;
  first.keypoints_right = 7;
  first.min_descriptors = 7;
  first.post_ratio_test = 7;
  first.post_spatial_filter = 7;
  first.post_ransac = 7;
  auto second = first;
  second.keypoints_left = 11;
  second.keypoints_right = 8;
  second.min_descriptors = 8;
  second.post_ratio_test = 8;
  second.post_spatial_filter = 8;
  second.post_ransac = 8;
  result.per_frame = {first, second};
  result.frames_used = 2;
  result.total_matches = 15;

  auto overflow = result;
  const auto maximum = std::numeric_limits<std::size_t>::max();
  for (auto& summary : overflow.per_frame) {
    summary.keypoints_left = maximum;
    summary.keypoints_right = maximum;
    summary.min_descriptors = maximum;
    summary.post_ratio_test = maximum;
    summary.post_spatial_filter = maximum;
    summary.post_ransac = maximum;
  }
  overflow.total_matches = maximum;
  expect_error([&] { (void)encode_calibration_worker_success(overflow); }, "inconsistent",
               "match summation overflow is rejected by the encoder");

  auto encoded = encode_calibration_worker_success(result);
  expect_eq(replace_u64(encoded, 9, maximum), 1U, "first keypoint overflow fixture mutation");
  expect_eq(replace_u64(encoded, 7, maximum), 5U, "first summary overflow fixture mutation");
  expect_eq(replace_u64(encoded, 11, maximum), 1U, "second keypoint overflow fixture mutation");
  expect_eq(replace_u64(encoded, 8, maximum), 5U, "second summary overflow fixture mutation");
  expect_eq(replace_u64(encoded, 15, maximum), 1U, "total overflow fixture mutation");
  expect_error([&] { (void)decode_calibration_worker_response(encoded); }, "inconsistent",
               "match summation overflow is rejected by the decoder");
}

void worker_converts_malformed_input_to_a_bounded_failure() {
  std::istringstream input("short");
  std::ostringstream output;
  expect_eq(run_calibration_worker(input, output), EXIT_FAILURE,
            "malformed worker input exits unsuccessfully");
  expect_true(output.str().size() <= kMaximumCalibrationWorkerFrameBytes,
              "worker failure remains bounded");
  expect_error([&] { (void)decode_calibration_worker_response(output.str()); }, "truncated",
               "worker reports malformed input without attempting calibration");
}

#if defined(__linux__)
std::uint64_t deadline_after(std::chrono::milliseconds duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch() + duration)
          .count());
}

void production_fd_transport_handles_delayed_request_and_response_backpressure() {
  std::array<int, 2> sockets{-1, -1};
  expect_true(
      ::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets.data()) == 0,
      "nonblocking worker transport socketpair opens");
  if (sockets[0] < 0 || sockets[1] < 0) {
    return;
  }
  std::vector<char> request(256U * 1024U);
  for (std::size_t index = 0; index < request.size(); ++index) {
    request[index] = static_cast<char>(index & 0xffU);
  }
  std::vector<char> received(request.size());
  std::exception_ptr read_error;
  std::thread reader([&] {
    try {
      read_calibration_worker_bytes_fd(sockets[1], received,
                                       deadline_after(std::chrono::seconds(2)));
    } catch (...) {
      read_error = std::current_exception();
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  try {
    write_calibration_worker_bytes_fd(sockets[0], std::string_view(request.data(), request.size()),
                                      deadline_after(std::chrono::seconds(2)));
  } catch (const std::exception& error) {
    std::cerr << "FAIL: delayed worker transport write error=" << error.what() << '\n';
    ++failures;
  }
  reader.join();
  expect_true(read_error == nullptr, "production worker transport waits for a delayed request");
  expect_true(received == request, "production worker transport preserves delayed request bytes");

  int send_buffer = 1024;
  expect_true(::setsockopt(sockets[1], SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer)) ==
                  0,
              "worker response send buffer is constrained");
  std::vector<char> response(512U * 1024U, 'r');
  std::vector<char> response_copy(response.size());
  std::exception_ptr write_error;
  std::thread writer([&] {
    try {
      write_calibration_worker_bytes_fd(sockets[1],
                                        std::string_view(response.data(), response.size()),
                                        deadline_after(std::chrono::seconds(2)));
    } catch (...) {
      write_error = std::current_exception();
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  try {
    read_calibration_worker_bytes_fd(sockets[0], response_copy,
                                     deadline_after(std::chrono::seconds(2)));
  } catch (const std::exception& error) {
    std::cerr << "FAIL: backpressured worker transport read error=" << error.what() << '\n';
    ++failures;
  }
  writer.join();
  expect_true(write_error == nullptr,
              "production worker transport waits for response backpressure to clear");
  expect_true(response_copy == response,
              "production worker transport preserves a backpressured response");

  (void)::close(sockets[0]);
  (void)::close(sockets[1]);
}

void production_fd_transport_enforces_its_deadline() {
  std::array<int, 2> sockets{-1, -1};
  expect_true(
      ::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets.data()) == 0,
      "deadline worker transport socketpair opens");
  if (sockets[0] < 0 || sockets[1] < 0) {
    return;
  }
  std::array<char, 1> byte{};
  const auto started = std::chrono::steady_clock::now();
  expect_error(
      [&] {
        read_calibration_worker_bytes_fd(sockets[0], byte,
                                         deadline_after(std::chrono::milliseconds(30)));
      },
      "truncated", "production worker transport rejects a request after its deadline");
  expect_true(std::chrono::steady_clock::now() - started < std::chrono::milliseconds(250),
              "production worker transport deadline is bounded");
  (void)::close(sockets[0]);
  (void)::close(sockets[1]);
}
#endif

} // namespace

int main() {
  request_round_trip_is_exact_and_bounded();
  result_round_trip_transfers_only_compact_summaries();
  malformed_frames_fail_before_unbounded_allocation();
  inconsistent_result_counts_are_rejected_on_encode_and_decode();
  result_match_summation_overflow_is_rejected_on_encode_and_decode();
  worker_converts_malformed_input_to_a_bounded_failure();
#if defined(__linux__)
  production_fd_transport_handles_delayed_request_and_response_backpressure();
  production_fd_transport_enforces_its_deadline();
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
