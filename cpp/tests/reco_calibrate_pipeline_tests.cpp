#include "reco/calibrate/pipeline.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

using namespace reco::calibrate;

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

GpuCalibrationRequest valid_request() {
  GpuCalibrationRequest request;
  request.left.path = "left.mp4";
  request.right.path = "right.mp4";
  request.left.lens_profile = "left-lens.json";
  request.right.lens_profile = "right-lens.json";
  request.config.num_frames = 8;
  request.config.akaze.detect_y_min = 0.25;
  request.config.akaze.detect_y_max = 0.85;
  request.config.matching.lowe_ratio = 0.75;
  request.config.matching.spatial_x_threshold = 0.5;
  request.config.optimizer.trim_fraction = 0.3;
  request.config.optimizer.seam_sigma = 0.08;
  request.auto_sync = false;
  request.output = "match.json";
  request.probe_worker = "/opt/reco/reco_video_probe_worker";
  return request;
}

CalibrationBackendStatus ready_backends() {
  return {
      .cuda = {.available = true, .detail = "cuda"},
      .gstreamer = {.available = true, .detail = "gstreamer"},
      .npp = {.available = true, .detail = "npp"},
      .nvbufsurface = {.available = true, .detail = "nvbufsurface"},
  };
}

void validation_rejects_invalid_requests() {
  auto request = valid_request();
  request.left.path.clear();
  auto error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "empty left path rejected");
  expect_true(error->find("left video path") != std::string::npos, "left path error detail");

  request = valid_request();
  request.config.num_frames = 0;
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "zero calibration frames rejected");

  request = valid_request();
  request.config.akaze.detect_y_min = 0.9;
  request.config.akaze.detect_y_max = 0.1;
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "invalid Y detection range rejected");

  request = valid_request();
  request.config.skip_start_secs = std::numeric_limits<double>::quiet_NaN();
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "nan skip start rejected");

  request = valid_request();
  request.config.akaze.detect_y_min = std::numeric_limits<double>::quiet_NaN();
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "nan Y detection range rejected");

  request = valid_request();
  request.manual_sync_offset = reco::core::kMaxSyncOffsetFrames + 1;
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "out-of-range sync offset rejected");

  request = valid_request();
  request.left.lens_profile.reset();
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "right-only lens profile rejected");

  request = valid_request();
  request.probe_worker = "relative/probe-worker";
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "relative probe worker rejected");
}

void plan_keeps_calibration_gpu_resident() {
  const auto plan = build_gpu_calibration_plan(valid_request(), ready_backends());
  expect_true(plan.gpu_resident, "ready plan is marked GPU resident");
  expect_eq(plan.steps.size(), 7U, "all Rust calibration stages are represented");
  expect_true(plan.ready, "complete GPU calibration pipeline is ready");
  expect_true(!plan.blocked_reason.has_value(), "ready plan has no blocked reason");

  const auto description = describe_calibration_plan(plan);
  expect_true(description.find("(device-resident)") != std::string::npos,
              "ready plan reports verified device residency");
  expect_true(description.find("nvv4l2decoder") != std::string::npos,
              "plan describes hardware decode pipeline");
  expect_true(description.find("qtdemux ! capsfilter caps=\"video/x-h264;video/x-h265\" ! "
                               "parsebin ! identity name=display_info silent=true ! "
                               "nvv4l2decoder") != std::string::npos,
              "containerized inputs select a supported video pad");
  expect_true(description.find("video/x-raw(memory:NVMM),format=NV12") != std::string::npos,
              "plan preserves NVMM decode caps");
  expect_true(description.find("Undistorting") != std::string::npos,
              "plan describes GPU undistort stage");
  expect_true(description.find("FeatureMatching") != std::string::npos,
              "plan describes feature matching stage");
}

void calibration_plan_selects_hevc_decode_for_hevc_paths() {
  auto request = valid_request();
  request.left.path = "left.hevc";
  request.right.path = "right.h265";
  const auto plan = build_gpu_calibration_plan(request, ready_backends());
  const auto description = describe_calibration_plan(plan);
  expect_true(description.find("h265parse ! identity name=display_info silent=true ! "
                               "nvv4l2decoder") != std::string::npos,
              "calibration HEVC inputs use HEVC parser");
  expect_true(description.find("qtdemux ! h265parse") == std::string::npos,
              "calibration raw HEVC inputs bypass qtdemux");
}

void plan_reports_missing_required_backend_first() {
  auto backends = ready_backends();
  backends.cuda = {.available = false, .detail = "no CUDA driver"};
  auto plan = build_gpu_calibration_plan(valid_request(), backends);
  expect_true(!plan.ready, "missing CUDA blocks plan");
  expect_true(plan.blocked_reason->find("CUDA is required") != std::string::npos,
              "CUDA block is reported");

  backends = ready_backends();
  backends.gstreamer = {.available = false, .detail = "no gst"};
  plan = build_gpu_calibration_plan(valid_request(), backends);
  expect_true(!plan.ready, "missing GStreamer blocks plan");
  expect_true(plan.blocked_reason->find("GStreamer is required") != std::string::npos,
              "GStreamer block is reported");

  backends = ready_backends();
  backends.npp = {.available = false, .detail = "no npp"};
  plan = build_gpu_calibration_plan(valid_request(), backends);
  expect_true(!plan.ready, "missing NPP blocks plan");
  expect_true(plan.blocked_reason->find("NPP is required") != std::string::npos,
              "NPP block is reported");

  backends = ready_backends();
  backends.nvbufsurface = {.available = false, .detail = "no nvbufsurface"};
  plan = build_gpu_calibration_plan(valid_request(), backends);
  expect_true(!plan.ready, "missing NvBufSurface blocks plan");
  expect_true(plan.blocked_reason->find("NvBufSurface is required") != std::string::npos,
              "NvBufSurface block is reported");
}

void plan_reports_unported_optional_stages() {
  auto request = valid_request();
  request.probe_worker.clear();
  auto plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "missing probe worker blocks plan");
  expect_true(plan.blocked_reason->find("probe worker") != std::string::npos,
              "missing probe worker is reported");

  request = valid_request();
  request.left.lens_profile.reset();
  request.right.lens_profile.reset();
  plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "automatic lens detection blocks plan");
  expect_true(plan.blocked_reason->find("lens profile detection") != std::string::npos,
              "automatic lens detection gap is reported");

  request = valid_request();
  request.auto_sync = true;
  plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "automatic sync blocks plan");
  expect_true(plan.blocked_reason->find("sync extraction") != std::string::npos,
              "automatic sync gap is reported");

  request = valid_request();
  request.debug_dir = "debug";
  plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "debug export blocks plan");
  expect_true(plan.blocked_reason->find("debug image export") != std::string::npos,
              "debug export gap is reported");
}

void invalid_decode_paths_block_calibration_plan() {
  auto request = valid_request();
  request.left.path = "left.mp4 ! fakesink";
  const auto plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "invalid decode path blocks plan");
  expect_true(plan.blocked_reason.has_value(), "invalid decode path reason present");
  expect_true(plan.blocked_reason->find("metacharacters") != std::string::npos,
              "invalid decode path explains launch-string risk");
}

} // namespace

int main() {
  validation_rejects_invalid_requests();
  plan_keeps_calibration_gpu_resident();
  calibration_plan_selects_hevc_decode_for_hevc_paths();
  plan_reports_missing_required_backend_first();
  plan_reports_unported_optional_stages();
  invalid_decode_paths_block_calibration_plan();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
