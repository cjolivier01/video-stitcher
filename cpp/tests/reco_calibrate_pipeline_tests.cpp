#include "reco/calibrate/gpu_features.hpp"
#include "reco/calibrate/pipeline.hpp"
#include "reco/io/gpu_video_probe.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace reco::calibrate;

namespace {

int failures = 0;

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::random_device random;
    const auto base = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 128; ++attempt) {
      path_ = base / ("reco-calibration-output-identity-" + std::to_string(random()) + "-" +
                      std::to_string(random()));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error && error != std::errc::file_exists) {
        throw std::filesystem::filesystem_error("cannot create test directory", path_, error);
      }
    }
    throw std::runtime_error("cannot create unique calibration identity test directory");
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

private:
  std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << "video fixture";
  if (!output) {
    throw std::runtime_error("cannot create test file " + path.string());
  }
}

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
  request.no_auto_imu = true;
  request.auto_sync = false;
  request.output = "match.json";
  request.probe_worker = "/opt/reco/reco_video_probe_worker";
  request.calibration_worker_path = "/opt/reco/reco_calibration_worker";
  return request;
}

reco::io::GpuVideoProbe probe_fixture() {
  return {.width = 3840,
          .height = 2160,
          .fps_numerator = 30,
          .fps_denominator = 1,
          .fps = 30.0,
          .duration_ns = 400'000'000'000ULL,
          .total_frames = 12'000,
          .first_stream_time_ns = 0,
          .duration_is_estimated = true,
          .total_frames_is_estimated = true,
          .selected_stream_caps_verified = false,
          .indexed_sampling_cadence_verified = false};
}

void calibration_requires_exhaustive_indexed_cadence_proof() {
  auto probe = probe_fixture();
  expect_true(validate_gpu_calibration_probe_metadata(probe).has_value(),
              "estimated long input is rejected without an exact AU count");

  probe.total_frames_is_estimated = false;
  expect_true(validate_gpu_calibration_probe_metadata(probe).has_value(),
              "exact count without full-stream caps proof remains rejected");

  probe.selected_stream_caps_verified = true;
  expect_true(validate_gpu_calibration_probe_metadata(probe).has_value(),
              "exact caps without full-stream cadence proof remains rejected");

  probe.indexed_sampling_cadence_verified = true;
  expect_true(!validate_gpu_calibration_probe_metadata(probe).has_value(),
              "exhaustively verified indexed metadata is accepted");

  probe.timestamp_multiplicity = 2;
  expect_true(!validate_gpu_calibration_probe_metadata(probe).has_value(),
              "complete duplicate timestamp groups are accepted");
  probe.total_frames = 11'999;
  expect_true(validate_gpu_calibration_probe_metadata(probe).has_value(),
              "an incomplete final duplicate timestamp group is rejected");
  probe.total_frames = 12'000;
  probe.timestamp_multiplicity = 0;
  expect_true(validate_gpu_calibration_probe_metadata(probe).has_value(),
              "zero timestamp multiplicity is rejected");

  probe = probe_fixture();
  probe.first_stream_time_ns.reset();
  expect_true(validate_gpu_calibration_probe_metadata(probe).has_value(),
              "long stream without an indexed timeline origin is rejected");

  probe = probe_fixture();
  probe.total_frames_is_estimated = false;
  probe.selected_stream_caps_verified = true;
  probe.indexed_sampling_cadence_verified = true;
  expect_true(!validate_gpu_calibration_probe_metadata(probe).has_value(),
              "EOS-verified cadence remains accepted regardless of stream length");
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
  request.config.akaze.max_keypoints = static_cast<std::size_t>(kMaxGpuAkazeFeatures) + 1U;
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value() && error->find("8192") != std::string::npos,
              "GPU keypoint launch limit rejected before calibration execution");

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

  request = valid_request();
  request.calibration_worker_path = "relative/calibration-worker";
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value(), "relative calibration worker rejected");
}

void validation_rejects_calibration_output_input_aliases() {
  TemporaryDirectory root;
  const auto left = root.path() / "left.mp4";
  const auto right = root.path() / "right.mp4";
  write_file(left);
  write_file(right);

  auto request = valid_request();
  request.left.path = left.string();
  request.right.path = right.string();
  request.output = left.string();
  auto error = validate_gpu_calibration_request(request);
  expect_true(error.has_value() && error->find("left video input") != std::string::npos,
              "direct input/output alias is rejected");

  request.output = std::filesystem::relative(left, std::filesystem::current_path()).string();
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value() && error->find("left video input") != std::string::npos,
              "relative output alias of an absolute input is rejected");

  const auto nested = root.path() / "nested";
  std::filesystem::create_directory(nested);
  request.output = (nested / ".." / left.filename()).string();
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value() && error->find("left video input") != std::string::npos,
              "lexically normalized input/output alias is rejected");

  const auto symlink_output = root.path() / "symlink-match.json";
  std::error_code symlink_error;
  std::filesystem::create_symlink(right, symlink_output, symlink_error);
#if !defined(_WIN32)
  expect_true(!symlink_error, "symlink alias fixture is available");
#endif
  if (!symlink_error) {
    request.output = symlink_output.string();
    error = validate_gpu_calibration_request(request);
    expect_true(error.has_value() && error->find("right video input") != std::string::npos,
                "symlink input/output alias is rejected");
  }

  const auto hardlink_output = root.path() / "hardlink-match.json";
  std::error_code hardlink_error;
  std::filesystem::create_hard_link(left, hardlink_output, hardlink_error);
  expect_true(!hardlink_error, "hard-link alias fixture is available");
  if (!hardlink_error) {
    request.output = hardlink_output.string();
    error = validate_gpu_calibration_request(request);
    expect_true(error.has_value() && error->find("left video input") != std::string::npos,
                "hard-link input/output alias is rejected");
  }

  const auto loop_output = root.path() / "loop-match.json";
  std::error_code loop_error;
  std::filesystem::create_symlink(loop_output.filename(), loop_output, loop_error);
#if !defined(_WIN32)
  expect_true(!loop_error, "identity-error symlink fixture is available");
#endif
  if (!loop_error) {
    request.output = loop_output.string();
    error = validate_gpu_calibration_request(request);
    expect_true(error.has_value() && error->find("cannot inspect") != std::string::npos,
                "output identity resolution errors fail closed");
  }
}

void validation_rejects_calibration_output_profile_aliases() {
  TemporaryDirectory root;
  const auto left = root.path() / "left.mp4";
  const auto right = root.path() / "right.mp4";
  const auto left_profile = root.path() / "left-lens.json";
  const auto right_profile = root.path() / "right-lens.json";
  write_file(left);
  write_file(right);
  write_file(left_profile);
  write_file(right_profile);

  auto request = valid_request();
  request.left.path = left.string();
  request.right.path = right.string();
  request.left.lens_profile = left_profile.string();
  request.right.lens_profile = right_profile.string();
  request.output = left_profile.string();
  auto error = validate_gpu_calibration_request(request);
  expect_true(error.has_value() && error->find("left lens profile") != std::string::npos,
              "direct lens-profile/output alias is rejected");

  request.output =
      std::filesystem::relative(right_profile, std::filesystem::current_path()).string();
  error = validate_gpu_calibration_request(request);
  expect_true(error.has_value() && error->find("right lens profile") != std::string::npos,
              "relative output alias of an absolute lens profile is rejected");

  const auto hardlink_output = root.path() / "hardlink-match.json";
  std::error_code hardlink_error;
  std::filesystem::create_hard_link(left_profile, hardlink_output, hardlink_error);
  expect_true(!hardlink_error, "lens-profile hard-link fixture is available");
  if (!hardlink_error) {
    request.output = hardlink_output.string();
    error = validate_gpu_calibration_request(request);
    expect_true(error.has_value() && error->find("left lens profile") != std::string::npos,
                "hard-link lens-profile/output alias is rejected");
  }

  const auto symlink_output = root.path() / "symlink-match.json";
  std::error_code symlink_error;
  std::filesystem::create_symlink(right_profile, symlink_output, symlink_error);
#if !defined(_WIN32)
  expect_true(!symlink_error, "lens-profile symlink fixture is available");
#endif
  if (!symlink_error) {
    request.output = symlink_output.string();
    error = validate_gpu_calibration_request(request);
    expect_true(error.has_value() && error->find("right lens profile") != std::string::npos,
                "symlink lens-profile/output alias is rejected");
  }
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
  expect_true(plan.ready, "calibration does not require NPP outside optional interop paths");

  backends = ready_backends();
  backends.nvbufsurface = {.available = false, .detail = "no nvbufsurface"};
  plan = build_gpu_calibration_plan(valid_request(), backends);
  expect_true(!plan.ready, "missing NvBufSurface blocks plan");
  expect_true(plan.blocked_reason->find("NvBufSurface is required") != std::string::npos,
              "NvBufSurface block is reported");
}

void plan_reports_unported_optional_stages() {
  GpuCalibrationRequest default_request;
  expect_true(!default_request.no_auto_imu,
              "calibration requests require an explicit automatic IMU opt-out");

  auto request = valid_request();
  request.probe_worker.clear();
  auto plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(plan.ready, "outer calibration isolation makes a nested probe worker unnecessary");

  request = valid_request();
  request.calibration_worker_path.clear();
  plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "missing calibration worker blocks plan");
  expect_true(plan.blocked_reason->find("calibration worker") != std::string::npos,
              "missing calibration worker is reported");

  request = valid_request();
  request.left.lens_profile.reset();
  request.right.lens_profile.reset();
  plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "automatic lens detection blocks plan");
  expect_true(plan.blocked_reason->find("lens profile detection") != std::string::npos,
              "automatic lens detection gap is reported");

  request = valid_request();
  request.no_auto_imu = false;
  request.auto_sync = false;
  plan = build_gpu_calibration_plan(request, ready_backends());
  expect_true(!plan.ready, "manual sync requires an explicit automatic IMU opt-out");
  expect_true(plan.blocked_reason->find("--no-auto-imu") != std::string::npos,
              "missing automatic IMU opt-out is actionable");

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
  calibration_requires_exhaustive_indexed_cadence_proof();
  validation_rejects_calibration_output_input_aliases();
  validation_rejects_calibration_output_profile_aliases();
  plan_keeps_calibration_gpu_resident();
  calibration_plan_selects_hevc_decode_for_hevc_paths();
  plan_reports_missing_required_backend_first();
  plan_reports_unported_optional_stages();
  invalid_decode_paths_block_calibration_plan();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
