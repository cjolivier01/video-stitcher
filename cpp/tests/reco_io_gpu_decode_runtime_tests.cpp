#include "reco/io/gpu_decode.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace reco::io;

namespace {

int failures = 0;

void expect_true(bool value, std::string_view message) {
  if (!value) {
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
void expect_gpu_decode_error(Function&& function, std::string_view fragment,
                             std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const GpuDecodeError& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing error fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::filesystem::path find_fake_runtime_runfile(std::string_view runtime_name) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || runfiles[0] == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    const auto filename = entry.path().filename().string();
    if (filename.find(runtime_name) != std::string::npos &&
        (ends_with(filename, ".so") || ends_with(filename, ".dylib") ||
         ends_with(filename, ".dll"))) {
      return entry.path();
    }
  }
  throw std::runtime_error("fake runtime runfile not found");
}

void set_environment(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

std::vector<std::string> read_events(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::string> events;
  for (std::string line; std::getline(input, line);) {
    events.push_back(std::move(line));
  }
  return events;
}

std::size_t count_event(const std::vector<std::string>& events, std::string_view expected) {
  std::size_t count = 0;
  for (const auto& event : events) {
    if (event == expected) {
      ++count;
    }
  }
  return count;
}

GpuFileDecodeConfig valid_config() {
  return {.path = "/data/left.mp4",
          .codec = GpuDecodeCodec::H264,
          .elementary_stream = false,
          .container = GpuDecodeContainer::QuickTime,
          .max_buffers = 4,
          .drop = false};
}

void set_scenario(std::string_view value) {
  set_environment("RECO_FAKE_GST_SCENARIO", std::string(value));
}

void production_source_retains_mapped_sample() {
  set_scenario("frame-eos");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);

  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  expect_true(source->gpu_resident(), "production source reports GPU residency");
  expect_true(source->pipeline().find("memory:NVMM") != std::string_view::npos,
              "production source exposes NVMM pipeline");
  auto first = source->read();
  expect_true(first.status == GpuDecodeFrameStatus::Frame, "first pull returns frame");
  expect_true(first.frame.has_value(), "first pull contains frame payload");
  if (!first.frame.has_value()) {
    return;
  }
  expect_eq(first.frame->frame_index, 0U, "source assigns zero-based frame index");
  expect_true(first.frame->pts_ns == 1'000'000'000U, "PTS preserved from GstBuffer");
  expect_true(first.frame->duration_ns == 33'333'333U, "duration preserved from GstBuffer");
  expect_true(first.frame->nvmm.abi == NvbufSurfaceAbi::DeepStream9_1,
              "selected DeepStream ABI preserved");
  expect_true(first.frame->nvmm.memory_type == NvmmMemoryType::CudaDevice,
              "dGPU CUDA-device memory preserved");
  expect_eq(first.frame->visible_width, 1280U, "pre-decoder visible width preserved");
  expect_eq(first.frame->visible_height, 720U, "pre-decoder visible height preserved");
  expect_true(first.frame->nvmm.color_matrix == Nv12ColorMatrix::Bt709,
              "NV12 matrix metadata preserved");

  auto events = read_events(event_path);
  expect_eq(count_event(events, "map"), 1U, "frame maps once");
  expect_eq(count_event(events, "unmap"), 0U, "map remains live while frame is owned");
  expect_eq(count_event(events, "sample-unref"), 0U, "sample remains live while frame is owned");

  source.reset();
  events = read_events(event_path);
  expect_eq(count_event(events, "state-null"), 0U,
            "source shutdown defers pipeline stop while a frame is retained");
  expect_eq(count_event(events, "remove-display-probe"), 0U,
            "source shutdown retains the geometry callback with the pipeline");
  expect_eq(count_event(events, "remove-output-probe"), 0U,
            "source shutdown retains the output callback with the pipeline");
  expect_eq(count_event(events, "probe-leaked"), 0U,
            "geometry callback does not outlive its source");
  expect_eq(count_event(events, "unmap"), 0U,
            "source shutdown does not invalidate an outstanding frame");

  first.frame.reset();
  events = read_events(event_path);
  expect_eq(count_event(events, "unmap"), 1U, "last frame owner unmaps GstBuffer");
  expect_eq(count_event(events, "sample-unref"), 1U, "last frame owner releases GstSample");
  expect_eq(count_event(events, "state-null"), 1U,
            "last frame release stops the deferred pipeline");
  expect_eq(count_event(events, "remove-display-probe"), 1U,
            "deferred shutdown removes the geometry callback");
  expect_eq(count_event(events, "destroy-display-probe-data"), 1U,
            "GStreamer owns and destroys the geometry callback state");
  expect_eq(count_event(events, "remove-output-probe"), 1U,
            "deferred shutdown removes the output metadata callback");
  expect_eq(count_event(events, "destroy-output-probe-data"), 1U,
            "GStreamer owns and destroys the output callback state");
  const auto unmap = std::find(events.begin(), events.end(), "unmap");
  const auto unref = std::find(events.begin(), events.end(), "sample-unref");
  const auto state_null = std::find(events.begin(), events.end(), "state-null");
  expect_true(unmap < unref, "buffer is unmapped before sample release");
  expect_true(unref < state_null, "sample is released before deferred pipeline teardown");

  source = open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  auto frame = source->read();
  frame.frame.reset();
  const auto eos = source->read();
  expect_true(eos.status == GpuDecodeFrameStatus::EndOfStream, "appsink EOS is distinct");
  const auto eos_again = source->read();
  expect_true(eos_again.status == GpuDecodeFrameStatus::EndOfStream,
              "reads after EOS remain idempotent");
  events = read_events(event_path);
  expect_eq(count_event(events, "pull"), 3U,
            "idempotent EOS does not perform a third pull on either source");
}

void persistent_stereo_session_pairs_gstreamer_sources() {
  set_scenario("frame-eos");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);

  auto left_config = valid_config();
  auto right_config = valid_config();
  right_config.path = "/data/right.mp4";
  GpuStereoDecodeConfig session_config{.queue_capacity = 1};
  auto session = std::make_unique<GpuStereoDecodeSession>(
      open_gstreamer_gpu_file_decode_source(std::move(left_config), NvbufSurfaceAbi::DeepStream9_1),
      open_gstreamer_gpu_file_decode_source(std::move(right_config),
                                            NvbufSurfaceAbi::DeepStream9_1),
      session_config);

  auto paired = session->read();
  expect_true(paired.status == GpuStereoDecodeStatus::FramePair,
              "persistent GStreamer stereo decode returns a frame pair");
  expect_true(paired.frames.has_value(), "persistent GStreamer pair contains both frames");
  if (paired.frames.has_value()) {
    expect_eq(paired.frames->left.frame_index, 0U, "persistent left frame index");
    expect_eq(paired.frames->right.frame_index, 0U, "persistent right frame index");
    expect_true(paired.frames->left.owner != paired.frames->right.owner,
                "persistent pair retains each decoder owner independently");
  }
  auto events = read_events(event_path);
  expect_eq(count_event(events, "map"), 2U, "persistent stereo sources each map one NVMM frame");
  expect_eq(count_event(events, "unmap"), 0U,
            "persistent paired frames retain both mappings while owned");

  expect_true(session->read().status == GpuStereoDecodeStatus::EndOfStream,
              "persistent GStreamer stereo decode reports EOS after its pair");
  events = read_events(event_path);
  expect_eq(count_event(events, "unmap"), 0U,
            "persistent pair remains mapped while retained across EOS");
  session.reset();
  events = read_events(event_path);
  expect_eq(count_event(events, "state-null"), 0U,
            "session destruction does not block on or stop retained frame owners");
  paired.frames.reset();
  events = read_events(event_path);
  expect_eq(count_event(events, "unmap"), 2U,
            "persistent pair release unmaps both decoder buffers");
  expect_eq(count_event(events, "state-null"), 2U,
            "persistent frame release stops both deferred GStreamer pipelines");
}

void orientation_tags_are_preserved() {
  set_scenario("orientation-180");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  const auto frame = source->read();
  expect_true(frame.frame.has_value(), "orientation fixture returns frame");
  if (frame.frame.has_value()) {
    expect_eq(frame.frame->rotation_degrees, 180U, "180-degree stream tag is preserved");
  }

  set_scenario("orientation-90");
  source = open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  const auto rotated = source->read();
  expect_true(rotated.frame.has_value(), "90-degree orientation fixture returns frame");
  if (rotated.frame.has_value()) {
    expect_eq(rotated.frame->rotation_degrees, 90U, "90-degree stream tag is preserved");
  }

  set_scenario("orientation-flip");
  source = open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  expect_gpu_decode_error([&] { (void)source->read(); }, "unsupported GStreamer image orientation",
                          "mirrored stream orientation fails closed");
}

void indexed_cadence_drives_frame_indices() {
  auto config = valid_config();
  config.indexed_fps_numerator = 30U;
  config.indexed_fps_denominator = 1U;

  set_scenario("indexed-cfr");
  auto source = open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
  const auto first = source->read();
  const auto second = source->read();
  expect_true(first.frame.has_value() && second.frame.has_value(),
              "indexed CFR fixture returns frames");
  if (first.frame.has_value() && second.frame.has_value()) {
    expect_eq(first.frame->frame_index, 0U, "indexed cadence origin is frame zero");
    expect_eq(second.frame->frame_index, 1U, "indexed cadence advances from PTS");
  }

  set_scenario("indexed-gap");
  source = open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
  (void)source->read();
  const auto after_gap = source->read();
  expect_true(after_gap.frame.has_value(), "indexed gap fixture returns second frame");
  if (after_gap.frame.has_value()) {
    expect_eq(after_gap.frame->frame_index, 2U, "PTS gap is not hidden by a pull counter");
  }

  set_scenario("indexed-off-cadence");
  source = open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
  (void)source->read();
  expect_gpu_decode_error([&] { (void)source->read(); }, "violates the probed cadence",
                          "off-cadence decoded PTS fails closed");
}

void indexed_decode_seeks_to_absolute_start_frame() {
  set_scenario("indexed-seek");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);
  auto config = valid_config();
  config.indexed_fps_numerator = 30U;
  config.indexed_fps_denominator = 1U;
  config.indexed_stream_time_origin_ns = 0U;
  config.start_frame_index = 300U;

  auto source = open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
  const auto first = source->read();
  const auto second = source->read();
  expect_true(first.frame.has_value() && second.frame.has_value(),
              "indexed seek fixture returns frames");
  if (first.frame.has_value() && second.frame.has_value()) {
    expect_eq(first.frame->frame_index, 300U, "seeked frame keeps absolute stream index");
    expect_eq(second.frame->frame_index, 301U, "seeked cadence advances absolute index");
    expect_true(first.frame->pts_ns == 10'000'000'000ULL,
                "initial seek uses exact rational frame timestamp");
  }
  const auto events = read_events(event_path);
  const auto paused = std::find(events.begin(), events.end(), "state-paused");
  const auto seek = std::find(events.begin(), events.end(), "seek-compressed");
  const auto playing = std::find(events.begin(), events.end(), "state-playing");
  expect_true(paused < seek && seek < playing,
              "decode pauses and seeks before entering PLAYING state");

  set_scenario("probe-seek-unsupported");
  expect_gpu_decode_error(
      [&] { (void)open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1); },
      "rejected the requested initial frame seek", "unsupported initial decode seek");

  set_scenario("indexed-seek-nonzero-origin");
  config.indexed_fps_numerator = 30'000U;
  config.indexed_fps_denominator = 1'001U;
  config.indexed_stream_time_origin_ns = 766'666'666ULL;
  source = open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
  const auto nonzero_first = source->read();
  const auto nonzero_second = source->read();
  expect_true(nonzero_first.frame.has_value() && nonzero_second.frame.has_value(),
              "nonzero-origin indexed seek returns frames");
  if (nonzero_first.frame.has_value() && nonzero_second.frame.has_value()) {
    expect_eq(nonzero_first.frame->frame_index, 300U,
              "nonzero-origin rational seek retains the requested absolute index");
    expect_eq(nonzero_second.frame->frame_index, 301U,
              "nonzero-origin rational cadence advances the absolute index");
    expect_true(nonzero_first.frame->pts_ns == 15'776'666'666ULL,
                "raw PTS remains distinct from converted presentation stream time");
  }

  set_scenario("indexed-seek-wrong-first");
  expect_gpu_decode_error(
      [&] {
        auto mismatched =
            open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
        (void)mismatched->read();
      },
      "does not match the requested absolute frame index",
      "post-seek sample from the wrong absolute frame is rejected");

  set_scenario("indexed-seek-invalid-segment");
  expect_gpu_decode_error(
      [&] {
        auto invalid_segment =
            open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
        (void)invalid_segment->read();
      },
      "cannot be converted to presentation stream time",
      "post-seek sample without a presentation stream-time mapping is rejected");
}

void stalled_appsink_reads_are_bounded() {
  set_scenario("read-timeout");
  auto config = valid_config();
  config.read_timeout_ns = 100'000'000ULL;
  auto source = open_gstreamer_gpu_file_decode_source(config, NvbufSurfaceAbi::DeepStream9_1);
  const auto start = std::chrono::steady_clock::now();
  expect_gpu_decode_error([&] { (void)source->read(); }, "read timed out",
                          "stalled appsink read reaches its deadline");
  const auto elapsed = std::chrono::steady_clock::now() - start;
  expect_true(elapsed < std::chrono::seconds(2), "appsink read deadline is bounded in wall time");
  expect_gpu_decode_error([&] { (void)source->read(); }, "read timed out",
                          "read timeout is latched");
}

void padded_sink_caps_preserve_predecoder_dimensions() {
  set_scenario("visible-crop");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  const auto result = source->read();
  expect_true(result.frame.has_value(), "visible-crop scenario returns a frame");
  if (!result.frame.has_value()) {
    return;
  }
  expect_eq(result.frame->nvmm.width, 864U, "aligned NVMM allocation width preserved");
  expect_eq(result.frame->nvmm.y_pitch, 1024U, "aligned NVMM allocation pitch preserved");
  expect_eq(result.frame->visible_width, 854U, "pre-decoder width preserved separately");
  expect_eq(result.frame->visible_height, 720U, "pre-decoder height preserved separately");
  const auto events = read_events(event_path);
  expect_eq(count_event(events, "pad-current-caps"), 1U,
            "display dimensions come from pre-decoder caps");
  expect_eq(count_event(events, "sample-caps"), 0U,
            "padded appsink caps are not treated as display dimensions");
}

void runahead_caps_are_correlated_by_timestamp() {
  set_scenario("caps-runahead");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  const auto first = source->read();
  expect_true(first.frame.has_value(), "caps-runahead first frame returned");
  if (!first.frame.has_value()) {
    return;
  }
  expect_eq(first.frame->nvmm.width, 864U, "first runahead allocation remains padded");
  expect_eq(first.frame->visible_width, 854U,
            "first runahead frame uses its timestamped pre-decoder geometry");
  auto events = read_events(event_path);
  expect_eq(count_event(events, "pad-current-caps"), 2U,
            "upstream advances to the second geometry before the first pull returns");

  const auto second = source->read();
  expect_true(second.frame.has_value(), "caps-runahead second frame returned");
  if (second.frame.has_value()) {
    expect_eq(second.frame->nvmm.width, 1280U, "second runahead allocation width");
    expect_eq(second.frame->visible_width, 1280U,
              "second runahead frame uses its own timestamped geometry");
  }
}

void duplicate_pts_at_geometry_transition_is_ambiguous() {
  set_scenario("duplicate-transition-pts");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  const auto first = source->read();
  expect_true(first.frame.has_value(), "duplicate-transition first old-geometry frame returned");
  expect_gpu_decode_error([&] { (void)source->read(); },
                          "timestamp is ambiguous at a geometry transition",
                          "duplicate PTS cannot relabel an old allocation with new geometry");
}

void valid_transition_with_unchanged_allocation_is_correlated() {
  set_scenario("same-allocation-runahead");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  const auto first = source->read();
  expect_true(first.frame.has_value(), "same-allocation first frame returned");
  if (first.frame.has_value()) {
    expect_eq(first.frame->nvmm.width, 1280U, "same-allocation first padded width");
    expect_eq(first.frame->visible_width, 1100U, "same-allocation first visible width");
  }

  const auto second = source->read();
  expect_true(second.frame.has_value(), "same-allocation transition frame returned");
  if (second.frame.has_value()) {
    expect_eq(second.frame->nvmm.width, 1280U, "same-allocation second padded width");
    expect_eq(second.frame->visible_width, 1200U, "same-allocation second visible width");
  }
}

void dropped_first_frame_can_land_on_exact_transition() {
  set_scenario("drop-transition-first");
  auto config = valid_config();
  config.drop = true;
  auto source =
      open_gstreamer_gpu_file_decode_source(std::move(config), NvbufSurfaceAbi::DeepStream9_1);

  const auto result = source->read();
  expect_true(result.frame.has_value(), "dropped exact-transition frame returned");
  if (result.frame.has_value()) {
    expect_eq(result.frame->pts_ns.value_or(0), 2'000'000'000ULL,
              "dropped exact-transition PTS preserved");
    expect_eq(result.frame->visible_width, 1200U,
              "dropped exact-transition frame uses current geometry");
  }
}

void dropped_first_duplicate_pts_at_transition_is_ambiguous() {
  set_scenario("drop-duplicate-transition-first");
  auto config = valid_config();
  config.drop = true;
  auto source =
      open_gstreamer_gpu_file_decode_source(std::move(config), NvbufSurfaceAbi::DeepStream9_1);

  expect_gpu_decode_error([&] { (void)source->read(); },
                          "timestamp is ambiguous at a geometry transition",
                          "first retained duplicate-PTS boundary frame remains ambiguous");
}

void nonmonotonic_parser_pts_preserve_duplicate_boundary_evidence() {
  set_scenario("nonmonotonic-duplicate-transition");
  auto config = valid_config();
  config.drop = true;
  auto source =
      open_gstreamer_gpu_file_decode_source(std::move(config), NvbufSurfaceAbi::DeepStream9_1);

  expect_gpu_decode_error([&] { (void)source->read(); },
                          "timestamp is ambiguous at a geometry transition",
                          "reordered parser PTS cannot hide duplicate-boundary ambiguity");
}

void nonmonotonic_parser_pts_correlate_unique_geometry() {
  set_scenario("nonmonotonic-unique-transition");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  const auto result = source->read();
  expect_true(result.frame.has_value(), "nonmonotonic unique transition frame returned");
  if (result.frame.has_value()) {
    expect_eq(result.frame->pts_ns.value_or(0), 2'000'000'000ULL,
              "nonmonotonic unique transition PTS preserved");
    expect_eq(result.frame->visible_width, 1200U,
              "nonmonotonic PTS uses its exact pre-decoder geometry");
  }
}

void retired_pts_do_not_pollute_later_geometry_epochs() {
  set_scenario("retired-pts-reuse");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  constexpr std::array<std::uint32_t, 3> expected_widths{1100, 1200, 1160};
  for (const auto expected_width : expected_widths) {
    const auto result = source->read();
    expect_true(result.frame.has_value(), "retired-PTS geometry frame returned");
    if (result.frame.has_value()) {
      expect_eq(result.frame->visible_width, expected_width,
                "retired PTS does not create a stale geometry candidate");
    }
  }
}

void pulled_sample_metadata_survives_post_pull_runahead() {
  set_scenario("post-pull-runahead");
  auto config = valid_config();
  config.drop = true;
  auto source =
      open_gstreamer_gpu_file_decode_source(std::move(config), NvbufSurfaceAbi::DeepStream9_1);

  const auto result = source->read();
  expect_true(result.frame.has_value(), "post-pull runahead frame returned");
  if (result.frame.has_value()) {
    expect_eq(result.frame->visible_width, 1280U,
              "attached geometry survives 5000 post-pull decoded frames");
  }
}

void stale_parser_caps_reject_allocation_changes() {
  set_scenario("caps-runahead-stale-caps");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  const auto first = source->read();
  expect_true(first.frame.has_value(), "stale-caps first frame returned");
  if (!first.frame.has_value()) {
    return;
  }
  expect_eq(first.frame->nvmm.width, 864U, "stale-caps initial padded allocation width");
  expect_eq(first.frame->nvmm.height, 480U, "stale-caps initial allocation height");
  expect_eq(first.frame->visible_width, 854U, "stale-caps initial visible width");
  expect_eq(first.frame->visible_height, 480U, "stale-caps initial visible height");

  expect_gpu_decode_error([&] { (void)source->read(); }, "visible geometry",
                          "stale parser caps cannot silently crop a resized NVMM frame");
}

void dropped_samples_do_not_accumulate_geometry_records() {
  set_scenario("drop-runahead");
  auto config = valid_config();
  config.drop = true;
  auto source =
      open_gstreamer_gpu_file_decode_source(std::move(config), NvbufSurfaceAbi::DeepStream9_1);

  const auto result = source->read();
  expect_true(result.frame.has_value(), "drop-runahead frame returned after 5000 discarded inputs");
  if (result.frame.has_value()) {
    expect_eq(result.frame->visible_width, 1280U,
              "drop-runahead static geometry remains correlated");
  }
}

void first_frame_rejects_grossly_stale_geometry() {
  set_scenario("drop-stale-caps-first");
  auto config = valid_config();
  config.drop = true;
  auto source =
      open_gstreamer_gpu_file_decode_source(std::move(config), NvbufSurfaceAbi::DeepStream9_1);
  expect_gpu_decode_error([&] { (void)source->read(); },
                          "allocation dimensions are inconsistent with visible geometry",
                          "first retained frame cannot use grossly stale parser caps");
}

void unknown_timestamps_are_not_fabricated() {
  set_scenario("unknown-time");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  const auto result = source->read();
  expect_true(result.frame.has_value(), "unknown-time scenario returns a frame");
  if (!result.frame.has_value()) {
    return;
  }
  expect_true(!result.frame->pts_ns.has_value(), "unknown PTS remains absent");
  expect_true(!result.frame->duration_ns.has_value(), "unknown duration remains absent");
}

void unknown_output_retires_only_unknown_observations() {
  set_scenario("mixed-unknown-reorder");
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  const auto unknown = source->read();
  expect_true(unknown.frame.has_value(), "mixed-PTS unknown frame returned");
  if (unknown.frame.has_value()) {
    expect_true(!unknown.frame->pts_ns.has_value(), "mixed-PTS unknown frame stays unknown");
  }
  const auto known = source->read();
  expect_true(known.frame.has_value(), "mixed-PTS reordered known frame returned");
  if (known.frame.has_value()) {
    expect_eq(known.frame->pts_ns.value_or(0), 1'000'000'000ULL,
              "mixed-PTS known observation remains correlated");
    expect_eq(known.frame->visible_width, 1280U, "mixed-PTS known geometry remains correlated");
  }
}

void dropped_ambiguous_unknown_output_keeps_batch_ambiguous() {
  set_scenario("dropped-unknown-transition");
  auto config = valid_config();
  config.drop = true;
  auto source =
      open_gstreamer_gpu_file_decode_source(std::move(config), NvbufSurfaceAbi::DeepStream9_1);

  expect_gpu_decode_error([&] { (void)source->read(); },
                          "timestamp cannot be correlated across a geometry change",
                          "dropped ambiguous unknown frame keeps remaining batch fail-closed");
}

void concurrent_reads_serialize_appsink_access() {
  set_scenario("frame-eos");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
  std::atomic<std::uint32_t> frame_count{0};
  std::atomic<std::uint32_t> eos_count{0};
  std::atomic<std::uint32_t> error_count{0};
  std::vector<std::thread> readers;
  readers.reserve(16);
  for (std::uint32_t index = 0; index < 16; ++index) {
    readers.emplace_back([&] {
      try {
        const auto result = source->read();
        if (result.status == GpuDecodeFrameStatus::Frame) {
          ++frame_count;
        } else {
          ++eos_count;
        }
      } catch (...) {
        ++error_count;
      }
    });
  }
  for (auto& reader : readers) {
    reader.join();
  }

  expect_eq(frame_count.load(), 1U, "concurrent reads return the one queued frame once");
  expect_eq(eos_count.load(), 15U, "concurrent reads return stable EOS after the frame");
  expect_eq(error_count.load(), 0U, "concurrent reads do not race");
  const auto events = read_events(event_path);
  expect_eq(count_event(events, "pull"), 2U, "concurrent reads serialize appsink pulls");
}

void fatal_pipeline_errors_are_latched() {
  set_scenario("stream-error");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  std::atomic<std::uint32_t> expected_errors{0};
  std::atomic<std::uint32_t> unexpected_results{0};
  std::vector<std::thread> readers;
  readers.reserve(16);
  for (std::uint32_t index = 0; index < 16; ++index) {
    readers.emplace_back([&] {
      try {
        (void)source->read();
        ++unexpected_results;
      } catch (const GpuDecodeError& error) {
        if (std::string_view(error.what()).find("fake decoder failure") != std::string_view::npos) {
          ++expected_errors;
        } else {
          ++unexpected_results;
        }
      } catch (...) {
        ++unexpected_results;
      }
    });
  }
  for (auto& reader : readers) {
    reader.join();
  }

  expect_eq(expected_errors.load(), 16U, "all concurrent readers receive the fatal bus error");
  expect_eq(unexpected_results.load(), 0U, "fatal bus errors return deterministic results");
  const auto events = read_events(event_path);
  expect_eq(count_event(events, "pull"), 0U,
            "queued fatal bus error prevents the first appsink pull");
}

void geometry_probe_errors_are_latched_before_further_pulls() {
  set_scenario("missing-caps");
  const auto event_path = std::filesystem::path(std::getenv("RECO_FAKE_GST_EVENT_PATH"));
  std::filesystem::remove(event_path);
  auto source =
      open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);

  expect_gpu_decode_error([&] { (void)source->read(); }, "does not contain negotiated caps",
                          "first geometry probe failure");
  expect_gpu_decode_error([&] { (void)source->read(); }, "does not contain negotiated caps",
                          "latched geometry probe failure");
  const auto events = read_events(event_path);
  expect_eq(count_event(events, "pull"), 1U,
            "latched geometry probe failure prevents further sample pulls");
}

void runtime_failures_are_reported() {
  for (const auto& [scenario_name, fragment] :
       std::array<std::pair<std::string_view, std::string_view>, 12>{
           {{"old-version", "1.10 or newer"},
            {"init-error", "fake initialization failure"},
            {"parse-error", "fake parse failure"},
            {"missing-sink", "does not contain appsink"},
            {"missing-display-info", "does not contain pre-decoder identity"},
            {"missing-display-pad", "does not provide a source pad"},
            {"probe-install-error", "failed to install"},
            {"missing-output-info", "does not contain output identity"},
            {"missing-output-pad", "output identity does not provide a source pad"},
            {"output-probe-install-error", "failed to install GStreamer output"},
            {"missing-bus", "does not provide a message bus"},
            {"state-error", "rejected the PLAYING state"}}}) {
    set_scenario(scenario_name);
    expect_gpu_decode_error(
        [&] {
          (void)open_gstreamer_gpu_file_decode_source(valid_config(),
                                                      NvbufSurfaceAbi::DeepStream9_1);
        },
        fragment, scenario_name);
  }

  for (const auto& [scenario_name, fragment] :
       std::array<std::pair<std::string_view, std::string_view>, 10>{
           {{"stream-error", "fake decoder failure"},
            {"delayed-stream-error", "fake decoder failure"},
            {"missing-buffer", "does not contain a buffer"},
            {"map-error", "failed to map"},
            {"invalid-surface", "filled-buffer count"},
            {"missing-caps", "does not contain negotiated caps"},
            {"missing-caps-structure", "do not contain a structure"},
            {"invalid-caps", "valid visible dimensions"},
            {"oversized-caps", "exceed the NVMM allocation"},
            {"caps-runahead-unknown-time", "cannot be correlated"}}}) {
    set_scenario(scenario_name);
    auto source =
        open_gstreamer_gpu_file_decode_source(valid_config(), NvbufSurfaceAbi::DeepStream9_1);
    expect_gpu_decode_error([&] { (void)source->read(); }, fragment, scenario_name);
  }
}

} // namespace

int main() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  const auto runtime = find_fake_runtime_runfile("fake_gstreamer_runtime");
  set_environment("RECO_GSTREAMER_DYLIB_PATH", runtime.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", runtime.string());
  set_environment("RECO_GLIB_DYLIB_PATH", runtime.string());
#if defined(__linux__)
  const auto nvbufsurface = find_fake_runtime_runfile("fake_nvbufsurface.so");
  set_environment("RECO_NVBUFSURFACE_DYLIB_PATH", nvbufsurface.string());
  set_environment("RECO_NVDS_UTILS_DYLIB_PATH", nvbufsurface.string());
#endif
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto event_path = std::filesystem::temp_directory_path() /
                          ("reco_fake_gstreamer_events_" + std::to_string(unique) + ".txt");
  set_environment("RECO_FAKE_GST_EVENT_PATH", event_path.string());

  production_source_retains_mapped_sample();
  persistent_stereo_session_pairs_gstreamer_sources();
  orientation_tags_are_preserved();
  indexed_cadence_drives_frame_indices();
  indexed_decode_seeks_to_absolute_start_frame();
  stalled_appsink_reads_are_bounded();
  padded_sink_caps_preserve_predecoder_dimensions();
  runahead_caps_are_correlated_by_timestamp();
  duplicate_pts_at_geometry_transition_is_ambiguous();
  valid_transition_with_unchanged_allocation_is_correlated();
  dropped_first_frame_can_land_on_exact_transition();
  dropped_first_duplicate_pts_at_transition_is_ambiguous();
  nonmonotonic_parser_pts_preserve_duplicate_boundary_evidence();
  nonmonotonic_parser_pts_correlate_unique_geometry();
  retired_pts_do_not_pollute_later_geometry_epochs();
  pulled_sample_metadata_survives_post_pull_runahead();
  stale_parser_caps_reject_allocation_changes();
  dropped_samples_do_not_accumulate_geometry_records();
  first_frame_rejects_grossly_stale_geometry();
  unknown_timestamps_are_not_fabricated();
  unknown_output_retires_only_unknown_observations();
  dropped_ambiguous_unknown_output_keeps_batch_ambiguous();
  concurrent_reads_serialize_appsink_access();
  fatal_pipeline_errors_are_latched();
  geometry_probe_errors_are_latched_before_further_pulls();
  runtime_failures_are_reported();
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
