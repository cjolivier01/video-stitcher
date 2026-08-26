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

std::filesystem::path find_fake_runtime_runfile() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || runfiles[0] == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    const auto filename = entry.path().filename().string();
    if (filename.find("fake_gstreamer_runtime") != std::string::npos &&
        (ends_with(filename, ".so") || ends_with(filename, ".dylib") ||
         ends_with(filename, ".dll"))) {
      return entry.path();
    }
  }
  throw std::runtime_error("fake GStreamer runtime runfile not found");
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
  expect_eq(count_event(events, "state-null"), 1U, "source shutdown stops pipeline");
  expect_eq(count_event(events, "remove-display-probe"), 1U,
            "source shutdown removes the geometry callback");
  expect_eq(count_event(events, "destroy-display-probe-data"), 1U,
            "GStreamer owns and destroys the geometry callback state");
  expect_eq(count_event(events, "probe-leaked"), 0U,
            "geometry callback does not outlive its source");
  expect_eq(count_event(events, "unmap"), 0U,
            "source shutdown does not invalidate an outstanding frame");

  first.frame.reset();
  events = read_events(event_path);
  expect_eq(count_event(events, "unmap"), 1U, "last frame owner unmaps GstBuffer");
  expect_eq(count_event(events, "sample-unref"), 1U, "last frame owner releases GstSample");
  const auto unmap = std::find(events.begin(), events.end(), "unmap");
  const auto unref = std::find(events.begin(), events.end(), "sample-unref");
  expect_true(unmap < unref, "buffer is unmapped before sample release");

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
  expect_eq(count_event(events, "pull"), 1U, "latched bus error prevents later appsink pulls");
}

void runtime_failures_are_reported() {
  for (const auto& [scenario_name, fragment] :
       std::array<std::pair<std::string_view, std::string_view>, 9>{
           {{"old-version", "1.10 or newer"},
            {"init-error", "fake initialization failure"},
            {"parse-error", "fake parse failure"},
            {"missing-sink", "does not contain appsink"},
            {"missing-display-info", "does not contain pre-decoder identity"},
            {"missing-display-pad", "does not provide a source pad"},
            {"probe-install-error", "failed to install"},
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
  const auto runtime = find_fake_runtime_runfile();
  set_environment("RECO_GSTREAMER_DYLIB_PATH", runtime.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", runtime.string());
  set_environment("RECO_GLIB_DYLIB_PATH", runtime.string());
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto event_path = std::filesystem::temp_directory_path() /
                          ("reco_fake_gstreamer_events_" + std::to_string(unique) + ".txt");
  set_environment("RECO_FAKE_GST_EVENT_PATH", event_path.string());

  production_source_retains_mapped_sample();
  padded_sink_caps_preserve_predecoder_dimensions();
  runahead_caps_are_correlated_by_timestamp();
  duplicate_pts_at_geometry_transition_is_ambiguous();
  valid_transition_with_unchanged_allocation_is_correlated();
  dropped_first_frame_can_land_on_exact_transition();
  stale_parser_caps_reject_allocation_changes();
  dropped_samples_do_not_accumulate_geometry_records();
  first_frame_rejects_grossly_stale_geometry();
  unknown_timestamps_are_not_fabricated();
  concurrent_reads_serialize_appsink_access();
  fatal_pipeline_errors_are_latched();
  runtime_failures_are_reported();
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
