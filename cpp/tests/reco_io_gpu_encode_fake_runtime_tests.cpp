#include "reco/io/gpu_encode.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace reco::io;

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
void expect_encode_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const GpuEncodeError& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing error fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_audio_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const AudioPassthroughError& error) {
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
  throw std::runtime_error("fake runtime runfile not found: " + std::string(runtime_name));
}

void set_environment(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void set_scenario(std::string_view scenario) {
  set_environment("RECO_FAKE_GST_SCENARIO", std::string(scenario));
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

bool wait_for_event(const std::filesystem::path& path, std::string_view expected) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  do {
    if (count_event(read_events(path), expected) != 0U) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

class Trace final : public GpuEncodeTraceSink {
public:
  void surface_allocated() noexcept override { ++allocated; }
  void surface_acquired() noexcept override { ++acquired; }
  void surface_submitted() noexcept override { ++submitted; }
  void surface_released() noexcept override { ++released; }

  std::atomic<std::uint32_t> allocated{0};
  std::atomic<std::uint32_t> acquired{0};
  std::atomic<std::uint32_t> submitted{0};
  std::atomic<std::uint32_t> released{0};
};

GpuEncodeConfig config() {
  return {
      .output_path = "fake-output.mp4",
      .width = 1280,
      .height = 720,
      .fps_numerator = 30,
      .fps_denominator = 1,
      .pool_capacity = 8,
      .acquire_timeout = std::chrono::milliseconds(5),
      .startup_timeout = std::chrono::milliseconds(5),
      .finalize_timeout = std::chrono::milliseconds(5),
  };
}

GpuEncodeConfig audio_config() {
  auto result = config();
  result.audio_caps = "audio/mpeg, mpegversion=(int)4, rate=(int)48000, channels=(int)2";
  return result;
}

CompressedAudioPacket audio_packet(std::size_t size, std::uint32_t flags = 0) {
  return {
      .bytes = std::vector<std::byte>(size, std::byte{0x5a}),
      .pts_ns = 0,
      .dts_ns = 0,
      .duration_ns = 21'333'333,
      .flags = flags,
  };
}

GpuVideoEncodeSession open_session(const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
                                   const std::shared_ptr<Trace>& trace = {}) {
  return GpuVideoEncodeSession::open(config(), runtime, trace);
}

void startup_failures_release_partial_resources(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const std::filesystem::path& event_path) {
  struct Failure {
    std::string_view scenario;
    std::string_view error;
  };
  constexpr Failure failures_to_test[] = {
      {"init-error", "fake initialization failure"},
      {"encode-parse-error", "fake parse failure"},
      {"encode-parse-partial-error", "fake partial parse failure"},
      {"encode-missing-source", "missing appsrc or bus"},
      {"encode-missing-bus", "missing appsrc or bus"},
      {"encode-state-error", "failed to enter PLAYING"},
      {"encode-startup-timeout", "timed out"},
      {"encode-startup-wrong-state", "timed out"},
  };
  for (const auto& failure : failures_to_test) {
    std::filesystem::remove(event_path);
    set_scenario(failure.scenario);
    expect_encode_error([&] { (void)open_session(runtime); }, failure.error, failure.scenario);
    if (failure.scenario == "encode-parse-partial-error") {
      expect_eq(count_event(read_events(event_path), "unref-pipeline"), 1U,
                "partial parse pipeline is released");
    }
  }
}

void wrapped_callbacks_and_move_assignment_release_exactly_once(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  set_scenario("encode-success");
  auto trace = std::make_shared<Trace>();
  auto session = open_session(runtime, trace);
  auto first = session.acquire_frame();
  auto second = session.acquire_frame();
  expect_true(first.view().y_plane().driver_validation() != nullptr &&
                  first.view().uv_plane().driver_validation() != nullptr,
              "encoder surface view retains CUDA driver validation");
  expect_true(
      first.view().y_plane().driver_validation()->permits(reco::core::CudaSpanAccess::ReadWrite) &&
          first.view().uv_plane().driver_validation()->permits(
              reco::core::CudaSpanAccess::ReadWrite),
      "encoder surface validation permits CUDA writes");
  first = std::move(second);
  expect_eq(trace->released.load(), 1U, "move assignment releases the displaced lease");
  session.submit_frame(std::move(first), 0, 33'333'333);
  expect_eq(trace->submitted.load(), 1U, "wrapped frame is submitted once");
  expect_eq(trace->released.load(), 2U, "wrapped-buffer callback releases the submitted frame");
  auto reusable = session.acquire_frame();
  expect_true(static_cast<bool>(reusable), "callback-released pool slot can be acquired again");
  reusable = session.acquire_frame();
  expect_eq(trace->released.load(), 3U, "lease reassignment returns the previous pool slot");
}

void wrapping_and_push_failures_preserve_pool_ownership(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  {
    set_scenario("encode-wrap-error");
    auto trace = std::make_shared<Trace>();
    auto session = open_session(runtime, trace);
    auto frame = session.acquire_frame();
    expect_encode_error([&] { session.submit_frame(std::move(frame), 0, 1); }, "failed to wrap",
                        "wrapped-buffer allocation failure");
    expect_true(!frame, "failed submit consumes its frame lease");
    expect_eq(trace->released.load(), 1U, "wrap failure returns the pool slot once");
  }
  {
    set_scenario("encode-push-error");
    auto trace = std::make_shared<Trace>();
    auto session = open_session(runtime, trace);
    auto frame = session.acquire_frame();
    expect_encode_error([&] { session.submit_frame(std::move(frame), 0, 1); }, "flow status",
                        "appsrc push failure");
    expect_eq(trace->released.load(), 1U, "appsrc rejection callback returns the pool slot");
    expect_encode_error([&] { (void)session.acquire_frame(); }, "flow status",
                        "appsrc rejection remains sticky");
  }
}

void bus_errors_and_early_eos_are_sticky(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  {
    set_scenario("encode-bus-error");
    auto session = open_session(runtime);
    auto frame = session.acquire_frame();
    session.submit_frame(std::move(frame), 0, 1);
    expect_encode_error([&] { (void)session.acquire_frame(); }, "fake encoder failure",
                        "encoder bus error is reported");
    expect_encode_error([&] { (void)session.acquire_frame(); }, "fake encoder failure",
                        "encoder bus error remains sticky");
  }
  {
    set_scenario("encode-early-eos");
    auto session = open_session(runtime);
    expect_encode_error([&] { (void)session.acquire_frame(); }, "before finalization",
                        "unexpected encoder EOS is rejected");
    expect_encode_error([&] { (void)session.acquire_frame(); }, "before finalization",
                        "unexpected encoder EOS remains sticky");
  }
}

void bounded_pool_times_out_and_releases_on_abort(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  set_scenario("encode-pool-backpressure");
  auto trace = std::make_shared<Trace>();
  {
    auto session = open_session(runtime, trace);
    for (std::uint32_t index = 0; index < 8; ++index) {
      auto frame = session.acquire_frame();
      session.submit_frame(std::move(frame), index, 1);
    }
    expect_encode_error([&] { (void)session.acquire_frame(); }, "timed out waiting",
                        "bounded output pool applies backpressure");
    expect_eq(trace->released.load(), 0U, "retained downstream buffers keep pool slots busy");
  }
  expect_eq(trace->released.load(), 8U, "abort releases every retained downstream buffer");
}

void finalization_failures_abort_and_release(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  for (const auto& [scenario, fragment] :
       std::vector<std::pair<std::string_view, std::string_view>>{
           {"encode-eos-rejected", "rejected end-of-stream"},
           {"encode-finalize-timeout", "timed out while finalizing"},
       }) {
    set_scenario(scenario);
    auto trace = std::make_shared<Trace>();
    auto session = open_session(runtime, trace);
    auto frame = session.acquire_frame();
    session.submit_frame(std::move(frame), 0, 1);
    expect_encode_error([&] { session.finish(); }, fragment, scenario);
    expect_eq(trace->released.load(), 1U, "failed finalization releases retained surfaces");
    expect_encode_error([&] { (void)session.acquire_frame(); }, "no longer accepting",
                        "failed finalization closes frame admission");
  }
}

void successful_finish_waits_for_downstream_release(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  set_scenario("encode-retain");
  auto trace = std::make_shared<Trace>();
  auto session = open_session(runtime, trace);
  auto frame = session.acquire_frame();
  session.submit_frame(std::move(frame), 0, 1);
  expect_eq(trace->released.load(), 0U, "encoder retains submitted surface before EOS");
  session.finish();
  expect_eq(trace->released.load(), 1U, "EOS completion releases retained surface");
  session.finish();
}

void outstanding_leases_survive_session_destruction(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  set_scenario("encode-success");
  auto trace = std::make_shared<Trace>();
  std::optional<GpuEncodeFrameLease> outstanding;
  {
    auto session = open_session(runtime, trace);
    outstanding.emplace(session.acquire_frame());
    expect_encode_error([&] { session.finish(); }, "outstanding",
                        "finish rejects an acquired frame lease");
  }
  expect_eq(trace->released.load(), 0U,
            "acquired lease retains its pool after session destruction");
  expect_true(outstanding->view().width() == 1280U,
              "outstanding lease view remains valid after session destruction");
  outstanding.reset();
  expect_eq(trace->released.load(), 1U, "last outstanding lease releases its retained pool");

  set_scenario("encode-retain");
  trace = std::make_shared<Trace>();
  {
    auto session = open_session(runtime, trace);
    auto submitted = session.acquire_frame();
    session.submit_frame(std::move(submitted), 0, 1);
  }
  expect_eq(trace->released.load(), 1U,
            "session destruction releases a downstream-retained submitted lease");
}

void compressed_audio_packets_use_the_bounded_audio_appsrc(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("encode-success");
  auto session = GpuVideoEncodeSession::open(audio_config(), runtime);
  session.submit_audio_packet(audio_packet(128));
  session.finish();
  const auto events = read_events(event_path);
  expect_eq(count_event(events, "get-audio-source"), 1U,
            "audio-enabled encoder retains its named appsrc");
  expect_eq(count_event(events, "push-audio-buffer"), 1U,
            "compressed audio packet is submitted exactly once");
  expect_eq(count_event(events, "wrapped-release"), 1U,
            "compressed audio storage is released exactly once");
  expect_eq(count_event(events, "audio-appsrc-eos"), 1U,
            "audio appsrc receives terminal EOS before mux finalization");
}

void compressed_audio_backpressure_bounds_packets_and_bytes(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("encode-audio-backpressure");
  {
    auto session = GpuVideoEncodeSession::open(audio_config(), runtime);
    for (std::size_t index = 0; index < 32; ++index) {
      session.submit_audio_packet(audio_packet(1));
    }
    expect_encode_error([&] { session.submit_audio_packet(audio_packet(1)); },
                        "timed out waiting for compressed audio mux capacity",
                        "compressed audio packet count applies backpressure");
    session.abort();
  }
  expect_eq(count_event(read_events(event_path), "wrapped-release"), 32U,
            "packet-count abort releases every retained compressed packet");

  std::filesystem::remove(event_path);
  {
    auto session = GpuVideoEncodeSession::open(audio_config(), runtime);
    session.submit_audio_packet(audio_packet(16U * 1024U * 1024U));
    session.submit_audio_packet(audio_packet(16U * 1024U * 1024U));
    expect_encode_error([&] { session.submit_audio_packet(audio_packet(1)); },
                        "timed out waiting for compressed audio mux capacity",
                        "compressed audio aggregate bytes apply backpressure");
    session.abort();
  }
  expect_eq(count_event(read_events(event_path), "wrapped-release"), 2U,
            "byte-budget abort releases every retained compressed packet");
}

void compressed_audio_preserves_only_buffer_semantic_flags(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("encode-success");
  auto session = GpuVideoEncodeSession::open(audio_config(), runtime);
  constexpr std::uint32_t mini_object_flags = 0x0f;
  constexpr std::uint32_t discont_and_delta_unit_flags = 0x40 | 0x2000;
  constexpr std::uint32_t tag_memory_flag = 0x4000;
  session.submit_audio_packet(
      audio_packet(1, mini_object_flags | discont_and_delta_unit_flags | tag_memory_flag));
  session.finish();
  expect_eq(count_event(read_events(event_path), "audio-buffer-flags-8256"), 1U,
            "audio mux retains semantic flags and filters mini-object and memory-tag flags");
}

void submission_is_serialized_with_finish_and_abort(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const std::filesystem::path& event_path) {
  using namespace std::chrono_literals;
  set_scenario("encode-blocking-audio-push");

  std::filesystem::remove(event_path);
  {
    auto session = GpuVideoEncodeSession::open(audio_config(), runtime);
    auto submit =
        std::async(std::launch::async, [&] { session.submit_audio_packet(audio_packet(1)); });
    expect_true(wait_for_event(event_path, "audio-push-enter"),
                "blocking audio submission entered the runtime");
    auto finish = std::async(std::launch::async, [&] { session.finish(); });
    expect_true(finish.wait_for(10ms) == std::future_status::timeout,
                "finish waits until the active audio submission returns");
    submit.get();
    finish.get();
  }

  std::filesystem::remove(event_path);
  {
    auto session = GpuVideoEncodeSession::open(audio_config(), runtime);
    auto submit =
        std::async(std::launch::async, [&] { session.submit_audio_packet(audio_packet(1)); });
    expect_true(wait_for_event(event_path, "audio-push-enter"),
                "second blocking audio submission entered the runtime");
    auto abort = std::async(std::launch::async, [&] { session.abort(); });
    expect_true(abort.wait_for(10ms) == std::future_status::timeout,
                "abort waits until the active audio submission returns");
    submit.get();
    abort.get();
  }
}

void compressed_audio_source_trims_and_rebases_chained_segments(
    const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("audio-packets");
  auto source = AudioPassthroughSource::open(
      {.paths = {"first.mp4", "second.mp4"}, .start_time_ns = 20'000'000ULL});
  expect_true(source.caps() == "audio/mpeg, mpegversion=(int)4, rate=(int)48000, channels=(int)2",
              "audio source exposes parser-negotiated compressed caps");

  std::vector<std::uint64_t> timestamps;
  std::vector<unsigned int> first_bytes;
  for (;;) {
    auto result = source.read();
    if (result.status == AudioPassthroughStatus::EndOfStream) {
      break;
    }
    expect_true(result.packet.has_value(), "audio packet result contains a packet");
    if (!result.packet.has_value()) {
      continue;
    }
    timestamps.push_back(result.packet->pts_ns.value_or(std::numeric_limits<std::uint64_t>::max()));
    expect_eq(result.packet->duration_ns, 20'000'000ULL, "audio packet duration is retained");
    expect_eq(result.packet->bytes.size(), 4U, "audio packet bytes stay bounded and exact");
    if (!result.packet->bytes.empty()) {
      first_bytes.push_back(std::to_integer<unsigned int>(result.packet->bytes.front()));
    }
  }
  expect_true(timestamps == std::vector<std::uint64_t>({0, 20'000'000, 40'000'000, 60'000'000,
                                                        80'000'000, 100'000'000, 120'000'000}),
              "trimmed chained audio is rebased to one gapless output timeline");
  expect_true(first_bytes == std::vector<unsigned int>({1, 2, 3, 0, 1, 2, 3}),
              "audio packet payloads are copied exactly once in segment order");
  const auto events = read_events(event_path);
  expect_eq(count_event(events, "parse-audio"), 2U,
            "audio source opens each compressed segment exactly once");
  expect_eq(count_event(events, "seek-compressed"), 1U,
            "audio start trim seeks only the first selected segment");
  expect_eq(count_event(events, "decoder-element"), 0U,
            "compressed audio passthrough never creates a decoder");
}

void compressed_audio_source_handles_absence_and_incompatible_caps(
    const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("audio-no-stream");
  auto absent = AudioPassthroughSource::open({.paths = {"first.mp4", "second.mp4"}});
  expect_true(!absent.caps().has_value(), "audio-free segments expose no mux caps");
  expect_true(absent.read().status == AudioPassthroughStatus::EndOfStream,
              "audio-free segment chain ends cleanly");
  const auto absent_events = read_events(event_path);
  expect_eq(count_event(absent_events, "discover-audio"), 2U,
            "audio-free segments are discovered explicitly");
  expect_eq(count_event(absent_events, "parse-audio"), 0U,
            "audio-free segments never open a compressed demux pipeline");

  std::filesystem::remove(event_path);
  set_scenario("audio-incompatible-segments");
  auto incompatible = AudioPassthroughSource::open({.paths = {"first.mp4", "second.mp4"}});
  for (int index = 0; index < 4; ++index) {
    expect_true(incompatible.read().status == AudioPassthroughStatus::Packet,
                "first compatible audio segment is readable");
  }
  expect_audio_error([&] { (void)incompatible.read(); }, "incompatible caps",
                     "incompatible chained compressed audio fails closed");
}

} // namespace

int main() {
#if defined(__linux__)
  const auto gstreamer = find_fake_runtime_runfile("fake_gstreamer_runtime");
  const auto nvbufsurface = find_fake_runtime_runfile("fake_nvbufsurface.so");
  const auto cuda = find_fake_runtime_runfile("fake_cuda_driver");
  set_environment("RECO_GSTREAMER_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GLIB_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GSTPBUTILS_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GOBJECT_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_NVBUFSURFACE_DYLIB_PATH", nvbufsurface.string());
  set_environment("RECO_NVDS_UTILS_DYLIB_PATH", nvbufsurface.string());
  set_environment("RECO_CUDA_DRIVER_DYLIB_PATH", cuda.string());
  const auto event_path =
      std::filesystem::temp_directory_path() /
      ("reco_fake_gpu_encode_events_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
  set_environment("RECO_FAKE_GST_EVENT_PATH", event_path.string());

  try {
    const auto runtime = discover_nvbufsurface_runtime();
    startup_failures_release_partial_resources(runtime, event_path);
    wrapped_callbacks_and_move_assignment_release_exactly_once(runtime);
    wrapping_and_push_failures_preserve_pool_ownership(runtime);
    bus_errors_and_early_eos_are_sticky(runtime);
    bounded_pool_times_out_and_releases_on_abort(runtime);
    finalization_failures_abort_and_release(runtime);
    successful_finish_waits_for_downstream_release(runtime);
    outstanding_leases_survive_session_destruction(runtime);
    compressed_audio_packets_use_the_bounded_audio_appsrc(runtime, event_path);
    compressed_audio_backpressure_bounds_packets_and_bytes(runtime, event_path);
    compressed_audio_preserves_only_buffer_semantic_flags(runtime, event_path);
    submission_is_serialized_with_finish_and_abort(runtime, event_path);
    compressed_audio_source_trims_and_rebases_chained_segments(event_path);
    compressed_audio_source_handles_absence_and_incompatible_caps(event_path);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected top-level error: " << error.what() << '\n';
    ++failures;
  }
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
