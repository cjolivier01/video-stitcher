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

#if defined(__linux__)
#include <dlfcn.h>
#endif

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

std::filesystem::path find_probe_worker_runfile() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || runfiles[0] == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
#if defined(_WIN32)
  constexpr std::string_view worker_name = "reco_video_probe_worker.exe";
#else
  constexpr std::string_view worker_name = "reco_video_probe_worker";
#endif
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    if (entry.path().filename() == worker_name && std::filesystem::is_regular_file(entry.path())) {
      return std::filesystem::absolute(entry.path());
    }
  }
  throw std::runtime_error("video probe worker runfile not found");
}

#if defined(__linux__)
class FakeNvbufSurfaceControl final {
public:
  explicit FakeNvbufSurfaceControl(const std::filesystem::path& path) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      throw std::runtime_error("failed to open fake NvBufSurface control library");
    }
    reset_ =
        reinterpret_cast<void (*)()>(dlsym(library_, "recoFakeNvbufSurfaceResetAllocationCount"));
    count_ = reinterpret_cast<std::uint64_t (*)()>(
        dlsym(library_, "recoFakeNvbufSurfaceAllocationCount"));
    if (reset_ == nullptr || count_ == nullptr) {
      dlclose(library_);
      library_ = nullptr;
      throw std::runtime_error("fake NvBufSurface allocation controls are missing");
    }
  }

  ~FakeNvbufSurfaceControl() {
    if (library_ != nullptr) {
      dlclose(library_);
    }
  }

  FakeNvbufSurfaceControl(const FakeNvbufSurfaceControl&) = delete;
  FakeNvbufSurfaceControl& operator=(const FakeNvbufSurfaceControl&) = delete;

  void reset() const { reset_(); }
  [[nodiscard]] std::uint64_t count() const { return count_(); }

private:
  void* library_ = nullptr;
  void (*reset_)() = nullptr;
  std::uint64_t (*count_)() = nullptr;
};
#endif

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

#if defined(__linux__)
void gpu_memory_preflight_prevents_partial_nvmm_pool_allocation(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const FakeNvbufSurfaceControl& nvbuf) {
  constexpr std::uint64_t mebibyte = 1024ULL * 1024ULL;
  constexpr std::uint64_t gibibyte = 1024ULL * mebibyte;
  set_scenario("encode-success");

  set_environment("RECO_FAKE_CUDA_TOTAL_BYTES", std::to_string(8ULL * gibibyte));
  set_environment("RECO_FAKE_CUDA_FREE_BYTES", std::to_string(64ULL * mebibyte));
  set_environment("RECO_FAKE_CUDA_INTEGRATED", "0");
  nvbuf.reset();
  auto trace = std::make_shared<Trace>();
  expect_encode_error([&] { (void)open_session(runtime, trace); }, "reduce output dimensions",
                      "insufficient discrete GPU memory fails preflight");
  expect_eq(nvbuf.count(), 0ULL, "failed preflight creates no NvBufSurface allocation");
  expect_eq(trace->allocated.load(), 0U, "failed preflight publishes no pool allocation");

  set_environment("RECO_FAKE_CUDA_FREE_BYTES", std::to_string(1536ULL * mebibyte));
  set_environment("RECO_FAKE_CUDA_INTEGRATED", "1");
  nvbuf.reset();
  expect_encode_error([&] { (void)open_session(runtime); }, "integrated CUDA device",
                      "integrated GPU keeps shared-memory safety reserve");
  expect_eq(nvbuf.count(), 0ULL,
            "integrated-memory preflight fails before the first NvBufSurface allocation");

  set_environment("RECO_FAKE_CUDA_FREE_BYTES", std::to_string(6ULL * gibibyte));
  set_environment("RECO_FAKE_CUDA_INTEGRATED", "0");
  nvbuf.reset();
  {
    auto session = open_session(runtime);
    session.abort();
  }
  expect_eq(nvbuf.count(), 8ULL, "normal GPU budget allocates the complete bounded pool");
}
#endif

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

void encoder_opening_observer_interrupts_startup(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const std::filesystem::path& event_path) {
  using namespace std::chrono_literals;
  std::filesystem::remove(event_path);
  set_scenario("encode-stop-blocked-open");
  std::atomic<GpuVideoEncodeSession*> opening_session{nullptr};
  auto opening = std::async(std::launch::async, [&] {
    return GpuVideoEncodeSession::open(config(), runtime, {}, [&](GpuVideoEncodeSession* session) {
      opening_session.store(session, std::memory_order_release);
      return true;
    });
  });

  expect_true(wait_for_event(event_path, "encode-get-state-blocked"),
              "encoder blocks after its opening observer attaches");
  auto* partial_session = opening_session.load(std::memory_order_acquire);
  expect_true(partial_session != nullptr, "opening observer exposes the partial encoder session");
  if (partial_session != nullptr) {
    partial_session->abort();
  }
  expect_true(opening.wait_for(500ms) == std::future_status::ready,
              "partial encoder abort interrupts native startup");
  auto session = opening.get();
  expect_true(opening_session.load(std::memory_order_acquire) == nullptr,
              "encoder opening observer is cleared before return");
  expect_encode_error([&] { (void)session.acquire_frame(); }, "no longer accepting frames",
                      "interrupted encoder returns in the aborted state");
  const auto events = read_events(event_path);
  expect_eq(count_event(events, "encode-get-state-unblocked"), 1U,
            "encoder startup returns after its abort request");
  expect_true(count_event(events, "state-null") >= 1U,
              "encoder opening cancellation stops the partial pipeline");
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
  {
    set_scenario("encode-wrap-error");
    auto session = GpuVideoEncodeSession::open(audio_config(), runtime);
    expect_encode_error([&] { session.submit_audio_packet(audio_packet(16U * 1024U * 1024U)); },
                        "failed to wrap", "compressed-audio owner allocation failure");
    set_scenario("encode-audio-backpressure");
    session.submit_audio_packet(audio_packet(16U * 1024U * 1024U));
    session.submit_audio_packet(audio_packet(16U * 1024U * 1024U));
    expect_encode_error([&] { session.submit_audio_packet(audio_packet(1)); }, "timed out waiting",
                        "audio allocation failure rolls back aggregate accounting");
    session.abort();
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

void concurrent_acquire_cannot_consume_final_eos(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime,
    const std::filesystem::path& event_path) {
  using namespace std::chrono_literals;
  std::filesystem::remove(event_path);
  auto gate_path = event_path;
  gate_path += ".final-poll-gate";
  std::filesystem::remove(gate_path);
  set_environment("RECO_FAKE_GST_FINAL_POLL_GATE_PATH", gate_path.string());
  set_scenario("encode-concurrent-acquire-finish");

  auto finish_config = config();
  finish_config.finalize_timeout = 2s;
  auto session = GpuVideoEncodeSession::open(std::move(finish_config), runtime);
  auto finish = std::async(std::launch::async, [&] { session.finish(); });
  expect_true(wait_for_event(event_path, "encode-final-poll-blocked"),
              "finish reaches the gated terminal bus poll");

  auto acquire = std::async(std::launch::async, [&] { return session.acquire_frame(); });
  expect_true(acquire.wait_for(50ms) == std::future_status::timeout,
              "concurrent acquire cannot poll the bus during finalization");
  {
    std::ofstream gate(gate_path);
    gate << "release\n";
  }

  finish.get();
  expect_encode_error([&] { (void)acquire.get(); }, "no longer accepting",
                      "concurrent acquire observes completed finalization");
  std::filesystem::remove(gate_path);
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

void finalized_output_requires_a_compressed_video_sample(const std::filesystem::path& worker,
                                                         const std::filesystem::path& event_path) {
  auto output_path = event_path;
  output_path.replace_extension(".mp4");
  {
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    output << "fake muxed output";
  }

  std::filesystem::remove(event_path);
  set_scenario("probe-exact-frame-count");
  verify_muxed_gpu_video_output(output_path, Codec::H264, Format::Mp4, worker,
                                std::chrono::seconds(10));
  auto events = read_events(event_path);
  expect_eq(count_event(events, "parse-probe"), 1U,
            "output verification uses the parser-only worker topology");
  expect_eq(count_event(events, "probe-codec-filter"), 1U,
            "output verification selects a compressed video stream");
  expect_eq(count_event(events, "probe-exact-codec-filter"), 1U,
            "output verification selects only the configured video codec");
  expect_eq(count_event(events, "probe-h264-parser"), 1U,
            "output verification uses the explicit H.264 parser");
  expect_eq(count_event(events, "probe-parsebin"), 0U,
            "output verification does not autoplug parser or decoder factories");
  expect_eq(count_event(events, "probe-decoder-caps"), 1U,
            "output verification requires compressed access-unit caps");
  expect_eq(count_event(events, "decoder-element"), 0U,
            "output verification never constructs a decoder element");
  expect_eq(count_event(events, "raw-video-caps"), 0U,
            "output verification never requests decoded pixels");
  expect_eq(count_event(events, "discover-audio"), 0U,
            "output verification never invokes GstDiscoverer");

  auto retained_path = output_path;
  retained_path += ".retained";
  const auto retained_output = StableMediaFile::open(output_path);
  std::filesystem::rename(output_path, retained_path);
  {
    std::ofstream substitute(output_path, std::ios::binary | std::ios::trunc);
    substitute << "pathname substitute that must not be probed";
  }
  std::filesystem::remove(event_path);
  verify_muxed_gpu_video_output(retained_output, Codec::H264, Format::Mp4, worker,
                                std::chrono::seconds(10));
  events = read_events(event_path);
  expect_eq(count_event(events, "probe-fd-source"), 1U,
            "output verification transfers the retained readable authority");
  expect_eq(count_event(events, "probe-file-source"), 0U,
            "output verification never reopens the substituted diagnostic pathname");
  std::filesystem::remove(output_path);
  std::filesystem::rename(retained_path, output_path);

  std::filesystem::remove(event_path);
  set_scenario("probe-av1-exact-frame-count");
  verify_muxed_gpu_video_output(output_path, Codec::AV1, Format::Mkv, worker,
                                std::chrono::seconds(10));
  events = read_events(event_path);
  expect_eq(count_event(events, "probe-av1-parser"), 1U,
            "AV1 output verification uses the explicit AV1 parser");
  expect_eq(count_event(events, "probe-exact-codec-filter"), 1U,
            "AV1 output verification selects only AV1 samples");
  expect_eq(count_event(events, "decoder-element"), 0U,
            "AV1 output verification never constructs a decoder");

  std::filesystem::remove(event_path);
  set_scenario("probe-exact-frame-count");
  verify_muxed_gpu_video_output(output_path, Codec::H264, Format::Flv, worker,
                                std::chrono::seconds(10));
  events = read_events(event_path);
  expect_eq(count_event(events, "probe-flv-demux"), 1U,
            "FLV output verification uses the explicit compressed demuxer");
  expect_eq(count_event(events, "decoder-element"), 0U,
            "FLV output verification never constructs a decoder");

  std::filesystem::remove(event_path);
  set_scenario("probe-video-caps-zero-samples");
  expect_encode_error(
      [&] {
        verify_muxed_gpu_video_output(output_path, Codec::H264, Format::Mp4, worker,
                                      std::chrono::seconds(10));
      },
      "found no H.264, HEVC, or AV1 moving-video stream",
      "video caps without a compressed sample are rejected before publication");
  events = read_events(event_path);
  expect_eq(count_event(events, "probe-codec-filter"), 1U,
            "zero-sample fixture still exposes selected compressed video caps");
  expect_eq(count_event(events, "probe-parsebin"), 0U,
            "zero-sample verification does not enable autoplugging");
  expect_eq(count_event(events, "decoder-element"), 0U,
            "zero-sample verification does not fall back to a decoder");

  std::filesystem::remove(event_path);
  set_scenario("probe-timeout");
  std::atomic<bool> cancel{false};
  std::thread requester([&] {
    (void)wait_for_event(event_path, "pull-probe");
    cancel.store(true, std::memory_order_release);
  });
  const auto started = std::chrono::steady_clock::now();
  bool cancellation_reported = false;
  try {
    verify_muxed_gpu_video_output(output_path, Codec::H264, Format::Mp4, worker,
                                  std::chrono::seconds(30),
                                  [&] { return cancel.load(std::memory_order_acquire); });
  } catch (const GpuVideoProbeCancelled&) {
    cancellation_reported = true;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: output verification returned the wrong cancellation error: " << error.what()
              << '\n';
    ++failures;
  }
  requester.join();
  expect_true(cancellation_reported, "output verification preserves explicit probe cancellation");
  expect_true(std::chrono::steady_clock::now() - started < std::chrono::seconds(3),
              "output-verification cancellation does not wait for the parser timeout");

  std::uint32_t cancellation_queries = 0;
  cancellation_reported = false;
  try {
    verify_muxed_gpu_video_output(output_path, Codec::H264, Format::Mp4, worker,
                                  std::chrono::seconds(10),
                                  [&] { return cancellation_queries++ == 0U; });
  } catch (const GpuVideoProbeCancelled&) {
    cancellation_reported = true;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: one-shot output cancellation returned the wrong error: " << error.what()
              << '\n';
    ++failures;
  }
  expect_true(cancellation_reported,
              "one-shot cancellation remains distinguishable without callback relatching");
  std::filesystem::remove(output_path);
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

void finish_is_serialized_and_abort_interrupts_waits(
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
    expect_true(abort.wait_for(50ms) == std::future_status::ready,
                "abort does not wait for an active audio submission");
    abort.get();
    submit.get();
  }

  std::filesystem::remove(event_path);
  set_scenario("encode-audio-backpressure");
  {
    auto backpressure_config = audio_config();
    backpressure_config.acquire_timeout = std::chrono::seconds(30);
    auto session = GpuVideoEncodeSession::open(std::move(backpressure_config), runtime);
    for (std::size_t index = 0; index < 32; ++index) {
      session.submit_audio_packet(audio_packet(1));
    }
    auto blocked_submit =
        std::async(std::launch::async, [&] { session.submit_audio_packet(audio_packet(1)); });
    expect_true(blocked_submit.wait_for(10ms) == std::future_status::timeout,
                "audio submission blocks at the bounded mux capacity");
    const auto abort_started = std::chrono::steady_clock::now();
    session.abort();
    expect_true(std::chrono::steady_clock::now() - abort_started < 500ms,
                "abort returns promptly during audio backpressure");
    expect_true(blocked_submit.wait_for(500ms) == std::future_status::ready,
                "abort wakes audio backpressure immediately");
    expect_encode_error([&] { blocked_submit.get(); }, "no longer accepting",
                        "aborted audio backpressure reports terminal state");
  }

  std::filesystem::remove(event_path);
  set_scenario("encode-finalize-timeout");
  {
    auto finalize_config = config();
    finalize_config.finalize_timeout = std::chrono::seconds(30);
    auto session = GpuVideoEncodeSession::open(std::move(finalize_config), runtime);
    auto frame = session.acquire_frame();
    session.submit_frame(std::move(frame), 0, 1);
    auto finish = std::async(std::launch::async, [&] { session.finish(); });
    expect_true(wait_for_event(event_path, "appsrc-eos"),
                "encoder entered mux finalization before abort");
    const auto abort_started = std::chrono::steady_clock::now();
    session.abort();
    expect_true(std::chrono::steady_clock::now() - abort_started < 500ms,
                "abort returns promptly during mux finalization");
    expect_true(finish.wait_for(500ms) == std::future_status::ready,
                "abort interrupts mux finalization polling");
    expect_encode_error([&] { finish.get(); }, "aborted", "interrupted finalization reports abort");
  }
}

void compressed_audio_source_trims_and_rebases_chained_segments(
    const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("audio-packets");
  auto source = AudioPassthroughSource::open(
      {.segments = {{.path = "first.mp4", .video_duration_ns = 80'000'000ULL},
                    {.path = "second.mp4", .video_duration_ns = 80'000'000ULL}},
       .start_time_ns = 20'000'000ULL});
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

  std::filesystem::remove(event_path);
  set_scenario("audio-silent-middle");
  auto with_silent_segment = AudioPassthroughSource::open(
      {.segments = {{.path = "first.mp4", .video_duration_ns = 80'000'000ULL},
                    {.path = "silent.mp4", .video_duration_ns = 100'000'000ULL},
                    {.path = "second.mp4", .video_duration_ns = 80'000'000ULL}}});
  timestamps.clear();
  for (;;) {
    auto result = with_silent_segment.read();
    if (result.status == AudioPassthroughStatus::EndOfStream) {
      break;
    }
    expect_true(result.packet.has_value(), "silent-gap audio result contains a packet");
    if (result.packet.has_value()) {
      timestamps.push_back(
          result.packet->pts_ns.value_or(std::numeric_limits<std::uint64_t>::max()));
    }
  }
  expect_true(timestamps ==
                  std::vector<std::uint64_t>({0, 20'000'000, 40'000'000, 60'000'000, 180'000'000,
                                              200'000'000, 220'000'000, 240'000'000}),
              "audio-free segment preserves its video-duration gap");
  const auto silent_events = read_events(event_path);
  expect_eq(count_event(silent_events, "discover-audio"), 3U,
            "every container segment is inspected for audio");
  expect_eq(count_event(silent_events, "parse-audio"), 2U,
            "audio-free segment does not construct a demux pipeline");
}

void compressed_audio_source_handles_absence_and_incompatible_caps(
    const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("audio-no-stream");
  auto absent = AudioPassthroughSource::open(
      {.segments = {{.path = "first.mp4", .video_duration_ns = 80'000'000ULL},
                    {.path = "second.mp4", .video_duration_ns = 80'000'000ULL}}});
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
  auto incompatible = AudioPassthroughSource::open(
      {.segments = {{.path = "first.mp4", .video_duration_ns = 80'000'000ULL},
                    {.path = "second.mp4", .video_duration_ns = 80'000'000ULL}}});
  for (int index = 0; index < 4; ++index) {
    expect_true(incompatible.read().status == AudioPassthroughStatus::Packet,
                "first compatible audio segment is readable");
  }
  expect_audio_error([&] { (void)incompatible.read(); }, "incompatible caps",
                     "incompatible chained compressed audio fails closed");
}

void compressed_audio_preserves_stream_time_offsets(const std::filesystem::path& event_path) {
  std::filesystem::remove(event_path);
  set_scenario("audio-nonzero-origin-delayed");
  auto source = AudioPassthroughSource::open(
      {.segments = {{.path = "first.mp4", .video_duration_ns = 80'000'000ULL},
                    {.path = "second.mp4", .video_duration_ns = 80'000'000ULL}},
       .start_time_ns = 20'000'000ULL});

  std::vector<std::uint64_t> timestamps;
  std::vector<std::uint64_t> durations;
  for (;;) {
    auto result = source.read();
    if (result.status == AudioPassthroughStatus::EndOfStream) {
      break;
    }
    expect_true(result.packet.has_value(), "offset audio result contains a packet");
    if (result.packet.has_value()) {
      timestamps.push_back(
          result.packet->pts_ns.value_or(std::numeric_limits<std::uint64_t>::max()));
      durations.push_back(result.packet->duration_ns);
    }
  }
  expect_true(timestamps ==
                  std::vector<std::uint64_t>({10'000'000, 30'000'000, 50'000'000, 70'000'000,
                                              90'000'000, 110'000'000, 130'000'000}),
              "stream-time conversion preserves delayed audio across trimmed video segments");
  expect_true(durations ==
                  std::vector<std::uint64_t>({20'000'000, 20'000'000, 10'000'000, 20'000'000,
                                              20'000'000, 20'000'000, 10'000'000}),
              "audio packets are clipped to cumulative video segment boundaries");
}

void compressed_audio_prime_failures_release_resources(const std::filesystem::path& event_path) {
  for (const auto& [scenario, error] : std::vector<std::pair<std::string_view, std::string_view>>{
           {"audio-prime-timeout", "timed out waiting"},
           {"audio-invalid-segment-time", "cannot be converted to stream time"},
           {"audio-seek-error", "failed to seek compressed audio"},
       }) {
    std::filesystem::remove(event_path);
    set_scenario(scenario);
    expect_audio_error(
        [&] {
          (void)AudioPassthroughSource::open(
              {.segments = {{.path = "first.mp4", .video_duration_ns = 80'000'000ULL}},
               .start_time_ns = scenario == "audio-seek-error" ? 20'000'000ULL : 0ULL});
        },
        error, "audio prime failure is reported");
    const auto events = read_events(event_path);
    expect_eq(count_event(events, "state-playing"), 1U, "audio prime failure starts one pipeline");
    expect_eq(count_event(events, "state-null"), 1U,
              "audio prime failure stops the partial pipeline");
    expect_eq(count_event(events, "unref-sink"), 1U,
              "audio prime failure releases the partial sink");
    expect_eq(count_event(events, "unref-bus"), 1U, "audio prime failure releases the partial bus");
    expect_eq(count_event(events, "unref-pipeline"), 1U,
              "audio prime failure releases the partial pipeline");
    expect_eq(count_event(events, "unref-discoverer"), 1U,
              "audio prime failure releases its discoverer");
  }
}

void compressed_audio_opening_observer_interrupts_discovery(
    const std::filesystem::path& event_path) {
  using namespace std::chrono_literals;
  std::filesystem::remove(event_path);
  set_scenario("audio-stop-blocked-discovery");
  std::atomic<AudioPassthroughSource*> opening_source{nullptr};
  auto opening = std::async(std::launch::async, [&] {
    return AudioPassthroughSource::open(
        {.segments = {{.path = "second.mp4", .video_duration_ns = 80'000'000ULL}},
         .read_timeout = std::chrono::seconds(30)},
        [&](AudioPassthroughSource* source) {
          opening_source.store(source, std::memory_order_release);
          return true;
        });
  });

  expect_true(wait_for_event(event_path, "discover-audio-blocked"),
              "audio discovery blocks after its opening observer attaches");
  auto* partial_source = opening_source.load(std::memory_order_acquire);
  expect_true(partial_source != nullptr, "opening observer exposes the partial audio source");
  if (partial_source != nullptr) {
    partial_source->request_stop();
  }
  expect_true(opening.wait_for(500ms) == std::future_status::ready,
              "partial audio stop interrupts initial stream discovery");
  auto source = opening.get();
  expect_true(opening_source.load(std::memory_order_acquire) == nullptr,
              "audio opening observer is cleared before return");
  expect_true(source.read().status == AudioPassthroughStatus::EndOfStream,
              "interrupted audio open returns in the stopped state");
  expect_eq(count_event(read_events(event_path), "unref-discoverer"), 1U,
            "interrupted initial audio discovery releases its discoverer");
}

void compressed_audio_stop_interrupts_concurrent_read(const std::filesystem::path& event_path) {
  using namespace std::chrono_literals;
  std::filesystem::remove(event_path);
  set_scenario("audio-stop-blocked-read");
  auto source = AudioPassthroughSource::open(
      {.segments = {{.path = "first.mp4", .video_duration_ns = 80'000'000ULL}},
       .read_timeout = std::chrono::seconds(30)});
  expect_true(source.read().status == AudioPassthroughStatus::Packet,
              "primed compressed audio packet is returned before blocked read");
  auto read = std::async(std::launch::async, [&] { return source.read(); });
  expect_true(wait_for_event(event_path, "audio-pull-blocked"),
              "compressed audio read entered the blocking runtime call");
  auto stop = std::async(std::launch::async, [&] { source.request_stop(); });
  expect_true(stop.wait_for(500ms) == std::future_status::ready,
              "audio request_stop interrupts and joins a concurrent read");
  stop.get();
  expect_true(read.wait_for(500ms) == std::future_status::ready,
              "interrupted compressed audio read returns promptly");
  expect_true(read.get().status == AudioPassthroughStatus::EndOfStream,
              "interrupted compressed audio read reports terminal status");
  source.request_stop();
}

void compressed_audio_stop_interrupts_discovery(const std::filesystem::path& event_path) {
  using namespace std::chrono_literals;
  std::filesystem::remove(event_path);
  set_scenario("audio-stop-blocked-discovery");
  auto source = AudioPassthroughSource::open(
      {.segments = {{.path = "first.mp4", .video_duration_ns = 20'000'000ULL},
                    {.path = "second.mp4", .video_duration_ns = 80'000'000ULL}},
       .read_timeout = std::chrono::seconds(30)});
  expect_true(source.read().status == AudioPassthroughStatus::Packet,
              "primed audio packet is returned before blocked discovery");
  auto read = std::async(std::launch::async, [&] { return source.read(); });
  expect_true(wait_for_event(event_path, "discover-audio-blocked"),
              "compressed audio read entered blocked segment discovery");
  auto stop = std::async(std::launch::async, [&] { source.request_stop(); });
  expect_true(stop.wait_for(500ms) == std::future_status::ready,
              "audio request_stop interrupts and joins segment discovery");
  stop.get();
  expect_true(read.wait_for(500ms) == std::future_status::ready,
              "interrupted segment discovery returns promptly");
  expect_true(read.get().status == AudioPassthroughStatus::EndOfStream,
              "interrupted segment discovery reports terminal status");
  expect_eq(count_event(read_events(event_path), "unref-discoverer"), 2U,
            "interrupted segment discovery releases every discoverer");
}

} // namespace

int main() {
#if defined(__linux__)
  const auto gstreamer = find_fake_runtime_runfile("fake_gstreamer_runtime");
  const auto nvbufsurface = find_fake_runtime_runfile("fake_nvbufsurface.so");
  const auto cuda = find_fake_runtime_runfile("fake_cuda_driver");
  const auto probe_worker = find_probe_worker_runfile();
  set_environment("RECO_GSTREAMER_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GSTAPP_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GLIB_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GSTPBUTILS_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_GOBJECT_DYLIB_PATH", gstreamer.string());
  set_environment("RECO_NVBUFSURFACE_DYLIB_PATH", nvbufsurface.string());
  set_environment("RECO_NVDS_UTILS_DYLIB_PATH", nvbufsurface.string());
  set_environment("RECO_CUDA_DRIVER_DYLIB_PATH", cuda.string());
  set_environment("RECO_FAKE_CUDA_TOTAL_BYTES", std::to_string(8ULL * 1024ULL * 1024ULL * 1024ULL));
  set_environment("RECO_FAKE_CUDA_FREE_BYTES", std::to_string(6ULL * 1024ULL * 1024ULL * 1024ULL));
  set_environment("RECO_FAKE_CUDA_INTEGRATED", "0");
  const auto event_path =
      std::filesystem::temp_directory_path() /
      ("reco_fake_gpu_encode_events_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
  set_environment("RECO_FAKE_GST_EVENT_PATH", event_path.string());

  try {
    const FakeNvbufSurfaceControl nvbuf_control(nvbufsurface);
    const auto runtime = discover_nvbufsurface_runtime();
    gpu_memory_preflight_prevents_partial_nvmm_pool_allocation(runtime, nvbuf_control);
    startup_failures_release_partial_resources(runtime, event_path);
    encoder_opening_observer_interrupts_startup(runtime, event_path);
    wrapped_callbacks_and_move_assignment_release_exactly_once(runtime);
    wrapping_and_push_failures_preserve_pool_ownership(runtime);
    bus_errors_and_early_eos_are_sticky(runtime);
    bounded_pool_times_out_and_releases_on_abort(runtime);
    finalization_failures_abort_and_release(runtime);
    successful_finish_waits_for_downstream_release(runtime);
    concurrent_acquire_cannot_consume_final_eos(runtime, event_path);
    outstanding_leases_survive_session_destruction(runtime);
    compressed_audio_packets_use_the_bounded_audio_appsrc(runtime, event_path);
    finalized_output_requires_a_compressed_video_sample(probe_worker, event_path);
    compressed_audio_backpressure_bounds_packets_and_bytes(runtime, event_path);
    compressed_audio_preserves_only_buffer_semantic_flags(runtime, event_path);
    finish_is_serialized_and_abort_interrupts_waits(runtime, event_path);
    compressed_audio_source_trims_and_rebases_chained_segments(event_path);
    compressed_audio_source_handles_absence_and_incompatible_caps(event_path);
    compressed_audio_preserves_stream_time_offsets(event_path);
    compressed_audio_prime_failures_release_resources(event_path);
    compressed_audio_opening_observer_interrupts_discovery(event_path);
    compressed_audio_stop_interrupts_concurrent_read(event_path);
    compressed_audio_stop_interrupts_discovery(event_path);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected top-level error: " << error.what() << '\n';
    ++failures;
  }
  std::filesystem::remove(event_path);
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
