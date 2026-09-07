#include "reco/gui/gpu_preview_controller.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace reco::gui;

namespace {

using namespace std::chrono_literals;

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
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable, std::string_view message) {
  try {
    callable();
    expect_true(false, message);
  } catch (const Exception&) {
    expect_true(true, message);
  } catch (...) {
    expect_true(false, message);
  }
}

GpuPreviewControllerConfig valid_config() {
  return {.left_path = "/video/left.mp4",
          .right_path = "/video/right.mp4",
          .calibration_path = "/calibration/match.json",
          .probe_worker_path = std::filesystem::current_path() / "app" / "reco_video_probe_worker",
          .native_window_handle = 1U,
          .output_width = 1280U,
          .output_height = 720U,
          .sink = reco::io::GpuPreviewSink::NvidiaEgl,
          .device_ordinal = 0U,
          .surface_pool_capacity = 3U,
          .decode_queue_capacity = 4U,
          .start_playing = false};
}

struct FakeBackendState {
  std::mutex mutex;
  std::condition_variable changed;
  GpuPreviewBackendStreamInfo info{.fps_numerator = 30U,
                                   .fps_denominator = 1U,
                                   .total_frames = 100U,
                                   .total_frames_exact = true};
  bool fail_initialize = false;
  bool hold_initialize = false;
  bool initialized = false;
  bool interrupted = false;
  bool shutdown = false;
  bool eos = false;
  std::uint64_t current_frame = 0U;
  std::size_t permits = 0U;
  std::size_t blocked_reads = 0U;
  std::size_t interrupt_count = 0U;
  std::size_t shutdown_count = 0U;
  std::size_t release_count = 0U;
  std::vector<std::uint64_t> rebuilds;
  std::vector<std::uint64_t> timestamps;
  std::vector<std::uint64_t> durations;
  std::vector<GpuPreviewViewport> viewports;
};

class FakeBackend final : public GpuPreviewControllerBackend {
public:
  explicit FakeBackend(std::shared_ptr<FakeBackendState> state) : state_(std::move(state)) {}

  void prepare_start() override {
    std::lock_guard lock(state_->mutex);
    state_->shutdown = false;
    state_->interrupted = false;
    state_->eos = false;
  }

  GpuPreviewBackendStreamInfo initialize(const GpuPreviewControllerConfig&) override {
    std::unique_lock lock(state_->mutex);
    if (state_->fail_initialize) {
      throw std::runtime_error("synthetic initialization failure");
    }
    state_->initialized = true;
    state_->changed.notify_all();
    state_->changed.wait(lock, [&] { return !state_->hold_initialize || state_->shutdown; });
    return state_->info;
  }

  void rebuild_decode(std::uint64_t frame_index) override {
    std::lock_guard lock(state_->mutex);
    state_->current_frame = frame_index;
    state_->interrupted = false;
    state_->rebuilds.push_back(frame_index);
    state_->changed.notify_all();
  }

  GpuPreviewBackendFrameResult present_next(const GpuPreviewViewport& viewport,
                                            std::uint64_t pts_ns,
                                            std::uint64_t duration_ns) override {
    std::unique_lock lock(state_->mutex);
    ++state_->blocked_reads;
    state_->changed.notify_all();
    state_->changed.wait(lock, [&] {
      return state_->shutdown || state_->interrupted || state_->eos || state_->permits != 0U;
    });
    if (state_->shutdown || state_->interrupted) {
      state_->interrupted = false;
      return {.status = GpuPreviewBackendFrameStatus::Stopped, .source_frame = std::nullopt};
    }
    if (state_->eos) {
      return {.status = GpuPreviewBackendFrameStatus::EndOfStream, .source_frame = std::nullopt};
    }
    --state_->permits;
    const auto source_frame = state_->current_frame++;
    state_->timestamps.push_back(pts_ns);
    state_->durations.push_back(duration_ns);
    state_->viewports.push_back(viewport);
    state_->changed.notify_all();
    return {.status = GpuPreviewBackendFrameStatus::Presented, .source_frame = source_frame};
  }

  void interrupt_decode() noexcept override {
    std::lock_guard lock(state_->mutex);
    ++state_->interrupt_count;
    state_->interrupted = true;
    state_->changed.notify_all();
  }

  void shutdown() noexcept override {
    std::lock_guard lock(state_->mutex);
    ++state_->shutdown_count;
    state_->shutdown = true;
    state_->changed.notify_all();
  }

  void release() noexcept override {
    std::lock_guard lock(state_->mutex);
    ++state_->release_count;
    state_->changed.notify_all();
  }

private:
  std::shared_ptr<FakeBackendState> state_;
};

template <typename Predicate>
bool wait_for_backend(const std::shared_ptr<FakeBackendState>& state, Predicate&& predicate) {
  std::unique_lock lock(state->mutex);
  return state->changed.wait_for(lock, 2s, std::forward<Predicate>(predicate));
}

template <typename Predicate>
bool wait_for_controller(GpuPreviewController& controller, Predicate&& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate(controller.snapshot())) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return predicate(controller.snapshot());
}

void validation_rejects_unsafe_boundaries() {
  auto config = valid_config();
  expect_true(!validate_gpu_preview_controller_config(config).has_value(),
              "valid controller config accepted");

  config.probe_worker_path = "relative/probe";
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "relative deployed probe path rejected");
  config = valid_config();
  config.output_width = 1279U;
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "odd output width rejected");
  config = valid_config();
  config.native_window_handle = 0U;
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "zero window handle rejected");
  config = valid_config();
  config.device_ordinal = 1U;
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "decode/render device mismatch is rejected");
  config = valid_config();
  config.surface_pool_capacity = 17U;
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "unbounded surface pool rejected");
  config = valid_config();
  config.decode_queue_capacity = 0U;
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "empty decode queue rejected");
  config = valid_config();
  config.probe_timeout = 999ms;
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "short probe timeout rejected");
  config = valid_config();
  config.blend_width_override = std::numeric_limits<float>::quiet_NaN();
  expect_true(validate_gpu_preview_controller_config(config).has_value(),
              "non-finite blend override rejected");

  expect_throws<std::invalid_argument>(
      [&] {
        GpuPreviewController controller(valid_config(),
                                        std::unique_ptr<GpuPreviewControllerBackend>{});
      },
      "null test backend rejected");
}

void controls_and_timestamps_are_deterministic() {
  constexpr float pi = 3.14159265358979323846F;
  auto viewport = clamp_gpu_preview_viewport(
      {.yaw_radians = 4.0F * pi, .pitch_radians = 4.0F, .fov_degrees = 500.0F});
  expect_near(viewport.yaw_radians, 0.0F, 1.0e-5F, "yaw wraps into one revolution");
  expect_near(viewport.pitch_radians, 89.0F * pi / 180.0F, 1.0e-5F,
              "pitch clamps below renderer pole");
  expect_near(viewport.fov_degrees, 150.0F, 1.0e-5F, "FOV clamps to GUI maximum");

  viewport = clamp_gpu_preview_viewport({.yaw_radians = std::numeric_limits<float>::quiet_NaN(),
                                         .pitch_radians = std::numeric_limits<float>::infinity(),
                                         .fov_degrees = std::numeric_limits<float>::quiet_NaN()});
  expect_near(viewport.yaw_radians, 0.0F, 0.0F, "non-finite yaw resets");
  expect_near(viewport.pitch_radians, 0.0F, 0.0F, "non-finite pitch resets");
  expect_near(viewport.fov_degrees, 75.0F, 0.0F, "non-finite FOV resets");

  expect_eq(gpu_preview_timestamp_for_frame(0U, 30'000U, 1'001U), 0ULL, "zero frame timestamp");
  expect_eq(gpu_preview_timestamp_for_frame(1U, 30'000U, 1'001U), 33'366'666ULL,
            "NTSC first frame timestamp floors exactly");
  expect_eq(gpu_preview_timestamp_for_frame(30'000U, 30'000U, 1'001U), 1'001'000'000'000ULL,
            "NTSC whole-cycle timestamp");
  expect_throws<std::invalid_argument>([] { (void)gpu_preview_timestamp_for_frame(1U, 0U, 1U); },
                                       "zero frame-rate numerator rejected");
  expect_throws<std::overflow_error>(
      [] {
        (void)gpu_preview_timestamp_for_frame(std::numeric_limits<std::uint64_t>::max(), 1U, 1U);
      },
      "timestamp overflow rejected");
}

void controller_transitions_seek_and_preserve_presentation_time() {
  auto state = std::make_shared<FakeBackendState>();
  std::atomic<std::size_t> notifications{0U};
  GpuPreviewController controller(valid_config(), std::make_unique<FakeBackend>(state),
                                  [&](GpuPreviewControllerSnapshot) {
                                    notifications.fetch_add(1U, std::memory_order_relaxed);
                                  });

  controller.start();
  expect_true(wait_for_controller(controller,
                                  [](const auto& snapshot) {
                                    return snapshot.state == GpuPreviewControllerState::Paused &&
                                           snapshot.ready;
                                  }),
              "controller initializes into configured pause state");
  auto snapshot = controller.snapshot();
  expect_eq(snapshot.total_frames, 100ULL, "aligned frame count published");
  expect_true(snapshot.total_frames_exact, "exact frame count published");
  expect_eq(snapshot.fps_numerator, 30U, "frame rate numerator published");

  controller.play();
  expect_true(wait_for_backend(state, [&] { return state->blocked_reads >= 1U; }),
              "play enters a backend decode read");
  controller.pause();
  const auto pause_state = controller.snapshot().state;
  expect_true(pause_state == GpuPreviewControllerState::Seeking ||
                  pause_state == GpuPreviewControllerState::Paused,
              "pause reports quiescing or an already-quiescent decoder");
  expect_true(wait_for_controller(controller,
                                  [](const auto& value) {
                                    return value.state == GpuPreviewControllerState::Paused;
                                  }),
              "pause reports Paused only after decode is rebuilt and idle");
  expect_true(wait_for_backend(
                  state, [&] { return !state->rebuilds.empty() && state->rebuilds.back() == 0U; }),
              "pause before the first frame preserves source frame zero");

  controller.seek(42U);
  expect_true(wait_for_backend(
                  state, [&] { return !state->rebuilds.empty() && state->rebuilds.back() == 42U; }),
              "seek interrupts and rebuilds indexed decode");
  expect_true(wait_for_controller(controller,
                                  [](const auto& value) {
                                    return value.state == GpuPreviewControllerState::Paused;
                                  }),
              "seek preserves paused playback state");
  {
    std::lock_guard lock(state->mutex);
    expect_true(state->interrupt_count >= 1U, "seek calls decode interruption");
  }

  controller.play();
  {
    std::lock_guard lock(state->mutex);
    ++state->permits;
    state->changed.notify_all();
  }
  expect_true(wait_for_controller(controller,
                                  [](const auto& value) { return value.presented_frames >= 1U; }),
              "first sought frame is presented");
  snapshot = controller.snapshot();
  expect_eq(*snapshot.current_frame, 42ULL, "first seek source frame published");
  expect_eq(*snapshot.presentation_timestamp_ns, 0ULL, "first presentation starts at zero");

  controller.seek(10U);
  expect_true(
      wait_for_backend(
          state, [&] { return state->rebuilds.size() >= 2U && state->rebuilds.back() == 10U; }),
      "second seek rebuild completes");
  {
    std::lock_guard lock(state->mutex);
    ++state->permits;
    state->changed.notify_all();
  }
  expect_true(wait_for_controller(controller,
                                  [](const auto& value) { return value.presented_frames >= 2U; }),
              "frame after second seek is presented");
  snapshot = controller.snapshot();
  expect_eq(*snapshot.current_frame, 10ULL, "second seek source frame published");
  expect_eq(*snapshot.presentation_timestamp_ns, 33'333'333ULL,
            "presentation timestamp remains monotonic after seek");

  {
    std::lock_guard lock(state->mutex);
    state->eos = true;
    state->changed.notify_all();
  }
  expect_true(wait_for_controller(controller,
                                  [](const auto& value) {
                                    return value.state == GpuPreviewControllerState::EndOfStream &&
                                           value.end_of_stream && value.ready;
                                  }),
              "EOS readiness snapshot is published");
  controller.play();
  expect_true(controller.snapshot().state == GpuPreviewControllerState::EndOfStream,
              "play does not leave EOS without a seek");
  expect_throws<std::out_of_range>([&] { controller.seek(100U); },
                                   "seek rejects the end-exclusive frame bound");

  controller.stop();
  snapshot = controller.snapshot();
  expect_true(snapshot.state == GpuPreviewControllerState::Stopped,
              "explicit stop joins into stopped state");
  expect_true(!snapshot.ready, "stopped controller is not ready");
  {
    std::lock_guard lock(state->mutex);
    expect_true(state->shutdown_count >= 1U, "stop interrupts backend blocking work");
    expect_true(state->release_count >= 1U, "stop releases persistent GPU resources");
  }
  expect_true(notifications.load(std::memory_order_relaxed) >= 8U,
              "state and frame notifications are delivered");
}

void initialization_errors_are_sticky_until_stop() {
  auto state = std::make_shared<FakeBackendState>();
  state->fail_initialize = true;
  GpuPreviewController controller(valid_config(), std::make_unique<FakeBackend>(state));
  controller.start();
  expect_true(wait_for_controller(controller,
                                  [](const auto& snapshot) {
                                    return snapshot.state == GpuPreviewControllerState::Error;
                                  }),
              "backend initialization error reaches snapshot");
  const auto snapshot = controller.snapshot();
  expect_true(snapshot.error.find("synthetic initialization failure") != std::string::npos,
              "error detail is retained");
  expect_true(!snapshot.ready, "failed controller is not ready");
  controller.stop();
  expect_true(controller.snapshot().error.empty(), "explicit stop clears sticky error detail");
}

void stop_wakes_a_blocked_decode_and_joins() {
  auto config = valid_config();
  config.start_playing = true;
  auto state = std::make_shared<FakeBackendState>();
  GpuPreviewController controller(config, std::make_unique<FakeBackend>(state));
  controller.start();
  expect_true(wait_for_backend(state, [&] { return state->blocked_reads != 0U; }),
              "backend decode is blocked before stop");
  const auto started = std::chrono::steady_clock::now();
  controller.stop();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  expect_true(elapsed < 1s, "stop interrupts blocked decode and joins promptly");
  expect_true(controller.snapshot().state == GpuPreviewControllerState::Stopped,
              "blocked decode stop reaches stopped state");
}

void state_names_are_stable() {
  expect_eq(gpu_preview_controller_state_name(GpuPreviewControllerState::Starting),
            std::string_view("starting"), "starting state name");
  expect_eq(gpu_preview_controller_state_name(GpuPreviewControllerState::EndOfStream),
            std::string_view("end_of_stream"), "EOS state name");
  expect_eq(gpu_preview_controller_state_name(GpuPreviewControllerState::Error),
            std::string_view("error"), "error state name");
}

void lifecycle_callback_can_stop_without_deadlock() {
  auto state = std::make_shared<FakeBackendState>();
  state->hold_initialize = true;
  GpuPreviewController* controller_ptr = nullptr;
  std::atomic<bool> stopped_from_callback{false};
  GpuPreviewController controller(
      valid_config(), std::make_unique<FakeBackend>(state),
      [&](const GpuPreviewControllerSnapshot& snapshot) {
        if (snapshot.state == GpuPreviewControllerState::Starting &&
            !stopped_from_callback.exchange(true, std::memory_order_relaxed)) {
          controller_ptr->stop();
        }
      });
  controller_ptr = &controller;
  controller.start();
  expect_true(stopped_from_callback.load(std::memory_order_relaxed),
              "starting callback invoked lifecycle control");
  expect_true(controller.snapshot().state == GpuPreviewControllerState::Stopped,
              "callback stop completes without lifecycle-lock deadlock");
}

} // namespace

int main() {
  validation_rejects_unsafe_boundaries();
  controls_and_timestamps_are_deterministic();
  controller_transitions_seek_and_preserve_presentation_time();
  initialization_errors_are_sticky_until_stop();
  stop_wakes_a_blocked_decode_and_joins();
  state_names_are_stable();
  lifecycle_callback_can_stop_without_deadlock();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
