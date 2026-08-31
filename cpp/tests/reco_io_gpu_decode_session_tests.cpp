#include "reco/io/gpu_decode.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace reco::io;

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
void expect_invalid_argument(Function&& function, std::string_view message) {
  try {
    function();
    expect_true(false, message);
  } catch (const std::invalid_argument&) {
  } catch (...) {
    expect_true(false, message);
  }
}

template <typename Predicate> bool wait_until(Predicate&& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

struct OwnerTracker {
  void created() {
    const auto current = alive.fetch_add(1, std::memory_order_acq_rel) + 1;
    auto observed = peak.load(std::memory_order_acquire);
    while (observed < current &&
           !peak.compare_exchange_weak(observed, current, std::memory_order_acq_rel)) {
    }
  }

  void destroyed() { alive.fetch_sub(1, std::memory_order_acq_rel); }

  std::atomic<std::size_t> alive{0};
  std::atomic<std::size_t> peak{0};
};

class TrackedOwner {
public:
  explicit TrackedOwner(std::shared_ptr<OwnerTracker> tracker) : tracker_(std::move(tracker)) {
    tracker_->created();
  }

  ~TrackedOwner() { tracker_->destroyed(); }

private:
  std::shared_ptr<OwnerTracker> tracker_;
};

class DecodeBarrier {
public:
  void arrive_and_wait() {
    std::unique_lock lock(mutex_);
    ++arrivals_;
    condition_.notify_all();
    condition_.wait(lock, [&] { return arrivals_ >= 2; });
  }

  [[nodiscard]] std::size_t arrivals() const {
    std::lock_guard lock(mutex_);
    return arrivals_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t arrivals_ = 0;
};

struct FakeSourceState {
  [[nodiscard]] std::size_t reads() const {
    std::lock_guard lock(mutex);
    return read_calls;
  }

  [[nodiscard]] std::size_t stops() const {
    std::lock_guard lock(mutex);
    return stop_calls;
  }

  [[nodiscard]] std::size_t owners_at_first_stop() const {
    std::lock_guard lock(mutex);
    return live_owners_at_first_stop;
  }

  [[nodiscard]] std::shared_ptr<void> owner_for(std::uint64_t frame_index) const {
    std::lock_guard lock(mutex);
    for (const auto& [index, owner] : owners) {
      if (index == frame_index) {
        return owner.lock();
      }
    }
    return {};
  }

  mutable std::mutex mutex;
  std::condition_variable condition;
  std::size_t read_calls = 0;
  std::size_t stop_calls = 0;
  std::size_t live_owners_at_first_stop = 0;
  bool stopped = false;
  std::vector<std::pair<std::uint64_t, std::weak_ptr<void>>> owners;
};

struct FakeSourcePlan {
  std::vector<std::uint64_t> frame_indices;
  std::optional<std::size_t> failure_at_read;
  std::string failure_message = "fixture decode failure";
  bool block_at_end = false;
  bool gpu_resident = true;
  std::shared_ptr<DecodeBarrier> first_read_barrier;
  std::shared_ptr<OwnerTracker> owner_tracker = std::make_shared<OwnerTracker>();
};

class FakeGpuDecodeSource final : public GpuFileDecodeSource {
public:
  FakeGpuDecodeSource(std::string label, FakeSourcePlan plan,
                      std::shared_ptr<FakeSourceState> state)
      : config_{.path = std::move(label) + ".mp4", .container = GpuDecodeContainer::QuickTime},
        plan_(std::move(plan)), state_(std::move(state)) {}

  [[nodiscard]] const GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return "fake-nvmm-pipeline"; }
  [[nodiscard]] bool gpu_resident() const override { return plan_.gpu_resident; }

  [[nodiscard]] GpuDecodeReadResult read() override {
    if (!entered_barrier_ && plan_.first_read_barrier) {
      entered_barrier_ = true;
      plan_.first_read_barrier->arrive_and_wait();
    }

    std::unique_lock lock(state_->mutex);
    const auto read_number = state_->read_calls++;
    state_->condition.notify_all();
    if (state_->stopped) {
      return make_gpu_decode_eos();
    }
    if (plan_.failure_at_read == read_number) {
      throw GpuDecodeError(plan_.failure_message);
    }
    if (next_frame_ >= plan_.frame_indices.size()) {
      if (plan_.block_at_end) {
        state_->condition.wait(lock, [&] { return state_->stopped; });
      }
      return make_gpu_decode_eos();
    }

    const auto frame_index = plan_.frame_indices[next_frame_++];
    lock.unlock();
    auto owner = std::make_shared<TrackedOwner>(plan_.owner_tracker);
    lock.lock();
    state_->owners.emplace_back(frame_index, owner);
    lock.unlock();
    return make_gpu_decode_frame({.nvmm = {.dmabuf_fd = 12,
                                           .width = 4,
                                           .height = 4,
                                           .y_offset = 0,
                                           .y_pitch = 4,
                                           .uv_offset = 16,
                                           .uv_pitch = 4,
                                           .total_size = 24,
                                           .surface_ptr = reinterpret_cast<void*>(0x1000),
                                           .memory_type = NvmmMemoryType::SurfaceArray,
                                           .gpu_id = 0,
                                           .y_size = 16,
                                           .uv_size = 8},
                                  .visible_width = 4,
                                  .visible_height = 4,
                                  .owner = std::move(owner),
                                  .frame_index = frame_index,
                                  .pts_ns = frame_index * 1'000'000ULL,
                                  .duration_ns = 1'000'000ULL});
  }

  void request_stop() noexcept override {
    std::lock_guard lock(state_->mutex);
    if (state_->stop_calls == 0) {
      state_->live_owners_at_first_stop = static_cast<std::size_t>(
          std::count_if(state_->owners.begin(), state_->owners.end(),
                        [](const auto& entry) { return !entry.second.expired(); }));
    }
    ++state_->stop_calls;
    state_->stopped = true;
    state_->condition.notify_all();
  }

private:
  GpuFileDecodeConfig config_;
  FakeSourcePlan plan_;
  std::shared_ptr<FakeSourceState> state_;
  std::size_t next_frame_ = 0;
  bool entered_barrier_ = false;
};

std::unique_ptr<GpuFileDecodeSource> make_source(std::string label, FakeSourcePlan plan,
                                                 const std::shared_ptr<FakeSourceState>& state) {
  return std::make_unique<FakeGpuDecodeSource>(std::move(label), std::move(plan), state);
}

void positive_sync_aligns_indices_and_preserves_owners() {
  const auto barrier = std::make_shared<DecodeBarrier>();
  const auto left_state = std::make_shared<FakeSourceState>();
  const auto right_state = std::make_shared<FakeSourceState>();
  FakeSourcePlan left_plan{.frame_indices = {0, 1, 2, 3, 4, 5}, .first_read_barrier = barrier};
  FakeSourcePlan right_plan{.frame_indices = {0, 1, 2, 3, 4, 5}, .first_read_barrier = barrier};
  GpuStereoDecodeSession session(make_source("left", std::move(left_plan), left_state),
                                 make_source("right", std::move(right_plan), right_state),
                                 {.sync_offset = 2, .queue_capacity = 2});

  for (std::uint64_t index = 0; index < 4; ++index) {
    auto result = session.read();
    expect_true(result.status == GpuStereoDecodeStatus::FramePair,
                "positive sync returns a frame pair");
    expect_true(result.frames.has_value(), "positive sync pair has payload");
    if (!result.frames.has_value()) {
      continue;
    }
    expect_eq(result.frames->left.frame_index, index, "positive sync left frame index");
    expect_eq(result.frames->right.frame_index, index + 2, "positive sync right frame index");
    expect_true(left_state->owner_for(index).get() == result.frames->left.owner.get(),
                "left decoder owner is returned without a copy");
    expect_true(right_state->owner_for(index + 2).get() == result.frames->right.owner.get(),
                "right decoder owner is returned without a copy");
  }

  const auto eos = session.read();
  expect_true(eos.status == GpuStereoDecodeStatus::EndOfStream,
              "positive sync reports distinct EOS");
  expect_true(!eos.frames.has_value(), "stereo EOS has no frame payload");
  expect_eq(barrier->arrivals(), 2U, "both persistent decode workers ran concurrently");
}

void negative_sync_aligns_indices() {
  const auto left_state = std::make_shared<FakeSourceState>();
  const auto right_state = std::make_shared<FakeSourceState>();
  GpuStereoDecodeSession session(
      make_source("left", {.frame_indices = {10, 11, 12, 13, 14}}, left_state),
      make_source("right", {.frame_indices = {10, 11, 12, 13, 14}}, right_state),
      {.sync_offset = -2, .queue_capacity = 1});

  for (std::uint64_t index = 10; index < 13; ++index) {
    const auto result = session.read();
    expect_true(result.status == GpuStereoDecodeStatus::FramePair,
                "negative sync returns a frame pair");
    expect_true(result.frames.has_value(), "negative sync pair has payload");
    if (result.frames.has_value()) {
      expect_eq(result.frames->left.frame_index, index + 2, "negative sync left frame index");
      expect_eq(result.frames->right.frame_index, index, "negative sync right frame index");
    }
  }
  expect_true(session.read().status == GpuStereoDecodeStatus::EndOfStream,
              "negative sync reports EOS after matchable pairs");
}

void retained_owner_queues_are_bounded() {
  constexpr std::uint32_t capacity = 3;
  const auto left_tracker = std::make_shared<OwnerTracker>();
  const auto right_tracker = std::make_shared<OwnerTracker>();
  const auto left_state = std::make_shared<FakeSourceState>();
  const auto right_state = std::make_shared<FakeSourceState>();
  std::vector<std::uint64_t> indices;
  for (std::uint64_t index = 0; index < 32; ++index) {
    indices.push_back(index);
  }
  GpuStereoDecodeSession session(
      make_source("left",
                  {.frame_indices = indices, .block_at_end = true, .owner_tracker = left_tracker},
                  left_state),
      make_source("right",
                  {.frame_indices = indices, .block_at_end = true, .owner_tracker = right_tracker},
                  right_state),
      {.queue_capacity = capacity});

  expect_true(wait_until([&] {
                return left_tracker->alive.load(std::memory_order_acquire) == capacity &&
                       right_tracker->alive.load(std::memory_order_acquire) == capacity;
              }),
              "both owner queues fill to their configured bound");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  expect_eq(left_tracker->peak.load(std::memory_order_acquire), capacity,
            "left queue never retains beyond capacity");
  expect_eq(right_tracker->peak.load(std::memory_order_acquire), capacity,
            "right queue never retains beyond capacity");

  session.request_stop();
  expect_eq(left_state->owners_at_first_stop(), 0U,
            "left queued owners are released before source interruption");
  expect_eq(right_state->owners_at_first_stop(), 0U,
            "right queued owners are released before source interruption");
  expect_true(wait_until([&] {
                return left_tracker->alive.load(std::memory_order_acquire) == 0 &&
                       right_tracker->alive.load(std::memory_order_acquire) == 0;
              }),
              "explicit stop releases all queued decoder owners");
}

void side_specific_failures_stop_the_peer() {
  for (const auto failing_side : {GpuDecodeSide::Left, GpuDecodeSide::Right}) {
    const auto left_state = std::make_shared<FakeSourceState>();
    const auto right_state = std::make_shared<FakeSourceState>();
    FakeSourcePlan failing{.failure_at_read = 0, .failure_message = "deterministic side failure"};
    FakeSourcePlan blocked{.block_at_end = true};
    GpuStereoDecodeSession session(
        make_source("left", failing_side == GpuDecodeSide::Left ? failing : blocked, left_state),
        make_source("right", failing_side == GpuDecodeSide::Right ? failing : blocked,
                    right_state));

    try {
      (void)session.read();
      expect_true(false, "side failure is propagated");
    } catch (const GpuStereoDecodeError& error) {
      expect_true(error.side() == failing_side, "stereo failure identifies the failed side");
      expect_true(std::string(error.what()).find("deterministic side failure") != std::string::npos,
                  "stereo failure preserves the source error");
    } catch (...) {
      expect_true(false, "side failure uses GpuStereoDecodeError");
    }
    const auto peer_state = failing_side == GpuDecodeSide::Left ? right_state : left_state;
    expect_true(wait_until([&] { return peer_state->stops() == 1; }),
                "source failure stops and wakes the peer");
  }
}

void explicit_stop_wakes_blocked_reads_idempotently() {
  const auto left_state = std::make_shared<FakeSourceState>();
  const auto right_state = std::make_shared<FakeSourceState>();
  GpuStereoDecodeSession session(make_source("left", {.block_at_end = true}, left_state),
                                 make_source("right", {.block_at_end = true}, right_state));
  expect_true(wait_until([&] { return left_state->reads() == 1 && right_state->reads() == 1; }),
              "both sources block in persistent reads");

  auto pending = std::async(std::launch::async, [&] { return session.read(); });
  session.request_stop();
  session.request_stop();
  expect_true(pending.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
              "explicit stop wakes a blocked session read");
  const auto stopped = pending.get();
  expect_true(stopped.status == GpuStereoDecodeStatus::Stopped,
              "explicit stop is distinct from EOS");
  expect_true(!stopped.frames.has_value(), "stopped result has no frame payload");
  expect_eq(left_state->stops(), 1U, "left source stop is idempotent");
  expect_eq(right_state->stops(), 1U, "right source stop is idempotent");
}

void eos_stops_a_blocked_peer() {
  const auto left_state = std::make_shared<FakeSourceState>();
  const auto right_state = std::make_shared<FakeSourceState>();
  GpuStereoDecodeSession session(make_source("left", {}, left_state),
                                 make_source("right", {.block_at_end = true}, right_state));
  const auto eos = session.read();
  expect_true(eos.status == GpuStereoDecodeStatus::EndOfStream,
              "natural source exhaustion reports EOS");
  expect_true(wait_until([&] { return right_state->stops() == 1; }),
              "EOS stops and wakes the peer source");
}

void invalid_configuration_and_cpu_sources_are_rejected() {
  expect_true(!validate_gpu_stereo_decode_config({.sync_offset = -100'000, .queue_capacity = 1})
                   .has_value(),
              "minimum sync and queue bounds are accepted");
  expect_true(!validate_gpu_stereo_decode_config({.sync_offset = 100'000, .queue_capacity = 16})
                   .has_value(),
              "maximum sync and queue bounds are accepted");

  auto construct = [](GpuStereoDecodeConfig config) {
    const auto left_state = std::make_shared<FakeSourceState>();
    const auto right_state = std::make_shared<FakeSourceState>();
    return std::make_unique<GpuStereoDecodeSession>(make_source("left", {}, left_state),
                                                    make_source("right", {}, right_state), config);
  };
  expect_invalid_argument([&] { (void)construct({.sync_offset = -100'001}); },
                          "sync offset below the lower bound is rejected");
  expect_invalid_argument([&] { (void)construct({.sync_offset = 100'001}); },
                          "sync offset above the upper bound is rejected");
  expect_invalid_argument([&] { (void)construct({.queue_capacity = 0}); },
                          "zero queue capacity is rejected");
  expect_invalid_argument([&] { (void)construct({.queue_capacity = 17}); },
                          "queue capacity above sixteen is rejected");

  const auto cpu_state = std::make_shared<FakeSourceState>();
  const auto gpu_state = std::make_shared<FakeSourceState>();
  expect_invalid_argument(
      [&] {
        (void)GpuStereoDecodeSession(make_source("cpu", {.gpu_resident = false}, cpu_state),
                                     make_source("gpu", {.gpu_resident = true}, gpu_state));
      },
      "CPU-resident left source is rejected before decoding starts");
  expect_eq(cpu_state->reads(), 0U, "rejected CPU source is never read");
  expect_eq(gpu_state->reads(), 0U, "peer is never started after CPU-source rejection");

  expect_invalid_argument(
      [&] { (void)GpuStereoDecodeSession(nullptr, make_source("right", {}, gpu_state)); },
      "missing stereo source is rejected");
}

} // namespace

int main() {
  positive_sync_aligns_indices_and_preserves_owners();
  negative_sync_aligns_indices();
  retained_owner_queues_are_bounded();
  side_specific_failures_stop_the_peer();
  explicit_stop_wakes_blocked_reads_idempotently();
  eos_stops_a_blocked_peer();
  invalid_configuration_and_cpu_sources_are_rejected();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
