#include "reco/io/gpu_decode.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace reco::io {
namespace {

std::string_view decode_side_name(GpuDecodeSide side) {
  return side == GpuDecodeSide::Left ? "left" : "right";
}

enum class FrontRelation {
  Pair,
  DropLeft,
  DropRight,
};

FrontRelation compare_frame_indices(std::uint64_t left, std::uint64_t right,
                                    std::int64_t sync_offset) {
  if (sync_offset >= 0) {
    const auto offset = static_cast<std::uint64_t>(sync_offset);
    if (left > std::numeric_limits<std::uint64_t>::max() - offset) {
      return FrontRelation::DropRight;
    }
    const auto expected_right = left + offset;
    if (expected_right < right) {
      return FrontRelation::DropLeft;
    }
    if (expected_right > right) {
      return FrontRelation::DropRight;
    }
    return FrontRelation::Pair;
  }

  const auto offset = static_cast<std::uint64_t>(-sync_offset);
  if (right > std::numeric_limits<std::uint64_t>::max() - offset) {
    return FrontRelation::DropLeft;
  }
  const auto expected_left = right + offset;
  if (left < expected_left) {
    return FrontRelation::DropLeft;
  }
  if (left > expected_left) {
    return FrontRelation::DropRight;
  }
  return FrontRelation::Pair;
}

} // namespace

GpuStereoDecodeError::GpuStereoDecodeError(GpuDecodeSide side, std::string message)
    : GpuDecodeError(std::string(decode_side_name(side)) + " GPU decode failed: " + message),
      side_(side) {}

class GpuStereoDecodeSession::Impl {
public:
  Impl(std::unique_ptr<GpuFileDecodeSource> left, std::unique_ptr<GpuFileDecodeSource> right,
       GpuStereoDecodeConfig config)
      : left_source_(std::move(left)), right_source_(std::move(right)), config_(config) {
    if (const auto error = validate_gpu_stereo_decode_config(config_); error.has_value()) {
      throw std::invalid_argument(*error);
    }
    if (!left_source_ || !right_source_) {
      throw std::invalid_argument("GPU stereo decode requires both sources");
    }
    if (!left_source_->gpu_resident()) {
      throw std::invalid_argument("left GPU stereo decode source is not GPU resident");
    }
    if (!right_source_->gpu_resident()) {
      throw std::invalid_argument("right GPU stereo decode source is not GPU resident");
    }
    left_.capacity = config_.queue_capacity;
    right_.capacity = config_.queue_capacity;
    start_threads();
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  ~Impl() {
    signal_stop();
    join_threads();
  }

  [[nodiscard]] GpuStereoDecodeReadResult read() {
    std::lock_guard read_lock(read_mutex_);
    for (;;) {
      std::unique_lock lock(mutex_);
      if (failure_.has_value()) {
        const auto failure = *failure_;
        lock.unlock();
        throw GpuStereoDecodeError(failure.side, failure.message);
      }
      if (terminal_eos_) {
        return {.status = GpuStereoDecodeStatus::EndOfStream, .frames = std::nullopt};
      }
      if (external_stop_) {
        return {.status = GpuStereoDecodeStatus::Stopped, .frames = std::nullopt};
      }

      if (!left_.frames.empty() && !right_.frames.empty()) {
        switch (compare_frame_indices(left_.frames.front().frame_index,
                                      right_.frames.front().frame_index, config_.sync_offset)) {
        case FrontRelation::Pair: {
          GpuDecodedFramePair pair{.left = std::move(left_.frames.front()),
                                   .right = std::move(right_.frames.front())};
          left_.frames.pop_front();
          right_.frames.pop_front();
          state_changed_.notify_all();
          return {.status = GpuStereoDecodeStatus::FramePair, .frames = std::move(pair)};
        }
        case FrontRelation::DropLeft: {
          auto discarded = std::move(left_.frames.front());
          left_.frames.pop_front();
          state_changed_.notify_all();
          lock.unlock();
          continue;
        }
        case FrontRelation::DropRight: {
          auto discarded = std::move(right_.frames.front());
          right_.frames.pop_front();
          state_changed_.notify_all();
          lock.unlock();
          continue;
        }
        }
      }

      if ((left_.eos && left_.frames.empty()) || (right_.eos && right_.frames.empty())) {
        terminal_eos_ = true;
        const bool first_stop = !workers_stopped_.exchange(true, std::memory_order_acq_rel);
        std::deque<GpuDecodedFrame> discarded_left;
        std::deque<GpuDecodedFrame> discarded_right;
        discarded_left.swap(left_.frames);
        discarded_right.swap(right_.frames);
        state_changed_.notify_all();
        lock.unlock();
        finish_stop(first_stop);
        return {.status = GpuStereoDecodeStatus::EndOfStream, .frames = std::nullopt};
      }

      state_changed_.wait(lock);
    }
  }

  void request_stop() noexcept {
    std::deque<GpuDecodedFrame> discarded_left;
    std::deque<GpuDecodedFrame> discarded_right;
    bool first_stop = false;
    {
      std::lock_guard lock(mutex_);
      if (failure_.has_value() || terminal_eos_ || external_stop_) {
        return;
      }
      external_stop_ = true;
      first_stop = !workers_stopped_.exchange(true, std::memory_order_acq_rel);
      discarded_left.swap(left_.frames);
      discarded_right.swap(right_.frames);
    }
    finish_stop(first_stop);
  }

private:
  struct SideState {
    std::deque<GpuDecodedFrame> frames;
    std::uint32_t capacity = 0;
    bool eos = false;
  };

  struct Failure {
    GpuDecodeSide side;
    std::string message;
  };

  void start_threads() {
    try {
      left_thread_ = std::thread([this] { decode(GpuDecodeSide::Left); });
      right_thread_ = std::thread([this] { decode(GpuDecodeSide::Right); });
    } catch (const std::exception& error) {
      const std::string message = error.what();
      signal_stop();
      join_threads();
      throw GpuDecodeError("failed to start GPU stereo decode threads: " + message);
    } catch (...) {
      signal_stop();
      join_threads();
      throw GpuDecodeError("failed to start GPU stereo decode threads");
    }
  }

  void join_threads() noexcept {
    if (left_thread_.joinable()) {
      left_thread_.join();
    }
    if (right_thread_.joinable()) {
      right_thread_.join();
    }
  }

  void signal_stop() noexcept {
    const bool first = !workers_stopped_.exchange(true, std::memory_order_acq_rel);
    finish_stop(first);
  }

  void finish_stop(bool first) noexcept {
    state_changed_.notify_all();
    if (first) {
      left_source_->request_stop();
      right_source_->request_stop();
    }
    state_changed_.notify_all();
  }

  void record_failure(GpuDecodeSide side, std::string message) noexcept {
    std::deque<GpuDecodedFrame> discarded_left;
    std::deque<GpuDecodedFrame> discarded_right;
    {
      std::lock_guard lock(mutex_);
      if (workers_stopped_.load(std::memory_order_acquire) || external_stop_) {
        return;
      }
      if (!failure_.has_value()) {
        failure_ = Failure{.side = side, .message = std::move(message)};
      }
      const bool first_stop = !workers_stopped_.exchange(true, std::memory_order_acq_rel);
      discarded_left.swap(left_.frames);
      discarded_right.swap(right_.frames);
      state_changed_.notify_all();
      if (!first_stop) {
        return;
      }
    }
    finish_stop(true);
  }

  void decode(GpuDecodeSide side) noexcept {
    auto& state = side == GpuDecodeSide::Left ? left_ : right_;
    auto& source = side == GpuDecodeSide::Left ? left_source_ : right_source_;
    std::optional<std::uint64_t> previous_index;
    try {
      for (;;) {
        {
          std::unique_lock lock(mutex_);
          state_changed_.wait(lock, [&] {
            return workers_stopped_.load(std::memory_order_acquire) ||
                   state.frames.size() < state.capacity;
          });
          if (workers_stopped_.load(std::memory_order_acquire)) {
            return;
          }
        }

        auto result = source->read();
        if (workers_stopped_.load(std::memory_order_acquire)) {
          return;
        }
        switch (result.status) {
        case GpuDecodeFrameStatus::EndOfStream: {
          if (result.frame.has_value()) {
            throw GpuDecodeError("end-of-stream result contains a frame");
          }
          std::lock_guard lock(mutex_);
          state.eos = true;
          state_changed_.notify_all();
          return;
        }
        case GpuDecodeFrameStatus::Frame:
          if (!result.frame.has_value()) {
            throw GpuDecodeError("frame result has no frame payload");
          }
          break;
        default:
          throw GpuDecodeError("source returned an unknown frame status");
        }

        if (const auto error = validate_gpu_decoded_frame(*result.frame); error.has_value()) {
          throw GpuDecodeError(*error);
        }
        if (previous_index.has_value() && result.frame->frame_index <= *previous_index) {
          throw GpuDecodeError("frame indices are not strictly increasing");
        }
        previous_index = result.frame->frame_index;

        std::lock_guard lock(mutex_);
        if (workers_stopped_.load(std::memory_order_acquire)) {
          return;
        }
        state.frames.push_back(std::move(*result.frame));
        state_changed_.notify_all();
      }
    } catch (const std::exception& error) {
      record_failure(side, error.what());
    } catch (...) {
      record_failure(side, "unknown source failure");
    }
  }

  std::unique_ptr<GpuFileDecodeSource> left_source_;
  std::unique_ptr<GpuFileDecodeSource> right_source_;
  GpuStereoDecodeConfig config_;
  std::mutex mutex_;
  std::mutex read_mutex_;
  std::condition_variable state_changed_;
  SideState left_;
  SideState right_;
  std::optional<Failure> failure_;
  std::atomic<bool> workers_stopped_{false};
  bool external_stop_ = false;
  bool terminal_eos_ = false;
  std::thread left_thread_;
  std::thread right_thread_;
};

GpuStereoDecodeSession::GpuStereoDecodeSession(std::unique_ptr<GpuFileDecodeSource> left,
                                               std::unique_ptr<GpuFileDecodeSource> right,
                                               GpuStereoDecodeConfig config)
    : impl_(std::make_unique<Impl>(std::move(left), std::move(right), config)) {}

GpuStereoDecodeSession::~GpuStereoDecodeSession() = default;

GpuStereoDecodeReadResult GpuStereoDecodeSession::read() { return impl_->read(); }

void GpuStereoDecodeSession::request_stop() noexcept { impl_->request_stop(); }

} // namespace reco::io
