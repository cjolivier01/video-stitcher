#include "reco/core/replay_buffer.hpp"

#include <utility>

namespace reco::core {

ReplayBuffer::ReplayBuffer(std::chrono::nanoseconds max_duration) : max_duration_(max_duration) {}

void ReplayBuffer::push(ReplayFrame frame) {
  frames_.push_back(std::move(frame));
  const auto newest = frames_.back().captured_at;
  const auto cutoff = newest > max_duration_ ? newest - max_duration_ : std::chrono::nanoseconds{};
  while (!frames_.empty() && frames_.front().captured_at < cutoff) {
    frames_.pop_front();
  }
}

std::size_t ReplayBuffer::len() const { return frames_.size(); }

bool ReplayBuffer::empty() const { return frames_.empty(); }

std::chrono::nanoseconds ReplayBuffer::max_duration() const { return max_duration_; }

const std::deque<ReplayFrame>& ReplayBuffer::frames() const { return frames_; }

const ReplayFrame* ReplayBuffer::latest() const {
  return frames_.empty() ? nullptr : &frames_.back();
}

const ReplayFrame* ReplayBuffer::oldest() const {
  return frames_.empty() ? nullptr : &frames_.front();
}

std::chrono::nanoseconds ReplayBuffer::buffered_duration() const {
  if (frames_.empty()) {
    return {};
  }
  if (frames_.back().captured_at <= frames_.front().captured_at) {
    return {};
  }
  return frames_.back().captured_at - frames_.front().captured_at;
}

void ReplayBuffer::clear() { frames_.clear(); }

std::vector<ReplayFrame> ReplayBuffer::snapshot() const { return {frames_.begin(), frames_.end()}; }

std::vector<ReplayFrame> ReplayBuffer::take() {
  std::vector<ReplayFrame> result;
  result.reserve(frames_.size());
  while (!frames_.empty()) {
    result.push_back(std::move(frames_.front()));
    frames_.pop_front();
  }
  return result;
}

} // namespace reco::core
