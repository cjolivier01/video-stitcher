#pragma once

#include "reco/core/viewport_position.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <vector>

namespace reco::core {

struct ReplayFrame {
  std::vector<std::uint8_t> rgba;
  std::chrono::nanoseconds captured_at{};
  ViewportPosition pose;
};

class ReplayBuffer {
public:
  explicit ReplayBuffer(std::chrono::nanoseconds max_duration);

  void push(ReplayFrame frame);
  [[nodiscard]] std::size_t len() const;
  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::chrono::nanoseconds max_duration() const;
  [[nodiscard]] const std::deque<ReplayFrame>& frames() const;
  [[nodiscard]] const ReplayFrame* latest() const;
  [[nodiscard]] const ReplayFrame* oldest() const;
  [[nodiscard]] std::chrono::nanoseconds buffered_duration() const;
  void clear();
  [[nodiscard]] std::vector<ReplayFrame> snapshot() const;
  [[nodiscard]] std::vector<ReplayFrame> take();

private:
  std::deque<ReplayFrame> frames_;
  std::chrono::nanoseconds max_duration_;
};

} // namespace reco::core
