#include "reco/control/keyboard.hpp"

#include <utility>

namespace reco::control {

void KeyboardTransport::push(ControlIntent intent) { queue_.push_back(std::move(intent)); }

std::size_t KeyboardTransport::pending() const { return queue_.size(); }

std::size_t KeyboardTransport::poll(std::vector<ControlIntent>& out) {
  const auto start = out.size();
  while (!queue_.empty()) {
    out.push_back(std::move(queue_.front()));
    queue_.pop_front();
  }
  return out.size() - start;
}

} // namespace reco::control

