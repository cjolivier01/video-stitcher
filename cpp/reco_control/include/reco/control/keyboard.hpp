#pragma once

#include <cstddef>
#include <deque>
#include <string_view>
#include <vector>

#include "reco/control/intents.hpp"

namespace reco::control {

class KeyboardTransport {
public:
  [[nodiscard]] static constexpr std::string_view name() { return "keyboard"; }

  void push(ControlIntent intent);
  [[nodiscard]] std::size_t pending() const;
  std::size_t poll(std::vector<ControlIntent>& out);

private:
  std::deque<ControlIntent> queue_;
};

} // namespace reco::control

