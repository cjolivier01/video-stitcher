#include "reco/gui/toast.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace reco::gui {
namespace {

std::uint32_t bits_from_i32(std::int32_t value) {
  if (value >= 0) {
    return static_cast<std::uint32_t>(value);
  }
  return static_cast<std::uint32_t>(static_cast<std::int64_t>(value) + (1LL << 32));
}

std::int32_t i32_from_bits(std::uint32_t bits) {
  if (bits <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    return static_cast<std::int32_t>(bits);
  }
  return static_cast<std::int32_t>(static_cast<std::int64_t>(bits) - (1LL << 32));
}

} // namespace

std::string_view severity_name(Severity severity) {
  switch (severity) {
  case Severity::Info:
    return "info";
  case Severity::Warn:
    return "warn";
  case Severity::Error:
    return "error";
  }
  return "info";
}

std::chrono::milliseconds default_ttl(Severity severity) {
  using namespace std::chrono_literals;
  switch (severity) {
  case Severity::Info:
    return 4000ms;
  case Severity::Warn:
    return 7000ms;
  case Severity::Error:
    return 10000ms;
  }
  return 4000ms;
}

ToastManager::ToastManager() = default;

ToastManager::ToastManager(std::size_t max_visible)
    : max_visible_(std::max<std::size_t>(1, max_visible)) {}

ToastManager::ToastManager(std::size_t max_visible, std::int32_t next_id)
    : next_id_bits_(bits_from_i32(next_id == 0 ? 1 : next_id)),
      max_visible_(std::max<std::size_t>(1, max_visible)) {}

std::int32_t ToastManager::push(Severity severity, std::string title, std::string body) {
  return push_with_ttl(severity, std::move(title), std::move(body), default_ttl(severity));
}

std::int32_t ToastManager::push_with_ttl(Severity severity, std::string title, std::string body,
                                         std::chrono::milliseconds ttl) {
  const std::int32_t id = i32_from_bits(next_id_bits_);
  ++next_id_bits_;
  if (next_id_bits_ == 0) {
    next_id_bits_ = 1;
  }
  entries_.push_back(Toast{.id = id,
                           .severity = severity,
                           .title = std::move(title),
                           .body = std::move(body),
                           .expires_at = std::chrono::steady_clock::now() + ttl});
  if (entries_.size() > max_visible_) {
    const auto overflow = entries_.size() - max_visible_;
    entries_.erase(entries_.begin(), entries_.begin() + static_cast<std::ptrdiff_t>(overflow));
  }
  return id;
}

void ToastManager::dismiss(std::int32_t id) {
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                [&](const Toast& toast) { return toast.id == id; }),
                 entries_.end());
}

bool ToastManager::expire(std::chrono::steady_clock::time_point now) {
  const auto before = entries_.size();
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                [&](const Toast& toast) { return toast.expires_at <= now; }),
                 entries_.end());
  return entries_.size() != before;
}

std::optional<std::pair<std::string_view, std::string_view>> ToastManager::latest() const {
  if (entries_.empty()) {
    return std::nullopt;
  }
  const auto& toast = entries_.back();
  return std::pair<std::string_view, std::string_view>{severity_name(toast.severity), toast.title};
}

} // namespace reco::gui
