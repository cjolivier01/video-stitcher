#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace reco::gui {

enum class Severity {
  Info,
  Warn,
  Error,
};

[[nodiscard]] std::string_view severity_name(Severity severity);
[[nodiscard]] std::chrono::milliseconds default_ttl(Severity severity);

struct Toast {
  std::int32_t id = 0;
  Severity severity = Severity::Info;
  std::string title;
  std::string body;
  std::chrono::steady_clock::time_point expires_at{};
};

class ToastManager {
public:
  ToastManager();
  explicit ToastManager(std::size_t max_visible);
  ToastManager(std::size_t max_visible, std::int32_t next_id);

  [[nodiscard]] std::int32_t push(Severity severity, std::string title, std::string body);
  [[nodiscard]] std::int32_t push_with_ttl(Severity severity, std::string title, std::string body,
                                           std::chrono::milliseconds ttl);
  void dismiss(std::int32_t id);
  [[nodiscard]] bool expire(std::chrono::steady_clock::time_point now);
  [[nodiscard]] std::optional<std::pair<std::string_view, std::string_view>> latest() const;
  [[nodiscard]] bool empty() const { return entries_.empty(); }
  [[nodiscard]] std::size_t size() const { return entries_.size(); }
  [[nodiscard]] const std::vector<Toast>& entries() const { return entries_; }

private:
  std::vector<Toast> entries_;
  std::uint32_t next_id_bits_ = 1;
  std::size_t max_visible_ = 4;
};

} // namespace reco::gui
