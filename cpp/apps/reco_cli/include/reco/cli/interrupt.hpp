#pragma once

#include <memory>

namespace reco::cli::detail {

/// Process-scoped Ctrl+C/termination monitor whose handlers only set signal-safe state.
class InterruptMonitor final {
public:
  /// Installs the native console/signal handlers and clears any prior request.
  InterruptMonitor();
  InterruptMonitor(const InterruptMonitor&) = delete;
  InterruptMonitor& operator=(const InterruptMonitor&) = delete;
  ~InterruptMonitor();

  /// Reports whether an installed native handler observed a cancellation request.
  [[nodiscard]] bool requested() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace reco::cli::detail
