#include "reco/cli/interrupt.hpp"

#include <csignal>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <signal.h>
#endif

namespace reco::cli::detail {
namespace {

#if defined(_WIN32)
volatile LONG interrupt_requested = 0;

BOOL WINAPI handle_console_interrupt(DWORD event) {
  switch (event) {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
  case CTRL_CLOSE_EVENT:
  case CTRL_LOGOFF_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    (void)InterlockedExchange(&interrupt_requested, 1);
    return TRUE;
  default:
    return FALSE;
  }
}

void handle_crt_interrupt(int) { (void)InterlockedExchange(&interrupt_requested, 1); }
#else
volatile std::sig_atomic_t interrupt_requested = 0;

extern "C" void handle_posix_interrupt(int) { interrupt_requested = 1; }
#endif

} // namespace

struct InterruptMonitor::Impl {
#if defined(_WIN32)
  using CrtSignalHandler = void (*)(int);
  CrtSignalHandler previous_interrupt = SIG_DFL;
#else
  struct sigaction previous_interrupt{};
  struct sigaction previous_termination{};
#endif
};

InterruptMonitor::InterruptMonitor() : impl_(std::make_unique<Impl>()) {
#if defined(_WIN32)
  (void)InterlockedExchange(&interrupt_requested, 0);
  if (SetConsoleCtrlHandler(handle_console_interrupt, TRUE) == 0) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                            "cannot install the Windows console cancellation handler");
  }
  impl_->previous_interrupt = std::signal(SIGINT, handle_crt_interrupt);
  if (impl_->previous_interrupt == SIG_ERR) {
    (void)SetConsoleCtrlHandler(handle_console_interrupt, FALSE);
    throw std::runtime_error("cannot install the Windows CRT interrupt handler");
  }
#else
  interrupt_requested = 0;
  struct sigaction action{};
  action.sa_handler = handle_posix_interrupt;
  if (::sigemptyset(&action.sa_mask) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "cannot initialize the cancellation signal mask");
  }
  if (::sigaction(SIGINT, &action, &impl_->previous_interrupt) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "cannot install the interrupt signal handler");
  }
  if (::sigaction(SIGTERM, &action, &impl_->previous_termination) != 0) {
    const int error = errno;
    (void)::sigaction(SIGINT, &impl_->previous_interrupt, nullptr);
    throw std::system_error(error, std::generic_category(),
                            "cannot install the termination signal handler");
  }
#endif
}

InterruptMonitor::~InterruptMonitor() {
#if defined(_WIN32)
  (void)std::signal(SIGINT, impl_->previous_interrupt);
  (void)SetConsoleCtrlHandler(handle_console_interrupt, FALSE);
#else
  (void)::sigaction(SIGTERM, &impl_->previous_termination, nullptr);
  (void)::sigaction(SIGINT, &impl_->previous_interrupt, nullptr);
#endif
}

bool InterruptMonitor::requested() const noexcept {
#if defined(_WIN32)
  return InterlockedCompareExchange(&interrupt_requested, 0, 0) != 0;
#else
  return interrupt_requested != 0;
#endif
}

} // namespace reco::cli::detail
