#pragma once

#include "reco/calibrate/pipeline.hpp"

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>

namespace reco::calibrate::detail {

inline constexpr std::string_view kCalibrationWorkerIpcArgument = "--reco-calibration-worker-ipc";
inline constexpr std::string_view kCalibrationGuardianArgument = "--reco-calibration-guardian-ipc";

/// File-backed implementation used only inside the isolated worker and tests.
[[nodiscard]] CalibrationResult
run_gpu_calibration_in_process(const GpuCalibrationRequest& request,
                               const CalibrationBackendStatus& backends);

/// Executes a file-backed calibration in a bounded subprocess.
[[nodiscard]] CalibrationResult
run_gpu_calibration_supervised(const GpuCalibrationRequest& request);

/// Connects to an abstract Unix socket after the executable and its loader have initialized.
[[nodiscard]] int connect_calibration_ipc(std::string_view address,
                                          std::uint64_t deadline_nanoseconds);
/// Receives and validates the cross-process admission-lock descriptor.
[[nodiscard]] int receive_calibration_admission_fd(int descriptor,
                                                   std::uint64_t deadline_nanoseconds);
/// Receives and validates the pinned calibration executable descriptor.
[[nodiscard]] int receive_calibration_executable_fd(int descriptor,
                                                    std::uint64_t deadline_nanoseconds);
/// Receives and validates a retained left calibration-media descriptor.
[[nodiscard]] int receive_calibration_left_input_fd(int descriptor,
                                                    std::uint64_t deadline_nanoseconds);
/// Receives and validates a retained right calibration-media descriptor.
[[nodiscard]] int receive_calibration_right_input_fd(int descriptor,
                                                     std::uint64_t deadline_nanoseconds);
/// Receives and validates the delegated cgroup-v2 directory descriptor.
[[nodiscard]] int receive_calibration_cgroup_fd(int descriptor, std::uint64_t deadline_nanoseconds);
/// Runs the Linux guardian that owns worker launch and cleanup.
int run_calibration_guardian_fd(int descriptor, const char* executable,
                                std::uint64_t deadline_nanoseconds);
/// Installs the post-exec worker syscall boundary before any IPC or parser work.
[[nodiscard]] bool install_calibration_worker_sandbox() noexcept;

/// Handles one framed worker request. Exposed only to the worker and protocol tests.
int run_calibration_worker(std::istream& input, std::ostream& output);
/// Handles one framed worker request on a post-exec authenticated channel.
int run_calibration_worker_fd(int descriptor, std::uint64_t deadline_nanoseconds);
/// Reads exactly one byte range using the production nonblocking worker transport.
void read_calibration_worker_bytes_fd(int descriptor, std::span<char> destination,
                                      std::uint64_t deadline_nanoseconds);
/// Writes one byte range using the production nonblocking worker transport.
void write_calibration_worker_bytes_fd(int descriptor, std::string_view bytes,
                                       std::uint64_t deadline_nanoseconds);

} // namespace reco::calibrate::detail
