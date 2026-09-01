#pragma once

#include "reco/io/gpu_video_probe.hpp"

#include <cstdint>
#include <filesystem>

namespace reco::io::detail {

[[nodiscard]] GpuVideoProbe probe_gpu_video_with_supervisor_start_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t supervisor_start_delay_ns);

[[nodiscard]] GpuVideoProbe probe_gpu_video_with_pre_worker_spawn_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_worker_spawn_delay_ns);

[[nodiscard]] GpuVideoProbe probe_gpu_video_with_pre_supervisor_exec_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_supervisor_exec_delay_ns,
    const std::filesystem::path& marker_path);

[[nodiscard]] GpuVideoProbe probe_gpu_video_with_pre_worker_report_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_worker_report_delay_ns);

[[nodiscard]] std::uint64_t maximum_probe_worker_address_space_bytes_for_test();
[[nodiscard]] std::uint64_t maximum_probe_executable_snapshot_bytes_for_test();
[[nodiscard]] std::uint64_t maximum_per_probe_memory_reservation_bytes_for_test();
[[nodiscard]] std::uint64_t maximum_aggregate_probe_memory_bytes_for_test();
[[nodiscard]] std::uint64_t reserved_probe_worker_address_space_bytes_for_test();
[[nodiscard]] std::uint64_t reserved_probe_memory_bytes_for_test();

[[nodiscard]] GpuVideoProbe probe_gpu_video_exhaustive_for_test(const GpuFileDecodeConfig& config,
                                                                std::uint64_t timeout_ns);
[[nodiscard]] GpuVideoProbe
probe_gpu_video_bounded_in_process_for_test(const GpuFileDecodeConfig& config,
                                            std::uint64_t timeout_ns);
void hold_probe_worker_memory_reservation_for_test(std::uint64_t hold_ns);

#if !defined(_WIN32)
void hold_probe_worker_admission_lock_for_test(std::uint64_t hold_ns, int ready_descriptor);

[[nodiscard]] GpuVideoProbe probe_gpu_video_with_pre_guardian_exec_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_guardian_exec_delay_ns,
    const std::filesystem::path& marker_path);

[[nodiscard]] bool guardian_watchdog_exit_is_fatal_for_test(bool memory_termination_sent,
                                                            int wait_result, int wait_error,
                                                            std::int64_t observed_pid,
                                                            std::int64_t watchdog_pid);
#endif

#if defined(__linux__) && defined(__x86_64__)
[[nodiscard]] bool probe_worker_rejects_x32_syscalls_for_test();
#endif

#if defined(__linux__)
void hold_linux_probe_executable_snapshot_for_test(const std::filesystem::path& worker_path,
                                                   int ready_descriptor, int release_descriptor);
#endif

} // namespace reco::io::detail
