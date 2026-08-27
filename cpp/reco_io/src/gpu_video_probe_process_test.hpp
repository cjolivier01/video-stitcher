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

[[nodiscard]] GpuVideoProbe probe_gpu_video_with_pre_worker_report_delay_for_test(
    const GpuFileDecodeConfig& config, const std::filesystem::path& worker_path,
    std::uint64_t timeout_ns, std::uint64_t pre_worker_report_delay_ns);

#if !defined(_WIN32)
[[nodiscard]] bool guardian_watchdog_exit_is_fatal_for_test(bool memory_termination_sent,
                                                            int wait_result, int wait_error,
                                                            std::int64_t observed_pid,
                                                            std::int64_t watchdog_pid);
#endif

} // namespace reco::io::detail
