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

} // namespace reco::io::detail
