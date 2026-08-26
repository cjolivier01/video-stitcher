#pragma once

#include "reco/io/gpu_video_probe.hpp"

#include <cstdint>

namespace reco::io::detail {

[[nodiscard]] GpuVideoProbe probe_gpu_video_in_process(const GpuFileDecodeConfig& config,
                                                       std::uint64_t timeout_ns);

} // namespace reco::io::detail
