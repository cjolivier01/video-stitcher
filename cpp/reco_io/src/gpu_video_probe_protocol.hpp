#pragma once

#include "reco/io/gpu_video_probe.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace reco::io::detail {

constexpr std::size_t kMaximumProbeIpcBytes = 256U * 1024U;

struct ProbeWorkerRequest {
  GpuFileDecodeConfig config;
  std::uint64_t timeout_ns = 0;
};

[[nodiscard]] std::string encode_probe_request(const GpuFileDecodeConfig& config,
                                               std::uint64_t timeout_ns);
[[nodiscard]] ProbeWorkerRequest decode_probe_request(std::string_view payload);
[[nodiscard]] std::string encode_probe_success(const GpuVideoProbe& probe);
[[nodiscard]] std::string encode_probe_failure(std::string_view kind, std::string_view message);
[[nodiscard]] GpuVideoProbe decode_probe_response(std::string_view payload);

} // namespace reco::io::detail
