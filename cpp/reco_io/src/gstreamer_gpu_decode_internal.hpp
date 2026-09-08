#pragma once

#include "reco/io/gpu_decode.hpp"

#include <functional>
#include <memory>

namespace reco::io::detail {

using GpuDecodeOpeningSourceObserver = std::function<bool(GpuFileDecodeSource*)>;

[[nodiscard]] std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_file_decode_source_interruptibly(GpuFileDecodeConfig config, NvbufSurfaceAbi abi,
                                                    const GpuDecodeOpeningSourceObserver& observer);

[[nodiscard]] std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_file_decode_source_interruptibly(
    GpuFileDecodeConfig config, std::shared_ptr<const NvbufSurfaceRuntime> runtime,
    const GpuDecodeOpeningSourceObserver& observer);

} // namespace reco::io::detail
