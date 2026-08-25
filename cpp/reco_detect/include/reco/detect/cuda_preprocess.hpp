#pragma once

#include "reco/core/cuda_backend.hpp"

#include <cstddef>
#include <cstdint>

namespace reco::detect {

void p010_to_nv12(core::CudaBackend& backend, core::CudaDevicePtr src, core::CudaDevicePtr dst,
                  std::uint32_t samples);
void p010_plane_to_nv12(core::CudaBackend& backend, core::CudaDevicePtr src, std::size_t src_pitch,
                        core::CudaDevicePtr dst, std::uint32_t width, std::uint32_t height);
void normalize_hwc_to_chw(core::CudaBackend& backend, core::CudaDevicePtr src,
                          core::CudaDevicePtr dst, std::uint32_t width, std::uint32_t height);
void normalize_rgba_to_chw(core::CudaBackend& backend, core::CudaDevicePtr src,
                           core::CudaDevicePtr dst, std::uint32_t width, std::uint32_t height);

} // namespace reco::detect
