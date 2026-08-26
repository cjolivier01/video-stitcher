#pragma once

#include "reco/core/cuda_backend.hpp"
#include "reco/core/video_format.hpp"

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
void nv12_to_rgb_chw(core::CudaBackend& backend, core::CudaDevicePtr y, core::CudaDevicePtr uv,
                     core::CudaDevicePtr dst, std::uint32_t y_pitch, std::uint32_t src_width,
                     std::uint32_t src_height, std::uint32_t dst_width, std::uint32_t dst_height,
                     std::uint32_t pad_x, std::uint32_t pad_y, float scale, int rotation_degrees,
                     core::YuvColorMatrix color_matrix, core::YuvColorRange color_range);
void nv12_to_rgb_chw_fullrange(core::CudaBackend& backend, core::CudaDevicePtr y,
                               core::CudaDevicePtr uv, core::CudaDevicePtr dst,
                               std::uint32_t y_pitch, std::uint32_t src_width,
                               std::uint32_t src_height, std::uint32_t dst_width,
                               std::uint32_t dst_height, std::uint32_t pad_x, std::uint32_t pad_y,
                               float scale, int rotation_degrees);

} // namespace reco::detect
