#pragma once

#include "reco/core/cuda_backend.hpp"
#include "reco/core/video_format.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace reco::detect {

class NppError : public std::runtime_error {
public:
  explicit NppError(const std::string& message) : std::runtime_error(message) {}
};

struct NppiSize {
  int width = 0;
  int height = 0;
};

struct NppiRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

[[nodiscard]] bool is_npp_available();
[[nodiscard]] std::string npp_availability_error();

void npp_nv12_to_rgb(core::CudaDevicePtr src_y, std::size_t y_pitch, core::CudaDevicePtr src_uv,
                     std::size_t uv_pitch, core::CudaDevicePtr dst, std::uint32_t width,
                     std::uint32_t height);
void npp_nv12_to_rgb(core::CudaDevicePtr src_y, std::size_t y_pitch, core::CudaDevicePtr src_uv,
                     std::size_t uv_pitch, core::CudaDevicePtr dst, std::uint32_t width,
                     std::uint32_t height, core::YuvColorMatrix color_matrix,
                     core::YuvColorRange color_range);

void npp_resize_c3(core::CudaDevicePtr src, std::uint32_t src_w, std::uint32_t src_h,
                   core::CudaDevicePtr dst, std::uint32_t dst_w, std::uint32_t dst_h,
                   NppiRect dst_roi);

void npp_resize_c4(core::CudaDevicePtr src, std::uint32_t src_w, std::uint32_t src_h,
                   core::CudaDevicePtr dst, std::uint32_t dst_w, std::uint32_t dst_h,
                   NppiRect dst_roi);

void npp_mirror_c3(core::CudaDevicePtr src, core::CudaDevicePtr dst, std::uint32_t width,
                   std::uint32_t height);

} // namespace reco::detect
