#pragma once

#include "reco/core/calibration.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/core/video_format.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace reco::calibrate {

struct GpuGrayFrame {
  reco::core::CudaDevicePtr ptr = 0;
  std::size_t pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  reco::core::YuvColorRange color_range = reco::core::YuvColorRange::Full;
};

struct GpuUndistortConfig {
  reco::core::CameraParams camera;
  std::uint32_t output_width = 0;
  std::uint32_t output_height = 0;
};

class GpuCalibrationUndistorter {
public:
  GpuCalibrationUndistorter(reco::core::CudaBackend& backend, GpuUndistortConfig config);
  ~GpuCalibrationUndistorter();

  GpuCalibrationUndistorter(const GpuCalibrationUndistorter&) = delete;
  GpuCalibrationUndistorter& operator=(const GpuCalibrationUndistorter&) = delete;
  GpuCalibrationUndistorter(GpuCalibrationUndistorter&&) noexcept;
  GpuCalibrationUndistorter& operator=(GpuCalibrationUndistorter&&) noexcept;

  [[nodiscard]] const GpuUndistortConfig& config() const;
  // Source and destination device ranges must not overlap.
  void undistort_y(const GpuGrayFrame& src, const GpuGrayFrame& dst) const;

private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace reco::calibrate
