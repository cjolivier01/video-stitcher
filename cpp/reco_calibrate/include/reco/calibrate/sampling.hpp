#pragma once

#include "reco/calibrate/gpu_undistort.hpp"
#include "reco/calibrate/types.hpp"
#include "reco/io/gpu_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reco::calibrate {

[[nodiscard]] std::vector<std::uint64_t> select_frame_indices(std::uint64_t total_frames,
                                                              double fps, std::size_t num_samples,
                                                              double skip_start_secs,
                                                              double skip_end_secs);

[[nodiscard]] GrayFrame downscale_if_needed(const GrayFrame& frame, std::uint32_t target_width);

/// Calibration-owned grayscale frame stored entirely in CUDA device memory.
struct GpuCalibrationFrame {
  reco::core::CudaDeviceBuffer y_plane;
  std::size_t pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint64_t frame_index = 0;
  std::optional<std::uint64_t> pts_ns;
  std::optional<std::uint64_t> duration_ns;

  [[nodiscard]] GpuGrayFrame view() const;
};

/// Failure while extracting indexed calibration frames on the GPU.
class GpuFrameExtractionError : public std::runtime_error {
public:
  explicit GpuFrameExtractionError(std::string message) : std::runtime_error(std::move(message)) {}
};

/// Extracts sorted unique frame indices into calibration-owned CUDA Y planes.
[[nodiscard]] std::vector<GpuCalibrationFrame>
extract_gpu_gray_frames(reco::core::CudaBackend& backend, reco::io::GpuFileDecodeSource& source,
                        std::span<const std::uint64_t> frame_indices);

/// Opens an NVDEC/NVMM source and extracts indexed CUDA-resident Y planes.
[[nodiscard]] std::vector<GpuCalibrationFrame> extract_gpu_gray_frames_from_file(
    reco::core::CudaBackend& backend, reco::io::GpuFileDecodeConfig config,
    reco::io::NvbufSurfaceAbi abi, std::span<const std::uint64_t> frame_indices);

} // namespace reco::calibrate
