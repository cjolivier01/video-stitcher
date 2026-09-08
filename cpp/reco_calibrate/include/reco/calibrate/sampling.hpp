#pragma once

#include "reco/calibrate/gpu_undistort.hpp"
#include "reco/calibrate/types.hpp"
#include "reco/io/gpu_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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

/// Selects paired indices while applying a signed right-camera frame offset.
/// The returned indices are always in range for their corresponding streams.
[[nodiscard]] std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>>
select_synchronized_frame_indices(std::uint64_t left_total_frames, std::uint64_t right_total_frames,
                                  double fps, std::size_t num_samples, double skip_start_secs,
                                  double skip_end_secs, std::int64_t sync_offset);

[[nodiscard]] GrayFrame downscale_if_needed(const GrayFrame& frame, std::uint32_t target_width);

/// Transforms lens intrinsics into the coordinate system of an applied display rotation.
/// Calibration currently supports only rotations that preserve frame dimensions.
[[nodiscard]] reco::core::CameraParams
camera_params_after_applied_rotation(const reco::core::CameraParams& camera,
                                     std::uint16_t rotation_degrees);

/// Calibration-owned grayscale frame stored entirely in CUDA device memory.
struct GpuCalibrationFrame {
  reco::core::CudaDeviceBuffer y_plane;
  std::size_t pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  reco::core::YuvColorRange color_range = reco::core::YuvColorRange::Full;
  std::uint64_t frame_index = 0;
  std::optional<std::uint64_t> pts_ns;
  std::optional<std::uint64_t> duration_ns;
  std::uint16_t applied_rotation_degrees = 0;

  [[nodiscard]] GpuGrayFrame view() const;
};

/// Extracts requested frames into calibration-owned CUDA memory. Indexed file
/// sources seek in place across sparse requests; other sources decode forward.
/// Requests must be strictly increasing.
class GpuCalibrationFrameReader {
public:
  GpuCalibrationFrameReader(reco::core::CudaBackend& backend,
                            reco::io::GpuFileDecodeSource& source);
  ~GpuCalibrationFrameReader();
  GpuCalibrationFrameReader(const GpuCalibrationFrameReader&) = delete;
  GpuCalibrationFrameReader& operator=(const GpuCalibrationFrameReader&) = delete;

  /// Seeks or decodes through `frame_index` and returns a device-resident luma copy.
  [[nodiscard]] GpuCalibrationFrame read(std::uint64_t frame_index);

private:
  struct Rotate180;

  reco::core::CudaBackend* backend_ = nullptr;
  reco::io::GpuFileDecodeSource* source_ = nullptr;
  std::optional<std::uint64_t> previous_requested_index_;
  std::optional<std::uint64_t> previous_source_index_;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> dimensions_;
  std::unique_ptr<Rotate180> rotate_180_;
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

using GpuFileDecodeSourceOpener = std::function<std::unique_ptr<reco::io::GpuFileDecodeSource>(
    reco::io::GpuFileDecodeConfig, reco::io::NvbufSurfaceAbi)>;

/// Opens one accurately seekable GPU source and reuses it for every selected frame.
/// The opener exists for pluggable backends and focused source-lifecycle tests.
[[nodiscard]] std::vector<GpuCalibrationFrame> extract_gpu_gray_frames_from_file(
    reco::core::CudaBackend& backend, reco::io::GpuFileDecodeConfig config,
    reco::io::NvbufSurfaceAbi abi, std::span<const std::uint64_t> frame_indices,
    const GpuFileDecodeSourceOpener& opener);

} // namespace reco::calibrate
