#pragma once

#include "reco/calibrate/features.hpp"
#include "reco/calibrate/gpu_undistort.hpp"
#include "reco/core/cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace reco::calibrate {

/// Hard upper bound for detector output and matcher input capacities.
inline constexpr std::uint32_t kMaxGpuAkazeFeatures = 8'192;

/// Hard ceiling for one AKAZE detection workspace, including allocation accounting margin.
inline constexpr std::size_t kMaxGpuAkazeWorkspaceBytes = std::size_t{2} * 1'024U * 1'024U * 1'024U;

/// Device memory left unused when admitting an AKAZE detection workspace.
inline constexpr std::size_t kGpuAkazeMemoryHeadroomBytes = std::size_t{256} * 1'024U * 1'024U;

/// Device ABI for an AKAZE keypoint. Descriptors are stored separately.
struct GpuFeaturePoint {
  /// Source-image x coordinate.
  float x = 0.0F;
  /// Source-image y coordinate.
  float y = 0.0F;
  /// Hessian detector response.
  float response = 0.0F;
  /// AKAZE descriptor support size in detector pixels.
  float size = 0.0F;
  /// Scale-space octave.
  std::uint32_t octave = 0;
  /// Flat scale-space level index.
  std::uint32_t level = 0;
};

/// Non-owning view of device-resident feature data.
struct GpuFeatureView {
  /// Device address of contiguous `GpuFeaturePoint` records.
  reco::core::CudaDevicePtr points = 0;
  /// Device address of contiguous 64-byte descriptors.
  reco::core::CudaDevicePtr descriptors = 0;
  /// Device address of one bounded `uint32_t` feature count.
  reco::core::CudaDevicePtr count = 0;
  /// Allocation capacity shared by points and descriptors.
  std::uint32_t capacity = 0;
};

/// Host-visible compact output from device-resident descriptor matching.
struct GpuMatchedPoint {
  /// Accepted left keypoint.
  KeyPoint left;
  /// Accepted right keypoint.
  KeyPoint right;
  /// Hamming distance between the accepted descriptors.
  std::uint32_t distance = 0;
};

/// Explicit AKAZE detection and matching limits.
struct GpuAkazeConfig {
  /// Maximum number of retained keypoints, in `[1, kMaxGpuAkazeFeatures]`.
  std::uint32_t max_keypoints = 2'000;
  /// Maximum detector-space width before luma downscaling.
  std::uint32_t max_detection_width = 1'920;
  /// Positive Hessian response threshold.
  float threshold = 0.001F;
  /// Bidirectional Lowe ratio in `(0, 1]`; one disables the ratio test.
  double lowe_ratio = 0.75;
  /// Normalized source-image region used when `use_region` is true.
  DetectRegion region;
  /// Enables region cropping and final point filtering.
  bool use_region = false;
  /// Source-pixel distance used by black-border rejection.
  std::uint32_t border_margin = 30;
  /// Number of nonlinear sublevels per full-size octave.
  std::uint32_t num_sublevels = 4;
  /// Maximum number of scale-space octaves.
  std::uint32_t max_octaves = 4;
};

/// Returns a conservative peak CUDA workspace bound without allocating device memory.
///
/// The bound includes worst-case candidate selection, retained scale-space images, detector
/// output, scan scratch, and allocation accounting margin. Invalid dimensions or configuration
/// values throw `std::invalid_argument`; arithmetic overflow throws `std::overflow_error`.
[[nodiscard]] std::size_t gpu_akaze_peak_workspace_bytes(std::uint32_t frame_width,
                                                         std::uint32_t frame_height,
                                                         const GpuAkazeConfig& config);

/// Reduces only `max_detection_width` to the largest GPU geometry that fits `workspace_limit`.
///
/// Full-resolution input and undistortion remain unchanged; the existing CUDA resize feeds the
/// fitted detector geometry. Invalid input/configuration or a limit too small for a usable AKAZE
/// scale space throws `std::invalid_argument`.
[[nodiscard]] GpuAkazeConfig
fit_gpu_akaze_workspace(std::uint32_t frame_width, std::uint32_t frame_height,
                        const GpuAkazeConfig& config,
                        std::size_t workspace_limit = kMaxGpuAkazeWorkspaceBytes);

/// Owning device-resident AKAZE feature set.
class GpuFeatureSet {
public:
  GpuFeatureSet();
  ~GpuFeatureSet();
  GpuFeatureSet(const GpuFeatureSet&) = delete;
  GpuFeatureSet& operator=(const GpuFeatureSet&) = delete;
  GpuFeatureSet(GpuFeatureSet&&) noexcept;
  GpuFeatureSet& operator=(GpuFeatureSet&&) noexcept;

  [[nodiscard]] GpuFeatureView view() const;
  [[nodiscard]] std::uint32_t capacity() const;

private:
  friend class GpuAkazePipeline;
  struct Impl;
  explicit GpuFeatureSet(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// CUDA-resident nonlinear scale-space detector, M-LDB descriptor extractor, and matcher.
class GpuAkazePipeline {
public:
  explicit GpuAkazePipeline(reco::core::CudaBackend& backend);
  ~GpuAkazePipeline();
  GpuAkazePipeline(const GpuAkazePipeline&) = delete;
  GpuAkazePipeline& operator=(const GpuAkazePipeline&) = delete;
  GpuAkazePipeline(GpuAkazePipeline&&) noexcept;
  GpuAkazePipeline& operator=(GpuAkazePipeline&&) noexcept;

  /// Detects keypoints and extracts M-LDB descriptors without frame or descriptor readback.
  [[nodiscard]] GpuFeatureSet detect(const GpuGrayFrame& frame, const GpuAkazeConfig& config) const;
  /// Runs bidirectional Hamming/Lowe matching and reads back only accepted point pairs.
  [[nodiscard]] std::vector<GpuMatchedPoint>
  match(const GpuFeatureView& left, const GpuFeatureView& right, double lowe_ratio) const;
  /// Reads back the bounded scalar feature count, never descriptor or image data.
  [[nodiscard]] std::uint32_t feature_count(const GpuFeatureView& features) const;
  [[nodiscard]] std::vector<GpuMatchedPoint> detect_and_match(const GpuGrayFrame& left,
                                                              const GpuGrayFrame& right,
                                                              const GpuAkazeConfig& config) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace reco::calibrate
