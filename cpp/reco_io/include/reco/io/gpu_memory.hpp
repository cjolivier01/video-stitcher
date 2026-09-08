#pragma once

#include "reco/core/cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace reco::io {

/// Inputs needed to estimate the peak retained GPU surfaces of file stitching.
struct GpuStitchMemoryConfig {
  /// Output width in pixels.
  std::uint32_t output_width = 0;
  /// Output height in pixels.
  std::uint32_t output_height = 0;
  /// Left decoded-frame width in pixels.
  std::uint32_t left_width = 0;
  /// Left decoded-frame height in pixels.
  std::uint32_t left_height = 0;
  /// Right decoded-frame width in pixels.
  std::uint32_t right_width = 0;
  /// Right decoded-frame height in pixels.
  std::uint32_t right_height = 0;
  /// Maximum frames retained by each GStreamer appsink.
  std::size_t decode_source_capacity = 0;
  /// Maximum frames retained by each side of the stereo pairing queue.
  std::size_t stereo_queue_capacity = 0;
  /// Number of writable NVMM surfaces retained by the encoder input pool.
  std::size_t encode_pool_capacity = 0;
};

/// Overflow-checked estimate of the peak GPU-resident stitching working set.
struct GpuStitchMemoryEstimate {
  /// Encoder-owned NVMM input surfaces.
  std::size_t encode_pool_bytes = 0;
  /// CUDA RGBA renderer output consumed by the NV12 converter.
  std::size_t rgba_output_bytes = 0;
  /// Retained NVDEC, conversion, appsink, pairing, and in-flight input surfaces.
  std::size_t decode_surfaces_bytes = 0;
  /// Sum of all estimated allocations above.
  std::size_t total_bytes = 0;
};

/// Result of applying a topology-aware safety reserve to a working-set estimate.
struct GpuMemoryPreflight {
  /// Estimated bytes retained by the requested operation.
  std::size_t working_set_bytes = 0;
  /// Bytes left unused for CUDA, codecs, display, and shared-memory system pressure.
  std::size_t safety_reserve_bytes = 0;
  /// Minimum free bytes required before allocation begins.
  std::size_t required_free_bytes = 0;
  /// CUDA-reported free bytes at preflight time.
  std::size_t available_free_bytes = 0;
  /// CUDA-reported total bytes at preflight time.
  std::size_t total_bytes = 0;
  /// Whether CUDA reports host/device integrated physical memory.
  bool integrated = false;
};

/// Estimates one pitch-aligned NV12 NVMM allocation, including tiling/metadata headroom.
[[nodiscard]] std::size_t estimate_nvmm_nv12_surface_bytes(std::uint32_t width,
                                                           std::uint32_t height);

/// Estimates a bounded pool of pitch-aligned NV12 NVMM allocations.
[[nodiscard]] std::size_t estimate_gpu_encode_pool_bytes(std::uint32_t width, std::uint32_t height,
                                                         std::size_t pool_capacity);

/// Estimates the complete retained GPU surface set of the file-stitch loop.
[[nodiscard]] GpuStitchMemoryEstimate
estimate_gpu_stitch_memory(const GpuStitchMemoryConfig& config);

/// Computes the safety-adjusted requirement without allocating GPU memory.
[[nodiscard]] GpuMemoryPreflight evaluate_gpu_memory_preflight(std::size_t working_set_bytes,
                                                               const core::CudaMemoryInfo& memory);

/// Throws an actionable error when the CUDA memory snapshot cannot admit the estimate.
void require_gpu_memory_preflight(const GpuMemoryPreflight& preflight, std::string_view operation);

} // namespace reco::io
