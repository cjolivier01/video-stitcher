#pragma once

#include "reco/core/calibration.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/core/cuda_frame.hpp"
#include "reco/core/nvrtc_compiler.hpp"

#include <cstdint>
#include <memory>

namespace reco::core {

/// Per-frame virtual-camera controls for the fused CUDA stitch renderer.
struct CudaStitchViewport {
  /// Horizontal camera rotation in radians.
  float yaw = 0.0F;
  /// Vertical camera rotation in radians.
  float pitch = 0.0F;
  /// Vertical perspective field of view in degrees.
  float fov_degrees = 75.0F;
  /// Applies the source's 180-degree display rotation without moving pixels to the CPU.
  bool flip_left_180 = false;
  /// Applies the source's 180-degree display rotation without moving pixels to the CPU.
  bool flip_right_180 = false;
};

/// Fixed calibration, output shape, and CUDA device for one renderer instance.
struct CudaStitchRendererConfig {
  /// Stereo intrinsics, distortion, physical plane layout, and render corrections.
  MatchCalibration calibration;
  /// Width of the pitched RGBA output in pixels.
  std::uint32_t output_width = 0;
  /// Height of the pitched RGBA output in pixels.
  std::uint32_t output_height = 0;
  /// CUDA device whose retained primary context owns all input and output pointers.
  int device_ordinal = 0;
};

/// Fused GPU-only stereo NV12-to-RGBA panoramic renderer.
///
/// Construction compiles exactly one NVRTC kernel and loads it into the selected
/// primary CUDA context. Every render reads borrowed pitched NV12 planes and
/// writes borrowed pitched RGBA storage without CPU pixel processing. The call
/// synchronizes the CUDA context before returning so decoder owners may be
/// released immediately; a future stream/fence API can relax this contract.
class CudaStereoStitchRenderer {
public:
  /// Creates a renderer using the process-default CUDA and NVRTC libraries.
  [[nodiscard]] static CudaStereoStitchRenderer create(CudaStitchRendererConfig config);
  /// Creates a renderer using explicit runtime handles, primarily for controlled deployments.
  [[nodiscard]] static CudaStereoStitchRenderer create(CudaStitchRendererConfig config,
                                                       CudaBackend backend, NvrtcCompiler compiler);

  CudaStereoStitchRenderer(const CudaStereoStitchRenderer&) = delete;
  CudaStereoStitchRenderer& operator=(const CudaStereoStitchRenderer&) = delete;
  CudaStereoStitchRenderer(CudaStereoStitchRenderer&&) noexcept;
  CudaStereoStitchRenderer& operator=(CudaStereoStitchRenderer&&) noexcept;
  ~CudaStereoStitchRenderer();

  /// Renders and synchronizes one stereo frame entirely in CUDA device memory.
  void render(const CudaNv12FrameView& left, const CudaNv12FrameView& right,
              const CudaRgbaFrameView& output, const CudaStitchViewport& viewport = {}) const;

  /// Process-local CUDA context identity accepted by this renderer.
  [[nodiscard]] CudaContextId context_id() const;
  /// CUDA device ordinal accepted by this renderer.
  [[nodiscard]] int device_ordinal() const;
  /// Fixed output width in pixels.
  [[nodiscard]] std::uint32_t output_width() const;
  /// Fixed output height in pixels.
  [[nodiscard]] std::uint32_t output_height() const;

private:
  struct Impl;
  explicit CudaStereoStitchRenderer(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace reco::core
