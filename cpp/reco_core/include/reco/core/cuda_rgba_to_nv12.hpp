#pragma once

#include "reco/core/cuda_backend.hpp"
#include "reco/core/cuda_frame.hpp"
#include "reco/core/nvrtc_compiler.hpp"

#include <cstdint>
#include <memory>

namespace reco::core {

/// Fixed frame shape and CUDA device for one RGBA-to-NV12 converter.
struct CudaRgbaToNv12Config {
  /// Width in pixels. The value must be non-zero and even for NV12 output.
  std::uint32_t width = 0;
  /// Height in pixels. The value must be non-zero and even for NV12 output.
  std::uint32_t height = 0;
  /// CUDA device whose retained primary context owns every borrowed plane.
  int device_ordinal = 0;
};

/// GPU-only converter from pitched 8-bit RGBA to pitched 8-bit NV12.
///
/// Construction compiles and loads one NVRTC kernel for a fixed frame shape.
/// Each conversion borrows its input and output allocations, performs one
/// bounded CUDA launch without allocating or transferring pixels to the host,
/// and waits for a retained stream completion event before returning. Converters
/// and renderers created from the same backend share this ordered stream without
/// synchronizing unrelated CUDA context work. Output matrix and range metadata
/// are read from the supplied `CudaNv12FrameView`; RGBA alpha is ignored when
/// producing gamma-coded YCbCr samples.
class CudaRgbaToNv12Converter {
public:
  /// Creates a converter using the process-default CUDA and NVRTC libraries.
  [[nodiscard]] static CudaRgbaToNv12Converter create(CudaRgbaToNv12Config config);
  /// Creates a converter with injectable runtime handles for controlled deployments and tests.
  [[nodiscard]] static CudaRgbaToNv12Converter create(CudaRgbaToNv12Config config,
                                                      CudaBackend backend, NvrtcCompiler compiler);

  CudaRgbaToNv12Converter(const CudaRgbaToNv12Converter&) = delete;
  CudaRgbaToNv12Converter& operator=(const CudaRgbaToNv12Converter&) = delete;
  CudaRgbaToNv12Converter(CudaRgbaToNv12Converter&&) noexcept;
  CudaRgbaToNv12Converter& operator=(CudaRgbaToNv12Converter&&) noexcept;
  ~CudaRgbaToNv12Converter();

  /// Converts one borrowed frame and waits for completion before either view may be released.
  void convert(const CudaRgbaFrameView& input, const CudaNv12FrameView& output) const;

  /// Process-local CUDA context identity accepted by this converter.
  [[nodiscard]] CudaContextId context_id() const;
  /// CUDA device ordinal accepted by this converter.
  [[nodiscard]] int device_ordinal() const;
  /// Fixed frame width in pixels.
  [[nodiscard]] std::uint32_t width() const;
  /// Fixed frame height in pixels.
  [[nodiscard]] std::uint32_t height() const;

private:
  struct Impl;
  explicit CudaRgbaToNv12Converter(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace reco::core
