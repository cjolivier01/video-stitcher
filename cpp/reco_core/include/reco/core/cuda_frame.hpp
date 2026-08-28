#pragma once

#include "reco/core/cuda_backend.hpp"
#include "reco/core/video_format.hpp"

#include <cstddef>
#include <cstdint>

namespace reco::core {

/// Process-local identity of the CUDA context in which a borrowed pointer is valid.
using CudaContextId = std::uintptr_t;

/// Borrowed view of pitched CUDA device memory.
///
/// The view does not own or retain the allocation. Its caller must keep the
/// allocation and CUDA context alive for the complete use of the view.
struct CudaPitchedPlaneView {
public:
  /// Creates a view after validating its address, capacity, shape, pitch, and provenance.
  CudaPitchedPlaneView(CudaDevicePtr ptr, std::size_t accessible_bytes, std::size_t pitch_bytes,
                       std::size_t row_bytes, std::uint32_t rows, CudaContextId context_id,
                       int device_ordinal = 0);

  /// First byte of the visible plane in CUDA device memory.
  [[nodiscard]] CudaDevicePtr ptr() const { return ptr_; }
  /// Distance between adjacent rows in bytes.
  [[nodiscard]] std::size_t pitch_bytes() const { return pitch_bytes_; }
  /// Number of visible bytes in each row.
  [[nodiscard]] std::size_t row_bytes() const { return row_bytes_; }
  /// Number of visible rows.
  [[nodiscard]] std::uint32_t rows() const { return rows_; }
  /// Bytes accessible from `ptr` within the borrowed allocation.
  [[nodiscard]] std::size_t accessible_bytes() const { return accessible_bytes_; }
  /// Process-local CUDA context identity in which the pointer is valid.
  [[nodiscard]] CudaContextId context_id() const { return context_id_; }
  /// CUDA device ordinal on which the pointer is valid.
  [[nodiscard]] int device_ordinal() const { return device_ordinal_; }
  /// Minimum address span covering all visible rows and inter-row padding.
  [[nodiscard]] std::size_t address_span_bytes() const { return address_span_bytes_; }

private:
  CudaDevicePtr ptr_ = 0;
  std::size_t accessible_bytes_ = 0;
  std::size_t pitch_bytes_ = 0;
  std::size_t row_bytes_ = 0;
  std::uint32_t rows_ = 0;
  CudaContextId context_id_ = 0;
  int device_ordinal_ = 0;
  std::size_t address_span_bytes_ = 0;
};

/// Borrowed, 8-bit, 4:2:0 NV12 CUDA frame with independently pitched planes.
///
/// Both planes must belong to the same CUDA device. The Y plane has `height`
/// rows of `width` bytes and the interleaved UV plane has `height / 2` rows of
/// `width` bytes. The view owns neither plane.
struct CudaNv12FrameView {
public:
  /// Creates a view after validating both planes, dimensions, and color metadata.
  CudaNv12FrameView(CudaPitchedPlaneView y_plane, CudaPitchedPlaneView uv_plane,
                    std::uint32_t width, std::uint32_t height, YuvColorMatrix color_matrix,
                    YuvColorRange color_range);

  /// Full-resolution luma plane.
  [[nodiscard]] const CudaPitchedPlaneView& y_plane() const { return y_plane_; }
  /// Half-height interleaved chroma plane.
  [[nodiscard]] const CudaPitchedPlaneView& uv_plane() const { return uv_plane_; }
  /// Visible frame width in pixels.
  [[nodiscard]] std::uint32_t width() const { return width_; }
  /// Visible frame height in pixels.
  [[nodiscard]] std::uint32_t height() const { return height_; }
  /// YCbCr coefficient matrix carried by the frame.
  [[nodiscard]] YuvColorMatrix color_matrix() const { return color_matrix_; }
  /// Encoded luma/chroma range carried by the frame.
  [[nodiscard]] YuvColorRange color_range() const { return color_range_; }
  /// CUDA device ordinal shared by both planes.
  [[nodiscard]] int device_ordinal() const { return y_plane_.device_ordinal(); }
  /// Process-local CUDA context identity shared by both planes.
  [[nodiscard]] CudaContextId context_id() const { return y_plane_.context_id(); }

private:
  CudaPitchedPlaneView y_plane_;
  CudaPitchedPlaneView uv_plane_;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
  YuvColorMatrix color_matrix_ = YuvColorMatrix::Bt601;
  YuvColorRange color_range_ = YuvColorRange::Limited;
};

/// Borrowed, pitched CUDA frame containing four 8-bit RGBA channels per pixel.
struct CudaRgbaFrameView {
public:
  /// Creates a view after validating its address, dimensions, pitch, and device ordinal.
  CudaRgbaFrameView(CudaPitchedPlaneView plane, std::uint32_t width, std::uint32_t height);

  /// Four-channel pixel plane.
  [[nodiscard]] const CudaPitchedPlaneView& plane() const { return plane_; }
  /// Visible frame width in pixels.
  [[nodiscard]] std::uint32_t width() const { return width_; }
  /// Visible frame height in pixels.
  [[nodiscard]] std::uint32_t height() const { return height_; }
  /// CUDA device ordinal on which the plane is valid.
  [[nodiscard]] int device_ordinal() const { return plane_.device_ordinal(); }
  /// Process-local CUDA context identity in which the plane is valid.
  [[nodiscard]] CudaContextId context_id() const { return plane_.context_id(); }

private:
  CudaPitchedPlaneView plane_;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
};

} // namespace reco::core
