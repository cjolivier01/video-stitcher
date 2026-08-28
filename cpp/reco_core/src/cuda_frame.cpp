#include "reco/core/cuda_frame.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace reco::core {
namespace {

std::size_t checked_address_span(std::size_t pitch_bytes, std::size_t row_bytes,
                                 std::uint32_t rows) {
  const auto preceding_rows = static_cast<std::size_t>(rows - 1);
  if (preceding_rows != 0 &&
      pitch_bytes > (std::numeric_limits<std::size_t>::max() - row_bytes) / preceding_rows) {
    throw std::overflow_error("CUDA pitched plane address span overflows size_t");
  }
  return preceding_rows * pitch_bytes + row_bytes;
}

void validate_device_address_span(CudaDevicePtr ptr, std::size_t address_span_bytes) {
  const auto last_byte_offset = address_span_bytes - 1;
  if (last_byte_offset > std::numeric_limits<CudaDevicePtr>::max() - ptr) {
    throw std::overflow_error("CUDA pitched plane address span overflows the device pointer");
  }
}

void validate_color_metadata(YuvColorMatrix matrix, YuvColorRange range) {
  switch (matrix) {
  case YuvColorMatrix::Bt601:
  case YuvColorMatrix::Bt709:
  case YuvColorMatrix::Bt2020:
    break;
  default:
    throw std::invalid_argument("CUDA NV12 frame has an unsupported YUV color matrix");
  }
  switch (range) {
  case YuvColorRange::Limited:
  case YuvColorRange::Full:
    break;
  default:
    throw std::invalid_argument("CUDA NV12 frame has an unsupported YUV color range");
  }
}

bool planes_overlap(const CudaPitchedPlaneView& lhs, const CudaPitchedPlaneView& rhs) {
  const auto lhs_last = lhs.ptr() + lhs.address_span_bytes() - 1;
  const auto rhs_last = rhs.ptr() + rhs.address_span_bytes() - 1;
  return lhs.ptr() <= rhs_last && rhs.ptr() <= lhs_last;
}

std::size_t rgba_row_bytes(std::uint32_t width) {
  constexpr std::size_t kRgbaChannels = 4;
  if (static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / kRgbaChannels) {
    throw std::overflow_error("CUDA RGBA row size overflows size_t");
  }
  return static_cast<std::size_t>(width) * kRgbaChannels;
}

} // namespace

CudaPitchedPlaneView::CudaPitchedPlaneView(CudaDevicePtr ptr, std::size_t accessible_bytes,
                                           std::size_t pitch_bytes, std::size_t row_bytes,
                                           std::uint32_t rows, CudaContextId context_id,
                                           int device_ordinal)
    : ptr_(ptr), accessible_bytes_(accessible_bytes), pitch_bytes_(pitch_bytes),
      row_bytes_(row_bytes), rows_(rows), context_id_(context_id), device_ordinal_(device_ordinal) {
  if (ptr == 0) {
    throw std::invalid_argument("CUDA pitched plane pointer must be non-zero");
  }
  if (row_bytes == 0 || rows == 0) {
    throw std::invalid_argument("CUDA pitched plane dimensions must be non-zero");
  }
  if (pitch_bytes < row_bytes) {
    throw std::invalid_argument("CUDA pitched plane pitch is smaller than its row size");
  }
  if (device_ordinal < 0) {
    throw std::invalid_argument("CUDA pitched plane device ordinal must be non-negative");
  }
  if (context_id == 0) {
    throw std::invalid_argument("CUDA pitched plane context identity must be non-zero");
  }
  address_span_bytes_ = checked_address_span(pitch_bytes, row_bytes, rows);
  if (accessible_bytes < address_span_bytes_) {
    throw std::invalid_argument("CUDA pitched plane capacity is smaller than its address span");
  }
  validate_device_address_span(ptr, address_span_bytes_);
  validate_device_address_span(ptr, accessible_bytes);
}

CudaNv12FrameView::CudaNv12FrameView(CudaPitchedPlaneView y_plane, CudaPitchedPlaneView uv_plane,
                                     std::uint32_t width, std::uint32_t height,
                                     YuvColorMatrix color_matrix, YuvColorRange color_range)
    : y_plane_(std::move(y_plane)), uv_plane_(std::move(uv_plane)), width_(width), height_(height),
      color_matrix_(color_matrix), color_range_(color_range) {
  if (width == 0 || height == 0 || (width % 2) != 0 || (height % 2) != 0) {
    throw std::invalid_argument("CUDA NV12 frame dimensions must be non-zero and even");
  }
  if (y_plane_.row_bytes() != width || y_plane_.rows() != height) {
    throw std::invalid_argument("CUDA NV12 Y plane shape does not match the frame dimensions");
  }
  if (uv_plane_.row_bytes() != width || uv_plane_.rows() != height / 2) {
    throw std::invalid_argument("CUDA NV12 UV plane shape does not match the frame dimensions");
  }
  if (y_plane_.device_ordinal() != uv_plane_.device_ordinal()) {
    throw std::invalid_argument("CUDA NV12 frame planes must use the same device ordinal");
  }
  if (y_plane_.context_id() != uv_plane_.context_id()) {
    throw std::invalid_argument("CUDA NV12 frame planes must use the same CUDA context");
  }
  if (planes_overlap(y_plane_, uv_plane_)) {
    throw std::invalid_argument("CUDA NV12 frame planes must not overlap");
  }
  validate_color_metadata(color_matrix, color_range);
}

CudaRgbaFrameView::CudaRgbaFrameView(CudaPitchedPlaneView plane, std::uint32_t width,
                                     std::uint32_t height)
    : plane_(std::move(plane)), width_(width), height_(height) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("CUDA RGBA frame dimensions must be non-zero");
  }
  if (plane_.row_bytes() != rgba_row_bytes(width) || plane_.rows() != height) {
    throw std::invalid_argument("CUDA RGBA plane shape does not match the frame dimensions");
  }
}

} // namespace reco::core
