#include "reco/calibrate/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace reco::calibrate {
namespace {

std::uint64_t rust_float_to_u64_saturating(double value) {
  if (std::isnan(value) || value <= 0.0) {
    return 0;
  }
  const auto max_u64 = static_cast<double>(std::numeric_limits<std::uint64_t>::max());
  if (value >= max_u64) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(value);
}

std::uint64_t saturating_sub(std::uint64_t lhs, std::uint64_t rhs) {
  return rhs > lhs ? 0 : lhs - rhs;
}

} // namespace

std::vector<std::uint64_t> select_frame_indices(std::uint64_t total_frames, double fps,
                                                std::size_t num_samples, double skip_start_secs,
                                                double skip_end_secs) {
  if (total_frames == 0 || num_samples == 0 || fps <= 0.0) {
    return {};
  }

  const std::uint64_t start =
      skip_start_secs > 0.0
          ? std::min(rust_float_to_u64_saturating(skip_start_secs * fps), total_frames)
          : rust_float_to_u64_saturating(static_cast<double>(total_frames) * 0.05);
  const std::uint64_t end =
      skip_end_secs > 0.0
          ? saturating_sub(total_frames, rust_float_to_u64_saturating(skip_end_secs * fps))
          : rust_float_to_u64_saturating(static_cast<double>(total_frames) * 0.95);
  const std::uint64_t usable = saturating_sub(end, start);

  if (usable == 0) {
    return {total_frames / 2};
  }

  const auto n = std::min<std::size_t>(num_samples, static_cast<std::size_t>(usable));
  const double segment_size = static_cast<double>(usable) / static_cast<double>(n);

  std::vector<std::uint64_t> indices;
  indices.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double mid = static_cast<double>(start) + (static_cast<double>(i) + 0.5) * segment_size;
    indices.push_back(std::min(rust_float_to_u64_saturating(mid), end - 1));
  }
  return indices;
}

GrayFrame downscale_if_needed(const GrayFrame& frame, std::uint32_t target_width) {
  if (target_width == 0) {
    throw std::invalid_argument("target_width must be nonzero");
  }
  const auto required = static_cast<std::uint64_t>(frame.width) * frame.height;
  if (frame.data.size() < required) {
    throw std::out_of_range("gray frame data is smaller than width * height");
  }
  if (frame.width <= target_width) {
    return frame;
  }

  const std::uint32_t factor = std::max<std::uint32_t>(frame.width / target_width, 1);
  const std::uint32_t new_w = frame.width / factor;
  const std::uint32_t new_h = frame.height / factor;

  std::vector<std::uint8_t> data(static_cast<std::size_t>(new_w) * new_h);
  const std::uint64_t factor_sq = static_cast<std::uint64_t>(factor) * factor;

  for (std::uint32_t out_y = 0; out_y < new_h; ++out_y) {
    for (std::uint32_t out_x = 0; out_x < new_w; ++out_x) {
      std::uint64_t sum = 0;
      for (std::uint32_t dy = 0; dy < factor; ++dy) {
        for (std::uint32_t dx = 0; dx < factor; ++dx) {
          const std::uint32_t src_x = out_x * factor + dx;
          const std::uint32_t src_y = out_y * factor + dy;
          const auto idx = static_cast<std::size_t>(src_y) * frame.width + src_x;
          sum += frame.data[idx];
        }
      }
      const auto out_idx = static_cast<std::size_t>(out_y) * new_w + out_x;
      data[out_idx] = static_cast<std::uint8_t>(sum / factor_sq);
    }
  }

  return {.data = std::move(data), .width = new_w, .height = new_h};
}

GpuGrayFrame GpuCalibrationFrame::view() const {
  return {.ptr = y_plane.ptr(), .pitch = pitch, .width = width, .height = height};
}

std::vector<GpuCalibrationFrame>
extract_gpu_gray_frames(reco::core::CudaBackend& backend, reco::io::GpuFileDecodeSource& source,
                        std::span<const std::uint64_t> frame_indices) {
  if (!source.gpu_resident()) {
    throw std::invalid_argument("calibration frame extraction requires a GPU-resident source");
  }
  if (source.config().drop) {
    throw std::invalid_argument(
        "calibration frame extraction requires decoder frame dropping to be disabled");
  }
  for (std::size_t i = 1; i < frame_indices.size(); ++i) {
    if (frame_indices[i] <= frame_indices[i - 1]) {
      throw std::invalid_argument("calibration frame indices must be sorted and unique");
    }
  }
  if (frame_indices.empty()) {
    return {};
  }

  std::vector<GpuCalibrationFrame> extracted;
  extracted.reserve(frame_indices.size());
  std::optional<std::uint64_t> previous_source_index;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> dimensions;
  std::size_t next_target = 0;

  while (next_target < frame_indices.size()) {
    auto result = source.read();
    if (result.status == reco::io::GpuDecodeFrameStatus::EndOfStream) {
      throw GpuFrameExtractionError("GPU decode reached EOS before calibration frame " +
                                    std::to_string(frame_indices[next_target]));
    }
    if (!result.frame.has_value()) {
      throw GpuFrameExtractionError("GPU decode returned frame status without a frame payload");
    }

    const auto& decoded = *result.frame;
    if (previous_source_index.has_value() && decoded.frame_index <= *previous_source_index) {
      throw GpuFrameExtractionError("GPU decode frame indices are not strictly increasing");
    }
    previous_source_index = decoded.frame_index;

    const auto target = frame_indices[next_target];
    if (decoded.frame_index < target) {
      continue;
    }
    if (decoded.frame_index > target) {
      throw GpuFrameExtractionError("GPU decode skipped requested calibration frame " +
                                    std::to_string(target));
    }

    const auto mapped = reco::io::map_gpu_decoded_frame_to_cuda(decoded);
    const auto current_dimensions = std::pair(mapped.width, mapped.height);
    if (!dimensions.has_value()) {
      dimensions = current_dimensions;
    } else if (*dimensions != current_dimensions) {
      throw GpuFrameExtractionError("GPU calibration frame dimensions changed during decode");
    }
    if (mapped.width > std::numeric_limits<std::size_t>::max() / mapped.height) {
      throw GpuFrameExtractionError("GPU calibration frame allocation size overflows");
    }

    auto allocation = backend.allocate_pitched(mapped.width, mapped.height, 16);
    backend.copy_device_to_device_2d({.src = mapped.y_ptr,
                                      .src_pitch = mapped.y_pitch,
                                      .dst = allocation.buffer.ptr(),
                                      .dst_pitch = allocation.pitch,
                                      .width_bytes = mapped.width,
                                      .height = mapped.height});
    // The decoder may recycle its surface as soon as this iteration releases
    // mapped. Confirm the D2D read is complete before returning that surface.
    backend.synchronize();
    extracted.push_back({.y_plane = std::move(allocation.buffer),
                         .pitch = allocation.pitch,
                         .width = mapped.width,
                         .height = mapped.height,
                         .frame_index = decoded.frame_index,
                         .pts_ns = decoded.pts_ns,
                         .duration_ns = decoded.duration_ns});
    ++next_target;
  }
  return extracted;
}

std::vector<GpuCalibrationFrame> extract_gpu_gray_frames_from_file(
    reco::core::CudaBackend& backend, reco::io::GpuFileDecodeConfig config,
    reco::io::NvbufSurfaceAbi abi, std::span<const std::uint64_t> frame_indices) {
  if (config.drop) {
    throw std::invalid_argument(
        "calibration frame extraction requires decoder frame dropping to be disabled");
  }
  auto source = reco::io::open_gstreamer_gpu_file_decode_source(std::move(config), abi);
  return extract_gpu_gray_frames(backend, *source, frame_indices);
}

} // namespace reco::calibrate
