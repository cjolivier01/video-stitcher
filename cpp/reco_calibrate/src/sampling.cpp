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

std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>>
select_synchronized_frame_indices(std::uint64_t left_total_frames, std::uint64_t right_total_frames,
                                  double fps, std::size_t num_samples, double skip_start_secs,
                                  double skip_end_secs, std::int64_t sync_offset) {
  if (sync_offset < -reco::core::kMaxSyncOffsetFrames ||
      sync_offset > reco::core::kMaxSyncOffsetFrames) {
    throw std::invalid_argument("calibration sync offset exceeds the supported limit");
  }

  const auto offset = static_cast<std::uint64_t>(sync_offset >= 0 ? sync_offset : -sync_offset);
  std::uint64_t usable_left = left_total_frames;
  std::uint64_t usable_right = right_total_frames;
  if (sync_offset >= 0) {
    if (offset >= right_total_frames) {
      return {};
    }
    usable_right -= offset;
  } else {
    if (offset >= left_total_frames) {
      return {};
    }
    usable_left -= offset;
  }

  auto base = select_frame_indices(std::min(usable_left, usable_right), fps, num_samples,
                                   skip_start_secs, skip_end_secs);
  auto left = base;
  auto right = base;
  if (sync_offset >= 0) {
    for (auto& index : right) {
      index += offset;
    }
  } else {
    for (auto& index : left) {
      index += offset;
    }
  }
  return {std::move(left), std::move(right)};
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
  return {.ptr = y_plane.ptr(),
          .pitch = pitch,
          .width = width,
          .height = height,
          .color_range = color_range};
}

GpuCalibrationFrameReader::GpuCalibrationFrameReader(reco::core::CudaBackend& backend,
                                                     reco::io::GpuFileDecodeSource& source)
    : backend_(&backend), source_(&source) {
  if (!source.gpu_resident()) {
    throw std::invalid_argument("calibration frame extraction requires a GPU-resident source");
  }
  if (source.config().drop) {
    throw std::invalid_argument(
        "calibration frame extraction requires decoder frame dropping to be disabled");
  }
}

GpuCalibrationFrame GpuCalibrationFrameReader::read(std::uint64_t frame_index) {
  if (previous_requested_index_.has_value() && frame_index <= *previous_requested_index_) {
    throw std::invalid_argument("calibration frame indices must be sorted and unique");
  }
  previous_requested_index_ = frame_index;

  while (true) {
    auto result = source_->read();
    if (result.status == reco::io::GpuDecodeFrameStatus::EndOfStream) {
      throw GpuFrameExtractionError("GPU decode reached EOS before calibration frame " +
                                    std::to_string(frame_index));
    }
    if (!result.frame.has_value()) {
      throw GpuFrameExtractionError("GPU decode returned frame status without a frame payload");
    }

    const auto& decoded = *result.frame;
    if (previous_source_index_.has_value() && decoded.frame_index <= *previous_source_index_) {
      throw GpuFrameExtractionError("GPU decode frame indices are not strictly increasing");
    }
    previous_source_index_ = decoded.frame_index;
    if (decoded.frame_index < frame_index) {
      continue;
    }
    if (decoded.frame_index > frame_index) {
      throw GpuFrameExtractionError("GPU decode skipped requested calibration frame " +
                                    std::to_string(frame_index));
    }

    const auto mapped = reco::io::map_gpu_decoded_frame_to_cuda(decoded);
    const auto current_dimensions = std::pair(mapped.width, mapped.height);
    if (!dimensions_.has_value()) {
      dimensions_ = current_dimensions;
    } else if (*dimensions_ != current_dimensions) {
      throw GpuFrameExtractionError("GPU calibration frame dimensions changed during decode");
    }
    if (mapped.width > std::numeric_limits<std::size_t>::max() / mapped.height) {
      throw GpuFrameExtractionError("GPU calibration frame allocation size overflows");
    }

    auto allocation = backend_->allocate_pitched(mapped.width, mapped.height, 16);
    backend_->copy_device_to_device_2d({.src = mapped.y_ptr,
                                        .src_pitch = mapped.y_pitch,
                                        .dst = allocation.buffer.ptr(),
                                        .dst_pitch = allocation.pitch,
                                        .width_bytes = mapped.width,
                                        .height = mapped.height});
    // The decoder may recycle its surface when `decoded` is released.
    backend_->synchronize();
    return {.y_plane = std::move(allocation.buffer),
            .pitch = allocation.pitch,
            .width = mapped.width,
            .height = mapped.height,
            .color_range = mapped.color_range,
            .frame_index = decoded.frame_index,
            .pts_ns = decoded.pts_ns,
            .duration_ns = decoded.duration_ns};
  }
}

std::vector<GpuCalibrationFrame>
extract_gpu_gray_frames(reco::core::CudaBackend& backend, reco::io::GpuFileDecodeSource& source,
                        std::span<const std::uint64_t> frame_indices) {
  for (std::size_t index = 1; index < frame_indices.size(); ++index) {
    if (frame_indices[index] <= frame_indices[index - 1U]) {
      throw std::invalid_argument("calibration frame indices must be sorted and unique");
    }
  }
  if (frame_indices.empty()) {
    return {};
  }

  GpuCalibrationFrameReader reader(backend, source);
  std::vector<GpuCalibrationFrame> extracted;
  extracted.reserve(frame_indices.size());
  for (const auto frame_index : frame_indices) {
    extracted.push_back(reader.read(frame_index));
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
