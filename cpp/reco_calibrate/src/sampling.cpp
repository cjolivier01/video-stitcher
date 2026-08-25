#include "reco/calibrate/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
    const double mid =
        static_cast<double>(start) + (static_cast<double>(i) + 0.5) * segment_size;
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

} // namespace reco::calibrate
