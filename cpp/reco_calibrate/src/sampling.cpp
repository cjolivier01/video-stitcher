#include "reco/calibrate/sampling.hpp"

#include "reco/core/nvrtc_compiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace reco::calibrate {
namespace {

constexpr char kRotate180KernelSource[] = R"cuda(
extern "C" __global__ void rotate_luma_180(
    const unsigned char* src,
    unsigned long long src_pitch,
    unsigned char* dst,
    unsigned long long dst_pitch,
    unsigned int width,
    unsigned int height) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  dst[(unsigned long long)y * dst_pitch + x] =
      src[(unsigned long long)(height - 1U - y) * src_pitch + (width - 1U - x)];
}
)cuda";

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

void validate_frame_indices(std::span<const std::uint64_t> frame_indices) {
  for (std::size_t index = 1; index < frame_indices.size(); ++index) {
    if (frame_indices[index] <= frame_indices[index - 1U]) {
      throw std::invalid_argument("calibration frame indices must be sorted and unique");
    }
  }
}

} // namespace

struct GpuCalibrationFrameReader::Rotate180 {
  explicit Rotate180(reco::core::CudaBackend& backend_in) : backend(backend_in) {
    auto compiler = reco::core::NvrtcCompiler::create();
    const auto capability = backend.compute_capability();
    const auto architecture =
        compiler.select_architecture(capability.major * 10 + capability.minor);
    reco::core::NvrtcCompileOptions options;
    options.values = {"--std=c++11", "--gpu-architecture=compute_" + std::to_string(architecture)};
    kernel = backend.load_kernel_from_ptx(
        compiler.compile(kRotate180KernelSource, "reco_calibrate_rotate_luma_180.cu", options).ptx,
        "rotate_luma_180");
  }

  void run(reco::core::CudaDevicePtr src, std::size_t src_pitch, reco::core::CudaDevicePtr dst,
           std::size_t dst_pitch, std::uint32_t width, std::uint32_t height) {
    auto src_pitch_u64 = static_cast<std::uint64_t>(src_pitch);
    auto dst_pitch_u64 = static_cast<std::uint64_t>(dst_pitch);
    std::array<void*, 6> args{&src, &src_pitch_u64, &dst, &dst_pitch_u64, &width, &height};
    kernel.launch({.grid = {.x = (width + 15U) / 16U, .y = (height + 15U) / 16U},
                   .block = {.x = 16U, .y = 16U}},
                  std::span<void*>{args});
  }

  reco::core::CudaBackend backend;
  reco::core::CudaKernel kernel;
};

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

reco::core::CameraParams
camera_params_after_applied_rotation(const reco::core::CameraParams& camera,
                                     std::uint16_t rotation_degrees) {
  if (rotation_degrees == 0U) {
    return camera;
  }
  if (rotation_degrees != 180U) {
    throw std::invalid_argument("calibration camera parameters support only 0- or 180-degree "
                                "applied rotation");
  }
  auto rotated = camera;
  rotated.cx = static_cast<double>(camera.width) - camera.cx;
  rotated.cy = static_cast<double>(camera.height) - camera.cy;
  return rotated;
}

GpuGrayFrame GpuCalibrationFrame::view() const {
  return {.ptr = y_plane.ptr(),
          .pitch = pitch,
          .width = width,
          .height = height,
          .color_range = color_range,
          .applied_rotation_degrees = applied_rotation_degrees};
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

GpuCalibrationFrameReader::~GpuCalibrationFrameReader() = default;

GpuCalibrationFrame GpuCalibrationFrameReader::read(std::uint64_t frame_index) {
  if (previous_requested_index_.has_value() && frame_index <= *previous_requested_index_) {
    throw std::invalid_argument("calibration frame indices must be sorted and unique");
  }
  previous_requested_index_ = frame_index;

  const auto& source_config = source_->config();
  const bool indexed_seek_available = source_config.indexed_fps_numerator.has_value() &&
                                      source_config.indexed_stream_time_origin_ns.has_value();
  const bool source_starts_at_request =
      !previous_source_index_.has_value() && source_config.start_frame_index == frame_index;
  const bool next_frame_is_request =
      previous_source_index_.has_value() &&
      *previous_source_index_ != std::numeric_limits<std::uint64_t>::max() &&
      *previous_source_index_ + 1U == frame_index;
  if (indexed_seek_available && !source_starts_at_request && !next_frame_is_request) {
    try {
      source_->seek_to_frame(frame_index);
    } catch (const reco::io::GpuDecodeError& error) {
      throw GpuFrameExtractionError("GPU calibration indexed seek failed: " +
                                    std::string(error.what()));
    }
    previous_source_index_.reset();
  }

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

    if (decoded.rotation_degrees != 0U && decoded.rotation_degrees != 180U) {
      throw GpuFrameExtractionError(
          "GPU calibration supports only 0- or 180-degree stream rotation");
    }
    auto allocation = backend_->allocate_pitched(mapped.width, mapped.height, 16);
    if (decoded.rotation_degrees == 180U) {
      if (!rotate_180_) {
        rotate_180_ = std::make_unique<Rotate180>(*backend_);
      }
      rotate_180_->run(mapped.y_ptr, mapped.y_pitch, allocation.buffer.ptr(), allocation.pitch,
                       mapped.width, mapped.height);
    } else {
      backend_->copy_device_to_device_2d({.src = mapped.y_ptr,
                                          .src_pitch = mapped.y_pitch,
                                          .dst = allocation.buffer.ptr(),
                                          .dst_pitch = allocation.pitch,
                                          .width_bytes = mapped.width,
                                          .height = mapped.height});
    }
    // The decoder may recycle its surface when `decoded` is released.
    backend_->synchronize();
    return {.y_plane = std::move(allocation.buffer),
            .pitch = allocation.pitch,
            .width = mapped.width,
            .height = mapped.height,
            .color_range = mapped.color_range,
            .frame_index = decoded.frame_index,
            .pts_ns = decoded.pts_ns,
            .duration_ns = decoded.duration_ns,
            .applied_rotation_degrees = decoded.rotation_degrees};
  }
}

std::vector<GpuCalibrationFrame>
extract_gpu_gray_frames(reco::core::CudaBackend& backend, reco::io::GpuFileDecodeSource& source,
                        std::span<const std::uint64_t> frame_indices) {
  validate_frame_indices(frame_indices);
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
  const GpuFileDecodeSourceOpener opener = [](reco::io::GpuFileDecodeConfig sample_config,
                                              reco::io::NvbufSurfaceAbi sample_abi) {
    return reco::io::open_gstreamer_gpu_file_decode_source(std::move(sample_config), sample_abi);
  };
  return extract_gpu_gray_frames_from_file(backend, std::move(config), abi, frame_indices, opener);
}

std::vector<GpuCalibrationFrame> extract_gpu_gray_frames_from_file(
    reco::core::CudaBackend& backend, reco::io::GpuFileDecodeConfig config,
    reco::io::NvbufSurfaceAbi abi, std::span<const std::uint64_t> frame_indices,
    const GpuFileDecodeSourceOpener& opener) {
  if (config.drop) {
    throw std::invalid_argument(
        "calibration frame extraction requires decoder frame dropping to be disabled");
  }
  validate_frame_indices(frame_indices);
  if (!opener) {
    throw std::invalid_argument("calibration frame extraction requires a GPU source opener");
  }
  if (frame_indices.empty()) {
    return {};
  }

  config.start_frame_index = frame_indices.front();
  auto source = opener(std::move(config), abi);
  if (!source) {
    throw GpuFrameExtractionError("GPU calibration source opener returned no source");
  }
  return extract_gpu_gray_frames(backend, *source, frame_indices);
}

} // namespace reco::calibrate
