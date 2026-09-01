#include "reco/calibrate/gpu_undistort.hpp"

#include "nvrtc_compiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace reco::calibrate {
namespace {

using reco::core::CudaDevicePtr;

constexpr char kUndistortKernelSource[] = R"cuda(
extern "C" __global__ void undistort_y_plane(
    const unsigned char* src,
    unsigned long long src_pitch,
    unsigned int src_width,
    unsigned int src_height,
    unsigned char* dst,
    unsigned long long dst_pitch,
    unsigned int dst_width,
    unsigned int dst_height,
    float src_fx,
    float src_fy,
    float src_cx,
    float src_cy,
    float out_fx,
    float out_fy,
    float out_cx,
    float out_cy,
    float k0,
    float k1,
    float k2,
    float k3,
    unsigned int black_level) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height) {
    return;
  }

  // Match WebGPU fragment-center interpolation: the shader evaluates the
  // remapped UV at the center of each destination pixel.
  const float nx = ((float)x + 0.5f - out_cx) / out_fx;
  const float ny = ((float)y + 0.5f - out_cy) / out_fy;
  const float r = sqrtf(nx * nx + ny * ny);

  float scale = 1.0f;
  if (r > 1.0e-7f) {
    const float theta = atanf(r);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float theta_d =
        theta * (1.0f + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8);
    scale = theta_d / r;
  }
  // A normalized texture coordinate maps to texel space as uv * size - 0.5.
  const float sx = src_fx * nx * scale + src_cx - 0.5f;
  const float sy = src_fy * ny * scale + src_cy - 0.5f;

  unsigned char value = (unsigned char)black_level;
  // The shader bounds-checks normalized coordinates in [0, 1], then its
  // clamp-to-edge sampler extends the edge texels by half a pixel.
  if (sx >= -0.5f && sy >= -0.5f && sx <= (float)src_width - 0.5f &&
      sy <= (float)src_height - 0.5f) {
    const float clamped_x = fminf(fmaxf(sx, 0.0f), (float)(src_width - 1U));
    const float clamped_y = fminf(fmaxf(sy, 0.0f), (float)(src_height - 1U));
    const unsigned int x0 = (unsigned int)floorf(clamped_x);
    const unsigned int y0 = (unsigned int)floorf(clamped_y);
    const unsigned int x1 = x0 + 1U < src_width ? x0 + 1U : x0;
    const unsigned int y1 = y0 + 1U < src_height ? y0 + 1U : y0;
    const float tx = clamped_x - (float)x0;
    const float ty = clamped_y - (float)y0;
    const unsigned char* row0 = src + (unsigned long long)y0 * src_pitch;
    const unsigned char* row1 = src + (unsigned long long)y1 * src_pitch;
    const float a = (1.0f - tx) * (float)row0[x0] + tx * (float)row0[x1];
    const float b = (1.0f - tx) * (float)row1[x0] + tx * (float)row1[x1];
    const float sample = (1.0f - ty) * a + ty * b;
    value = (unsigned char)(sample + 0.5f);
  }

  unsigned char* out = dst + (unsigned long long)y * dst_pitch;
  out[x] = value;
}
)cuda";

void validate_camera(const reco::core::CameraParams& camera) {
  if (camera.width == 0 || camera.height == 0) {
    throw std::invalid_argument("GPU undistort camera dimensions must be non-zero");
  }
  if (camera.width > reco::core::kMaxCalibrationDimension ||
      camera.height > reco::core::kMaxCalibrationDimension) {
    throw std::invalid_argument("GPU undistort camera dimensions exceed calibration limit");
  }
  if (!(std::isfinite(camera.fx) && camera.fx > 0.0 && std::isfinite(camera.fy) &&
        camera.fy > 0.0)) {
    throw std::invalid_argument("GPU undistort camera focal lengths must be finite and positive");
  }
  if (!(std::isfinite(camera.cx) && std::isfinite(camera.cy))) {
    throw std::invalid_argument("GPU undistort camera principal point must be finite");
  }
  for (double coefficient : camera.d) {
    if (!std::isfinite(coefficient)) {
      throw std::invalid_argument("GPU undistort distortion coefficients must be finite");
    }
  }
}

void validate_config(const GpuUndistortConfig& config) {
  validate_camera(config.camera);
  if (config.output_width == 0 || config.output_height == 0) {
    throw std::invalid_argument("GPU undistort output dimensions must be non-zero");
  }
  if (config.output_width > reco::core::kMaxCalibrationDimension ||
      config.output_height > reco::core::kMaxCalibrationDimension) {
    throw std::invalid_argument("GPU undistort output dimensions exceed calibration limit");
  }
}

void validate_frame(const GpuGrayFrame& frame, std::string_view label) {
  if (frame.ptr == 0) {
    throw std::invalid_argument(std::string("GPU undistort ") + std::string(label) +
                                " frame pointer must be non-null");
  }
  if (frame.width == 0 || frame.height == 0) {
    throw std::invalid_argument(std::string("GPU undistort ") + std::string(label) +
                                " dimensions must be non-zero");
  }
  if (frame.pitch < frame.width) {
    throw std::invalid_argument(std::string("GPU undistort ") + std::string(label) +
                                " pitch is smaller than width");
  }
  switch (frame.color_range) {
  case reco::core::YuvColorRange::Limited:
  case reco::core::YuvColorRange::Full:
    break;
  default:
    throw std::invalid_argument(std::string("GPU undistort ") + std::string(label) +
                                " frame has an invalid color range");
  }
}

std::uint64_t frame_span_bytes(const GpuGrayFrame& frame, std::string_view label) {
  const auto pitch = static_cast<std::uint64_t>(frame.pitch);
  const auto rows_before_last = static_cast<std::uint64_t>(frame.height - 1U);
  const auto width = static_cast<std::uint64_t>(frame.width);
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (rows_before_last != 0U && pitch > (maximum - width) / rows_before_last) {
    throw std::overflow_error(std::string("GPU undistort ") + std::string(label) +
                              " frame byte span overflow");
  }
  return rows_before_last * pitch + width;
}

bool ranges_overlap(CudaDevicePtr lhs, std::uint64_t lhs_size, CudaDevicePtr rhs,
                    std::uint64_t rhs_size) {
  if (lhs_size == 0 || rhs_size == 0) {
    return false;
  }
  if (lhs > std::numeric_limits<std::uint64_t>::max() - lhs_size ||
      rhs > std::numeric_limits<std::uint64_t>::max() - rhs_size) {
    throw std::overflow_error("GPU undistort device pointer range overflow");
  }
  return lhs < rhs + rhs_size && rhs < lhs + lhs_size;
}

} // namespace

struct GpuCalibrationUndistorter::Impl {
  Impl(reco::core::CudaBackend& backend_in, GpuUndistortConfig config_in)
      : backend(backend_in), config(std::move(config_in)) {
    validate_config(config);
    backend.ensure_primary_context();
    detail::NvrtcCompiler compiler;
    const std::string ptx =
        compiler.compile(kUndistortKernelSource, "reco_calibrate_gpu_undistort.cu");
    kernel = backend.load_kernel_from_ptx(ptx, "undistort_y_plane");
  }

  reco::core::CudaBackend backend;
  GpuUndistortConfig config;
  reco::core::CudaKernel kernel;
};

GpuCalibrationUndistorter::GpuCalibrationUndistorter(reco::core::CudaBackend& backend,
                                                     GpuUndistortConfig config)
    : impl_(std::make_unique<Impl>(backend, std::move(config))) {}

GpuCalibrationUndistorter::~GpuCalibrationUndistorter() = default;

GpuCalibrationUndistorter::GpuCalibrationUndistorter(GpuCalibrationUndistorter&&) noexcept =
    default;

GpuCalibrationUndistorter&
GpuCalibrationUndistorter::operator=(GpuCalibrationUndistorter&&) noexcept = default;

const GpuUndistortConfig& GpuCalibrationUndistorter::config() const { return impl_->config; }

void GpuCalibrationUndistorter::undistort_y(const GpuGrayFrame& src,
                                            const GpuGrayFrame& dst) const {
  validate_frame(src, "source");
  validate_frame(dst, "destination");
  if (src.color_range != dst.color_range) {
    throw std::invalid_argument("GPU undistort source and destination color ranges must match");
  }
  const auto& config = impl_->config;
  if (dst.width != config.output_width || dst.height != config.output_height) {
    throw std::invalid_argument(
        "GPU undistort destination dimensions must match output dimensions");
  }
  if (src.pitch > std::numeric_limits<std::uint64_t>::max() ||
      dst.pitch > std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("GPU undistort pitch exceeds CUDA kernel parameter range");
  }
  if (ranges_overlap(src.ptr, frame_span_bytes(src, "source"), dst.ptr,
                     frame_span_bytes(dst, "destination"))) {
    throw std::invalid_argument("GPU undistort source and destination ranges must not overlap");
  }

  auto src_ptr = src.ptr;
  auto src_pitch = static_cast<std::uint64_t>(src.pitch);
  auto src_width = src.width;
  auto src_height = src.height;
  auto dst_ptr = dst.ptr;
  auto dst_pitch = static_cast<std::uint64_t>(dst.pitch);
  auto dst_width = dst.width;
  auto dst_height = dst.height;
  const double src_scale_x =
      static_cast<double>(src.width) / static_cast<double>(config.camera.width);
  const double src_scale_y =
      static_cast<double>(src.height) / static_cast<double>(config.camera.height);
  auto src_fx = static_cast<float>(config.camera.fx * src_scale_x);
  auto src_fy = static_cast<float>(config.camera.fy * src_scale_y);
  auto src_cx = static_cast<float>(config.camera.cx * src_scale_x);
  auto src_cy = static_cast<float>(config.camera.cy * src_scale_y);
  const double out_scale_x = static_cast<double>(dst.width) / static_cast<double>(src.width);
  const double out_scale_y = static_cast<double>(dst.height) / static_cast<double>(src.height);
  auto out_fx = static_cast<float>(config.camera.fx * src_scale_x * out_scale_x * 0.5);
  auto out_fy = static_cast<float>(config.camera.fy * src_scale_y * out_scale_y * 0.5);
  auto out_cx = static_cast<float>(
      (static_cast<double>(dst.width) + 2.0 * config.camera.cx * src_scale_x * out_scale_x) * 0.25);
  auto out_cy = static_cast<float>(
      (static_cast<double>(dst.height) + 2.0 * config.camera.cy * src_scale_y * out_scale_y) *
      0.25);
  auto k0 = static_cast<float>(config.camera.d[0]);
  auto k1 = static_cast<float>(config.camera.d[1]);
  auto k2 = static_cast<float>(config.camera.d[2]);
  auto k3 = static_cast<float>(config.camera.d[3]);
  std::uint32_t black_level = src.color_range == reco::core::YuvColorRange::Limited ? 16U : 0U;

  std::array<void*, 21> args{
      &src_ptr,    &src_pitch, &src_width, &src_height, &dst_ptr, &dst_pitch, &dst_width,
      &dst_height, &src_fx,    &src_fy,    &src_cx,     &src_cy,  &out_fx,    &out_fy,
      &out_cx,     &out_cy,    &k0,        &k1,         &k2,      &k3,        &black_level,
  };
  impl_->kernel.launch({.grid = {.x = (dst.width + 15U) / 16U, .y = (dst.height + 15U) / 16U},
                        .block = {.x = 16, .y = 16}},
                       std::span<void*>{args});
  impl_->kernel.synchronize();
}

} // namespace reco::calibrate
