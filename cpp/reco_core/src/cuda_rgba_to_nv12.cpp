#include "reco/core/cuda_rgba_to_nv12.hpp"

#include "reco/core/video_format.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace reco::core {
namespace {

constexpr std::string_view kKernelName = "reco_rgba_to_nv12";
constexpr std::uint32_t kBlockWidth = 16;
constexpr std::uint32_t kBlockHeight = 16;
constexpr std::uint32_t kMaximumGridX = 2'147'483'647U;
constexpr std::uint32_t kMaximumGridY = 65'535U;

struct KernelColorParams {
  float red_luma = 0.0F;
  float green_luma = 0.0F;
  float blue_luma = 0.0F;
  float luma_scale = 1.0F;
  float luma_offset = 0.0F;
  float chroma_scale = 1.0F;
  float chroma_center = 128.0F;
  float reserved = 0.0F;
};

static_assert(sizeof(KernelColorParams) == 32);

constexpr std::string_view kCudaSource = R"cuda(
struct ColorParams {
  float red_luma;
  float green_luma;
  float blue_luma;
  float luma_scale;
  float luma_offset;
  float chroma_scale;
  float chroma_center;
  float reserved;
};

static_assert(sizeof(ColorParams) == 32, "ColorParams ABI mismatch");

__device__ __forceinline__ unsigned char quantize(float value) {
  value = fminf(255.0f, fmaxf(0.0f, value));
  return static_cast<unsigned char>(floorf(value + 0.5f));
}

__device__ __forceinline__ void load_rgb(const unsigned char* input, unsigned long long pitch,
                                         unsigned int x, unsigned int y,
                                         float* red, float* green, float* blue) {
  const auto* pixel = input + static_cast<unsigned long long>(y) * pitch +
                      static_cast<unsigned long long>(x) * 4ULL;
  *red = static_cast<float>(pixel[0]);
  *green = static_cast<float>(pixel[1]);
  *blue = static_cast<float>(pixel[2]);
}

__device__ __forceinline__ float luma(const ColorParams& color, float red, float green,
                                      float blue) {
  return color.red_luma * red + color.green_luma * green + color.blue_luma * blue;
}

extern "C" __global__ void reco_rgba_to_nv12(
    unsigned long long input_ptr, unsigned long long input_pitch,
    unsigned long long output_y_ptr, unsigned long long output_y_pitch,
    unsigned long long output_uv_ptr, unsigned long long output_uv_pitch,
    unsigned int width, unsigned int height, ColorParams color) {
  const unsigned int chroma_x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int chroma_y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int x = chroma_x * 2U;
  const unsigned int y = chroma_y * 2U;
  if (x >= width || y >= height) {
    return;
  }

  const auto* input = reinterpret_cast<const unsigned char*>(input_ptr);
  auto* output_y = reinterpret_cast<unsigned char*>(output_y_ptr);
  auto* output_uv = reinterpret_cast<unsigned char*>(output_uv_ptr);
  float red[4];
  float green[4];
  float blue[4];
  load_rgb(input, input_pitch, x, y, &red[0], &green[0], &blue[0]);
  load_rgb(input, input_pitch, x + 1U, y, &red[1], &green[1], &blue[1]);
  load_rgb(input, input_pitch, x, y + 1U, &red[2], &green[2], &blue[2]);
  load_rgb(input, input_pitch, x + 1U, y + 1U, &red[3], &green[3], &blue[3]);

  auto* y_row_0 = output_y + static_cast<unsigned long long>(y) * output_y_pitch + x;
  auto* y_row_1 = y_row_0 + output_y_pitch;
  y_row_0[0] = quantize(color.luma_offset +
                         color.luma_scale * luma(color, red[0], green[0], blue[0]));
  y_row_0[1] = quantize(color.luma_offset +
                         color.luma_scale * luma(color, red[1], green[1], blue[1]));
  y_row_1[0] = quantize(color.luma_offset +
                         color.luma_scale * luma(color, red[2], green[2], blue[2]));
  y_row_1[1] = quantize(color.luma_offset +
                         color.luma_scale * luma(color, red[3], green[3], blue[3]));

  const float average_red = (red[0] + red[1] + red[2] + red[3]) * 0.25f;
  const float average_green = (green[0] + green[1] + green[2] + green[3]) * 0.25f;
  const float average_blue = (blue[0] + blue[1] + blue[2] + blue[3]) * 0.25f;
  const float average_luma = luma(color, average_red, average_green, average_blue);
  const float cb = color.chroma_center + color.chroma_scale *
      (average_blue - average_luma) / (2.0f * (1.0f - color.blue_luma));
  const float cr = color.chroma_center + color.chroma_scale *
      (average_red - average_luma) / (2.0f * (1.0f - color.red_luma));
  auto* uv = output_uv + static_cast<unsigned long long>(chroma_y) * output_uv_pitch + x;
  uv[0] = quantize(cb);
  uv[1] = quantize(cr);
}
)cuda";

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, std::string_view label) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error("CUDA RGBA-to-NV12 " + std::string(label) + " overflows size_t");
  }
  return lhs * rhs;
}

void validate_config_shape(const CudaRgbaToNv12Config& config) {
  if (config.width == 0 || config.height == 0 || (config.width % 2U) != 0U ||
      (config.height % 2U) != 0U) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 dimensions must be non-zero and even");
  }

  const auto width = static_cast<std::size_t>(config.width);
  const auto height = static_cast<std::size_t>(config.height);
  const auto rgba_row_bytes = checked_multiply(width, 4U, "RGBA row size");
  (void)checked_multiply(rgba_row_bytes, height, "RGBA frame size");
  const auto y_bytes = checked_multiply(width, height, "luma plane size");
  if (y_bytes > std::numeric_limits<std::size_t>::max() - y_bytes / 2U) {
    throw std::overflow_error("CUDA RGBA-to-NV12 NV12 frame size overflows size_t");
  }

  const auto chroma_columns = config.width / 2U;
  const auto chroma_rows = config.height / 2U;
  const auto grid_x = chroma_columns / kBlockWidth +
                      static_cast<std::uint32_t>((chroma_columns % kBlockWidth) != 0U);
  const auto grid_y =
      chroma_rows / kBlockHeight + static_cast<std::uint32_t>((chroma_rows % kBlockHeight) != 0U);
  if (grid_x > kMaximumGridX || grid_y > kMaximumGridY) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 dimensions exceed CUDA grid limits");
  }
}

void validate_config(const CudaRgbaToNv12Config& config, const CudaBackend& backend) {
  validate_config_shape(config);
  const auto device_count = backend.device_count();
  if (config.device_ordinal < 0 || config.device_ordinal >= device_count) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 device ordinal is out of range");
  }
}

int encoded_architecture(CudaComputeCapability capability) {
  if (capability.major > (std::numeric_limits<int>::max() - capability.minor) / 10) {
    throw std::overflow_error("CUDA RGBA-to-NV12 compute capability encoding overflows int");
  }
  return capability.major * 10 + capability.minor;
}

KernelColorParams color_params(YuvColorMatrix matrix, YuvColorRange range) {
  KernelColorParams params;
  switch (matrix) {
  case YuvColorMatrix::Bt601:
    params.red_luma = 0.299F;
    params.blue_luma = 0.114F;
    break;
  case YuvColorMatrix::Bt709:
    params.red_luma = 0.2126F;
    params.blue_luma = 0.0722F;
    break;
  case YuvColorMatrix::Bt2020:
    params.red_luma = 0.2627F;
    params.blue_luma = 0.0593F;
    break;
  default:
    throw std::invalid_argument("CUDA RGBA-to-NV12 output has an unsupported YUV color matrix");
  }
  params.green_luma = 1.0F - params.red_luma - params.blue_luma;

  switch (range) {
  case YuvColorRange::Limited:
    params.luma_scale = 219.0F / 255.0F;
    params.luma_offset = 16.0F;
    params.chroma_scale = 224.0F / 255.0F;
    params.chroma_center = 128.0F;
    break;
  case YuvColorRange::Full:
    params.luma_scale = 1.0F;
    params.luma_offset = 0.0F;
    params.chroma_scale = 1.0F;
    params.chroma_center = 127.5F;
    break;
  default:
    throw std::invalid_argument("CUDA RGBA-to-NV12 output has an unsupported YUV color range");
  }
  return params;
}

CudaValidatedSpan validate_plane_allocation(const CudaBackend& backend,
                                            const CudaPitchedPlaneView& plane,
                                            std::string_view label, CudaSpanAccess required_access,
                                            int device_ordinal) {
  try {
    if (const auto* validation = plane.driver_validation(); validation != nullptr) {
      if (!validation->permits(required_access)) {
        throw std::invalid_argument(required_access == CudaSpanAccess::ReadWrite
                                        ? "CUDA device span does not permit device writes"
                                        : "CUDA device span does not permit device reads");
      }
      return *validation;
    }
    return backend.retain_device_span(plane.ptr(), plane.address_span_bytes(), required_access,
                                      device_ordinal);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 " + std::string(label) +
                                " plane is invalid: " + error.what());
  }
}

void validate_frame(const CudaRgbaToNv12Config& config, CudaContextId context_id,
                    const CudaRgbaFrameView& input, const CudaNv12FrameView& output) {
  if (input.width() != config.width || input.height() != config.height ||
      output.width() != config.width || output.height() != config.height) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 frame dimensions do not match the converter");
  }
  if (input.context_id() != context_id || output.context_id() != context_id) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 frame belongs to a different CUDA context");
  }
  if (input.device_ordinal() != config.device_ordinal ||
      output.device_ordinal() != config.device_ordinal) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 frame belongs to a different CUDA device");
  }
}

std::uint64_t checked_pitch(std::size_t pitch) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (pitch > std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("CUDA RGBA-to-NV12 pitch does not fit the kernel ABI");
    }
  }
  return static_cast<std::uint64_t>(pitch);
}

} // namespace

struct CudaRgbaToNv12Converter::Impl {
  Impl(CudaRgbaToNv12Config config_in, CudaContextId context_id_in, CudaBackend backend_in,
       CudaKernel kernel_in)
      : config(config_in), context_id(context_id_in), backend(std::move(backend_in)),
        kernel(std::move(kernel_in)) {}

  CudaRgbaToNv12Config config;
  CudaContextId context_id = 0;
  CudaBackend backend;
  CudaKernel kernel;
  mutable std::mutex convert_mutex;
};

CudaRgbaToNv12Converter CudaRgbaToNv12Converter::create(CudaRgbaToNv12Config config) {
  validate_config_shape(config);
  return create(config, CudaBackend::create(), NvrtcCompiler::create());
}

CudaRgbaToNv12Converter CudaRgbaToNv12Converter::create(CudaRgbaToNv12Config config,
                                                        CudaBackend backend,
                                                        NvrtcCompiler compiler) {
  validate_config(config, backend);
  const auto capability = backend.compute_capability(config.device_ordinal);
  const auto target_architecture = compiler.select_architecture(encoded_architecture(capability));
  const auto context_id = backend.primary_context_id(config.device_ordinal);
  NvrtcCompileOptions options;
  options.values = {"--std=c++17",
                    "--gpu-architecture=compute_" + std::to_string(target_architecture)};
  const auto compiled = compiler.compile(kCudaSource, "reco_cuda_rgba_to_nv12.cu", options);
  auto module = backend.load_module_from_ptx(compiled.ptx, config.device_ordinal);
  auto kernel = module.load_kernel(kKernelName);
  return CudaRgbaToNv12Converter(
      std::make_unique<Impl>(config, context_id, std::move(backend), std::move(kernel)));
}

CudaRgbaToNv12Converter::CudaRgbaToNv12Converter(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CudaRgbaToNv12Converter::CudaRgbaToNv12Converter(CudaRgbaToNv12Converter&&) noexcept = default;
CudaRgbaToNv12Converter&
CudaRgbaToNv12Converter::operator=(CudaRgbaToNv12Converter&&) noexcept = default;
CudaRgbaToNv12Converter::~CudaRgbaToNv12Converter() = default;

void CudaRgbaToNv12Converter::convert(const CudaRgbaFrameView& input,
                                      const CudaNv12FrameView& output) const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA RGBA-to-NV12 converter");
  }
  const auto& state = *impl_;
  std::lock_guard<std::mutex> lock(state.convert_mutex);
  validate_frame(state.config, state.context_id, input, output);
  const auto input_span =
      validate_plane_allocation(state.backend, input.plane(), "RGBA input", CudaSpanAccess::Read,
                                state.config.device_ordinal);
  const auto output_y_span =
      validate_plane_allocation(state.backend, output.y_plane(), "Y output",
                                CudaSpanAccess::ReadWrite, state.config.device_ordinal);
  const auto output_uv_span =
      validate_plane_allocation(state.backend, output.uv_plane(), "UV output",
                                CudaSpanAccess::ReadWrite, state.config.device_ordinal);
  if (input_span.aliases(output_y_span) || input_span.aliases(output_uv_span)) {
    throw std::invalid_argument("CUDA RGBA-to-NV12 input and output memory must not overlap");
  }

  auto input_ptr = input.plane().ptr();
  auto input_pitch = checked_pitch(input.plane().pitch_bytes());
  auto output_y_ptr = output.y_plane().ptr();
  auto output_y_pitch = checked_pitch(output.y_plane().pitch_bytes());
  auto output_uv_ptr = output.uv_plane().ptr();
  auto output_uv_pitch = checked_pitch(output.uv_plane().pitch_bytes());
  auto width = state.config.width;
  auto height = state.config.height;
  auto color = color_params(output.color_matrix(), output.color_range());
  std::array<void*, 9> arguments{&input_ptr,      &input_pitch,   &output_y_ptr,
                                 &output_y_pitch, &output_uv_ptr, &output_uv_pitch,
                                 &width,          &height,        &color};
  const auto chroma_columns = width / 2U;
  const auto chroma_rows = height / 2U;
  const auto grid_x = chroma_columns / kBlockWidth +
                      static_cast<std::uint32_t>((chroma_columns % kBlockWidth) != 0U);
  const auto grid_y =
      chroma_rows / kBlockHeight + static_cast<std::uint32_t>((chroma_rows % kBlockHeight) != 0U);
  state.kernel.launch({.grid = {grid_x, grid_y, 1},
                       .block = {kBlockWidth, kBlockHeight, 1},
                       .shared_memory_bytes = 0},
                      std::span<void*>(arguments));
  state.kernel.synchronize();
}

CudaContextId CudaRgbaToNv12Converter::context_id() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA RGBA-to-NV12 converter");
  }
  return impl_->context_id;
}

int CudaRgbaToNv12Converter::device_ordinal() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA RGBA-to-NV12 converter");
  }
  return impl_->config.device_ordinal;
}

std::uint32_t CudaRgbaToNv12Converter::width() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA RGBA-to-NV12 converter");
  }
  return impl_->config.width;
}

std::uint32_t CudaRgbaToNv12Converter::height() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA RGBA-to-NV12 converter");
  }
  return impl_->config.height;
}

} // namespace reco::core
