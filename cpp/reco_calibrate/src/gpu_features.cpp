#include "reco/calibrate/gpu_features.hpp"

#include "nvrtc_compiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reco::calibrate {
namespace {

#include "gpu_akaze_kernels.inc"
#include "gpu_match_kernels.inc"

constexpr std::uint32_t kImageBlockWidth = 16;
constexpr std::uint32_t kImageBlockHeight = 16;
constexpr std::uint32_t kLinearBlockSize = 256;
constexpr std::uint32_t kCropMargin = 32;
constexpr double kBaseScaleOffset = 1.6;
constexpr double kDerivativeFactor = 1.5;
constexpr double kContrastPercentile = 0.7;
constexpr float kMinimumContrast = 1.0e-6F;
constexpr double kFedMaximumStep = 0.25;
constexpr std::size_t kCandidateBytes = 76;

static_assert(sizeof(GpuFeaturePoint) == 24);

struct GpuCompactMatch {
  float left_x;
  float left_y;
  float left_response;
  float right_x;
  float right_y;
  float right_response;
  std::uint32_t distance;
};

static_assert(sizeof(GpuCompactMatch) == 28);

struct GpuLevelView {
  std::uint64_t lt;
  std::uint64_t lx;
  std::uint64_t ly;
  std::uint64_t lt_pitch;
  std::uint64_t lx_pitch;
  std::uint64_t ly_pitch;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t octave;
  std::uint32_t reserved;
};

static_assert(sizeof(GpuLevelView) == 64);

struct GpuSelectionLevel {
  std::uint32_t pixel_base;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t octave;
};

static_assert(sizeof(GpuSelectionLevel) == 16);

struct DeviceImage {
  reco::core::CudaDeviceBuffer buffer;
  std::size_t pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct EvolutionSpec {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t octave = 0;
  std::uint32_t sublevel = 0;
  std::uint32_t pixel_base = 0;
  double sigma = 0.0;
  double time = 0.0;
};

struct EvolutionLevel {
  EvolutionSpec spec;
  DeviceImage lt;
  DeviceImage lx;
  DeviceImage ly;
  DeviceImage response;
};

struct DetectionGeometry {
  std::uint32_t crop_x = 0;
  std::uint32_t crop_y = 0;
  std::uint32_t crop_width = 0;
  std::uint32_t crop_height = 0;
  std::uint32_t detect_width = 0;
  std::uint32_t detect_height = 0;
  float detector_scale = 1.0F;
};

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " size overflow");
  }
  return left * right;
}

[[nodiscard]] std::uint32_t checked_u32(std::size_t value, const char* label) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string(label) + " exceeds uint32 range");
  }
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint32_t divide_round_up(std::uint32_t value, std::uint32_t divisor) {
  return value / divisor + (value % divisor != 0U ? 1U : 0U);
}

[[nodiscard]] reco::core::CudaLaunchConfig image_launch(std::uint32_t width, std::uint32_t height) {
  const auto grid_y = divide_round_up(height, kImageBlockHeight);
  if (grid_y > 65'535U) {
    throw std::invalid_argument("GPU AKAZE image height exceeds the CUDA 2D grid limit");
  }
  return {.grid = {.x = divide_round_up(width, kImageBlockWidth), .y = grid_y, .z = 1},
          .block = {.x = kImageBlockWidth, .y = kImageBlockHeight, .z = 1}};
}

[[nodiscard]] reco::core::CudaLaunchConfig linear_launch(std::uint32_t count) {
  return {.grid = {.x = divide_round_up(count, kLinearBlockSize), .y = 1, .z = 1},
          .block = {.x = kLinearBlockSize, .y = 1, .z = 1}};
}

template <typename... Args>
void launch(const reco::core::CudaKernel& kernel, const reco::core::CudaLaunchConfig& config,
            Args&... args) {
  std::array<void*, sizeof...(Args)> pointers{
      const_cast<void*>(static_cast<const void*>(&args))...};
  kernel.launch(config, std::span<void*>(pointers));
}

[[nodiscard]] DeviceImage allocate_float_image(const reco::core::CudaBackend& backend,
                                               std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("GPU AKAZE image dimensions must be non-zero");
  }
  const auto width_bytes = checked_multiply(width, sizeof(float), "GPU AKAZE image row");
  auto allocation = backend.allocate_pitched(width_bytes, height, sizeof(float));
  return {.buffer = std::move(allocation.buffer),
          .pitch = allocation.pitch,
          .width = width,
          .height = height};
}

void copy_image(reco::core::CudaBackend& backend, const DeviceImage& source,
                const DeviceImage& destination) {
  if (source.width != destination.width || source.height != destination.height) {
    throw std::logic_error("GPU AKAZE internal image copy dimension mismatch");
  }
  backend.copy_device_to_device_2d(
      {.src = source.buffer.ptr(),
       .src_pitch = source.pitch,
       .dst = destination.buffer.ptr(),
       .dst_pitch = destination.pitch,
       .width_bytes = checked_multiply(source.width, sizeof(float), "GPU AKAZE image copy"),
       .height = source.height});
}

void upload_bytes(reco::core::CudaBackend& backend, const void* source, std::size_t size,
                  const reco::core::CudaDeviceBuffer& destination) {
  if (size == 0 || source == nullptr || destination.size() < size) {
    throw std::logic_error("GPU AKAZE internal upload extent is invalid");
  }
  backend.copy_host_to_device_2d({.src = source,
                                  .src_pitch = size,
                                  .dst = destination.ptr(),
                                  .dst_pitch = size,
                                  .width_bytes = size,
                                  .height = 1});
}

[[nodiscard]] std::uint32_t download_u32(reco::core::CudaBackend& backend,
                                         reco::core::CudaDevicePtr source) {
  std::uint32_t value = 0;
  backend.copy_device_to_host_2d({.dst = &value,
                                  .dst_pitch = sizeof(value),
                                  .src = source,
                                  .src_pitch = sizeof(value),
                                  .width_bytes = sizeof(value),
                                  .height = 1});
  return value;
}

void validate_frame(const GpuGrayFrame& frame) {
  if (frame.ptr == 0) {
    throw std::invalid_argument("GPU AKAZE frame pointer must be non-zero");
  }
  if (frame.width == 0 || frame.height == 0) {
    throw std::invalid_argument("GPU AKAZE frame dimensions must be non-zero");
  }
  if (frame.pitch < frame.width) {
    throw std::invalid_argument("GPU AKAZE frame pitch is smaller than its visible width");
  }
  const auto pitch = static_cast<std::uint64_t>(frame.pitch);
  const auto rows_before_last = static_cast<std::uint64_t>(frame.height - 1U);
  const auto width = static_cast<std::uint64_t>(frame.width);
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (rows_before_last != 0U && pitch > (maximum - width) / rows_before_last) {
    throw std::invalid_argument("GPU AKAZE frame visible byte span overflows");
  }
  const auto visible_span = rows_before_last * pitch + width;
  if (frame.ptr > maximum - visible_span) {
    throw std::invalid_argument("GPU AKAZE frame device pointer range overflows");
  }
  switch (frame.color_range) {
  case reco::core::YuvColorRange::Limited:
  case reco::core::YuvColorRange::Full:
    break;
  default:
    throw std::invalid_argument("GPU AKAZE frame has an invalid luma range");
  }
}

void validate_region(const DetectRegion& region) {
  const bool finite = std::isfinite(region.x_min) && std::isfinite(region.x_max) &&
                      std::isfinite(region.y_min) && std::isfinite(region.y_max);
  if (!finite || region.x_min < 0.0F || region.x_max > 1.0F || region.y_min < 0.0F ||
      region.y_max > 1.0F || !(region.x_min < region.x_max) || !(region.y_min < region.y_max)) {
    throw std::invalid_argument("GPU AKAZE region must be finite, non-empty, and normalized");
  }
}

void validate_config(const GpuAkazeConfig& config) {
  if (config.max_keypoints == 0) {
    throw std::invalid_argument("GPU AKAZE maximum keypoint count must be non-zero");
  }
  if (config.max_detection_width == 0) {
    throw std::invalid_argument("GPU AKAZE maximum detection width must be non-zero");
  }
  if (!std::isfinite(config.threshold) || !(config.threshold > 0.0F)) {
    throw std::invalid_argument("GPU AKAZE threshold must be finite and positive");
  }
  if (!std::isfinite(config.lowe_ratio) || !(config.lowe_ratio > 0.0) || config.lowe_ratio > 1.0) {
    throw std::invalid_argument("GPU AKAZE Lowe ratio must be finite and in (0, 1]");
  }
  if (config.num_sublevels == 0 || config.num_sublevels > 8) {
    throw std::invalid_argument("GPU AKAZE sublevel count must be in [1, 8]");
  }
  if (config.max_octaves == 0 || config.max_octaves > 8) {
    throw std::invalid_argument("GPU AKAZE octave count must be in [1, 8]");
  }
  if (config.use_region) {
    validate_region(config.region);
  }
}

[[nodiscard]] DetectionGeometry detection_geometry(const GpuGrayFrame& frame,
                                                   const GpuAkazeConfig& config) {
  DetectionGeometry result{.crop_width = frame.width, .crop_height = frame.height};
  if (config.use_region) {
    const auto endpoint = [](float normalized, std::uint32_t dimension, bool upper) {
      const double scaled = static_cast<double>(normalized) * static_cast<double>(dimension);
      const double rounded = upper ? std::ceil(scaled) : std::floor(scaled);
      if (!(rounded >= 0.0) || rounded > static_cast<double>(dimension)) {
        throw std::invalid_argument("GPU AKAZE region endpoint exceeds frame dimensions");
      }
      return static_cast<std::uint32_t>(rounded);
    };
    const auto x_low = endpoint(config.region.x_min, frame.width, false);
    const auto y_low = endpoint(config.region.y_min, frame.height, false);
    const auto x_high = endpoint(config.region.x_max, frame.width, true);
    const auto y_high = endpoint(config.region.y_max, frame.height, true);
    const auto crop_x = x_low > kCropMargin ? x_low - kCropMargin : 0U;
    const auto crop_y = y_low > kCropMargin ? y_low - kCropMargin : 0U;
    const auto crop_right = x_high > frame.width - std::min(frame.width, kCropMargin)
                                ? frame.width
                                : x_high + kCropMargin;
    const auto crop_bottom = y_high > frame.height - std::min(frame.height, kCropMargin)
                                 ? frame.height
                                 : y_high + kCropMargin;
    const auto crop_width = crop_right - crop_x;
    const auto crop_height = crop_bottom - crop_y;
    const std::uint64_t crop_area = static_cast<std::uint64_t>(crop_width) * crop_height;
    const std::uint64_t full_area = static_cast<std::uint64_t>(frame.width) * frame.height;
    const std::uint64_t three_quarters = (full_area / 4U) * 3U + ((full_area % 4U) * 3U) / 4U;
    if (crop_area < three_quarters) {
      result.crop_x = crop_x;
      result.crop_y = crop_y;
      result.crop_width = crop_width;
      result.crop_height = crop_height;
    }
  }

  if (result.crop_width > config.max_detection_width) {
    result.detector_scale =
        static_cast<float>(config.max_detection_width) / static_cast<float>(result.crop_width);
    result.detect_width = config.max_detection_width;
    result.detect_height =
        static_cast<std::uint32_t>(static_cast<float>(result.crop_height) * result.detector_scale);
  } else {
    result.detect_width = result.crop_width;
    result.detect_height = result.crop_height;
  }
  if (result.detect_width == 0 || result.detect_height == 0) {
    throw std::invalid_argument("GPU AKAZE detection geometry is empty");
  }
  return result;
}

[[nodiscard]] std::vector<EvolutionSpec> make_evolution_specs(const DetectionGeometry& geometry,
                                                              const GpuAkazeConfig& config) {
  std::vector<EvolutionSpec> specs;
  std::size_t pixel_base = 0;
  for (std::uint32_t octave = 0; octave < config.max_octaves; ++octave) {
    const auto divisor = std::uint32_t{1} << octave;
    const auto width = geometry.detect_width / divisor;
    const auto height = geometry.detect_height / divisor;
    const auto smallest = std::min(width, height);
    if (smallest < 40U) {
      continue;
    }
    const auto sublevels = smallest < 80U ? 1U : config.num_sublevels;
    for (std::uint32_t sublevel = 0; sublevel < sublevels; ++sublevel) {
      const double sigma = kBaseScaleOffset *
                           std::pow(2.0, static_cast<double>(octave) +
                                             static_cast<double>(sublevel) / config.num_sublevels);
      specs.push_back({.width = width,
                       .height = height,
                       .octave = octave,
                       .sublevel = sublevel,
                       .pixel_base = checked_u32(pixel_base, "GPU AKAZE pixel base"),
                       .sigma = sigma,
                       .time = 0.5 * sigma * sigma});
      pixel_base += checked_multiply(width, height, "GPU AKAZE scale space");
      (void)checked_u32(pixel_base, "GPU AKAZE scale-space pixel count");
    }
  }
  if (specs.empty()) {
    throw std::invalid_argument("GPU AKAZE image is too small; both dimensions must reach 40px");
  }
  return specs;
}

[[nodiscard]] bool is_prime(std::size_t value) {
  if (value < 2) {
    return false;
  }
  for (std::size_t divisor = 2; divisor <= value / divisor; ++divisor) {
    if (value % divisor == 0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::vector<float> fed_steps(double process_time) {
  if (!(process_time > 0.0) || !std::isfinite(process_time)) {
    throw std::logic_error("GPU AKAZE evolution time must be finite and increasing");
  }
  const double estimate = std::sqrt(3.0 * process_time / kFedMaximumStep + 0.25) - 0.5 - 1.0e-8;
  const auto count = static_cast<std::size_t>(std::ceil(estimate) + 0.5);
  if (count == 0) {
    return {};
  }
  const double scale =
      3.0 * process_time / (kFedMaximumStep * static_cast<double>(count * (count + 1U)));
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const double c = 1.0 / (4.0 * static_cast<double>(count) + 2.0);
    const double d = scale * kFedMaximumStep / 2.0;
    const double cosine = std::cos(std::acos(-1.0) * (2.0 * static_cast<double>(index) + 1.0) * c);
    values[index] = static_cast<float>(d / (cosine * cosine));
  }
  if (count == 1) {
    return values;
  }

  const std::size_t stride = count / 2U;
  std::size_t prime = count + 1U;
  while (!is_prime(prime)) {
    ++prime;
  }
  std::vector<float> reordered;
  reordered.reserve(count);
  std::size_t cursor = 0;
  for (std::size_t output = 0; output < count; ++output) {
    std::size_t selected = ((cursor + 1U) * stride) % prime;
    while (selected == 0 || selected - 1U >= count) {
      ++cursor;
      selected = ((cursor + 1U) * stride) % prime;
    }
    reordered.push_back(values[selected - 1U]);
    ++cursor;
  }
  return reordered;
}

void validate_feature_view(const GpuFeatureView& view, const char* label) {
  if (view.capacity == 0) {
    throw std::invalid_argument(std::string("GPU ") + label +
                                " feature view capacity must be non-zero");
  }
  if (view.points == 0 || view.descriptors == 0 || view.count == 0) {
    throw std::invalid_argument(std::string("GPU ") + label +
                                " feature view has a null device pointer");
  }
  if (view.points % alignof(GpuFeaturePoint) != 0 || view.descriptors % 8U != 0 ||
      view.count % alignof(std::uint32_t) != 0) {
    throw std::invalid_argument(std::string("GPU ") + label +
                                " feature view has a misaligned device pointer");
  }
  (void)checked_multiply(view.capacity, sizeof(GpuFeaturePoint), "GPU feature points");
  (void)checked_multiply(view.capacity, kDescriptorBytes, "GPU feature descriptors");
}

} // namespace

struct GpuFeatureSet::Impl {
  reco::core::CudaDeviceBuffer points;
  reco::core::CudaDeviceBuffer descriptors;
  reco::core::CudaDeviceBuffer count;
  std::uint32_t capacity = 0;
};

struct GpuAkazePipeline::Impl {
  explicit Impl(reco::core::CudaBackend& backend_in) : backend(backend_in) {
    backend.ensure_primary_context();
    detail::NvrtcCompiler compiler;
    const auto akaze_ptx = compiler.compile(kGpuAkazeKernelSource, "reco_calibrate_gpu_akaze.cu",
                                            {.disable_fmad = true});
    y_to_float = backend.load_kernel_from_ptx(akaze_ptx, "akaze_y_to_float");
    triangle_vertical_y = backend.load_kernel_from_ptx(akaze_ptx, "akaze_triangle_vertical_y");
    triangle_horizontal = backend.load_kernel_from_ptx(akaze_ptx, "akaze_triangle_horizontal");
    gaussian_x = backend.load_kernel_from_ptx(akaze_ptx, "akaze_gaussian_x");
    gaussian_y = backend.load_kernel_from_ptx(akaze_ptx, "akaze_gaussian_y");
    half_size = backend.load_kernel_from_ptx(akaze_ptx, "akaze_half_size");
    scharr = backend.load_kernel_from_ptx(akaze_ptx, "akaze_scharr");
    clear_histogram = backend.load_kernel_from_ptx(akaze_ptx, "akaze_clear_histogram");
    gradient_histogram = backend.load_kernel_from_ptx(akaze_ptx, "akaze_gradient_histogram");
    histogram_percentile = backend.load_kernel_from_ptx(akaze_ptx, "akaze_histogram_percentile");
    conductivity = backend.load_kernel_from_ptx(akaze_ptx, "akaze_conductivity");
    diffusion_step = backend.load_kernel_from_ptx(akaze_ptx, "akaze_diffusion_step");
    hessian_response = backend.load_kernel_from_ptx(akaze_ptx, "akaze_hessian_response");
    nms_flags = backend.load_kernel_from_ptx(akaze_ptx, "akaze_nms_flags");
    akaze_scan_block = backend.load_kernel_from_ptx(akaze_ptx, "akaze_scan_block");
    akaze_scan_add = backend.load_kernel_from_ptx(akaze_ptx, "akaze_scan_add");
    scatter_candidates = backend.load_kernel_from_ptx(akaze_ptx, "akaze_scatter_candidates");
    select_scale_extrema = backend.load_kernel_from_ptx(akaze_ptx, "akaze_select_scale_extrema");
    prepare_feature_keys = backend.load_kernel_from_ptx(akaze_ptx, "akaze_prepare_feature_keys");
    radix_zero_flags = backend.load_kernel_from_ptx(akaze_ptx, "akaze_radix_zero_flags");
    radix_scatter = backend.load_kernel_from_ptx(akaze_ptx, "akaze_radix_scatter");
    emit_features = backend.load_kernel_from_ptx(akaze_ptx, "akaze_emit_features");
    describe_features = backend.load_kernel_from_ptx(akaze_ptx, "akaze_describe_features");

    const auto match_ptx = compiler.compile(kGpuMatchKernelSource, "reco_calibrate_gpu_match.cu");
    match_one_way = backend.load_kernel_from_ptx(match_ptx, "match_one_way");
    match_crosscheck = backend.load_kernel_from_ptx(match_ptx, "match_crosscheck_flags");
    match_scan_block = backend.load_kernel_from_ptx(match_ptx, "match_scan_block");
    match_scan_add = backend.load_kernel_from_ptx(match_ptx, "match_scan_add");
    match_scatter = backend.load_kernel_from_ptx(match_ptx, "match_scatter");
    match_sort = backend.load_kernel_from_ptx(match_ptx, "match_stable_sort");
  }

  [[nodiscard]] DeviceImage gaussian(const DeviceImage& source, float sigma) const {
    auto temporary = allocate_float_image(backend, source.width, source.height);
    auto output = allocate_float_image(backend, source.width, source.height);
    auto src = source.buffer.ptr();
    auto src_pitch = static_cast<std::uint64_t>(source.pitch);
    auto temp = temporary.buffer.ptr();
    auto temp_pitch = static_cast<std::uint64_t>(temporary.pitch);
    auto out = output.buffer.ptr();
    auto out_pitch = static_cast<std::uint64_t>(output.pitch);
    auto width = source.width;
    auto height = source.height;
    auto radius = static_cast<std::uint32_t>(std::ceil(sigma));
    launch(gaussian_x, image_launch(width, height), src, src_pitch, temp, temp_pitch, width, height,
           sigma, radius);
    launch(gaussian_y, image_launch(width, height), temp, temp_pitch, out, out_pitch, width, height,
           sigma, radius);
    return output;
  }

  void derivatives(const DeviceImage& source, std::uint32_t sigma_size, DeviceImage& lx,
                   DeviceImage& ly) const {
    auto src = source.buffer.ptr();
    auto src_pitch = static_cast<std::uint64_t>(source.pitch);
    auto lx_ptr = lx.buffer.ptr();
    auto lx_pitch = static_cast<std::uint64_t>(lx.pitch);
    auto ly_ptr = ly.buffer.ptr();
    auto ly_pitch = static_cast<std::uint64_t>(ly.pitch);
    auto width = source.width;
    auto height = source.height;
    // Rust AKAZE names the vertical response lx and the horizontal response ly.
    // Preserve that convention because it defines orientation and M-LDB bytes.
    launch(scharr, image_launch(width, height), src, src_pitch, ly_ptr, ly_pitch, lx_ptr, lx_pitch,
           width, height, sigma_size);
  }

  void one_derivative(const DeviceImage& source, std::uint32_t sigma_size, bool horizontal,
                      DeviceImage& output) const {
    auto src = source.buffer.ptr();
    auto src_pitch = static_cast<std::uint64_t>(source.pitch);
    reco::core::CudaDevicePtr zero = 0;
    auto out = output.buffer.ptr();
    auto out_pitch = static_cast<std::uint64_t>(output.pitch);
    std::uint64_t zero_pitch = 0;
    auto width = source.width;
    auto height = source.height;
    if (horizontal) {
      launch(scharr, image_launch(width, height), src, src_pitch, zero, zero_pitch, out, out_pitch,
             width, height, sigma_size);
    } else {
      launch(scharr, image_launch(width, height), src, src_pitch, out, out_pitch, zero, zero_pitch,
             width, height, sigma_size);
    }
  }

  void compute_contrast(const DeviceImage& lx, const DeviceImage& ly,
                        const reco::core::CudaDeviceBuffer& histogram,
                        const reco::core::CudaDeviceBuffer& maximum,
                        const reco::core::CudaDeviceBuffer& nonzero,
                        const reco::core::CudaDeviceBuffer& contrast) const {
    auto histogram_ptr = histogram.ptr();
    auto maximum_ptr = maximum.ptr();
    auto nonzero_ptr = nonzero.ptr();
    launch(clear_histogram,
           {.grid = {.x = 2, .y = 1, .z = 1}, .block = {.x = kLinearBlockSize, .y = 1, .z = 1}},
           histogram_ptr, maximum_ptr, nonzero_ptr);

    auto lx_ptr = lx.buffer.ptr();
    auto lx_pitch = static_cast<std::uint64_t>(lx.pitch);
    auto ly_ptr = ly.buffer.ptr();
    auto ly_pitch = static_cast<std::uint64_t>(ly.pitch);
    auto width = lx.width;
    auto height = lx.height;
    std::uint32_t mode = 0;
    launch(gradient_histogram, image_launch(width, height), lx_ptr, lx_pitch, ly_ptr, ly_pitch,
           width, height, histogram_ptr, maximum_ptr, nonzero_ptr, mode);
    mode = 1;
    launch(gradient_histogram, image_launch(width, height), lx_ptr, lx_pitch, ly_ptr, ly_pitch,
           width, height, histogram_ptr, maximum_ptr, nonzero_ptr, mode);

    auto percentile = kContrastPercentile;
    auto minimum = kMinimumContrast;
    auto contrast_ptr = contrast.ptr();
    launch(histogram_percentile,
           {.grid = {.x = 1, .y = 1, .z = 1}, .block = {.x = 1, .y = 1, .z = 1}}, histogram_ptr,
           maximum_ptr, nonzero_ptr, percentile, minimum, contrast_ptr);
  }

  void exclusive_scan(const reco::core::CudaKernel& scan_block_kernel,
                      const reco::core::CudaKernel& scan_add_kernel,
                      reco::core::CudaDevicePtr input, reco::core::CudaDevicePtr output,
                      std::uint32_t count, bool embedded_shared_memory,
                      std::vector<reco::core::CudaDeviceBuffer>& scratch) const {
    const auto blocks = divide_round_up(count, kLinearBlockSize);
    auto sums =
        backend.allocate(checked_multiply(blocks, sizeof(std::uint32_t), "GPU scan block sums"));
    auto sums_ptr = sums.ptr();
    auto count_arg = count;
    reco::core::CudaLaunchConfig config{
        .grid = {.x = blocks, .y = 1, .z = 1},
        .block = {.x = kLinearBlockSize, .y = 1, .z = 1},
        .shared_memory_bytes =
            embedded_shared_memory
                ? 0U
                : static_cast<std::uint32_t>(kLinearBlockSize * sizeof(std::uint32_t))};
    launch(scan_block_kernel, config, input, output, sums_ptr, count_arg);

    if (blocks > 1U) {
      auto offsets = backend.allocate(
          checked_multiply(blocks, sizeof(std::uint32_t), "GPU scan block offsets"));
      auto offsets_ptr = offsets.ptr();
      exclusive_scan(scan_block_kernel, scan_add_kernel, sums_ptr, offsets_ptr, blocks,
                     embedded_shared_memory, scratch);
      auto scan_count = count;
      if (embedded_shared_memory) {
        auto scan_block_size = kLinearBlockSize;
        launch(scan_add_kernel, linear_launch(count), output, offsets_ptr, scan_count,
               scan_block_size);
      } else {
        launch(scan_add_kernel, linear_launch(count), output, offsets_ptr, scan_count);
      }
      scratch.push_back(std::move(offsets));
    }
    scratch.push_back(std::move(sums));
  }

  reco::core::CudaBackend backend;
  reco::core::CudaKernel y_to_float;
  reco::core::CudaKernel triangle_vertical_y;
  reco::core::CudaKernel triangle_horizontal;
  reco::core::CudaKernel gaussian_x;
  reco::core::CudaKernel gaussian_y;
  reco::core::CudaKernel half_size;
  reco::core::CudaKernel scharr;
  reco::core::CudaKernel clear_histogram;
  reco::core::CudaKernel gradient_histogram;
  reco::core::CudaKernel histogram_percentile;
  reco::core::CudaKernel conductivity;
  reco::core::CudaKernel diffusion_step;
  reco::core::CudaKernel hessian_response;
  reco::core::CudaKernel nms_flags;
  reco::core::CudaKernel akaze_scan_block;
  reco::core::CudaKernel akaze_scan_add;
  reco::core::CudaKernel scatter_candidates;
  reco::core::CudaKernel select_scale_extrema;
  reco::core::CudaKernel prepare_feature_keys;
  reco::core::CudaKernel radix_zero_flags;
  reco::core::CudaKernel radix_scatter;
  reco::core::CudaKernel emit_features;
  reco::core::CudaKernel describe_features;
  reco::core::CudaKernel match_one_way;
  reco::core::CudaKernel match_crosscheck;
  reco::core::CudaKernel match_scan_block;
  reco::core::CudaKernel match_scan_add;
  reco::core::CudaKernel match_scatter;
  reco::core::CudaKernel match_sort;
};

GpuFeatureSet::GpuFeatureSet() = default;
GpuFeatureSet::~GpuFeatureSet() = default;
GpuFeatureSet::GpuFeatureSet(GpuFeatureSet&&) noexcept = default;
GpuFeatureSet& GpuFeatureSet::operator=(GpuFeatureSet&&) noexcept = default;
GpuFeatureSet::GpuFeatureSet(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GpuFeatureView GpuFeatureSet::view() const {
  if (impl_ == nullptr) {
    return {};
  }
  return {.points = impl_->points.ptr(),
          .descriptors = impl_->descriptors.ptr(),
          .count = impl_->count.ptr(),
          .capacity = impl_->capacity};
}

std::uint32_t GpuFeatureSet::capacity() const { return impl_ != nullptr ? impl_->capacity : 0U; }

GpuAkazePipeline::GpuAkazePipeline(reco::core::CudaBackend& backend)
    : impl_(std::make_unique<Impl>(backend)) {}
GpuAkazePipeline::~GpuAkazePipeline() = default;
GpuAkazePipeline::GpuAkazePipeline(GpuAkazePipeline&&) noexcept = default;
GpuAkazePipeline& GpuAkazePipeline::operator=(GpuAkazePipeline&&) noexcept = default;

GpuFeatureSet GpuAkazePipeline::detect(const GpuGrayFrame& frame,
                                       const GpuAkazeConfig& config) const {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot use a moved-from GPU AKAZE pipeline");
  }
  validate_frame(frame);
  validate_config(config);
  auto& backend = impl_->backend;
  backend.ensure_primary_context();

  const auto geometry = detection_geometry(frame, config);
  const auto specs = make_evolution_specs(geometry, config);
  const auto& final_spec = specs.back();
  const std::uint32_t total_pixels = checked_u32(
      static_cast<std::size_t>(final_spec.pixel_base) +
          checked_multiply(final_spec.width, final_spec.height, "GPU AKAZE final level"),
      "GPU AKAZE total pixels");

  auto base = allocate_float_image(backend, geometry.detect_width, geometry.detect_height);
  auto src = frame.ptr;
  auto src_pitch = static_cast<std::uint64_t>(frame.pitch);
  auto src_width = frame.width;
  auto src_height = frame.height;
  auto dst = base.buffer.ptr();
  auto dst_pitch = static_cast<std::uint64_t>(base.pitch);
  auto dst_width = base.width;
  auto dst_height = base.height;
  auto crop_x = geometry.crop_x;
  auto crop_y = geometry.crop_y;
  auto crop_width = geometry.crop_width;
  auto crop_height = geometry.crop_height;
  std::uint32_t limited_range = frame.color_range == reco::core::YuvColorRange::Limited ? 1U : 0U;
  if (crop_width == dst_width && crop_height == dst_height) {
    launch(impl_->y_to_float, image_launch(dst_width, dst_height), src, src_pitch, src_width,
           src_height, dst, dst_pitch, dst_width, dst_height, crop_x, crop_y, crop_width,
           crop_height, limited_range);
  } else {
    auto vertical = allocate_float_image(backend, crop_width, dst_height);
    auto vertical_ptr = vertical.buffer.ptr();
    auto vertical_pitch = static_cast<std::uint64_t>(vertical.pitch);
    launch(impl_->triangle_vertical_y, image_launch(crop_width, dst_height), src, src_pitch,
           vertical_ptr, vertical_pitch, crop_x, crop_y, crop_width, crop_height, dst_height,
           limited_range);
    launch(impl_->triangle_horizontal, image_launch(dst_width, dst_height), vertical_ptr,
           vertical_pitch, crop_width, dst, dst_pitch, dst_width, dst_height);
  }

  auto histogram = backend.allocate(300U * sizeof(std::uint32_t));
  auto maximum = backend.allocate(sizeof(std::uint32_t));
  auto nonzero = backend.allocate(sizeof(std::uint32_t));
  auto contrast = backend.allocate(sizeof(float));
  bool contrast_ready = false;
  std::vector<EvolutionLevel> levels;
  levels.reserve(specs.size());

  for (std::size_t level_index = 0; level_index < specs.size(); ++level_index) {
    const auto& spec = specs[level_index];
    DeviceImage lt;
    if (level_index == 0) {
      lt = impl_->gaussian(base, static_cast<float>(kBaseScaleOffset));
    } else if (spec.octave == specs[level_index - 1U].octave) {
      lt = allocate_float_image(backend, spec.width, spec.height);
      copy_image(backend, levels.back().lt, lt);
    } else {
      lt = allocate_float_image(backend, spec.width, spec.height);
      auto previous = levels.back().lt.buffer.ptr();
      auto previous_pitch = static_cast<std::uint64_t>(levels.back().lt.pitch);
      auto previous_width = levels.back().lt.width;
      auto previous_height = levels.back().lt.height;
      auto current = lt.buffer.ptr();
      auto current_pitch = static_cast<std::uint64_t>(lt.pitch);
      auto current_width = lt.width;
      auto current_height = lt.height;
      launch(impl_->half_size, image_launch(current_width, current_height), previous,
             previous_pitch, previous_width, previous_height, current, current_pitch, current_width,
             current_height);
    }

    DeviceImage smooth;
    if (level_index == 0) {
      smooth = allocate_float_image(backend, spec.width, spec.height);
      copy_image(backend, lt, smooth);
    } else {
      smooth = impl_->gaussian(lt, 1.0F);
    }

    DeviceImage flow_lx;
    DeviceImage flow_ly;
    if (level_index > 0) {
      flow_lx = allocate_float_image(backend, spec.width, spec.height);
      flow_ly = allocate_float_image(backend, spec.width, spec.height);
      impl_->derivatives(smooth, 1U, flow_lx, flow_ly);
      if (!contrast_ready) {
        impl_->compute_contrast(flow_lx, flow_ly, histogram, maximum, nonzero, contrast);
        contrast_ready = true;
      }

      auto conductivity = allocate_float_image(backend, spec.width, spec.height);
      auto lx_ptr = flow_lx.buffer.ptr();
      auto lx_pitch = static_cast<std::uint64_t>(flow_lx.pitch);
      auto ly_ptr = flow_ly.buffer.ptr();
      auto ly_pitch = static_cast<std::uint64_t>(flow_ly.pitch);
      auto conductivity_ptr = conductivity.buffer.ptr();
      auto conductivity_pitch = static_cast<std::uint64_t>(conductivity.pitch);
      auto width = spec.width;
      auto height = spec.height;
      auto contrast_ptr = contrast.ptr();
      const auto contrast_scale = std::pow(0.75F, static_cast<float>(spec.octave));
      launch(impl_->conductivity, image_launch(width, height), lx_ptr, lx_pitch, ly_ptr, ly_pitch,
             conductivity_ptr, conductivity_pitch, width, height, contrast_ptr, contrast_scale);

      auto diffusion_target = allocate_float_image(backend, spec.width, spec.height);
      for (float step_size : fed_steps(spec.time - specs[level_index - 1U].time)) {
        auto diffusion_source = lt.buffer.ptr();
        auto diffusion_source_pitch = static_cast<std::uint64_t>(lt.pitch);
        auto diffusion_output = diffusion_target.buffer.ptr();
        auto diffusion_output_pitch = static_cast<std::uint64_t>(diffusion_target.pitch);
        launch(impl_->diffusion_step, image_launch(width, height), diffusion_source,
               diffusion_source_pitch, conductivity_ptr, conductivity_pitch, diffusion_output,
               diffusion_output_pitch, width, height, step_size);
        std::swap(lt, diffusion_target);
      }
    }

    if (!contrast_ready && level_index + 1U == specs.size()) {
      auto contrast_smooth = impl_->gaussian(smooth, 1.0F);
      auto contrast_lx = allocate_float_image(backend, spec.width, spec.height);
      auto contrast_ly = allocate_float_image(backend, spec.width, spec.height);
      impl_->derivatives(contrast_smooth, 1U, contrast_lx, contrast_ly);
      impl_->compute_contrast(contrast_lx, contrast_ly, histogram, maximum, nonzero, contrast);
      contrast_ready = true;
    }

    auto detector_lx = allocate_float_image(backend, spec.width, spec.height);
    auto detector_ly = allocate_float_image(backend, spec.width, spec.height);
    const auto octave_ratio = static_cast<double>(std::uint32_t{1} << spec.octave);
    const auto sigma_size =
        static_cast<std::uint32_t>(std::round(spec.sigma * kDerivativeFactor / octave_ratio));
    impl_->derivatives(smooth, sigma_size, detector_lx, detector_ly);
    auto lxx = allocate_float_image(backend, spec.width, spec.height);
    auto lyy = allocate_float_image(backend, spec.width, spec.height);
    auto lxy = allocate_float_image(backend, spec.width, spec.height);
    impl_->one_derivative(detector_lx, sigma_size, true, lxx);
    impl_->one_derivative(detector_ly, sigma_size, false, lyy);
    impl_->one_derivative(detector_lx, sigma_size, false, lxy);
    auto response = allocate_float_image(backend, spec.width, spec.height);
    auto lxx_ptr = lxx.buffer.ptr();
    auto lxx_pitch = static_cast<std::uint64_t>(lxx.pitch);
    auto lyy_ptr = lyy.buffer.ptr();
    auto lyy_pitch = static_cast<std::uint64_t>(lyy.pitch);
    auto lxy_ptr = lxy.buffer.ptr();
    auto lxy_pitch = static_cast<std::uint64_t>(lxy.pitch);
    auto response_ptr = response.buffer.ptr();
    auto response_pitch = static_cast<std::uint64_t>(response.pitch);
    auto width = spec.width;
    auto height = spec.height;
    auto sigma_size_arg = sigma_size;
    launch(impl_->hessian_response, image_launch(width, height), lxx_ptr, lxx_pitch, lyy_ptr,
           lyy_pitch, lxy_ptr, lxy_pitch, response_ptr, response_pitch, width, height,
           sigma_size_arg);
    levels.push_back({.spec = spec,
                      .lt = std::move(lt),
                      .lx = std::move(detector_lx),
                      .ly = std::move(detector_ly),
                      .response = std::move(response)});
  }

  auto flags = backend.allocate(
      checked_multiply(total_pixels, sizeof(std::uint32_t), "GPU AKAZE NMS flags"));
  auto offsets = backend.allocate(
      checked_multiply(total_pixels, sizeof(std::uint32_t), "GPU AKAZE NMS offsets"));
  for (const auto& level : levels) {
    auto response = level.response.buffer.ptr();
    auto response_pitch = static_cast<std::uint64_t>(level.response.pitch);
    auto width = level.spec.width;
    auto height = level.spec.height;
    auto threshold = config.threshold;
    auto level_flags =
        flags.ptr() + static_cast<std::uint64_t>(level.spec.pixel_base) * sizeof(std::uint32_t);
    launch(impl_->nms_flags, image_launch(width, height), response, response_pitch, width, height,
           threshold, level_flags);
  }

  std::vector<reco::core::CudaDeviceBuffer> scan_scratch;
  impl_->exclusive_scan(impl_->akaze_scan_block, impl_->akaze_scan_add, flags.ptr(), offsets.ptr(),
                        total_pixels, false, scan_scratch);
  const auto last_byte = static_cast<std::uint64_t>(total_pixels - 1U) * sizeof(std::uint32_t);
  const auto candidate_count = download_u32(backend, offsets.ptr() + last_byte) +
                               download_u32(backend, flags.ptr() + last_byte);
  if (candidate_count > total_pixels) {
    throw std::runtime_error("GPU AKAZE candidate count exceeded its deterministic capacity");
  }

  auto result = std::make_unique<GpuFeatureSet::Impl>();
  result->capacity = config.max_keypoints;
  const auto allocated_capacity = std::max(config.max_keypoints, 1U);
  result->points = backend.allocate(
      checked_multiply(allocated_capacity, sizeof(GpuFeaturePoint), "GPU AKAZE features"));
  result->descriptors = backend.allocate(
      checked_multiply(allocated_capacity, kDescriptorBytes, "GPU AKAZE descriptors"));
  result->count = backend.allocate(sizeof(std::uint32_t));
  backend.memset_d8(result->count, 0);

  if (candidate_count != 0U) {
    auto candidates = backend.allocate(
        checked_multiply(candidate_count, kCandidateBytes, "GPU AKAZE candidates"));
    for (const auto& level : levels) {
      auto level_flags =
          flags.ptr() + static_cast<std::uint64_t>(level.spec.pixel_base) * sizeof(std::uint32_t);
      auto level_offsets =
          offsets.ptr() + static_cast<std::uint64_t>(level.spec.pixel_base) * sizeof(std::uint32_t);
      auto response = level.response.buffer.ptr();
      auto response_pitch = static_cast<std::uint64_t>(level.response.pitch);
      auto width = level.spec.width;
      auto height = level.spec.height;
      auto feature_size = static_cast<float>(level.spec.sigma * kDerivativeFactor);
      auto octave = level.spec.octave;
      auto level_index = checked_u32(&level - levels.data(), "GPU AKAZE level index");
      auto base_order = level.spec.pixel_base;
      auto candidates_ptr = candidates.ptr();
      auto capacity = candidate_count;
      reco::core::CudaDevicePtr overflow_ptr = 0;
      const auto pixel_count = checked_u32(
          checked_multiply(width, height, "GPU AKAZE level pixels"), "GPU AKAZE level pixels");
      launch(impl_->scatter_candidates, linear_launch(pixel_count), level_flags, level_offsets,
             response, response_pitch, width, height, feature_size, octave, level_index, base_order,
             candidates_ptr, capacity, overflow_ptr);
    }

    std::vector<GpuSelectionLevel> host_selection_levels;
    host_selection_levels.reserve(levels.size());
    for (const auto& level : levels) {
      host_selection_levels.push_back({.pixel_base = level.spec.pixel_base,
                                       .width = level.spec.width,
                                       .height = level.spec.height,
                                       .octave = level.spec.octave});
    }
    auto selection_levels = backend.allocate(checked_multiply(
        host_selection_levels.size(), sizeof(GpuSelectionLevel), "GPU AKAZE selection levels"));
    upload_bytes(backend, host_selection_levels.data(),
                 host_selection_levels.size() * sizeof(GpuSelectionLevel), selection_levels);

    const auto candidate_bytes =
        checked_multiply(candidate_count, sizeof(std::uint32_t), "GPU AKAZE candidate selection");
    auto selected_flags = backend.allocate(candidate_bytes);
    auto cache_indices = backend.allocate(candidate_bytes);
    auto cache_cells = backend.allocate(
        checked_multiply(total_pixels, sizeof(std::uint32_t), "GPU AKAZE selection cells"));
    auto cache_count = backend.allocate(sizeof(std::uint32_t));
    backend.memset_d8(selected_flags, 0U);
    backend.memset_d8(cache_cells, 0xffU);
    backend.memset_d8(cache_count, 0U);
    auto keys_a = backend.allocate(candidate_bytes);
    auto keys_b = backend.allocate(candidate_bytes);
    auto indices_a = backend.allocate(candidate_bytes);
    auto indices_b = backend.allocate(candidate_bytes);
    auto radix_flags = backend.allocate(candidate_bytes);
    auto radix_offsets = backend.allocate(candidate_bytes);
    auto valid_count = backend.allocate(sizeof(std::uint32_t));
    backend.memset_d8(valid_count, 0);

    auto candidates_ptr = candidates.ptr();
    auto candidate_count_arg = candidate_count;
    auto selection_levels_ptr = selection_levels.ptr();
    auto level_count = checked_u32(levels.size(), "GPU AKAZE selection level count");
    auto selected_flags_ptr = selected_flags.ptr();
    auto cache_indices_ptr = cache_indices.ptr();
    auto cache_cells_ptr = cache_cells.ptr();
    auto cache_count_ptr = cache_count.ptr();
    launch(impl_->select_scale_extrema,
           {.grid = {.x = 1, .y = 1, .z = 1}, .block = {.x = 1, .y = 1, .z = 1}}, candidates_ptr,
           candidate_count_arg, selection_levels_ptr, level_count, cache_indices_ptr,
           cache_cells_ptr, cache_count_ptr, selected_flags_ptr);

    auto inverse_detection_scale = 1.0F / geometry.detector_scale;
    auto crop_x_float = static_cast<float>(geometry.crop_x);
    auto crop_y_float = static_cast<float>(geometry.crop_y);
    auto use_roi = config.use_region ? 1U : 0U;
    auto roi_x_min = config.region.x_min * frame.width;
    auto roi_x_max = config.region.x_max * frame.width;
    auto roi_y_min = config.region.y_min * frame.height;
    auto roi_y_max = config.region.y_max * frame.height;
    auto border_y = frame.ptr;
    auto border_pitch = static_cast<std::uint64_t>(frame.pitch);
    auto border_width = frame.width;
    auto border_height = frame.height;
    auto border_margin = config.border_margin;
    std::uint32_t black_threshold =
        frame.color_range == reco::core::YuvColorRange::Limited ? 25U : 10U;
    auto keys_input = keys_a.ptr();
    auto keys_output = keys_b.ptr();
    auto indices_input = indices_a.ptr();
    auto indices_output = indices_b.ptr();
    auto valid_count_ptr = valid_count.ptr();
    launch(impl_->prepare_feature_keys, linear_launch(candidate_count), candidates_ptr,
           candidate_count_arg, selected_flags_ptr, keys_input, indices_input, valid_count_ptr,
           inverse_detection_scale, crop_x_float, crop_y_float, use_roi, roi_x_min, roi_x_max,
           roi_y_min, roi_y_max, border_y, border_pitch, border_width, border_height, border_margin,
           black_threshold);

    std::vector<reco::core::CudaDeviceBuffer> radix_scan_scratch;
    auto radix_flags_ptr = radix_flags.ptr();
    auto radix_offsets_ptr = radix_offsets.ptr();
    for (std::uint32_t bit = 0; bit < 32U; ++bit) {
      auto bit_arg = bit;
      launch(impl_->radix_zero_flags, linear_launch(candidate_count), keys_input,
             candidate_count_arg, bit_arg, radix_flags_ptr);
      impl_->exclusive_scan(impl_->akaze_scan_block, impl_->akaze_scan_add, radix_flags_ptr,
                            radix_offsets_ptr, candidate_count, false, radix_scan_scratch);
      launch(impl_->radix_scatter, linear_launch(candidate_count), keys_input, indices_input,
             radix_flags_ptr, radix_offsets_ptr, candidate_count_arg, keys_output, indices_output);
      std::swap(keys_input, keys_output);
      std::swap(indices_input, indices_output);
    }

    auto features_ptr = result->points.ptr();
    auto descriptor_positions = backend.allocate(
        checked_multiply(checked_multiply(allocated_capacity, 2U, "GPU AKAZE descriptor positions"),
                         sizeof(float), "GPU AKAZE descriptor positions"));
    auto descriptor_positions_ptr = descriptor_positions.ptr();
    auto max_features = config.max_keypoints;
    auto feature_count = result->count.ptr();
    launch(impl_->emit_features, linear_launch(max_features), candidates_ptr, indices_input,
           valid_count_ptr, features_ptr, descriptor_positions_ptr, max_features, feature_count,
           inverse_detection_scale, crop_x_float, crop_y_float, use_roi, roi_x_min, roi_x_max,
           roi_y_min, roi_y_max, border_y, border_pitch, border_width, border_height, border_margin,
           black_threshold);

    if (config.max_keypoints != 0U) {
      std::vector<GpuLevelView> host_views;
      host_views.reserve(levels.size());
      for (const auto& level : levels) {
        host_views.push_back({.lt = level.lt.buffer.ptr(),
                              .lx = level.lx.buffer.ptr(),
                              .ly = level.ly.buffer.ptr(),
                              .lt_pitch = level.lt.pitch,
                              .lx_pitch = level.lx.pitch,
                              .ly_pitch = level.ly.pitch,
                              .width = level.spec.width,
                              .height = level.spec.height,
                              .octave = level.spec.octave,
                              .reserved = 0});
      }
      auto device_views = backend.allocate(
          checked_multiply(host_views.size(), sizeof(GpuLevelView), "GPU AKAZE level views"));
      upload_bytes(backend, host_views.data(), host_views.size() * sizeof(GpuLevelView),
                   device_views);
      auto level_views = device_views.ptr();
      auto level_count = checked_u32(host_views.size(), "GPU AKAZE level count");
      auto descriptors = result->descriptors.ptr();
      std::uint64_t descriptor_pitch = kDescriptorBytes;
      launch(impl_->describe_features, linear_launch(config.max_keypoints), features_ptr,
             descriptor_positions_ptr, feature_count, level_views, level_count, descriptors,
             descriptor_pitch);
      impl_->describe_features.synchronize();
    } else {
      impl_->emit_features.synchronize();
    }
  }
  return GpuFeatureSet(std::move(result));
}

std::vector<GpuMatchedPoint> GpuAkazePipeline::match(const GpuFeatureView& left,
                                                     const GpuFeatureView& right,
                                                     double lowe_ratio) const {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot use a moved-from GPU AKAZE pipeline");
  }
  validate_feature_view(left, "left");
  validate_feature_view(right, "right");
  if (!std::isfinite(lowe_ratio) || !(lowe_ratio > 0.0) || lowe_ratio > 1.0) {
    throw std::invalid_argument("GPU matcher Lowe ratio must be finite and in (0, 1]");
  }
  if (left.capacity == 0 || right.capacity == 0) {
    return {};
  }

  auto& backend = impl_->backend;
  backend.ensure_primary_context();
  struct DirectionBuffers {
    reco::core::CudaDeviceBuffer best;
    reco::core::CudaDeviceBuffer best_distance;
    reco::core::CudaDeviceBuffer second_distance;
    reco::core::CudaDeviceBuffer accepted;
  };
  const auto allocate_direction = [&backend](std::uint32_t capacity) {
    const auto bytes = checked_multiply(capacity, sizeof(std::uint32_t), "GPU match direction");
    return DirectionBuffers{.best = backend.allocate(bytes),
                            .best_distance = backend.allocate(bytes),
                            .second_distance = backend.allocate(bytes),
                            .accepted = backend.allocate(bytes)};
  };
  auto forward = allocate_direction(left.capacity);
  auto backward = allocate_direction(right.capacity);

  const auto launch_direction = [&](const GpuFeatureView& query, const GpuFeatureView& train,
                                    DirectionBuffers& output) {
    auto query_descriptors = query.descriptors;
    auto query_count = query.count;
    auto query_capacity = query.capacity;
    auto train_descriptors = train.descriptors;
    auto train_count = train.count;
    auto train_capacity = train.capacity;
    auto ratio = lowe_ratio;
    auto best = output.best.ptr();
    auto best_distance = output.best_distance.ptr();
    auto second_distance = output.second_distance.ptr();
    auto accepted = output.accepted.ptr();
    launch(impl_->match_one_way, linear_launch(query.capacity), query_descriptors, query_count,
           query_capacity, train_descriptors, train_count, train_capacity, ratio, best,
           best_distance, second_distance, accepted);
  };
  launch_direction(left, right, forward);
  launch_direction(right, left, backward);

  const auto left_bytes = checked_multiply(left.capacity, sizeof(std::uint32_t), "GPU match flags");
  auto flags = backend.allocate(left_bytes);
  auto offsets = backend.allocate(left_bytes);
  auto left_best = forward.best.ptr();
  auto left_accepted = forward.accepted.ptr();
  auto right_best = backward.best.ptr();
  auto right_accepted = backward.accepted.ptr();
  auto left_count = left.count;
  auto left_capacity = left.capacity;
  auto right_count = right.count;
  auto right_capacity = right.capacity;
  auto flags_ptr = flags.ptr();
  launch(impl_->match_crosscheck, linear_launch(left.capacity), left_best, left_accepted,
         right_best, right_accepted, left_count, left_capacity, right_count, right_capacity,
         flags_ptr);

  std::vector<reco::core::CudaDeviceBuffer> scan_scratch;
  impl_->exclusive_scan(impl_->match_scan_block, impl_->match_scan_add, flags.ptr(), offsets.ptr(),
                        left.capacity, true, scan_scratch);
  const auto compact_capacity = std::min(left.capacity, right.capacity);
  auto compact = backend.allocate(
      checked_multiply(compact_capacity, sizeof(GpuCompactMatch), "GPU compact matches"));
  auto sorted = backend.allocate(
      checked_multiply(compact_capacity, sizeof(GpuCompactMatch), "GPU sorted matches"));
  auto compact_count = backend.allocate(sizeof(std::uint32_t));
  backend.memset_d8(compact_count, 0);
  auto left_points = left.points;
  auto right_points = right.points;
  auto left_distances = forward.best_distance.ptr();
  auto offsets_ptr = offsets.ptr();
  auto compact_ptr = compact.ptr();
  auto compact_count_ptr = compact_count.ptr();
  launch(impl_->match_scatter, linear_launch(left.capacity), left_points, left_count, left_capacity,
         right_points, right_count, right_capacity, left_best, left_distances, flags_ptr,
         offsets_ptr, compact_ptr, compact_capacity, compact_count_ptr);
  auto sorted_ptr = sorted.ptr();
  launch(impl_->match_sort, linear_launch(compact_capacity), compact_ptr, compact_count_ptr,
         compact_capacity, sorted_ptr);

  const auto match_count = download_u32(backend, compact_count.ptr());
  if (match_count > compact_capacity) {
    throw std::runtime_error("GPU matcher result count exceeded its compact capacity");
  }
  if (match_count == 0) {
    return {};
  }
  std::vector<GpuCompactMatch> host_matches(match_count);
  const auto bytes = checked_multiply(match_count, sizeof(GpuCompactMatch), "GPU match readback");
  backend.copy_device_to_host_2d({.dst = host_matches.data(),
                                  .dst_pitch = bytes,
                                  .src = sorted.ptr(),
                                  .src_pitch = bytes,
                                  .width_bytes = bytes,
                                  .height = 1});
  std::vector<GpuMatchedPoint> result;
  result.reserve(host_matches.size());
  for (const auto& match : host_matches) {
    result.push_back(
        {.left = {.x = match.left_x, .y = match.left_y, .response = match.left_response},
         .right = {.x = match.right_x, .y = match.right_y, .response = match.right_response},
         .distance = match.distance});
  }
  return result;
}

std::vector<GpuMatchedPoint>
GpuAkazePipeline::detect_and_match(const GpuGrayFrame& left, const GpuGrayFrame& right,
                                   const GpuAkazeConfig& config) const {
  auto left_features = detect(left, config);
  auto right_features = detect(right, config);
  return match(left_features.view(), right_features.view(), config.lowe_ratio);
}

} // namespace reco::calibrate
