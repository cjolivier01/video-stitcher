// Runtime-compiled CUDA source for the GPU-resident AKAZE pipeline.
//
// All pointer parameters cross the driver API as 64-bit device addresses. Pitches
// are byte pitches. The host orchestration must launch the histogram and scan
// kernels in the documented multi-pass order and must validate dimensions,
// capacities, shifts, and allocation sizes before launch.
constexpr char kGpuAkazeKernelSource[] = R"cuda(
template <typename T>
__device__ __forceinline__ T* akaze_pointer(unsigned long long address) {
  return reinterpret_cast<T*>(address);
}

template <typename T>
__device__ __forceinline__ const T* akaze_const_pointer(unsigned long long address) {
  return reinterpret_cast<const T*>(address);
}

template <typename T>
__device__ __forceinline__ T* akaze_row(
    unsigned long long address, unsigned long long pitch, unsigned int y) {
  unsigned char* base = akaze_pointer<unsigned char>(address);
  return reinterpret_cast<T*>(base + (unsigned long long)y * pitch);
}

template <typename T>
__device__ __forceinline__ const T* akaze_const_row(
    unsigned long long address, unsigned long long pitch, unsigned int y) {
  const unsigned char* base = akaze_const_pointer<unsigned char>(address);
  return reinterpret_cast<const T*>(base + (unsigned long long)y * pitch);
}

__device__ __forceinline__ int akaze_clamp_int(int value, int lower, int upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

__device__ __forceinline__ float akaze_load_replicated(
    unsigned long long image,
    unsigned long long pitch,
    unsigned int width,
    unsigned int height,
    int x,
    int y) {
  const int cx = akaze_clamp_int(x, 0, (int)width - 1);
  const int cy = akaze_clamp_int(y, 0, (int)height - 1);
  return akaze_const_row<float>(image, pitch, (unsigned int)cy)[cx];
}

__device__ __forceinline__ float akaze_normalize_y(
    unsigned char value, unsigned int limited_range) {
  if (limited_range != 0U) {
    if (value <= 16U) {
      return 0.0f;
    }
    if (value >= 235U) {
      return 1.0f;
    }
    return ((float)value - 16.0f) * (1.0f / 219.0f);
  }
  if (value == 0U) {
    return 0.0f;
  }
  if (value == 255U) {
    return 1.0f;
  }
  return (float)value * (1.0f / 255.0f);
}

// The layout intentionally matches reco::calibrate::GpuFeaturePoint.
struct AkazeFeaturePoint {
  float x;
  float y;
  float response;
  float size;
  unsigned int octave;
  unsigned int level;
};

// Internal row-major candidate record. response_patch is ordered y-major and
// contains the 3x3 detector-response neighborhood used for subpixel refinement.
struct AkazeCandidate {
  float x;
  float y;
  float response;
  float size;
  unsigned int octave;
  unsigned int level;
  unsigned int width;
  unsigned int height;
  unsigned int order;
  unsigned int valid;
  float response_patch[9];
};

// Host mirror: six uint64 fields followed by four uint32 fields (64 bytes).
struct AkazeLevelView {
  unsigned long long lt;
  unsigned long long lx;
  unsigned long long ly;
  unsigned long long lt_pitch;
  unsigned long long lx_pitch;
  unsigned long long ly_pitch;
  unsigned int width;
  unsigned int height;
  unsigned int octave;
  unsigned int reserved;
};

// Spatial lookup metadata for deterministic scale-space suppression. pixel_base
// addresses the level in the global NMS flag/offset arrays.
struct AkazeSelectionLevel {
  unsigned int pixel_base;
  unsigned int width;
  unsigned int height;
  unsigned int octave;
};

static_assert(sizeof(AkazeFeaturePoint) == 24, "feature point ABI must be 24 bytes");
static_assert(sizeof(AkazeCandidate) == 76, "candidate ABI must be 76 bytes");
static_assert(sizeof(AkazeLevelView) == 64, "level view ABI must be 64 bytes");
static_assert(sizeof(AkazeSelectionLevel) == 16, "selection level ABI must be 16 bytes");

extern "C" __global__ void akaze_y_to_float(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned int src_width,
    unsigned int src_height,
    unsigned long long dst,
    unsigned long long dst_pitch,
    unsigned int dst_width,
    unsigned int dst_height,
    unsigned int crop_x,
    unsigned int crop_y,
    unsigned int crop_width,
    unsigned int crop_height,
    unsigned int limited_range) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height) {
    return;
  }

  const float scale_x = (float)crop_width / (float)dst_width;
  const float scale_y = (float)crop_height / (float)dst_height;
  float sx = (float)crop_x + ((float)x + 0.5f) * scale_x - 0.5f;
  float sy = (float)crop_y + ((float)y + 0.5f) * scale_y - 0.5f;
  const float max_x = (float)(crop_x + crop_width - 1U);
  const float max_y = (float)(crop_y + crop_height - 1U);
  sx = sx < (float)crop_x ? (float)crop_x : (sx > max_x ? max_x : sx);
  sy = sy < (float)crop_y ? (float)crop_y : (sy > max_y ? max_y : sy);

  unsigned int x0 = (unsigned int)floorf(sx);
  unsigned int y0 = (unsigned int)floorf(sy);
  unsigned int x1 = x0 + 1U;
  unsigned int y1 = y0 + 1U;
  x1 = x1 < src_width ? x1 : src_width - 1U;
  y1 = y1 < src_height ? y1 : src_height - 1U;
  const float tx = sx - (float)x0;
  const float ty = sy - (float)y0;
  const unsigned char* row0 = akaze_const_row<unsigned char>(src, src_pitch, y0);
  const unsigned char* row1 = akaze_const_row<unsigned char>(src, src_pitch, y1);
  const float upper =
      (1.0f - tx) * akaze_normalize_y(row0[x0], limited_range) +
      tx * akaze_normalize_y(row0[x1], limited_range);
  const float lower =
      (1.0f - tx) * akaze_normalize_y(row1[x0], limited_range) +
      tx * akaze_normalize_y(row1[x1], limited_range);
  akaze_row<float>(dst, dst_pitch, y)[x] = (1.0f - ty) * upper + ty * lower;
}

// First pass of image::imageops::FilterType::Triangle downscaling. The filter
// support expands with the reduction ratio, providing the antialiasing that a
// single bilinear sample lacks. Source values are normalized before filtering
// so full- and limited-range luma retain identical endpoint behavior.
extern "C" __global__ void akaze_triangle_vertical_y(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned long long dst,
    unsigned long long dst_pitch,
    unsigned int crop_x,
    unsigned int crop_y,
    unsigned int crop_width,
    unsigned int crop_height,
    unsigned int dst_height,
    unsigned int limited_range) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= crop_width || y >= dst_height) {
    return;
  }

  const float ratio = (float)crop_height / (float)dst_height;
  const float sample_ratio = ratio < 1.0f ? 1.0f : ratio;
  const float input = ((float)y + 0.5f) * ratio;
  int first = (int)floorf(input - sample_ratio);
  int last = (int)ceilf(input + sample_ratio);
  first = akaze_clamp_int(first, 0, (int)crop_height - 1);
  last = akaze_clamp_int(last, first + 1, (int)crop_height);
  const float center = input - 0.5f;
  float weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  for (int source_y = first; source_y < last; ++source_y) {
    const float distance = fabsf(((float)source_y - center) / sample_ratio);
    const float weight = distance < 1.0f ? 1.0f - distance : 0.0f;
    const unsigned char value = akaze_const_row<unsigned char>(
        src, src_pitch, crop_y + (unsigned int)source_y)[crop_x + x];
    weighted_sum += weight * akaze_normalize_y(value, limited_range);
    weight_sum += weight;
  }
  akaze_row<float>(dst, dst_pitch, y)[x] = weighted_sum / weight_sum;
}

// Horizontal Triangle pass. Quantizing the final normalized sample to an
// 8-bit step matches DynamicImage::resize_exact before AKAZE converts the
// resized grayscale image to unit floats.
extern "C" __global__ void akaze_triangle_horizontal(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned int src_width,
    unsigned long long dst,
    unsigned long long dst_pitch,
    unsigned int dst_width,
    unsigned int height) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= height) {
    return;
  }

  const float ratio = (float)src_width / (float)dst_width;
  const float sample_ratio = ratio < 1.0f ? 1.0f : ratio;
  const float input = ((float)x + 0.5f) * ratio;
  int first = (int)floorf(input - sample_ratio);
  int last = (int)ceilf(input + sample_ratio);
  first = akaze_clamp_int(first, 0, (int)src_width - 1);
  last = akaze_clamp_int(last, first + 1, (int)src_width);
  const float center = input - 0.5f;
  float weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  const float* row = akaze_const_row<float>(src, src_pitch, y);
  for (int source_x = first; source_x < last; ++source_x) {
    const float distance = fabsf(((float)source_x - center) / sample_ratio);
    const float weight = distance < 1.0f ? 1.0f - distance : 0.0f;
    weighted_sum += weight * row[source_x];
    weight_sum += weight;
  }
  const float filtered = weighted_sum / weight_sum;
  const float quantized = roundf(fminf(fmaxf(filtered, 0.0f), 1.0f) * 255.0f) *
      (1.0f / 255.0f);
  akaze_row<float>(dst, dst_pitch, y)[x] = quantized;
}

extern "C" __global__ void akaze_gaussian_x(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned long long dst,
    unsigned long long dst_pitch,
    unsigned int width,
    unsigned int height,
    float sigma,
    unsigned int radius) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const float inverse_two_sigma_squared = 0.5f / (sigma * sigma);
  float weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  for (int tap = -(int)radius; tap <= (int)radius; ++tap) {
    const float weight = expf(-(float)(tap * tap) * inverse_two_sigma_squared);
    weighted_sum += weight * akaze_load_replicated(
        src, src_pitch, width, height, (int)x + tap, (int)y);
    weight_sum += weight;
  }
  akaze_row<float>(dst, dst_pitch, y)[x] = weighted_sum / weight_sum;
}

extern "C" __global__ void akaze_gaussian_y(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned long long dst,
    unsigned long long dst_pitch,
    unsigned int width,
    unsigned int height,
    float sigma,
    unsigned int radius) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const float inverse_two_sigma_squared = 0.5f / (sigma * sigma);
  float weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  for (int tap = -(int)radius; tap <= (int)radius; ++tap) {
    const float weight = expf(-(float)(tap * tap) * inverse_two_sigma_squared);
    weighted_sum += weight * akaze_load_replicated(
        src, src_pitch, width, height, (int)x, (int)y + tap);
    weight_sum += weight;
  }
  akaze_row<float>(dst, dst_pitch, y)[x] = weighted_sum / weight_sum;
}

extern "C" __global__ void akaze_half_size(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned int src_width,
    unsigned int src_height,
    unsigned long long dst,
    unsigned long long dst_pitch,
    unsigned int dst_width,
    unsigned int dst_height) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height) {
    return;
  }
  unsigned int sx = (unsigned int)floorf(
      ((float)x + 0.5f) * (float)src_width / (float)dst_width);
  unsigned int sy = (unsigned int)floorf(
      ((float)y + 0.5f) * (float)src_height / (float)dst_height);
  sx = sx < src_width ? sx : src_width - 1U;
  sy = sy < src_height ? sy : src_height - 1U;
  akaze_row<float>(dst, dst_pitch, y)[x] =
      akaze_const_row<float>(src, src_pitch, sy)[sx];
}

// Computes both normalized Scharr derivatives at the requested integer stride.
// Either destination address may be zero when only one derivative is needed.
extern "C" __global__ void akaze_scharr(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned long long dx,
    unsigned long long dx_pitch,
    unsigned long long dy,
    unsigned long long dy_pitch,
    unsigned int width,
    unsigned int height,
    unsigned int sigma_size) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int step = (int)sigma_size;
  const float scharr_weight = 10.0f / 3.0f;
  const float norm = 1.0f / (2.0f * (float)sigma_size * (scharr_weight + 2.0f));

  const float x_top = akaze_load_replicated(
      src, src_pitch, width, height, (int)x + step, (int)y - step) -
      akaze_load_replicated(src, src_pitch, width, height, (int)x - step, (int)y - step);
  const float x_mid = akaze_load_replicated(
      src, src_pitch, width, height, (int)x + step, (int)y) -
      akaze_load_replicated(src, src_pitch, width, height, (int)x - step, (int)y);
  const float x_bottom = akaze_load_replicated(
      src, src_pitch, width, height, (int)x + step, (int)y + step) -
      akaze_load_replicated(src, src_pitch, width, height, (int)x - step, (int)y + step);

  const float y_left = akaze_load_replicated(
      src, src_pitch, width, height, (int)x - step, (int)y + step) -
      akaze_load_replicated(src, src_pitch, width, height, (int)x - step, (int)y - step);
  const float y_mid = akaze_load_replicated(
      src, src_pitch, width, height, (int)x, (int)y + step) -
      akaze_load_replicated(src, src_pitch, width, height, (int)x, (int)y - step);
  const float y_right = akaze_load_replicated(
      src, src_pitch, width, height, (int)x + step, (int)y + step) -
      akaze_load_replicated(src, src_pitch, width, height, (int)x + step, (int)y - step);

  if (dx != 0ULL) {
    akaze_row<float>(dx, dx_pitch, y)[x] =
        norm * (x_top + scharr_weight * x_mid + x_bottom);
  }
  if (dy != 0ULL) {
    akaze_row<float>(dy, dy_pitch, y)[x] =
        norm * (y_left + scharr_weight * y_mid + y_right);
  }
}

// Launch once before each contrast estimate. The histogram has exactly 300 bins.
extern "C" __global__ void akaze_clear_histogram(
    unsigned long long histogram,
    unsigned long long max_gradient_squared,
    unsigned long long nonzero_count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 300U) {
    akaze_pointer<unsigned int>(histogram)[index] = 0U;
  }
  if (index == 0U) {
    *akaze_pointer<unsigned int>(max_gradient_squared) = 0U;
    *akaze_pointer<unsigned int>(nonzero_count) = 0U;
  }
}

// mode 0 finds max(|gradient|^2) and counts nonzero samples. mode 1 fills
// the 300 bins using that frozen maximum. Integer atomics make both results
// independent of warp scheduling; no candidate ordering depends on atomics.
extern "C" __global__ void akaze_gradient_histogram(
    unsigned long long lx,
    unsigned long long lx_pitch,
    unsigned long long ly,
    unsigned long long ly_pitch,
    unsigned int width,
    unsigned int height,
    unsigned long long histogram,
    unsigned long long max_gradient_squared,
    unsigned long long nonzero_count,
    unsigned int mode) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x == 0U || y == 0U || x + 1U >= width || y + 1U >= height) {
    return;
  }
  const float gx = akaze_const_row<float>(lx, lx_pitch, y)[x];
  const float gy = akaze_const_row<float>(ly, ly_pitch, y)[x];
  const float magnitude_squared = gx * gx + gy * gy;
  if (!(magnitude_squared > 0.0f)) {
    return;
  }
  if (mode == 0U) {
    atomicMax(akaze_pointer<unsigned int>(max_gradient_squared),
              __float_as_uint(magnitude_squared));
    atomicAdd(akaze_pointer<unsigned int>(nonzero_count), 1U);
    return;
  }
  const float maximum = __uint_as_float(
      *akaze_const_pointer<unsigned int>(max_gradient_squared));
  if (!(maximum > 0.0f)) {
    return;
  }
  unsigned int bin = (unsigned int)floorf(300.0f * sqrtf(magnitude_squared / maximum));
  bin = bin < 300U ? bin : 299U;
  atomicAdd(akaze_pointer<unsigned int>(histogram) + bin, 1U);
}

extern "C" __global__ void akaze_histogram_percentile(
    unsigned long long histogram,
    unsigned long long max_gradient_squared,
    unsigned long long nonzero_count,
    double percentile,
    float minimum_contrast,
    unsigned long long contrast) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  const unsigned int count = *akaze_const_pointer<unsigned int>(nonzero_count);
  const unsigned int threshold = (unsigned int)((double)count * percentile);
  unsigned int elements = 0U;
  unsigned int bin = 0U;
  const unsigned int* bins = akaze_const_pointer<unsigned int>(histogram);
  while (elements < threshold && bin < 300U) {
    elements += bins[bin];
    ++bin;
  }
  const float maximum = __uint_as_float(
      *akaze_const_pointer<unsigned int>(max_gradient_squared));
  float result = elements >= threshold
      ? sqrtf(maximum) * (float)bin * (1.0f / 300.0f)
      : 0.03f;
  result = result > minimum_contrast ? result : minimum_contrast;
  *akaze_pointer<float>(contrast) = result;
}

extern "C" __global__ void akaze_conductivity(
    unsigned long long lx,
    unsigned long long lx_pitch,
    unsigned long long ly,
    unsigned long long ly_pitch,
    unsigned long long conductivity,
    unsigned long long conductivity_pitch,
    unsigned int width,
    unsigned int height,
    unsigned long long base_contrast,
    float contrast_scale) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const float gx = akaze_const_row<float>(lx, lx_pitch, y)[x];
  const float gy = akaze_const_row<float>(ly, ly_pitch, y)[x];
  const float contrast =
      *akaze_const_pointer<float>(base_contrast) * contrast_scale;
  const float k = contrast > 1.0e-6f ? contrast : 1.0e-6f;
  const float inverse_k_squared = 1.0f / (k * k);
  akaze_row<float>(conductivity, conductivity_pitch, y)[x] =
      1.0f / (1.0f + inverse_k_squared * (gx * gx + gy * gy));
}

// src and dst must be distinct. This is the four-neighbor conservative FED
// update equivalent to forming horizontal/vertical edge flows first.
extern "C" __global__ void akaze_diffusion_step(
    unsigned long long src,
    unsigned long long src_pitch,
    unsigned long long conductivity,
    unsigned long long conductivity_pitch,
    unsigned long long dst,
    unsigned long long dst_pitch,
    unsigned int width,
    unsigned int height,
    float step_size) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const float center = akaze_const_row<float>(src, src_pitch, y)[x];
  const float center_c = akaze_const_row<float>(conductivity, conductivity_pitch, y)[x];
  float delta = 0.0f;
  if (x > 0U) {
    const float neighbor = akaze_const_row<float>(src, src_pitch, y)[x - 1U];
    const float neighbor_c =
        akaze_const_row<float>(conductivity, conductivity_pitch, y)[x - 1U];
    delta += center_c * neighbor_c * (neighbor - center);
  }
  if (x + 1U < width) {
    const float neighbor = akaze_const_row<float>(src, src_pitch, y)[x + 1U];
    const float neighbor_c =
        akaze_const_row<float>(conductivity, conductivity_pitch, y)[x + 1U];
    delta += center_c * neighbor_c * (neighbor - center);
  }
  if (y > 0U) {
    const float neighbor = akaze_const_row<float>(src, src_pitch, y - 1U)[x];
    const float neighbor_c =
        akaze_const_row<float>(conductivity, conductivity_pitch, y - 1U)[x];
    delta += center_c * neighbor_c * (neighbor - center);
  }
  if (y + 1U < height) {
    const float neighbor = akaze_const_row<float>(src, src_pitch, y + 1U)[x];
    const float neighbor_c =
        akaze_const_row<float>(conductivity, conductivity_pitch, y + 1U)[x];
    delta += center_c * neighbor_c * (neighbor - center);
  }
  akaze_row<float>(dst, dst_pitch, y)[x] = center + step_size * delta;
}

extern "C" __global__ void akaze_hessian_response(
    unsigned long long lxx,
    unsigned long long lxx_pitch,
    unsigned long long lyy,
    unsigned long long lyy_pitch,
    unsigned long long lxy,
    unsigned long long lxy_pitch,
    unsigned long long response,
    unsigned long long response_pitch,
    unsigned int width,
    unsigned int height,
    unsigned int sigma_size) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const float xx = akaze_const_row<float>(lxx, lxx_pitch, y)[x];
  const float yy = akaze_const_row<float>(lyy, lyy_pitch, y)[x];
  const float xy = akaze_const_row<float>(lxy, lxy_pitch, y)[x];
  const float sigma_squared = (float)sigma_size * (float)sigma_size;
  akaze_row<float>(response, response_pitch, y)[x] =
      (xx * yy - xy * xy) * sigma_squared * sigma_squared;
}

extern "C" __global__ void akaze_nms_flags(
    unsigned long long response,
    unsigned long long response_pitch,
    unsigned int width,
    unsigned int height,
    float threshold,
    unsigned long long flags) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const unsigned int linear = y * width + x;
  unsigned int keep = 0U;
  if (x > 0U && y > 0U && x + 1U < width && y + 1U < height) {
    const float center = akaze_const_row<float>(response, response_pitch, y)[x];
    keep = center > threshold;
    for (int oy = -1; oy <= 1 && keep != 0U; ++oy) {
      const float* row = akaze_const_row<float>(
          response, response_pitch, (unsigned int)((int)y + oy));
      for (int ox = -1; ox <= 1; ++ox) {
        if ((ox != 0 || oy != 0) && !(center > row[(int)x + ox])) {
          keep = 0U;
          break;
        }
      }
    }
  }
  akaze_pointer<unsigned int>(flags)[linear] = keep;
}

// Deterministic per-block exclusive scan. Launch recursively over block_sums,
// then invoke akaze_scan_add at each lower level. Shared memory is
// blockDim.x * sizeof(unsigned int); blockDim.x need not be a power of two.
extern "C" __global__ void akaze_scan_block(
    unsigned long long input,
    unsigned long long output,
    unsigned long long block_sums,
    unsigned int count) {
  extern __shared__ unsigned int values[];
  const unsigned int lane = threadIdx.x;
  const unsigned int index = blockIdx.x * blockDim.x + lane;
  const unsigned int input_value =
      index < count ? akaze_const_pointer<unsigned int>(input)[index] : 0U;
  values[lane] = input_value;
  __syncthreads();
  for (unsigned int offset = 1U; offset < blockDim.x; offset <<= 1U) {
    const unsigned int addend = lane >= offset ? values[lane - offset] : 0U;
    __syncthreads();
    values[lane] += addend;
    __syncthreads();
  }
  if (index < count) {
    akaze_pointer<unsigned int>(output)[index] = values[lane] - input_value;
  }
  const unsigned int base = blockIdx.x * blockDim.x;
  const unsigned int remaining = count > base ? count - base : 0U;
  const unsigned int valid = remaining < blockDim.x ? remaining : blockDim.x;
  if (block_sums != 0ULL && valid != 0U && lane + 1U == valid) {
    akaze_pointer<unsigned int>(block_sums)[blockIdx.x] = values[lane];
  }
}

extern "C" __global__ void akaze_scan_add(
    unsigned long long scanned,
    unsigned long long block_offsets,
    unsigned int count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    akaze_pointer<unsigned int>(scanned)[index] +=
        akaze_const_pointer<unsigned int>(block_offsets)[blockIdx.x];
  }
}

extern "C" __global__ void akaze_scatter_candidates(
    unsigned long long flags,
    unsigned long long offsets,
    unsigned long long response,
    unsigned long long response_pitch,
    unsigned int width,
    unsigned int height,
    float feature_size,
    unsigned int octave,
    unsigned int level,
    unsigned int base_order,
    unsigned long long candidates,
    unsigned int capacity,
    unsigned long long overflow) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int pixel_count = width * height;
  if (index >= pixel_count || akaze_const_pointer<unsigned int>(flags)[index] == 0U) {
    return;
  }
  const unsigned int output_index = akaze_const_pointer<unsigned int>(offsets)[index];
  if (output_index >= capacity) {
    if (overflow != 0ULL) {
      atomicExch(akaze_pointer<unsigned int>(overflow), 1U);
    }
    return;
  }
  const unsigned int x = index % width;
  const unsigned int y = index / width;
  AkazeCandidate candidate;
  candidate.x = (float)x;
  candidate.y = (float)y;
  candidate.response = akaze_const_row<float>(response, response_pitch, y)[x];
  candidate.size = feature_size;
  candidate.octave = octave;
  candidate.level = level;
  candidate.width = width;
  candidate.height = height;
  candidate.order = base_order + index;
  candidate.valid = 1U;
  unsigned int patch_index = 0U;
  for (int oy = -1; oy <= 1; ++oy) {
    const float* row = akaze_const_row<float>(
        response, response_pitch, (unsigned int)((int)y + oy));
    for (int ox = -1; ox <= 1; ++ox) {
      candidate.response_patch[patch_index++] = row[(int)x + ox];
    }
  }
  akaze_pointer<AkazeCandidate>(candidates)[output_index] = candidate;
}

__device__ __forceinline__ float akaze_octave_ratio(unsigned int octave) {
  return (float)(1U << octave);
}

__device__ __forceinline__ float akaze_global_coordinate(float local, float ratio) {
  return local * ratio + 0.5f * (ratio - 1.0f);
}

__device__ __forceinline__ bool akaze_overlaps(
    const AkazeCandidate& current, const AkazeCandidate& previous) {
  if (!(current.level == previous.level ||
        (current.level != 0U && current.level - 1U == previous.level))) {
    return false;
  }
  const float current_ratio = akaze_octave_ratio(current.octave);
  const float previous_ratio = akaze_octave_ratio(previous.octave);
  const float current_x = akaze_global_coordinate(current.x, current_ratio);
  const float current_y = akaze_global_coordinate(current.y, current_ratio);
  const float previous_x = akaze_global_coordinate(previous.x, previous_ratio);
  const float previous_y = akaze_global_coordinate(previous.y, previous_ratio);
  const float dx = current_x - previous_x;
  const float dy = current_y - previous_y;
  return dx * dx + dy * dy <= current.size * current.size;
}

__device__ __forceinline__ bool akaze_descriptor_bounds_ok(const AkazeCandidate& candidate) {
  const float ratio = akaze_octave_ratio(candidate.octave);
  const float sigma_size = roundf(candidate.size / ratio);
  const float radius = 10.0f * 1.4142135623730951f * sigma_size;
  const float left = roundf(candidate.x - radius) - 1.0f;
  const float right = roundf(candidate.x + radius) + 1.0f;
  const float top = roundf(candidate.y - radius) - 1.0f;
  const float bottom = roundf(candidate.y + radius) + 1.0f;
  return left >= 0.0f && top >= 0.0f &&
      right < (float)candidate.width && bottom < (float)candidate.height;
}

__device__ __forceinline__ bool akaze_refine(
    const AkazeCandidate& candidate, float* refined_x, float* refined_y) {
  const float* p = candidate.response_patch;
  const float center = p[4];
  const float dx = 0.5f * (p[5] - p[3]);
  const float dy = 0.5f * (p[7] - p[1]);
  const float dxx = p[5] + p[3] - 2.0f * center;
  const float dyy = p[7] + p[1] - 2.0f * center;
  const float dxy = 0.25f * (p[8] + p[0] - p[2] - p[6]);
  const float determinant = dxx * dyy - dxy * dxy;
  const float inverse = 1.0f / determinant;
  const float offset_x = (-dx * dyy + dy * dxy) * inverse;
  const float offset_y = (dx * dxy - dy * dxx) * inverse;
  if (!(offset_x == offset_x && offset_y == offset_y) ||
      fabsf(offset_x) > 1.0f || fabsf(offset_y) > 1.0f) {
    return false;
  }
  *refined_x = candidate.x + offset_x;
  *refined_y = candidate.y + offset_y;
  return true;
}

__device__ __forceinline__ bool akaze_border_ok(
    unsigned long long border_y,
    unsigned long long border_pitch,
    unsigned int width,
    unsigned int height,
    unsigned int margin,
    unsigned int black_threshold,
    float x,
    float y) {
  if (border_y == 0ULL || margin == 0U || width < 1000U) {
    return true;
  }
  const int cx = (int)x;
  const int cy = (int)y;
  const int offsets_x[8] = {
      -(int)margin, (int)margin, 0, 0,
      -(int)margin, (int)margin, -(int)margin, (int)margin};
  const int offsets_y[8] = {
      0, 0, -(int)margin, (int)margin,
      -(int)margin, -(int)margin, (int)margin, (int)margin};
  for (unsigned int i = 0U; i < 8U; ++i) {
    const int sx = cx + offsets_x[i];
    const int sy = cy + offsets_y[i];
    if (sx < 0 || sy < 0 || sx >= (int)width || sy >= (int)height ||
        akaze_const_row<unsigned char>(border_y, border_pitch, (unsigned int)sy)[sx] <
            black_threshold) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ bool akaze_response_precedes(
    const AkazeCandidate& candidate, const AkazeCandidate& current) {
  return candidate.response > current.response ||
      (candidate.response == current.response && candidate.order < current.order);
}

// Searches only the geometrically possible overlap window in one scale-space
// level. AKAZE feature radius divided by the target octave ratio is bounded by
// the configured sublevel range, so work does not grow with candidate_count.
__device__ bool akaze_has_scale_dominator(
    const AkazeCandidate& current,
    const AkazeSelectionLevel& level,
    unsigned long long nms_flags,
    unsigned long long nms_offsets,
    unsigned long long candidates,
    unsigned int candidate_count) {
  const float current_ratio = akaze_octave_ratio(current.octave);
  const float current_x = akaze_global_coordinate(current.x, current_ratio);
  const float current_y = akaze_global_coordinate(current.y, current_ratio);
  const float target_ratio = akaze_octave_ratio(level.octave);
  const float target_offset = 0.5f * (target_ratio - 1.0f);
  const float target_x = (current_x - target_offset) / target_ratio;
  const float target_y = (current_y - target_offset) / target_ratio;
  const float local_radius = current.size / target_ratio;
  const int first_x = akaze_clamp_int(
      (int)floorf(target_x - local_radius), 0, (int)level.width - 1);
  const int last_x = akaze_clamp_int(
      (int)ceilf(target_x + local_radius), 0, (int)level.width - 1);
  const int first_y = akaze_clamp_int(
      (int)floorf(target_y - local_radius), 0, (int)level.height - 1);
  const int last_y = akaze_clamp_int(
      (int)ceilf(target_y + local_radius), 0, (int)level.height - 1);
  const unsigned int* flags = akaze_const_pointer<unsigned int>(nms_flags);
  const unsigned int* offsets = akaze_const_pointer<unsigned int>(nms_offsets);
  const AkazeCandidate* values = akaze_const_pointer<AkazeCandidate>(candidates);
  for (int y = first_y; y <= last_y; ++y) {
    for (int x = first_x; x <= last_x; ++x) {
      const unsigned int pixel = level.pixel_base +
          (unsigned int)y * level.width + (unsigned int)x;
      if (flags[pixel] == 0U) {
        continue;
      }
      const unsigned int candidate_index = offsets[pixel];
      if (candidate_index >= candidate_count) {
        continue;
      }
      const AkazeCandidate candidate = values[candidate_index];
      if (candidate.order == current.order ||
          !akaze_descriptor_bounds_ok(candidate) ||
          !akaze_response_precedes(candidate, current)) {
        continue;
      }
      const float ratio = akaze_octave_ratio(candidate.octave);
      const float dx = current_x - akaze_global_coordinate(candidate.x, ratio);
      const float dy = current_y - akaze_global_coordinate(candidate.y, ratio);
      if (dx * dx + dy * dy <= current.size * current.size) {
        return true;
      }
    }
  }
  return false;
}

extern "C" __global__ void akaze_mark_scale_extrema(
    unsigned long long candidates,
    unsigned int candidate_count,
    unsigned long long levels,
    unsigned int level_count,
    unsigned long long nms_flags,
    unsigned long long nms_offsets,
    unsigned long long selected_flags) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const AkazeCandidate current = akaze_const_pointer<AkazeCandidate>(candidates)[index];
  unsigned int keep = akaze_descriptor_bounds_ok(current) ? 1U : 0U;
  if (keep != 0U && current.level < level_count) {
    const AkazeSelectionLevel* selection_levels =
        akaze_const_pointer<AkazeSelectionLevel>(levels);
    keep = akaze_has_scale_dominator(
               current, selection_levels[current.level], nms_flags, nms_offsets,
               candidates, candidate_count)
        ? 0U
        : 1U;
    if (keep != 0U && current.level != 0U) {
      keep = akaze_has_scale_dominator(
                 current, selection_levels[current.level - 1U], nms_flags, nms_offsets,
                 candidates, candidate_count)
          ? 0U
          : 1U;
    }
  }
  akaze_pointer<unsigned int>(selected_flags)[index] = keep;
}

__device__ bool akaze_has_upper_level_overlap(
    const AkazeCandidate& current,
    const AkazeSelectionLevel& level,
    unsigned long long nms_flags,
    unsigned long long nms_offsets,
    unsigned long long candidates,
    unsigned int candidate_count,
    unsigned long long scale_flags) {
  const float current_ratio = akaze_octave_ratio(current.octave);
  const float current_x = akaze_global_coordinate(current.x, current_ratio);
  const float current_y = akaze_global_coordinate(current.y, current_ratio);
  const float target_ratio = akaze_octave_ratio(level.octave);
  const float target_offset = 0.5f * (target_ratio - 1.0f);
  const float target_x = (current_x - target_offset) / target_ratio;
  const float target_y = (current_y - target_offset) / target_ratio;
  const float local_radius = current.size / target_ratio;
  const int first_x = akaze_clamp_int(
      (int)floorf(target_x - local_radius), 0, (int)level.width - 1);
  const int last_x = akaze_clamp_int(
      (int)ceilf(target_x + local_radius), 0, (int)level.width - 1);
  const int first_y = akaze_clamp_int(
      (int)floorf(target_y - local_radius), 0, (int)level.height - 1);
  const int last_y = akaze_clamp_int(
      (int)ceilf(target_y + local_radius), 0, (int)level.height - 1);
  const unsigned int* flags = akaze_const_pointer<unsigned int>(nms_flags);
  const unsigned int* offsets = akaze_const_pointer<unsigned int>(nms_offsets);
  const unsigned int* extrema = akaze_const_pointer<unsigned int>(scale_flags);
  const AkazeCandidate* values = akaze_const_pointer<AkazeCandidate>(candidates);
  for (int y = first_y; y <= last_y; ++y) {
    for (int x = first_x; x <= last_x; ++x) {
      const unsigned int pixel = level.pixel_base +
          (unsigned int)y * level.width + (unsigned int)x;
      if (flags[pixel] == 0U) {
        continue;
      }
      const unsigned int candidate_index = offsets[pixel];
      if (candidate_index >= candidate_count || extrema[candidate_index] == 0U) {
        continue;
      }
      const AkazeCandidate candidate = values[candidate_index];
      const float ratio = akaze_octave_ratio(candidate.octave);
      const float dx = current_x - akaze_global_coordinate(candidate.x, ratio);
      const float dy = current_y - akaze_global_coordinate(candidate.y, ratio);
      if (dx * dx + dy * dy <= current.size * current.size) {
        return true;
      }
    }
  }
  return false;
}

extern "C" __global__ void akaze_mark_upper_level(
    unsigned long long candidates,
    unsigned int candidate_count,
    unsigned long long levels,
    unsigned int level_count,
    unsigned long long nms_flags,
    unsigned long long nms_offsets,
    unsigned long long scale_flags,
    unsigned long long selected_flags) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const unsigned int keep = akaze_const_pointer<unsigned int>(scale_flags)[index];
  if (keep == 0U) {
    akaze_pointer<unsigned int>(selected_flags)[index] = 0U;
    return;
  }
  const AkazeCandidate current = akaze_const_pointer<AkazeCandidate>(candidates)[index];
  unsigned int output = keep;
  if (current.level + 1U < level_count) {
    const AkazeSelectionLevel upper =
        akaze_const_pointer<AkazeSelectionLevel>(levels)[current.level + 1U];
    output = akaze_has_upper_level_overlap(
                 current, upper, nms_flags, nms_offsets, candidates, candidate_count, scale_flags)
        ? 0U
        : 1U;
  }
  akaze_pointer<unsigned int>(selected_flags)[index] = output;
}

__device__ bool akaze_make_feature(
    const AkazeCandidate& candidate,
    float inverse_detection_scale,
    float crop_x,
    float crop_y,
    unsigned int use_roi,
    float roi_x_min,
    float roi_x_max,
    float roi_y_min,
    float roi_y_max,
    unsigned long long border_y,
    unsigned long long border_pitch,
    unsigned int border_width,
    unsigned int border_height,
    unsigned int border_margin,
    unsigned int black_threshold,
    AkazeFeaturePoint* point) {
  float refined_x;
  float refined_y;
  if (!akaze_refine(candidate, &refined_x, &refined_y)) {
    return false;
  }
  const float ratio = akaze_octave_ratio(candidate.octave);
  const float detector_x = akaze_global_coordinate(refined_x, ratio);
  const float detector_y = akaze_global_coordinate(refined_y, ratio);
  const float mapped_x = detector_x * inverse_detection_scale + crop_x;
  const float mapped_y = detector_y * inverse_detection_scale + crop_y;
  if (use_roi != 0U &&
      (mapped_x < roi_x_min || mapped_x > roi_x_max ||
       mapped_y < roi_y_min || mapped_y > roi_y_max)) {
    return false;
  }
  if (!akaze_border_ok(border_y, border_pitch, border_width, border_height,
                       border_margin, black_threshold, mapped_x, mapped_y)) {
    return false;
  }
  point->x = mapped_x;
  point->y = mapped_y;
  point->response = candidate.response;
  point->size = candidate.size;
  point->octave = candidate.octave;
  point->level = candidate.level;
  return true;
}

// Creates descending-response radix keys. Candidate order is already
// level-major/row-major; stable radix passes therefore preserve response ties.
extern "C" __global__ void akaze_prepare_feature_keys(
    unsigned long long candidates,
    unsigned int candidate_count,
    unsigned long long selected_flags,
    unsigned long long keys,
    unsigned long long indices,
    unsigned long long valid_count,
    float inverse_detection_scale,
    float crop_x,
    float crop_y,
    unsigned int use_roi,
    float roi_x_min,
    float roi_x_max,
    float roi_y_min,
    float roi_y_max,
    unsigned long long border_y,
    unsigned long long border_pitch,
    unsigned int border_width,
    unsigned int border_height,
    unsigned int border_margin,
    unsigned int black_threshold) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  akaze_pointer<unsigned int>(indices)[index] = index;
  unsigned int key = 0xffffffffU;
  if (akaze_const_pointer<unsigned int>(selected_flags)[index] != 0U) {
    const AkazeCandidate candidate = akaze_const_pointer<AkazeCandidate>(candidates)[index];
    AkazeFeaturePoint point;
    if (akaze_make_feature(candidate, inverse_detection_scale, crop_x, crop_y,
                           use_roi, roi_x_min, roi_x_max, roi_y_min, roi_y_max,
                           border_y, border_pitch, border_width, border_height,
                           border_margin, black_threshold, &point)) {
      key = ~__float_as_uint(candidate.response);
      atomicAdd(akaze_pointer<unsigned int>(valid_count), 1U);
    }
  }
  akaze_pointer<unsigned int>(keys)[index] = key;
}

extern "C" __global__ void akaze_radix_zero_flags(
    unsigned long long keys,
    unsigned int count,
    unsigned int bit,
    unsigned long long zero_flags) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const unsigned int key = akaze_const_pointer<unsigned int>(keys)[index];
    akaze_pointer<unsigned int>(zero_flags)[index] = ((key >> bit) & 1U) == 0U ? 1U : 0U;
  }
}

extern "C" __global__ void akaze_radix_scatter(
    unsigned long long input_keys,
    unsigned long long input_indices,
    unsigned long long zero_flags,
    unsigned long long zero_offsets,
    unsigned int count,
    unsigned long long output_keys,
    unsigned long long output_indices) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const unsigned int* flags = akaze_const_pointer<unsigned int>(zero_flags);
  const unsigned int* offsets = akaze_const_pointer<unsigned int>(zero_offsets);
  const unsigned int zero_count = offsets[count - 1U] + flags[count - 1U];
  const unsigned int output_index = flags[index] != 0U
      ? offsets[index]
      : zero_count + index - offsets[index];
  akaze_pointer<unsigned int>(output_keys)[output_index] =
      akaze_const_pointer<unsigned int>(input_keys)[index];
  akaze_pointer<unsigned int>(output_indices)[output_index] =
      akaze_const_pointer<unsigned int>(input_indices)[index];
}

extern "C" __global__ void akaze_emit_features(
    unsigned long long candidates,
    unsigned long long sorted_indices,
    unsigned long long valid_count,
    unsigned long long features,
    unsigned int max_features,
    unsigned long long feature_count,
    float inverse_detection_scale,
    float crop_x,
    float crop_y,
    unsigned int use_roi,
    float roi_x_min,
    float roi_x_max,
    float roi_y_min,
    float roi_y_max,
    unsigned long long border_y,
    unsigned long long border_pitch,
    unsigned int border_width,
    unsigned int border_height,
    unsigned int border_margin,
    unsigned int black_threshold) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int available = *akaze_const_pointer<unsigned int>(valid_count);
  const unsigned int output_count = available < max_features ? available : max_features;
  if (index == 0U) {
    *akaze_pointer<unsigned int>(feature_count) = output_count;
  }
  if (index >= output_count) {
    return;
  }
  const unsigned int candidate_index =
      akaze_const_pointer<unsigned int>(sorted_indices)[index];
  const AkazeCandidate candidate =
      akaze_const_pointer<AkazeCandidate>(candidates)[candidate_index];
  AkazeFeaturePoint point;
  if (akaze_make_feature(candidate, inverse_detection_scale, crop_x, crop_y,
                         use_roi, roi_x_min, roi_x_max, roi_y_min, roi_y_max,
                         border_y, border_pitch, border_width, border_height,
                         border_margin, black_threshold, &point)) {
    akaze_pointer<AkazeFeaturePoint>(features)[index] = point;
  }
}

__device__ __constant__ float kAkazeGauss25[49] = {
  0.02546481f, 0.02350698f, 0.01849125f, 0.01239505f, 0.00708017f, 0.00344629f, 0.00142946f,
  0.02350698f, 0.02169968f, 0.01706957f, 0.01144208f, 0.00653582f, 0.00318132f, 0.00131956f,
  0.01849125f, 0.01706957f, 0.01342740f, 0.00900066f, 0.00514126f, 0.00250252f, 0.00103800f,
  0.01239505f, 0.01144208f, 0.00900066f, 0.00603332f, 0.00344629f, 0.00167749f, 0.00069579f,
  0.00708017f, 0.00653582f, 0.00514126f, 0.00344629f, 0.00196855f, 0.00095820f, 0.00039744f,
  0.00344629f, 0.00318132f, 0.00250252f, 0.00167749f, 0.00095820f, 0.00046640f, 0.00019346f,
  0.00142946f, 0.00131956f, 0.00103800f, 0.00069579f, 0.00039744f, 0.00019346f, 0.00008024f
};

__device__ __forceinline__ float akaze_level_sample(
    unsigned long long image,
    unsigned long long pitch,
    unsigned int width,
    unsigned int height,
    int x,
    int y) {
  return akaze_load_replicated(image, pitch, width, height, x, y);
}

__device__ float akaze_orientation(
    const AkazeFeaturePoint& point,
    const AkazeLevelView& level,
    float detector_scale,
    float crop_x,
    float crop_y) {
  float responses_x[109];
  float responses_y[109];
  float angles[109];
  const float pi = 3.14159265358979323846f;
  const int gaussian_index[13] = {6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6};
  const float ratio = akaze_octave_ratio(point.octave);
  const float scale = roundf(0.5f * point.size / ratio);
  const float detector_x = (point.x - crop_x) * detector_scale;
  const float detector_y = (point.y - crop_y) * detector_scale;
  const float xf = detector_x / ratio;
  const float yf = detector_y / ratio;
  unsigned int index = 0U;
  for (int i = -6; i <= 6; ++i) {
    for (int j = -6; j <= 6; ++j) {
      if (i * i + j * j < 36) {
        const int ix = (int)roundf(xf + (float)i * scale);
        const int iy = (int)roundf(yf + (float)j * scale);
        const float weight =
            kAkazeGauss25[gaussian_index[i + 6] * 7 + gaussian_index[j + 6]];
        responses_x[index] = weight * akaze_level_sample(
            level.lx, level.lx_pitch, level.width, level.height, ix, iy);
        responses_y[index] = weight * akaze_level_sample(
            level.ly, level.ly_pitch, level.width, level.height, ix, iy);
        const float angle = atan2f(responses_y[index], responses_x[index]);
        angles[index] = angle < 0.0f ? angle + 2.0f * pi : angle;
        ++index;
      }
    }
  }

  float angle1 = 0.0f;
  float maximum = 0.0f;
  float orientation = 0.0f;
  while (angle1 < 2.0f * pi) {
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    const float angle2 = angle1 + pi / 3.0f > 2.0f * pi
        ? angle1 - 5.0f * pi / 3.0f
        : angle1 + pi / 3.0f;
    for (unsigned int k = 0U; k < 109U; ++k) {
      const float angle = angles[k];
      if ((angle1 < angle2 && angle1 < angle && angle < angle2) ||
          (angle2 < angle1 &&
           ((angle > 0.0f && angle < angle2) ||
            (angle > angle1 && angle < 2.0f * pi)))) {
        sum_x += responses_x[k];
        sum_y += responses_y[k];
      }
    }
    const float magnitude = sum_x * sum_x + sum_y * sum_y;
    if (magnitude > maximum) {
      maximum = magnitude;
      orientation = atan2f(sum_y, sum_x);
    }
    angle1 += 0.15f;
  }
  return orientation;
}

__device__ void akaze_mldb_values(
    float* values,
    unsigned int sample_step,
    const AkazeFeaturePoint& point,
    const AkazeLevelView& level,
    float xf,
    float yf,
    float cosine,
    float sine,
    float scale) {
  unsigned int value_position = 0U;
  for (int i = -10; i < 10; i += (int)sample_step) {
    for (int j = -10; j < 10; j += (int)sample_step) {
      float intensity = 0.0f;
      float derivative_x = 0.0f;
      float derivative_y = 0.0f;
      unsigned int samples = 0U;
      for (int k = i; k < i + (int)sample_step; ++k) {
        for (int l = j; l < j + (int)sample_step; ++l) {
          const float sample_l = (float)l + 0.5f;
          const float sample_k = (float)k + 0.5f;
          const float sample_y =
              yf + (sample_l * cosine * scale + sample_k * sine * scale);
          const float sample_x =
              xf + (-sample_l * sine * scale + sample_k * cosine * scale);
          const int iy = (int)roundf(sample_y);
          const int ix = (int)roundf(sample_x);
          const float image_value = akaze_level_sample(
              level.lt, level.lt_pitch, level.width, level.height, ix, iy);
          const float gx = akaze_level_sample(
              level.lx, level.lx_pitch, level.width, level.height, ix, iy);
          const float gy = akaze_level_sample(
              level.ly, level.ly_pitch, level.width, level.height, ix, iy);
          intensity += image_value;
          derivative_x += -gx * sine + gy * cosine;
          derivative_y += gx * cosine + gy * sine;
          ++samples;
        }
      }
      const float inverse_samples = 1.0f / (float)samples;
      values[value_position] = intensity * inverse_samples;
      values[value_position + 1U] = derivative_x * inverse_samples;
      values[value_position + 2U] = derivative_y * inverse_samples;
      value_position += 3U;
    }
  }
}

__device__ void akaze_mldb_compare(
    const float* values,
    unsigned char* descriptor,
    unsigned int count,
    unsigned int* bit_position) {
  for (unsigned int channel = 0U; channel < 3U; ++channel) {
    for (unsigned int i = 0U; i < count; ++i) {
      const float left = values[3U * i + channel];
      for (unsigned int j = i + 1U; j < count; ++j) {
        if (left > values[3U * j + channel]) {
          descriptor[*bit_position >> 3U] |=
              (unsigned char)(1U << (*bit_position & 7U));
        }
        ++*bit_position;
      }
    }
  }
}

extern "C" __global__ void akaze_describe_features(
    unsigned long long features,
    unsigned long long feature_count,
    unsigned long long levels,
    unsigned int level_count,
    float detector_scale,
    float crop_x,
    float crop_y,
    unsigned long long descriptors,
    unsigned long long descriptor_pitch) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int count = *akaze_const_pointer<unsigned int>(feature_count);
  if (index >= count) {
    return;
  }
  const AkazeFeaturePoint point = akaze_const_pointer<AkazeFeaturePoint>(features)[index];
  if (point.level >= level_count) {
    return;
  }
  const AkazeLevelView level = akaze_const_pointer<AkazeLevelView>(levels)[point.level];
  unsigned char* descriptor = akaze_row<unsigned char>(
      descriptors, descriptor_pitch, index);
  for (unsigned int byte = 0U; byte < 64U; ++byte) {
    descriptor[byte] = 0U;
  }

  const float angle = akaze_orientation(
      point, level, detector_scale, crop_x, crop_y);
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float ratio = akaze_octave_ratio(point.octave);
  const float scale = roundf(0.5f * point.size / ratio);
  const float detector_x = (point.x - crop_x) * detector_scale;
  const float detector_y = (point.y - crop_y) * detector_scale;
  const float xf = detector_x / ratio;
  const float yf = detector_y / ratio;
  const unsigned int sample_steps[3] = {10U, 7U, 5U};
  const unsigned int value_counts[3] = {4U, 9U, 16U};
  float values[48];
  unsigned int bit_position = 0U;
  for (unsigned int descriptor_level = 0U; descriptor_level < 3U; ++descriptor_level) {
    akaze_mldb_values(values, sample_steps[descriptor_level], point, level,
                      xf, yf, cosine, sine, scale);
    akaze_mldb_compare(values, descriptor, value_counts[descriptor_level], &bit_position);
  }
}
)cuda";
