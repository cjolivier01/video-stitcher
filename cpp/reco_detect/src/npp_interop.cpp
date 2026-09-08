#include "reco/detect/npp_interop.hpp"

#include "reco/core/windows_runtime_library.hpp"

#include <array>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reco::detect {
namespace {

constexpr int kNppiInterLinear = 2;
constexpr int kNppiInterSuper = 8;
constexpr int kNppiAxisBoth = 2;
constexpr std::uint32_t kCudaStreamNonBlocking = 1;

using CudaStream = void*;
using CudaError = int;
using CudaSetDevice = CudaError (*)(int);
using CudaGetDevice = CudaError (*)(int*);
using CudaDeviceGetAttribute = CudaError (*)(int*, int, int);
using CudaStreamCreateWithFlags = CudaError (*)(CudaStream*, std::uint32_t);
using CudaStreamDestroy = CudaError (*)(CudaStream);
using CudaStreamGetFlags = CudaError (*)(CudaStream, std::uint32_t*);

struct NppStreamContext {
  CudaStream h_stream = nullptr;
  int n_cuda_device_id = 0;
  int n_multi_processor_count = 0;
  int n_max_threads_per_multi_processor = 0;
  int n_max_threads_per_block = 0;
  std::size_t n_shared_mem_per_block = 0;
  int n_cuda_dev_attr_compute_capability_major = 0;
  int n_cuda_dev_attr_compute_capability_minor = 0;
  std::uint32_t n_stream_flags = 0;
  int n_reserved0 = 0;
};

constexpr int kCudaDevAttrMaxThreadsPerBlock = 1;
constexpr int kCudaDevAttrMaxSharedMemoryPerBlock = 8;
constexpr int kCudaDevAttrMultiProcessorCount = 16;
constexpr int kCudaDevAttrMaxThreadsPerMultiProcessor = 39;
constexpr int kCudaDevAttrComputeCapabilityMajor = 75;
constexpr int kCudaDevAttrComputeCapabilityMinor = 76;

static_assert(sizeof(NppiSize) == 8);
static_assert(sizeof(NppiRect) == 16);

class DynamicLibrary {
public:
  DynamicLibrary() = default;

  explicit DynamicLibrary(const char* name) { open(name); }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  DynamicLibrary(DynamicLibrary&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
      close();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~DynamicLibrary() { close(); }

  template <typename Fn> Fn symbol(const char* name) const {
#if defined(_WIN32)
    auto* sym = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* sym = dlsym(handle_, name);
#endif
    if (sym == nullptr) {
      throw NppError(std::string("missing NPP symbol ") + name);
    }
    return reinterpret_cast<Fn>(sym);
  }

private:
  void open(const char* name) {
#if defined(_WIN32)
    handle_ = core::detail::load_windows_runtime_library(name);
#else
    handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw NppError(std::string("failed to load ") + name);
    }
  }

  void close() noexcept {
    if (handle_ == nullptr) {
      return;
    }
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
  }

  void* handle_ = nullptr;
};

DynamicLibrary load_first(std::initializer_list<const char*> names, const char* label) {
  std::string last_error;
  for (const char* name : names) {
    try {
      return DynamicLibrary(name);
    } catch (const std::exception& error) {
      last_error = error.what();
    }
  }
  throw NppError(std::string(label) + " not available: " + last_error);
}

std::optional<DynamicLibrary> try_load_first(std::initializer_list<const char*> names) {
  for (const char* name : names) {
    try {
      return DynamicLibrary(name);
    } catch (const std::exception&) {
    }
  }
  return std::nullopt;
}

using NppiNv12ToRgb = int (*)(const std::uint8_t* const*, int, std::uint8_t*, int, NppiSize,
                              NppStreamContext);
using NppiNv12ToRgbColorTwist = int (*)(const std::uint8_t* const*, int*, std::uint8_t*, int,
                                        NppiSize, const float (*)[4], NppStreamContext);
using NppiResize = int (*)(const std::uint8_t*, int, NppiSize, NppiRect, std::uint8_t*, int,
                           NppiSize, NppiRect, int, NppStreamContext);
using NppiMirror = int (*)(const std::uint8_t*, int, std::uint8_t*, int, NppiSize, int,
                           NppStreamContext);

struct NppFunctions {
  NppFunctions(DynamicLibrary nppicc_library, DynamicLibrary nppig_library,
               std::optional<DynamicLibrary> cudart_library)
      : nppicc(std::move(nppicc_library)), nppig(std::move(nppig_library)),
        cudart(std::move(cudart_library)) {}

  NppFunctions(const NppFunctions&) = delete;
  NppFunctions& operator=(const NppFunctions&) = delete;
  NppFunctions(NppFunctions&&) = delete;
  NppFunctions& operator=(NppFunctions&&) = delete;

  DynamicLibrary nppicc;
  DynamicLibrary nppig;
  std::optional<DynamicLibrary> cudart;
  NppStreamContext stream_ctx;
  std::optional<CudaStreamDestroy> stream_destroy;
  NppiNv12ToRgb nv12_to_rgb = nullptr;
  NppiNv12ToRgbColorTwist nv12_to_rgb_color_twist = nullptr;
  NppiResize resize_c1 = nullptr;
  NppiResize resize_c3 = nullptr;
  NppiResize resize_c4 = nullptr;
  NppiMirror mirror_c3 = nullptr;

  ~NppFunctions() {
    if (stream_destroy.has_value() && stream_ctx.h_stream != nullptr) {
      (void)(*stream_destroy)(stream_ctx.h_stream);
    }
  }
};

std::pair<NppStreamContext, std::optional<CudaStreamDestroy>>
create_stream_context(const std::optional<DynamicLibrary>& cudart) {
  try {
    if (!cudart.has_value()) {
      return {};
    }

    auto set_device = cudart->symbol<CudaSetDevice>("cudaSetDevice");
    auto get_device = cudart->symbol<CudaGetDevice>("cudaGetDevice");
    auto device_get_attribute = cudart->symbol<CudaDeviceGetAttribute>("cudaDeviceGetAttribute");
    auto stream_create = cudart->symbol<CudaStreamCreateWithFlags>("cudaStreamCreateWithFlags");
    auto stream_destroy = cudart->symbol<CudaStreamDestroy>("cudaStreamDestroy");
    auto stream_get_flags = cudart->symbol<CudaStreamGetFlags>("cudaStreamGetFlags");

    const CudaError set_rc = set_device(0);
    if (set_rc != 0) {
      return {};
    }

    CudaStream stream = nullptr;
    const CudaError create_rc = stream_create(&stream, kCudaStreamNonBlocking);
    if (create_rc != 0 || stream == nullptr) {
      return {};
    }

    int device_id = 0;
    if (get_device(&device_id) != 0) {
      (void)stream_destroy(stream);
      return {};
    }

    auto get_attr = [&](int attr, bool allow_zero = false) -> std::optional<int> {
      int value = 0;
      if (device_get_attribute(&value, attr, device_id) != 0 ||
          (allow_zero ? value < 0 : value <= 0)) {
        return std::nullopt;
      }
      return value;
    };
    const auto multi_processor_count = get_attr(kCudaDevAttrMultiProcessorCount);
    const auto max_threads_per_multi_processor = get_attr(kCudaDevAttrMaxThreadsPerMultiProcessor);
    const auto max_threads_per_block = get_attr(kCudaDevAttrMaxThreadsPerBlock);
    const auto shared_mem_per_block = get_attr(kCudaDevAttrMaxSharedMemoryPerBlock);
    const auto compute_capability_major = get_attr(kCudaDevAttrComputeCapabilityMajor);
    const auto compute_capability_minor = get_attr(kCudaDevAttrComputeCapabilityMinor, true);
    std::uint32_t stream_flags = 0;
    if (!multi_processor_count.has_value() || !max_threads_per_multi_processor.has_value() ||
        !max_threads_per_block.has_value() || !shared_mem_per_block.has_value() ||
        !compute_capability_major.has_value() || !compute_capability_minor.has_value() ||
        stream_get_flags(stream, &stream_flags) != 0) {
      (void)stream_destroy(stream);
      return {};
    }

    NppStreamContext context;
    context.h_stream = stream;
    context.n_cuda_device_id = device_id;
    context.n_multi_processor_count = *multi_processor_count;
    context.n_max_threads_per_multi_processor = *max_threads_per_multi_processor;
    context.n_max_threads_per_block = *max_threads_per_block;
    context.n_shared_mem_per_block = static_cast<std::size_t>(*shared_mem_per_block);
    context.n_cuda_dev_attr_compute_capability_major = *compute_capability_major;
    context.n_cuda_dev_attr_compute_capability_minor = *compute_capability_minor;
    context.n_stream_flags = stream_flags;
    return {context, stream_destroy};
  } catch (const std::exception&) {
    return {};
  }
}

std::unique_ptr<NppFunctions> load_npp() {
  auto functions = std::make_unique<NppFunctions>(
#if defined(_WIN32)
      load_first({"nppicc64_13.dll", "nppicc64_12.dll", "nppicc64_11.dll"}, "nppicc"),
      load_first({"nppig64_13.dll", "nppig64_12.dll", "nppig64_11.dll"}, "nppig"),
      try_load_first({"cudart64_130.dll", "cudart64_120.dll", "cudart64_110.dll"})
#else
      load_first({"libnppicc.so.13", "libnppicc.so.12", "libnppicc.so.11", "libnppicc.so"},
                 "libnppicc"),
      load_first({"libnppig.so.13", "libnppig.so.12", "libnppig.so.11", "libnppig.so"}, "libnppig"),
      try_load_first({"libcudart.so.13", "libcudart.so.12", "libcudart.so.11", "libcudart.so"})
#endif
  );

  auto [stream_ctx, stream_destroy] = create_stream_context(functions->cudart);
  if (!functions->cudart.has_value() || stream_ctx.h_stream == nullptr ||
      !stream_destroy.has_value()) {
    throw NppError("NPP requires libcudart and a CUDA stream context");
  }
  functions->stream_ctx = stream_ctx;
  functions->stream_destroy = stream_destroy;
  functions->nv12_to_rgb = functions->nppicc.symbol<NppiNv12ToRgb>("nppiNV12ToRGB_8u_P2C3R_Ctx");
  functions->nv12_to_rgb_color_twist =
      functions->nppicc.symbol<NppiNv12ToRgbColorTwist>("nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx");
  functions->resize_c1 = functions->nppig.symbol<NppiResize>("nppiResize_8u_C1R_Ctx");
  functions->resize_c3 = functions->nppig.symbol<NppiResize>("nppiResize_8u_C3R_Ctx");
  functions->resize_c4 = functions->nppig.symbol<NppiResize>("nppiResize_8u_C4R_Ctx");
  functions->mirror_c3 = functions->nppig.symbol<NppiMirror>("nppiMirror_8u_C3R_Ctx");
  return functions;
}

struct NppState {
  std::unique_ptr<NppFunctions> functions;
  std::string error;
};

const NppState& npp_state() {
  static const NppState state = [] {
    try {
      return NppState{.functions = load_npp(), .error = {}};
    } catch (const std::exception& error) {
      return NppState{.functions = nullptr, .error = error.what()};
    }
  }();
  return state;
}

const NppFunctions& npp() {
  const auto& state = npp_state();
  if (!state.functions) {
    throw NppError("NPP not available: " + state.error);
  }
  return *state.functions;
}

void check_npp(const char* function, int status) {
  if (status < 0) {
    std::ostringstream message;
    message << "NPP error " << status << " in " << function;
    throw NppError(message.str());
  }
}

void require_ptr(core::CudaDevicePtr ptr, const char* name) {
  if (ptr == 0) {
    throw std::invalid_argument(std::string("NPP ") + name + " pointer must be non-zero");
  }
}

void require_dims(std::uint32_t width, std::uint32_t height, const char* name) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument(std::string("NPP ") + name + " dimensions must be non-zero");
  }
  if (width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string("NPP ") + name + " dimensions exceed i32 range");
  }
}

int checked_step(std::uint32_t width, std::uint32_t channels, const char* name) {
  const std::uint64_t step = static_cast<std::uint64_t>(width) * channels;
  if (step > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string("NPP ") + name + " row step exceeds i32 range");
  }
  return static_cast<int>(step);
}

int checked_pitch(std::size_t pitch, const char* name) {
  if (pitch == 0) {
    throw std::invalid_argument(std::string("NPP ") + name + " pitch must be non-zero");
  }
  if (pitch > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string("NPP ") + name + " pitch exceeds i32 range");
  }
  return static_cast<int>(pitch);
}

void require_roi(NppiRect roi, std::uint32_t dst_w, std::uint32_t dst_h) {
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0) {
    throw std::invalid_argument("NPP destination ROI must be positive and in bounds");
  }
  const auto right = static_cast<std::uint64_t>(roi.x) + static_cast<std::uint64_t>(roi.width);
  const auto bottom = static_cast<std::uint64_t>(roi.y) + static_cast<std::uint64_t>(roi.height);
  if (right > dst_w || bottom > dst_h) {
    throw std::invalid_argument("NPP destination ROI must be positive and in bounds");
  }
}

} // namespace

bool is_npp_available() { return npp_state().functions != nullptr; }

std::string npp_availability_error() { return npp_state().error; }

void npp_nv12_to_rgb(core::CudaDevicePtr src_y, std::size_t y_pitch, core::CudaDevicePtr src_uv,
                     std::size_t uv_pitch, core::CudaDevicePtr dst, std::uint32_t width,
                     std::uint32_t height) {
  require_ptr(src_y, "NV12 Y source");
  require_ptr(src_uv, "NV12 UV source");
  require_ptr(dst, "RGB destination");
  require_dims(width, height, "NV12");
  if ((width % 2) != 0 || (height % 2) != 0) {
    throw std::invalid_argument("NPP NV12 dimensions must be even");
  }
  if (y_pitch < width || uv_pitch < width) {
    throw std::invalid_argument("NPP NV12 pitch is smaller than width");
  }
  if (y_pitch != uv_pitch) {
    throw std::invalid_argument("NPP NV12 Y and UV pitch must match");
  }

  const auto& functions = npp();
  const std::array<const std::uint8_t*, 2> src_ptrs{
      reinterpret_cast<const std::uint8_t*>(src_y),
      reinterpret_cast<const std::uint8_t*>(src_uv),
  };
  check_npp("nppiNV12ToRGB_8u_P2C3R_Ctx",
            functions.nv12_to_rgb(
                src_ptrs.data(), checked_pitch(y_pitch, "NV12 Y"),
                reinterpret_cast<std::uint8_t*>(dst), checked_step(width, 3, "RGB"),
                NppiSize{static_cast<int>(width), static_cast<int>(height)}, functions.stream_ctx));
}

void npp_nv12_to_rgb(core::CudaDevicePtr src_y, std::size_t y_pitch, core::CudaDevicePtr src_uv,
                     std::size_t uv_pitch, core::CudaDevicePtr dst, std::uint32_t width,
                     std::uint32_t height, core::YuvColorMatrix color_matrix,
                     core::YuvColorRange color_range) {
  require_ptr(src_y, "NV12 Y source");
  require_ptr(src_uv, "NV12 UV source");
  require_ptr(dst, "RGB destination");
  require_dims(width, height, "NV12");
  if ((width % 2) != 0 || (height % 2) != 0) {
    throw std::invalid_argument("NPP NV12 dimensions must be even");
  }
  if (y_pitch < width || uv_pitch < width) {
    throw std::invalid_argument("NPP NV12 pitch is smaller than width");
  }

  const auto& functions = npp();
  const std::array<const std::uint8_t*, 2> src_ptrs{
      reinterpret_cast<const std::uint8_t*>(src_y),
      reinterpret_cast<const std::uint8_t*>(src_uv),
  };
  std::array<int, 2> src_steps{
      checked_pitch(y_pitch, "NV12 Y"),
      checked_pitch(uv_pitch, "NV12 UV"),
  };
  const auto coefficients = core::yuv_to_rgb_coefficients(color_matrix, color_range);
  const std::array<std::array<float, 4>, 3> twist{{
      {coefficients.y_scale, 0.0F, coefficients.red_from_v,
       coefficients.y_scale * coefficients.y_offset - 128.0F * coefficients.red_from_v},
      {coefficients.y_scale, coefficients.green_from_u, coefficients.green_from_v,
       coefficients.y_scale * coefficients.y_offset -
           128.0F * (coefficients.green_from_u + coefficients.green_from_v)},
      {coefficients.y_scale, coefficients.blue_from_u, 0.0F,
       coefficients.y_scale * coefficients.y_offset - 128.0F * coefficients.blue_from_u},
  }};
  check_npp("nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx",
            functions.nv12_to_rgb_color_twist(
                src_ptrs.data(), src_steps.data(), reinterpret_cast<std::uint8_t*>(dst),
                checked_step(width, 3, "RGB"),
                NppiSize{static_cast<int>(width), static_cast<int>(height)},
                reinterpret_cast<const float (*)[4]>(twist.data()), functions.stream_ctx));
}

void npp_resize_c3(core::CudaDevicePtr src, std::uint32_t src_w, std::uint32_t src_h,
                   core::CudaDevicePtr dst, std::uint32_t dst_w, std::uint32_t dst_h,
                   NppiRect dst_roi) {
  require_ptr(src, "C3 resize source");
  require_ptr(dst, "C3 resize destination");
  require_dims(src_w, src_h, "C3 source");
  require_dims(dst_w, dst_h, "C3 destination");
  require_roi(dst_roi, dst_w, dst_h);

  const auto& functions = npp();
  check_npp("nppiResize_8u_C3R_Ctx",
            functions.resize_c3(
                reinterpret_cast<const std::uint8_t*>(src), checked_step(src_w, 3, "C3 source"),
                NppiSize{static_cast<int>(src_w), static_cast<int>(src_h)},
                NppiRect{0, 0, static_cast<int>(src_w), static_cast<int>(src_h)},
                reinterpret_cast<std::uint8_t*>(dst), checked_step(dst_w, 3, "C3 destination"),
                NppiSize{static_cast<int>(dst_w), static_cast<int>(dst_h)}, dst_roi,
                kNppiInterLinear, functions.stream_ctx));
}

void npp_resize_c1(core::CudaDevicePtr src, std::size_t src_pitch, std::uint32_t src_w,
                   std::uint32_t src_h, core::CudaDevicePtr dst, std::size_t dst_pitch,
                   std::uint32_t dst_w, std::uint32_t dst_h) {
  require_ptr(src, "C1 resize source");
  require_ptr(dst, "C1 resize destination");
  require_dims(src_w, src_h, "C1 source");
  require_dims(dst_w, dst_h, "C1 destination");
  if (src_pitch < src_w || dst_pitch < dst_w) {
    throw std::invalid_argument("NPP C1 resize pitch is smaller than width");
  }

  const auto& functions = npp();
  check_npp("nppiResize_8u_C1R_Ctx",
            functions.resize_c1(
                reinterpret_cast<const std::uint8_t*>(src), checked_pitch(src_pitch, "C1 source"),
                NppiSize{static_cast<int>(src_w), static_cast<int>(src_h)},
                NppiRect{0, 0, static_cast<int>(src_w), static_cast<int>(src_h)},
                reinterpret_cast<std::uint8_t*>(dst), checked_pitch(dst_pitch, "C1 destination"),
                NppiSize{static_cast<int>(dst_w), static_cast<int>(dst_h)},
                NppiRect{0, 0, static_cast<int>(dst_w), static_cast<int>(dst_h)},
                dst_w < src_w && dst_h < src_h ? kNppiInterSuper : kNppiInterLinear,
                functions.stream_ctx));
}

void npp_resize_c4(core::CudaDevicePtr src, std::uint32_t src_w, std::uint32_t src_h,
                   core::CudaDevicePtr dst, std::uint32_t dst_w, std::uint32_t dst_h,
                   NppiRect dst_roi) {
  require_ptr(src, "C4 resize source");
  require_ptr(dst, "C4 resize destination");
  require_dims(src_w, src_h, "C4 source");
  require_dims(dst_w, dst_h, "C4 destination");
  require_roi(dst_roi, dst_w, dst_h);

  const auto& functions = npp();
  check_npp("nppiResize_8u_C4R_Ctx",
            functions.resize_c4(
                reinterpret_cast<const std::uint8_t*>(src), checked_step(src_w, 4, "C4 source"),
                NppiSize{static_cast<int>(src_w), static_cast<int>(src_h)},
                NppiRect{0, 0, static_cast<int>(src_w), static_cast<int>(src_h)},
                reinterpret_cast<std::uint8_t*>(dst), checked_step(dst_w, 4, "C4 destination"),
                NppiSize{static_cast<int>(dst_w), static_cast<int>(dst_h)}, dst_roi,
                kNppiInterLinear, functions.stream_ctx));
}

void npp_mirror_c3(core::CudaDevicePtr src, core::CudaDevicePtr dst, std::uint32_t width,
                   std::uint32_t height) {
  require_ptr(src, "C3 mirror source");
  require_ptr(dst, "C3 mirror destination");
  require_dims(width, height, "C3 mirror");

  const auto& functions = npp();
  const int step = checked_step(width, 3, "C3 mirror");
  check_npp("nppiMirror_8u_C3R_Ctx",
            functions.mirror_c3(reinterpret_cast<const std::uint8_t*>(src), step,
                                reinterpret_cast<std::uint8_t*>(dst), step,
                                NppiSize{static_cast<int>(width), static_cast<int>(height)},
                                kNppiAxisBoth, functions.stream_ctx));
}

} // namespace reco::detect
