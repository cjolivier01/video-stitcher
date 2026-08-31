#include "reco/io/nvmm.hpp"

#include "reco/io/detail/nvbufsurface_7_1.hpp"
#include "reco/io/detail/nvbufsurface_9_1.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <link.h>
#endif

namespace reco::io {
namespace {

namespace abi7 = detail::nvbufsurface_7_1;
namespace abi9 = detail::nvbufsurface_9_1;

static_assert(sizeof(NvBufSurfacePlaneParams) == sizeof(abi7::PlaneParams));
static_assert(sizeof(NvBufSurfaceMappedAddr) == sizeof(abi7::MappedAddr));
static_assert(sizeof(NvBufSurfaceParams) == sizeof(abi7::SurfaceParams));
static_assert(sizeof(NvBufSurface) == sizeof(abi7::Surface));
static_assert(offsetof(NvBufSurfaceParams, buffer_desc) ==
              offsetof(abi7::SurfaceParams, buffer_desc));
static_assert(offsetof(NvBufSurfaceParams, plane_params) ==
              offsetof(abi7::SurfaceParams, plane_params));
static_assert(offsetof(NvBufSurface, surface_list) == offsetof(abi7::Surface, surface_list));

constexpr std::uint32_t kMemCudaDevice = 2;
constexpr std::uint32_t kMemSurfaceArray = 4;
constexpr std::uint32_t kLayoutPitch = 0;
constexpr std::uint32_t kColorNv12 = 6;
constexpr std::uint32_t kColorNv12Er = 7;
constexpr std::uint32_t kColorNv12_709 = 33;
constexpr std::uint32_t kColorNv12_709Er = 34;
constexpr std::uint32_t kColorNv12_2020 = 36;

bool is_nv12_color_format(std::uint32_t format) {
  return format == kColorNv12 || format == kColorNv12Er || format == kColorNv12_709 ||
         format == kColorNv12_709Er || format == kColorNv12_2020;
}

std::pair<Nv12ColorMatrix, Nv12ColorRange> color_description(std::uint32_t format) {
  switch (format) {
  case kColorNv12:
    return {Nv12ColorMatrix::Bt601, Nv12ColorRange::Limited};
  case kColorNv12Er:
    return {Nv12ColorMatrix::Bt601, Nv12ColorRange::Full};
  case kColorNv12_709:
    return {Nv12ColorMatrix::Bt709, Nv12ColorRange::Limited};
  case kColorNv12_709Er:
    return {Nv12ColorMatrix::Bt709, Nv12ColorRange::Full};
  case kColorNv12_2020:
    return {Nv12ColorMatrix::Bt2020, Nv12ColorRange::Limited};
  default:
    throw NvmmError("NvBufSurface color format is not 8-bit NV12");
  }
}

bool same_frame_info(const NvmmFrameInfo& lhs, const NvmmFrameInfo& rhs) {
  return lhs.abi == rhs.abi && lhs.memory_type == rhs.memory_type && lhs.gpu_id == rhs.gpu_id &&
         lhs.dmabuf_fd == rhs.dmabuf_fd && lhs.cuda_base_ptr == rhs.cuda_base_ptr &&
         lhs.width == rhs.width && lhs.height == rhs.height && lhs.y_offset == rhs.y_offset &&
         lhs.y_pitch == rhs.y_pitch && lhs.y_size == rhs.y_size && lhs.uv_offset == rhs.uv_offset &&
         lhs.uv_pitch == rhs.uv_pitch && lhs.uv_size == rhs.uv_size &&
         lhs.total_size == rhs.total_size && lhs.color_matrix == rhs.color_matrix &&
         lhs.color_range == rhs.color_range && lhs.surface_ptr == rhs.surface_ptr;
}

core::CudaDevicePtr checked_device_pointer(void* base, std::uint32_t offset) {
  if (base == nullptr) {
    throw NvmmError("NvBufSurface CUDA mapping returned a null data pointer");
  }
  const auto value = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base));
  if (value > std::numeric_limits<core::CudaDevicePtr>::max() - offset) {
    throw NvmmError("NvBufSurface CUDA plane pointer overflows");
  }
  return value + offset;
}

struct CudaPointerProvenance {
  std::uintptr_t context_id = 0;
  int device_ordinal = -1;
  core::CudaDevicePtr mapping_base = 0;
  std::size_t mapping_bytes = 0;
  std::size_t accessible_bytes = 0;
};

#if defined(__linux__)
class DynamicLibrary {
public:
  explicit DynamicLibrary(const char* path, int flags = RTLD_NOW | RTLD_LOCAL) : path_(path) {
    handle_ = dlopen(path, flags);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw NvmmError("failed to load " + path_ +
                      (error == nullptr ? "" : ": " + std::string(error)));
    }
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  template <typename Fn> Fn symbol(const char* name) const {
    dlerror();
    void* value = dlsym(handle_, name);
    if (value == nullptr) {
      throw NvmmError("missing runtime symbol " + std::string(name));
    }
    return reinterpret_cast<Fn>(value);
  }

  template <typename Fn> Fn optional_symbol(const char* name) const {
    dlerror();
    return reinterpret_cast<Fn>(dlsym(handle_, name));
  }

private:
  std::string path_;
  void* handle_ = nullptr;
};

struct NvbufFunctions {
  using Map = int (*)(void*, int);

  explicit NvbufFunctions(const char* path) : library(path, RTLD_NOW | RTLD_GLOBAL) {
    map_cuda = library.optional_symbol<Map>("NvBufSurfaceMapCudaBuffer");
    unmap_cuda = library.optional_symbol<Map>("NvBufSurfaceUnMapCudaBuffer");
    map_egl = library.symbol<Map>("NvBufSurfaceMapEglImage");
    unmap_egl = library.symbol<Map>("NvBufSurfaceUnMapEglImage");
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(map_egl), &info) == 0 || info.dli_fbase == nullptr ||
        info.dli_fname == nullptr) {
      throw NvmmError("cannot identify the loaded NvBufSurface runtime object");
    }
    provider_base = info.dli_fbase;
    provider_path = info.dli_fname;
  }

  DynamicLibrary library;
  Map map_cuda = nullptr;
  Map unmap_cuda = nullptr;
  Map map_egl = nullptr;
  Map unmap_egl = nullptr;
  void* provider_base = nullptr;
  std::string provider_path;
};

std::shared_ptr<NvbufFunctions> nvbuf_functions() {
  static const std::shared_ptr<NvbufFunctions> functions = [] {
    const char* path = std::getenv("RECO_NVBUFSURFACE_DYLIB_PATH");
    if (path == nullptr || path[0] == '\0') {
      path = "libnvbufsurface.so";
    }
    return std::make_shared<NvbufFunctions>(path);
  }();
  return functions;
}

using CudaResult = int;
using CudaDevice = int;
using CudaContext = void*;
using CudaGraphicsResource = void*;
constexpr CudaResult kCudaSuccess = 0;
constexpr int kCudaMemoryTypeDevice = 2;
constexpr int kCudaPointerAttributeContext = 1;
constexpr int kCudaPointerAttributeMemoryType = 2;
constexpr int kCudaPointerAttributeDeviceOrdinal = 9;
constexpr int kCudaPointerAttributeMapped = 13;
constexpr int kCudaPointerAttributeMappingSize = 18;
constexpr int kCudaPointerAttributeMappingBase = 19;
constexpr std::uint32_t kCudaEglFrameTypePitch = 1;
constexpr std::uint32_t kCudaArrayFormatUnsignedInt8 = 1;
constexpr std::uint32_t kCudaEglColorYuv420Semiplanar = 0x01;
constexpr std::uint32_t kCudaEglColorYuv420SemiplanarEr = 0x26;
constexpr std::uint32_t kCudaEglColorYuv420Semiplanar2020 = 0x53;
constexpr std::uint32_t kCudaEglColorYuv420Semiplanar709 = 0x57;

std::uint32_t expected_egl_color_format(const NvmmFrameInfo& info) {
  switch (info.color_matrix) {
  case Nv12ColorMatrix::Bt601:
    return info.color_range == Nv12ColorRange::Full ? kCudaEglColorYuv420SemiplanarEr
                                                    : kCudaEglColorYuv420Semiplanar;
  case Nv12ColorMatrix::Bt709:
    return kCudaEglColorYuv420Semiplanar709;
  case Nv12ColorMatrix::Bt2020:
    return kCudaEglColorYuv420Semiplanar2020;
  }
  throw NvmmError("unsupported NV12 color description");
}

struct CudaEglFrame {
  union {
    std::array<void*, 3> arrays;
    std::array<void*, 3> pitches;
  } frame{};
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t depth = 0;
  std::uint32_t pitch = 0;
  std::uint32_t plane_count = 0;
  std::uint32_t channel_count = 0;
  std::uint32_t frame_type = 0;
  std::uint32_t color_format = 0;
  std::uint32_t array_format = 0;
};

static_assert(sizeof(CudaEglFrame) == 64);

void check_cuda(const char* function, CudaResult result) {
  if (result != kCudaSuccess) {
    throw NvmmError(std::string(function) + " returned CUDA error " + std::to_string(result));
  }
}

struct CudaFunctions {
  using Init = CudaResult (*)(unsigned int);
  using DeviceGet = CudaResult (*)(CudaDevice*, int);
  using PrimaryContextRetain = CudaResult (*)(CudaContext*, CudaDevice);
  using PrimaryContextRelease = CudaResult (*)(CudaDevice);
  using ContextGetCurrent = CudaResult (*)(CudaContext*);
  using ContextSetCurrent = CudaResult (*)(CudaContext);
  using PointerGetAttribute = CudaResult (*)(void*, int, core::CudaDevicePtr);
  using RegisterEglImage = CudaResult (*)(CudaGraphicsResource*, void*, unsigned int);
  using GetMappedEglFrame = CudaResult (*)(CudaEglFrame*, CudaGraphicsResource, unsigned int,
                                           unsigned int);
  using UnregisterResource = CudaResult (*)(CudaGraphicsResource);

  explicit CudaFunctions(const char* path) : library(path) {
    init = library.symbol<Init>("cuInit");
    device_get = library.symbol<DeviceGet>("cuDeviceGet");
    primary_context_retain = library.symbol<PrimaryContextRetain>("cuDevicePrimaryCtxRetain");
    primary_context_release = library.symbol<PrimaryContextRelease>("cuDevicePrimaryCtxRelease");
    context_get_current = library.symbol<ContextGetCurrent>("cuCtxGetCurrent");
    context_set_current = library.symbol<ContextSetCurrent>("cuCtxSetCurrent");
    pointer_get_attribute = library.symbol<PointerGetAttribute>("cuPointerGetAttribute");
    register_egl_image = library.symbol<RegisterEglImage>("cuGraphicsEGLRegisterImage");
    get_mapped_egl_frame = library.symbol<GetMappedEglFrame>("cuGraphicsResourceGetMappedEglFrame");
    unregister_resource = library.symbol<UnregisterResource>("cuGraphicsUnregisterResource");
    check_cuda("cuInit", init(0));
    check_cuda("cuDeviceGet", device_get(&device, 0));
    check_cuda("cuDevicePrimaryCtxRetain", primary_context_retain(&primary_context, device));
  }

  ~CudaFunctions() {
    if (primary_context != nullptr) {
      (void)primary_context_release(device);
    }
  }

  DynamicLibrary library;
  Init init = nullptr;
  DeviceGet device_get = nullptr;
  PrimaryContextRetain primary_context_retain = nullptr;
  PrimaryContextRelease primary_context_release = nullptr;
  ContextGetCurrent context_get_current = nullptr;
  ContextSetCurrent context_set_current = nullptr;
  PointerGetAttribute pointer_get_attribute = nullptr;
  RegisterEglImage register_egl_image = nullptr;
  GetMappedEglFrame get_mapped_egl_frame = nullptr;
  UnregisterResource unregister_resource = nullptr;
  CudaDevice device = 0;
  CudaContext primary_context = nullptr;
};

std::shared_ptr<CudaFunctions> cuda_functions() {
  static const std::shared_ptr<CudaFunctions> functions = [] {
    const char* path = std::getenv("RECO_CUDA_DRIVER_DYLIB_PATH");
    if (path == nullptr || path[0] == '\0') {
      path = "libcuda.so.1";
    }
    return std::make_shared<CudaFunctions>(path);
  }();
  return functions;
}

class DeviceZeroContext {
public:
  explicit DeviceZeroContext(const std::shared_ptr<CudaFunctions>& functions)
      : functions_(functions) {
    check_cuda("cuCtxGetCurrent", functions_->context_get_current(&previous_));
    check_cuda("cuCtxSetCurrent", functions_->context_set_current(functions_->primary_context));
  }

  DeviceZeroContext(const DeviceZeroContext&) = delete;
  DeviceZeroContext& operator=(const DeviceZeroContext&) = delete;

  ~DeviceZeroContext() {
    if (functions_) {
      (void)functions_->context_set_current(previous_);
    }
  }

private:
  std::shared_ptr<CudaFunctions> functions_;
  CudaContext previous_ = nullptr;
};

template <typename T>
T cuda_pointer_attribute(const std::shared_ptr<CudaFunctions>& cuda, int attribute,
                         core::CudaDevicePtr pointer, const char* name) {
  T value{};
  check_cuda(name, cuda->pointer_get_attribute(&value, attribute, pointer));
  return value;
}

CudaPointerProvenance validate_cuda_plane_pointer(const std::shared_ptr<CudaFunctions>& cuda,
                                                  core::CudaDevicePtr pointer, std::size_t pitch,
                                                  std::size_t row_bytes, std::uint32_t rows,
                                                  std::size_t plane_bytes, int expected_device,
                                                  const char* plane_name) {
  const auto context = cuda_pointer_attribute<CudaContext>(
      cuda, kCudaPointerAttributeContext, pointer, "cuPointerGetAttribute(CONTEXT)");
  const auto memory_type = cuda_pointer_attribute<int>(
      cuda, kCudaPointerAttributeMemoryType, pointer, "cuPointerGetAttribute(MEMORY_TYPE)");
  const auto device = cuda_pointer_attribute<int>(cuda, kCudaPointerAttributeDeviceOrdinal, pointer,
                                                  "cuPointerGetAttribute(DEVICE_ORDINAL)");
  const auto mapped = cuda_pointer_attribute<int>(cuda, kCudaPointerAttributeMapped, pointer,
                                                  "cuPointerGetAttribute(MAPPED)");
  const auto mapping_size = cuda_pointer_attribute<std::size_t>(
      cuda, kCudaPointerAttributeMappingSize, pointer, "cuPointerGetAttribute(MAPPING_SIZE)");
  const auto mapping_base = cuda_pointer_attribute<core::CudaDevicePtr>(
      cuda, kCudaPointerAttributeMappingBase, pointer, "cuPointerGetAttribute(MAPPING_BASE_ADDR)");
  if (context != cuda->primary_context || memory_type != kCudaMemoryTypeDevice ||
      device != expected_device || mapped == 0) {
    throw NvmmError(std::string("NvBufSurface ") + plane_name +
                    " plane is not memory on the expected device in the retained CUDA context");
  }
  if (pointer < mapping_base) {
    throw NvmmError(std::string("NvBufSurface ") + plane_name + " plane precedes its CUDA mapping");
  }
  const auto offset = pointer - mapping_base;
  if (offset > mapping_size) {
    throw NvmmError(std::string("NvBufSurface ") + plane_name + " plane exceeds its CUDA mapping");
  }
  const auto mapping_remaining = mapping_size - static_cast<std::size_t>(offset);
  const auto accessible_bytes = std::min(plane_bytes, mapping_remaining);
  const auto required = static_cast<std::uint64_t>(rows - 1U) * pitch + row_bytes;
  if (required > accessible_bytes) {
    throw NvmmError(std::string("NvBufSurface ") + plane_name +
                    " plane exceeds its CUDA mapping or plane allocation");
  }
  return {
      .context_id = reinterpret_cast<std::uintptr_t>(context),
      .device_ordinal = device,
      .mapping_base = mapping_base,
      .mapping_bytes = mapping_size,
      .accessible_bytes = accessible_bytes,
  };
}

enum class SurfaceMappingKind {
  CudaBuffer,
  EglImage,
};

enum class SurfaceCleanupFailure {
  None,
  Context,
  CudaUnregister,
  EglUnmap,
  CudaUnmap,
  Unknown,
};

struct CudaMappingState;

struct CudaMappingRegistry {
  std::mutex mutex;
  std::unordered_map<void*, std::shared_ptr<CudaMappingState>> mappings;
};

std::shared_ptr<CudaMappingRegistry> mapping_registry() {
  static const auto registry = std::make_shared<CudaMappingRegistry>();
  return registry;
}

struct CudaMappingState {
  std::shared_ptr<NvbufFunctions> functions;
  std::shared_ptr<CudaFunctions> cuda;
  std::shared_ptr<const NvbufSurfaceRuntime> runtime;
  std::shared_ptr<void> decoder_owner;
  NvmmFrameInfo frame_info;
  void* surface = nullptr;
  CudaGraphicsResource graphics_resource = nullptr;
  core::CudaDevicePtr y_ptr = 0;
  core::CudaDevicePtr uv_ptr = 0;
  CudaPointerProvenance y_provenance;
  CudaPointerProvenance uv_provenance;
  SurfaceMappingKind kind = SurfaceMappingKind::CudaBuffer;
  std::size_t active_leases = 0;
  SurfaceCleanupFailure cleanup_failure = SurfaceCleanupFailure::None;
  CudaResult cleanup_cuda_result = kCudaSuccess;
};

std::string cleanup_failure_message(const CudaMappingState& mapping) {
  switch (mapping.cleanup_failure) {
  case SurfaceCleanupFailure::None:
    return {};
  case SurfaceCleanupFailure::Context:
    return "failed to select CUDA device 0 context during cleanup";
  case SurfaceCleanupFailure::CudaUnregister:
    return "cuGraphicsUnregisterResource returned CUDA error " +
           std::to_string(mapping.cleanup_cuda_result);
  case SurfaceCleanupFailure::EglUnmap:
    return "NvBufSurfaceUnMapEglImage failed";
  case SurfaceCleanupFailure::CudaUnmap:
    return "NvBufSurfaceUnMapCudaBuffer failed";
  case SurfaceCleanupFailure::Unknown:
    return "unknown CUDA mapping cleanup failure";
  }
  return "unknown CUDA mapping cleanup failure";
}

bool release_surface_mapping(CudaMappingState& mapping) noexcept {
  try {
    DeviceZeroContext context(mapping.cuda);
    if (mapping.kind == SurfaceMappingKind::EglImage) {
      if (mapping.graphics_resource != nullptr) {
        const auto result = mapping.cuda->unregister_resource(mapping.graphics_resource);
        if (result != kCudaSuccess) {
          mapping.cleanup_failure = SurfaceCleanupFailure::CudaUnregister;
          mapping.cleanup_cuda_result = result;
          return false;
        }
        mapping.graphics_resource = nullptr;
      }
      if (mapping.functions->unmap_egl(mapping.surface, 0) != 0) {
        mapping.cleanup_failure = SurfaceCleanupFailure::EglUnmap;
        return false;
      }
    } else if (mapping.functions->unmap_cuda == nullptr ||
               mapping.functions->unmap_cuda(mapping.surface, 0) != 0) {
      mapping.cleanup_failure = SurfaceCleanupFailure::CudaUnmap;
      return false;
    }
    mapping.surface = nullptr;
    mapping.cleanup_failure = SurfaceCleanupFailure::None;
    mapping.cleanup_cuda_result = kCudaSuccess;
    return true;
  } catch (const std::exception&) {
    mapping.cleanup_failure = SurfaceCleanupFailure::Context;
    return false;
  } catch (...) {
    mapping.cleanup_failure = SurfaceCleanupFailure::Unknown;
    return false;
  }
}

struct CudaMappingLease {
  std::shared_ptr<CudaMappingState> mapping;
  std::shared_ptr<CudaMappingRegistry> registry;
  bool active = false;

  ~CudaMappingLease() {
    if (!active || !mapping || !registry) {
      return;
    }
    std::lock_guard<std::mutex> lock(registry->mutex);
    if (mapping->active_leases == 0) {
      return;
    }
    --mapping->active_leases;
    if (mapping->active_leases != 0) {
      return;
    }
    const auto it = registry->mappings.find(mapping->surface);
    if (it == registry->mappings.end() || it->second.get() != mapping.get()) {
      return;
    }
    if (release_surface_mapping(*mapping)) {
      registry->mappings.erase(it);
    }
  }
};

std::shared_ptr<void> acquire_mapping_lease(const std::shared_ptr<CudaMappingState>& mapping,
                                            const std::shared_ptr<CudaMappingRegistry>& registry) {
  auto lease = std::make_shared<CudaMappingLease>();
  lease->mapping = mapping;
  lease->registry = registry;
  ++mapping->active_leases;
  lease->active = true;
  return lease;
}

struct DirectCudaOwner {
  std::shared_ptr<const NvbufSurfaceRuntime> runtime;
  std::shared_ptr<CudaFunctions> cuda;
  std::shared_ptr<void> decoder_owner;
};

bool same_owner(const std::shared_ptr<void>& lhs, const std::shared_ptr<void>& rhs) {
  return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}
#endif

template <typename Surface>
NvmmFrameInfo extract_frame_info(const void* mapped_data, NvbufSurfaceAbi selected_abi) {
  using SurfaceParams = std::remove_pointer_t<decltype(std::declval<Surface>().surface_list)>;
  if (mapped_data == nullptr) {
    throw NvmmError("NvBufSurface mapped data is null");
  }
  if ((reinterpret_cast<std::uintptr_t>(mapped_data) % alignof(Surface)) != 0) {
    throw NvmmError("NvBufSurface pointer is misaligned for the selected ABI");
  }
  const auto* surface = static_cast<const Surface*>(mapped_data);
  if (surface->mem_type != kMemSurfaceArray && surface->mem_type != kMemCudaDevice) {
    throw NvmmError("expected NVBUF_MEM_SURFACE_ARRAY or NVBUF_MEM_CUDA_DEVICE");
  }
  if (surface->batch_size == 0 || surface->num_filled == 0 ||
      surface->num_filled > surface->batch_size) {
    throw NvmmError("NvBufSurface filled-buffer count is invalid");
  }
  if (surface->surface_list == nullptr) {
    throw NvmmError("NvBufSurface surface_list is null");
  }
  if ((reinterpret_cast<std::uintptr_t>(surface->surface_list) % alignof(SurfaceParams)) != 0) {
    throw NvmmError("NvBufSurface surface_list is misaligned for the selected ABI");
  }

  const auto& params = surface->surface_list[0];
  const auto& planes = params.plane_params;
  if (!is_nv12_color_format(params.color_format)) {
    throw NvmmError("NvBufSurface color format is not 8-bit NV12");
  }
  if (params.layout != kLayoutPitch) {
    throw NvmmError("NvBufSurface must use pitch-linear layout for CUDA/NPP");
  }
  if (params.width == 0 || params.height == 0 || (params.width % 2U) != 0 ||
      (params.height % 2U) != 0) {
    throw NvmmError("NvBufSurface NV12 dimensions must be non-zero and even");
  }
  if (planes.num_planes != 2) {
    throw NvmmError("NvBufSurface must contain exactly 2 NV12 planes");
  }
  if (planes.width[0] != params.width || planes.height[0] != params.height ||
      planes.width[1] != params.width / 2U || planes.height[1] != params.height / 2U) {
    throw NvmmError("NvBufSurface NV12 plane dimensions are inconsistent");
  }
  if (planes.bytes_per_pix[0] != 1 || planes.bytes_per_pix[1] != 2) {
    throw NvmmError("NvBufSurface NV12 bytes-per-pixel metadata is invalid");
  }
  if (params.pitch != planes.pitch[0] || planes.pitch[0] < params.width ||
      planes.pitch[1] < params.width || planes.pitch[0] != planes.pitch[1]) {
    throw NvmmError("NvBufSurface NV12 pitch metadata is unsupported");
  }
  if (params.data_size == 0) {
    throw NvmmError("NvBufSurface allocation size is zero");
  }

  const auto y_required = static_cast<std::uint64_t>(planes.pitch[0]) * planes.height[0];
  const auto uv_required = static_cast<std::uint64_t>(planes.pitch[1]) * planes.height[1];
  if (planes.psize[0] < y_required || planes.psize[1] < uv_required) {
    throw NvmmError("NvBufSurface plane allocation is smaller than its pitched extent");
  }
  const auto y_alloc_end = static_cast<std::uint64_t>(planes.offset[0]) + planes.psize[0];
  const auto uv_alloc_end = static_cast<std::uint64_t>(planes.offset[1]) + planes.psize[1];
  if (planes.offset[1] < y_alloc_end) {
    throw NvmmError("NvBufSurface NV12 plane allocations overlap");
  }
  if (y_alloc_end > params.data_size || uv_alloc_end > params.data_size) {
    throw NvmmError("NvBufSurface plane allocation exceeds dataSize");
  }

  const bool surface_array = surface->mem_type == kMemSurfaceArray;
  std::int32_t dmabuf_fd = -1;
  if (surface_array) {
    using BufferDescriptor = std::remove_cv_t<decltype(params.buffer_desc)>;
    bool negative = false;
    if constexpr (std::is_signed_v<BufferDescriptor>) {
      negative = params.buffer_desc < 0;
    }
    if (negative || static_cast<std::uint64_t>(params.buffer_desc) >
                        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
      throw NvmmError("NvBufSurface DMA-BUF descriptor is out of range");
    }
    dmabuf_fd = static_cast<std::int32_t>(params.buffer_desc);
  } else if (params.data_ptr == nullptr) {
    throw NvmmError("NvBufSurface CUDA device data pointer is null");
  }

  const auto [color_matrix, color_range] = color_description(params.color_format);
  NvmmFrameInfo info{
      .dmabuf_fd = dmabuf_fd,
      .width = params.width,
      .height = params.height,
      .y_offset = planes.offset[0],
      .y_pitch = planes.pitch[0],
      .y_size = planes.psize[0],
      .uv_offset = planes.offset[1],
      .uv_pitch = planes.pitch[1],
      .uv_size = planes.psize[1],
      .total_size = params.data_size,
      .surface_ptr = const_cast<void*>(mapped_data),
      .abi = selected_abi,
      .memory_type = surface_array ? NvmmMemoryType::SurfaceArray : NvmmMemoryType::CudaDevice,
      .gpu_id = surface->gpu_id,
      .cuda_base_ptr = surface_array ? 0
                                     : static_cast<core::CudaDevicePtr>(
                                           reinterpret_cast<std::uintptr_t>(params.data_ptr)),
      .color_matrix = color_matrix,
      .color_range = color_range,
  };
  if (const auto error = validate_nvmm_frame_info(info); error.has_value()) {
    throw NvmmError(*error);
  }
  return info;
}

void* direct_cuda_data_ptr(const NvmmFrameInfo& info) {
  switch (info.abi) {
  case NvbufSurfaceAbi::DeepStream7_1:
    return static_cast<abi7::Surface*>(info.surface_ptr)->surface_list[0].data_ptr;
  case NvbufSurfaceAbi::DeepStream9_1:
    return static_cast<abi9::Surface*>(info.surface_ptr)->surface_list[0].data_ptr;
  }
  return nullptr;
}

#if defined(__linux__)
void* mapped_cuda_buffer_handle(const NvmmFrameInfo& info) {
  if (info.abi != NvbufSurfaceAbi::DeepStream9_1) {
    return nullptr;
  }
  auto* surface = static_cast<abi9::Surface*>(info.surface_ptr);
  return surface->surface_list[0].mapped_addr.cuda_ptr;
}

void* mapped_cuda_buffer_ptr(const NvmmFrameInfo& info) {
  auto* buffer = static_cast<abi9::CudaBuffer*>(mapped_cuda_buffer_handle(info));
  return buffer == nullptr ? nullptr : buffer->data_ptr;
}

void* mapped_egl_image(const NvmmFrameInfo& info) {
  switch (info.abi) {
  case NvbufSurfaceAbi::DeepStream7_1:
    return static_cast<abi7::Surface*>(info.surface_ptr)->surface_list[0].mapped_addr.egl_image;
  case NvbufSurfaceAbi::DeepStream9_1:
    return static_cast<abi9::Surface*>(info.surface_ptr)->surface_list[0].mapped_addr.egl_image;
  }
  return nullptr;
}
#endif

} // namespace

struct NvbufSurfaceRuntime::State {
#if defined(__linux__)
  std::shared_ptr<NvbufFunctions> functions;
  std::unique_ptr<DynamicLibrary> version_library;
#endif
  NvbufSurfaceAbi abi = NvbufSurfaceAbi::DeepStream7_1;
  std::string library;
};

NvbufSurfaceRuntime::NvbufSurfaceRuntime(std::unique_ptr<State> state) : state_(std::move(state)) {}

NvbufSurfaceRuntime::~NvbufSurfaceRuntime() = default;

NvbufSurfaceAbi NvbufSurfaceRuntime::abi() const noexcept { return state_->abi; }

std::string_view NvbufSurfaceRuntime::library() const noexcept { return state_->library; }

std::shared_ptr<const NvbufSurfaceRuntime> discover_nvbufsurface_runtime() {
#if defined(__linux__)
  const char* nvbufsurface_path = std::getenv("RECO_NVBUFSURFACE_DYLIB_PATH");
  if (nvbufsurface_path == nullptr || nvbufsurface_path[0] == '\0') {
    nvbufsurface_path = "libnvbufsurface.so";
  }
  auto functions = std::make_shared<NvbufFunctions>(nvbufsurface_path);

  const char* deepstream_utils_path = std::getenv("RECO_NVDS_UTILS_DYLIB_PATH");
  if (deepstream_utils_path == nullptr || deepstream_utils_path[0] == '\0') {
    deepstream_utils_path = "libnvds_utils.so";
  }
  auto deepstream_utils = std::make_unique<DynamicLibrary>(deepstream_utils_path);
  using DeepStreamVersion = void (*)(unsigned int*, unsigned int*);
  const auto version = deepstream_utils->symbol<DeepStreamVersion>("nvds_version");
  unsigned int major = 0;
  unsigned int minor = 0;
  version(&major, &minor);

  NvbufSurfaceAbi abi;
  if (major == 7U && minor == 1U) {
    if (functions->map_cuda != nullptr || functions->unmap_cuda != nullptr) {
      throw NvmmError(
          "DeepStream 7.1 version metadata does not match the loaded NvBufSurface runtime: "
          "CUDA-buffer mapping symbols are forbidden for the 7.1 ABI");
    }
    abi = NvbufSurfaceAbi::DeepStream7_1;
  } else if (major == 9U && minor == 1U) {
    if (functions->map_cuda == nullptr || functions->unmap_cuda == nullptr) {
      throw NvmmError(
          "DeepStream 9.1 version metadata does not match the loaded NvBufSurface runtime: "
          "both CUDA-buffer mapping symbols are required for the 9.1 ABI");
    }
    abi = NvbufSurfaceAbi::DeepStream9_1;
  } else {
    throw NvmmError("unsupported DeepStream NvBufSurface ABI version " + std::to_string(major) +
                    "." + std::to_string(minor) + "; supported versions are 7.1 and 9.1");
  }

  auto state = std::make_unique<NvbufSurfaceRuntime::State>();
  state->functions = std::move(functions);
  state->version_library = std::move(deepstream_utils);
  state->abi = abi;
  state->library = state->functions->provider_path;
  return std::shared_ptr<const NvbufSurfaceRuntime>(new NvbufSurfaceRuntime(std::move(state)));
#else
  throw NvmmError("DeepStream NvBufSurface ABI discovery is only supported on Linux");
#endif
}

NvbufSurfaceAbi discover_nvbufsurface_abi() { return discover_nvbufsurface_runtime()->abi(); }

std::optional<std::string> validate_nvbufsurface_runtime_provenance(
    const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
#if defined(__linux__)
  if (!runtime || !runtime->state_ || !runtime->state_->functions) {
    return "NvBufSurface runtime binding is missing";
  }
  struct LoadedObject {
    std::uintptr_t base = 0;
    std::string path;
  };
  std::vector<LoadedObject> loaded_objects;
  const auto collect = [](dl_phdr_info* info, std::size_t, void* raw_objects) {
    if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0') {
      static_cast<std::vector<LoadedObject>*>(raw_objects)
          ->push_back(LoadedObject{.base = static_cast<std::uintptr_t>(info->dlpi_addr),
                                   .path = info->dlpi_name});
    }
    return 0;
  };
  (void)dl_iterate_phdr(collect, &loaded_objects);

  const auto expected_base = runtime->state_->functions->provider_base;
  for (const auto& object : loaded_objects) {
    void* handle = dlopen(object.path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle == nullptr) {
      continue;
    }
    dlerror();
    void* symbol = dlsym(handle, "NvBufSurfaceMapEglImage");
    Dl_info symbol_info{};
    const bool owns_provider =
        symbol != nullptr && dladdr(symbol, &symbol_info) != 0 &&
        symbol_info.dli_fbase != nullptr &&
        reinterpret_cast<std::uintptr_t>(symbol_info.dli_fbase) == object.base;
    (void)dlclose(handle);
    if (owns_provider && symbol_info.dli_fbase != expected_base) {
      return "multiple NvBufSurface runtime providers are loaded: retained " +
             runtime->state_->functions->provider_path + " but also found " + object.path;
    }
  }
  return std::nullopt;
#else
  (void)runtime;
  return "NvBufSurface runtime provenance is only supported on Linux";
#endif
}

std::optional<std::string> validate_nvmm_frame_info(const NvmmFrameInfo& info) {
  if (info.abi != NvbufSurfaceAbi::DeepStream7_1 && info.abi != NvbufSurfaceAbi::DeepStream9_1) {
    return "unsupported NvBufSurface ABI";
  }
  if (info.memory_type != NvmmMemoryType::SurfaceArray &&
      info.memory_type != NvmmMemoryType::CudaDevice) {
    return "unsupported NvBufSurface memory type";
  }
  if (info.runtime && info.runtime->abi() != info.abi) {
    return "NvBufSurface frame ABI does not match its retained runtime";
  }
  if (info.surface_ptr == nullptr) {
    return "NvBufSurface pointer is null";
  }
  if (info.gpu_id != 0) {
    return "only CUDA device 0 is supported by current detector consumers";
  }
  if (info.width == 0 || info.height == 0) {
    return "NV12 dimensions must be non-zero";
  }
  if ((info.width % 2U) != 0 || (info.height % 2U) != 0) {
    return "NV12 dimensions must be even";
  }
  if (info.y_pitch < info.width || info.uv_pitch < info.width) {
    return "NV12 pitches must cover the frame width";
  }
  if (info.y_pitch != info.uv_pitch) {
    return "NV12 Y and UV pitches must match for the CUDA/NPP path";
  }
  const auto y_required = static_cast<std::uint64_t>(info.y_pitch) * info.height;
  const auto uv_required = static_cast<std::uint64_t>(info.uv_pitch) * (info.height / 2U);
  if (info.y_size < y_required || info.uv_size < uv_required) {
    return "NV12 plane allocation is smaller than its pitched extent";
  }
  const auto y_end = static_cast<std::uint64_t>(info.y_offset) + info.y_size;
  const auto uv_end = static_cast<std::uint64_t>(info.uv_offset) + info.uv_size;
  if (info.uv_offset < y_end) {
    return "NV12 planes overlap";
  }
  if (info.total_size == 0 || y_end > info.total_size || uv_end > info.total_size) {
    return "NV12 planes exceed the NvBufSurface allocation";
  }
  if (info.memory_type == NvmmMemoryType::SurfaceArray && info.dmabuf_fd < 0) {
    return "NVBUF_MEM_SURFACE_ARRAY requires a valid DMA-BUF descriptor";
  }
  if (info.memory_type == NvmmMemoryType::CudaDevice && info.cuda_base_ptr == 0) {
    return "NVBUF_MEM_CUDA_DEVICE requires a CUDA device pointer";
  }
  return std::nullopt;
}

NvmmFrameInfo extract_nvmm_frame_info(const void* mapped_data) {
  return extract_frame_info<abi7::Surface>(mapped_data, NvbufSurfaceAbi::DeepStream7_1);
}

NvmmFrameInfo extract_nvmm_frame_info(const void* mapped_data, NvbufSurfaceAbi requested_abi) {
  switch (requested_abi) {
  case NvbufSurfaceAbi::DeepStream7_1:
    return extract_frame_info<abi7::Surface>(mapped_data, requested_abi);
  case NvbufSurfaceAbi::DeepStream9_1:
    return extract_frame_info<abi9::Surface>(mapped_data, requested_abi);
  }
  throw NvmmError("unsupported NvBufSurface ABI");
}

bool is_nvmm_cuda_interop_available() {
#if defined(__linux__)
  try {
    (void)nvbuf_functions();
    (void)cuda_functions();
    return true;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

std::string nvmm_cuda_interop_availability_error() {
#if defined(__linux__)
  try {
    (void)nvbuf_functions();
    (void)cuda_functions();
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
#else
  return "NvBufSurface CUDA interop is only supported on Linux";
#endif
}

NvmmCudaFrame map_nvmm_frame_to_cuda(const NvmmFrameInfo& info, std::shared_ptr<void> owner) {
  if (!owner) {
    throw NvmmError("NvBufSurface CUDA mapping requires a retained decoder owner");
  }
  if (const auto error = validate_nvmm_frame_info(info); error.has_value()) {
    throw NvmmError(*error);
  }
  const auto current = extract_nvmm_frame_info(info.surface_ptr, info.abi);
  if (!same_frame_info(info, current)) {
    throw NvmmError("NvBufSurface metadata changed before CUDA mapping");
  }

  auto make_frame = [&](core::CudaDevicePtr y_ptr, core::CudaDevicePtr uv_ptr,
                        const CudaPointerProvenance& y_provenance,
                        const CudaPointerProvenance& uv_provenance,
                        std::shared_ptr<void> mapped_owner) {
    if (y_provenance.context_id != uv_provenance.context_id ||
        y_provenance.device_ordinal != uv_provenance.device_ordinal) {
      throw NvmmError("NvBufSurface CUDA planes do not share one context and device");
    }
    return NvmmCudaFrame{
        .y_ptr = y_ptr,
        .uv_ptr = uv_ptr,
        .y_pitch = info.y_pitch,
        .uv_pitch = info.uv_pitch,
        .y_accessible_bytes = y_provenance.accessible_bytes,
        .uv_accessible_bytes = uv_provenance.accessible_bytes,
        .y_mapping_base = y_provenance.mapping_base,
        .uv_mapping_base = uv_provenance.mapping_base,
        .y_mapping_bytes = y_provenance.mapping_bytes,
        .uv_mapping_bytes = uv_provenance.mapping_bytes,
        .width = info.width,
        .height = info.height,
        .gpu_id = info.gpu_id,
        .context_id = y_provenance.context_id,
        .device_ordinal = y_provenance.device_ordinal,
        .color_matrix = info.color_matrix,
        .color_range = info.color_range,
        .runtime = info.runtime,
        .owner = std::move(mapped_owner),
    };
  };

#if defined(__linux__)
  if (info.runtime) {
    if (const auto error = validate_nvbufsurface_runtime_provenance(info.runtime);
        error.has_value()) {
      throw NvmmError(*error);
    }
  }
  auto cuda = cuda_functions();
  DeviceZeroContext context(cuda);
  if (info.memory_type == NvmmMemoryType::CudaDevice) {
    void* base = direct_cuda_data_ptr(info);
    const auto y_ptr = checked_device_pointer(base, info.y_offset);
    const auto uv_ptr = checked_device_pointer(base, info.uv_offset);
    const auto y_provenance =
        validate_cuda_plane_pointer(cuda, y_ptr, info.y_pitch, info.width, info.height, info.y_size,
                                    static_cast<int>(info.gpu_id), "Y");
    const auto uv_provenance =
        validate_cuda_plane_pointer(cuda, uv_ptr, info.uv_pitch, info.width, info.height / 2U,
                                    info.uv_size, static_cast<int>(info.gpu_id), "UV");
    auto direct_owner = std::make_shared<DirectCudaOwner>();
    direct_owner->decoder_owner = std::move(owner);
    direct_owner->runtime = info.runtime;
    direct_owner->cuda = std::move(cuda);
    return make_frame(y_ptr, uv_ptr, y_provenance, uv_provenance, direct_owner);
  }

  auto registry = mapping_registry();
  std::unique_lock<std::mutex> lock(registry->mutex);
  if (auto it = registry->mappings.find(info.surface_ptr); it != registry->mappings.end()) {
    const auto& existing = it->second;
    if (existing->cleanup_failure != SurfaceCleanupFailure::None) {
      throw NvmmError("NvBufSurface CUDA mapping cleanup failed: " +
                      cleanup_failure_message(*existing));
    }
    if (!same_owner(existing->decoder_owner, owner)) {
      throw NvmmError("NvBufSurface is already CUDA-mapped by a different decoder owner");
    }
    if (existing->runtime.get() != info.runtime.get()) {
      throw NvmmError("NvBufSurface is already CUDA-mapped by a different retained runtime");
    }
    if (!same_frame_info(existing->frame_info, info)) {
      throw NvmmError("NvBufSurface metadata changed while its CUDA mapping remained active");
    }
    return make_frame(existing->y_ptr, existing->uv_ptr, existing->y_provenance,
                      existing->uv_provenance, acquire_mapping_lease(existing, registry));
  }

  auto functions = info.runtime ? info.runtime->state_->functions : nvbuf_functions();
  if (mapped_cuda_buffer_handle(info) != nullptr || mapped_egl_image(info) != nullptr) {
    throw NvmmError("NvBufSurface already has an external CUDA or EGL mapping");
  }
  if (info.abi == NvbufSurfaceAbi::DeepStream9_1 && functions->map_cuda != nullptr &&
      functions->unmap_cuda != nullptr) {
    auto mapping = std::make_shared<CudaMappingState>();
    mapping->decoder_owner = owner;
    mapping->runtime = info.runtime;
    mapping->frame_info = info;
    mapping->functions = functions;
    mapping->cuda = cuda;
    mapping->surface = info.surface_ptr;
    mapping->kind = SurfaceMappingKind::CudaBuffer;
    registry->mappings.emplace(info.surface_ptr, mapping);
    if (functions->map_cuda(info.surface_ptr, 0) != 0) {
      if (mapped_cuda_buffer_handle(info) == nullptr || release_surface_mapping(*mapping)) {
        registry->mappings.erase(info.surface_ptr);
      }
      throw NvmmError("NvBufSurfaceMapCudaBuffer failed");
    }
    try {
      void* base = mapped_cuda_buffer_ptr(info);
      const auto y_ptr = checked_device_pointer(base, info.y_offset);
      const auto uv_ptr = checked_device_pointer(base, info.uv_offset);
      const auto y_provenance =
          validate_cuda_plane_pointer(cuda, y_ptr, info.y_pitch, info.width, info.height,
                                      info.y_size, static_cast<int>(info.gpu_id), "Y");
      const auto uv_provenance =
          validate_cuda_plane_pointer(cuda, uv_ptr, info.uv_pitch, info.width, info.height / 2U,
                                      info.uv_size, static_cast<int>(info.gpu_id), "UV");
      mapping->y_ptr = y_ptr;
      mapping->uv_ptr = uv_ptr;
      mapping->y_provenance = y_provenance;
      mapping->uv_provenance = uv_provenance;
      auto lease = acquire_mapping_lease(mapping, registry);
      return make_frame(y_ptr, uv_ptr, y_provenance, uv_provenance, std::move(lease));
    } catch (...) {
      if (release_surface_mapping(*mapping)) {
        registry->mappings.erase(info.surface_ptr);
      }
      throw;
    }
  }

  auto mapping = std::make_shared<CudaMappingState>();
  mapping->decoder_owner = owner;
  mapping->runtime = info.runtime;
  mapping->frame_info = info;
  mapping->functions = functions;
  mapping->cuda = cuda;
  mapping->surface = info.surface_ptr;
  mapping->kind = SurfaceMappingKind::EglImage;
  registry->mappings.emplace(info.surface_ptr, mapping);
  if (functions->map_egl(info.surface_ptr, 0) != 0) {
    if (mapped_egl_image(info) == nullptr || release_surface_mapping(*mapping)) {
      registry->mappings.erase(info.surface_ptr);
    }
    throw NvmmError("NvBufSurfaceMapEglImage failed");
  }
  CudaGraphicsResource graphics_resource = nullptr;
  try {
    void* egl_image = mapped_egl_image(info);
    if (egl_image == nullptr) {
      throw NvmmError("NvBufSurfaceMapEglImage returned no EGL image");
    }
    check_cuda("cuGraphicsEGLRegisterImage",
               cuda->register_egl_image(&graphics_resource, egl_image, 0));
    mapping->graphics_resource = graphics_resource;
    CudaEglFrame egl_frame;
    check_cuda("cuGraphicsResourceGetMappedEglFrame",
               cuda->get_mapped_egl_frame(&egl_frame, graphics_resource, 0, 0));
    if (egl_frame.frame_type != kCudaEglFrameTypePitch || egl_frame.plane_count != 2 ||
        egl_frame.width != info.width || egl_frame.height != info.height || egl_frame.depth != 1 ||
        egl_frame.pitch != info.y_pitch || egl_frame.channel_count != 1 ||
        egl_frame.color_format != expected_egl_color_format(info) ||
        egl_frame.array_format != kCudaArrayFormatUnsignedInt8 ||
        egl_frame.frame.pitches[0] == nullptr || egl_frame.frame.pitches[1] == nullptr) {
      throw NvmmError("CUDA EGL frame does not match pitch-linear NV12 metadata");
    }

    const auto y_ptr = checked_device_pointer(egl_frame.frame.pitches[0], 0);
    const auto uv_ptr = checked_device_pointer(egl_frame.frame.pitches[1], 0);
    const auto y_provenance =
        validate_cuda_plane_pointer(cuda, y_ptr, info.y_pitch, info.width, info.height, info.y_size,
                                    static_cast<int>(info.gpu_id), "Y");
    const auto uv_provenance =
        validate_cuda_plane_pointer(cuda, uv_ptr, info.uv_pitch, info.width, info.height / 2U,
                                    info.uv_size, static_cast<int>(info.gpu_id), "UV");
    mapping->y_ptr = y_ptr;
    mapping->uv_ptr = uv_ptr;
    mapping->y_provenance = y_provenance;
    mapping->uv_provenance = uv_provenance;
    auto lease = acquire_mapping_lease(mapping, registry);
    return make_frame(y_ptr, uv_ptr, y_provenance, uv_provenance, std::move(lease));
  } catch (...) {
    if (release_surface_mapping(*mapping)) {
      registry->mappings.erase(info.surface_ptr);
    }
    throw;
  }
#else
  throw NvmmError("NvBufSurface CUDA mapping is only supported on Linux");
#endif
}

} // namespace reco::io
