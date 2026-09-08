#include "reco/core/cuda_backend.hpp"
#include "reco/core/path.hpp"
#include "reco/core/windows_runtime_library.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace reco::core {
namespace {

using CUdevice = int;
using CUcontext = void*;
using CUfunction = void*;
using CUmodule = void*;
using CUresult = int;
using CUstream = void*;
using CUevent = void*;
#if defined(_WIN32)
#define RECO_CUDA_API __stdcall
#else
#define RECO_CUDA_API
#endif

using CUdeviceptr = unsigned long long;
using CUmemGenericAllocationHandle = unsigned long long;
using CudaMemAccessFlags = unsigned long long;

static_assert(sizeof(void*) == 8, "the CUDA backend supports only 64-bit targets");
static_assert(sizeof(CUdeviceptr) == sizeof(void*));
static_assert(sizeof(CUmemGenericAllocationHandle) == 8);

constexpr CUresult kCudaSuccess = 0;
constexpr unsigned int kStreamNonBlocking = 1;
constexpr unsigned int kEventDisableTiming = 2;
constexpr unsigned int kMemoryTypeHost = 1;
constexpr unsigned int kMemoryTypeDevice = 2;
constexpr int kPointerAttributeContext = 1;
constexpr int kPointerAttributeMemoryType = 2;
constexpr int kPointerAttributeDeviceOrdinal = 9;
constexpr int kPointerAttributeRangeStartAddress = 11;
constexpr int kPointerAttributeRangeSize = 12;
constexpr int kPointerAttributeMapped = 13;
constexpr int kPointerAttributeAccessFlags = 16;
constexpr int kPointerAttributeMappingSize = 18;
constexpr int kPointerAttributeMappingBaseAddress = 19;
constexpr int kPointerAttributeMemoryBlockId = 20;
constexpr unsigned int kMemAllocationTypePinned = 1;
constexpr unsigned int kMemLocationTypeDevice = 1;
#if defined(_WIN32)
constexpr unsigned int kMemHandleType = 2;
#else
constexpr unsigned int kMemHandleType = 1;
#endif
constexpr unsigned int kMemAccessFlagsProtReadWrite = 3;
constexpr unsigned int kMemAccessFlagsProtRead = 1;
constexpr unsigned int kMemAllocGranularityMinimum = 0;
constexpr int kDeviceAttributeComputeCapabilityMajor = 75;
constexpr int kDeviceAttributeComputeCapabilityMinor = 76;
constexpr std::size_t kMaximumDriverLibraryPathBytes = 32U * 1024U;

struct CUuuid {
  std::uint8_t bytes[16];
};

struct CudaMemLocation {
  unsigned int type = 0;
  int id = 0;
};

struct CudaMemAllocationProp {
  unsigned int type = 0;
  unsigned int requested_handle_types = 0;
  CudaMemLocation location;
  void* win32_handle_meta_data = nullptr;
  std::uint64_t reserved[8]{};
};

struct CudaMemAccessDesc {
  CudaMemLocation location;
  unsigned int flags = 0;
};

struct CudaMemcpy2D {
  std::size_t src_x_in_bytes = 0;
  std::size_t src_y = 0;
  unsigned int src_memory_type = 0;
  const void* src_host = nullptr;
  CUdeviceptr src_device = 0;
  const void* src_array = nullptr;
  std::size_t src_pitch = 0;
  std::size_t dst_x_in_bytes = 0;
  std::size_t dst_y = 0;
  unsigned int dst_memory_type = 0;
  void* dst_host = nullptr;
  CUdeviceptr dst_device = 0;
  const void* dst_array = nullptr;
  std::size_t dst_pitch = 0;
  std::size_t width_in_bytes = 0;
  std::size_t height = 0;
};

class DynamicLibrary {
public:
  explicit DynamicLibrary(std::string name) : name_(std::move(name)) {
#if defined(_WIN32)
    handle_ = detail::load_windows_runtime_library(path_from_utf8(name_));
#else
    handle_ = dlopen(name_.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw std::runtime_error("failed to load " + name_);
    }
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif
  }

  template <typename Fn> Fn symbol(const char* name) const {
#if defined(_WIN32)
    auto* sym = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* sym = dlsym(handle_, name);
#endif
    if (sym == nullptr) {
      throw std::runtime_error(std::string("missing CUDA symbol ") + name);
    }
    return reinterpret_cast<Fn>(sym);
  }

  template <typename Fn> Fn optional_symbol(const char* name) const noexcept {
#if defined(_WIN32)
    auto* sym = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* sym = dlsym(handle_, name);
#endif
    return reinterpret_cast<Fn>(sym);
  }

private:
  std::string name_;
  void* handle_ = nullptr;
};

std::string default_cuda_driver_name() {
#if defined(_WIN32)
  return "nvcuda.dll";
#else
  return "libcuda.so.1";
#endif
}

[[noreturn]] void throw_cuda(const char* function, CUresult result) {
  throw std::runtime_error(std::string(function) + " returned CUDA error " +
                           std::to_string(result));
}

void check_cuda(const char* function, CUresult result) {
  if (result != kCudaSuccess) {
    throw_cuda(function, result);
  }
}

void check_cuda_pointer(const char* function, CUresult result) {
  if (result != kCudaSuccess) {
    throw std::invalid_argument(std::string(function) +
                                " rejected the CUDA device pointer (error " +
                                std::to_string(result) + ")");
  }
}

void validate_2d_shape(std::size_t src_pitch, std::size_t dst_pitch, std::size_t width_bytes,
                       std::size_t height) {
  if (width_bytes == 0 || height == 0) {
    throw std::invalid_argument("CUDA 2D copy dimensions must be non-zero");
  }
  if (src_pitch < width_bytes) {
    throw std::invalid_argument("CUDA 2D copy source pitch is smaller than width");
  }
  if (dst_pitch < width_bytes) {
    throw std::invalid_argument("CUDA 2D copy destination pitch is smaller than width");
  }
}

void validate_dim3(CudaDim3 dim, const char* name) {
  if (dim.x == 0 || dim.y == 0 || dim.z == 0) {
    throw std::invalid_argument(std::string("CUDA kernel ") + name +
                                " dimensions must be non-zero");
  }
}

void validate_no_nul(std::string_view value, const char* name) {
  if (value.empty()) {
    throw std::invalid_argument(std::string("CUDA ") + name + " must be non-empty");
  }
  if (std::find(value.begin(), value.end(), '\0') != value.end()) {
    throw std::invalid_argument(std::string("CUDA ") + name + " must not contain NUL bytes");
  }
}

std::string validated_driver_library_path(std::string_view library_path) {
  validate_no_nul(library_path, "driver library path");
  if (library_path.size() > kMaximumDriverLibraryPathBytes) {
    throw std::invalid_argument("CUDA driver library path exceeds 32768 bytes");
  }
  return std::string(library_path);
}

void validate_ptx(std::string_view ptx) {
  if (ptx.empty()) {
    throw std::invalid_argument("CUDA PTX must be non-empty");
  }
  const auto nul = std::find(ptx.begin(), ptx.end(), '\0');
  if (nul != ptx.end() && nul + 1 != ptx.end()) {
    throw std::invalid_argument("CUDA PTX must not contain interior NUL bytes");
  }
}

std::size_t round_up_to_granularity(std::size_t bytes, std::size_t granularity) {
  if (granularity == 0) {
    throw std::runtime_error("CUDA VMM allocation granularity is zero");
  }
  const std::size_t remainder = bytes % granularity;
  if (remainder == 0) {
    return bytes;
  }
  const std::size_t delta = granularity - remainder;
  if (bytes > std::numeric_limits<std::size_t>::max() - delta) {
    throw std::overflow_error("CUDA VMM allocation size overflow");
  }
  return bytes + delta;
}

bool valid_shareable_handle(CudaShareableHandle handle) {
#if defined(_WIN32)
  return handle != nullptr;
#else
  return handle >= 0;
#endif
}

void close_shareable_handle(CudaShareableHandle handle) {
  if (!valid_shareable_handle(handle)) {
    return;
  }
#if defined(_WIN32)
  CloseHandle(handle);
#else
  close(handle);
#endif
}

CudaShareableHandle invalid_shareable_handle() {
#if defined(_WIN32)
  return nullptr;
#else
  return -1;
#endif
}

} // namespace

struct CudaBackend::Impl {
  explicit Impl(std::string driver_path) : driver(std::move(driver_path)) {
    cu_init = driver.symbol<decltype(cu_init)>("cuInit");
    cu_device_get_count = driver.symbol<decltype(cu_device_get_count)>("cuDeviceGetCount");
    cu_device_get = driver.symbol<decltype(cu_device_get)>("cuDeviceGet");
    cu_device_get_attribute =
        driver.symbol<decltype(cu_device_get_attribute)>("cuDeviceGetAttribute");
    cu_device_get_name = driver.symbol<decltype(cu_device_get_name)>("cuDeviceGetName");
    cu_device_get_uuid = driver.symbol<decltype(cu_device_get_uuid)>("cuDeviceGetUuid");
    cu_device_primary_ctx_retain =
        driver.symbol<decltype(cu_device_primary_ctx_retain)>("cuDevicePrimaryCtxRetain");
    cu_device_primary_ctx_release =
        driver.symbol<decltype(cu_device_primary_ctx_release)>("cuDevicePrimaryCtxRelease_v2");
    cu_ctx_get_current = driver.symbol<decltype(cu_ctx_get_current)>("cuCtxGetCurrent");
    cu_ctx_get_device = driver.symbol<decltype(cu_ctx_get_device)>("cuCtxGetDevice");
    cu_ctx_set_current = driver.symbol<decltype(cu_ctx_set_current)>("cuCtxSetCurrent");
    cu_ctx_synchronize = driver.symbol<decltype(cu_ctx_synchronize)>("cuCtxSynchronize");
    cu_mem_alloc = driver.symbol<decltype(cu_mem_alloc)>("cuMemAlloc_v2");
    cu_mem_alloc_pitch = driver.symbol<decltype(cu_mem_alloc_pitch)>("cuMemAllocPitch_v2");
    cu_mem_free = driver.symbol<decltype(cu_mem_free)>("cuMemFree_v2");
    cu_memset_d8 = driver.symbol<decltype(cu_memset_d8)>("cuMemsetD8_v2");
    cu_memcpy_2d = driver.symbol<decltype(cu_memcpy_2d)>("cuMemcpy2D_v2");
    cu_memcpy_dtoh = driver.symbol<decltype(cu_memcpy_dtoh)>("cuMemcpyDtoH_v2");
    cu_mem_get_info = driver.symbol<decltype(cu_mem_get_info)>("cuMemGetInfo_v2");
    cu_pointer_get_attribute =
        driver.symbol<decltype(cu_pointer_get_attribute)>("cuPointerGetAttribute");
    cu_mem_get_allocation_granularity =
        driver.symbol<decltype(cu_mem_get_allocation_granularity)>("cuMemGetAllocationGranularity");
    cu_mem_get_access = driver.symbol<decltype(cu_mem_get_access)>("cuMemGetAccess");
    cu_mem_retain_allocation_handle =
        driver.symbol<decltype(cu_mem_retain_allocation_handle)>("cuMemRetainAllocationHandle");
    cu_mem_address_reserve = driver.symbol<decltype(cu_mem_address_reserve)>("cuMemAddressReserve");
    cu_mem_create = driver.symbol<decltype(cu_mem_create)>("cuMemCreate");
    cu_mem_export_to_shareable_handle =
        driver.symbol<decltype(cu_mem_export_to_shareable_handle)>("cuMemExportToShareableHandle");
    cu_mem_map = driver.symbol<decltype(cu_mem_map)>("cuMemMap");
    cu_mem_set_access = driver.symbol<decltype(cu_mem_set_access)>("cuMemSetAccess");
    cu_mem_release = driver.symbol<decltype(cu_mem_release)>("cuMemRelease");
    cu_mem_unmap = driver.symbol<decltype(cu_mem_unmap)>("cuMemUnmap");
    cu_mem_address_free = driver.symbol<decltype(cu_mem_address_free)>("cuMemAddressFree");
    cu_module_load_data = driver.symbol<decltype(cu_module_load_data)>("cuModuleLoadData");
    cu_module_unload = driver.symbol<decltype(cu_module_unload)>("cuModuleUnload");
    cu_module_get_function = driver.symbol<decltype(cu_module_get_function)>("cuModuleGetFunction");
    cu_launch_kernel = driver.symbol<decltype(cu_launch_kernel)>("cuLaunchKernel");
    cu_stream_create = driver.optional_symbol<decltype(cu_stream_create)>("cuStreamCreate");
    cu_stream_destroy = driver.optional_symbol<decltype(cu_stream_destroy)>("cuStreamDestroy_v2");
    cu_stream_synchronize =
        driver.optional_symbol<decltype(cu_stream_synchronize)>("cuStreamSynchronize");
    cu_event_create = driver.optional_symbol<decltype(cu_event_create)>("cuEventCreate");
    cu_event_destroy = driver.optional_symbol<decltype(cu_event_destroy)>("cuEventDestroy_v2");
    cu_event_record = driver.optional_symbol<decltype(cu_event_record)>("cuEventRecord");
    cu_event_synchronize =
        driver.optional_symbol<decltype(cu_event_synchronize)>("cuEventSynchronize");
    check_cuda("cuInit", cu_init(0));
  }

  ~Impl() {
    for (const auto& [ordinal, context] : retained_contexts) {
      (void)context;
      CUdevice retained_device = 0;
      if (cu_device_get(&retained_device, ordinal) == kCudaSuccess) {
        (void)cu_device_primary_ctx_release(retained_device);
      }
    }
  }

  CUdevice device(int ordinal) const {
    CUdevice device = 0;
    check_cuda("cuDeviceGet", cu_device_get(&device, ordinal));
    return device;
  }

  void require_healthy_context() const {
    if (context_restore_failed.load(std::memory_order_acquire)) {
      throw std::runtime_error("CUDA caller context restoration previously failed");
    }
  }

  void mark_context_restore_failed() noexcept {
    context_restore_failed.store(true, std::memory_order_release);
  }

  void ensure_primary_context(int ordinal) {
    require_healthy_context();
    const CUdevice requested_device = device(ordinal);
    CUcontext context = nullptr;
    {
      std::lock_guard<std::mutex> lock(context_mutex);
      auto it = retained_contexts.find(ordinal);
      if (it == retained_contexts.end()) {
        check_cuda("cuDevicePrimaryCtxRetain",
                   cu_device_primary_ctx_retain(&context, requested_device));
        retained_contexts.emplace(ordinal, context);
      } else {
        context = it->second;
      }
    }
    CUcontext current = nullptr;
    check_cuda("cuCtxGetCurrent", cu_ctx_get_current(&current));
    if (current == context) {
      return;
    }
    check_cuda("cuCtxSetCurrent", cu_ctx_set_current(context));
  }

  void ensure_current_context_for_copy() { ensure_primary_context(0); }

  CUcontext current_context() {
    require_healthy_context();
    CUcontext current = nullptr;
    check_cuda("cuCtxGetCurrent", cu_ctx_get_current(&current));
    return current;
  }

  void set_current_context(CUcontext context) {
    require_healthy_context();
    check_cuda("cuCtxSetCurrent", cu_ctx_set_current(context));
  }

  void restore_context(CUcontext context) {
    const auto result = cu_ctx_set_current(context);
    if (result != kCudaSuccess) {
      mark_context_restore_failed();
      throw_cuda("cuCtxSetCurrent (restore)", result);
    }
  }

  void restore_context_noexcept(CUcontext context) noexcept {
    if (cu_ctx_set_current(context) != kCudaSuccess) {
      mark_context_restore_failed();
    }
  }

  DynamicLibrary driver;
  std::mutex context_mutex;
  std::unordered_map<int, CUcontext> retained_contexts;
  std::mutex execution_stream_mutex;
  std::unordered_map<int, std::weak_ptr<void>> execution_streams;
  std::atomic<bool> context_restore_failed{false};
  CUresult (*cu_init)(unsigned int) = nullptr;
  CUresult (*cu_device_get_count)(int*) = nullptr;
  CUresult (*cu_device_get)(CUdevice*, int) = nullptr;
  CUresult (*cu_device_get_attribute)(int*, int, CUdevice) = nullptr;
  CUresult (*cu_device_get_name)(char*, int, CUdevice) = nullptr;
  CUresult (*cu_device_get_uuid)(CUuuid*, CUdevice) = nullptr;
  CUresult (*cu_device_primary_ctx_retain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*cu_device_primary_ctx_release)(CUdevice) = nullptr;
  CUresult (*cu_ctx_get_current)(CUcontext*) = nullptr;
  CUresult (*cu_ctx_get_device)(CUdevice*) = nullptr;
  CUresult (*cu_ctx_set_current)(CUcontext) = nullptr;
  CUresult (*cu_ctx_synchronize)() = nullptr;
  CUresult (*cu_mem_alloc)(CUdeviceptr*, std::size_t) = nullptr;
  CUresult (*cu_mem_alloc_pitch)(CUdeviceptr*, std::size_t*, std::size_t, std::size_t,
                                 unsigned int) = nullptr;
  CUresult (*cu_mem_free)(CUdeviceptr) = nullptr;
  CUresult (*cu_memset_d8)(CUdeviceptr, unsigned char, std::size_t) = nullptr;
  CUresult (*cu_memcpy_2d)(const CudaMemcpy2D*) = nullptr;
  CUresult (*cu_memcpy_dtoh)(void*, CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cu_mem_get_info)(std::size_t*, std::size_t*) = nullptr;
  CUresult(RECO_CUDA_API* cu_pointer_get_attribute)(void*, int, CUdeviceptr) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_get_allocation_granularity)(std::size_t*,
                                                             const CudaMemAllocationProp*,
                                                             unsigned int) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_get_access)(CudaMemAccessFlags*, const CudaMemLocation*,
                                             CUdeviceptr) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_retain_allocation_handle)(CUmemGenericAllocationHandle*,
                                                           void*) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_address_reserve)(CUdeviceptr*, std::size_t, std::size_t,
                                                  CUdeviceptr, unsigned long long) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_create)(CUmemGenericAllocationHandle*, std::size_t,
                                         const CudaMemAllocationProp*,
                                         unsigned long long) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_export_to_shareable_handle)(void*, CUmemGenericAllocationHandle,
                                                             unsigned int,
                                                             unsigned long long) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_map)(CUdeviceptr, std::size_t, std::size_t,
                                      CUmemGenericAllocationHandle, unsigned long long) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_set_access)(CUdeviceptr, std::size_t, const CudaMemAccessDesc*,
                                             std::size_t) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_release)(CUmemGenericAllocationHandle) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_unmap)(CUdeviceptr, std::size_t) = nullptr;
  CUresult(RECO_CUDA_API* cu_mem_address_free)(CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cu_module_load_data)(CUmodule*, const void*) = nullptr;
  CUresult (*cu_module_unload)(CUmodule) = nullptr;
  CUresult (*cu_module_get_function)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*cu_launch_kernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int, CUstream, void**,
                               void**) = nullptr;
  CUresult (*cu_stream_create)(CUstream*, unsigned int) = nullptr;
  CUresult (*cu_stream_destroy)(CUstream) = nullptr;
  CUresult (*cu_stream_synchronize)(CUstream) = nullptr;
  CUresult (*cu_event_create)(CUevent*, unsigned int) = nullptr;
  CUresult (*cu_event_destroy)(CUevent) = nullptr;
  CUresult (*cu_event_record)(CUevent, CUstream) = nullptr;
  CUresult (*cu_event_synchronize)(CUevent) = nullptr;
};

#undef RECO_CUDA_API

namespace {

template <typename Backend> class PrimaryContextScope {
public:
  PrimaryContextScope(Backend& backend, int device_ordinal)
      : backend_(&backend), previous_(backend.current_context()) {
    try {
      backend_->ensure_primary_context(device_ordinal);
      current_ = backend_->current_context();
    } catch (...) {
      backend_->restore_context_noexcept(previous_);
      throw;
    }
  }

  PrimaryContextScope(const PrimaryContextScope&) = delete;
  PrimaryContextScope& operator=(const PrimaryContextScope&) = delete;

  ~PrimaryContextScope() {
    if (backend_ != nullptr) {
      backend_->restore_context_noexcept(previous_);
    }
  }

  [[nodiscard]] CUcontext current() const { return current_; }

  void restore() {
    if (backend_ == nullptr) {
      return;
    }
    backend_->restore_context(previous_);
    backend_ = nullptr;
  }

private:
  Backend* backend_ = nullptr;
  CUcontext previous_ = nullptr;
  CUcontext current_ = nullptr;
};

void validate_access_flags(std::uint64_t access_flags, CudaSpanAccess required_access) {
  if ((access_flags & kMemAccessFlagsProtRead) == 0U) {
    throw std::invalid_argument("CUDA device span does not permit device reads");
  }
  if (required_access == CudaSpanAccess::ReadWrite &&
      (access_flags & kMemAccessFlagsProtReadWrite) != kMemAccessFlagsProtReadWrite) {
    throw std::invalid_argument("CUDA device span does not permit device writes");
  }
}

struct VmmProvenanceRange {
  unsigned long long memory_block_id = 0;
  CudaDevicePtr mapping_base = 0;
  std::size_t mapping_size = 0;
  std::size_t first_offset = 0;
  std::size_t last_offset = 0;
};

void append_vmm_provenance_range(std::vector<VmmProvenanceRange>& ranges,
                                 VmmProvenanceRange range) {
  if (!ranges.empty()) {
    auto& previous = ranges.back();
    if (previous.memory_block_id == range.memory_block_id &&
        previous.mapping_base == range.mapping_base &&
        previous.mapping_size == range.mapping_size &&
        previous.last_offset + 1U == range.first_offset) {
      previous.last_offset = range.last_offset;
      return;
    }
  }
  ranges.push_back(range);
}

} // namespace

struct CudaValidatedSpan::State {
  State(std::shared_ptr<CudaBackend::Impl> backend_in, CudaDevicePtr ptr_in, std::size_t size_in,
        std::uintptr_t context_id_in, int device_ordinal_in, CudaSpanAccess access_in,
        CudaDevicePtr validated_range_base_in, std::size_t validated_range_bytes_in,
        std::vector<CUmemGenericAllocationHandle> allocation_handles_in,
        std::vector<VmmProvenanceRange> vmm_provenance_ranges_in, bool is_vmm_in,
        bool vmm_identity_complete_in)
      : backend(std::move(backend_in)), ptr(ptr_in), size(size_in), context_id(context_id_in),
        device_ordinal(device_ordinal_in), access(access_in),
        validated_range_base(validated_range_base_in),
        validated_range_bytes(validated_range_bytes_in),
        allocation_handles(std::move(allocation_handles_in)),
        vmm_provenance_ranges(std::move(vmm_provenance_ranges_in)), is_vmm(is_vmm_in),
        vmm_identity_complete(vmm_identity_complete_in) {}

  State(const State&) = delete;
  State& operator=(const State&) = delete;

  ~State() {
    if (backend == nullptr) {
      return;
    }
    for (const auto handle : allocation_handles) {
      (void)backend->cu_mem_release(handle);
    }
  }

  std::shared_ptr<CudaBackend::Impl> backend;
  CudaDevicePtr ptr = 0;
  std::size_t size = 0;
  std::uintptr_t context_id = 0;
  int device_ordinal = -1;
  CudaSpanAccess access = CudaSpanAccess::Read;
  CudaDevicePtr validated_range_base = 0;
  std::size_t validated_range_bytes = 0;
  std::vector<CUmemGenericAllocationHandle> allocation_handles;
  std::vector<VmmProvenanceRange> vmm_provenance_ranges;
  bool is_vmm = false;
  bool vmm_identity_complete = false;
};

struct CudaExecutionStream::State {
  State(std::shared_ptr<CudaBackend::Impl> backend_in, CUcontext context_in, int device_ordinal_in,
        CUstream stream_in, CUevent completion_event_in)
      : backend(std::move(backend_in)), context(context_in), device_ordinal(device_ordinal_in),
        stream(stream_in), completion_event(completion_event_in) {}

  State(const State&) = delete;
  State& operator=(const State&) = delete;

  ~State() {
    if (backend == nullptr || context == nullptr) {
      return;
    }
    try {
      const CUcontext previous_context = backend->current_context();
      backend->set_current_context(context);
      if (completion_event != nullptr) {
        (void)backend->cu_event_destroy(completion_event);
      }
      if (stream != nullptr) {
        (void)backend->cu_stream_destroy(stream);
      }
      backend->restore_context(previous_context);
    } catch (...) {
    }
  }

  std::shared_ptr<CudaBackend::Impl> backend;
  CUcontext context = nullptr;
  int device_ordinal = -1;
  CUstream stream = nullptr;
  CUevent completion_event = nullptr;
  std::mutex submission_mutex;
};

CudaExecutionStream::CudaExecutionStream(std::shared_ptr<State> state) : state_(std::move(state)) {}

std::uintptr_t CudaExecutionStream::context_id() const {
  if (state_ == nullptr) {
    throw std::logic_error("cannot inspect an empty CUDA execution stream");
  }
  return reinterpret_cast<std::uintptr_t>(state_->context);
}

int CudaExecutionStream::device_ordinal() const {
  if (state_ == nullptr) {
    throw std::logic_error("cannot inspect an empty CUDA execution stream");
  }
  return state_->device_ordinal;
}

CudaValidatedSpan::CudaValidatedSpan(std::shared_ptr<State> state) : state_(std::move(state)) {}

CudaDevicePtr CudaValidatedSpan::ptr() const {
  if (!state_) {
    throw std::logic_error("cannot inspect an empty CUDA validated span");
  }
  return state_->ptr;
}

std::size_t CudaValidatedSpan::size() const {
  if (!state_) {
    throw std::logic_error("cannot inspect an empty CUDA validated span");
  }
  return state_->size;
}

std::uintptr_t CudaValidatedSpan::context_id() const {
  if (!state_) {
    throw std::logic_error("cannot inspect an empty CUDA validated span");
  }
  return state_->context_id;
}

int CudaValidatedSpan::device_ordinal() const {
  if (!state_) {
    throw std::logic_error("cannot inspect an empty CUDA validated span");
  }
  return state_->device_ordinal;
}

CudaSpanAccess CudaValidatedSpan::access() const {
  if (!state_) {
    throw std::logic_error("cannot inspect an empty CUDA validated span");
  }
  return state_->access;
}

CudaDevicePtr CudaValidatedSpan::validated_range_base() const {
  if (!state_) {
    throw std::logic_error("cannot inspect an empty CUDA validated span");
  }
  return state_->validated_range_base;
}

std::size_t CudaValidatedSpan::validated_range_bytes() const {
  if (!state_) {
    throw std::logic_error("cannot inspect an empty CUDA validated span");
  }
  return state_->validated_range_bytes;
}

bool CudaValidatedSpan::permits(CudaSpanAccess required_access) const {
  if (!state_) {
    return false;
  }
  if (required_access == CudaSpanAccess::Read) {
    return state_->access == CudaSpanAccess::Read || state_->access == CudaSpanAccess::ReadWrite;
  }
  return required_access == CudaSpanAccess::ReadWrite &&
         state_->access == CudaSpanAccess::ReadWrite;
}

bool CudaValidatedSpan::aliases(const CudaValidatedSpan& other) const {
  if (!state_ || !other.state_) {
    throw std::logic_error("cannot compare an empty CUDA validated span");
  }
  const auto this_last = state_->ptr + state_->size - 1U;
  const auto other_last = other.state_->ptr + other.state_->size - 1U;
  if (state_->ptr <= other_last && other.state_->ptr <= this_last) {
    return true;
  }
  if (!state_->is_vmm || !other.state_->is_vmm) {
    return false;
  }
  if (!state_->vmm_identity_complete || !other.state_->vmm_identity_complete) {
    return true;
  }
  for (const auto& range : state_->vmm_provenance_ranges) {
    for (const auto& other_range : other.state_->vmm_provenance_ranges) {
      if (range.memory_block_id != other_range.memory_block_id) {
        continue;
      }
      // CUDA identifies the physical allocation but does not expose a retained mapping's
      // cuMemMap offset. Distinct mappings of one allocation must therefore fail closed.
      if (range.mapping_base != other_range.mapping_base ||
          range.mapping_size != other_range.mapping_size) {
        return true;
      }
      if (range.first_offset <= other_range.last_offset &&
          other_range.first_offset <= range.last_offset) {
        return true;
      }
    }
  }
  return false;
}

struct CudaModule::State {
  State(std::shared_ptr<CudaBackend::Impl> backend_in, CUcontext context_in, CUmodule module_in)
      : backend(std::move(backend_in)), context(context_in), module(module_in) {}

  State(const State&) = delete;
  State& operator=(const State&) = delete;

  ~State() { unload(backend, context, module); }

  static void unload(const std::shared_ptr<CudaBackend::Impl>& backend, CUcontext context,
                     CUmodule module) noexcept {
    if (backend == nullptr || context == nullptr || module == nullptr) {
      return;
    }
    try {
      const CUcontext previous_context = backend->current_context();
      backend->set_current_context(context);
      try {
        check_cuda("cuModuleUnload", backend->cu_module_unload(module));
      } catch (...) {
        backend->restore_context_noexcept(previous_context);
        throw;
      }
      backend->restore_context(previous_context);
    } catch (...) {
    }
  }

  std::shared_ptr<CudaBackend::Impl> backend;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
};

CudaDeviceBuffer::CudaDeviceBuffer(CudaDevicePtr ptr, std::size_t size,
                                   std::function<void(CudaDevicePtr)> free)
    : ptr_(ptr), size_(size), free_(std::move(free)) {}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, 0)), size_(std::exchange(other.size_, 0)),
      free_(std::move(other.free_)) {}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = std::exchange(other.ptr_, 0);
    size_ = std::exchange(other.size_, 0);
    free_ = std::move(other.free_);
  }
  return *this;
}

CudaDeviceBuffer::~CudaDeviceBuffer() { reset(); }

void CudaDeviceBuffer::reset() {
  if (ptr_ == 0) {
    return;
  }
  try {
    if (free_) {
      free_(ptr_);
    }
  } catch (...) {
  }
  ptr_ = 0;
  size_ = 0;
  free_ = {};
}

CudaSharedMemory::CudaSharedMemory(std::shared_ptr<CudaBackend::Impl> backend, void* context,
                                   CudaDevicePtr ptr, std::size_t size,
                                   CudaShareableHandle shareable_handle)
    : backend_(std::move(backend)), context_(context), ptr_(ptr), size_(size),
      shareable_handle_(shareable_handle),
      owns_shareable_handle_(valid_shareable_handle(shareable_handle)) {}

CudaSharedMemory::CudaSharedMemory(CudaSharedMemory&& other) noexcept
    : backend_(std::move(other.backend_)), context_(std::exchange(other.context_, nullptr)),
      ptr_(std::exchange(other.ptr_, 0)), size_(std::exchange(other.size_, 0)),
      shareable_handle_(std::exchange(other.shareable_handle_, invalid_shareable_handle())),
      owns_shareable_handle_(std::exchange(other.owns_shareable_handle_, false)) {}

CudaSharedMemory& CudaSharedMemory::operator=(CudaSharedMemory&& other) noexcept {
  if (this != &other) {
    reset();
    backend_ = std::move(other.backend_);
    context_ = std::exchange(other.context_, nullptr);
    ptr_ = std::exchange(other.ptr_, 0);
    size_ = std::exchange(other.size_, 0);
    shareable_handle_ = std::exchange(other.shareable_handle_, invalid_shareable_handle());
    owns_shareable_handle_ = std::exchange(other.owns_shareable_handle_, false);
  }
  return *this;
}

CudaSharedMemory::~CudaSharedMemory() { reset(); }

CudaShareableHandle CudaSharedMemory::release_shareable_handle() {
  owns_shareable_handle_ = false;
  return std::exchange(shareable_handle_, invalid_shareable_handle());
}

void CudaSharedMemory::reset() {
  if (owns_shareable_handle_) {
    close_shareable_handle(shareable_handle_);
  }
  shareable_handle_ = invalid_shareable_handle();
  owns_shareable_handle_ = false;

  if (ptr_ == 0) {
    size_ = 0;
    context_ = nullptr;
    backend_.reset();
    return;
  }
  try {
    const CUcontext previous_context = backend_->current_context();
    backend_->set_current_context(static_cast<CUcontext>(context_));
    const auto unmap_result = backend_->cu_mem_unmap(ptr_, size_);
    (void)unmap_result;
    const auto free_result = backend_->cu_mem_address_free(ptr_, size_);
    (void)free_result;
    backend_->restore_context(previous_context);
  } catch (...) {
  }
  ptr_ = 0;
  size_ = 0;
  context_ = nullptr;
  backend_.reset();
}

CudaModule::CudaModule(std::shared_ptr<State> state) : state_(std::move(state)) {}

CudaModule::CudaModule(CudaModule&& other) noexcept : state_(std::move(other.state_)) {}

CudaModule& CudaModule::operator=(CudaModule&& other) noexcept {
  if (this != &other) {
    state_ = std::move(other.state_);
  }
  return *this;
}

CudaModule::~CudaModule() = default;

CudaKernel CudaModule::load_kernel(std::string_view function_name) const {
  validate_no_nul(function_name, "kernel function name");
  const auto state = state_;
  if (state == nullptr) {
    throw std::invalid_argument("CUDA kernel load requires a live module");
  }

  std::string terminated_name(function_name);
  terminated_name.push_back('\0');
  const CUcontext previous_context = state->backend->current_context();
  state->backend->set_current_context(state->context);
  CUfunction function = nullptr;
  try {
    check_cuda("cuModuleGetFunction", state->backend->cu_module_get_function(
                                          &function, state->module, terminated_name.data()));
  } catch (...) {
    state->backend->restore_context_noexcept(previous_context);
    throw;
  }
  state->backend->restore_context(previous_context);
  return CudaKernel(state, function);
}

void CudaModule::reset() { state_.reset(); }

CudaKernel::CudaKernel(std::shared_ptr<CudaModule::State> module, void* function)
    : module_state_(std::move(module)), function_(function) {}

CudaKernel::CudaKernel(CudaKernel&& other) noexcept
    : module_state_(std::move(other.module_state_)),
      function_(std::exchange(other.function_, nullptr)) {}

CudaKernel& CudaKernel::operator=(CudaKernel&& other) noexcept {
  if (this != &other) {
    reset();
    module_state_ = std::move(other.module_state_);
    function_ = std::exchange(other.function_, nullptr);
  }
  return *this;
}

CudaKernel::~CudaKernel() { reset(); }

void CudaKernel::launch_and_synchronize(const CudaLaunchConfig& config,
                                        std::span<void*> args) const {
  if (!*this) {
    throw std::invalid_argument("CUDA kernel launch requires a live kernel");
  }
  validate_dim3(config.grid, "grid");
  validate_dim3(config.block, "block");
  const auto& backend = module_state_->backend;
  const CUcontext previous_context = backend->current_context();
  backend->set_current_context(module_state_->context);
  void** kernel_args = args.empty() ? nullptr : args.data();
  try {
    check_cuda("cuLaunchKernel",
               backend->cu_launch_kernel(static_cast<CUfunction>(function_), config.grid.x,
                                         config.grid.y, config.grid.z, config.block.x,
                                         config.block.y, config.block.z, config.shared_memory_bytes,
                                         nullptr, kernel_args, nullptr));
    check_cuda("cuCtxSynchronize", backend->cu_ctx_synchronize());
  } catch (...) {
    backend->restore_context_noexcept(previous_context);
    throw;
  }
  backend->restore_context(previous_context);
}

void CudaKernel::launch_and_synchronize(const CudaExecutionStream& stream,
                                        const CudaLaunchConfig& config,
                                        std::span<void*> args) const {
  if (!*this) {
    throw std::invalid_argument("CUDA kernel launch requires a live kernel");
  }
  const auto stream_state = stream.state_;
  if (stream_state == nullptr) {
    throw std::invalid_argument("CUDA kernel launch requires a live execution stream");
  }
  validate_dim3(config.grid, "grid");
  validate_dim3(config.block, "block");
  const auto& backend = module_state_->backend;
  if (stream_state->backend != backend || stream_state->context != module_state_->context) {
    throw std::invalid_argument("CUDA kernel and execution stream must share one context");
  }

  std::lock_guard<std::mutex> lock(stream_state->submission_mutex);
  const CUcontext previous_context = backend->current_context();
  backend->set_current_context(module_state_->context);
  void** kernel_args = args.empty() ? nullptr : args.data();
  bool launch_submitted = false;
  try {
    check_cuda("cuLaunchKernel",
               backend->cu_launch_kernel(static_cast<CUfunction>(function_), config.grid.x,
                                         config.grid.y, config.grid.z, config.block.x,
                                         config.block.y, config.block.z, config.shared_memory_bytes,
                                         stream_state->stream, kernel_args, nullptr));
    launch_submitted = true;
    check_cuda("cuEventRecord",
               backend->cu_event_record(stream_state->completion_event, stream_state->stream));
    check_cuda("cuEventSynchronize", backend->cu_event_synchronize(stream_state->completion_event));
  } catch (...) {
    if (launch_submitted) {
      (void)backend->cu_stream_synchronize(stream_state->stream);
    }
    backend->restore_context_noexcept(previous_context);
    throw;
  }
  backend->restore_context(previous_context);
}

void CudaKernel::launch(const CudaLaunchConfig& config, std::span<void*> args) const {
  if (!*this) {
    throw std::invalid_argument("CUDA kernel launch requires a live kernel");
  }
  validate_dim3(config.grid, "grid");
  validate_dim3(config.block, "block");
  const auto& backend = module_state_->backend;
  const CUcontext previous_context = backend->current_context();
  backend->set_current_context(module_state_->context);
  void** kernel_args = args.empty() ? nullptr : args.data();
  try {
    check_cuda("cuLaunchKernel",
               backend->cu_launch_kernel(static_cast<CUfunction>(function_), config.grid.x,
                                         config.grid.y, config.grid.z, config.block.x,
                                         config.block.y, config.block.z, config.shared_memory_bytes,
                                         nullptr, kernel_args, nullptr));
  } catch (...) {
    backend->restore_context_noexcept(previous_context);
    throw;
  }
  backend->restore_context(previous_context);
}

void CudaKernel::synchronize() const {
  if (!*this) {
    throw std::invalid_argument("CUDA kernel synchronize requires a live kernel");
  }
  const auto& backend = module_state_->backend;
  const CUcontext previous_context = backend->current_context();
  backend->set_current_context(module_state_->context);
  try {
    check_cuda("cuCtxSynchronize", backend->cu_ctx_synchronize());
  } catch (...) {
    backend->restore_context_noexcept(previous_context);
    throw;
  }
  backend->restore_context(previous_context);
}

void CudaKernel::reset() {
  function_ = nullptr;
  module_state_.reset();
}

bool CudaBackend::is_available() {
  try {
    const auto backend = create();
    return backend.device_count() > 0;
  } catch (...) {
    return false;
  }
}

std::string CudaBackend::availability_error() {
  try {
    const auto backend = create();
    if (backend.device_count() <= 0) {
      return "CUDA driver loaded but no CUDA devices were reported";
    }
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}

std::string CudaBackend::availability_error(std::string_view library_path) {
  try {
    const auto backend =
        CudaBackend(std::make_shared<Impl>(validated_driver_library_path(library_path)));
    if (backend.device_count() <= 0) {
      return "CUDA driver loaded but no CUDA devices were reported";
    }
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}

CudaBackend CudaBackend::create() {
  static const std::shared_ptr<Impl> impl = std::make_shared<Impl>(default_cuda_driver_name());
  return CudaBackend(impl);
}

CudaBackend CudaBackend::load(std::string_view library_path) {
  const auto path = validated_driver_library_path(library_path);
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, std::shared_ptr<Impl>> cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  if (const auto it = cache.find(path); it != cache.end()) {
    return CudaBackend(it->second);
  }
  auto impl = std::make_shared<Impl>(path);
  cache.emplace(path, impl);
  return CudaBackend(std::move(impl));
}

CudaBackend::CudaBackend(std::shared_ptr<Impl> impl,
                         std::shared_ptr<CudaBackendTraceSink> trace_sink)
    : impl_(std::move(impl)), trace_sink_(std::move(trace_sink)) {}

CudaBackend CudaBackend::with_trace_sink(std::shared_ptr<CudaBackendTraceSink> trace_sink) const {
  if (!trace_sink) {
    throw std::invalid_argument("CUDA trace sink must be non-null");
  }
  return CudaBackend(impl_, std::move(trace_sink));
}

bool CudaBackend::context_healthy() const noexcept {
  return !impl_->context_restore_failed.load(std::memory_order_acquire);
}

std::string CudaBackend::context_health_error() const {
  if (context_healthy()) {
    return {};
  }
  return "CUDA caller context restoration previously failed";
}

int CudaBackend::device_count() const {
  int count = 0;
  check_cuda("cuDeviceGetCount", impl_->cu_device_get_count(&count));
  return count;
}

CudaDeviceInfo CudaBackend::device_info(int ordinal) const {
  const CUdevice device = impl_->device(ordinal);

  std::array<char, 256> name{};
  check_cuda("cuDeviceGetName",
             impl_->cu_device_get_name(name.data(), static_cast<int>(name.size()), device));
  CUuuid uuid{};
  check_cuda("cuDeviceGetUuid", impl_->cu_device_get_uuid(&uuid, device));

  CudaDeviceInfo info;
  info.ordinal = ordinal;
  info.name = name.data();
  std::memcpy(info.uuid.data(), uuid.bytes, info.uuid.size());
  return info;
}

CudaComputeCapability CudaBackend::compute_capability(int ordinal) const {
  const CUdevice device = impl_->device(ordinal);
  CudaComputeCapability capability;
  check_cuda("cuDeviceGetAttribute (compute capability major)",
             impl_->cu_device_get_attribute(&capability.major,
                                            kDeviceAttributeComputeCapabilityMajor, device));
  check_cuda("cuDeviceGetAttribute (compute capability minor)",
             impl_->cu_device_get_attribute(&capability.minor,
                                            kDeviceAttributeComputeCapabilityMinor, device));
  if (capability.major <= 0 || capability.minor < 0 || capability.minor > 9) {
    throw std::runtime_error("CUDA driver returned an invalid compute capability");
  }
  return capability;
}

void CudaBackend::ensure_primary_context(int ordinal) const {
  impl_->ensure_primary_context(ordinal);
}

std::uintptr_t CudaBackend::primary_context_id(int ordinal) const {
  PrimaryContextScope scope(*impl_, ordinal);
  const auto context = scope.current();
  if (context == nullptr) {
    throw std::runtime_error("CUDA primary context identity is null");
  }
  const auto identity = reinterpret_cast<std::uintptr_t>(context);
  scope.restore();
  return identity;
}

CudaMemoryInfo CudaBackend::memory_info() const {
  impl_->ensure_primary_context(0);
  CudaMemoryInfo info;
  check_cuda("cuMemGetInfo_v2", impl_->cu_mem_get_info(&info.free_bytes, &info.total_bytes));
  return info;
}

CudaDeviceBuffer CudaBackend::allocate(std::size_t bytes) const {
  if (bytes == 0) {
    throw std::invalid_argument("CUDA allocation size must be non-zero");
  }
  auto impl = impl_;
  auto trace_sink = trace_sink_;
  std::function<void(CudaDevicePtr)> deleter = [impl, trace_sink, bytes](CudaDevicePtr ptr) {
    impl->ensure_primary_context(0);
    check_cuda("cuMemFree_v2", impl->cu_mem_free(ptr));
    if (trace_sink) {
      trace_sink->device_allocation_released(bytes);
    }
  };
  impl_->ensure_primary_context(0);
  CUdeviceptr ptr = 0;
  check_cuda("cuMemAlloc_v2", impl_->cu_mem_alloc(&ptr, bytes));
  if (trace_sink_) {
    trace_sink_->device_allocation_created(bytes);
  }
  return CudaDeviceBuffer(ptr, bytes, std::move(deleter));
}

CudaPitchedAllocation CudaBackend::allocate_pitched(std::size_t width_bytes, std::size_t height,
                                                    unsigned int element_size_bytes) const {
  if (width_bytes == 0 || height == 0) {
    throw std::invalid_argument("CUDA pitched allocation dimensions must be non-zero");
  }
  if (element_size_bytes != 4 && element_size_bytes != 8 && element_size_bytes != 16) {
    throw std::invalid_argument("CUDA pitched allocation element size must be 4, 8, or 16");
  }
  if (width_bytes > std::numeric_limits<std::size_t>::max() / height) {
    throw std::overflow_error("CUDA pitched allocation size overflow");
  }

  impl_->ensure_primary_context(0);
  CUdeviceptr ptr = 0;
  std::size_t pitch = 0;
  check_cuda("cuMemAllocPitch_v2",
             impl_->cu_mem_alloc_pitch(&ptr, &pitch, width_bytes, height, element_size_bytes));
  if (pitch < width_bytes || pitch > std::numeric_limits<std::size_t>::max() / height) {
    const auto free_result = impl_->cu_mem_free(ptr);
    (void)free_result;
    throw std::runtime_error("CUDA pitched allocation returned an invalid pitch");
  }

  const std::size_t size = pitch * height;
  try {
    auto impl = impl_;
    auto trace_sink = trace_sink_;
    std::function<void(CudaDevicePtr)> deleter = [impl, trace_sink,
                                                  size](CudaDevicePtr allocation) {
      impl->ensure_primary_context(0);
      check_cuda("cuMemFree_v2", impl->cu_mem_free(allocation));
      if (trace_sink) {
        trace_sink->device_allocation_released(size);
      }
    };
    if (trace_sink_) {
      trace_sink_->device_allocation_created(size);
    }
    return {.buffer = CudaDeviceBuffer(ptr, size, std::move(deleter)), .pitch = pitch};
  } catch (...) {
    const auto free_result = impl_->cu_mem_free(ptr);
    (void)free_result;
    throw;
  }
}

CudaSharedMemory CudaBackend::allocate_shared_memory(std::size_t bytes) const {
  if (bytes == 0) {
    throw std::invalid_argument("CUDA shared allocation size must be non-zero");
  }
  impl_->ensure_current_context_for_copy();
  const CUcontext context = impl_->current_context();
  if (context == nullptr) {
    throw std::runtime_error("CUDA shared allocation did not establish a current context");
  }
  CUdevice current_device = 0;
  check_cuda("cuCtxGetDevice", impl_->cu_ctx_get_device(&current_device));

  const CudaMemAllocationProp prop = {
      .type = kMemAllocationTypePinned,
      .requested_handle_types = kMemHandleType,
      .location = {.type = kMemLocationTypeDevice, .id = current_device},
  };
  std::size_t granularity = 0;
  check_cuda(
      "cuMemGetAllocationGranularity",
      impl_->cu_mem_get_allocation_granularity(&granularity, &prop, kMemAllocGranularityMinimum));
  const std::size_t alloc_size = round_up_to_granularity(bytes, granularity);

  CUdeviceptr ptr = 0;
  CUmemGenericAllocationHandle allocation_handle = 0;
  bool allocation_handle_live = false;
  bool mapped = false;
  CudaShareableHandle shareable_handle = invalid_shareable_handle();

  try {
    check_cuda("cuMemAddressReserve",
               impl_->cu_mem_address_reserve(&ptr, alloc_size, granularity, 0, 0));
    check_cuda("cuMemCreate", impl_->cu_mem_create(&allocation_handle, alloc_size, &prop, 0));
    allocation_handle_live = true;
    check_cuda("cuMemExportToShareableHandle",
               impl_->cu_mem_export_to_shareable_handle(&shareable_handle, allocation_handle,
                                                        kMemHandleType, 0));
    check_cuda("cuMemMap", impl_->cu_mem_map(ptr, alloc_size, 0, allocation_handle, 0));
    mapped = true;
    check_cuda("cuMemRelease", impl_->cu_mem_release(allocation_handle));
    allocation_handle_live = false;
    const CudaMemAccessDesc access = {
        .location = {.type = kMemLocationTypeDevice, .id = current_device},
        .flags = kMemAccessFlagsProtReadWrite,
    };
    check_cuda("cuMemSetAccess", impl_->cu_mem_set_access(ptr, alloc_size, &access, 1));
  } catch (...) {
    if (allocation_handle_live) {
      const auto release_result = impl_->cu_mem_release(allocation_handle);
      (void)release_result;
    }
    if (mapped) {
      const auto unmap_result = impl_->cu_mem_unmap(ptr, alloc_size);
      (void)unmap_result;
    }
    if (ptr != 0) {
      const auto free_result = impl_->cu_mem_address_free(ptr, alloc_size);
      (void)free_result;
    }
    close_shareable_handle(shareable_handle);
    throw;
  }

  return CudaSharedMemory(impl_, context, ptr, alloc_size, shareable_handle);
}

void CudaBackend::memset_d8(const CudaDeviceBuffer& buffer, std::uint8_t value) const {
  if (!buffer) {
    throw std::invalid_argument("CUDA memset requires a live device buffer");
  }
  impl_->ensure_primary_context(0);
  check_cuda("cuMemsetD8_v2", impl_->cu_memset_d8(buffer.ptr(), value, buffer.size()));
}

std::vector<std::uint8_t> CudaBackend::copy_to_host(const CudaDeviceBuffer& buffer) const {
  if (!buffer) {
    throw std::invalid_argument("CUDA copy requires a live device buffer");
  }
  impl_->ensure_primary_context(0);
  std::vector<std::uint8_t> out(buffer.size());
  check_cuda("cuMemcpyDtoH_v2", impl_->cu_memcpy_dtoh(out.data(), buffer.ptr(), buffer.size()));
  if (trace_sink_) {
    trace_sink_->device_to_host_copy_submitted(buffer.size(), 1);
  }
  return out;
}

void CudaBackend::copy_host_to_device_2d(const CudaHostToDevice2DCopy& copy) const {
  if (copy.src == nullptr || copy.dst == 0) {
    throw std::invalid_argument("CUDA HtoD 2D copy requires live source and destination");
  }
  validate_2d_shape(copy.src_pitch, copy.dst_pitch, copy.width_bytes, copy.height);
  impl_->ensure_current_context_for_copy();
  const CudaMemcpy2D desc = {
      .src_memory_type = kMemoryTypeHost,
      .src_host = copy.src,
      .src_pitch = copy.src_pitch,
      .dst_memory_type = kMemoryTypeDevice,
      .dst_device = copy.dst,
      .dst_pitch = copy.dst_pitch,
      .width_in_bytes = copy.width_bytes,
      .height = copy.height,
  };
  check_cuda("cuMemcpy2D_v2 (HtoD)", impl_->cu_memcpy_2d(&desc));
}

void CudaBackend::copy_device_to_device_2d(const Cuda2DCopy& copy) const {
  if (copy.src == 0 || copy.dst == 0) {
    throw std::invalid_argument("CUDA DtoD 2D copy requires live source and destination");
  }
  validate_2d_shape(copy.src_pitch, copy.dst_pitch, copy.width_bytes, copy.height);
  impl_->ensure_current_context_for_copy();
  const CudaMemcpy2D desc = {
      .src_memory_type = kMemoryTypeDevice,
      .src_device = copy.src,
      .src_pitch = copy.src_pitch,
      .dst_memory_type = kMemoryTypeDevice,
      .dst_device = copy.dst,
      .dst_pitch = copy.dst_pitch,
      .width_in_bytes = copy.width_bytes,
      .height = copy.height,
  };
  check_cuda("cuMemcpy2D_v2 (DtoD)", impl_->cu_memcpy_2d(&desc));
  if (trace_sink_) {
    trace_sink_->device_to_device_copy_submitted();
  }
}

void CudaBackend::copy_device_to_host_2d(const CudaDeviceToHost2DCopy& copy) const {
  if (copy.dst == nullptr || copy.src == 0) {
    throw std::invalid_argument("CUDA DtoH 2D copy requires live source and destination");
  }
  validate_2d_shape(copy.src_pitch, copy.dst_pitch, copy.width_bytes, copy.height);
  impl_->ensure_current_context_for_copy();
  const CudaMemcpy2D desc = {
      .src_memory_type = kMemoryTypeDevice,
      .src_device = copy.src,
      .src_pitch = copy.src_pitch,
      .dst_memory_type = kMemoryTypeHost,
      .dst_host = copy.dst,
      .dst_pitch = copy.dst_pitch,
      .width_in_bytes = copy.width_bytes,
      .height = copy.height,
  };
  check_cuda("cuMemcpy2D_v2 (DtoH)", impl_->cu_memcpy_2d(&desc));
  if (trace_sink_) {
    trace_sink_->device_to_host_copy_submitted(copy.width_bytes, copy.height);
  }
}

CudaValidatedSpan CudaBackend::retain_device_span(CudaDevicePtr ptr, std::size_t accessible_bytes,
                                                  CudaSpanAccess required_access,
                                                  int device_ordinal) const {
  if (ptr == 0 || accessible_bytes == 0) {
    throw std::invalid_argument("CUDA device span requires a non-zero pointer and size");
  }
  if (device_ordinal < 0) {
    throw std::invalid_argument("CUDA device span ordinal must be non-negative");
  }
  if (required_access != CudaSpanAccess::Read && required_access != CudaSpanAccess::ReadWrite) {
    throw std::invalid_argument("CUDA device span access requirement is invalid");
  }
  if (accessible_bytes - 1U > std::numeric_limits<CudaDevicePtr>::max() - ptr) {
    throw std::overflow_error("CUDA device span address overflows the device pointer");
  }

  PrimaryContextScope scope(*impl_, device_ordinal);
  const CUcontext retained_context = scope.current();
  if (retained_context == nullptr) {
    throw std::runtime_error("CUDA device span validation did not establish a context");
  }

  CUcontext pointer_context = nullptr;
  unsigned int memory_type = 0;
  int pointer_device = -1;
  unsigned int mapped = 0;
  check_cuda_pointer(
      "cuPointerGetAttribute(CONTEXT)",
      impl_->cu_pointer_get_attribute(&pointer_context, kPointerAttributeContext, ptr));
  check_cuda_pointer(
      "cuPointerGetAttribute(MEMORY_TYPE)",
      impl_->cu_pointer_get_attribute(&memory_type, kPointerAttributeMemoryType, ptr));
  check_cuda_pointer(
      "cuPointerGetAttribute(DEVICE_ORDINAL)",
      impl_->cu_pointer_get_attribute(&pointer_device, kPointerAttributeDeviceOrdinal, ptr));
  check_cuda_pointer("cuPointerGetAttribute(MAPPED)",
                     impl_->cu_pointer_get_attribute(&mapped, kPointerAttributeMapped, ptr));
  // CUDA VMM mappings are context-independent and report a null owning context.
  if (pointer_context != nullptr && pointer_context != retained_context) {
    throw std::invalid_argument("CUDA device span belongs to a different CUDA context");
  }
  if (memory_type != kMemoryTypeDevice) {
    throw std::invalid_argument("CUDA device span does not reference device memory");
  }
  if (pointer_device != device_ordinal) {
    throw std::invalid_argument("CUDA device span belongs to a different CUDA device");
  }
  if (mapped == 0U) {
    throw std::invalid_argument("CUDA device span is not mapped to a live allocation");
  }

  CudaDevicePtr validated_range_base = ptr;
  std::size_t validated_range_bytes = accessible_bytes;
  std::vector<CUmemGenericAllocationHandle> allocation_handles;
  std::vector<VmmProvenanceRange> vmm_provenance_ranges;
  bool vmm_identity_complete = pointer_context == nullptr;
  if (pointer_context != nullptr) {
    unsigned int access_flags = 0;
    check_cuda_pointer(
        "cuPointerGetAttribute(ACCESS_FLAGS)",
        impl_->cu_pointer_get_attribute(&access_flags, kPointerAttributeAccessFlags, ptr));
    validate_access_flags(access_flags, required_access);

    std::size_t allocation_size = 0;
    CUdeviceptr allocation_base = 0;
    check_cuda_pointer(
        "cuPointerGetAttribute(RANGE_SIZE)",
        impl_->cu_pointer_get_attribute(&allocation_size, kPointerAttributeRangeSize, ptr));
    check_cuda_pointer(
        "cuPointerGetAttribute(RANGE_START_ADDR)",
        impl_->cu_pointer_get_attribute(&allocation_base, kPointerAttributeRangeStartAddress, ptr));
    if (ptr < allocation_base) {
      throw std::invalid_argument("CUDA device span precedes its allocation");
    }
    const auto allocation_offset = ptr - allocation_base;
    if (allocation_offset > allocation_size ||
        accessible_bytes > allocation_size - static_cast<std::size_t>(allocation_offset)) {
      throw std::invalid_argument("CUDA device span exceeds its allocation or mapping");
    }
    validated_range_base = allocation_base;
    validated_range_bytes = allocation_size;
  } else {
    const CudaMemAllocationProp prop = {
        .type = kMemAllocationTypePinned,
        .location = {.type = kMemLocationTypeDevice, .id = device_ordinal},
    };
    std::size_t granularity = 0;
    check_cuda(
        "cuMemGetAllocationGranularity",
        impl_->cu_mem_get_allocation_granularity(&granularity, &prop, kMemAllocGranularityMinimum));
    if (granularity == 0 || granularity > std::numeric_limits<CudaDevicePtr>::max()) {
      throw std::runtime_error("CUDA VMM allocation granularity is invalid");
    }
    const auto granularity_ptr = static_cast<CudaDevicePtr>(granularity);
    const auto last = ptr + static_cast<CudaDevicePtr>(accessible_bytes - 1U);
    auto region = ptr - ptr % granularity_ptr;
    const auto last_region = last - last % granularity_ptr;
    const CudaMemLocation location = {.type = kMemLocationTypeDevice, .id = device_ordinal};
    try {
      for (;;) {
        CudaMemAccessFlags access_flags = 0;
        check_cuda_pointer("cuMemGetAccess",
                           impl_->cu_mem_get_access(&access_flags, &location, region));
        validate_access_flags(access_flags, required_access);

        if (vmm_identity_complete) {
          unsigned long long block_id = 0;
          std::size_t mapping_size = 0;
          CudaDevicePtr mapping_base = 0;
          if (impl_->cu_pointer_get_attribute(&block_id, kPointerAttributeMemoryBlockId, region) ==
                  kCudaSuccess &&
              impl_->cu_pointer_get_attribute(&mapping_size, kPointerAttributeMappingSize,
                                              region) == kCudaSuccess &&
              impl_->cu_pointer_get_attribute(&mapping_base, kPointerAttributeMappingBaseAddress,
                                              region) == kCudaSuccess) {
            const auto region_remaining = std::numeric_limits<CudaDevicePtr>::max() - region;
            const auto region_last = region + std::min(granularity_ptr - 1U, region_remaining);
            const auto covered_first = std::max(ptr, region);
            const auto covered_last = std::min(last, region_last);
            if (mapping_size == 0 || covered_first < mapping_base || covered_last < mapping_base ||
                covered_first - mapping_base >= mapping_size ||
                covered_last - mapping_base >= mapping_size) {
              throw std::runtime_error("CUDA VMM mapping metadata does not cover the device span");
            }
            append_vmm_provenance_range(
                vmm_provenance_ranges,
                {.memory_block_id = block_id,
                 .mapping_base = mapping_base,
                 .mapping_size = mapping_size,
                 .first_offset = static_cast<std::size_t>(covered_first - mapping_base),
                 .last_offset = static_cast<std::size_t>(covered_last - mapping_base)});
          } else {
            vmm_provenance_ranges.clear();
            vmm_identity_complete = false;
          }
        }

        CUmemGenericAllocationHandle handle = 0;
        check_cuda_pointer(
            "cuMemRetainAllocationHandle",
            impl_->cu_mem_retain_allocation_handle(
                &handle, reinterpret_cast<void*>(static_cast<std::uintptr_t>(region))));
        if (std::find(allocation_handles.begin(), allocation_handles.end(), handle) ==
            allocation_handles.end()) {
          try {
            allocation_handles.push_back(handle);
          } catch (...) {
            (void)impl_->cu_mem_release(handle);
            throw;
          }
        } else {
          check_cuda("cuMemRelease", impl_->cu_mem_release(handle));
        }
        if (region == last_region) {
          break;
        }
        if (region > std::numeric_limits<CudaDevicePtr>::max() - granularity_ptr) {
          throw std::overflow_error("CUDA VMM validation region overflows the device pointer");
        }
        region += granularity_ptr;
      }
    } catch (...) {
      for (const auto handle : allocation_handles) {
        (void)impl_->cu_mem_release(handle);
      }
      throw;
    }
  }

  std::shared_ptr<CudaValidatedSpan::State> state;
  try {
    state = std::make_shared<CudaValidatedSpan::State>(
        impl_, ptr, accessible_bytes, reinterpret_cast<std::uintptr_t>(retained_context),
        device_ordinal, required_access, validated_range_base, validated_range_bytes,
        std::move(allocation_handles), std::move(vmm_provenance_ranges), pointer_context == nullptr,
        vmm_identity_complete);
  } catch (...) {
    for (const auto handle : allocation_handles) {
      (void)impl_->cu_mem_release(handle);
    }
    throw;
  }
  scope.restore();
  return CudaValidatedSpan(std::move(state));
}

void CudaBackend::validate_device_span(CudaDevicePtr ptr, std::size_t accessible_bytes,
                                       CudaSpanAccess required_access, int device_ordinal) const {
  (void)retain_device_span(ptr, accessible_bytes, required_access, device_ordinal);
}

CudaModule CudaBackend::load_module_from_ptx(std::string_view ptx, int device_ordinal) const {
  validate_ptx(ptx);
  PrimaryContextScope scope(*impl_, device_ordinal);
  const CUcontext context = scope.current();
  if (context == nullptr) {
    throw std::runtime_error("CUDA kernel load did not establish a current context");
  }
  std::string terminated_ptx(ptx);
  if (terminated_ptx.back() != '\0') {
    terminated_ptx.push_back('\0');
  }
  CUmodule module = nullptr;
  check_cuda("cuModuleLoadData", impl_->cu_module_load_data(&module, terminated_ptx.data()));
  std::shared_ptr<CudaModule::State> state;
  try {
    state = std::make_shared<CudaModule::State>(impl_, context, module);
  } catch (...) {
    CudaModule::State::unload(impl_, context, module);
    throw;
  }
  scope.restore();
  return CudaModule(std::move(state));
}

CudaKernel CudaBackend::load_kernel_from_ptx(std::string_view ptx, std::string_view function_name,
                                             int device_ordinal) const {
  validate_ptx(ptx);
  validate_no_nul(function_name, "kernel function name");
  return load_module_from_ptx(ptx, device_ordinal).load_kernel(function_name);
}

CudaExecutionStream CudaBackend::execution_stream(int device_ordinal) const {
  if (device_ordinal < 0 || device_ordinal >= device_count()) {
    throw std::invalid_argument("CUDA execution stream device ordinal is out of range");
  }
  if (impl_->cu_stream_create == nullptr || impl_->cu_stream_destroy == nullptr ||
      impl_->cu_stream_synchronize == nullptr || impl_->cu_event_create == nullptr ||
      impl_->cu_event_destroy == nullptr || impl_->cu_event_record == nullptr ||
      impl_->cu_event_synchronize == nullptr) {
    throw std::runtime_error("CUDA driver does not expose required stream/event APIs");
  }

  std::lock_guard<std::mutex> lock(impl_->execution_stream_mutex);
  if (const auto existing = impl_->execution_streams[device_ordinal].lock()) {
    return CudaExecutionStream(std::static_pointer_cast<CudaExecutionStream::State>(existing));
  }

  PrimaryContextScope scope(*impl_, device_ordinal);
  const CUcontext context = scope.current();
  if (context == nullptr) {
    throw std::runtime_error("CUDA execution stream did not establish a current context");
  }
  CUstream stream = nullptr;
  CUevent completion_event = nullptr;
  std::shared_ptr<CudaExecutionStream::State> state;
  check_cuda("cuStreamCreate", impl_->cu_stream_create(&stream, kStreamNonBlocking));
  try {
    check_cuda("cuEventCreate", impl_->cu_event_create(&completion_event, kEventDisableTiming));
    state = std::make_shared<CudaExecutionStream::State>(impl_, context, device_ordinal, stream,
                                                         completion_event);
    impl_->execution_streams[device_ordinal] = state;
    scope.restore();
    return CudaExecutionStream(std::move(state));
  } catch (...) {
    if (state != nullptr) {
      state.reset();
    } else {
      if (completion_event != nullptr) {
        (void)impl_->cu_event_destroy(completion_event);
      }
      (void)impl_->cu_stream_destroy(stream);
    }
    throw;
  }
}

void CudaBackend::synchronize() const {
  impl_->ensure_current_context_for_copy();
  check_cuda("cuCtxSynchronize", impl_->cu_ctx_synchronize());
  if (trace_sink_) {
    trace_sink_->context_synchronized();
  }
}

} // namespace reco::core
