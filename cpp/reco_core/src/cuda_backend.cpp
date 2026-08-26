#include "reco/core/cuda_backend.hpp"

#include <array>
#include <algorithm>
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
using CUdeviceptr = std::uint64_t;
using CUmemGenericAllocationHandle = std::uint64_t;

constexpr CUresult kCudaSuccess = 0;
constexpr unsigned int kMemoryTypeHost = 1;
constexpr unsigned int kMemoryTypeDevice = 2;
constexpr unsigned int kMemAllocationTypePinned = 1;
constexpr unsigned int kMemLocationTypeDevice = 1;
#if defined(_WIN32)
constexpr unsigned int kMemHandleType = 2;
#else
constexpr unsigned int kMemHandleType = 1;
#endif
constexpr unsigned int kMemAccessFlagsProtReadWrite = 3;
constexpr unsigned int kMemAllocGranularityMinimum = 0;

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
  explicit DynamicLibrary(const char* name) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(name);
#else
    handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw std::runtime_error(std::string("failed to load ") + name);
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

private:
  void* handle_ = nullptr;
};

[[noreturn]] void throw_cuda(const char* function, CUresult result) {
  throw std::runtime_error(std::string(function) + " returned CUDA error " +
                           std::to_string(result));
}

void check_cuda(const char* function, CUresult result) {
  if (result != kCudaSuccess) {
    throw_cuda(function, result);
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
    throw std::invalid_argument(std::string("CUDA kernel ") + name + " dimensions must be non-zero");
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
  explicit Impl()
#if defined(_WIN32)
      : driver("nvcuda.dll") {
#else
      : driver("libcuda.so.1") {
#endif
    cu_init = driver.symbol<decltype(cu_init)>("cuInit");
    cu_device_get_count = driver.symbol<decltype(cu_device_get_count)>("cuDeviceGetCount");
    cu_device_get = driver.symbol<decltype(cu_device_get)>("cuDeviceGet");
    cu_device_get_name = driver.symbol<decltype(cu_device_get_name)>("cuDeviceGetName");
    cu_device_get_uuid = driver.symbol<decltype(cu_device_get_uuid)>("cuDeviceGetUuid");
    cu_device_primary_ctx_retain =
        driver.symbol<decltype(cu_device_primary_ctx_retain)>("cuDevicePrimaryCtxRetain");
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
    cu_mem_get_allocation_granularity =
        driver.symbol<decltype(cu_mem_get_allocation_granularity)>(
            "cuMemGetAllocationGranularity");
    cu_mem_address_reserve = driver.symbol<decltype(cu_mem_address_reserve)>("cuMemAddressReserve");
    cu_mem_create = driver.symbol<decltype(cu_mem_create)>("cuMemCreate");
    cu_mem_export_to_shareable_handle =
        driver.symbol<decltype(cu_mem_export_to_shareable_handle)>(
            "cuMemExportToShareableHandle");
    cu_mem_map = driver.symbol<decltype(cu_mem_map)>("cuMemMap");
    cu_mem_set_access = driver.symbol<decltype(cu_mem_set_access)>("cuMemSetAccess");
    cu_mem_release = driver.symbol<decltype(cu_mem_release)>("cuMemRelease");
    cu_mem_unmap = driver.symbol<decltype(cu_mem_unmap)>("cuMemUnmap");
    cu_mem_address_free = driver.symbol<decltype(cu_mem_address_free)>("cuMemAddressFree");
    cu_module_load_data = driver.symbol<decltype(cu_module_load_data)>("cuModuleLoadData");
    cu_module_unload = driver.symbol<decltype(cu_module_unload)>("cuModuleUnload");
    cu_module_get_function = driver.symbol<decltype(cu_module_get_function)>("cuModuleGetFunction");
    cu_launch_kernel = driver.symbol<decltype(cu_launch_kernel)>("cuLaunchKernel");
    check_cuda("cuInit", cu_init(0));
  }

  CUdevice device(int ordinal) const {
    CUdevice device = 0;
    check_cuda("cuDeviceGet", cu_device_get(&device, ordinal));
    return device;
  }

  void ensure_primary_context(int ordinal) {
    const CUdevice requested_device = device(ordinal);
    CUcontext current = nullptr;
    check_cuda("cuCtxGetCurrent", cu_ctx_get_current(&current));
    if (current != nullptr) {
      CUdevice current_device = 0;
      check_cuda("cuCtxGetDevice", cu_ctx_get_device(&current_device));
      if (current_device == requested_device) {
        return;
      }
    }

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
    check_cuda("cuCtxSetCurrent", cu_ctx_set_current(context));
  }

  void ensure_current_context_for_copy() {
    CUcontext current = nullptr;
    check_cuda("cuCtxGetCurrent", cu_ctx_get_current(&current));
    if (current != nullptr) {
      return;
    }
    ensure_primary_context(0);
  }

  CUcontext current_context() {
    CUcontext current = nullptr;
    check_cuda("cuCtxGetCurrent", cu_ctx_get_current(&current));
    return current;
  }

  void set_current_context(CUcontext context) {
    check_cuda("cuCtxSetCurrent", cu_ctx_set_current(context));
  }

  DynamicLibrary driver;
  std::mutex context_mutex;
  std::unordered_map<int, CUcontext> retained_contexts;
  CUresult (*cu_init)(unsigned int) = nullptr;
  CUresult (*cu_device_get_count)(int*) = nullptr;
  CUresult (*cu_device_get)(CUdevice*, int) = nullptr;
  CUresult (*cu_device_get_name)(char*, int, CUdevice) = nullptr;
  CUresult (*cu_device_get_uuid)(CUuuid*, CUdevice) = nullptr;
  CUresult (*cu_device_primary_ctx_retain)(CUcontext*, CUdevice) = nullptr;
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
  CUresult (*cu_mem_get_allocation_granularity)(std::size_t*, const CudaMemAllocationProp*,
                                                unsigned int) = nullptr;
  CUresult (*cu_mem_address_reserve)(CUdeviceptr*, std::size_t, std::size_t, CUdeviceptr,
                                     std::uint64_t) = nullptr;
  CUresult (*cu_mem_create)(CUmemGenericAllocationHandle*, std::size_t,
                            const CudaMemAllocationProp*, std::uint64_t) = nullptr;
  CUresult (*cu_mem_export_to_shareable_handle)(void*, CUmemGenericAllocationHandle, unsigned int,
                                                std::uint64_t) = nullptr;
  CUresult (*cu_mem_map)(CUdeviceptr, std::size_t, std::size_t, CUmemGenericAllocationHandle,
                         std::uint64_t) = nullptr;
  CUresult (*cu_mem_set_access)(CUdeviceptr, std::size_t, const CudaMemAccessDesc*,
                                std::size_t) = nullptr;
  CUresult (*cu_mem_release)(CUmemGenericAllocationHandle) = nullptr;
  CUresult (*cu_mem_unmap)(CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cu_mem_address_free)(CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cu_module_load_data)(CUmodule*, const void*) = nullptr;
  CUresult (*cu_module_unload)(CUmodule) = nullptr;
  CUresult (*cu_module_get_function)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*cu_launch_kernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int, CUstream, void**, void**) =
      nullptr;
};

CudaDeviceBuffer::CudaDeviceBuffer(CudaDevicePtr ptr, std::size_t size,
                                   std::function<void(CudaDevicePtr)> free)
    : ptr_(ptr), size_(size), free_(std::move(free)) {}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, 0)),
      size_(std::exchange(other.size_, 0)),
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
    : backend_(std::move(backend)),
      context_(context),
      ptr_(ptr),
      size_(size),
      shareable_handle_(shareable_handle),
      owns_shareable_handle_(valid_shareable_handle(shareable_handle)) {}

CudaSharedMemory::CudaSharedMemory(CudaSharedMemory&& other) noexcept
    : backend_(std::move(other.backend_)),
      context_(std::exchange(other.context_, nullptr)),
      ptr_(std::exchange(other.ptr_, 0)),
      size_(std::exchange(other.size_, 0)),
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
    backend_->set_current_context(previous_context);
  } catch (...) {
  }
  ptr_ = 0;
  size_ = 0;
  context_ = nullptr;
  backend_.reset();
}

CudaKernel::CudaKernel(std::shared_ptr<CudaBackend::Impl> backend, void* context, void* module,
                       void* function)
    : backend_(std::move(backend)), context_(context), module_(module), function_(function) {}

CudaKernel::CudaKernel(CudaKernel&& other) noexcept
    : backend_(std::move(other.backend_)),
      context_(std::exchange(other.context_, nullptr)),
      module_(std::exchange(other.module_, nullptr)),
      function_(std::exchange(other.function_, nullptr)) {}

CudaKernel& CudaKernel::operator=(CudaKernel&& other) noexcept {
  if (this != &other) {
    reset();
    backend_ = std::move(other.backend_);
    context_ = std::exchange(other.context_, nullptr);
    module_ = std::exchange(other.module_, nullptr);
    function_ = std::exchange(other.function_, nullptr);
  }
  return *this;
}

CudaKernel::~CudaKernel() { reset(); }

void CudaKernel::launch(const CudaLaunchConfig& config, std::span<void*> args) const {
  if (!*this) {
    throw std::invalid_argument("CUDA kernel launch requires a live kernel");
  }
  validate_dim3(config.grid, "grid");
  validate_dim3(config.block, "block");
  const CUcontext previous_context = backend_->current_context();
  backend_->set_current_context(static_cast<CUcontext>(context_));
  void** kernel_args = args.empty() ? nullptr : args.data();
  try {
    check_cuda("cuLaunchKernel",
               backend_->cu_launch_kernel(static_cast<CUfunction>(function_), config.grid.x,
                                          config.grid.y, config.grid.z, config.block.x,
                                          config.block.y, config.block.z, config.shared_memory_bytes,
                                          nullptr, kernel_args, nullptr));
  } catch (...) {
    backend_->set_current_context(previous_context);
    throw;
  }
  backend_->set_current_context(previous_context);
}

void CudaKernel::synchronize() const {
  if (!*this) {
    throw std::invalid_argument("CUDA kernel synchronize requires a live kernel");
  }
  const CUcontext previous_context = backend_->current_context();
  backend_->set_current_context(static_cast<CUcontext>(context_));
  try {
    check_cuda("cuCtxSynchronize", backend_->cu_ctx_synchronize());
  } catch (...) {
    backend_->set_current_context(previous_context);
    throw;
  }
  backend_->set_current_context(previous_context);
}

void CudaKernel::reset() {
  if (module_ == nullptr) {
    return;
  }
  try {
    const CUcontext previous_context = backend_->current_context();
    backend_->set_current_context(static_cast<CUcontext>(context_));
    try {
      check_cuda("cuModuleUnload", backend_->cu_module_unload(static_cast<CUmodule>(module_)));
    } catch (...) {
      backend_->set_current_context(previous_context);
      throw;
    }
    backend_->set_current_context(previous_context);
  } catch (...) {
  }
  context_ = nullptr;
  module_ = nullptr;
  function_ = nullptr;
  backend_.reset();
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

CudaBackend CudaBackend::create() {
  static const std::shared_ptr<Impl> impl = std::make_shared<Impl>();
  return CudaBackend(impl);
}

CudaBackend::CudaBackend(std::shared_ptr<Impl> impl, std::function<void()> synchronization_observer)
    : impl_(std::move(impl)), synchronization_observer_(std::move(synchronization_observer)) {}

CudaBackend CudaBackend::with_synchronization_observer(std::function<void()> observer) const {
  if (!observer) {
    throw std::invalid_argument("CUDA synchronization observer must be callable");
  }
  return CudaBackend(impl_, std::move(observer));
}

int CudaBackend::device_count() const {
  int count = 0;
  check_cuda("cuDeviceGetCount", impl_->cu_device_get_count(&count));
  return count;
}

CudaDeviceInfo CudaBackend::device_info(int ordinal) const {
  const CUdevice device = impl_->device(ordinal);

  std::array<char, 256> name{};
  check_cuda("cuDeviceGetName", impl_->cu_device_get_name(name.data(), static_cast<int>(name.size()),
                                                          device));
  CUuuid uuid{};
  check_cuda("cuDeviceGetUuid", impl_->cu_device_get_uuid(&uuid, device));

  CudaDeviceInfo info;
  info.ordinal = ordinal;
  info.name = name.data();
  std::memcpy(info.uuid.data(), uuid.bytes, info.uuid.size());
  return info;
}

void CudaBackend::ensure_primary_context(int ordinal) const {
  impl_->ensure_primary_context(ordinal);
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
  impl_->ensure_primary_context(0);
  CUdeviceptr ptr = 0;
  check_cuda("cuMemAlloc_v2", impl_->cu_mem_alloc(&ptr, bytes));
  auto impl = impl_;
  return CudaDeviceBuffer(ptr, bytes, [impl](CudaDevicePtr ptr) {
    impl->ensure_primary_context(0);
    check_cuda("cuMemFree_v2", impl->cu_mem_free(ptr));
  });
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
  auto impl = impl_;
  return {.buffer = CudaDeviceBuffer(ptr, size,
                                     [impl](CudaDevicePtr allocation) {
                                       impl->ensure_primary_context(0);
                                       check_cuda("cuMemFree_v2", impl->cu_mem_free(allocation));
                                     }),
          .pitch = pitch};
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
  check_cuda("cuMemGetAllocationGranularity",
             impl_->cu_mem_get_allocation_granularity(&granularity, &prop,
                                                      kMemAllocGranularityMinimum));
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
}

CudaKernel CudaBackend::load_kernel_from_ptx(std::string_view ptx,
                                             std::string_view function_name) const {
  validate_ptx(ptx);
  validate_no_nul(function_name, "kernel function name");
  impl_->ensure_current_context_for_copy();
  const CUcontext context = impl_->current_context();
  if (context == nullptr) {
    throw std::runtime_error("CUDA kernel load did not establish a current context");
  }
  std::string terminated_ptx(ptx);
  if (terminated_ptx.back() != '\0') {
    terminated_ptx.push_back('\0');
  }
  std::string terminated_name(function_name);
  terminated_name.push_back('\0');

  CUmodule module = nullptr;
  check_cuda("cuModuleLoadData", impl_->cu_module_load_data(&module, terminated_ptx.data()));
  CUfunction function = nullptr;
  try {
    check_cuda("cuModuleGetFunction",
               impl_->cu_module_get_function(&function, module, terminated_name.data()));
  } catch (...) {
    const auto unload_result = impl_->cu_module_unload(module);
    (void)unload_result;
    throw;
  }
  return CudaKernel(impl_, context, module, function);
}

void CudaBackend::synchronize() const {
  impl_->ensure_current_context_for_copy();
  check_cuda("cuCtxSynchronize", impl_->cu_ctx_synchronize());
  if (synchronization_observer_) {
    synchronization_observer_();
  }
}

} // namespace reco::core
