#include "reco/core/cuda_backend.hpp"

#include <array>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
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

constexpr CUresult kCudaSuccess = 0;
constexpr unsigned int kMemoryTypeHost = 1;
constexpr unsigned int kMemoryTypeDevice = 2;

struct CUuuid {
  std::uint8_t bytes[16];
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
    cu_mem_free = driver.symbol<decltype(cu_mem_free)>("cuMemFree_v2");
    cu_memset_d8 = driver.symbol<decltype(cu_memset_d8)>("cuMemsetD8_v2");
    cu_memcpy_2d = driver.symbol<decltype(cu_memcpy_2d)>("cuMemcpy2D_v2");
    cu_memcpy_dtoh = driver.symbol<decltype(cu_memcpy_dtoh)>("cuMemcpyDtoH_v2");
    cu_mem_get_info = driver.symbol<decltype(cu_mem_get_info)>("cuMemGetInfo_v2");
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
  CUresult (*cu_mem_free)(CUdeviceptr) = nullptr;
  CUresult (*cu_memset_d8)(CUdeviceptr, unsigned char, std::size_t) = nullptr;
  CUresult (*cu_memcpy_2d)(const CudaMemcpy2D*) = nullptr;
  CUresult (*cu_memcpy_dtoh)(void*, CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cu_mem_get_info)(std::size_t*, std::size_t*) = nullptr;
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

CudaBackend::CudaBackend(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

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
}

} // namespace reco::core
