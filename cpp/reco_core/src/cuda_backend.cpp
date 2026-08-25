#include "reco/core/cuda_backend.hpp"

#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>
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
using CUresult = int;
using CUdeviceptr = std::uint64_t;

constexpr CUresult kCudaSuccess = 0;

struct CUuuid {
  std::uint8_t bytes[16];
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
    cu_memcpy_dtoh = driver.symbol<decltype(cu_memcpy_dtoh)>("cuMemcpyDtoH_v2");
    cu_mem_get_info = driver.symbol<decltype(cu_mem_get_info)>("cuMemGetInfo_v2");
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
  CUresult (*cu_memcpy_dtoh)(void*, CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cu_mem_get_info)(std::size_t*, std::size_t*) = nullptr;
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

void CudaBackend::synchronize() const {
  impl_->ensure_primary_context(0);
  check_cuda("cuCtxSynchronize", impl_->cu_ctx_synchronize());
}

} // namespace reco::core
