#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifndef RECO_FAKE_CUDA_DRIVER_MARKER
#define RECO_FAKE_CUDA_DRIVER_MARKER 1
#endif

#if defined(_WIN32)
#define RECO_TEST_EXPORT __declspec(dllexport)
#else
#define RECO_TEST_EXPORT __attribute__((visibility("default")))
#endif

namespace {

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

thread_local void* current_context = nullptr;
thread_local void* fail_set_current_target = nullptr;
thread_local bool fail_set_current_once = false;
int primary_context_retain_count = 0;

constexpr std::uintptr_t kDefaultBase = 0x40000000;
constexpr std::uintptr_t kContextIndependentBase = 0x50000000;
constexpr std::uintptr_t kNoAccessBase = 0x51000000;
constexpr std::uintptr_t kReadOnlyBase = 0x52000000;
constexpr std::uintptr_t kYPlaneSize = 1280U * 720U;
constexpr std::uintptr_t kAllocationSize = 1280U * 1080U;
constexpr std::size_t kVmmGranularity = 0x10000U;
constexpr std::size_t kDefaultTotalMemoryBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kDefaultFreeMemoryBytes = 6ULL * 1024ULL * 1024ULL * 1024ULL;

std::size_t memory_bytes_from_environment(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return fallback;
  }
  return static_cast<std::size_t>(parsed);
}

std::uintptr_t allocation_base(std::uintptr_t pointer) {
  if (pointer >= kDefaultBase && pointer < kDefaultBase + kAllocationSize) {
    return kDefaultBase;
  }
  for (const auto base : {kContextIndependentBase, kNoAccessBase, kReadOnlyBase}) {
    if (pointer >= base && pointer < base + kAllocationSize) {
      return base;
    }
  }
  if (pointer < 100U) {
    return pointer;
  }
  if (pointer >= kYPlaneSize && pointer < kYPlaneSize + 100U) {
    return pointer - kYPlaneSize;
  }
  return pointer;
}

} // namespace

extern "C" RECO_TEST_EXPORT int recoFakeRuntimeMarker() { return RECO_FAKE_CUDA_DRIVER_MARKER; }

extern "C" RECO_TEST_EXPORT void recoFakeCudaFailNextSetCurrent(std::uintptr_t context) {
  fail_set_current_target = reinterpret_cast<void*>(context);
  fail_set_current_once = true;
}

extern "C" RECO_TEST_EXPORT std::uintptr_t recoFakeCudaCurrentContext() {
  return reinterpret_cast<std::uintptr_t>(current_context);
}

extern "C" int cuInit(unsigned int) { return 0; }

extern "C" int cuDeviceGetCount(int* count) {
  if (count == nullptr) {
    return 1;
  }
  *count = 1;
  return 0;
}

extern "C" int cuDeviceGet(int* device, int ordinal) {
  if (device == nullptr || ordinal != 0) {
    return 1;
  }
  *device = 0;
  return 0;
}

extern "C" int cuDeviceGetAttribute(int* value, int attribute, int device) {
  if (value == nullptr || device != 0) {
    return 1;
  }
  if (attribute == 75) {
    *value = 8;
    return 0;
  }
  if (attribute == 76) {
    *value = 6;
    return 0;
  }
  if (attribute == 18) {
    const char* integrated = std::getenv("RECO_FAKE_CUDA_INTEGRATED");
    *value = integrated != nullptr && std::strcmp(integrated, "1") == 0 ? 1 : 0;
    return 0;
  }
  return 1;
}

extern "C" int cuDeviceGetName(char* name, int length, int device) {
  if (name == nullptr || length < 5 || device != 0) {
    return 1;
  }
  std::memcpy(name, "fake", 5);
  return 0;
}

extern "C" int cuDeviceGetUuid(void* uuid, int device) {
  if (uuid == nullptr || device != 0) {
    return 1;
  }
  std::memset(uuid, 0x42, 16);
  return 0;
}

extern "C" int cuDevicePrimaryCtxRetain(void** context, int device) {
  if (context == nullptr || device != 0) {
    return 1;
  }
  ++primary_context_retain_count;
  *context = reinterpret_cast<void*>(0xC0DA);
  return 0;
}

extern "C" int cuDevicePrimaryCtxRelease(int device) {
  if (device != 0 || primary_context_retain_count <= 0) {
    return 1;
  }
  --primary_context_retain_count;
  return 0;
}

extern "C" int cuDevicePrimaryCtxRelease_v2(int device) {
  return cuDevicePrimaryCtxRelease(device);
}

extern "C" int cuCtxGetCurrent(void** context) {
  if (context == nullptr) {
    return 1;
  }
  *context = current_context;
  return 0;
}

extern "C" int cuCtxSetCurrent(void* context) {
  if (fail_set_current_once && context == fail_set_current_target) {
    fail_set_current_once = false;
    return 901;
  }
  current_context = context;
  return 0;
}

extern "C" int cuCtxGetDevice(int* device) {
  if (device == nullptr || current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  *device = 0;
  return 0;
}

extern "C" int cuCtxSynchronize() {
  return current_context == reinterpret_cast<void*>(0xC0DA) ? 0 : 1;
}

extern "C" int cuPointerGetAttribute(void* data, int attribute, unsigned long long pointer) {
  if (data == nullptr || current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  const auto base = allocation_base(static_cast<std::uintptr_t>(pointer));
  switch (attribute) {
  case 1:
    *static_cast<void**>(data) =
        base == 11 ? reinterpret_cast<void*>(0xBAD)
        : base == kContextIndependentBase || base == kNoAccessBase || base == kReadOnlyBase
            ? nullptr
            : reinterpret_cast<void*>(0xC0DA);
    return 0;
  case 2:
    *static_cast<int*>(data) = base == 9 ? 1 : 2;
    return 0;
  case 9:
    *static_cast<int*>(data) = base == 12 ? 1 : 0;
    return 0;
  case 11:
    *static_cast<std::uint64_t*>(data) = base;
    return 0;
  case 12:
    *static_cast<std::size_t*>(data) = base == 10 || base == 13 ? 1U : kAllocationSize;
    return 0;
  case 13:
    *static_cast<int*>(data) = 1;
    return 0;
  case 16:
    *static_cast<unsigned int*>(data) =
        base == kNoAccessBase ? 0U : (base == kReadOnlyBase ? 1U : 3U);
    return 0;
  case 18:
    *static_cast<std::size_t*>(data) = kAllocationSize;
    return 0;
  case 19:
    *static_cast<std::uint64_t*>(data) = base;
    return 0;
  case 20:
    *static_cast<unsigned long long*>(data) = base;
    return 0;
  default:
    return 1;
  }
}

extern "C" int cuMemGetAllocationGranularity(std::size_t* granularity, const void*,
                                             unsigned int option) {
  if (granularity == nullptr || option != 0U) {
    return 1;
  }
  *granularity = kVmmGranularity;
  return 0;
}

extern "C" int cuMemGetAccess(unsigned long long* flags, const void*, unsigned long long pointer) {
  if (flags == nullptr || current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  const auto base = allocation_base(static_cast<std::uintptr_t>(pointer));
  *flags = base == kNoAccessBase ? 0U : (base == kReadOnlyBase ? 1U : 3U);
  return base == kContextIndependentBase || base == kNoAccessBase || base == kReadOnlyBase ? 0 : 1;
}

extern "C" int cuMemRetainAllocationHandle(unsigned long long* handle, void* address) {
  if (handle == nullptr || current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  const auto base = allocation_base(reinterpret_cast<std::uintptr_t>(address));
  if (base != kContextIndependentBase && base != kNoAccessBase && base != kReadOnlyBase) {
    return 1;
  }
  *handle = static_cast<unsigned long long>(base);
  return 0;
}

extern "C" int cuMemRelease(unsigned long long handle) {
  return handle == kContextIndependentBase || handle == kNoAccessBase || handle == kReadOnlyBase
             ? 0
             : 1;
}

extern "C" int cuMemAlloc_v2(std::uint64_t*, std::size_t) { return 1; }
extern "C" int cuMemAllocPitch_v2(std::uint64_t*, std::size_t*, std::size_t, std::size_t,
                                  unsigned int) {
  return 1;
}
extern "C" int cuMemFree_v2(std::uint64_t) { return 1; }
extern "C" int cuMemsetD8_v2(std::uint64_t, unsigned char, std::size_t) { return 1; }
extern "C" int cuMemcpy2D_v2(const void*) { return 1; }
extern "C" int cuMemcpyDtoH_v2(void*, std::uint64_t, std::size_t) { return 1; }
extern "C" int cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
  if (free_bytes == nullptr || total_bytes == nullptr ||
      current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  *free_bytes = memory_bytes_from_environment("RECO_FAKE_CUDA_FREE_BYTES", kDefaultFreeMemoryBytes);
  *total_bytes =
      memory_bytes_from_environment("RECO_FAKE_CUDA_TOTAL_BYTES", kDefaultTotalMemoryBytes);
  return 0;
}
extern "C" int cuMemAddressReserve(unsigned long long*, std::size_t, std::size_t,
                                   unsigned long long, unsigned long long) {
  return 1;
}
extern "C" int cuMemCreate(unsigned long long*, std::size_t, const void*, unsigned long long) {
  return 1;
}
extern "C" int cuMemExportToShareableHandle(void*, unsigned long long, unsigned int,
                                            unsigned long long) {
  return 1;
}
extern "C" int cuMemMap(unsigned long long, std::size_t, std::size_t, unsigned long long,
                        unsigned long long) {
  return 1;
}
extern "C" int cuMemSetAccess(unsigned long long, std::size_t, const void*, std::size_t) {
  return 1;
}
extern "C" int cuMemUnmap(unsigned long long, std::size_t) { return 1; }
extern "C" int cuMemAddressFree(unsigned long long, std::size_t) { return 1; }
extern "C" int cuModuleLoadData(void**, const void*) { return 1; }
extern "C" int cuModuleUnload(void*) { return 1; }
extern "C" int cuModuleGetFunction(void**, void*, const char*) { return 1; }
extern "C" int cuLaunchKernel(void*, unsigned int, unsigned int, unsigned int, unsigned int,
                              unsigned int, unsigned int, unsigned int, void*, void**, void**) {
  return 1;
}

extern "C" int cuGraphicsEGLRegisterImage(void** resource, void* image, unsigned int) {
  if (resource == nullptr || image == nullptr ||
      current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  *resource = image;
  return 0;
}

extern "C" int cuGraphicsResourceGetMappedEglFrame(CudaEglFrame* frame, void* resource,
                                                   unsigned int, unsigned int) {
  if (frame == nullptr || resource == nullptr ||
      current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  frame->frame.pitches[0] = resource;
  frame->frame.pitches[1] =
      reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(resource) + 1280U * 720U);
  frame->width = 1280;
  frame->height = 720;
  frame->depth = 1;
  frame->pitch = 1280;
  frame->plane_count = 2;
  frame->channel_count = 1;
  frame->frame_type = 1;
  if (resource == reinterpret_cast<void*>(2) || resource == reinterpret_cast<void*>(7)) {
    frame->color_format = 0;
  } else if (resource == reinterpret_cast<void*>(8)) {
    frame->color_format = 0x01;
  } else {
    frame->color_format = 0x57;
  }
  frame->array_format = 1;
  return 0;
}

extern "C" int cuGraphicsUnregisterResource(void* resource) {
  return resource == nullptr || resource == reinterpret_cast<void*>(4) ||
                 current_context != reinterpret_cast<void*>(0xC0DA)
             ? 1
             : 0;
}
