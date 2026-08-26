#include <array>
#include <cstddef>
#include <cstdint>

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
int primary_context_retain_count = 0;

constexpr std::uintptr_t kDefaultBase = 0x40000000;
constexpr std::uintptr_t kYPlaneSize = 1280U * 720U;
constexpr std::uintptr_t kAllocationSize = 1280U * 1080U;

std::uintptr_t allocation_base(std::uintptr_t pointer) {
  if (pointer >= kDefaultBase && pointer < kDefaultBase + kAllocationSize) {
    return kDefaultBase;
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

extern "C" int cuInit(unsigned int) { return 0; }

extern "C" int cuDeviceGet(int* device, int ordinal) {
  if (device == nullptr || ordinal != 0) {
    return 1;
  }
  *device = 0;
  return 0;
}

extern "C" int cuDevicePrimaryCtxRetain(void** context, int device) {
  if (context == nullptr || device != 0 || primary_context_retain_count != 0) {
    return 1;
  }
  ++primary_context_retain_count;
  *context = reinterpret_cast<void*>(0xC0DA);
  return 0;
}

extern "C" int cuDevicePrimaryCtxRelease(int device) {
  if (device != 0 || primary_context_retain_count != 1) {
    return 1;
  }
  --primary_context_retain_count;
  return 0;
}

extern "C" int cuCtxGetCurrent(void** context) {
  if (context == nullptr) {
    return 1;
  }
  *context = current_context;
  return 0;
}

extern "C" int cuCtxSetCurrent(void* context) {
  current_context = context;
  return 0;
}

extern "C" int cuPointerGetAttribute(void* data, int attribute, std::uint64_t pointer) {
  if (data == nullptr || current_context != reinterpret_cast<void*>(0xC0DA)) {
    return 1;
  }
  const auto base = allocation_base(static_cast<std::uintptr_t>(pointer));
  switch (attribute) {
  case 1:
    *static_cast<void**>(data) =
        base == 11 ? reinterpret_cast<void*>(0xBAD) : reinterpret_cast<void*>(0xC0DA);
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
    *static_cast<std::size_t*>(data) = kAllocationSize;
    return 0;
  case 13:
    *static_cast<int*>(data) = 1;
    return 0;
  case 18:
    *static_cast<std::size_t*>(data) = base == 10 || base == 13 ? 1U : kAllocationSize;
    return 0;
  case 19:
    *static_cast<std::uint64_t*>(data) = base;
    return 0;
  default:
    return 1;
  }
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
