#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#if defined(_WIN32)
#define RECO_FAKE_CUDA_EXPORT extern "C" __declspec(dllexport)
#else
#define RECO_FAKE_CUDA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr std::uintptr_t kContextIdentity = 0xCAFE0001U;
thread_local void* current_context = nullptr;
std::atomic<int> retain_count{0};
std::atomic<int> launch_count{0};
std::atomic<int> synchronize_count{0};
std::atomic<int> sequence{0};
std::atomic<int> last_launch_sequence{0};
std::atomic<int> last_synchronize_sequence{0};
std::array<std::uint64_t, 8> captured_u64{};
std::array<std::uint32_t, 12> captured_u32{};
std::array<float, 16> captured_float{};

struct PlanePrefix {
  std::uint64_t y_ptr;
  std::uint64_t uv_ptr;
  std::uint64_t y_pitch;
  std::uint64_t uv_pitch;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t flip_180;
  std::uint32_t reserved;
  float intrinsics[4];
  float distortion[4];
  float color[8];
};

struct ViewPrefix {
  float eye[4];
  float forward[4];
  float right[4];
  float up[4];
  float projection[4];
  float blend_clip[4];
};

} // namespace

RECO_FAKE_CUDA_EXPORT void recoFakeCudaStitchReset() {
  launch_count = 0;
  synchronize_count = 0;
  sequence = 0;
  last_launch_sequence = 0;
  last_synchronize_sequence = 0;
  captured_u64.fill(0);
  captured_u32.fill(0);
  captured_float.fill(0.0F);
}

RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchLaunchCount() { return launch_count.load(); }
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchSynchronizeCount() { return synchronize_count.load(); }
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchLaunchSequence() { return last_launch_sequence.load(); }
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchSynchronizeSequence() {
  return last_synchronize_sequence.load();
}
RECO_FAKE_CUDA_EXPORT std::uint64_t recoFakeCudaStitchCapturedU64(int index) {
  return index >= 0 && static_cast<std::size_t>(index) < captured_u64.size()
             ? captured_u64[static_cast<std::size_t>(index)]
             : 0;
}
RECO_FAKE_CUDA_EXPORT std::uint32_t recoFakeCudaStitchCapturedU32(int index) {
  return index >= 0 && static_cast<std::size_t>(index) < captured_u32.size()
             ? captured_u32[static_cast<std::size_t>(index)]
             : 0;
}
RECO_FAKE_CUDA_EXPORT float recoFakeCudaStitchCapturedFloat(int index) {
  return index >= 0 && static_cast<std::size_t>(index) < captured_float.size()
             ? captured_float[static_cast<std::size_t>(index)]
             : 0.0F;
}

RECO_FAKE_CUDA_EXPORT int cuInit(unsigned int) { return 0; }
RECO_FAKE_CUDA_EXPORT int cuDeviceGetCount(int* count) {
  if (count == nullptr) {
    return 1;
  }
  *count = 1;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuDeviceGet(int* device, int ordinal) {
  if (device == nullptr || ordinal != 0) {
    return 1;
  }
  *device = 0;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuDeviceGetAttribute(int* value, int attribute, int device) {
  if (value == nullptr || device != 0) {
    return 1;
  }
  if (attribute == 75) {
    *value = 8;
    return 0;
  }
  if (attribute == 76) {
    *value = 9;
    return 0;
  }
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuDeviceGetName(char* name, int length, int device) {
  if (name == nullptr || length <= 0 || device != 0) {
    return 1;
  }
  constexpr std::string_view kName = "fake stitch CUDA device";
  const auto count = std::min<std::size_t>(kName.size(), static_cast<std::size_t>(length - 1));
  std::memcpy(name, kName.data(), count);
  name[count] = '\0';
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuDeviceGetUuid(void* uuid, int device) {
  if (uuid == nullptr || device != 0) {
    return 1;
  }
  std::memset(uuid, 0x5A, 16);
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuDevicePrimaryCtxRetain(void** context, int device) {
  if (context == nullptr || device != 0) {
    return 1;
  }
  ++retain_count;
  *context = reinterpret_cast<void*>(kContextIdentity);
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuDevicePrimaryCtxRelease_v2(int device) {
  if (device != 0 || retain_count.load() <= 0) {
    return 1;
  }
  --retain_count;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuCtxGetCurrent(void** context) {
  if (context == nullptr) {
    return 1;
  }
  *context = current_context;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuCtxGetDevice(int* device) {
  if (device == nullptr || current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  *device = 0;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuCtxSetCurrent(void* context) {
  current_context = context;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuCtxSynchronize() {
  if (current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  ++synchronize_count;
  last_synchronize_sequence = ++sequence;
  return 0;
}

RECO_FAKE_CUDA_EXPORT int cuModuleLoadData(void** module, const void* image) {
  if (module == nullptr || image == nullptr ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  *module = reinterpret_cast<void*>(0x1234U);
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuModuleUnload(void* module) {
  return module == reinterpret_cast<void*>(0x1234U) ? 0 : 1;
}
RECO_FAKE_CUDA_EXPORT int cuModuleGetFunction(void** function, void* module, const char* name) {
  if (function == nullptr || module != reinterpret_cast<void*>(0x1234U) || name == nullptr ||
      std::string_view(name) != "reco_stitch_nv12_rgba") {
    return 1;
  }
  *function = reinterpret_cast<void*>(0x5678U);
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuLaunchKernel(void* function, unsigned int grid_x, unsigned int grid_y,
                                         unsigned int grid_z, unsigned int block_x,
                                         unsigned int block_y, unsigned int block_z,
                                         unsigned int shared_memory, void*, void** parameters,
                                         void**) {
  if (function != reinterpret_cast<void*>(0x5678U) || parameters == nullptr ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  const auto& left = *static_cast<const PlanePrefix*>(parameters[0]);
  const auto& right = *static_cast<const PlanePrefix*>(parameters[1]);
  const auto& view = *static_cast<const ViewPrefix*>(parameters[2]);
  captured_u64 = {left.y_ptr,
                  left.uv_ptr,
                  left.y_pitch,
                  left.uv_pitch,
                  right.y_ptr,
                  right.uv_ptr,
                  *static_cast<const std::uint64_t*>(parameters[3]),
                  *static_cast<const std::uint64_t*>(parameters[4])};
  captured_u32 = {left.width, left.height, left.flip_180, right.width, right.height, right.flip_180,
                  grid_x,     grid_y,      grid_z,        block_x,     block_y,      block_z};
  std::copy(std::begin(left.color), std::end(left.color), captured_float.begin());
  std::copy(std::begin(view.projection), std::end(view.projection), captured_float.begin() + 8);
  captured_float[12] = view.blend_clip[0];
  captured_float[13] = right.color[2];
  captured_float[14] = static_cast<float>(*static_cast<const std::uint32_t*>(parameters[5]));
  captured_float[15] = static_cast<float>(*static_cast<const std::uint32_t*>(parameters[6]));
  if (shared_memory != 0U) {
    return 1;
  }
  ++launch_count;
  last_launch_sequence = ++sequence;
  return 0;
}

RECO_FAKE_CUDA_EXPORT int cuMemAlloc_v2(std::uint64_t*, std::size_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemAllocPitch_v2(std::uint64_t*, std::size_t*, std::size_t, std::size_t,
                                             unsigned int) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemFree_v2(std::uint64_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemsetD8_v2(std::uint64_t, unsigned char, std::size_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemcpy2D_v2(const void*) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemcpyDtoH_v2(void*, std::uint64_t, std::size_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemGetInfo_v2(std::size_t*, std::size_t*) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemGetAllocationGranularity(std::size_t*, const void*, unsigned int) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemAddressReserve(std::uint64_t*, std::size_t, std::size_t,
                                              std::uint64_t, std::uint64_t) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemCreate(std::uint64_t*, std::size_t, const void*, std::uint64_t) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemExportToShareableHandle(void*, std::uint64_t, unsigned int,
                                                       std::uint64_t) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemMap(std::uint64_t, std::size_t, std::size_t, std::uint64_t,
                                   std::uint64_t) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemSetAccess(std::uint64_t, std::size_t, const void*, std::size_t) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemRelease(std::uint64_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemUnmap(std::uint64_t, std::size_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemAddressFree(std::uint64_t, std::size_t) { return 1; }
