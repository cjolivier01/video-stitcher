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
constexpr std::uintptr_t kForeignContextIdentity = 0xCAFE0002U;
constexpr std::uint64_t kAllocationAlignment = 0x1000U;
constexpr std::size_t kAllocationSize = 0x1000U;
constexpr std::uint64_t kHostAllocation = 0x80000U;
constexpr std::uint64_t kFreedAllocation = 0x90000U;
constexpr std::uint64_t kUndersizedAllocation = 0xA0000U;
constexpr std::uint64_t kForeignContextAllocation = 0xB0000U;
constexpr std::uint64_t kForeignDeviceAllocation = 0xC0000U;
constexpr std::uint64_t kUnmappedAllocation = 0xD0000U;
constexpr std::uint64_t kNoAccessAllocation = 0xE0000U;
constexpr std::uint64_t kReadOnlyAllocation = 0xF0000U;
constexpr std::uint64_t kContextIndependentMapping = 0x50000U;
constexpr std::uint64_t kPhysicalAliasMapping = 0x60000U;
constexpr std::uint64_t kSplitAccessMapping = 0x100000U;
constexpr std::uint64_t kUnknownBlockIdMapping = 0x110000U;
thread_local void* current_context = nullptr;
std::atomic<int> retain_count{0};
std::atomic<int> release_count{0};
std::atomic<int> launch_count{0};
std::atomic<int> synchronize_count{0};
std::atomic<int> pointer_attribute_count{0};
std::atomic<int> memory_access_count{0};
std::atomic<bool> fail_module_load{false};
std::atomic<int> sequence{0};
std::atomic<int> last_launch_sequence{0};
std::atomic<int> last_synchronize_sequence{0};
std::atomic<int> last_restore_sequence{0};
thread_local void* fail_set_current_target = nullptr;
thread_local bool fail_set_current_once = false;
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

std::uint64_t allocation_base(std::uint64_t pointer) {
  return pointer - pointer % kAllocationAlignment;
}

std::size_t allocation_size(std::uint64_t base) {
  return base == kUndersizedAllocation ? 8U : kAllocationSize;
}

} // namespace

RECO_FAKE_CUDA_EXPORT void recoFakeCudaStitchReset() {
  launch_count = 0;
  synchronize_count = 0;
  pointer_attribute_count = 0;
  memory_access_count = 0;
  fail_module_load = false;
  sequence = 0;
  last_launch_sequence = 0;
  last_synchronize_sequence = 0;
  last_restore_sequence = 0;
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
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchRestoreSequence() {
  return last_restore_sequence.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchPointerAttributeCount() {
  return pointer_attribute_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchMemoryAccessCount() {
  return memory_access_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchRetainCount() { return retain_count.load(); }
RECO_FAKE_CUDA_EXPORT int recoFakeCudaStitchReleaseCount() { return release_count.load(); }
RECO_FAKE_CUDA_EXPORT void recoFakeCudaStitchSetCurrentContext(std::uintptr_t context) {
  current_context = reinterpret_cast<void*>(context);
}
RECO_FAKE_CUDA_EXPORT void recoFakeCudaStitchFailNextSetCurrent(std::uintptr_t context) {
  fail_set_current_target = reinterpret_cast<void*>(context);
  fail_set_current_once = true;
}
RECO_FAKE_CUDA_EXPORT std::uintptr_t recoFakeCudaStitchCurrentContext() {
  return reinterpret_cast<std::uintptr_t>(current_context);
}
RECO_FAKE_CUDA_EXPORT void recoFakeCudaStitchFailModuleLoad(int fail) {
  fail_module_load = fail != 0;
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
  ++release_count;
  if (retain_count.load() == 0 && current_context == reinterpret_cast<void*>(kContextIdentity)) {
    current_context = nullptr;
  }
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
  if (device == nullptr || retain_count.load() <= 0 ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  *device = 0;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuCtxSetCurrent(void* context) {
  if (fail_set_current_once && context == fail_set_current_target) {
    fail_set_current_once = false;
    return 901;
  }
  if (context == reinterpret_cast<void*>(kContextIdentity) && retain_count.load() <= 0) {
    return 1;
  }
  current_context = context;
  if (context == reinterpret_cast<void*>(kForeignContextIdentity)) {
    last_restore_sequence = ++sequence;
  }
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuCtxSynchronize() {
  if (retain_count.load() <= 0 || current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  ++synchronize_count;
  last_synchronize_sequence = ++sequence;
  return 0;
}

RECO_FAKE_CUDA_EXPORT int cuPointerGetAttribute(void* data, int attribute,
                                                unsigned long long pointer) {
  ++pointer_attribute_count;
  const auto base = allocation_base(pointer);
  if (data == nullptr || retain_count.load() <= 0 ||
      current_context != reinterpret_cast<void*>(kContextIdentity) || pointer == 0 ||
      base == kFreedAllocation) {
    return 1;
  }
  switch (attribute) {
  case 1:
    *static_cast<void**>(data) =
        base == kForeignContextAllocation
            ? reinterpret_cast<void*>(kForeignContextIdentity)
            : (base == kContextIndependentMapping || base == kPhysicalAliasMapping ||
                       base == kSplitAccessMapping || base == kSplitAccessMapping + 0x1000U ||
                       base == kUnknownBlockIdMapping
                   ? nullptr
                   : reinterpret_cast<void*>(kContextIdentity));
    return 0;
  case 2:
    *static_cast<unsigned int*>(data) = base == kHostAllocation ? 1U : 2U;
    return 0;
  case 9:
    *static_cast<int*>(data) = base == kForeignDeviceAllocation ? 1 : 0;
    return 0;
  case 11:
    *static_cast<std::uint64_t*>(data) = base;
    return 0;
  case 12:
    *static_cast<std::size_t*>(data) = allocation_size(base);
    return 0;
  case 13:
    *static_cast<unsigned int*>(data) = base == kUnmappedAllocation ? 0U : 1U;
    return 0;
  case 16:
    *static_cast<unsigned int*>(data) =
        base == kNoAccessAllocation ? 0U : (base == kReadOnlyAllocation ? 1U : 3U);
    return 0;
  case 18:
    *static_cast<std::size_t*>(data) = kAllocationSize;
    return 0;
  case 19:
    *static_cast<std::uint64_t*>(data) = base;
    return 0;
  case 20:
    if (base == kUnknownBlockIdMapping) {
      return 1;
    }
    *static_cast<unsigned long long*>(data) =
        base == kContextIndependentMapping || base == kPhysicalAliasMapping ? 0xB10CU : base;
    return 0;
  default:
    return 1;
  }
}

RECO_FAKE_CUDA_EXPORT int cuModuleLoadData(void** module, const void* image) {
  if (module == nullptr || image == nullptr || retain_count.load() <= 0 ||
      current_context != reinterpret_cast<void*>(kContextIdentity) || fail_module_load.load()) {
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
      retain_count.load() <= 0 || current_context != reinterpret_cast<void*>(kContextIdentity)) {
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
RECO_FAKE_CUDA_EXPORT int cuMemGetAllocationGranularity(std::size_t* granularity, const void*,
                                                        unsigned int option) {
  if (granularity == nullptr || option != 0U) {
    return 1;
  }
  *granularity = kAllocationAlignment;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuMemGetAccess(unsigned long long* flags, const void*,
                                         unsigned long long pointer) {
  if (flags == nullptr || current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  ++memory_access_count;
  *flags = pointer == kSplitAccessMapping + kAllocationAlignment ? 0U : 3U;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuMemRetainAllocationHandle(unsigned long long* handle, void* address) {
  if (handle == nullptr || current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  const auto pointer = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(address));
  const auto base = allocation_base(pointer);
  if (base == kContextIndependentMapping || base == kPhysicalAliasMapping ||
      base == kUnknownBlockIdMapping) {
    *handle = base;
    return 0;
  }
  if (base == kSplitAccessMapping || base == kSplitAccessMapping + kAllocationAlignment) {
    *handle = 0x1000U;
    return 0;
  }
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemAddressReserve(unsigned long long*, std::size_t, std::size_t,
                                              unsigned long long, unsigned long long) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemCreate(unsigned long long*, std::size_t, const void*,
                                      unsigned long long) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemExportToShareableHandle(void*, unsigned long long, unsigned int,
                                                       unsigned long long) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemMap(unsigned long long, std::size_t, std::size_t, unsigned long long,
                                   unsigned long long) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemSetAccess(unsigned long long, std::size_t, const void*,
                                         std::size_t) {
  return 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemRelease(unsigned long long handle) {
  return handle == kContextIndependentMapping || handle == kPhysicalAliasMapping ||
                 handle == kUnknownBlockIdMapping || handle == 0x1000U
             ? 0
             : 1;
}
RECO_FAKE_CUDA_EXPORT int cuMemUnmap(unsigned long long, std::size_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemAddressFree(unsigned long long, std::size_t) { return 1; }
