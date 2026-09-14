#if defined(RECO_CUDA_RGBA_TO_NV12_FAKE_DRIVER)

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

constexpr std::uintptr_t kContextIdentity = 0xCAFE2901U;
constexpr std::uintptr_t kForeignContextIdentity = 0xCAFE2902U;
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
constexpr std::uint64_t kContextIndependentMapping = 0x100000U;
constexpr std::uint64_t kPhysicalAliasMapping = 0x110000U;
constexpr std::uint64_t kUnknownPhysicalIdentityMapping = 0x120000U;
constexpr std::uintptr_t kExecutionStream = 0xCAFE3901U;
constexpr std::uintptr_t kCompletionEvent = 0xCAFE3902U;
thread_local void* current_context = nullptr;
std::atomic<int> retain_count{0};
std::atomic<int> launch_count{0};
std::atomic<int> synchronize_count{0};
std::atomic<int> pointer_attribute_count{0};
std::atomic<int> sequence{0};
std::atomic<int> launch_sequence{0};
std::atomic<int> synchronize_sequence{0};
std::atomic<int> stream_create_count{0};
std::atomic<int> stream_destroy_count{0};
std::atomic<int> event_create_count{0};
std::atomic<int> event_destroy_count{0};
std::atomic<int> event_record_count{0};
std::atomic<int> event_synchronize_count{0};
std::atomic<int> surface_busy_count{0};
std::array<std::uint64_t, 6> captured_u64{};
std::array<std::uint32_t, 8> captured_u32{};
std::array<float, 8> captured_color{};

struct ColorParams {
  float values[8];
};

std::uint64_t allocation_base(std::uint64_t pointer) {
  return pointer - pointer % kAllocationAlignment;
}

std::size_t allocation_size(std::uint64_t base) {
  return base == kUndersizedAllocation ? 8U : kAllocationSize;
}

} // namespace

RECO_FAKE_CUDA_EXPORT void recoFakeCudaRgbaToNv12Reset() {
  launch_count = 0;
  synchronize_count = 0;
  pointer_attribute_count = 0;
  sequence = 0;
  launch_sequence = 0;
  synchronize_sequence = 0;
  stream_create_count = 0;
  stream_destroy_count = 0;
  event_create_count = 0;
  event_destroy_count = 0;
  event_record_count = 0;
  event_synchronize_count = 0;
  surface_busy_count = 0;
  captured_u64.fill(0);
  captured_u32.fill(0);
  captured_color.fill(0.0F);
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12LaunchCount() { return launch_count.load(); }
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12SynchronizeCount() {
  return synchronize_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12PointerAttributeCount() {
  return pointer_attribute_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12LaunchSequence() { return launch_sequence.load(); }
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12SynchronizeSequence() {
  return synchronize_sequence.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12StreamCreateCount() {
  return stream_create_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12StreamDestroyCount() {
  return stream_destroy_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12EventCreateCount() {
  return event_create_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12EventDestroyCount() {
  return event_destroy_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12EventRecordCount() {
  return event_record_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12EventSynchronizeCount() {
  return event_synchronize_count.load();
}
RECO_FAKE_CUDA_EXPORT int recoFakeCudaRgbaToNv12SurfaceBusyCount() {
  return surface_busy_count.load();
}
RECO_FAKE_CUDA_EXPORT std::uint64_t recoFakeCudaRgbaToNv12CapturedU64(int index) {
  return index >= 0 && static_cast<std::size_t>(index) < captured_u64.size()
             ? captured_u64[static_cast<std::size_t>(index)]
             : 0;
}
RECO_FAKE_CUDA_EXPORT std::uint32_t recoFakeCudaRgbaToNv12CapturedU32(int index) {
  return index >= 0 && static_cast<std::size_t>(index) < captured_u32.size()
             ? captured_u32[static_cast<std::size_t>(index)]
             : 0;
}
RECO_FAKE_CUDA_EXPORT float recoFakeCudaRgbaToNv12CapturedColor(int index) {
  return index >= 0 && static_cast<std::size_t>(index) < captured_color.size()
             ? captured_color[static_cast<std::size_t>(index)]
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
  constexpr std::string_view kName = "fake RGBA-to-NV12 CUDA device";
  const auto count = std::min<std::size_t>(kName.size(), static_cast<std::size_t>(length - 1));
  std::memcpy(name, kName.data(), count);
  name[count] = '\0';
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuDeviceGetUuid(void* uuid, int device) {
  if (uuid == nullptr || device != 0) {
    return 1;
  }
  std::memset(uuid, 0x29, 16);
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
  synchronize_sequence = ++sequence;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuStreamCreate(void** stream, unsigned int flags) {
  if (stream == nullptr || flags != 1U ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  *stream = reinterpret_cast<void*>(kExecutionStream);
  ++stream_create_count;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuStreamDestroy_v2(void* stream) {
  if (stream != reinterpret_cast<void*>(kExecutionStream) || surface_busy_count.load() != 0) {
    return 1;
  }
  ++stream_destroy_count;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuStreamSynchronize(void* stream) {
  if (stream != reinterpret_cast<void*>(kExecutionStream) ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  surface_busy_count = 0;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuEventCreate(void** event, unsigned int flags) {
  if (event == nullptr || flags != 2U ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  *event = reinterpret_cast<void*>(kCompletionEvent);
  ++event_create_count;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuEventDestroy_v2(void* event) {
  if (event != reinterpret_cast<void*>(kCompletionEvent) || surface_busy_count.load() != 0) {
    return 1;
  }
  ++event_destroy_count;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuEventRecord(void* event, void* stream) {
  if (event != reinterpret_cast<void*>(kCompletionEvent) ||
      stream != reinterpret_cast<void*>(kExecutionStream) || surface_busy_count.load() <= 0 ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  ++event_record_count;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuEventSynchronize(void* event) {
  if (event != reinterpret_cast<void*>(kCompletionEvent) || surface_busy_count.load() <= 0 ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  ++event_synchronize_count;
  synchronize_sequence = ++sequence;
  surface_busy_count = 0;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuPointerGetAttribute(void* data, int attribute, std::uint64_t pointer) {
  ++pointer_attribute_count;
  const auto base = allocation_base(pointer);
  if (data == nullptr || current_context != reinterpret_cast<void*>(kContextIdentity) ||
      pointer == 0 || base == kFreedAllocation) {
    return 1;
  }
  switch (attribute) {
  case 1:
    *static_cast<void**>(data) =
        base == kForeignContextAllocation ? reinterpret_cast<void*>(kForeignContextIdentity)
        : base == kContextIndependentMapping || base == kPhysicalAliasMapping ||
                base == kUnknownPhysicalIdentityMapping
            ? nullptr
            : reinterpret_cast<void*>(kContextIdentity);
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
    if (base == kUnknownPhysicalIdentityMapping) {
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
  if (module == nullptr || image == nullptr ||
      current_context != reinterpret_cast<void*>(kContextIdentity)) {
    return 1;
  }
  *module = reinterpret_cast<void*>(0x2902U);
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuModuleUnload(void* module) {
  return module == reinterpret_cast<void*>(0x2902U) ? 0 : 1;
}
RECO_FAKE_CUDA_EXPORT int cuModuleGetFunction(void** function, void* module, const char* name) {
  if (function == nullptr || module != reinterpret_cast<void*>(0x2902U) || name == nullptr ||
      std::string_view(name) != "reco_rgba_to_nv12") {
    return 1;
  }
  *function = reinterpret_cast<void*>(0x2903U);
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuLaunchKernel(void* function, unsigned int grid_x, unsigned int grid_y,
                                         unsigned int grid_z, unsigned int block_x,
                                         unsigned int block_y, unsigned int block_z,
                                         unsigned int shared_memory, void* stream,
                                         void** parameters, void**) {
  if (function != reinterpret_cast<void*>(0x2903U) || parameters == nullptr ||
      current_context != reinterpret_cast<void*>(kContextIdentity) || shared_memory != 0U ||
      stream != reinterpret_cast<void*>(kExecutionStream)) {
    return 1;
  }
  for (std::size_t index = 0; index < captured_u64.size(); ++index) {
    captured_u64[index] = *static_cast<const std::uint64_t*>(parameters[index]);
  }
  captured_u32 = {
      *static_cast<const std::uint32_t*>(parameters[6]),
      *static_cast<const std::uint32_t*>(parameters[7]),
      grid_x,
      grid_y,
      grid_z,
      block_x,
      block_y,
      block_z,
  };
  const auto& color = *static_cast<const ColorParams*>(parameters[8]);
  std::copy(std::begin(color.values), std::end(color.values), captured_color.begin());
  ++launch_count;
  launch_sequence = ++sequence;
  ++surface_busy_count;
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
RECO_FAKE_CUDA_EXPORT int cuMemGetAccess(std::uint64_t* flags, const void*, std::uint64_t pointer) {
  const auto base = allocation_base(pointer);
  if (flags == nullptr || current_context != reinterpret_cast<void*>(kContextIdentity) ||
      (base != kContextIndependentMapping && base != kPhysicalAliasMapping &&
       base != kUnknownPhysicalIdentityMapping)) {
    return 1;
  }
  *flags = 3U;
  return 0;
}
RECO_FAKE_CUDA_EXPORT int cuMemRetainAllocationHandle(std::uint64_t* handle, void* address) {
  const auto base = allocation_base(reinterpret_cast<std::uintptr_t>(address));
  if (handle == nullptr || current_context != reinterpret_cast<void*>(kContextIdentity) ||
      (base != kContextIndependentMapping && base != kPhysicalAliasMapping &&
       base != kUnknownPhysicalIdentityMapping)) {
    return 1;
  }
  *handle = 0x2904U;
  return 0;
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
RECO_FAKE_CUDA_EXPORT int cuMemRelease(std::uint64_t handle) { return handle == 0x2904U ? 0 : 1; }
RECO_FAKE_CUDA_EXPORT int cuMemUnmap(std::uint64_t, std::size_t) { return 1; }
RECO_FAKE_CUDA_EXPORT int cuMemAddressFree(std::uint64_t, std::size_t) { return 1; }

#else

#include "reco/core/cuda_rgba_to_nv12.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "rules_cc/cc/runfiles/runfiles.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using namespace reco::core;

constexpr CudaDevicePtr kContextIndependentMapping = 0x100000U;
constexpr CudaDevicePtr kPhysicalAliasMapping = 0x110000U;
constexpr CudaDevicePtr kUnknownPhysicalIdentityMapping = 0x120000U;

static_assert(!std::is_copy_constructible_v<CudaRgbaToNv12Converter>);
static_assert(!std::is_copy_assignable_v<CudaRgbaToNv12Converter>);
static_assert(std::is_nothrow_move_constructible_v<CudaRgbaToNv12Converter>);
static_assert(std::is_nothrow_move_assignable_v<CudaRgbaToNv12Converter>);

int failures = 0;

template <typename Function> void run_case(std::string_view name, Function&& function) {
  std::cerr << "RUN: " << name << std::endl;
  try {
    function();
    std::cout << "PASS: " << name << '\n';
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
  }
}

void expect_true(bool value, std::string_view message) {
  if (!value) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Actual, typename Expected>
void expect_eq(const Actual& actual, const Expected& expected, std::string_view message) {
  if (!(actual == expected)) {
    throw std::runtime_error(std::string(message));
  }
}

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual));
  }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
  } catch (const Exception& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      throw std::runtime_error(std::string(message) + ": unexpected diagnostic: " + error.what());
    }
    return;
  }
  throw std::runtime_error(std::string(message) + ": expected exception");
}

class DynamicControl {
public:
  explicit DynamicControl(const std::filesystem::path& path) {
#if defined(_WIN32)
    handle_ = LoadLibraryW(path.c_str());
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw std::runtime_error("failed to load test runtime: " + path.string());
    }
  }

  DynamicControl(const DynamicControl&) = delete;
  DynamicControl& operator=(const DynamicControl&) = delete;

  ~DynamicControl() {
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
  }

  template <typename Function> Function symbol(const char* name) const {
#if defined(_WIN32)
    auto* result = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* result = dlsym(handle_, name);
#endif
    if (result == nullptr) {
      throw std::runtime_error(std::string("missing test control symbol ") + name);
    }
    return reinterpret_cast<Function>(result);
  }

private:
  void* handle_ = nullptr;
};

struct FakeCudaControl {
  explicit FakeCudaControl(const std::filesystem::path& path) : library(path) {
    reset_fn = library.symbol<void (*)()>("recoFakeCudaRgbaToNv12Reset");
    launch_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12LaunchCount");
    synchronize_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12SynchronizeCount");
    launch_sequence_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12LaunchSequence");
    synchronize_sequence_fn =
        library.symbol<int (*)()>("recoFakeCudaRgbaToNv12SynchronizeSequence");
    stream_create_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12StreamCreateCount");
    stream_destroy_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12StreamDestroyCount");
    event_create_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12EventCreateCount");
    event_destroy_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12EventDestroyCount");
    event_record_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12EventRecordCount");
    event_synchronize_count_fn =
        library.symbol<int (*)()>("recoFakeCudaRgbaToNv12EventSynchronizeCount");
    surface_busy_count_fn = library.symbol<int (*)()>("recoFakeCudaRgbaToNv12SurfaceBusyCount");
    pointer_attribute_count_fn =
        library.symbol<int (*)()>("recoFakeCudaRgbaToNv12PointerAttributeCount");
    captured_u64_fn = library.symbol<std::uint64_t (*)(int)>("recoFakeCudaRgbaToNv12CapturedU64");
    captured_u32_fn = library.symbol<std::uint32_t (*)(int)>("recoFakeCudaRgbaToNv12CapturedU32");
    captured_color_fn = library.symbol<float (*)(int)>("recoFakeCudaRgbaToNv12CapturedColor");
  }

  void reset() const { reset_fn(); }
  int launch_count() const { return launch_count_fn(); }
  int synchronize_count() const { return synchronize_count_fn(); }
  int pointer_attribute_count() const { return pointer_attribute_count_fn(); }
  int launch_sequence() const { return launch_sequence_fn(); }
  int synchronize_sequence() const { return synchronize_sequence_fn(); }
  int stream_create_count() const { return stream_create_count_fn(); }
  int stream_destroy_count() const { return stream_destroy_count_fn(); }
  int event_create_count() const { return event_create_count_fn(); }
  int event_destroy_count() const { return event_destroy_count_fn(); }
  int event_record_count() const { return event_record_count_fn(); }
  int event_synchronize_count() const { return event_synchronize_count_fn(); }
  int surface_busy_count() const { return surface_busy_count_fn(); }
  std::uint64_t captured_u64(int index) const { return captured_u64_fn(index); }
  std::uint32_t captured_u32(int index) const { return captured_u32_fn(index); }
  float captured_color(int index) const { return captured_color_fn(index); }

  DynamicControl library;
  void (*reset_fn)() = nullptr;
  int (*launch_count_fn)() = nullptr;
  int (*synchronize_count_fn)() = nullptr;
  int (*pointer_attribute_count_fn)() = nullptr;
  int (*launch_sequence_fn)() = nullptr;
  int (*synchronize_sequence_fn)() = nullptr;
  int (*stream_create_count_fn)() = nullptr;
  int (*stream_destroy_count_fn)() = nullptr;
  int (*event_create_count_fn)() = nullptr;
  int (*event_destroy_count_fn)() = nullptr;
  int (*event_record_count_fn)() = nullptr;
  int (*event_synchronize_count_fn)() = nullptr;
  int (*surface_busy_count_fn)() = nullptr;
  std::uint64_t (*captured_u64_fn)(int) = nullptr;
  std::uint32_t (*captured_u32_fn)(int) = nullptr;
  float (*captured_color_fn)(int) = nullptr;
};

struct FakeNvrtcControl {
  explicit FakeNvrtcControl(const std::filesystem::path& path) : library(path) {
    reset_fn = library.symbol<void (*)()>("recoFakeNvrtcReset");
    create_count_fn = library.symbol<int (*)()>("recoFakeNvrtcCreateCount");
    destroy_count_fn = library.symbol<int (*)()>("recoFakeNvrtcDestroyCount");
    last_architecture_fn = library.symbol<int (*)()>("recoFakeNvrtcLastArchitecture");
  }

  void reset() const { reset_fn(); }
  int create_count() const { return create_count_fn(); }
  int destroy_count() const { return destroy_count_fn(); }
  int last_architecture() const { return last_architecture_fn(); }

  DynamicControl library;
  void (*reset_fn)() = nullptr;
  int (*create_count_fn)() = nullptr;
  int (*destroy_count_fn)() = nullptr;
  int (*last_architecture_fn)() = nullptr;
};

std::filesystem::path runtime_path(const char* environment_name, std::string_view runfile) {
  if (const char* explicit_path = std::getenv(environment_name); explicit_path != nullptr) {
    return explicit_path;
  }
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
  const auto logical_path = std::string(workspace) + "/" + std::string(runfile);
  const auto resolved = std::filesystem::path(runfiles->Rlocation(logical_path));
  if (!resolved.empty() && std::filesystem::is_regular_file(resolved)) {
    return resolved;
  }
  throw std::runtime_error(std::string("test runtime runfile not found: ") + std::string(runfile));
}

CudaRgbaFrameView rgba_frame(CudaDevicePtr base, CudaContextId context, int device = 0,
                             std::uint32_t width = 34, std::uint32_t height = 18) {
  const auto row_bytes = static_cast<std::size_t>(width) * 4U;
  const auto pitch = row_bytes + 20U;
  return CudaRgbaFrameView(
      CudaPitchedPlaneView(base, pitch * height, pitch, row_bytes, height, context, device), width,
      height);
}

CudaNv12FrameView nv12_frame(CudaDevicePtr y_base, CudaDevicePtr uv_base, CudaContextId context,
                             YuvColorMatrix matrix = YuvColorMatrix::Bt709,
                             YuvColorRange range = YuvColorRange::Limited, int device = 0,
                             std::uint32_t width = 34, std::uint32_t height = 18) {
  const auto y_pitch = static_cast<std::size_t>(width) + 14U;
  const auto uv_pitch = static_cast<std::size_t>(width) + 30U;
  return CudaNv12FrameView(
      CudaPitchedPlaneView(y_base, y_pitch * height, y_pitch, width, height, context, device),
      CudaPitchedPlaneView(uv_base, uv_pitch * (height / 2U), uv_pitch, width, height / 2U, context,
                           device),
      width, height, matrix, range);
}

CudaRgbaToNv12Converter create_converter(const CudaRgbaToNv12Config& config,
                                         const std::filesystem::path& cuda_runtime,
                                         const std::filesystem::path& nvrtc_runtime) {
  return CudaRgbaToNv12Converter::create(config, CudaBackend::load(cuda_runtime.string()),
                                         NvrtcCompiler::load(nvrtc_runtime.string()));
}

void compiles_once_and_uses_event_scoped_completion(const std::filesystem::path& cuda_runtime,
                                                    const std::filesystem::path& nvrtc_runtime,
                                                    const FakeCudaControl& cuda_control,
                                                    const FakeNvrtcControl& nvrtc_control) {
  cuda_control.reset();
  nvrtc_control.reset();
  auto converter = create_converter({.width = 34, .height = 18}, cuda_runtime, nvrtc_runtime);
  expect_eq(nvrtc_control.create_count(), 1, "converter performs one NVRTC compilation");
  expect_eq(nvrtc_control.destroy_count(), 1, "converter releases its NVRTC program");
  expect_eq(nvrtc_control.last_architecture(), 86,
            "converter selects the highest compatible NVRTC architecture");
  expect_eq(converter.device_ordinal(), 0, "converter device ordinal");
  expect_eq(converter.width(), 34U, "converter width");
  expect_eq(converter.height(), 18U, "converter height");

  const auto context = converter.context_id();
  const auto input = rgba_frame(0x10000U, context);
  const auto limited =
      nv12_frame(0x40000U, 0x50000U, context, YuvColorMatrix::Bt601, YuvColorRange::Limited);
  converter.convert(input, limited);
  expect_near(cuda_control.captured_color(0), 0.299F, 1.0e-6F, "BT.601 red luma");
  expect_near(cuda_control.captured_color(1), 0.587F, 1.0e-6F, "BT.601 green luma");
  expect_near(cuda_control.captured_color(2), 0.114F, 1.0e-6F, "BT.601 blue luma");
  expect_near(cuda_control.captured_color(3), 219.0F / 255.0F, 1.0e-6F, "limited luma scale");
  expect_near(cuda_control.captured_color(4), 16.0F, 1.0e-6F, "limited luma offset");
  expect_near(cuda_control.captured_color(5), 224.0F / 255.0F, 1.0e-6F, "limited chroma scale");
  expect_near(cuda_control.captured_color(6), 128.0F, 1.0e-6F, "limited chroma center");

  const auto full =
      nv12_frame(0x60000U, 0x70000U, context, YuvColorMatrix::Bt2020, YuvColorRange::Full);
  converter.convert(input, full);
  expect_eq(nvrtc_control.create_count(), 1, "conversion does not recompile");
  expect_eq(cuda_control.launch_count(), 2, "one kernel launch per conversion");
  expect_eq(cuda_control.synchronize_count(), 0,
            "conversion does not synchronize unrelated CUDA context work");
  expect_eq(cuda_control.stream_create_count(), 1, "converter retains one execution stream");
  expect_eq(cuda_control.event_create_count(), 1, "converter retains one completion event");
  expect_eq(cuda_control.event_record_count(), 2, "each conversion records completion");
  expect_eq(cuda_control.event_synchronize_count(), 2,
            "each conversion waits for its completion event");
  expect_eq(cuda_control.surface_busy_count(), 0,
            "NV12 output is complete before its surface can be reused");
  expect_eq(cuda_control.pointer_attribute_count(), 42,
            "each conversion validates all three pointers through seven CUDA attributes");
  expect_true(cuda_control.launch_sequence() < cuda_control.synchronize_sequence(),
              "kernel launch precedes synchronization");
  expect_eq(cuda_control.captured_u64(0), input.plane().ptr(), "input pointer propagated");
  expect_eq(cuda_control.captured_u64(1), input.plane().pitch_bytes(), "input pitch propagated");
  expect_eq(cuda_control.captured_u64(2), full.y_plane().ptr(), "Y pointer propagated");
  expect_eq(cuda_control.captured_u64(3), full.y_plane().pitch_bytes(), "Y pitch propagated");
  expect_eq(cuda_control.captured_u64(4), full.uv_plane().ptr(), "UV pointer propagated");
  expect_eq(cuda_control.captured_u64(5), full.uv_plane().pitch_bytes(), "UV pitch propagated");
  expect_eq(cuda_control.captured_u32(0), 34U, "width propagated");
  expect_eq(cuda_control.captured_u32(1), 18U, "height propagated");
  expect_eq(cuda_control.captured_u32(2), 2U, "bounded grid X");
  expect_eq(cuda_control.captured_u32(3), 1U, "bounded grid Y");
  expect_eq(cuda_control.captured_u32(5), 16U, "block X");
  expect_eq(cuda_control.captured_u32(6), 16U, "block Y");
  expect_near(cuda_control.captured_color(0), 0.2627F, 1.0e-6F, "BT.2020 red luma");
  expect_near(cuda_control.captured_color(1), 0.6780F, 1.0e-6F, "BT.2020 green luma");
  expect_near(cuda_control.captured_color(2), 0.0593F, 1.0e-6F, "BT.2020 blue luma");
  expect_near(cuda_control.captured_color(3), 1.0F, 1.0e-6F, "full luma scale");
  expect_near(cuda_control.captured_color(4), 0.0F, 1.0e-6F, "full luma offset");
  expect_near(cuda_control.captured_color(5), 1.0F, 1.0e-6F, "full chroma scale");
  expect_near(cuda_control.captured_color(6), 127.5F, 1.0e-6F, "full chroma center");
}

void enforces_device_access_permissions(const std::filesystem::path& cuda_runtime,
                                        const std::filesystem::path& nvrtc_runtime,
                                        const FakeCudaControl& cuda_control) {
  auto converter = create_converter({.width = 34, .height = 18}, cuda_runtime, nvrtc_runtime);
  const auto context = converter.context_id();
  const auto output = nv12_frame(0x40000U, 0x50000U, context);
  cuda_control.reset();

  converter.convert(rgba_frame(0xF0000U, context), output);
  expect_eq(cuda_control.launch_count(), 1, "read-only RGBA input is accepted");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(rgba_frame(0xE0000U, context), output); }, "reads",
      "inaccessible RGBA input is rejected");
  expect_throws<std::invalid_argument>(
      [&] {
        converter.convert(rgba_frame(0x10000U, context), nv12_frame(0xF0000U, 0x70000U, context));
      },
      "writes", "read-only Y output is rejected");
  expect_throws<std::invalid_argument>(
      [&] {
        converter.convert(rgba_frame(0x10000U, context), nv12_frame(0x60000U, 0xF0000U, context));
      },
      "writes", "read-only UV output is rejected");
  expect_eq(cuda_control.launch_count(), 1, "denied converter access never launches");
}

void retained_validation_avoids_conversion_time_queries(const std::filesystem::path& cuda_runtime,
                                                        const std::filesystem::path& nvrtc_runtime,
                                                        const FakeCudaControl& cuda_control) {
  auto backend = CudaBackend::load(cuda_runtime.string());
  auto converter = CudaRgbaToNv12Converter::create({.width = 34, .height = 18}, backend,
                                                   NvrtcCompiler::load(nvrtc_runtime.string()));
  const auto input_pitch = 34U * 4U + 20U;
  const auto y_pitch = 34U + 14U;
  const auto uv_pitch = 34U + 30U;
  const CudaRgbaFrameView input(
      CudaPitchedPlaneView(
          backend.retain_device_span(0x10000U, input_pitch * 18U, CudaSpanAccess::Read),
          input_pitch, 34U * 4U, 18U),
      34U, 18U);
  const CudaNv12FrameView output(
      CudaPitchedPlaneView(
          backend.retain_device_span(0x40000U, y_pitch * 18U, CudaSpanAccess::ReadWrite), y_pitch,
          34U, 18U),
      CudaPitchedPlaneView(
          backend.retain_device_span(0x50000U, uv_pitch * 9U, CudaSpanAccess::ReadWrite), uv_pitch,
          34U, 9U),
      34U, 18U, YuvColorMatrix::Bt709, YuvColorRange::Limited);

  cuda_control.reset();
  converter.convert(input, output);
  expect_eq(cuda_control.pointer_attribute_count(), 0,
            "retained converter views avoid per-frame CUDA pointer queries");
  expect_eq(cuda_control.launch_count(), 1, "retained converter views launch normally");
}

void rejects_physical_vmm_aliases(const std::filesystem::path& cuda_runtime,
                                  const std::filesystem::path& nvrtc_runtime,
                                  const FakeCudaControl& cuda_control) {
  auto converter = create_converter({.width = 34, .height = 18}, cuda_runtime, nvrtc_runtime);
  const auto context = converter.context_id();
  cuda_control.reset();
  expect_throws<std::invalid_argument>(
      [&] {
        converter.convert(rgba_frame(kContextIndependentMapping, context),
                          nv12_frame(kPhysicalAliasMapping, 0x70000U, context));
      },
      "overlap", "distinct VMM mappings of one physical allocation are rejected");
  expect_eq(cuda_control.launch_count(), 0, "physical alias rejection prevents conversion");

  expect_throws<std::invalid_argument>(
      [&] {
        converter.convert(rgba_frame(0x10000U, context),
                          nv12_frame(kContextIndependentMapping, kPhysicalAliasMapping, context));
      },
      "Y and UV", "physical aliasing between distinct VMM output mappings is rejected");
  expect_eq(cuda_control.launch_count(), 0, "aliased VMM outputs do not launch the converter");

  converter.convert(
      rgba_frame(0x10000U, context),
      nv12_frame(kContextIndependentMapping, kContextIndependentMapping + 0x800U, context));
  expect_eq(cuda_control.launch_count(), 1,
            "disjoint Y and UV ranges in one VMM mapping remain supported");

  expect_throws<std::invalid_argument>(
      [&] {
        converter.convert(
            rgba_frame(0x10000U, context),
            nv12_frame(kContextIndependentMapping, kUnknownPhysicalIdentityMapping, context));
      },
      "Y and UV", "unknown VMM output identity fails closed");
  expect_eq(cuda_control.launch_count(), 1, "unknown VMM identity does not launch the converter");
}

void rejects_invalid_configuration(const std::filesystem::path& cuda_runtime,
                                   const std::filesystem::path& nvrtc_runtime,
                                   const FakeNvrtcControl& nvrtc_control) {
  const auto check = [&](CudaRgbaToNv12Config config, std::string_view fragment,
                         std::string_view label) {
    nvrtc_control.reset();
    expect_throws<std::exception>(
        [&] { (void)create_converter(config, cuda_runtime, nvrtc_runtime); }, fragment, label);
    expect_eq(nvrtc_control.create_count(), 0, std::string(label) + " does not compile");
  };
  check({.width = 0, .height = 2}, "non-zero and even", "zero width");
  check({.width = 3, .height = 2}, "non-zero and even", "odd width");
  check({.width = 2, .height = 3}, "non-zero and even", "odd height");
  check({.width = std::numeric_limits<std::uint32_t>::max() - 1U,
         .height = std::numeric_limits<std::uint32_t>::max() - 1U},
        "overflows", "overflowing frame extent");
  check({.width = 2, .height = 2U * 16U * 65'536U}, "grid limits", "oversized grid");
  check({.width = 2, .height = 2, .device_ordinal = -1}, "out of range", "negative device");
  check({.width = 2, .height = 2, .device_ordinal = 1}, "out of range", "missing device");
}

void rejects_unsafe_frames(const std::filesystem::path& cuda_runtime,
                           const std::filesystem::path& nvrtc_runtime,
                           const FakeCudaControl& cuda_control) {
  auto converter = create_converter({.width = 34, .height = 18}, cuda_runtime, nvrtc_runtime);
  const auto context = converter.context_id();
  const auto input = rgba_frame(0x10000U, context);
  const auto output = nv12_frame(0x40000U, 0x50000U, context);
  cuda_control.reset();

  expect_throws<std::invalid_argument>(
      [&] { converter.convert(rgba_frame(0x60000U, context, 0, 32, 18), output); }, "dimensions",
      "input shape mismatch");
  expect_throws<std::invalid_argument>(
      [&] {
        converter.convert(input, nv12_frame(0x60000U, 0x70000U, context, YuvColorMatrix::Bt709,
                                            YuvColorRange::Limited, 0, 32, 18));
      },
      "dimensions", "output shape mismatch");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(rgba_frame(0x60000U, context + 1U), output); }, "context",
      "input context mismatch");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(input, nv12_frame(0x60000U, 0x70000U, context + 1U)); }, "context",
      "output context mismatch");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(rgba_frame(0x60000U, context, 1), output); }, "device",
      "input device mismatch");
  expect_throws<std::invalid_argument>(
      [&] {
        converter.convert(input, nv12_frame(0x60000U, 0x70000U, context, YuvColorMatrix::Bt709,
                                            YuvColorRange::Limited, 1));
      },
      "device", "output device mismatch");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(input, nv12_frame(input.plane().ptr(), 0x70000U, context)); },
      "overlap", "input and Y overlap");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(input, nv12_frame(0x70000U, input.plane().ptr(), context)); },
      "overlap", "input and UV overlap");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(input, nv12_frame(0x60000U, 0x60000U, context)); }, "overlap",
      "Y and UV outputs overlap");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(rgba_frame(0x80000U, context), output); }, "device memory",
      "host input pointer");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(rgba_frame(0x90000U, context), output); }, "rejected",
      "freed input pointer");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(rgba_frame(0xA0000U, context), output); }, "exceeds",
      "undersized input allocation");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(input, nv12_frame(0xB0000U, 0x70000U, context)); },
      "different CUDA context", "foreign-context Y output");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(input, nv12_frame(0x60000U, 0xC0000U, context)); },
      "different CUDA device", "foreign-device UV output");
  expect_throws<std::invalid_argument>(
      [&] { converter.convert(input, nv12_frame(0x60000U, 0xD0000U, context)); }, "not mapped",
      "unmapped UV output");
  expect_eq(cuda_control.launch_count(), 0, "invalid frames do not launch");
  expect_eq(cuda_control.synchronize_count(), 0, "invalid frames do not synchronize");
}

void moved_from_converter_is_diagnosed(const std::filesystem::path& cuda_runtime,
                                       const std::filesystem::path& nvrtc_runtime) {
  auto source = create_converter({.width = 34, .height = 18}, cuda_runtime, nvrtc_runtime);
  const auto context = source.context_id();
  auto converter = std::move(source);
  expect_eq(converter.context_id(), context, "moved converter retains context");
  expect_throws<std::logic_error>([&] { (void)source.context_id(); }, "moved-from",
                                  "moved-from context access");
  expect_throws<std::logic_error>(
      [&] {
        source.convert(rgba_frame(0x10000U, context), nv12_frame(0x40000U, 0x50000U, context));
      },
      "moved-from", "moved-from conversion");
}

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) != "0";
}

bool skip_cuda() {
  const char* value = std::getenv("RECO_SKIP_CUDA_TEST");
  return value != nullptr && std::string_view(value) != "0";
}

struct CpuColor {
  float kr = 0.0F;
  float kb = 0.0F;
  float y_scale = 1.0F;
  float y_offset = 0.0F;
  float chroma_scale = 1.0F;
  float chroma_center = 128.0F;
};

CpuColor cpu_color(YuvColorMatrix matrix, YuvColorRange range) {
  CpuColor color;
  switch (matrix) {
  case YuvColorMatrix::Bt601:
    color.kr = 0.299F;
    color.kb = 0.114F;
    break;
  case YuvColorMatrix::Bt709:
    color.kr = 0.2126F;
    color.kb = 0.0722F;
    break;
  case YuvColorMatrix::Bt2020:
    color.kr = 0.2627F;
    color.kb = 0.0593F;
    break;
  }
  if (range == YuvColorRange::Limited) {
    color.y_scale = 219.0F / 255.0F;
    color.y_offset = 16.0F;
    color.chroma_scale = 224.0F / 255.0F;
  } else {
    color.chroma_center = 127.5F;
  }
  return color;
}

std::uint8_t quantize(float value) {
  return static_cast<std::uint8_t>(std::floor(std::clamp(value, 0.0F, 255.0F) + 0.5F));
}

void hardware_parity_if_available() {
  if (skip_cuda()) {
    std::cout << "SKIP: hardware CUDA RGBA-to-NV12 test disabled\n";
    return;
  }
  const auto cuda_error = CudaBackend::availability_error();
  const auto nvrtc_error = NvrtcCompiler::availability_error();
  if (!cuda_error.empty() || !nvrtc_error.empty()) {
    const auto diagnostic =
        "CUDA=" + (cuda_error.empty() ? std::string("available") : cuda_error) +
        " NVRTC=" + (nvrtc_error.empty() ? std::string("available") : nvrtc_error);
    if (require_cuda()) {
      throw std::runtime_error("required CUDA RGBA-to-NV12 test unavailable: " + diagnostic);
    }
    std::cout << "SKIP: hardware CUDA RGBA-to-NV12 test unavailable: " << diagnostic << '\n';
    return;
  }

  constexpr std::uint32_t width = 4;
  constexpr std::uint32_t height = 2;
  const std::array<std::uint8_t, width * height * 4U> rgba = {
      255, 0, 0, 1, 0,   255, 0,   2, 0,   0,   255, 3, 255, 255, 255, 4,
      0,   0, 0, 5, 127, 127, 127, 6, 255, 255, 0,   7, 0,   255, 255, 8,
  };

  auto backend = CudaBackend::create();
  const auto context = backend.primary_context_id();
  auto input_storage = backend.allocate_pitched(width * 4U, height, 4);
  auto y_storage = backend.allocate_pitched(width, height, 4);
  auto uv_storage = backend.allocate_pitched(width, height / 2U, 4);
  backend.copy_host_to_device_2d({.src = rgba.data(),
                                  .src_pitch = width * 4U,
                                  .dst = input_storage.buffer.ptr(),
                                  .dst_pitch = input_storage.pitch,
                                  .width_bytes = width * 4U,
                                  .height = height});
  const CudaRgbaFrameView input(
      CudaPitchedPlaneView(input_storage.buffer.ptr(), input_storage.buffer.size(),
                           input_storage.pitch, width * 4U, height, context),
      width, height);
  auto converter = CudaRgbaToNv12Converter::create({.width = width, .height = height}, backend,
                                                   NvrtcCompiler::create());

  const std::array<YuvColorMatrix, 3> matrices = {YuvColorMatrix::Bt601, YuvColorMatrix::Bt709,
                                                  YuvColorMatrix::Bt2020};
  const std::array<YuvColorRange, 2> ranges = {YuvColorRange::Limited, YuvColorRange::Full};
  for (const auto matrix : matrices) {
    for (const auto range : ranges) {
      const CudaNv12FrameView output(
          CudaPitchedPlaneView(y_storage.buffer.ptr(), y_storage.buffer.size(), y_storage.pitch,
                               width, height, context),
          CudaPitchedPlaneView(uv_storage.buffer.ptr(), uv_storage.buffer.size(), uv_storage.pitch,
                               width, height / 2U, context),
          width, height, matrix, range);
      converter.convert(input, output);

      std::array<std::uint8_t, width * height> actual_y{};
      std::array<std::uint8_t, width> actual_uv{};
      backend.copy_device_to_host_2d({.dst = actual_y.data(),
                                      .dst_pitch = width,
                                      .src = y_storage.buffer.ptr(),
                                      .src_pitch = y_storage.pitch,
                                      .width_bytes = width,
                                      .height = height});
      backend.copy_device_to_host_2d({.dst = actual_uv.data(),
                                      .dst_pitch = width,
                                      .src = uv_storage.buffer.ptr(),
                                      .src_pitch = uv_storage.pitch,
                                      .width_bytes = width,
                                      .height = height / 2U});

      const auto color = cpu_color(matrix, range);
      const float kg = 1.0F - color.kr - color.kb;
      std::array<float, width * height> luma{};
      for (std::size_t pixel = 0; pixel < luma.size(); ++pixel) {
        const float red = rgba[pixel * 4U];
        const float green = rgba[pixel * 4U + 1U];
        const float blue = rgba[pixel * 4U + 2U];
        luma[pixel] = color.kr * red + kg * green + color.kb * blue;
        const auto expected = quantize(color.y_offset + color.y_scale * luma[pixel]);
        expect_true(std::abs(static_cast<int>(actual_y[pixel]) - static_cast<int>(expected)) <= 1,
                    "hardware luma parity");
      }
      for (std::size_t block = 0; block < width / 2U; ++block) {
        const std::array<std::size_t, 4> pixels = {block * 2U, block * 2U + 1U, width + block * 2U,
                                                   width + block * 2U + 1U};
        float red = 0.0F;
        float green = 0.0F;
        float blue = 0.0F;
        for (const auto pixel : pixels) {
          red += rgba[pixel * 4U];
          green += rgba[pixel * 4U + 1U];
          blue += rgba[pixel * 4U + 2U];
        }
        red *= 0.25F;
        green *= 0.25F;
        blue *= 0.25F;
        const float average_y = color.kr * red + kg * green + color.kb * blue;
        const auto expected_u =
            quantize(color.chroma_center +
                     color.chroma_scale * (blue - average_y) / (2.0F * (1.0F - color.kb)));
        const auto expected_v =
            quantize(color.chroma_center +
                     color.chroma_scale * (red - average_y) / (2.0F * (1.0F - color.kr)));
        expect_true(
            std::abs(static_cast<int>(actual_uv[block * 2U]) - static_cast<int>(expected_u)) <= 1,
            "hardware chroma U parity");
        expect_true(std::abs(static_cast<int>(actual_uv[block * 2U + 1U]) -
                             static_cast<int>(expected_v)) <= 1,
                    "hardware chroma V parity");
      }
    }
  }
  std::cout << "hardware CUDA RGBA-to-NV12 parity executed\n";
}

} // namespace

int main() {
  try {
    constexpr std::string_view kFakeCudaRunfile =
        "cpp/tests/libreco_core_fake_cuda_rgba_to_nv12_driver.so";
    constexpr std::string_view kFakeNvrtcRunfile = "cpp/tests/libreco_core_fake_nvrtc_runtime.so";
    std::cerr << "RUN: fake runtime setup" << std::endl;
    const auto cuda_runtime = runtime_path("RECO_TEST_FAKE_CUDA_DRIVER", kFakeCudaRunfile);
    const auto nvrtc_runtime = runtime_path("RECO_TEST_FAKE_NVRTC_RUNTIME", kFakeNvrtcRunfile);
    FakeCudaControl cuda_control(cuda_runtime);
    FakeNvrtcControl nvrtc_control(nvrtc_runtime);

    run_case("compile once with event-scoped completion", [&] {
      compiles_once_and_uses_event_scoped_completion(cuda_runtime, nvrtc_runtime, cuda_control,
                                                     nvrtc_control);
    });
    run_case("configuration validation",
             [&] { rejects_invalid_configuration(cuda_runtime, nvrtc_runtime, nvrtc_control); });
    run_case("frame contract validation",
             [&] { rejects_unsafe_frames(cuda_runtime, nvrtc_runtime, cuda_control); });
    run_case("device access validation", [&] {
      enforces_device_access_permissions(cuda_runtime, nvrtc_runtime, cuda_control);
    });
    run_case("retained validation conversion path", [&] {
      retained_validation_avoids_conversion_time_queries(cuda_runtime, nvrtc_runtime, cuda_control);
    });
    run_case("physical VMM alias validation",
             [&] { rejects_physical_vmm_aliases(cuda_runtime, nvrtc_runtime, cuda_control); });
    run_case("moved-from converter",
             [&] { moved_from_converter_is_diagnosed(cuda_runtime, nvrtc_runtime); });
    run_case("hardware parity", hardware_parity_if_available);

    if (failures != 0) {
      std::cerr << failures << " test(s) failed\n";
      return EXIT_FAILURE;
    }
    std::cout << "all tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: test setup: " << error.what() << '\n';
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "FAIL: test setup threw a non-standard exception\n";
    return EXIT_FAILURE;
  }
}

#endif
