#include "reco/core/cuda_backend.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

using namespace reco::core;

namespace {

static_assert(!std::is_copy_constructible_v<CudaModule>);
static_assert(!std::is_copy_assignable_v<CudaModule>);
static_assert(std::is_nothrow_move_constructible_v<CudaModule>);
static_assert(std::is_nothrow_move_assignable_v<CudaModule>);
static_assert(!std::is_copy_constructible_v<CudaKernel>);
static_assert(!std::is_copy_assignable_v<CudaKernel>);
static_assert(std::is_nothrow_move_constructible_v<CudaKernel>);
static_assert(std::is_nothrow_move_assignable_v<CudaKernel>);

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_invalid_argument(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_overflow_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::overflow_error&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_runtime_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::runtime_error&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

bool address_sanitizer_build() {
#if defined(__SANITIZE_ADDRESS__)
  return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
  return true;
#else
  return false;
#endif
#else
  return false;
#endif
}

bool valid_shareable_handle(CudaShareableHandle handle) {
#if defined(_WIN32)
  return handle != nullptr;
#else
  return handle >= 0;
#endif
}

void close_released_handle(CudaShareableHandle handle) {
#if defined(_WIN32)
  CloseHandle(handle);
#else
  close(handle);
#endif
}

void primary_context_replaces_same_device_secondary(const CudaBackend& backend) {
#if defined(__linux__)
  void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (driver == nullptr) {
    throw std::runtime_error("failed to load CUDA driver for context ownership test");
  }
  struct DriverCloser {
    void* driver;
    ~DriverCloser() { dlclose(driver); }
  } closer{driver};
  const auto symbol = [driver](const char* name) {
    auto* value = dlsym(driver, name);
    if (value == nullptr) {
      throw std::runtime_error(std::string("missing CUDA test symbol ") + name);
    }
    return value;
  };
  using Result = int;
  using Device = int;
  using Context = void*;
  const auto device_get = reinterpret_cast<Result (*)(Device*, int)>(symbol("cuDeviceGet"));
  const auto context_create =
      reinterpret_cast<Result (*)(Context*, unsigned int, Device)>(symbol("cuCtxCreate_v2"));
  const auto context_destroy = reinterpret_cast<Result (*)(Context)>(symbol("cuCtxDestroy_v2"));
  const auto context_get_current =
      reinterpret_cast<Result (*)(Context*)>(symbol("cuCtxGetCurrent"));
  const auto context_set_current = reinterpret_cast<Result (*)(Context)>(symbol("cuCtxSetCurrent"));
  Device device = 0;
  Context secondary = nullptr;
  if (device_get(&device, 0) != 0 || context_create(&secondary, 0, device) != 0 ||
      secondary == nullptr) {
    throw std::runtime_error("failed to create CUDA secondary context for ownership test");
  }
  try {
    backend.ensure_primary_context(0);
    Context observed = nullptr;
    if (context_get_current(&observed) != 0) {
      throw std::runtime_error("failed to inspect CUDA context after primary selection");
    }
    expect_true(observed != secondary,
                "primary context replaces a current same-device secondary context");
    if (context_set_current(secondary) != 0 || context_destroy(secondary) != 0) {
      throw std::runtime_error("failed to destroy CUDA secondary context");
    }
    secondary = nullptr;
    backend.ensure_primary_context(0);
  } catch (...) {
    if (secondary != nullptr) {
      (void)context_set_current(secondary);
      (void)context_destroy(secondary);
    }
    throw;
  }
#else
  (void)backend;
#endif
}

constexpr char kFillBytesPtx[] = R"ptx(
.version 8.0
.target sm_50
.address_size 64

.visible .entry fill_bytes(
    .param .u64 fill_bytes_param_0,
    .param .u32 fill_bytes_param_1,
    .param .u32 fill_bytes_param_2
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [fill_bytes_param_0];
    ld.param.u32 %r1, [fill_bytes_param_1];
    ld.param.u32 %r2, [fill_bytes_param_2];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;
    cvt.u64.u32 %rd2, %r6;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u8 [%rd3], %r1;

DONE:
    ret;
}

.visible .entry increment_bytes(
    .param .u64 increment_bytes_param_0,
    .param .u32 increment_bytes_param_1
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [increment_bytes_param_0];
    ld.param.u32 %r1, [increment_bytes_param_1];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra INCREMENT_DONE;
    cvt.u64.u32 %rd2, %r5;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u8 %r6, [%rd3];
    add.u32 %r7, %r6, 1;
    st.global.u8 [%rd3], %r7;

INCREMENT_DONE:
    ret;
}
)ptx";

struct CountingCudaTrace final : CudaBackendTraceSink {
  void device_allocation_created(std::size_t bytes) noexcept override {
    ++allocations;
    allocated_bytes += bytes;
  }
  void device_allocation_released(std::size_t bytes) noexcept override {
    ++releases;
    released_bytes += bytes;
  }
  void device_to_device_copy_submitted() noexcept override { ++copies; }
  void device_to_host_copy_submitted(std::size_t width_bytes,
                                     std::size_t height) noexcept override {
    ++host_copies;
    host_copy_bytes += width_bytes * height;
  }
  void context_synchronized() noexcept override { ++synchronizations; }

  std::size_t allocations = 0;
  std::size_t releases = 0;
  std::size_t allocated_bytes = 0;
  std::size_t released_bytes = 0;
  std::size_t copies = 0;
  std::size_t host_copies = 0;
  std::size_t host_copy_bytes = 0;
  std::size_t synchronizations = 0;
};

} // namespace

int main() {
  if (address_sanitizer_build() && !require_cuda()) {
    std::cerr << "SKIP: CUDA driver smoke test is skipped under ASan unless explicitly required\n";
    return EXIT_SUCCESS;
  }

  if (!CudaBackend::is_available()) {
    const auto error = CudaBackend::availability_error();
    if (require_cuda()) {
      std::cerr << "FAIL: CUDA unavailable: " << error << '\n';
      return EXIT_FAILURE;
    }
    std::cerr << "SKIP: CUDA unavailable: " << error << '\n';
    return EXIT_SUCCESS;
  }

  try {
    const auto backend = CudaBackend::create();
    expect_true(backend.device_count() > 0, "device count");
    const auto device = backend.device_info(0);
    expect_true(!device.name.empty(), "device name");
    expect_eq(device.uuid.size(), 16U, "device uuid size");

    const auto fresh_memory_info = CudaBackend::create().memory_info();
    expect_true(fresh_memory_info.total_bytes > 0, "fresh memory info total");
    expect_true(fresh_memory_info.free_bytes > 0, "fresh memory info free");

    backend.ensure_primary_context(0);
    backend.ensure_primary_context(0);
    primary_context_replaces_same_device_secondary(backend);
    const auto before = backend.memory_info();
    expect_true(before.total_bytes > 0, "total device memory");
    expect_true(before.free_bytes > 0, "free device memory");
    if (backend.device_count() > 1) {
      backend.ensure_primary_context(1);
      const auto second_device = backend.memory_info();
      expect_true(second_device.total_bytes > 0, "second device total memory");
      backend.ensure_primary_context(0);
    }

    std::atomic<int> thread_failures = 0;
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
      threads.emplace_back([&] {
        const auto threaded_backend = CudaBackend::create();
        threaded_backend.ensure_primary_context(0);
        const auto info = threaded_backend.memory_info();
        if (info.total_bytes == 0 || info.free_bytes == 0) {
          thread_failures.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    expect_eq(thread_failures.load(std::memory_order_relaxed), 0, "threaded backend failures");

    auto buffer = backend.allocate(4096);
    expect_true(static_cast<bool>(buffer), "allocated device buffer");
    expect_eq(buffer.size(), 4096U, "device buffer size");
    backend.memset_d8(buffer, 0xA5);
    backend.synchronize();
    expect_invalid_argument(
        [&] {
          auto ignored = backend.with_trace_sink({});
          (void)ignored;
        },
        "null CUDA trace sink validation");
    auto trace = std::make_shared<CountingCudaTrace>();
    auto observed_backend = backend.with_trace_sink(trace);
    auto observed_buffer = observed_backend.allocate(257);
    expect_eq(trace->allocations, 1U, "successful CUDA allocation observed");
    expect_eq(trace->allocated_bytes, 257U, "CUDA allocation byte count observed");
    observed_buffer.reset();
    expect_eq(trace->releases, 1U, "successful CUDA allocation release observed");
    expect_eq(trace->released_bytes, 257U, "CUDA release byte count observed");
    observed_backend.synchronize();
    expect_eq(trace->synchronizations, 1U, "successful CUDA synchronization observed");
    const auto host = observed_backend.copy_to_host(buffer);
    expect_eq(host.size(), 4096U, "host copy size");
    expect_eq(trace->host_copies, 1U, "successful CUDA D2H submission observed");
    expect_eq(trace->host_copy_bytes, 4096U, "linear CUDA D2H byte count observed");
    for (const auto byte : host) {
      if (byte != 0xA5) {
        expect_true(false, "device memset byte pattern");
        break;
      }
    }

    expect_invalid_argument(
        [&] {
          auto ignored = backend.load_kernel_from_ptx("", "fill_bytes");
          (void)ignored;
        },
        "empty PTX validation");
    expect_invalid_argument(
        [&] {
          auto ignored = backend.load_kernel_from_ptx(kFillBytesPtx, "");
          (void)ignored;
        },
        "empty kernel name validation");
    std::string ptx_with_interior_nul(kFillBytesPtx);
    ptx_with_interior_nul.insert(ptx_with_interior_nul.size() / 2, 1, '\0');
    expect_invalid_argument(
        [&] {
          auto ignored = backend.load_kernel_from_ptx(ptx_with_interior_nul, "fill_bytes");
          (void)ignored;
        },
        "interior PTX NUL validation");
    expect_invalid_argument(
        [&] {
          auto ignored = backend.load_module_from_ptx("");
          (void)ignored;
        },
        "empty module PTX validation");
    expect_invalid_argument(
        [&] {
          auto ignored = backend.load_module_from_ptx(ptx_with_interior_nul);
          (void)ignored;
        },
        "interior module PTX NUL validation");
    std::string null_terminated_ptx(kFillBytesPtx);
    null_terminated_ptx.push_back('\0');
    auto null_terminated_kernel = backend.load_kernel_from_ptx(null_terminated_ptx, "fill_bytes");
    expect_true(static_cast<bool>(null_terminated_kernel), "loaded null-terminated CUDA kernel");
    null_terminated_kernel.reset();

    CudaModule empty_module;
    expect_true(!static_cast<bool>(empty_module), "default CUDA module is empty");
    expect_invalid_argument(
        [&] {
          auto ignored = empty_module.load_kernel("fill_bytes");
          (void)ignored;
        },
        "empty CUDA module kernel lookup validation");
    auto shared_module = backend.load_module_from_ptx(null_terminated_ptx);
    expect_true(static_cast<bool>(shared_module), "loaded shared CUDA module");
    expect_invalid_argument(
        [&] {
          auto ignored = shared_module.load_kernel("");
          (void)ignored;
        },
        "empty shared-module kernel name validation");
    expect_runtime_error(
        [&] {
          auto ignored = shared_module.load_kernel("missing_function");
          (void)ignored;
        },
        "missing shared-module kernel validation");
    auto shared_fill = shared_module.load_kernel("fill_bytes");
    auto shared_increment = shared_module.load_kernel("increment_bytes");
    expect_true(static_cast<bool>(shared_fill), "resolved first shared-module kernel");
    expect_true(static_cast<bool>(shared_increment), "resolved second shared-module kernel");

    auto moved_module = std::move(shared_module);
    expect_true(!static_cast<bool>(shared_module), "moved-from CUDA module is empty");
    expect_true(static_cast<bool>(moved_module), "move-constructed CUDA module remains live");
    expect_invalid_argument(
        [&] {
          auto ignored = shared_module.load_kernel("fill_bytes");
          (void)ignored;
        },
        "moved-from CUDA module lookup validation");
    CudaModule assigned_module;
    assigned_module = std::move(moved_module);
    expect_true(!static_cast<bool>(moved_module), "move-assigned source CUDA module is empty");
    expect_true(static_cast<bool>(assigned_module), "move-assigned CUDA module remains live");
    assigned_module.reset();
    expect_true(!static_cast<bool>(assigned_module), "reset CUDA module is empty");
    expect_true(static_cast<bool>(shared_fill), "first kernel retains reset shared module");
    expect_true(static_cast<bool>(shared_increment), "second kernel retains reset shared module");

    auto shared_buffer = backend.allocate(257);
    backend.memset_d8(shared_buffer, 0);
    CudaDevicePtr shared_ptr = shared_buffer.ptr();
    std::uint32_t shared_fill_value = 0x3A;
    std::uint32_t shared_count = static_cast<std::uint32_t>(shared_buffer.size());
    void* shared_fill_args[] = {&shared_ptr, &shared_fill_value, &shared_count};
    shared_fill.launch({.grid = {.x = 9}, .block = {.x = 32}}, shared_fill_args);
    shared_fill.reset();
    expect_true(!static_cast<bool>(shared_fill), "reset first shared-module kernel is empty");

    auto moved_increment = std::move(shared_increment);
    expect_true(!static_cast<bool>(shared_increment), "moved-from CUDA kernel is empty");
    expect_true(static_cast<bool>(moved_increment), "move-constructed CUDA kernel remains live");
    CudaKernel assigned_increment;
    assigned_increment = std::move(moved_increment);
    expect_true(!static_cast<bool>(moved_increment), "move-assigned source CUDA kernel is empty");
    expect_true(static_cast<bool>(assigned_increment), "move-assigned CUDA kernel remains live");
    void* increment_args[] = {&shared_ptr, &shared_count};
    assigned_increment.launch({.grid = {.x = 9}, .block = {.x = 32}}, increment_args);
    assigned_increment.synchronize();
    const auto shared_result = backend.copy_to_host(shared_buffer);
    for (const auto byte : shared_result) {
      if (byte != static_cast<std::uint8_t>(shared_fill_value + 1U)) {
        expect_true(false, "shared-module kernels execute after module handle reset");
        break;
      }
    }

    const auto load_module_without_backend_owner = [] {
      const auto short_lived_backend = CudaBackend::create();
      return short_lived_backend.load_module_from_ptx(kFillBytesPtx);
    };
    auto independently_owned_module = load_module_without_backend_owner();
    auto independently_owned_kernel = independently_owned_module.load_kernel("increment_bytes");
    independently_owned_module.reset();
    independently_owned_kernel.launch({.grid = {.x = 9}, .block = {.x = 32}}, increment_args);
    independently_owned_kernel.synchronize();
    const auto independently_owned_result = backend.copy_to_host(shared_buffer);
    for (const auto byte : independently_owned_result) {
      if (byte != static_cast<std::uint8_t>(shared_fill_value + 2U)) {
        expect_true(false, "shared module retains CUDA backend lifetime");
        break;
      }
    }

    const auto kernel = backend.load_kernel_from_ptx(kFillBytesPtx, "fill_bytes");
    expect_true(static_cast<bool>(kernel), "loaded CUDA kernel");
    auto kernel_buffer = backend.allocate(257);
    backend.memset_d8(kernel_buffer, 0);
    CudaDevicePtr kernel_ptr = kernel_buffer.ptr();
    std::uint32_t fill_value = 0x5C;
    std::uint32_t fill_count = static_cast<std::uint32_t>(kernel_buffer.size());
    void* kernel_args[] = {&kernel_ptr, &fill_value, &fill_count};
    expect_invalid_argument(
        [&] {
          kernel.launch({.grid = {.x = 0, .y = 1, .z = 1}, .block = {.x = 32, .y = 1, .z = 1}},
                        kernel_args);
        },
        "zero kernel grid validation");
    expect_invalid_argument(
        [&] {
          kernel.launch({.grid = {.x = 9, .y = 1, .z = 1}, .block = {.x = 0, .y = 1, .z = 1}},
                        kernel_args);
        },
        "zero kernel block validation");
    kernel.launch({.grid = {.x = 9, .y = 1, .z = 1}, .block = {.x = 32, .y = 1, .z = 1}},
                  kernel_args);
    kernel.synchronize();
    const auto filled = backend.copy_to_host(kernel_buffer);
    for (const auto byte : filled) {
      if (byte != static_cast<std::uint8_t>(fill_value)) {
        expect_true(false, "kernel-filled byte pattern");
        break;
      }
    }

    constexpr std::size_t width = 17;
    constexpr std::size_t height = 5;
    constexpr std::size_t src_pitch = 23;
    constexpr std::size_t dst_pitch = 29;

    std::vector<std::uint8_t> host_src(src_pitch * height, 0x11);
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        host_src[row * src_pitch + col] = static_cast<std::uint8_t>(row * 31 + col);
      }
    }
    expect_invalid_argument(
        [&] {
          auto ignored = backend.allocate_pitched(0, height, 16);
          (void)ignored;
        },
        "pitched allocation zero width validation");
    expect_invalid_argument(
        [&] {
          auto ignored = backend.allocate_pitched(width, 0, 16);
          (void)ignored;
        },
        "pitched allocation zero height validation");
    expect_invalid_argument(
        [&] {
          auto ignored = backend.allocate_pitched(width, height, 1);
          (void)ignored;
        },
        "pitched allocation element size validation");
    expect_overflow_error(
        [&] {
          auto ignored = backend.allocate_pitched(std::numeric_limits<std::size_t>::max(), 2, 16);
          (void)ignored;
        },
        "pitched allocation overflow validation");
    auto src_allocation = backend.allocate_pitched(width, height, 16);
    auto dst_allocation = backend.allocate_pitched(width, height, 16);
    auto& src_device = src_allocation.buffer;
    auto& dst_device = dst_allocation.buffer;
    const auto device_src_pitch = src_allocation.pitch;
    const auto device_dst_pitch = dst_allocation.pitch;
    expect_true(device_src_pitch >= width, "pitched source covers requested width");
    expect_true(device_dst_pitch >= width, "pitched destination covers requested width");
    expect_eq(src_device.size(), device_src_pitch * height, "pitched source allocation size");
    expect_eq(dst_device.size(), device_dst_pitch * height, "pitched destination allocation size");
    backend.memset_d8(src_device, 0);
    backend.memset_d8(dst_device, 0xEE);
    expect_invalid_argument(
        [&] {
          backend.copy_host_to_device_2d({.src = nullptr,
                                          .src_pitch = src_pitch,
                                          .dst = src_device.ptr(),
                                          .dst_pitch = device_src_pitch,
                                          .width_bytes = width,
                                          .height = height});
        },
        "2D HtoD null source validation");
    expect_invalid_argument(
        [&] {
          backend.copy_device_to_device_2d({.src = 0,
                                            .src_pitch = device_src_pitch,
                                            .dst = dst_device.ptr(),
                                            .dst_pitch = device_dst_pitch,
                                            .width_bytes = width,
                                            .height = height});
        },
        "2D DtoD null source validation");
    expect_invalid_argument(
        [&] {
          backend.copy_device_to_host_2d({.dst = nullptr,
                                          .dst_pitch = dst_pitch,
                                          .src = dst_device.ptr(),
                                          .src_pitch = device_dst_pitch,
                                          .width_bytes = width,
                                          .height = height});
        },
        "2D DtoH null destination validation");
    expect_invalid_argument(
        [&] {
          backend.copy_device_to_device_2d({.src = src_device.ptr(),
                                            .src_pitch = device_src_pitch,
                                            .dst = dst_device.ptr(),
                                            .dst_pitch = device_dst_pitch,
                                            .width_bytes = 0,
                                            .height = height});
        },
        "2D zero width validation");
    expect_invalid_argument(
        [&] {
          backend.copy_device_to_device_2d({.src = src_device.ptr(),
                                            .src_pitch = device_src_pitch,
                                            .dst = dst_device.ptr(),
                                            .dst_pitch = device_dst_pitch,
                                            .width_bytes = width,
                                            .height = 0});
        },
        "2D zero height validation");
    expect_invalid_argument(
        [&] {
          backend.copy_device_to_device_2d({.src = src_device.ptr(),
                                            .src_pitch = width - 1,
                                            .dst = dst_device.ptr(),
                                            .dst_pitch = device_dst_pitch,
                                            .width_bytes = width,
                                            .height = height});
        },
        "2D source pitch validation");
    expect_invalid_argument(
        [&] {
          backend.copy_device_to_device_2d({.src = src_device.ptr(),
                                            .src_pitch = device_src_pitch,
                                            .dst = dst_device.ptr(),
                                            .dst_pitch = width - 1,
                                            .width_bytes = width,
                                            .height = height});
        },
        "2D destination pitch validation");
    backend.copy_host_to_device_2d({.src = host_src.data(),
                                    .src_pitch = src_pitch,
                                    .dst = src_device.ptr(),
                                    .dst_pitch = device_src_pitch,
                                    .width_bytes = width,
                                    .height = height});
    backend.copy_device_to_device_2d({.src = src_device.ptr(),
                                      .src_pitch = device_src_pitch,
                                      .dst = dst_device.ptr(),
                                      .dst_pitch = device_dst_pitch,
                                      .width_bytes = width,
                                      .height = height});
    observed_backend.copy_device_to_device_2d({.src = src_device.ptr(),
                                               .src_pitch = device_src_pitch,
                                               .dst = dst_device.ptr(),
                                               .dst_pitch = device_dst_pitch,
                                               .width_bytes = width,
                                               .height = height});
    expect_eq(trace->copies, 1U, "successful CUDA D2D submission observed");
    std::vector<std::uint8_t> host_dst(dst_pitch * height, 0xCD);
    observed_backend.copy_device_to_host_2d({.dst = host_dst.data(),
                                             .dst_pitch = dst_pitch,
                                             .src = dst_device.ptr(),
                                             .src_pitch = device_dst_pitch,
                                             .width_bytes = width,
                                             .height = height});
    expect_eq(trace->host_copies, 2U, "2D CUDA D2H submission observed");
    expect_eq(trace->host_copy_bytes, 4096U + width * height,
              "2D CUDA D2H payload byte count observed");
    backend.synchronize();
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        expect_eq(host_dst[row * dst_pitch + col], host_src[row * src_pitch + col],
                  "2D copied payload byte");
      }
      for (std::size_t col = width; col < dst_pitch; ++col) {
        expect_eq(host_dst[row * dst_pitch + col], static_cast<std::uint8_t>(0xCD),
                  "2D destination padding untouched");
      }
    }

    expect_invalid_argument(
        [&] {
          auto ignored = backend.allocate_shared_memory(0);
          (void)ignored;
        },
        "zero shared allocation validation");
    auto shared_memory = backend.allocate_shared_memory(12345);
    expect_true(static_cast<bool>(shared_memory), "shared memory allocation");
    expect_true(shared_memory.ptr() != 0, "shared memory device pointer");
    expect_true(shared_memory.size() >= 12345, "shared memory size rounded up");
    expect_true(valid_shareable_handle(shared_memory.shareable_handle()),
                "shared memory OS handle");
    auto released_memory = backend.allocate_shared_memory(4096);
    const CudaShareableHandle released_handle = released_memory.release_shareable_handle();
    expect_true(valid_shareable_handle(released_handle), "released shared memory OS handle");
    expect_true(!valid_shareable_handle(released_memory.shareable_handle()),
                "released shared memory no longer owns handle");
    close_released_handle(released_handle);
    released_memory.reset();
    constexpr std::size_t shared_width = 19;
    std::vector<std::uint8_t> shared_src(shared_width);
    for (std::size_t i = 0; i < shared_src.size(); ++i) {
      shared_src[i] = static_cast<std::uint8_t>(0x40 + i);
    }
    backend.copy_host_to_device_2d({.src = shared_src.data(),
                                    .src_pitch = shared_width,
                                    .dst = shared_memory.ptr(),
                                    .dst_pitch = shared_width,
                                    .width_bytes = shared_width,
                                    .height = 1});
    std::vector<std::uint8_t> shared_dst(shared_width, 0);
    backend.copy_device_to_host_2d({.dst = shared_dst.data(),
                                    .dst_pitch = shared_width,
                                    .src = shared_memory.ptr(),
                                    .src_pitch = shared_width,
                                    .width_bytes = shared_width,
                                    .height = 1});
    backend.synchronize();
    for (std::size_t i = 0; i < shared_width; ++i) {
      expect_eq(shared_dst[i], shared_src[i], "shared memory copied byte");
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    ++failures;
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
