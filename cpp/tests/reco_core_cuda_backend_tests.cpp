#include "reco/core/cuda_backend.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

using namespace reco::core;

namespace {

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
    const auto host = backend.copy_to_host(buffer);
    expect_eq(host.size(), 4096U, "host copy size");
    for (const auto byte : host) {
      if (byte != 0xA5) {
        expect_true(false, "device memset byte pattern");
        break;
      }
    }

    constexpr std::size_t width = 17;
    constexpr std::size_t height = 5;
    constexpr std::size_t src_pitch = 23;
    constexpr std::size_t device_src_pitch = 32;
    constexpr std::size_t device_dst_pitch = 40;
    constexpr std::size_t dst_pitch = 29;

    std::vector<std::uint8_t> host_src(src_pitch * height, 0x11);
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        host_src[row * src_pitch + col] = static_cast<std::uint8_t>(row * 31 + col);
      }
    }
    auto src_device = backend.allocate(device_src_pitch * height);
    auto dst_device = backend.allocate(device_dst_pitch * height);
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
    std::vector<std::uint8_t> host_dst(dst_pitch * height, 0xCD);
    backend.copy_device_to_host_2d({.dst = host_dst.data(),
                                    .dst_pitch = dst_pitch,
                                    .src = dst_device.ptr(),
                                    .src_pitch = device_dst_pitch,
                                    .width_bytes = width,
                                    .height = height});
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
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    ++failures;
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
