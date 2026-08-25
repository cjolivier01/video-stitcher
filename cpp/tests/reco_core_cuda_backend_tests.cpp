#include "reco/core/cuda_backend.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
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
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    ++failures;
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
