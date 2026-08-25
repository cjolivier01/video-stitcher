#include "reco/core/cuda_backend.hpp"
#include "reco/detect/cuda_preprocess.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace reco::core;
using namespace reco::detect;

namespace {

int failures = 0;

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
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

void upload_bytes(CudaBackend& backend, const std::vector<std::uint8_t>& src,
                  const CudaDeviceBuffer& dst) {
  backend.copy_host_to_device_2d({.src = src.data(),
                                  .src_pitch = src.size(),
                                  .dst = dst.ptr(),
                                  .dst_pitch = src.size(),
                                  .width_bytes = src.size(),
                                  .height = 1});
}

std::vector<float> copy_floats(CudaBackend& backend, const CudaDeviceBuffer& buffer,
                               std::size_t count) {
  const auto bytes = backend.copy_to_host(buffer);
  std::vector<float> out(count);
  std::memcpy(out.data(), bytes.data(), count * sizeof(float));
  return out;
}

} // namespace

int main() {
  if (address_sanitizer_build() && !require_cuda()) {
    std::cerr << "SKIP: CUDA detector preprocess tests are skipped under ASan unless required\n";
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
    auto backend = CudaBackend::create();

    expect_invalid_argument([&] { p010_to_nv12(backend, 0, 1, 1); }, "p010 null source");
    expect_invalid_argument([&] { p010_to_nv12(backend, 1, 0, 1); }, "p010 null destination");
    expect_invalid_argument([&] { p010_to_nv12(backend, 1, 1, 0); }, "p010 zero samples");
    expect_invalid_argument([&] { p010_plane_to_nv12(backend, 1, 5, 2, 3, 1); },
                            "p010 pitched source pitch");
    expect_invalid_argument([&] { normalize_hwc_to_chw(backend, 1, 2, 0, 1); },
                            "normalize zero width");
    expect_invalid_argument([&] { normalize_rgba_to_chw(backend, 1, 2, 1, 0); },
                            "normalize zero height");

    const std::vector<std::uint16_t> p010_samples{0x0000, 0x0100, 0x8000, 0xAB00, 0xFFFF};
    std::vector<std::uint8_t> p010_bytes(p010_samples.size() * sizeof(std::uint16_t));
    std::memcpy(p010_bytes.data(), p010_samples.data(), p010_bytes.size());
    auto p010_src = backend.allocate(p010_bytes.size());
    auto p010_dst = backend.allocate(p010_samples.size());
    upload_bytes(backend, p010_bytes, p010_src);
    p010_to_nv12(backend, p010_src.ptr(), p010_dst.ptr(),
                 static_cast<std::uint32_t>(p010_samples.size()));
    const auto nv12 = backend.copy_to_host(p010_dst);
    expect_eq(nv12[0], 0x00U, "p010 sample 0");
    expect_eq(nv12[1], 0x01U, "p010 sample 1");
    expect_eq(nv12[2], 0x80U, "p010 sample 2");
    expect_eq(nv12[3], 0xABU, "p010 sample 3");
    expect_eq(nv12[4], 0xFFU, "p010 sample 4");
    auto p010_plane_dst = backend.allocate(p010_samples.size());
    p010_plane_to_nv12(backend, p010_src.ptr(), p010_bytes.size(), p010_plane_dst.ptr(),
                       static_cast<std::uint32_t>(p010_samples.size()), 1);
    const auto contiguous_plane = backend.copy_to_host(p010_plane_dst);
    for (std::size_t i = 0; i < contiguous_plane.size(); ++i) {
      expect_eq(contiguous_plane[i], nv12[i], "contiguous p010 plane fast path");
    }

    constexpr std::uint32_t pitched_width = 3;
    constexpr std::uint32_t pitched_height = 2;
    constexpr std::size_t src_pitch = 8;
    std::vector<std::uint8_t> pitched(src_pitch * pitched_height, 0xCD);
    const std::uint16_t pitched_values[] = {0x1000, 0x2200, 0x3300, 0x4400, 0x5500, 0x6600};
    for (std::uint32_t row = 0; row < pitched_height; ++row) {
      std::memcpy(pitched.data() + row * src_pitch, pitched_values + row * pitched_width,
                  pitched_width * sizeof(std::uint16_t));
    }
    auto pitched_src = backend.allocate(pitched.size());
    auto pitched_dst = backend.allocate(pitched_width * pitched_height);
    upload_bytes(backend, pitched, pitched_src);
    p010_plane_to_nv12(backend, pitched_src.ptr(), src_pitch, pitched_dst.ptr(), pitched_width,
                       pitched_height);
    const auto pitched_out = backend.copy_to_host(pitched_dst);
    const std::uint8_t expected_pitched[] = {0x10, 0x22, 0x33, 0x44, 0x55, 0x66};
    for (std::size_t i = 0; i < pitched_out.size(); ++i) {
      expect_eq(pitched_out[i], expected_pitched[i], "pitched p010 sample");
    }

    constexpr std::uint32_t width = 2;
    constexpr std::uint32_t height = 2;
    const std::vector<std::uint8_t> rgb{
        255, 0, 0,
        0, 128, 0,
        0, 0, 64,
        25, 50, 75,
    };
    auto rgb_src = backend.allocate(rgb.size());
    auto rgb_dst = backend.allocate(3 * width * height * sizeof(float));
    upload_bytes(backend, rgb, rgb_src);
    normalize_hwc_to_chw(backend, rgb_src.ptr(), rgb_dst.ptr(), width, height);
    const auto rgb_chw = copy_floats(backend, rgb_dst, 3 * width * height);
    expect_near(rgb_chw[0], 1.0F, 1.0e-6F, "rgb r0");
    expect_near(rgb_chw[1], 0.0F, 1.0e-6F, "rgb r1");
    expect_near(rgb_chw[2], 0.0F, 1.0e-6F, "rgb r2");
    expect_near(rgb_chw[3], 25.0F / 255.0F, 1.0e-6F, "rgb r3");
    expect_near(rgb_chw[4], 0.0F, 1.0e-6F, "rgb g0");
    expect_near(rgb_chw[5], 128.0F / 255.0F, 1.0e-6F, "rgb g1");
    expect_near(rgb_chw[8], 0.0F, 1.0e-6F, "rgb b0");
    expect_near(rgb_chw[10], 64.0F / 255.0F, 1.0e-6F, "rgb b2");
    expect_near(rgb_chw[11], 75.0F / 255.0F, 1.0e-6F, "rgb b3");

    const std::vector<std::uint8_t> rgba{
        255, 0, 0, 7,
        0, 128, 0, 8,
        0, 0, 64, 9,
        25, 50, 75, 10,
    };
    auto rgba_src = backend.allocate(rgba.size());
    auto rgba_dst = backend.allocate(3 * width * height * sizeof(float));
    upload_bytes(backend, rgba, rgba_src);
    normalize_rgba_to_chw(backend, rgba_src.ptr(), rgba_dst.ptr(), width, height);
    const auto rgba_chw = copy_floats(backend, rgba_dst, 3 * width * height);
    expect_near(rgba_chw[0], 1.0F, 1.0e-6F, "rgba r0");
    expect_near(rgba_chw[5], 128.0F / 255.0F, 1.0e-6F, "rgba g1");
    expect_near(rgba_chw[10], 64.0F / 255.0F, 1.0e-6F, "rgba b2");
    expect_near(rgba_chw[11], 75.0F / 255.0F, 1.0e-6F, "rgba b3");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    ++failures;
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
