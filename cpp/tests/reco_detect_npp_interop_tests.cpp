#include "reco/core/cuda_backend.hpp"
#include "reco/detect/npp_interop.hpp"

#include <array>
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

bool require_npp() {
  const char* value = std::getenv("RECO_REQUIRE_NPP_TEST");
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

void expect_bytes(CudaBackend& backend, const CudaDeviceBuffer& buffer,
                  const std::vector<std::uint8_t>& expected, std::string_view label) {
  const auto actual = backend.copy_to_host(buffer);
  expect_eq(actual.size(), expected.size(), label);
  const auto n = std::min(actual.size(), expected.size());
  for (std::size_t i = 0; i < n; ++i) {
    expect_eq(actual[i], expected[i], label);
  }
}

} // namespace

int main() {
  expect_invalid_argument([] { npp_nv12_to_rgb(0, 2, 1, 2, 3, 2, 2); }, "nv12 null y");
  expect_invalid_argument([] { npp_nv12_to_rgb(1, 2, 0, 2, 3, 2, 2); }, "nv12 null uv");
  expect_invalid_argument([] { npp_nv12_to_rgb(1, 2, 3, 2, 0, 2, 2); }, "nv12 null dst");
  expect_invalid_argument([] { npp_nv12_to_rgb(1, 2, 3, 2, 4, 3, 2); }, "nv12 odd width");
  expect_invalid_argument([] { npp_nv12_to_rgb(1, 1, 3, 2, 4, 2, 2); }, "nv12 narrow y pitch");
  expect_invalid_argument([] { npp_nv12_to_rgb(1, 8, 3, 4, 4, 4, 4); }, "nv12 mismatched pitch");
  expect_invalid_argument([] { npp_resize_c3(0, 2, 2, 1, 2, 2, {0, 0, 2, 2}); },
                          "resize c3 null src");
  expect_invalid_argument([] { npp_resize_c1(0, 2, 2, 2, 1, 2, 2, 2); }, "resize c1 null src");
  expect_invalid_argument([] { npp_resize_c1(1, 1, 2, 2, 3, 2, 2, 2); },
                          "resize c1 narrow source pitch");
  expect_invalid_argument([] { npp_resize_c3(1, 2, 2, 0, 2, 2, {0, 0, 2, 2}); },
                          "resize c3 null dst");
  expect_invalid_argument([] { npp_resize_c4(1, 2, 2, 3, 2, 2, {1, 0, 2, 2}); },
                          "resize c4 roi outside");
  expect_invalid_argument([] { npp_mirror_c3(1, 2, 0, 2); }, "mirror zero width");

  if (address_sanitizer_build() && !require_cuda()) {
    std::cerr << "SKIP: NPP interop tests are skipped under ASan unless required\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (!CudaBackend::is_available()) {
    const auto error = CudaBackend::availability_error();
    if (require_cuda() || require_npp()) {
      std::cerr << "FAIL: CUDA unavailable: " << error << '\n';
      return EXIT_FAILURE;
    }
    std::cerr << "SKIP: CUDA unavailable: " << error << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (!is_npp_available()) {
    const auto error = npp_availability_error();
    if (require_npp()) {
      std::cerr << "FAIL: NPP unavailable: " << error << '\n';
      return EXIT_FAILURE;
    }
    std::cerr << "SKIP: NPP unavailable: " << error << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  try {
    auto backend = CudaBackend::create();

    constexpr std::uint32_t width = 2;
    constexpr std::uint32_t height = 2;
    const std::vector<std::uint8_t> rgb{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    auto rgb_src = backend.allocate(rgb.size());
    auto rgb_dst = backend.allocate(rgb.size());
    upload_bytes(backend, rgb, rgb_src);
    npp_resize_c3(rgb_src.ptr(), width, height, rgb_dst.ptr(), width, height, {0, 0, 2, 2});
    backend.synchronize();
    expect_bytes(backend, rgb_dst, rgb, "resize c3 identity");

    const std::vector<std::uint8_t> gray{1, 2, 3, 4};
    auto gray_src = backend.allocate(gray.size());
    auto gray_dst = backend.allocate(gray.size());
    upload_bytes(backend, gray, gray_src);
    npp_resize_c1(gray_src.ptr(), width, width, height, gray_dst.ptr(), width, width, height);
    backend.synchronize();
    expect_bytes(backend, gray_dst, gray, "resize c1 identity");

    const std::vector<std::uint8_t> gray_large{
        10, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40,
    };
    auto gray_large_src = backend.allocate(gray_large.size());
    auto gray_small_dst = backend.allocate(4);
    upload_bytes(backend, gray_large, gray_large_src);
    npp_resize_c1(gray_large_src.ptr(), 4, 4, 4, gray_small_dst.ptr(), 2, 2, 2);
    backend.synchronize();
    expect_bytes(backend, gray_small_dst, {10, 20, 30, 40}, "resize c1 area downscale");

    const auto expect_mixed_axis_resize = [&](std::uint32_t src_width, std::uint32_t src_height,
                                              std::uint32_t dst_width, std::uint32_t dst_height,
                                              std::uint8_t value, std::string_view label) {
      const std::vector<std::uint8_t> src(src_width * src_height, value);
      auto src_buffer = backend.allocate(src.size());
      auto dst_buffer = backend.allocate(dst_width * dst_height);
      upload_bytes(backend, src, src_buffer);
      npp_resize_c1(src_buffer.ptr(), src_width, src_width, src_height, dst_buffer.ptr(), dst_width,
                    dst_width, dst_height);
      backend.synchronize();
      expect_bytes(backend, dst_buffer, std::vector<std::uint8_t>(dst_width * dst_height, value),
                   label);
    };
    expect_mixed_axis_resize(4, 2, 2, 4, 17, "resize c1 width shrink height grow");
    expect_mixed_axis_resize(2, 4, 4, 2, 31, "resize c1 width grow height shrink");
    expect_mixed_axis_resize(4, 2, 2, 2, 47, "resize c1 width shrink height unchanged");

    auto mirror_dst = backend.allocate(rgb.size());
    npp_mirror_c3(rgb_src.ptr(), mirror_dst.ptr(), width, height);
    backend.synchronize();
    expect_bytes(backend, mirror_dst,
                 {
                     10,
                     11,
                     12,
                     7,
                     8,
                     9,
                     4,
                     5,
                     6,
                     1,
                     2,
                     3,
                 },
                 "mirror c3 both axes");

    const std::vector<std::uint8_t> rgba{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    };
    auto rgba_src = backend.allocate(rgba.size());
    auto rgba_dst = backend.allocate(rgba.size());
    upload_bytes(backend, rgba, rgba_src);
    npp_resize_c4(rgba_src.ptr(), width, height, rgba_dst.ptr(), width, height, {0, 0, 2, 2});
    backend.synchronize();
    expect_bytes(backend, rgba_dst, rgba, "resize c4 identity");

    constexpr std::uint32_t nv12_width = 4;
    constexpr std::uint32_t nv12_height = 4;
    const std::vector<std::uint8_t> y_plane{
        16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136,
    };
    const std::vector<std::uint8_t> uv_plane{
        128, 128, 128, 128, 128, 128, 128, 128,
    };
    auto y_buffer = backend.allocate(y_plane.size());
    auto uv_buffer = backend.allocate(uv_plane.size());
    auto nv12_rgb = backend.allocate(nv12_width * nv12_height * 3);
    upload_bytes(backend, y_plane, y_buffer);
    upload_bytes(backend, uv_plane, uv_buffer);
    npp_nv12_to_rgb(y_buffer.ptr(), nv12_width, uv_buffer.ptr(), nv12_width, nv12_rgb.ptr(),
                    nv12_width, nv12_height);
    backend.synchronize();
    expect_bytes(backend, nv12_rgb,
                 {
                     16, 16,  16,  24,  24,  24,  32,  32,  32,  40,  40,  40,  48,  48,  48,  56,
                     56, 56,  64,  64,  64,  72,  72,  72,  80,  80,  80,  88,  88,  88,  96,  96,
                     96, 104, 104, 104, 112, 112, 112, 120, 120, 120, 128, 128, 128, 136, 136, 136,
                 },
                 "nv12 neutral rgb golden");

    const std::vector<std::uint8_t> limited_black_y(nv12_width * nv12_height, 16);
    upload_bytes(backend, limited_black_y, y_buffer);
    npp_nv12_to_rgb(y_buffer.ptr(), nv12_width, uv_buffer.ptr(), nv12_width, nv12_rgb.ptr(),
                    nv12_width, nv12_height, YuvColorMatrix::Bt709, YuvColorRange::Limited);
    backend.synchronize();
    expect_bytes(backend, nv12_rgb, std::vector<std::uint8_t>(nv12_width * nv12_height * 3, 0),
                 "limited-range NV12 black level");

    const std::vector<std::uint8_t> color_y(nv12_width * nv12_height, 120);
    const std::vector<std::uint8_t> color_uv{
        140, 150, 140, 150, 140, 150, 140, 150,
    };
    upload_bytes(backend, color_y, y_buffer);
    upload_bytes(backend, color_uv, uv_buffer);
    npp_nv12_to_rgb(y_buffer.ptr(), nv12_width, uv_buffer.ptr(), nv12_width, nv12_rgb.ptr(),
                    nv12_width, nv12_height);
    backend.synchronize();
    std::vector<std::uint8_t> legacy_expected;
    legacy_expected.reserve(nv12_width * nv12_height * 3);
    for (std::size_t pixel = 0; pixel < nv12_width * nv12_height; ++pixel) {
      legacy_expected.insert(legacy_expected.end(), {145, 102, 144});
    }
    expect_bytes(backend, nv12_rgb, legacy_expected, "legacy NV12 conversion");

    struct ColorGolden {
      YuvColorMatrix matrix;
      YuvColorRange range;
      std::array<std::uint8_t, 3> rgb;
      const char* label;
    };
    const std::array color_goldens{
        ColorGolden{YuvColorMatrix::Bt601, YuvColorRange::Full, {151, 100, 141}, "BT.601 full"},
        ColorGolden{
            YuvColorMatrix::Bt601, YuvColorRange::Limited, {156, 99, 145}, "BT.601 limited"},
        ColorGolden{YuvColorMatrix::Bt709, YuvColorRange::Full, {155, 107, 142}, "BT.709 full"},
        ColorGolden{
            YuvColorMatrix::Bt709, YuvColorRange::Limited, {161, 107, 146}, "BT.709 limited"},
        ColorGolden{YuvColorMatrix::Bt2020, YuvColorRange::Full, {152, 105, 143}, "BT.2020 full"},
        ColorGolden{
            YuvColorMatrix::Bt2020, YuvColorRange::Limited, {158, 105, 147}, "BT.2020 limited"},
    };
    for (const auto& golden : color_goldens) {
      npp_nv12_to_rgb(y_buffer.ptr(), nv12_width, uv_buffer.ptr(), nv12_width, nv12_rgb.ptr(),
                      nv12_width, nv12_height, golden.matrix, golden.range);
      backend.synchronize();
      std::vector<std::uint8_t> expected;
      expected.reserve(nv12_width * nv12_height * 3);
      for (std::size_t pixel = 0; pixel < nv12_width * nv12_height; ++pixel) {
        expected.insert(expected.end(), golden.rgb.begin(), golden.rgb.end());
      }
      expect_bytes(backend, nv12_rgb, expected, golden.label);
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    ++failures;
  }

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
