#include "reco/calibrate/gpu_undistort.hpp"
#include "reco/core/cuda_backend.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace reco::calibrate;
using namespace reco::core;

namespace {

int failures = 0;

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_u8_near(std::uint8_t actual, std::uint8_t expected, int tolerance,
                    std::string_view message) {
  const int delta = std::abs(static_cast<int>(actual) - static_cast<int>(expected));
  if (delta > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << static_cast<int>(expected)
              << " actual=" << static_cast<int>(actual) << '\n';
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

CameraParams flat_camera(std::uint32_t width, std::uint32_t height) {
  return {
      .width = width,
      .height = height,
      .fx = static_cast<double>(width),
      .fy = static_cast<double>(height),
      .cx = (static_cast<double>(width) - 1.0) * 0.5,
      .cy = (static_cast<double>(height) - 1.0) * 0.5,
  };
}

void upload_y_plane(CudaBackend& backend, const std::vector<std::uint8_t>& src,
                    std::size_t src_pitch, const CudaDeviceBuffer& dst, std::size_t dst_pitch,
                    std::uint32_t width, std::uint32_t height) {
  backend.copy_host_to_device_2d({.src = src.data(),
                                  .src_pitch = src_pitch,
                                  .dst = dst.ptr(),
                                  .dst_pitch = dst_pitch,
                                  .width_bytes = width,
                                  .height = height});
}

std::vector<std::uint8_t> download_y_plane(CudaBackend& backend, const CudaDeviceBuffer& src,
                                           std::size_t src_pitch, std::uint32_t width,
                                           std::uint32_t height) {
  std::vector<std::uint8_t> out(width * height);
  backend.copy_device_to_host_2d({.dst = out.data(),
                                  .dst_pitch = width,
                                  .src = src.ptr(),
                                  .src_pitch = src_pitch,
                                  .width_bytes = width,
                                  .height = height});
  return out;
}

double kb4_forward_scale(double r, const std::array<double, 4>& d) {
  if (r < 1.0e-10) {
    return 1.0;
  }
  const double theta = std::atan(r);
  const double theta2 = theta * theta;
  const double theta4 = theta2 * theta2;
  const double theta6 = theta4 * theta2;
  const double theta8 = theta4 * theta4;
  const double theta_d =
      theta * (1.0 + d[0] * theta2 + d[1] * theta4 + d[2] * theta6 + d[3] * theta8);
  return theta_d / r;
}

std::uint8_t bilinear_sample(const std::vector<std::uint8_t>& data, std::size_t pitch,
                             std::uint32_t width, std::uint32_t height, double x, double y) {
  if (x < 0.0 || y < 0.0 || x >= static_cast<double>(width - 1) ||
      y >= static_cast<double>(height - 1)) {
    return 0;
  }
  const auto x0 = static_cast<std::uint32_t>(x);
  const auto y0 = static_cast<std::uint32_t>(y);
  const double tx = x - static_cast<double>(x0);
  const double ty = y - static_cast<double>(y0);
  const double p00 = data[y0 * pitch + x0];
  const double p10 = data[y0 * pitch + x0 + 1];
  const double p01 = data[(y0 + 1) * pitch + x0];
  const double p11 = data[(y0 + 1) * pitch + x0 + 1];
  const double value =
      p00 * (1.0 - tx) * (1.0 - ty) + p10 * tx * (1.0 - ty) + p01 * (1.0 - tx) * ty + p11 * tx * ty;
  return static_cast<std::uint8_t>(std::round(value));
}

std::vector<std::uint8_t> reference_undistort(const std::vector<std::uint8_t>& input,
                                              std::size_t input_pitch, std::uint32_t width,
                                              std::uint32_t height, const CameraParams& camera) {
  const double src_scale_x = static_cast<double>(width) / static_cast<double>(camera.width);
  const double src_scale_y = static_cast<double>(height) / static_cast<double>(camera.height);
  const double src_fx = camera.fx * src_scale_x;
  const double src_fy = camera.fy * src_scale_y;
  const double src_cx = camera.cx * src_scale_x;
  const double src_cy = camera.cy * src_scale_y;
  const double out_fx = src_fx / 2.0;
  const double out_fy = src_fy / 2.0;
  const double out_cx = (static_cast<double>(width) + 2.0 * src_cx) / 4.0;
  const double out_cy = (static_cast<double>(height) + 2.0 * src_cy) / 4.0;
  std::vector<std::uint8_t> out(width * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const double nx = (static_cast<double>(x) - out_cx) / out_fx;
      const double ny = (static_cast<double>(y) - out_cy) / out_fy;
      const double scale = kb4_forward_scale(std::sqrt(nx * nx + ny * ny), camera.d);
      out[y * width + x] =
          bilinear_sample(input, input_pitch, width, height, src_fx * nx * scale + src_cx,
                          src_fy * ny * scale + src_cy);
    }
  }
  return out;
}

} // namespace

int main() {
  if (address_sanitizer_build() && !require_cuda()) {
    std::cerr << "SKIP: CUDA calibration undistort tests are skipped under ASan unless required\n";
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

    expect_invalid_argument(
        [&] {
          GpuCalibrationUndistorter(
              backend, {.camera = flat_camera(0, 4), .output_width = 4, .output_height = 4});
        },
        "zero camera width");
    expect_invalid_argument(
        [&] {
          auto camera = flat_camera(4, 4);
          camera.fx = 0.0;
          GpuCalibrationUndistorter(backend,
                                    {.camera = camera, .output_width = 4, .output_height = 4});
        },
        "zero focal length");
    expect_invalid_argument(
        [&] {
          GpuCalibrationUndistorter(
              backend, {.camera = flat_camera(4, 4), .output_width = 0, .output_height = 4});
        },
        "zero output width");

    constexpr std::uint32_t width = 7;
    constexpr std::uint32_t height = 5;
    constexpr std::size_t src_pitch = 16;
    constexpr std::size_t dst_pitch = 12;
    std::vector<std::uint8_t> input(src_pitch * height, 0xCD);
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        input[y * src_pitch + x] = static_cast<std::uint8_t>(10U + y * width + x);
      }
    }

    auto src = backend.allocate(src_pitch * height);
    auto dst = backend.allocate(dst_pitch * height);
    upload_y_plane(backend, input, src_pitch, src, src_pitch, width, height);

    GpuCalibrationUndistorter undistorter(
        backend,
        {.camera = flat_camera(width, height), .output_width = width, .output_height = height});
    expect_invalid_argument(
        [&] {
          undistorter.undistort_y(
              {.ptr = 0, .pitch = src_pitch, .width = width, .height = height},
              {.ptr = dst.ptr(), .pitch = dst_pitch, .width = width, .height = height});
        },
        "null source");
    expect_invalid_argument(
        [&] {
          undistorter.undistort_y(
              {.ptr = src.ptr(), .pitch = width - 1, .width = width, .height = height},
              {.ptr = dst.ptr(), .pitch = dst_pitch, .width = width, .height = height});
        },
        "narrow source pitch");
    expect_invalid_argument(
        [&] {
          undistorter.undistort_y(
              {.ptr = src.ptr(), .pitch = src_pitch, .width = width, .height = height},
              {.ptr = dst.ptr(), .pitch = dst_pitch, .width = width - 1, .height = height});
        },
        "destination dimension mismatch");
    expect_invalid_argument(
        [&] {
          undistorter.undistort_y(
              {.ptr = src.ptr(), .pitch = src_pitch, .width = width, .height = height},
              {.ptr = src.ptr(), .pitch = src_pitch, .width = width, .height = height});
        },
        "in-place undistort");

    undistorter.undistort_y(
        {.ptr = src.ptr(), .pitch = src_pitch, .width = width, .height = height},
        {.ptr = dst.ptr(), .pitch = dst_pitch, .width = width, .height = height});
    backend.synchronize();
    const auto copied = download_y_plane(backend, dst, dst_pitch, width, height);
    const auto expected =
        reference_undistort(input, src_pitch, width, height, undistorter.config().camera);
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        expect_u8_near(copied[y * width + x], expected[y * width + x], 1,
                       "zero-coefficient undistort pixel");
      }
    }

    auto scaled_profile = flat_camera(width, height);
    scaled_profile.width = width * 2;
    scaled_profile.height = height * 2;
    scaled_profile.fx *= 2.0;
    scaled_profile.fy *= 2.0;
    scaled_profile.cx = (static_cast<double>(scaled_profile.width) - 1.0) * 0.5;
    scaled_profile.cy = (static_cast<double>(scaled_profile.height) - 1.0) * 0.5;
    auto scaled_dst = backend.allocate(dst_pitch * height);
    GpuCalibrationUndistorter scaled(
        backend, {.camera = scaled_profile, .output_width = width, .output_height = height});
    scaled.undistort_y(
        {.ptr = src.ptr(), .pitch = src_pitch, .width = width, .height = height},
        {.ptr = scaled_dst.ptr(), .pitch = dst_pitch, .width = width, .height = height});
    backend.synchronize();
    const auto scaled_pixels = download_y_plane(backend, scaled_dst, dst_pitch, width, height);
    const auto expected_scaled =
        reference_undistort(input, src_pitch, width, height, scaled_profile);
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        const std::size_t index = y * width + x;
        expect_u8_near(scaled_pixels[index], expected_scaled[index], 1,
                       "scaled-profile undistort pixel");
      }
    }

    auto warped_dst = backend.allocate(dst_pitch * height);
    auto camera = flat_camera(width, height);
    camera.fx = static_cast<double>(width);
    camera.fy = static_cast<double>(height);
    camera.d = {0.4, 0.0, 0.0, 0.0};
    GpuCalibrationUndistorter warped(
        backend, {.camera = camera, .output_width = width, .output_height = height});
    warped.undistort_y(
        {.ptr = src.ptr(), .pitch = src_pitch, .width = width, .height = height},
        {.ptr = warped_dst.ptr(), .pitch = dst_pitch, .width = width, .height = height});
    backend.synchronize();
    const auto warped_pixels = download_y_plane(backend, warped_dst, dst_pitch, width, height);
    const auto expected_warped = reference_undistort(input, src_pitch, width, height, camera);
    bool changed = false;
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        const std::size_t index = y * width + x;
        expect_u8_near(warped_pixels[index], expected_warped[index], 1,
                       "nonzero-distortion undistort pixel");
        changed = changed || warped_pixels[index] != copied[index];
      }
    }
    if (!changed) {
      std::cerr << "FAIL: nonzero distortion should alter at least one sample\n";
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  if (failures != 0) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
