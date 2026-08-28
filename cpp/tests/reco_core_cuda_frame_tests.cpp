#include "reco/core/cuda_frame.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>

using namespace reco::core;

namespace {

int failures = 0;
constexpr CudaContextId kContextA = 0xCAFE;
constexpr CudaContextId kContextB = 0xBEEF;

template <typename Actual, typename Expected>
void expect_eq(const Actual& actual, const Expected& expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_invalid_argument(Function&& function, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function> void expect_overflow(Function&& function, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::overflow_error&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

void pitched_plane_validation() {
  const CudaPitchedPlaneView plane(0x1000, 512, 128, 96, 4, kContextA, 2);
  expect_eq(plane.ptr(), CudaDevicePtr{0x1000}, "plane pointer");
  expect_eq(plane.accessible_bytes(), std::size_t{512}, "plane capacity");
  expect_eq(plane.pitch_bytes(), std::size_t{128}, "plane pitch");
  expect_eq(plane.row_bytes(), std::size_t{96}, "plane row bytes");
  expect_eq(plane.rows(), std::uint32_t{4}, "plane row count");
  expect_eq(plane.context_id(), kContextA, "plane context identity");
  expect_eq(plane.device_ordinal(), 2, "plane device ordinal");
  expect_eq(plane.address_span_bytes(), std::size_t{480}, "plane address span");

  const CudaPitchedPlaneView exact_capacity(0x2000, 480, 128, 96, 4, kContextA);
  expect_eq(exact_capacity.accessible_bytes(), exact_capacity.address_span_bytes(),
            "plane accepts exact capacity");

  expect_invalid_argument([] { (void)CudaPitchedPlaneView(0, 16, 16, 16, 1, kContextA); },
                          "plane rejects null pointer");
  expect_invalid_argument([] { (void)CudaPitchedPlaneView(1, 16, 16, 0, 1, kContextA); },
                          "plane rejects zero row size");
  expect_invalid_argument([] { (void)CudaPitchedPlaneView(1, 16, 16, 16, 0, kContextA); },
                          "plane rejects zero rows");
  expect_invalid_argument([] { (void)CudaPitchedPlaneView(1, 16, 15, 16, 1, kContextA); },
                          "plane rejects short pitch");
  expect_invalid_argument([] { (void)CudaPitchedPlaneView(1, 15, 16, 16, 1, kContextA); },
                          "plane rejects insufficient capacity");
  expect_invalid_argument([] { (void)CudaPitchedPlaneView(1, 16, 16, 16, 1, 0); },
                          "plane rejects missing context identity");
  expect_invalid_argument([] { (void)CudaPitchedPlaneView(1, 16, 16, 16, 1, kContextA, -1); },
                          "plane rejects negative device ordinal");
  expect_overflow(
      [] {
        (void)CudaPitchedPlaneView(1, std::numeric_limits<std::size_t>::max(),
                                   std::numeric_limits<std::size_t>::max(), 1, 3, kContextA);
      },
      "plane rejects size span overflow");
  expect_overflow(
      [] {
        (void)CudaPitchedPlaneView(std::numeric_limits<CudaDevicePtr>::max() - 3, 8, 8, 8, 1,
                                   kContextA);
      },
      "plane rejects pointer span overflow");
  expect_overflow(
      [] {
        (void)CudaPitchedPlaneView(std::numeric_limits<CudaDevicePtr>::max() - 7, 16, 4, 4, 1,
                                   kContextA);
      },
      "plane rejects capacity address overflow independently of visible span");
}

void nv12_frame_validation() {
  const CudaPitchedPlaneView y_plane(0x10000, 2048U * 1080U, 2048, 1920, 1080, kContextA, 1);
  const CudaPitchedPlaneView uv_plane(0x400000, 2112U * 540U, 2112, 1920, 540, kContextA, 1);
  const CudaNv12FrameView frame(y_plane, uv_plane, 1920, 1080, YuvColorMatrix::Bt709,
                                YuvColorRange::Limited);
  expect_eq(frame.width(), std::uint32_t{1920}, "NV12 width");
  expect_eq(frame.height(), std::uint32_t{1080}, "NV12 height");
  expect_eq(frame.y_plane().row_bytes(), std::size_t{1920}, "NV12 Y row size");
  expect_eq(frame.y_plane().rows(), std::uint32_t{1080}, "NV12 Y rows");
  expect_eq(frame.uv_plane().row_bytes(), std::size_t{1920}, "NV12 UV row size");
  expect_eq(frame.uv_plane().rows(), std::uint32_t{540}, "NV12 UV rows");
  expect_eq(frame.uv_plane().pitch_bytes(), std::size_t{2112}, "NV12 independent UV pitch");
  expect_eq(frame.color_matrix(), YuvColorMatrix::Bt709, "NV12 color matrix");
  expect_eq(frame.color_range(), YuvColorRange::Limited, "NV12 color range");
  expect_eq(frame.context_id(), kContextA, "NV12 context identity");
  expect_eq(frame.device_ordinal(), 1, "NV12 device ordinal");

  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 4, 2, kContextA),
                                CudaPitchedPlaneView(0x2000, 4, 4, 4, 1, kContextA), 3, 2,
                                YuvColorMatrix::Bt601, YuvColorRange::Full);
      },
      "NV12 rejects odd width");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 12, 4, 4, 3, kContextA),
                                CudaPitchedPlaneView(0x2000, 4, 4, 4, 1, kContextA), 4, 3,
                                YuvColorMatrix::Bt601, YuvColorRange::Full);
      },
      "NV12 rejects odd height");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 2, 2, kContextA),
                                CudaPitchedPlaneView(0x2000, 4, 4, 4, 1, kContextA), 4, 2,
                                YuvColorMatrix::Bt601, YuvColorRange::Full);
      },
      "NV12 rejects mismatched Y shape");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 4, 2, kContextA),
                                CudaPitchedPlaneView(0x2000, 2, 2, 2, 1, kContextA), 4, 2,
                                YuvColorMatrix::Bt601, YuvColorRange::Full);
      },
      "NV12 rejects mismatched UV shape");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 4, 2, kContextA, 0),
                                CudaPitchedPlaneView(0x2000, 4, 4, 4, 1, kContextA, 1), 4, 2,
                                YuvColorMatrix::Bt601, YuvColorRange::Full);
      },
      "NV12 rejects mismatched device ordinals");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 4, 2, kContextA),
                                CudaPitchedPlaneView(0x2000, 4, 4, 4, 1, kContextB), 4, 2,
                                YuvColorMatrix::Bt601, YuvColorRange::Full);
      },
      "NV12 rejects mismatched contexts");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 4, 2, kContextA),
                                CudaPitchedPlaneView(0x1002, 4, 4, 4, 1, kContextA), 4, 2,
                                YuvColorMatrix::Bt601, YuvColorRange::Full);
      },
      "NV12 rejects overlapping planes");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 4, 2, kContextA),
                                CudaPitchedPlaneView(0x2000, 4, 4, 4, 1, kContextA), 4, 2,
                                static_cast<YuvColorMatrix>(99), YuvColorRange::Full);
      },
      "NV12 rejects invalid color matrix");
  expect_invalid_argument(
      [] {
        (void)CudaNv12FrameView(CudaPitchedPlaneView(0x1000, 8, 4, 4, 2, kContextA),
                                CudaPitchedPlaneView(0x2000, 4, 4, 4, 1, kContextA), 4, 2,
                                YuvColorMatrix::Bt2020, static_cast<YuvColorRange>(99));
      },
      "NV12 rejects invalid color range");
}

void rgba_frame_validation() {
  const CudaRgbaFrameView frame(
      CudaPitchedPlaneView(0x30000, 4096U * 480U, 4096, 2560, 480, kContextA, 3), 640, 480);
  expect_eq(frame.width(), std::uint32_t{640}, "RGBA width");
  expect_eq(frame.height(), std::uint32_t{480}, "RGBA height");
  expect_eq(frame.plane().row_bytes(), std::size_t{2560}, "RGBA row size");
  expect_eq(frame.plane().pitch_bytes(), std::size_t{4096}, "RGBA pitch");
  expect_eq(frame.context_id(), kContextA, "RGBA context identity");
  expect_eq(frame.device_ordinal(), 3, "RGBA device ordinal");

  expect_invalid_argument(
      [] { (void)CudaRgbaFrameView(CudaPitchedPlaneView(1, 1, 1, 1, 1, kContextA), 0, 1); },
      "RGBA rejects zero width");
  expect_invalid_argument(
      [] { (void)CudaRgbaFrameView(CudaPitchedPlaneView(1, 1, 1, 1, 1, kContextA), 1, 0); },
      "RGBA rejects zero height");
  expect_invalid_argument(
      [] { (void)CudaRgbaFrameView(CudaPitchedPlaneView(1, 15, 15, 15, 1, kContextA), 4, 1); },
      "RGBA rejects mismatched row size");
  expect_invalid_argument(
      [] { (void)CudaRgbaFrameView(CudaPitchedPlaneView(1, 32, 16, 16, 2, kContextA), 4, 1); },
      "RGBA rejects mismatched row count");
}

} // namespace

int main() {
  static_assert(std::is_copy_constructible_v<CudaPitchedPlaneView>);
  static_assert(std::is_copy_constructible_v<CudaNv12FrameView>);
  static_assert(std::is_copy_constructible_v<CudaRgbaFrameView>);

  pitched_plane_validation();
  nv12_frame_validation();
  rgba_frame_validation();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
