#pragma once

#include <stdexcept>

namespace reco::core {

/// YCbCr coefficient matrix carried by an NV12 video frame.
enum class YuvColorMatrix {
  Bt601,
  Bt709,
  Bt2020,
};

/// Encoded luma/chroma range carried by an NV12 video frame.
enum class YuvColorRange {
  Limited,
  Full,
};

/// Coefficients for converting 8-bit YCbCr samples to 8-bit RGB values.
struct YuvToRgbCoefficients {
  float y_offset = 0.0F;
  float y_scale = 1.0F;
  float red_from_v = 0.0F;
  float green_from_u = 0.0F;
  float green_from_v = 0.0F;
  float blue_from_u = 0.0F;
};

/// Returns the RGB conversion coefficients for a matrix/range pair.
[[nodiscard]] inline YuvToRgbCoefficients yuv_to_rgb_coefficients(YuvColorMatrix matrix,
                                                                  YuvColorRange range) {
  float kr = 0.0F;
  float kb = 0.0F;
  switch (matrix) {
  case YuvColorMatrix::Bt601:
    kr = 0.299F;
    kb = 0.114F;
    break;
  case YuvColorMatrix::Bt709:
    kr = 0.2126F;
    kb = 0.0722F;
    break;
  case YuvColorMatrix::Bt2020:
    kr = 0.2627F;
    kb = 0.0593F;
    break;
  default:
    throw std::invalid_argument("unsupported YUV color matrix");
  }

  const float kg = 1.0F - kr - kb;
  const float y_scale = range == YuvColorRange::Limited ? 255.0F / 219.0F : 1.0F;
  const float chroma_scale = range == YuvColorRange::Limited ? 255.0F / 224.0F : 1.0F;
  if (range != YuvColorRange::Limited && range != YuvColorRange::Full) {
    throw std::invalid_argument("unsupported YUV color range");
  }
  return {
      .y_offset = range == YuvColorRange::Limited ? -16.0F : 0.0F,
      .y_scale = y_scale,
      .red_from_v = (2.0F - 2.0F * kr) * chroma_scale,
      .green_from_u = -(kb * (2.0F - 2.0F * kb) / kg) * chroma_scale,
      .green_from_v = -(kr * (2.0F - 2.0F * kr) / kg) * chroma_scale,
      .blue_from_u = (2.0F - 2.0F * kb) * chroma_scale,
  };
}

} // namespace reco::core
