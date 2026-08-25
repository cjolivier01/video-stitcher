#pragma once

#include "reco/core/cuda_backend.hpp"
#include "reco/core/pipeline_event.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace reco::detect {

using reco::core::CameraId;

struct Detection {
  CameraId camera = CameraId::Left;
  std::uint16_t class_id = 0;
  float confidence = 0.0F;
  float center_x = 0.0F;
  float center_y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
};

struct Yuv420pChroma {
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct Nv12Chroma {
  std::span<const std::uint8_t> uv;
};

using ChromaFormat = std::variant<Yuv420pChroma, Nv12Chroma>;

struct RawFrame {
  std::span<const std::uint8_t> y;
  ChromaFormat chroma;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct GpuNv12Frame {
  core::CudaDevicePtr y_ptr = 0;
  core::CudaDevicePtr uv_ptr = 0;
  std::size_t y_pitch = 0;
  std::size_t uv_pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  int rotation = 0;
  bool is_10bit = false;
};

struct PreprocessedChwFrame {
  std::span<const float> data;
  std::uint32_t input_size = 0;
  std::uint32_t src_width = 0;
  std::uint32_t src_height = 0;
};

struct RgbaFrame {
  std::span<const std::uint8_t> data;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct CudaRgbaFrame {
  core::CudaDevicePtr ptr = 0;
  std::size_t pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct CudaRgbaLetterboxedFrame {
  core::CudaDevicePtr ptr = 0;
  std::uint32_t src_width = 0;
  std::uint32_t src_height = 0;
};

struct MetalFrame {
  void* cv_pixel_buffer = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct WgpuNv12Frame {
  const void* y_view = nullptr;
  const void* uv_view = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  int rotation = 0;
};

// Non-owning detector input. Span and opaque pointer members must remain valid
// only for the duration of a UnifiedDetector::detect call; DetectorFrame is
// move-only to discourage queueing or storing borrowed frame views.
class DetectorFrame {
public:
  using Variant = std::variant<RawFrame, GpuNv12Frame, PreprocessedChwFrame, RgbaFrame,
                               CudaRgbaFrame, CudaRgbaLetterboxedFrame, MetalFrame, WgpuNv12Frame>;

  explicit DetectorFrame(Variant frame) : frame_(std::move(frame)) {}
  DetectorFrame(const DetectorFrame&) = delete;
  DetectorFrame& operator=(const DetectorFrame&) = delete;
  DetectorFrame(DetectorFrame&&) noexcept = default;
  DetectorFrame& operator=(DetectorFrame&&) noexcept = default;

  [[nodiscard]] const Variant& variant() const { return frame_; }
  [[nodiscard]] std::string_view variant_name() const;

private:
  Variant frame_;
};

enum class DetectorErrorKind {
  InferenceFailed,
  Timeout,
  UnsupportedFrameKind,
  Transport,
  Canceled,
};

class DetectorError : public std::runtime_error {
public:
  [[nodiscard]] DetectorErrorKind kind() const { return kind_; }
  [[nodiscard]] const std::string& detail() const { return detail_; }
  [[nodiscard]] std::optional<std::chrono::nanoseconds> after() const { return after_; }

  [[nodiscard]] static DetectorError inference_failed(std::string message);
  [[nodiscard]] static DetectorError timeout(std::chrono::nanoseconds after);
  [[nodiscard]] static DetectorError unsupported_frame_kind();
  [[nodiscard]] static DetectorError transport(std::string message);
  [[nodiscard]] static DetectorError canceled();

private:
  DetectorError(DetectorErrorKind kind, std::string message, std::string detail = {},
                std::optional<std::chrono::nanoseconds> after = std::nullopt);

  DetectorErrorKind kind_;
  std::string detail_;
  std::optional<std::chrono::nanoseconds> after_;
};

class UnifiedDetector {
public:
  virtual ~UnifiedDetector() = default;
  [[nodiscard]] virtual const char* name() const = 0;
  [[nodiscard]] virtual std::vector<Detection> detect(CameraId camera,
                                                      const DetectorFrame& frame) = 0;
  [[nodiscard]] virtual std::optional<std::span<const std::string>> class_names() const {
    return std::nullopt;
  }
};

[[nodiscard]] std::vector<Detection> postprocess(const std::vector<float>& data, std::size_t n,
                                                 CameraId camera, float confidence_threshold,
                                                 float scale, float pad_x, float pad_y,
                                                 std::uint32_t frame_width,
                                                 std::uint32_t frame_height);

[[nodiscard]] std::vector<Detection>
postprocess_balldet(const std::vector<float>& data, std::size_t n, CameraId camera,
                    float confidence_threshold, float scale, float pad_x, float pad_y,
                    std::uint32_t frame_width, std::uint32_t frame_height);

[[nodiscard]] float box_iou(const Detection& a, const Detection& b);
[[nodiscard]] std::vector<Detection> greedy_nms(std::vector<Detection> detections,
                                                float iou_threshold);
[[nodiscard]] std::vector<std::string> read_labels_file(const std::string& path);
[[nodiscard]] std::optional<std::vector<std::string>>
parse_names_dict_string(std::string_view names_str);

} // namespace reco::detect
