#pragma once

#include "reco/core/cuda_backend.hpp"
#include "reco/core/pipeline_event.hpp"
#include "reco/core/video_format.hpp"
#include "reco/detect/ncnn_session.hpp"
#include "reco/detect/ort_session.hpp"
#include "reco/detect/trt_engine.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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
  std::optional<core::YuvColorMatrix> color_matrix;
  std::optional<core::YuvColorRange> color_range;
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

class CpuYoloDetector final : public UnifiedDetector {
public:
  explicit CpuYoloDetector(std::filesystem::path model_path);
  CpuYoloDetector(std::filesystem::path model_path, float confidence_threshold,
                  std::vector<std::string> labels);

  [[nodiscard]] const char* name() const override;
  [[nodiscard]] std::vector<Detection> detect(CameraId camera, const DetectorFrame& frame) override;
  [[nodiscard]] std::optional<std::span<const std::string>> class_names() const override;
  [[nodiscard]] std::uint32_t input_size() const;

private:
  [[nodiscard]] std::vector<Detection> detect_raw(CameraId camera, const RawFrame& frame);
  [[nodiscard]] std::vector<Detection>
  detect_preprocessed(CameraId camera, std::span<const float> data, std::uint32_t input_size,
                      std::uint32_t src_width, std::uint32_t src_height);
  [[nodiscard]] std::tuple<float, float, float> preprocess(const RawFrame& frame);

  OrtSession session_;
  float confidence_threshold_ = 0.10F;
  std::vector<float> rgb_chw_buf_;
};

class OrtCudaYoloDetector final : public UnifiedDetector {
public:
  OrtCudaYoloDetector(std::filesystem::path model_path, std::uint32_t frame_width,
                      std::uint32_t frame_height, float confidence_threshold,
                      std::vector<std::string> labels, bool supports_p010);

  [[nodiscard]] const char* name() const override;
  [[nodiscard]] std::vector<Detection> detect(CameraId camera, const DetectorFrame& frame) override;
  [[nodiscard]] std::optional<std::span<const std::string>> class_names() const override;
  [[nodiscard]] std::uint32_t input_size() const;

private:
  [[nodiscard]] std::vector<Detection> detect_gpu_raw(CameraId camera, const GpuNv12Frame& frame);

  core::CudaBackend backend_;
  OrtSession session_;
  std::uint32_t frame_width_ = 0;
  std::uint32_t frame_height_ = 0;
  float confidence_threshold_ = 0.10F;
  float scale_ = 1.0F;
  float pad_x_ = 0.0F;
  float pad_y_ = 0.0F;
  core::CudaDeviceBuffer tensor_f32_;
  core::CudaDeviceBuffer nv12_8bit_y_;
  core::CudaDeviceBuffer nv12_8bit_uv_;
};

class TrtGpuDetector final : public UnifiedDetector {
public:
  TrtGpuDetector(std::filesystem::path engine_path, std::uint32_t frame_width,
                 std::uint32_t frame_height, float confidence_threshold,
                 std::vector<std::string> labels, bool supports_p010);

  [[nodiscard]] const char* name() const override;
  [[nodiscard]] std::vector<Detection> detect(CameraId camera, const DetectorFrame& frame) override;
  [[nodiscard]] std::optional<std::span<const std::string>> class_names() const override;
  [[nodiscard]] std::uint32_t input_size() const { return input_size_; }

private:
  [[nodiscard]] std::vector<Detection> detect_gpu_raw(CameraId camera, const GpuNv12Frame& frame);
  [[nodiscard]] std::vector<void*> build_binding_ptrs();

  core::CudaBackend backend_;
  std::optional<TrtEngine> engine_;
  std::optional<TrtContext> context_;
  std::uint32_t frame_width_ = 0;
  std::uint32_t frame_height_ = 0;
  std::uint32_t input_size_ = 0;
  float confidence_threshold_ = 0.10F;
  float scale_ = 1.0F;
  float pad_x_ = 0.0F;
  float pad_y_ = 0.0F;
  std::uint32_t new_w_ = 0;
  std::uint32_t new_h_ = 0;
  std::vector<std::string> labels_;
  std::size_t input_idx_ = 0;
  std::size_t output_idx_ = 0;
  std::size_t binding_count_ = 0;
  std::size_t output_floats_ = 0;
  core::CudaDeviceBuffer rgb_u8_;
  core::CudaDeviceBuffer rgb_scratch_;
  core::CudaDeviceBuffer resized_u8_;
  core::CudaDeviceBuffer tensor_f32_;
  core::CudaDeviceBuffer output_;
  core::CudaDeviceBuffer nv12_8bit_y_;
  core::CudaDeviceBuffer nv12_8bit_uv_;
};

class NcnnYoloDetector final : public UnifiedDetector {
public:
  NcnnYoloDetector(std::filesystem::path model_dir, std::uint32_t input_size,
                   std::uint32_t frame_width, std::uint32_t frame_height,
                   float confidence_threshold, std::vector<std::string> labels);

  [[nodiscard]] const char* name() const override;
  [[nodiscard]] std::vector<Detection> detect(CameraId camera, const DetectorFrame& frame) override;
  [[nodiscard]] std::optional<std::span<const std::string>> class_names() const override;
  [[nodiscard]] std::uint32_t input_size() const { return input_size_; }

private:
  [[nodiscard]] std::vector<Detection> detect_preprocessed(CameraId camera,
                                                           const PreprocessedChwFrame& frame);

  NcnnSession session_;
  std::uint32_t input_size_ = 0;
  std::uint32_t frame_width_ = 0;
  std::uint32_t frame_height_ = 0;
  float confidence_threshold_ = 0.10F;
  float nms_threshold_ = 0.45F;
  float scale_ = 1.0F;
  float pad_x_ = 0.0F;
  float pad_y_ = 0.0F;
  std::vector<std::string> labels_;
};

[[nodiscard]] std::vector<Detection> postprocess(const std::vector<float>& data, std::size_t n,
                                                 CameraId camera, float confidence_threshold,
                                                 float scale, float pad_x, float pad_y,
                                                 std::uint32_t frame_width,
                                                 std::uint32_t frame_height);

[[nodiscard]] std::vector<Detection>
postprocess_yolo_transposed(const std::vector<float>& data, std::size_t num_proposals,
                            std::size_t num_features, CameraId camera, float confidence_threshold,
                            float scale, float pad_x, float pad_y, std::uint32_t frame_width,
                            std::uint32_t frame_height, float nms_threshold);

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
