#include "reco/detect/detectors.hpp"

#include "reco/detect/cuda_preprocess.hpp"
#include "reco/detect/npp_interop.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

namespace reco::detect {
namespace {

float clamp01(float value) { return std::clamp(value, 0.0F, 1.0F); }

constexpr std::size_t kMaxClassCount = 10000;
constexpr float kLetterboxGrey = 114.0F / 255.0F;

std::string_view trim_ascii(std::string_view value) {
  constexpr std::string_view whitespace = " \t\r\n";
  const auto first = value.find_first_not_of(whitespace);
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(whitespace);
  return value.substr(first, last - first + 1);
}

std::string_view trim_matching(std::string_view value, char ch) {
  while (!value.empty() && value.front() == ch) {
    value.remove_prefix(1);
  }
  while (!value.empty() && value.back() == ch) {
    value.remove_suffix(1);
  }
  return value;
}

std::optional<std::size_t> parse_usize(std::string_view value) {
  value = trim_ascii(value);
  if (value.empty()) {
    return std::nullopt;
  }
  std::size_t parsed = 0;
  const char* begin = value.data();
  const char* end = value.data() + value.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end) {
    return std::nullopt;
  }
  return parsed;
}

std::size_t checked_plane_size(std::uint32_t width, std::uint32_t height,
                               std::string_view label) {
  const auto w = static_cast<std::size_t>(width);
  const auto h = static_cast<std::size_t>(height);
  if (w != 0 && h > std::numeric_limits<std::size_t>::max() / w) {
    throw DetectorError::inference_failed(std::string(label) + " dimensions overflow");
  }
  return w * h;
}

void validate_raw_frame(const RawFrame& frame) {
  if (frame.width == 0 || frame.height == 0) {
    throw DetectorError::inference_failed("RawFrame dimensions must be non-zero");
  }
  if ((frame.width % 2) != 0 || (frame.height % 2) != 0) {
    throw DetectorError::inference_failed("RawFrame dimensions must be even for 4:2:0 chroma");
  }
  const auto y_size = checked_plane_size(frame.width, frame.height, "RawFrame Y");
  if (frame.y.size() < y_size) {
    throw DetectorError::inference_failed("RawFrame Y plane is shorter than width*height");
  }
  const auto chroma_size = checked_plane_size(frame.width / 2, frame.height / 2, "RawFrame chroma");
  std::visit(
      [&](const auto& chroma) {
        using T = std::decay_t<decltype(chroma)>;
        if constexpr (std::is_same_v<T, Yuv420pChroma>) {
          if (chroma.u.size() < chroma_size || chroma.v.size() < chroma_size) {
            throw DetectorError::inference_failed("RawFrame YUV420p chroma plane is too short");
          }
        } else if constexpr (std::is_same_v<T, Nv12Chroma>) {
          if (chroma.uv.size() < chroma_size * 2) {
            throw DetectorError::inference_failed("RawFrame NV12 chroma plane is too short");
          }
        }
      },
      frame.chroma);
}

std::pair<float, float> chroma_sample(const RawFrame& frame, std::uint32_t x, std::uint32_t y) {
  const auto cx = static_cast<std::size_t>(x / 2);
  const auto cy = static_cast<std::size_t>(y / 2);
  const auto cw = static_cast<std::size_t>(frame.width / 2);
  return std::visit(
      [&](const auto& chroma) -> std::pair<float, float> {
        using T = std::decay_t<decltype(chroma)>;
        if constexpr (std::is_same_v<T, Yuv420pChroma>) {
          const auto idx = cy * cw + cx;
          return {static_cast<float>(chroma.u[idx]), static_cast<float>(chroma.v[idx])};
        } else {
          const auto idx = cy * static_cast<std::size_t>(frame.width) + cx * 2;
          return {static_cast<float>(chroma.uv[idx]), static_cast<float>(chroma.uv[idx + 1])};
        }
      },
      frame.chroma);
}

std::array<float, 3> sample_rgb(const RawFrame& frame, std::uint32_t x, std::uint32_t y) {
  const auto y_value = static_cast<float>(frame.y[static_cast<std::size_t>(y) * frame.width + x]);
  const auto [u_value, v_value] = chroma_sample(frame, x, y);
  const float r = std::clamp(y_value + 1.402F * (v_value - 128.0F), 0.0F, 255.0F);
  const float g =
      std::clamp(y_value - 0.344136F * (u_value - 128.0F) - 0.714136F * (v_value - 128.0F),
                 0.0F, 255.0F);
  const float b = std::clamp(y_value + 1.772F * (u_value - 128.0F), 0.0F, 255.0F);
  return {r, g, b};
}

std::size_t detection_count_from_output_shape(const OrtTensorOutput& output) {
  if (output.shape.size() < 2) {
    throw DetectorError::inference_failed("ORT detector output rank is less than 2");
  }
  if (output.shape[1] < 0) {
    throw DetectorError::inference_failed("ORT detector output count dimension is dynamic");
  }
  return static_cast<std::size_t>(output.shape[1]);
}

} // namespace

std::string_view DetectorFrame::variant_name() const {
  return std::visit(
      [](const auto& frame) -> std::string_view {
        using T = std::decay_t<decltype(frame)>;
        if constexpr (std::is_same_v<T, RawFrame>) {
          return "Cpu";
        } else if constexpr (std::is_same_v<T, GpuNv12Frame>) {
          return "Cuda";
        } else if constexpr (std::is_same_v<T, PreprocessedChwFrame>) {
          return "PreprocessedChw";
        } else if constexpr (std::is_same_v<T, RgbaFrame>) {
          return "Rgba";
        } else if constexpr (std::is_same_v<T, CudaRgbaFrame>) {
          return "CudaRgba";
        } else if constexpr (std::is_same_v<T, CudaRgbaLetterboxedFrame>) {
          return "CudaRgbaLetterboxed";
        } else if constexpr (std::is_same_v<T, MetalFrame>) {
          return "Metal";
        } else if constexpr (std::is_same_v<T, WgpuNv12Frame>) {
          return "WgpuNv12";
        }
      },
      frame_);
}

DetectorError::DetectorError(DetectorErrorKind kind, std::string message, std::string detail,
                             std::optional<std::chrono::nanoseconds> after)
    : std::runtime_error(std::move(message)), kind_(kind), detail_(std::move(detail)),
      after_(after) {}

DetectorError DetectorError::inference_failed(std::string message) {
  const auto detail = message;
  return DetectorError(DetectorErrorKind::InferenceFailed, "inference failed: " + message, detail);
}

DetectorError DetectorError::timeout(std::chrono::nanoseconds after) {
  if (after < std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("detector timeout duration must be non-negative");
  }
  std::ostringstream message;
  message << "detector timed out after " << after.count() << "ns";
  return DetectorError(DetectorErrorKind::Timeout, message.str(), {}, after);
}

DetectorError DetectorError::unsupported_frame_kind() {
  return DetectorError(DetectorErrorKind::UnsupportedFrameKind,
                       "detector does not support this frame variant");
}

DetectorError DetectorError::transport(std::string message) {
  const auto detail = message;
  return DetectorError(DetectorErrorKind::Transport, "transport error: " + message, detail);
}

DetectorError DetectorError::canceled() {
  return DetectorError(DetectorErrorKind::Canceled, "detection canceled");
}

CpuYoloDetector::CpuYoloDetector(std::filesystem::path model_path)
    : CpuYoloDetector(std::move(model_path), 0.10F, {}) {}

CpuYoloDetector::CpuYoloDetector(std::filesystem::path model_path, float confidence_threshold,
                                 std::vector<std::string> labels)
    : session_(OrtSessionConfig{
          .model_path = std::move(model_path),
          .fallback_labels = std::move(labels),
          .providers = {OrtExecutionProvider::Cpu},
      }),
      confidence_threshold_(confidence_threshold) {
  const auto sz = static_cast<std::size_t>(session_.metadata().input_size);
  if (sz == 0 || sz > std::numeric_limits<std::size_t>::max() / sz / 3) {
    throw DetectorError::inference_failed("CpuYoloDetector input dimensions overflow");
  }
  rgb_chw_buf_.assign(3 * sz * sz, kLetterboxGrey);
}

const char* CpuYoloDetector::name() const { return "ort-cpu"; }

std::optional<std::span<const std::string>> CpuYoloDetector::class_names() const {
  return std::span<const std::string>(session_.metadata().labels);
}

std::uint32_t CpuYoloDetector::input_size() const { return session_.metadata().input_size; }

std::vector<Detection> CpuYoloDetector::detect(CameraId camera, const DetectorFrame& frame) {
  const auto& variant = frame.variant();
  if (const auto* raw = std::get_if<RawFrame>(&variant); raw != nullptr) {
    return detect_raw(camera, *raw);
  }
  if (const auto* preprocessed = std::get_if<PreprocessedChwFrame>(&variant);
      preprocessed != nullptr) {
    return detect_preprocessed(camera, preprocessed->data, preprocessed->input_size,
                               preprocessed->src_width, preprocessed->src_height);
  }
  throw DetectorError::unsupported_frame_kind();
}

std::tuple<float, float, float> CpuYoloDetector::preprocess(const RawFrame& frame) {
  validate_raw_frame(frame);
  const auto input_size = session_.metadata().input_size;
  const float fw = static_cast<float>(frame.width);
  const float fh = static_cast<float>(frame.height);
  const float is = static_cast<float>(input_size);
  const float scale = std::min(is / fw, is / fh);
  const auto new_w = static_cast<std::uint32_t>(std::round(fw * scale));
  const auto new_h = static_cast<std::uint32_t>(std::round(fh * scale));
  const float pad_x = static_cast<float>(input_size - new_w) / 2.0F;
  const float pad_y = static_cast<float>(input_size - new_h) / 2.0F;
  const auto pad_x_i = static_cast<std::uint32_t>(pad_x);
  const auto pad_y_i = static_cast<std::uint32_t>(pad_y);
  const auto sz = static_cast<std::size_t>(input_size);
  const auto plane = sz * sz;

  rgb_chw_buf_.assign(rgb_chw_buf_.size(), kLetterboxGrey);
  const auto w_max = frame.width - 1;
  const auto h_max = frame.height - 1;

  for (std::uint32_t dy = 0; dy < new_h; ++dy) {
    for (std::uint32_t dx = 0; dx < new_w; ++dx) {
      const float src_x = static_cast<float>(dx) / scale;
      const float src_y = static_cast<float>(dy) / scale;
      const auto x0 = std::min(static_cast<std::uint32_t>(std::floor(src_x)), w_max);
      const auto y0 = std::min(static_cast<std::uint32_t>(std::floor(src_y)), h_max);
      const auto x1 = std::min(x0 + 1, w_max);
      const auto y1 = std::min(y0 + 1, h_max);
      const float fx = src_x - std::floor(src_x);
      const float fy = src_y - std::floor(src_y);

      const auto c00 = sample_rgb(frame, x0, y0);
      const auto c10 = sample_rgb(frame, x1, y0);
      const auto c01 = sample_rgb(frame, x0, y1);
      const auto c11 = sample_rgb(frame, x1, y1);
      const auto lerp = [&](std::size_t channel) {
        return c00[channel] * (1.0F - fx) * (1.0F - fy) +
               c10[channel] * fx * (1.0F - fy) + c01[channel] * (1.0F - fx) * fy +
               c11[channel] * fx * fy;
      };

      const auto ox = static_cast<std::size_t>(pad_x_i + dx);
      const auto oy = static_cast<std::size_t>(pad_y_i + dy);
      rgb_chw_buf_[oy * sz + ox] = lerp(0) / 255.0F;
      rgb_chw_buf_[plane + oy * sz + ox] = lerp(1) / 255.0F;
      rgb_chw_buf_[2 * plane + oy * sz + ox] = lerp(2) / 255.0F;
    }
  }
  return {scale, pad_x, pad_y};
}

std::vector<Detection> CpuYoloDetector::detect_raw(CameraId camera, const RawFrame& frame) {
  const auto [scale, pad_x, pad_y] = preprocess(frame);
  const auto input_size = session_.metadata().input_size;
  const std::vector<std::int64_t> shape{1, 3, static_cast<std::int64_t>(input_size),
                                        static_cast<std::int64_t>(input_size)};
  const auto outputs = session_.run_cpu_f32(rgb_chw_buf_, shape);
  if (outputs.empty()) {
    return {};
  }
  const std::size_t n = detection_count_from_output_shape(outputs[0]);
  const auto& output = outputs[0].data;
  return outputs.size() > 1
             ? postprocess_balldet(output, n, camera, confidence_threshold_, scale, pad_x, pad_y,
                                   frame.width, frame.height)
             : postprocess(output, n, camera, confidence_threshold_, scale, pad_x, pad_y,
                           frame.width, frame.height);
}

std::vector<Detection> CpuYoloDetector::detect_preprocessed(CameraId camera,
                                                            std::span<const float> data,
                                                            std::uint32_t input_size,
                                                            std::uint32_t src_width,
                                                            std::uint32_t src_height) {
  const auto sz = static_cast<std::size_t>(input_size);
  if (input_size == 0 || src_width == 0 || src_height == 0 ||
      sz > std::numeric_limits<std::size_t>::max() / sz / 3) {
    throw DetectorError::inference_failed("PreprocessedChw dimensions are invalid");
  }
  const auto expected = 3 * sz * sz;
  if (data.size() != expected) {
    throw DetectorError::inference_failed("PreprocessedChw input length does not match input_size");
  }
  const std::vector<std::int64_t> shape{1, 3, static_cast<std::int64_t>(input_size),
                                        static_cast<std::int64_t>(input_size)};
  const auto outputs = session_.run_cpu_f32(data, shape);
  if (outputs.empty()) {
    return {};
  }

  const float fw = static_cast<float>(src_width);
  const float fh = static_cast<float>(src_height);
  const float is = static_cast<float>(input_size);
  const float scale = std::min(is / fw, is / fh);
  const float pad_x = (is - std::round(fw * scale)) / 2.0F;
  const float pad_y = (is - std::round(fh * scale)) / 2.0F;
  const std::size_t n = detection_count_from_output_shape(outputs[0]);
  const auto& output = outputs[0].data;
  return outputs.size() > 1
             ? postprocess_balldet(output, n, camera, confidence_threshold_, scale, pad_x, pad_y,
                                   src_width, src_height)
             : postprocess(output, n, camera, confidence_threshold_, scale, pad_x, pad_y,
                           src_width, src_height);
}

OrtCudaYoloDetector::OrtCudaYoloDetector(std::filesystem::path model_path,
                                         std::uint32_t frame_width,
                                         std::uint32_t frame_height,
                                         float confidence_threshold,
                                         std::vector<std::string> labels, bool supports_p010)
    : backend_(core::CudaBackend::create()),
      session_(OrtSessionConfig{
          .model_path = std::move(model_path),
          .fallback_labels = std::move(labels),
          .providers = {OrtExecutionProvider::Cuda},
      }),
      frame_width_(frame_width),
      frame_height_(frame_height),
      confidence_threshold_(confidence_threshold) {
  if (frame_width_ == 0 || frame_height_ == 0) {
    throw DetectorError::inference_failed("OrtCudaYoloDetector frame dimensions must be non-zero");
  }
  const auto input_size = session_.metadata().input_size;
  if (input_size == 0) {
    throw DetectorError::inference_failed("OrtCudaYoloDetector model input size must be non-zero");
  }
  const float fw = static_cast<float>(frame_width_);
  const float fh = static_cast<float>(frame_height_);
  const float is = static_cast<float>(input_size);
  scale_ = std::min(is / fw, is / fh);
  pad_x_ = static_cast<float>(input_size - static_cast<std::uint32_t>(std::round(fw * scale_))) /
           2.0F;
  pad_y_ = static_cast<float>(input_size - static_cast<std::uint32_t>(std::round(fh * scale_))) /
           2.0F;

  const auto sz = static_cast<std::size_t>(input_size);
  if (sz > std::numeric_limits<std::size_t>::max() / sz / 3 / sizeof(float)) {
    throw DetectorError::inference_failed("OrtCudaYoloDetector tensor dimensions overflow");
  }
  tensor_f32_ = backend_.allocate(3 * sz * sz * sizeof(float));
  if (supports_p010) {
    const auto y_size = checked_plane_size(frame_width_, frame_height_, "P010 Y scratch");
    const auto uv_size = checked_plane_size(frame_width_, frame_height_ / 2, "P010 UV scratch");
    nv12_8bit_y_ = backend_.allocate(y_size);
    nv12_8bit_uv_ = backend_.allocate(uv_size);
  }
}

const char* OrtCudaYoloDetector::name() const { return "ort-cuda"; }

std::optional<std::span<const std::string>> OrtCudaYoloDetector::class_names() const {
  return std::span<const std::string>(session_.metadata().labels);
}

std::uint32_t OrtCudaYoloDetector::input_size() const { return session_.metadata().input_size; }

std::vector<Detection> OrtCudaYoloDetector::detect(CameraId camera, const DetectorFrame& frame) {
  const auto& variant = frame.variant();
  if (const auto* gpu = std::get_if<GpuNv12Frame>(&variant); gpu != nullptr) {
    return detect_gpu_raw(camera, *gpu);
  }
  throw DetectorError::unsupported_frame_kind();
}

std::vector<Detection> OrtCudaYoloDetector::detect_gpu_raw(CameraId camera,
                                                           const GpuNv12Frame& frame) {
  if (frame.y_ptr == 0 || frame.uv_ptr == 0) {
    throw DetectorError::inference_failed("GpuNv12Frame has null CUDA plane pointer");
  }
  if (frame.width != frame_width_ || frame.height != frame_height_) {
    throw DetectorError::inference_failed("GpuNv12Frame dimensions do not match detector");
  }
  if ((frame.width % 2) != 0 || (frame.height % 2) != 0) {
    throw DetectorError::inference_failed("GpuNv12Frame dimensions must be even for 4:2:0 chroma");
  }
  core::CudaDevicePtr nv12_y = frame.y_ptr;
  core::CudaDevicePtr nv12_uv = frame.uv_ptr;
  std::uint32_t nv12_y_pitch = frame.width;

  if (frame.is_10bit) {
    if (!nv12_8bit_y_ || !nv12_8bit_uv_) {
      throw DetectorError::inference_failed("P010 frame received but detector lacks P010 scratch");
    }
    const auto p010_min_pitch = static_cast<std::size_t>(frame.width) * 2U;
    if (frame.y_pitch < p010_min_pitch || frame.uv_pitch < p010_min_pitch) {
      throw DetectorError::inference_failed("GpuNv12Frame P010 pitch is smaller than width*2");
    }
    p010_plane_to_nv12(backend_, frame.y_ptr, frame.y_pitch, nv12_8bit_y_.ptr(), frame.width,
                       frame.height);
    p010_plane_to_nv12(backend_, frame.uv_ptr, frame.uv_pitch, nv12_8bit_uv_.ptr(), frame.width,
                       frame.height / 2);
    nv12_y = nv12_8bit_y_.ptr();
    nv12_uv = nv12_8bit_uv_.ptr();
    nv12_y_pitch = frame.width;
  } else if (frame.y_pitch != frame.uv_pitch) {
    throw DetectorError::inference_failed("GpuNv12Frame Y and UV pitches must match");
  } else {
    if (frame.y_pitch < frame.width) {
      throw DetectorError::inference_failed("GpuNv12Frame pitch is smaller than width");
    }
    nv12_y_pitch = static_cast<std::uint32_t>(frame.y_pitch);
    if (static_cast<std::size_t>(nv12_y_pitch) != frame.y_pitch) {
      throw DetectorError::inference_failed("GpuNv12Frame pitch exceeds CUDA kernel u32 limit");
    }
  }

  const auto input_size = session_.metadata().input_size;
  nv12_to_rgb_chw_fullrange(backend_, nv12_y, nv12_uv, tensor_f32_.ptr(), nv12_y_pitch,
                            frame.width, frame.height, input_size, input_size,
                            static_cast<std::uint32_t>(pad_x_), static_cast<std::uint32_t>(pad_y_),
                            scale_, frame.rotation);
  const std::vector<std::int64_t> shape{1, 3, static_cast<std::int64_t>(input_size),
                                        static_cast<std::int64_t>(input_size)};
  const auto outputs = session_.run_cuda_f32(tensor_f32_.ptr(), tensor_f32_.size(), shape);
  if (outputs.empty()) {
    return {};
  }
  const std::size_t n = detection_count_from_output_shape(outputs[0]);
  const auto& output = outputs[0].data;
  return postprocess(output, n, camera, confidence_threshold_, scale_, pad_x_, pad_y_, frame.width,
                     frame.height);
}

TrtGpuDetector::TrtGpuDetector(std::filesystem::path engine_path, std::uint32_t frame_width,
                               std::uint32_t frame_height, float confidence_threshold,
                               std::vector<std::string> labels, bool supports_p010)
    : backend_(core::CudaBackend::create()), labels_(std::move(labels)) {
  if (!is_npp_available()) {
    throw DetectorError::inference_failed("NPP is required for TensorRT detector preprocessing");
  }
  if (frame_width == 0 || frame_height == 0) {
    throw DetectorError::inference_failed("TrtGpuDetector frame dimensions must be non-zero");
  }
  backend_.ensure_primary_context(0);
  engine_.emplace(engine_path);
  frame_width_ = frame_width;
  frame_height_ = frame_height;
  confidence_threshold_ = confidence_threshold;

  const auto bindings = engine_->bindings();
  binding_count_ = bindings.size();
  const auto input_it = std::find_if(bindings.begin(), bindings.end(),
                                     [](const TrtBindingInfo& binding) { return binding.is_input; });
  const auto output_it = std::find_if(bindings.begin(), bindings.end(),
                                      [](const TrtBindingInfo& binding) { return !binding.is_input; });
  if (input_it == bindings.end() || output_it == bindings.end()) {
    throw DetectorError::inference_failed("TensorRT engine must have at least one input and output binding");
  }
  input_idx_ = static_cast<std::size_t>(std::distance(bindings.begin(), input_it));
  output_idx_ = static_cast<std::size_t>(std::distance(bindings.begin(), output_it));
  if (input_it->data_type != TrtDataType::Float || input_it->dims.size() != 4 ||
      input_it->dims[0] != 1 || input_it->dims[1] != 3 || input_it->dims[2] <= 0 ||
      input_it->dims[3] != input_it->dims[2]) {
    throw DetectorError::inference_failed("TensorRT input binding must be float32 [1,3,S,S]");
  }
  if (output_it->data_type != TrtDataType::Float || output_it->byte_size % sizeof(float) != 0) {
    throw DetectorError::inference_failed("TensorRT output binding must be float32");
  }
  input_size_ = static_cast<std::uint32_t>(input_it->dims[2]);
  output_floats_ = output_it->byte_size / sizeof(float);
  if (output_floats_ == 0 || (output_floats_ % 6) != 0) {
    throw DetectorError::inference_failed("TensorRT output binding must contain 6-float detections");
  }

  const float fw = static_cast<float>(frame_width_);
  const float fh = static_cast<float>(frame_height_);
  const float is = static_cast<float>(input_size_);
  scale_ = std::min(is / fw, is / fh);
  new_w_ = static_cast<std::uint32_t>(std::round(fw * scale_));
  new_h_ = static_cast<std::uint32_t>(std::round(fh * scale_));
  pad_x_ = static_cast<float>(input_size_ - new_w_) / 2.0F;
  pad_y_ = static_cast<float>(input_size_ - new_h_) / 2.0F;

  const auto frame_rgb_size = checked_plane_size(frame_width_, frame_height_, "TensorRT RGB");
  if (frame_rgb_size > std::numeric_limits<std::size_t>::max() / 3U) {
    throw DetectorError::inference_failed("TensorRT RGB dimensions overflow");
  }
  const auto resized_rgb_size = checked_plane_size(input_size_, input_size_, "TensorRT resized RGB");
  if (resized_rgb_size > std::numeric_limits<std::size_t>::max() / 3U) {
    throw DetectorError::inference_failed("TensorRT resized RGB dimensions overflow");
  }
  const auto tensor_plane_size = checked_plane_size(input_size_, input_size_, "TensorRT tensor");
  if (tensor_plane_size > std::numeric_limits<std::size_t>::max() / 3U / sizeof(float)) {
    throw DetectorError::inference_failed("TensorRT tensor dimensions overflow");
  }
  rgb_u8_ = backend_.allocate(frame_rgb_size * 3U);
  rgb_scratch_ = backend_.allocate(frame_rgb_size * 3U);
  resized_u8_ = backend_.allocate(resized_rgb_size * 3U);
  tensor_f32_ = backend_.allocate(tensor_plane_size * 3U * sizeof(float));
  output_ = backend_.allocate(output_it->byte_size);
  backend_.memset_d8(resized_u8_, 114);
  backend_.memset_d8(output_, 0);
  if (supports_p010) {
    nv12_8bit_y_ =
        backend_.allocate(checked_plane_size(frame_width_, frame_height_, "TensorRT P010 Y"));
    nv12_8bit_uv_ =
        backend_.allocate(checked_plane_size(frame_width_, frame_height_ / 2, "TensorRT P010 UV"));
  }
  context_.emplace(engine_->create_context());
}

const char* TrtGpuDetector::name() const { return "tensorrt-native"; }

std::optional<std::span<const std::string>> TrtGpuDetector::class_names() const {
  return std::span<const std::string>(labels_);
}

std::vector<Detection> TrtGpuDetector::detect(CameraId camera, const DetectorFrame& frame) {
  const auto& variant = frame.variant();
  if (const auto* gpu = std::get_if<GpuNv12Frame>(&variant); gpu != nullptr) {
    try {
      return detect_gpu_raw(camera, *gpu);
    } catch (const DetectorError&) {
      throw;
    } catch (const std::exception& error) {
      throw DetectorError::inference_failed(error.what());
    }
  }
  throw DetectorError::unsupported_frame_kind();
}

std::vector<void*> TrtGpuDetector::build_binding_ptrs() {
  std::vector<void*> bindings(binding_count_, nullptr);
  bindings[input_idx_] = reinterpret_cast<void*>(tensor_f32_.ptr());
  bindings[output_idx_] = reinterpret_cast<void*>(output_.ptr());
  return bindings;
}

std::vector<Detection> TrtGpuDetector::detect_gpu_raw(CameraId camera, const GpuNv12Frame& frame) {
  backend_.ensure_primary_context(0);
  if (frame.y_ptr == 0 || frame.uv_ptr == 0) {
    throw DetectorError::inference_failed("GpuNv12Frame has null CUDA plane pointer");
  }
  if (frame.width != frame_width_ || frame.height != frame_height_) {
    throw DetectorError::inference_failed("GpuNv12Frame dimensions do not match TensorRT detector");
  }
  if ((frame.width % 2) != 0 || (frame.height % 2) != 0) {
    throw DetectorError::inference_failed("GpuNv12Frame dimensions must be even for 4:2:0 chroma");
  }

  core::CudaDevicePtr nv12_y = frame.y_ptr;
  core::CudaDevicePtr nv12_uv = frame.uv_ptr;
  std::size_t nv12_y_pitch = frame.y_pitch;
  std::size_t nv12_uv_pitch = frame.uv_pitch;
  if (frame.is_10bit) {
    if (!nv12_8bit_y_ || !nv12_8bit_uv_) {
      throw DetectorError::inference_failed("P010 frame received but TensorRT detector lacks scratch");
    }
    const auto p010_min_pitch = static_cast<std::size_t>(frame.width) * 2U;
    if (frame.y_pitch < p010_min_pitch || frame.uv_pitch < p010_min_pitch) {
      throw DetectorError::inference_failed("GpuNv12Frame P010 pitch is smaller than width*2");
    }
    p010_plane_to_nv12(backend_, frame.y_ptr, frame.y_pitch, nv12_8bit_y_.ptr(), frame.width,
                       frame.height);
    p010_plane_to_nv12(backend_, frame.uv_ptr, frame.uv_pitch, nv12_8bit_uv_.ptr(), frame.width,
                       frame.height / 2);
    nv12_y = nv12_8bit_y_.ptr();
    nv12_uv = nv12_8bit_uv_.ptr();
    nv12_y_pitch = frame.width;
    nv12_uv_pitch = frame.width;
  } else {
    if (frame.y_pitch < frame.width || frame.uv_pitch < frame.width) {
      throw DetectorError::inference_failed("GpuNv12Frame pitch is smaller than width");
    }
    if (frame.y_pitch != frame.uv_pitch) {
      throw DetectorError::inference_failed("GpuNv12Frame Y and UV pitches must match");
    }
  }

  npp_nv12_to_rgb(nv12_y, nv12_y_pitch, nv12_uv, nv12_uv_pitch, rgb_u8_.ptr(), frame.width,
                  frame.height);
  core::CudaDevicePtr rgb_for_resize = rgb_u8_.ptr();
  if (frame.rotation == 180) {
    npp_mirror_c3(rgb_u8_.ptr(), rgb_scratch_.ptr(), frame.width, frame.height);
    rgb_for_resize = rgb_scratch_.ptr();
  }
  npp_resize_c3(rgb_for_resize, frame.width, frame.height, resized_u8_.ptr(), input_size_,
                input_size_, NppiRect{static_cast<int>(pad_x_), static_cast<int>(pad_y_),
                                      static_cast<int>(new_w_), static_cast<int>(new_h_)});
  backend_.synchronize();
  normalize_hwc_to_chw(backend_, resized_u8_.ptr(), tensor_f32_.ptr(), input_size_, input_size_);
  backend_.synchronize();

  auto bindings = build_binding_ptrs();
  context_->enqueue(bindings, nullptr);
  backend_.synchronize();

  const auto output_bytes = backend_.copy_to_host(output_);
  std::vector<float> output(output_floats_, 0.0F);
  std::memcpy(output.data(), output_bytes.data(), output.size() * sizeof(float));
  return postprocess(output, output_floats_ / 6, camera, confidence_threshold_, scale_, pad_x_,
                     pad_y_, frame.width, frame.height);
}

NcnnYoloDetector::NcnnYoloDetector(std::filesystem::path model_dir, std::uint32_t input_size,
                                   std::uint32_t frame_width, std::uint32_t frame_height,
                                   float confidence_threshold, std::vector<std::string> labels)
    : session_(NcnnSessionConfig{.model_dir = std::move(model_dir)}),
      input_size_(input_size),
      frame_width_(frame_width),
      frame_height_(frame_height),
      confidence_threshold_(confidence_threshold),
      labels_(std::move(labels)) {
  if (input_size_ == 0 || frame_width_ == 0 || frame_height_ == 0) {
    throw DetectorError::inference_failed("NcnnYoloDetector dimensions must be non-zero");
  }
  const float fw = static_cast<float>(frame_width_);
  const float fh = static_cast<float>(frame_height_);
  const float is = static_cast<float>(input_size_);
  scale_ = std::min(is / fw, is / fh);
  pad_x_ = static_cast<float>(input_size_ - static_cast<std::uint32_t>(std::round(fw * scale_))) /
           2.0F;
  pad_y_ = static_cast<float>(input_size_ - static_cast<std::uint32_t>(std::round(fh * scale_))) /
           2.0F;
}

const char* NcnnYoloDetector::name() const { return "ncnn"; }

std::optional<std::span<const std::string>> NcnnYoloDetector::class_names() const {
  return std::span<const std::string>(labels_);
}

std::vector<Detection> NcnnYoloDetector::detect(CameraId camera, const DetectorFrame& frame) {
  const auto& variant = frame.variant();
  if (const auto* preprocessed = std::get_if<PreprocessedChwFrame>(&variant);
      preprocessed != nullptr) {
    return detect_preprocessed(camera, *preprocessed);
  }
  throw DetectorError::unsupported_frame_kind();
}

std::vector<Detection> NcnnYoloDetector::detect_preprocessed(CameraId camera,
                                                             const PreprocessedChwFrame& frame) {
  if (frame.input_size != input_size_ || frame.src_width != frame_width_ ||
      frame.src_height != frame_height_) {
    throw DetectorError::inference_failed("PreprocessedChw dimensions do not match NCNN detector");
  }
  try {
    const auto output = session_.run_preprocessed_chw(frame.data, frame.input_size);
    if (output.width <= 0 || output.height <= 0) {
      return {};
    }
    return postprocess_yolo_transposed(output.data, static_cast<std::size_t>(output.width),
                                       static_cast<std::size_t>(output.height), camera,
                                       confidence_threshold_, scale_, pad_x_, pad_y_,
                                       frame.src_width, frame.src_height, nms_threshold_);
  } catch (const NcnnError& error) {
    throw DetectorError::inference_failed(error.what());
  }
}

std::vector<Detection> postprocess(const std::vector<float>& data, std::size_t n, CameraId camera,
                                   float confidence_threshold, float scale, float pad_x,
                                   float pad_y, std::uint32_t frame_width,
                                   std::uint32_t frame_height) {
  const std::size_t expected_len = n * 6;
  if (data.size() < expected_len) {
    return {};
  }

  const float fw = static_cast<float>(frame_width);
  const float fh = static_cast<float>(frame_height);
  std::vector<Detection> detections;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t base = i * 6;
    const float conf = data[base + 4];
    if (!std::isfinite(conf) || conf < confidence_threshold) {
      continue;
    }
    const float x1 = data[base];
    const float y1 = data[base + 1];
    const float x2 = data[base + 2];
    const float y2 = data[base + 3];
    if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
      continue;
    }
    const float class_id_f = data[base + 5];
    if (!std::isfinite(class_id_f) || class_id_f < 0.0F ||
        class_id_f > static_cast<float>(std::numeric_limits<std::uint16_t>::max())) {
      continue;
    }
    const auto class_id = static_cast<std::uint16_t>(class_id_f);

    const float orig_x1 = (x1 - pad_x) / scale;
    const float orig_y1 = (y1 - pad_y) / scale;
    const float orig_x2 = (x2 - pad_x) / scale;
    const float orig_y2 = (y2 - pad_y) / scale;
    const float cx = ((orig_x1 + orig_x2) / 2.0F) / fw;
    const float cy = ((orig_y1 + orig_y2) / 2.0F) / fh;
    const float w = std::abs(orig_x2 - orig_x1) / fw;
    const float h = std::abs(orig_y2 - orig_y1) / fh;
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h)) {
      continue;
    }
    if (cx < 0.0F || cx > 1.0F || cy < 0.0F || cy > 1.0F) {
      continue;
    }
    detections.push_back({.camera = camera,
                          .class_id = class_id,
                          .confidence = conf,
                          .center_x = clamp01(cx),
                          .center_y = clamp01(cy),
                          .width = clamp01(w),
                          .height = clamp01(h)});
  }
  return detections;
}

std::vector<Detection> postprocess_yolo_transposed(const std::vector<float>& data,
                                                   std::size_t num_proposals,
                                                   std::size_t num_features, CameraId camera,
                                                   float confidence_threshold, float scale,
                                                   float pad_x, float pad_y,
                                                   std::uint32_t frame_width,
                                                   std::uint32_t frame_height,
                                                   float nms_threshold) {
  if (num_proposals == 0 || num_features <= 4) {
    return {};
  }
  if (num_features > std::numeric_limits<std::size_t>::max() / num_proposals ||
      data.size() < num_features * num_proposals) {
    return {};
  }

  const auto num_classes = num_features - 4U;
  const float fw = static_cast<float>(frame_width);
  const float fh = static_cast<float>(frame_height);
  std::vector<Detection> candidates;
  for (std::size_t proposal = 0; proposal < num_proposals; ++proposal) {
    std::size_t best_class = 0;
    float best_score = 0.0F;
    for (std::size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
      const float score = data[(4U + class_idx) * num_proposals + proposal];
      if (std::isfinite(score) && score > best_score) {
        best_score = score;
        best_class = class_idx;
      }
    }
    if (best_score < confidence_threshold ||
        best_class > std::numeric_limits<std::uint16_t>::max()) {
      continue;
    }

    const float cx_p = data[proposal];
    const float cy_p = data[num_proposals + proposal];
    const float w_p = data[(2U * num_proposals) + proposal];
    const float h_p = data[(3U * num_proposals) + proposal];
    if (!std::isfinite(cx_p) || !std::isfinite(cy_p) || !std::isfinite(w_p) ||
        !std::isfinite(h_p)) {
      continue;
    }

    const float x1 = cx_p - w_p / 2.0F;
    const float y1 = cy_p - h_p / 2.0F;
    const float x2 = cx_p + w_p / 2.0F;
    const float y2 = cy_p + h_p / 2.0F;
    const float orig_x1 = (x1 - pad_x) / scale;
    const float orig_y1 = (y1 - pad_y) / scale;
    const float orig_x2 = (x2 - pad_x) / scale;
    const float orig_y2 = (y2 - pad_y) / scale;
    const float center_x = ((orig_x1 + orig_x2) / 2.0F) / fw;
    const float center_y = ((orig_y1 + orig_y2) / 2.0F) / fh;
    const float width = std::abs(orig_x2 - orig_x1) / fw;
    const float height = std::abs(orig_y2 - orig_y1) / fh;
    if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(width) ||
        !std::isfinite(height)) {
      continue;
    }
    if (center_x < 0.0F || center_x > 1.0F || center_y < 0.0F || center_y > 1.0F) {
      continue;
    }
    candidates.push_back({.camera = camera,
                          .class_id = static_cast<std::uint16_t>(best_class),
                          .confidence = best_score,
                          .center_x = clamp01(center_x),
                          .center_y = clamp01(center_y),
                          .width = clamp01(width),
                          .height = clamp01(height)});
  }

  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.confidence > b.confidence;
  });
  std::vector<Detection> keep;
  for (const auto& detection : candidates) {
    const auto same_class_overlap = [&](const auto& kept) {
      return kept.class_id == detection.class_id && box_iou(kept, detection) > nms_threshold;
    };
    if (std::none_of(keep.begin(), keep.end(), same_class_overlap)) {
      keep.push_back(detection);
    }
  }
  return keep;
}

std::vector<Detection> postprocess_balldet(const std::vector<float>& data, std::size_t n,
                                           CameraId camera, float confidence_threshold, float scale,
                                           float pad_x, float pad_y, std::uint32_t frame_width,
                                           std::uint32_t frame_height) {
  const std::size_t expected_len = n * 6;
  if (data.size() < expected_len) {
    return {};
  }

  const float fw = static_cast<float>(frame_width);
  const float fh = static_cast<float>(frame_height);
  std::vector<Detection> candidates;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t base = i * 6;
    const float conf = data[base + 5];
    if (!std::isfinite(conf) || conf < confidence_threshold) {
      continue;
    }
    const float cx_p = data[base];
    const float cy_p = data[base + 1];
    const float w_p = data[base + 2];
    const float h_p = data[base + 3];
    if (!std::isfinite(cx_p) || !std::isfinite(cy_p) || !std::isfinite(w_p) ||
        !std::isfinite(h_p)) {
      continue;
    }

    const float cx = ((cx_p - pad_x) / scale) / fw;
    const float cy = ((cy_p - pad_y) / scale) / fh;
    const float w = (w_p / scale) / fw;
    const float h = (h_p / scale) / fh;
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h)) {
      continue;
    }
    if (cx < 0.0F || cx > 1.0F || cy < 0.0F || cy > 1.0F) {
      continue;
    }
    candidates.push_back({.camera = camera,
                          .class_id = 0,
                          .confidence = conf,
                          .center_x = clamp01(cx),
                          .center_y = clamp01(cy),
                          .width = clamp01(w),
                          .height = clamp01(h)});
  }
  return greedy_nms(std::move(candidates), 0.45F);
}

float box_iou(const Detection& a, const Detection& b) {
  const float ax1 = a.center_x - a.width / 2.0F;
  const float ay1 = a.center_y - a.height / 2.0F;
  const float ax2 = a.center_x + a.width / 2.0F;
  const float ay2 = a.center_y + a.height / 2.0F;
  const float bx1 = b.center_x - b.width / 2.0F;
  const float by1 = b.center_y - b.height / 2.0F;
  const float bx2 = b.center_x + b.width / 2.0F;
  const float by2 = b.center_y + b.height / 2.0F;
  const float iw = std::max(std::min(ax2, bx2) - std::max(ax1, bx1), 0.0F);
  const float ih = std::max(std::min(ay2, by2) - std::max(ay1, by1), 0.0F);
  const float inter = iw * ih;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni <= 0.0F ? 0.0F : inter / uni;
}

std::vector<Detection> greedy_nms(std::vector<Detection> detections, float iou_threshold) {
  std::stable_sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
    if (std::isnan(a.confidence) || std::isnan(b.confidence)) {
      return false;
    }
    return a.confidence > b.confidence;
  });

  std::vector<Detection> keep;
  for (const auto& detection : detections) {
    if (std::all_of(keep.begin(), keep.end(),
                    [&](const auto& kept) { return box_iou(kept, detection) < iou_threshold; })) {
      keep.push_back(detection);
    }
  }
  return keep;
}

std::vector<std::string> read_labels_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    return {};
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file, line)) {
    const auto begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      continue;
    }
    const auto end = line.find_last_not_of(" \t\r\n");
    labels.push_back(line.substr(begin, end - begin + 1));
  }
  return labels;
}

std::optional<std::vector<std::string>> parse_names_dict_string(std::string_view names_str) {
  const auto trimmed = trim_ascii(names_str);
  if (trimmed.size() < 2 || trimmed.front() != '{' || trimmed.back() != '}') {
    return std::nullopt;
  }
  const auto inner = trimmed.substr(1, trimmed.size() - 2);
  if (inner.empty()) {
    return std::nullopt;
  }

  std::vector<std::pair<std::size_t, std::string>> labels;
  std::size_t start = 0;
  while (start <= inner.size()) {
    const auto comma = inner.find(',', start);
    const auto entry = trim_ascii(inner.substr(
        start, comma == std::string_view::npos ? std::string_view::npos : comma - start));
    const auto colon = entry.find(':');
    if (colon != std::string_view::npos) {
      if (const auto idx = parse_usize(entry.substr(0, colon)); idx.has_value()) {
        auto name = trim_ascii(entry.substr(colon + 1));
        name = trim_matching(name, '\'');
        name = trim_matching(name, '"');
        labels.emplace_back(*idx, std::string(name));
      }
    }
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }

  std::stable_sort(labels.begin(), labels.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  if (labels.empty()) {
    return std::nullopt;
  }

  const auto max_idx = labels.back().first;
  if (max_idx >= kMaxClassCount) {
    return std::nullopt;
  }

  std::vector<std::string> result;
  result.reserve(max_idx + 1);
  auto label = labels.begin();
  for (std::size_t i = 0; i <= max_idx; ++i) {
    if (label != labels.end() && label->first == i) {
      result.push_back(std::move(label->second));
      ++label;
    } else {
      result.push_back("class_" + std::to_string(i));
    }
  }
  return result;
}

} // namespace reco::detect
