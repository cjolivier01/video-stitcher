#include "reco/detect/detectors.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
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
