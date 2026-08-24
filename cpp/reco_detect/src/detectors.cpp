#include "reco/detect/detectors.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <string_view>

namespace reco::detect {
namespace {

float clamp01(float value) { return std::clamp(value, 0.0F, 1.0F); }

constexpr std::size_t kMaxClassCount = 10000;

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

} // namespace

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
                                           CameraId camera, float confidence_threshold,
                                           float scale, float pad_x, float pad_y,
                                           std::uint32_t frame_width,
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
    const auto entry = trim_ascii(inner.substr(start, comma == std::string_view::npos
                                                         ? std::string_view::npos
                                                         : comma - start));
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
