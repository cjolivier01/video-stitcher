#pragma once

#include "reco/core/pipeline_event.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
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

[[nodiscard]] std::vector<Detection> postprocess(const std::vector<float>& data, std::size_t n,
                                                 CameraId camera, float confidence_threshold,
                                                 float scale, float pad_x, float pad_y,
                                                 std::uint32_t frame_width,
                                                 std::uint32_t frame_height);

[[nodiscard]] std::vector<Detection> postprocess_balldet(
    const std::vector<float>& data, std::size_t n, CameraId camera, float confidence_threshold,
    float scale, float pad_x, float pad_y, std::uint32_t frame_width, std::uint32_t frame_height);

[[nodiscard]] float box_iou(const Detection& a, const Detection& b);
[[nodiscard]] std::vector<Detection> greedy_nms(std::vector<Detection> detections,
                                                float iou_threshold);
[[nodiscard]] std::vector<std::string> read_labels_file(const std::string& path);
[[nodiscard]] std::optional<std::vector<std::string>> parse_names_dict_string(
    std::string_view names_str);

} // namespace reco::detect
