#include "reco/control/intents.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace reco::control {
namespace {

std::string quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  out << std::hex;
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (ch < 0x20) {
        constexpr char digits[] = "0123456789abcdef";
        out << "\\u00" << digits[(ch >> 4) & 0x0F] << digits[ch & 0x0F];
      } else {
        out << static_cast<char>(ch);
      }
      break;
    }
  }
  out << '"';
  return out.str();
}

std::string normalize_scientific(std::string value, int exponent) {
  const auto exponent_pos = value.find_first_of("eE");
  if (exponent_pos == std::string::npos) {
    return value;
  }
  std::string mantissa = value.substr(0, exponent_pos);
  const auto dot_pos = mantissa.find('.');
  if (dot_pos != std::string::npos) {
    while (!mantissa.empty() && mantissa.back() == '0') {
      mantissa.pop_back();
    }
    if (!mantissa.empty() && mantissa.back() == '.') {
      mantissa.pop_back();
    }
  }
  return mantissa + (exponent >= 0 ? "e+" : "e") + std::to_string(exponent);
}

std::string rust_like_float_from_scientific(std::string value) {
  const auto exponent_pos = value.find_first_of("eE");
  if (exponent_pos == std::string::npos) {
    return value;
  }

  std::string mantissa = value.substr(0, exponent_pos);
  int exponent = 0;
  auto exponent_text = value.substr(exponent_pos + 1);
  if (!exponent_text.empty() && exponent_text.front() == '+') {
    exponent_text.erase(exponent_text.begin());
  }
  const auto result =
      std::from_chars(exponent_text.data(), exponent_text.data() + exponent_text.size(), exponent);
  if (result.ec != std::errc()) {
    return value;
  }
  if (exponent < -6 || exponent > 20) {
    return normalize_scientific(std::move(value), exponent);
  }

  std::string sign;
  if (!mantissa.empty() && mantissa.front() == '-') {
    sign = "-";
    mantissa.erase(mantissa.begin());
  }

  const auto dot_pos = mantissa.find('.');
  const int fractional_digits =
      dot_pos == std::string::npos ? 0 : static_cast<int>(mantissa.size() - dot_pos - 1);
  if (dot_pos != std::string::npos) {
    mantissa.erase(dot_pos, 1);
  }

  const int decimal_index = static_cast<int>(mantissa.size()) - fractional_digits + exponent;
  if (decimal_index <= 0) {
    return sign + "0." + std::string(static_cast<std::size_t>(-decimal_index), '0') + mantissa;
  }
  if (decimal_index >= static_cast<int>(mantissa.size())) {
    return sign + mantissa +
           std::string(static_cast<std::size_t>(decimal_index - static_cast<int>(mantissa.size())),
                       '0') +
           ".0";
  }
  mantissa.insert(static_cast<std::size_t>(decimal_index), ".");
  return sign + mantissa;
}

std::string scientific_from_float(float value) {
  std::array<char, 64> buffer{};
  const auto result =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific);
  if (result.ec != std::errc()) {
    throw std::runtime_error("failed to serialize float");
  }
  std::string out(buffer.data(), result.ptr);
  const auto exponent_pos = out.find_first_of("eE");
  if (exponent_pos == std::string::npos) {
    return out;
  }
  int exponent = 0;
  auto exponent_text = out.substr(exponent_pos + 1);
  if (!exponent_text.empty() && exponent_text.front() == '+') {
    exponent_text.erase(exponent_text.begin());
  }
  const auto exponent_result =
      std::from_chars(exponent_text.data(), exponent_text.data() + exponent_text.size(), exponent);
  if (exponent_result.ec != std::errc()) {
    return out;
  }
  return normalize_scientific(std::move(out), exponent);
}

std::string numeric(float value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  const float magnitude = std::abs(value);
  if (magnitude >= 1.0e20F || (magnitude > 0.0F && magnitude < 1.0e-6F)) {
    return scientific_from_float(value);
  }
  std::array<char, 64> buffer{};
  const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (result.ec != std::errc()) {
    throw std::runtime_error("failed to serialize float");
  }
  std::string out(buffer.data(), result.ptr);
  out = rust_like_float_from_scientific(std::move(out));
  if (out.find_first_of(".eE") == std::string::npos) {
    out += ".0";
  }
  return out;
}

std::string action_value(std::string_view action, const std::string& value_json) {
  return "{\"action\":\"" + std::string(action) + "\",\"value\":" + value_json + "}";
}

} // namespace

std::string to_json(HotkeyIntent intent) {
  switch (intent) {
  case HotkeyIntent::YawLeft:
    return "\"yaw_left\"";
  case HotkeyIntent::YawRight:
    return "\"yaw_right\"";
  case HotkeyIntent::PitchUp:
    return "\"pitch_up\"";
  case HotkeyIntent::PitchDown:
    return "\"pitch_down\"";
  case HotkeyIntent::ZoomIn:
    return "\"zoom_in\"";
  case HotkeyIntent::ZoomOut:
    return "\"zoom_out\"";
  case HotkeyIntent::Reset:
    return "\"reset\"";
  case HotkeyIntent::ToggleConstrained:
    return "\"toggle_constrained\"";
  }
  throw std::logic_error("unknown HotkeyIntent");
}

std::string to_json(CaptureIntent intent) {
  switch (intent) {
  case CaptureIntent::StartRecord:
    return "\"start_record\"";
  case CaptureIntent::StopRecord:
    return "\"stop_record\"";
  case CaptureIntent::Snapshot:
    return "\"snapshot\"";
  case CaptureIntent::ClearReplay:
    return "\"clear_replay\"";
  case CaptureIntent::SaveReplay:
    return "\"save_replay\"";
  }
  throw std::logic_error("unknown CaptureIntent");
}

std::string to_json(const ControlIntent& intent) {
  return std::visit(
      [](const auto& item) -> std::string {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, HotkeyControlIntent>) {
          return "{\"kind\":\"hotkey\",\"data\":" + to_json(item.value) + "}";
        } else if constexpr (std::is_same_v<T, CaptureControlIntent>) {
          return "{\"kind\":\"capture\",\"data\":" + to_json(item.value) + "}";
        } else if constexpr (std::is_same_v<T, PoseControlIntent>) {
          return std::visit(
              [](const auto& pose) -> std::string {
                using P = std::decay_t<decltype(pose)>;
                if constexpr (std::is_same_v<P, PoseSetYawRad>) {
                  return "{\"kind\":\"pose\",\"data\":" +
                         action_value("set_yaw_rad", numeric(pose.value)) + "}";
                } else if constexpr (std::is_same_v<P, PoseSetPitchRad>) {
                  return "{\"kind\":\"pose\",\"data\":" +
                         action_value("set_pitch_rad", numeric(pose.value)) + "}";
                } else if constexpr (std::is_same_v<P, PoseSetFovDeg>) {
                  return "{\"kind\":\"pose\",\"data\":" +
                         action_value("set_fov_deg", numeric(pose.value)) + "}";
                } else if constexpr (std::is_same_v<P, PoseDeltaYawRad>) {
                  return "{\"kind\":\"pose\",\"data\":" +
                         action_value("delta_yaw_rad", numeric(pose.value)) + "}";
                } else if constexpr (std::is_same_v<P, PoseDeltaPitchRad>) {
                  return "{\"kind\":\"pose\",\"data\":" +
                         action_value("delta_pitch_rad", numeric(pose.value)) + "}";
                } else if constexpr (std::is_same_v<P, PoseDeltaFovDeg>) {
                  return "{\"kind\":\"pose\",\"data\":" +
                         action_value("delta_fov_deg", numeric(pose.value)) + "}";
                } else {
                  return "{\"kind\":\"pose\",\"data\":{\"action\":\"reset\"}}";
                }
              },
              item.value);
        } else if constexpr (std::is_same_v<T, QualityControlIntent>) {
          return std::visit(
              [](const auto& quality) -> std::string {
                using Q = std::decay_t<decltype(quality)>;
                if constexpr (std::is_same_v<Q, QualitySetBitrate>) {
                  return "{\"kind\":\"quality\",\"data\":" +
                         action_value("set_bitrate", std::to_string(quality.value)) + "}";
                } else if constexpr (std::is_same_v<Q, QualitySetCodec>) {
                  return "{\"kind\":\"quality\",\"data\":" +
                         action_value("set_codec", quote(quality.value)) + "}";
                } else if constexpr (std::is_same_v<Q, QualitySetResolution>) {
                  return "{\"kind\":\"quality\",\"data\":{\"action\":\"set_resolution\",\"value\":{\"width\":" +
                         std::to_string(quality.width) + ",\"height\":" +
                         std::to_string(quality.height) + "}}}";
                } else if constexpr (std::is_same_v<Q, QualitySetPreset>) {
                  return "{\"kind\":\"quality\",\"data\":" +
                         action_value("set_preset", quote(quality.value)) + "}";
                } else {
                  return "{\"kind\":\"quality\",\"data\":" +
                         action_value("set_crf", std::to_string(quality.value)) + "}";
                }
              },
              item.value);
        } else {
          return std::visit(
              [](const auto& model) -> std::string {
                using M = std::decay_t<decltype(model)>;
                if constexpr (std::is_same_v<M, ModelSetDetectorModel>) {
                  return "{\"kind\":\"model_select\",\"data\":" +
                         action_value("set_detector_model", quote(model.value)) + "}";
                } else if constexpr (std::is_same_v<M, ModelSetDetectionInterval>) {
                  return "{\"kind\":\"model_select\",\"data\":" +
                         action_value("set_detection_interval", std::to_string(model.value)) + "}";
                } else {
                  return "{\"kind\":\"model_select\",\"data\":{\"action\":\"disable_detection\"}}";
                }
              },
              item.value);
        }
      },
      intent);
}

} // namespace reco::control
