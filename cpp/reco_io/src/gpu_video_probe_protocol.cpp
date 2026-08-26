#include "gpu_video_probe_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace reco::io::detail {
namespace {

constexpr std::size_t kMaximumErrorMessageBytes = 1024;
constexpr std::uint32_t kProbeProtocolVersion = 1;
constexpr double kMaximumSaneFps = 1000.0;

int encode_container(const std::optional<GpuDecodeContainer>& container) {
  return container.has_value() ? static_cast<int>(*container) : -1;
}

GpuDecodeCodec decode_codec(int value) {
  if (value < static_cast<int>(GpuDecodeCodec::H264) ||
      value > static_cast<int>(GpuDecodeCodec::Hevc)) {
    throw std::invalid_argument("video probe worker request has an invalid codec");
  }
  return static_cast<GpuDecodeCodec>(value);
}

std::optional<GpuDecodeContainer> decode_container(int value) {
  if (value == -1) {
    return std::nullopt;
  }
  if (value < static_cast<int>(GpuDecodeContainer::QuickTime) ||
      value > static_cast<int>(GpuDecodeContainer::MpegTs)) {
    throw std::invalid_argument("video probe worker request has an invalid container");
  }
  return static_cast<GpuDecodeContainer>(value);
}

nlohmann::json parse_payload(std::string_view payload, std::string_view description) {
  if (payload.empty() || payload.size() > kMaximumProbeIpcBytes) {
    throw GpuVideoProbeError(std::string(description) + " has an invalid size");
  }
  try {
    return nlohmann::json::from_cbor(payload.begin(), payload.end(), true, true);
  } catch (const nlohmann::json::exception& error) {
    throw GpuVideoProbeError(std::string(description) + " is not valid CBOR: " + error.what());
  }
}

std::string serialize_payload(const nlohmann::json& payload) {
  const auto bytes = nlohmann::json::to_cbor(payload);
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

nlohmann::json::binary_t bytes(std::string_view value) {
  return nlohmann::json::binary_t(std::vector<std::uint8_t>(value.begin(), value.end()));
}

std::string required_bytes(const nlohmann::json& json, std::string_view key,
                           std::string_view description) {
  try {
    const auto& value = json.at(std::string(key));
    if (!value.is_binary()) {
      throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key));
    }
    const auto& encoded = value.get_binary();
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
  } catch (const nlohmann::json::exception& error) {
    throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key) +
                             ": " + error.what());
  }
}

template <typename Value>
Value required_value(const nlohmann::json& json, std::string_view key,
                     std::string_view description) {
  try {
    return json.at(std::string(key)).get<Value>();
  } catch (const nlohmann::json::exception& error) {
    throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key) +
                             ": " + error.what());
  }
}

} // namespace

std::string encode_probe_request(const GpuFileDecodeConfig& config, std::uint64_t timeout_ns) {
  const nlohmann::json request{{"protocol_version", kProbeProtocolVersion},
                               {"path", nlohmann::json::binary(bytes(config.path))},
                               {"codec", static_cast<int>(config.codec)},
                               {"elementary_stream", config.elementary_stream},
                               {"container", encode_container(config.container)},
                               {"max_buffers", config.max_buffers},
                               {"drop", config.drop},
                               {"timeout_ns", timeout_ns}};
  auto payload = serialize_payload(request);
  if (payload.size() > kMaximumProbeIpcBytes) {
    throw std::invalid_argument("video probe worker request exceeds the IPC size limit");
  }
  return payload;
}

ProbeWorkerRequest decode_probe_request(std::string_view payload) {
  const auto request = parse_payload(payload, "video probe worker request");
  if (!request.is_object()) {
    throw GpuVideoProbeError("video probe worker request must be a CBOR map");
  }
  if (required_value<std::uint32_t>(request, "protocol_version", "video probe worker request") !=
      kProbeProtocolVersion) {
    throw GpuVideoProbeError("video probe worker request uses an unsupported protocol version");
  }
  ProbeWorkerRequest decoded;
  decoded.config.path = required_bytes(request, "path", "video probe worker request");
  decoded.config.codec =
      decode_codec(required_value<int>(request, "codec", "video probe worker request"));
  decoded.config.elementary_stream =
      required_value<bool>(request, "elementary_stream", "video probe worker request");
  decoded.config.container =
      decode_container(required_value<int>(request, "container", "video probe worker request"));
  decoded.config.max_buffers =
      required_value<std::uint32_t>(request, "max_buffers", "video probe worker request");
  decoded.config.drop = required_value<bool>(request, "drop", "video probe worker request");
  decoded.timeout_ns =
      required_value<std::uint64_t>(request, "timeout_ns", "video probe worker request");
  return decoded;
}

std::string encode_probe_success(const GpuVideoProbe& probe) {
  return serialize_payload(
      nlohmann::json{{"protocol_version", kProbeProtocolVersion},
                     {"ok", true},
                     {"width", probe.width},
                     {"height", probe.height},
                     {"fps_numerator", probe.fps_numerator},
                     {"fps_denominator", probe.fps_denominator},
                     {"duration_ns", probe.duration_ns},
                     {"total_frames", probe.total_frames},
                     {"duration_is_estimated", probe.duration_is_estimated},
                     {"total_frames_is_estimated", probe.total_frames_is_estimated}});
}

std::string encode_probe_failure(std::string_view kind, std::string_view message) {
  const auto bounded_message =
      message.substr(0, std::min(message.size(), kMaximumErrorMessageBytes));
  return serialize_payload(
      nlohmann::json{{"protocol_version", kProbeProtocolVersion},
                     {"ok", false},
                     {"kind", kind},
                     {"message", nlohmann::json::binary(bytes(bounded_message))}});
}

GpuVideoProbe decode_probe_response(std::string_view payload) {
  const auto response = parse_payload(payload, "video probe worker response");
  if (!response.is_object()) {
    throw GpuVideoProbeError("video probe worker response must be a CBOR map");
  }
  if (required_value<std::uint32_t>(response, "protocol_version", "video probe worker response") !=
      kProbeProtocolVersion) {
    throw GpuVideoProbeError("video probe worker response uses an unsupported protocol version");
  }
  if (!required_value<bool>(response, "ok", "video probe worker response")) {
    const auto kind = required_value<std::string>(response, "kind", "video probe worker response");
    const auto message = required_bytes(response, "message", "video probe worker response");
    if (kind == "invalid_argument") {
      throw std::invalid_argument(message);
    }
    throw GpuVideoProbeError(message);
  }

  GpuVideoProbe probe;
  probe.width = required_value<std::uint32_t>(response, "width", "video probe worker response");
  probe.height = required_value<std::uint32_t>(response, "height", "video probe worker response");
  probe.fps_numerator =
      required_value<std::uint32_t>(response, "fps_numerator", "video probe worker response");
  probe.fps_denominator =
      required_value<std::uint32_t>(response, "fps_denominator", "video probe worker response");
  probe.duration_ns =
      required_value<std::uint64_t>(response, "duration_ns", "video probe worker response");
  probe.total_frames =
      required_value<std::uint64_t>(response, "total_frames", "video probe worker response");
  probe.duration_is_estimated =
      required_value<bool>(response, "duration_is_estimated", "video probe worker response");
  probe.total_frames_is_estimated =
      required_value<bool>(response, "total_frames_is_estimated", "video probe worker response");
  if (probe.width == 0 || probe.height == 0 || (probe.width % 2U) != 0 ||
      (probe.height % 2U) != 0 || probe.fps_numerator == 0 || probe.fps_denominator == 0 ||
      probe.duration_ns == 0 || probe.total_frames == 0) {
    throw GpuVideoProbeError("video probe worker returned invalid metadata");
  }
  probe.fps = static_cast<double>(probe.fps_numerator) / probe.fps_denominator;
  if (!std::isfinite(probe.fps) || probe.fps <= 0.0 || probe.fps > kMaximumSaneFps) {
    throw GpuVideoProbeError("video probe worker returned invalid metadata");
  }
  return probe;
}

} // namespace reco::io::detail
