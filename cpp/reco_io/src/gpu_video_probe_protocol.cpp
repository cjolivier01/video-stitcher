#include "gpu_video_probe_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace reco::io::detail {
namespace {

constexpr std::size_t kMaximumErrorMessageBytes = 1024;
constexpr std::size_t kMaximumCborNestingDepth = 32;
constexpr std::uint32_t kProbeProtocolVersion = 5;
constexpr double kMaximumSaneFps = 1000.0;
constexpr std::uint32_t kMaximumGpuVideoDimension = 8'192;
constexpr std::uint64_t kMaximumNv12FrameBytes = 8'192ULL * 8'192ULL * 3ULL / 2ULL;
constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr std::uint64_t kExactFrameCountTolerance = 1;
constexpr std::uint64_t kEstimatedFrameCountRatio = 16;
constexpr std::uint64_t kEstimatedFrameCountSlack = 64;

struct DivisionResult {
  std::uint64_t quotient;
  std::uint64_t remainder;
};

bool add_modulo(std::uint64_t left, std::uint64_t right, std::uint64_t divisor,
                std::uint64_t& result) {
  if (left >= divisor - right) {
    result = left - (divisor - right);
    return true;
  }
  result = left + right;
  return false;
}

std::optional<DivisionResult> multiply_divide(std::uint64_t multiplicand, std::uint32_t multiplier,
                                              std::uint64_t divisor) {
  const auto whole = multiplicand / divisor;
  if (whole != 0 && multiplier > std::numeric_limits<std::uint64_t>::max() / whole) {
    return std::nullopt;
  }

  std::uint64_t quotient = whole * multiplier;
  std::uint64_t partial_quotient = 0;
  std::uint64_t partial_remainder = 0;
  const auto remainder = multiplicand % divisor;
  for (std::uint32_t bit = 32; bit-- > 0;) {
    if (partial_quotient > std::numeric_limits<std::uint64_t>::max() / 2U) {
      return std::nullopt;
    }
    partial_quotient *= 2U;
    std::uint64_t next_remainder = 0;
    const bool doubled_remainder_wrapped =
        add_modulo(partial_remainder, partial_remainder, divisor, next_remainder);
    if (doubled_remainder_wrapped &&
        partial_quotient == std::numeric_limits<std::uint64_t>::max()) {
      return std::nullopt;
    }
    partial_quotient += doubled_remainder_wrapped ? 1U : 0U;
    partial_remainder = next_remainder;
    if (((multiplier >> bit) & 1U) != 0U) {
      const bool added_remainder_wrapped =
          add_modulo(partial_remainder, remainder, divisor, next_remainder);
      if (added_remainder_wrapped &&
          partial_quotient == std::numeric_limits<std::uint64_t>::max()) {
        return std::nullopt;
      }
      partial_quotient += added_remainder_wrapped ? 1U : 0U;
      partial_remainder = next_remainder;
    }
  }
  if (partial_quotient > std::numeric_limits<std::uint64_t>::max() - quotient) {
    return std::nullopt;
  }
  return DivisionResult{quotient + partial_quotient, partial_remainder};
}

std::optional<std::uint64_t> rounded_frame_count(std::uint64_t duration_ns,
                                                 std::uint32_t fps_numerator,
                                                 std::uint32_t fps_denominator) {
  const auto divisor = kNanosecondsPerSecond * static_cast<std::uint64_t>(fps_denominator);
  const auto division = multiply_divide(duration_ns, fps_numerator, divisor);
  if (!division.has_value()) {
    return std::nullopt;
  }
  const bool round_up = division->remainder >= divisor - division->remainder;
  if (round_up && division->quotient == std::numeric_limits<std::uint64_t>::max()) {
    return std::nullopt;
  }
  return division->quotient + (round_up ? 1U : 0U);
}

bool exact_frame_count_is_consistent(const GpuVideoProbe& probe) {
  const auto expected =
      rounded_frame_count(probe.duration_ns, probe.fps_numerator, probe.fps_denominator);
  if (!expected.has_value()) {
    return false;
  }
  const auto difference = probe.total_frames > *expected ? probe.total_frames - *expected
                                                         : *expected - probe.total_frames;
  return difference <= kExactFrameCountTolerance;
}

bool estimated_frame_count_is_plausible(const GpuVideoProbe& probe) {
  const auto expected =
      rounded_frame_count(probe.duration_ns, probe.fps_numerator, probe.fps_denominator);
  if (!expected.has_value()) {
    return false;
  }
  const auto divided_lower = *expected / kEstimatedFrameCountRatio;
  const auto lower =
      divided_lower > kEstimatedFrameCountSlack ? divided_lower - kEstimatedFrameCountSlack : 0U;
  const auto multiplied_upper =
      *expected > (std::numeric_limits<std::uint64_t>::max() - kEstimatedFrameCountSlack) /
                      kEstimatedFrameCountRatio
          ? std::numeric_limits<std::uint64_t>::max()
          : *expected * kEstimatedFrameCountRatio + kEstimatedFrameCountSlack;
  return probe.total_frames >= lower && probe.total_frames <= multiplied_upper;
}

bool gpu_geometry_is_valid(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0 || width > kMaximumGpuVideoDimension ||
      height > kMaximumGpuVideoDimension || (width % 2U) != 0 || (height % 2U) != 0 ||
      width > std::numeric_limits<std::uint64_t>::max() / height) {
    return false;
  }
  const auto pixels = static_cast<std::uint64_t>(width) * height;
  if (pixels > std::numeric_limits<std::uint64_t>::max() - pixels / 2U) {
    return false;
  }
  return pixels + pixels / 2U <= kMaximumNv12FrameBytes;
}

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

[[noreturn]] void throw_invalid_cbor(std::string_view description, std::string_view reason) {
  throw GpuVideoProbeError(std::string(description) + " is not valid CBOR: " + std::string(reason));
}

std::uint64_t read_cbor_argument(std::string_view payload, std::size_t& offset,
                                 unsigned char additional, std::string_view description) {
  if (additional <= 23U) {
    return additional;
  }
  std::size_t byte_count = 0;
  switch (additional) {
  case 24U:
    byte_count = 1;
    break;
  case 25U:
    byte_count = 2;
    break;
  case 26U:
    byte_count = 4;
    break;
  case 27U:
    byte_count = 8;
    break;
  default:
    throw_invalid_cbor(description, "indefinite or reserved item length");
  }
  if (byte_count > payload.size() - offset) {
    throw_invalid_cbor(description, "truncated item argument");
  }
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < byte_count; ++index) {
    value =
        (value << 8U) | static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++]));
  }
  return value;
}

void validate_cbor_structure(std::string_view payload, std::string_view description) {
  std::vector<std::uint64_t> remaining_items{1};
  std::size_t offset = 0;
  while (!remaining_items.empty()) {
    if (remaining_items.back() == 0) {
      remaining_items.pop_back();
      continue;
    }
    if (offset >= payload.size()) {
      throw_invalid_cbor(description, "truncated container");
    }
    --remaining_items.back();
    const auto initial = static_cast<unsigned char>(payload[offset++]);
    const auto major = static_cast<unsigned char>(initial >> 5U);
    const auto additional = static_cast<unsigned char>(initial & 0x1FU);
    const auto argument = read_cbor_argument(payload, offset, additional, description);

    if (major == 2U || major == 3U) {
      if (argument > payload.size() - offset) {
        throw_invalid_cbor(description, "truncated string");
      }
      offset += static_cast<std::size_t>(argument);
      continue;
    }
    if (major == 4U || major == 5U) {
      if (major == 5U && argument > std::numeric_limits<std::uint64_t>::max() / 2U) {
        throw_invalid_cbor(description, "map length overflow");
      }
      const auto child_items = major == 5U ? argument * 2U : argument;
      if (child_items != 0) {
        if (remaining_items.size() > kMaximumCborNestingDepth) {
          throw_invalid_cbor(description, "nesting exceeds the protocol limit");
        }
        remaining_items.push_back(child_items);
      }
      continue;
    }
    if (major == 6U) {
      throw_invalid_cbor(description, "semantic tags are not permitted");
    }
    if (major > 7U) {
      throw_invalid_cbor(description, "unknown major type");
    }
  }
  if (offset != payload.size()) {
    throw_invalid_cbor(description, "trailing data");
  }
}

nlohmann::json parse_payload(std::string_view payload, std::string_view description) {
  if (payload.empty() || payload.size() > kMaximumProbeIpcBytes) {
    throw GpuVideoProbeError(std::string(description) + " has an invalid size");
  }
  validate_cbor_structure(payload, description);
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
    const auto& value = json.at(std::string(key));
    if constexpr (std::is_same_v<Value, bool>) {
      if (!value.is_boolean()) {
        throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key));
      }
      return value.get<bool>();
    } else if constexpr (std::is_same_v<Value, std::string>) {
      if (!value.is_string()) {
        throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key));
      }
      return value.get<std::string>();
    } else if constexpr (std::is_integral_v<Value>) {
      if (value.is_number_unsigned()) {
        const auto raw = value.get<std::uint64_t>();
        if (raw > static_cast<std::uint64_t>(std::numeric_limits<Value>::max())) {
          throw GpuVideoProbeError(std::string(description) + " has an out-of-range " +
                                   std::string(key));
        }
        return static_cast<Value>(raw);
      }
      if (value.is_number_integer()) {
        const auto raw = value.get<std::int64_t>();
        if constexpr (std::is_unsigned_v<Value>) {
          if (raw < 0 || static_cast<std::uint64_t>(raw) >
                             static_cast<std::uint64_t>(std::numeric_limits<Value>::max())) {
            throw GpuVideoProbeError(std::string(description) + " has an out-of-range " +
                                     std::string(key));
          }
        } else if (raw < static_cast<std::int64_t>(std::numeric_limits<Value>::min()) ||
                   raw > static_cast<std::int64_t>(std::numeric_limits<Value>::max())) {
          throw GpuVideoProbeError(std::string(description) + " has an out-of-range " +
                                   std::string(key));
        }
        return static_cast<Value>(raw);
      }
      throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key));
    } else {
      static_assert(!sizeof(Value), "required_value does not support this type");
    }
  } catch (const nlohmann::json::exception& error) {
    throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key) +
                             ": " + error.what());
  }
}

std::optional<std::uint64_t> required_optional_u64(const nlohmann::json& json, std::string_view key,
                                                   std::string_view description) {
  try {
    if (json.at(std::string(key)).is_null()) {
      return std::nullopt;
    }
  } catch (const nlohmann::json::exception& error) {
    throw GpuVideoProbeError(std::string(description) + " has an invalid " + std::string(key) +
                             ": " + error.what());
  }
  return required_value<std::uint64_t>(json, key, description);
}

nlohmann::json optional_u64(const std::optional<std::uint64_t>& value) {
  return value.has_value() ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

} // namespace

ProbeIpcFrameHeader encode_probe_ipc_frame_header(std::size_t payload_size) {
  if (payload_size == 0 || payload_size > kMaximumProbeIpcBytes) {
    throw GpuVideoProbeError("video probe worker IPC frame length is invalid");
  }
  const auto size = static_cast<std::uint32_t>(payload_size);
  return {static_cast<char>((size >> 24U) & 0xFFU), static_cast<char>((size >> 16U) & 0xFFU),
          static_cast<char>((size >> 8U) & 0xFFU), static_cast<char>(size & 0xFFU)};
}

std::size_t decode_probe_ipc_frame_header(const ProbeIpcFrameHeader& header) {
  const auto byte = [](char value) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(value));
  };
  const auto size = (byte(header[0]) << 24U) | (byte(header[1]) << 16U) | (byte(header[2]) << 8U) |
                    byte(header[3]);
  if (size == 0 || size > kMaximumProbeIpcBytes) {
    throw GpuVideoProbeError("video probe worker IPC frame length is invalid");
  }
  return size;
}

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
  return serialize_payload(nlohmann::json{
      {"protocol_version", kProbeProtocolVersion},
      {"ok", true},
      {"width", probe.width},
      {"height", probe.height},
      {"fps_numerator", probe.fps_numerator},
      {"fps_denominator", probe.fps_denominator},
      {"duration_ns", probe.duration_ns},
      {"total_frames", probe.total_frames},
      {"first_stream_time_ns", optional_u64(probe.first_stream_time_ns)},
      {"timestamp_multiplicity", probe.timestamp_multiplicity},
      {"duration_is_estimated", probe.duration_is_estimated},
      {"total_frames_is_estimated", probe.total_frames_is_estimated},
      {"selected_stream_caps_verified", probe.selected_stream_caps_verified},
      {"indexed_sampling_cadence_verified", probe.indexed_sampling_cadence_verified}});
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
  probe.first_stream_time_ns =
      required_optional_u64(response, "first_stream_time_ns", "video probe worker response");
  probe.timestamp_multiplicity = required_value<std::uint32_t>(response, "timestamp_multiplicity",
                                                               "video probe worker response");
  probe.duration_is_estimated =
      required_value<bool>(response, "duration_is_estimated", "video probe worker response");
  probe.total_frames_is_estimated =
      required_value<bool>(response, "total_frames_is_estimated", "video probe worker response");
  probe.selected_stream_caps_verified = required_value<bool>(
      response, "selected_stream_caps_verified", "video probe worker response");
  probe.indexed_sampling_cadence_verified = required_value<bool>(
      response, "indexed_sampling_cadence_verified", "video probe worker response");
  if (!gpu_geometry_is_valid(probe.width, probe.height) || probe.fps_numerator == 0 ||
      probe.fps_denominator == 0 || probe.duration_ns == 0 || probe.total_frames == 0 ||
      probe.timestamp_multiplicity == 0 ||
      probe.timestamp_multiplicity > kMaximumIndexedTimestampMultiplicity) {
    throw GpuVideoProbeError("video probe worker returned invalid metadata");
  }
  probe.fps = static_cast<double>(probe.fps_numerator) / probe.fps_denominator;
  if (!std::isfinite(probe.fps) || probe.fps <= 0.0 || probe.fps > kMaximumSaneFps) {
    throw GpuVideoProbeError("video probe worker returned invalid metadata");
  }
  const auto average_fps = static_cast<long double>(probe.total_frames) * 1'000'000'000.0L /
                           static_cast<long double>(probe.duration_ns);
  if (!std::isfinite(average_fps) || average_fps > kMaximumSaneFps * 1.01L) {
    throw GpuVideoProbeError("video probe worker returned invalid metadata");
  }
  if (!probe.duration_is_estimated && !probe.total_frames_is_estimated &&
      !exact_frame_count_is_consistent(probe)) {
    throw GpuVideoProbeError("video probe worker returned inconsistent exact metadata");
  }
  if ((probe.duration_is_estimated || probe.total_frames_is_estimated) &&
      !estimated_frame_count_is_plausible(probe)) {
    throw GpuVideoProbeError("video probe worker returned implausible estimated metadata");
  }
  if (probe.indexed_sampling_cadence_verified &&
      (probe.total_frames < 3 || probe.total_frames_is_estimated ||
       !probe.selected_stream_caps_verified || !probe.first_stream_time_ns.has_value() ||
       !exact_frame_count_is_consistent(probe) ||
       probe.total_frames % probe.timestamp_multiplicity != 0U)) {
    throw GpuVideoProbeError("video probe worker returned inconsistent cadence proof metadata");
  }
  if (probe.selected_stream_caps_verified && probe.total_frames_is_estimated) {
    throw GpuVideoProbeError("video probe worker returned inconsistent caps proof metadata");
  }
  return probe;
}

} // namespace reco::io::detail
