#include "gpu_video_probe_protocol.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>

namespace {

int failures = 0;

std::string encode(const nlohmann::json& value) {
  const auto bytes = nlohmann::json::to_cbor(value);
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

nlohmann::json valid_request() {
  return {{"protocol_version", 5U},
          {"path", nlohmann::json::binary({0x76, 0x69, 0x64, 0x65, 0x6f})},
          {"codec", 0},
          {"elementary_stream", false},
          {"container", 0},
          {"max_buffers", 4U},
          {"drop", false},
          {"timeout_ns", 1'000'000'000ULL}};
}

template <typename Function>
void expect_probe_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const reco::io::GpuVideoProbeError& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function> void expect_success(Function&& function, std::string_view message) {
  try {
    function();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw: " << error.what() << '\n';
    ++failures;
  }
}

nlohmann::json valid_response() {
  return {{"protocol_version", 5U},
          {"ok", true},
          {"width", 1920U},
          {"height", 1080U},
          {"fps_numerator", 30U},
          {"fps_denominator", 1U},
          {"duration_ns", 1'000'000'000ULL},
          {"total_frames", 30ULL},
          {"first_stream_time_ns", 766'666'666ULL},
          {"timestamp_multiplicity", 1U},
          {"duration_is_estimated", false},
          {"total_frames_is_estimated", false},
          {"selected_stream_caps_verified", true},
          {"indexed_sampling_cadence_verified", true}};
}

void request_numeric_domains_are_enforced() {
  auto previous_protocol = valid_request();
  previous_protocol["protocol_version"] = 3U;
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_request(encode(previous_protocol)); },
      "unsupported protocol version", "previous request schema is rejected explicitly");

  for (const auto& [key, value] : std::initializer_list<std::pair<std::string, nlohmann::json>>{
           {"protocol_version", std::numeric_limits<std::uint64_t>::max()},
           {"codec", std::numeric_limits<std::uint64_t>::max()},
           {"container", std::numeric_limits<std::uint64_t>::max()},
           {"max_buffers", -1},
           {"timeout_ns", -1}}) {
    auto request = valid_request();
    request[key] = value;
    expect_probe_error([&] { (void)reco::io::detail::decode_probe_request(encode(request)); }, key,
                       "out-of-domain request field " + key);
  }

  auto floating_codec = valid_request();
  floating_codec["codec"] = 0.0;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_request(encode(floating_codec)); },
                     "codec", "floating-point codec is rejected");
}

void response_numeric_domains_are_enforced() {
  const auto base = valid_response();
  auto previous_protocol = base;
  previous_protocol["protocol_version"] = 3U;
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(previous_protocol)); },
      "unsupported protocol version", "previous response schema is rejected explicitly");

  auto missing_cadence_proof = base;
  missing_cadence_proof.erase("indexed_sampling_cadence_verified");
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(missing_cadence_proof)); },
      "indexed_sampling_cadence_verified", "cadence proof is mandatory in protocol version 5");

  auto missing_caps_proof = base;
  missing_caps_proof.erase("selected_stream_caps_verified");
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(missing_caps_proof)); },
      "selected_stream_caps_verified", "caps proof is mandatory in protocol version 5");

  auto missing_stream_origin = base;
  missing_stream_origin.erase("first_stream_time_ns");
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(missing_stream_origin)); },
      "first_stream_time_ns", "stream-time origin is mandatory in protocol version 5");

  auto missing_multiplicity = base;
  missing_multiplicity.erase("timestamp_multiplicity");
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(missing_multiplicity)); },
      "timestamp_multiplicity", "timestamp multiplicity is mandatory in protocol version 5");

  for (const auto& [key, value] : std::initializer_list<std::pair<std::string, nlohmann::json>>{
           {"protocol_version", std::numeric_limits<std::uint64_t>::max()},
           {"width", -2},
           {"duration_ns", -1},
           {"total_frames", -1},
           {"timestamp_multiplicity", -1}}) {
    auto response = base;
    response[key] = value;
    expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(response)); },
                       key, "out-of-domain response field " + key);
  }

  auto maximum_dimensions = base;
  maximum_dimensions["width"] = 8'192U;
  maximum_dimensions["height"] = 8'192U;
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(maximum_dimensions)); },
                 "maximum supported NVDEC geometry is accepted");

  auto impossible_dimensions = maximum_dimensions;
  impossible_dimensions["width"] = 8'194U;
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(impossible_dimensions)); },
      "invalid metadata", "impossible GPU dimensions are rejected");

  auto impossible_frame_count = base;
  impossible_frame_count["total_frames"] = std::numeric_limits<std::uint64_t>::max();
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(impossible_frame_count)); },
      "invalid metadata", "impossible average frame counts are rejected");
}

void stream_time_origin_round_trips() {
  const auto decoded = reco::io::detail::decode_probe_response(encode(valid_response()));
  if (decoded.first_stream_time_ns != std::optional<std::uint64_t>(766'666'666ULL)) {
    std::cerr << "FAIL: nonzero stream-time origin decodes from protocol\n";
    ++failures;
  }

  const auto round_tripped =
      reco::io::detail::decode_probe_response(reco::io::detail::encode_probe_success(decoded));
  if (round_tripped.first_stream_time_ns != decoded.first_stream_time_ns) {
    std::cerr << "FAIL: nonzero stream-time origin survives protocol round trip\n";
    ++failures;
  }

  auto untimed = valid_response();
  untimed["first_stream_time_ns"] = nullptr;
  untimed["selected_stream_caps_verified"] = false;
  untimed["indexed_sampling_cadence_verified"] = false;
  const auto decoded_untimed = reco::io::detail::decode_probe_response(encode(untimed));
  if (decoded_untimed.first_stream_time_ns.has_value()) {
    std::cerr << "FAIL: missing stream-time origin remains explicit\n";
    ++failures;
  }

  untimed["indexed_sampling_cadence_verified"] = true;
  untimed["selected_stream_caps_verified"] = true;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(untimed)); },
                     "cadence proof metadata",
                     "cadence proof without a stream-time origin is rejected");
}

void timestamp_multiplicity_round_trips() {
  auto duplicate = valid_response();
  duplicate["fps_numerator"] = 50U;
  duplicate["duration_ns"] = 4'000'000'000ULL;
  duplicate["total_frames"] = 200ULL;
  duplicate["timestamp_multiplicity"] = 2U;
  const auto decoded = reco::io::detail::decode_probe_response(encode(duplicate));
  if (decoded.timestamp_multiplicity != 2U) {
    std::cerr << "FAIL: duplicate timestamp multiplicity decodes from protocol\n";
    ++failures;
  }
  const auto round_tripped =
      reco::io::detail::decode_probe_response(reco::io::detail::encode_probe_success(decoded));
  if (round_tripped.timestamp_multiplicity != 2U) {
    std::cerr << "FAIL: duplicate timestamp multiplicity survives protocol round trip\n";
    ++failures;
  }

  duplicate["timestamp_multiplicity"] = 0U;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(duplicate)); },
                     "invalid metadata", "zero timestamp multiplicity is rejected");
  duplicate["timestamp_multiplicity"] = 3U;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(duplicate)); },
                     "cadence proof metadata",
                     "exact frame count must contain complete timestamp groups");
}

void exact_response_metadata_must_be_consistent() {
  auto contradiction = valid_response();
  contradiction["fps_numerator"] = 1000U;
  contradiction["duration_ns"] = 3'600'000'000'000ULL;
  contradiction["total_frames"] = 1ULL;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(contradiction)); },
                     "inconsistent exact metadata",
                     "contradictory exact response metadata is rejected");

  auto estimated = contradiction;
  estimated["duration_is_estimated"] = true;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(estimated)); },
                     "implausible estimated metadata",
                     "estimated duration cannot bypass bounded plausibility validation");
  estimated["duration_is_estimated"] = false;
  estimated["total_frames_is_estimated"] = true;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(estimated)); },
                     "implausible estimated metadata",
                     "estimated frame count cannot bypass bounded plausibility validation");

  auto plausible_estimate = valid_response();
  plausible_estimate["duration_ns"] = 10'000'000'000ULL;
  plausible_estimate["total_frames"] = 240ULL;
  plausible_estimate["duration_is_estimated"] = true;
  plausible_estimate["total_frames_is_estimated"] = true;
  plausible_estimate["selected_stream_caps_verified"] = false;
  plausible_estimate["indexed_sampling_cadence_verified"] = false;
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(plausible_estimate)); },
                 "bounded estimated metadata remains accepted");

  auto impossible_cadence_proof = plausible_estimate;
  impossible_cadence_proof["fps_numerator"] = 30U;
  impossible_cadence_proof["duration_ns"] = 10'000'000'000ULL;
  impossible_cadence_proof["total_frames"] = 1ULL;
  impossible_cadence_proof["indexed_sampling_cadence_verified"] = true;
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(impossible_cadence_proof)); },
      "cadence proof metadata", "estimated metadata cannot claim an impossible cadence proof");

  auto impossible_caps_proof = plausible_estimate;
  impossible_caps_proof["selected_stream_caps_verified"] = true;
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(impossible_caps_proof)); },
      "caps proof metadata", "EOS-complete caps proof cannot retain an estimated frame count");

  auto boundary = valid_response();
  boundary["fps_numerator"] = 100U;
  boundary["duration_ns"] = 10'000'000'000ULL;
  boundary["total_frames"] = 999ULL;
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(boundary)); },
                 "exact response at the lower dropped-frame tolerance boundary");
  boundary["total_frames"] = 1001ULL;
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(boundary)); },
                 "exact response at the upper dropped-frame tolerance boundary");

  boundary["total_frames"] = 998ULL;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(boundary)); },
                     "inconsistent exact metadata",
                     "exact response below the dropped-frame tolerance is rejected");
  boundary["total_frames"] = 1002ULL;
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(encode(boundary)); },
                     "inconsistent exact metadata",
                     "exact response above the dropped-frame tolerance is rejected");

  auto short_contradiction = valid_response();
  short_contradiction["total_frames"] = 1ULL;
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(short_contradiction)); },
      "inconsistent exact metadata", "short exact response cannot hide a relative contradiction");

  auto rounded_duration = valid_response();
  rounded_duration["fps_numerator"] = 30'000U;
  rounded_duration["fps_denominator"] = 1001U;
  rounded_duration["duration_ns"] = 1'000'000'001ULL;
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(rounded_duration)); },
                 "plausibly rounded exact duration");

  auto large_duration = valid_response();
  large_duration["fps_numerator"] = 1000U;
  large_duration["duration_ns"] = std::numeric_limits<std::uint64_t>::max();
  large_duration["total_frames"] = 18'446'744'073'710ULL;
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(large_duration)); },
                 "large exact metadata with an overflowing raw product");
}

void ipc_frame_lengths_are_enforced() {
  const auto one_byte = reco::io::detail::encode_probe_ipc_frame_header(1);
  if (one_byte != reco::io::detail::ProbeIpcFrameHeader{0, 0, 0, 1} ||
      reco::io::detail::decode_probe_ipc_frame_header(one_byte) != 1) {
    std::cerr << "FAIL: one-byte IPC frame length round trip\n";
    ++failures;
  }

  const auto maximum =
      reco::io::detail::encode_probe_ipc_frame_header(reco::io::detail::kMaximumProbeIpcBytes);
  if (reco::io::detail::decode_probe_ipc_frame_header(maximum) !=
      reco::io::detail::kMaximumProbeIpcBytes) {
    std::cerr << "FAIL: maximum IPC frame length round trip\n";
    ++failures;
  }

  expect_probe_error([] { (void)reco::io::detail::encode_probe_ipc_frame_header(0); },
                     "frame length", "empty IPC frame rejected");
  expect_probe_error(
      [] {
        (void)reco::io::detail::encode_probe_ipc_frame_header(
            reco::io::detail::kMaximumProbeIpcBytes + 1U);
      },
      "frame length", "oversized IPC frame rejected");
  expect_probe_error([] { (void)reco::io::detail::decode_probe_ipc_frame_header({0, 0, 0, 0}); },
                     "frame length", "zero wire IPC frame rejected");
  expect_probe_error([] { (void)reco::io::detail::decode_probe_ipc_frame_header({0, 4, 0, 1}); },
                     "frame length", "oversized wire IPC frame rejected");
}

void cbor_nesting_is_bounded_before_parsing() {
  std::string nested(64, static_cast<char>(0x81));
  nested.push_back(static_cast<char>(0xF6));
  expect_probe_error([&] { (void)reco::io::detail::decode_probe_response(nested); }, "nesting",
                     "deep CBOR response rejected before recursive parsing");
}

} // namespace

int main() {
  request_numeric_domains_are_enforced();
  response_numeric_domains_are_enforced();
  stream_time_origin_round_trips();
  timestamp_multiplicity_round_trips();
  exact_response_metadata_must_be_consistent();
  ipc_frame_lengths_are_enforced();
  cbor_nesting_is_bounded_before_parsing();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
