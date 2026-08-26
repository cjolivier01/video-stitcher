#include "gpu_video_probe_protocol.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace {

int failures = 0;

std::string encode(const nlohmann::json& value) {
  const auto bytes = nlohmann::json::to_cbor(value);
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

nlohmann::json valid_request() {
  return {{"protocol_version", 1U},
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
  return {{"protocol_version", 1U},
          {"ok", true},
          {"width", 1920U},
          {"height", 1080U},
          {"fps_numerator", 30U},
          {"fps_denominator", 1U},
          {"duration_ns", 1'000'000'000ULL},
          {"total_frames", 30ULL},
          {"duration_is_estimated", false},
          {"total_frames_is_estimated", false}};
}

void request_numeric_domains_are_enforced() {
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
  for (const auto& [key, value] : std::initializer_list<std::pair<std::string, nlohmann::json>>{
           {"protocol_version", std::numeric_limits<std::uint64_t>::max()},
           {"width", -2},
           {"duration_ns", -1},
           {"total_frames", -1}}) {
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
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(estimated)); },
                 "response with estimated duration bypasses exact consistency validation");
  estimated["duration_is_estimated"] = false;
  estimated["total_frames_is_estimated"] = true;
  expect_success([&] { (void)reco::io::detail::decode_probe_response(encode(estimated)); },
                 "response with estimated frame count bypasses exact consistency validation");

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
  exact_response_metadata_must_be_consistent();
  ipc_frame_lengths_are_enforced();
  cbor_nesting_is_bounded_before_parsing();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
