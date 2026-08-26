#include "gpu_video_probe_protocol.hpp"

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
  const auto base = nlohmann::json{{"protocol_version", 1U},
                                   {"ok", true},
                                   {"width", 1920U},
                                   {"height", 1080U},
                                   {"fps_numerator", 30U},
                                   {"fps_denominator", 1U},
                                   {"duration_ns", 1'000'000'000ULL},
                                   {"total_frames", 30ULL},
                                   {"duration_is_estimated", false},
                                   {"total_frames_is_estimated", false}};
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

  auto impossible_dimensions = base;
  impossible_dimensions["width"] = std::numeric_limits<std::uint32_t>::max();
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(impossible_dimensions)); },
      "invalid metadata", "impossible GPU dimensions are rejected");

  auto impossible_frame_count = base;
  impossible_frame_count["total_frames"] = std::numeric_limits<std::uint64_t>::max();
  expect_probe_error(
      [&] { (void)reco::io::detail::decode_probe_response(encode(impossible_frame_count)); },
      "invalid metadata", "impossible average frame counts are rejected");
}

} // namespace

int main() {
  request_numeric_domains_are_enforced();
  response_numeric_domains_are_enforced();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
