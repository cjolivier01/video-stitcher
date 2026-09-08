#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "rules_cc/cc/runfiles/runfiles.h"

namespace reco::tests {

inline std::filesystem::path find_runfile(std::string_view filename) {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
  const auto logical_path = std::string(workspace) + "/cpp/tests/fixtures/" + std::string(filename);
  const auto resolved = std::filesystem::path(runfiles->Rlocation(logical_path));
  if (!resolved.empty() && std::filesystem::is_regular_file(resolved)) {
    return resolved;
  }
  throw std::runtime_error("fixture runfile not found: " + std::string(filename));
}

inline std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<std::uint8_t> decoded;
  std::uint32_t accumulator = 0;
  int bits = -8;
  for (const unsigned char character : encoded) {
    if (std::isspace(character) != 0) {
      continue;
    }
    if (character == '=') {
      break;
    }
    const auto value = alphabet.find(static_cast<char>(character));
    if (value == std::string_view::npos) {
      throw std::runtime_error("base64 fixture contains an invalid character");
    }
    accumulator = (accumulator << 6U) | static_cast<std::uint32_t>(value);
    bits += 6;
    if (bits >= 0) {
      decoded.push_back(static_cast<std::uint8_t>((accumulator >> bits) & 0xff));
      bits -= 8;
    }
  }
  return decoded;
}

inline void materialize_base64_fixture(const std::filesystem::path& encoded_path,
                                       const std::filesystem::path& output_path) {
  std::ifstream input(encoded_path);
  if (!input) {
    throw std::runtime_error("cannot open base64 fixture: " + encoded_path.string());
  }
  const std::string encoded((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  const auto decoded = decode_base64(encoded);
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(decoded.data()),
               static_cast<std::streamsize>(decoded.size()));
  if (!output) {
    throw std::runtime_error("cannot materialize fixture: " + output_path.string());
  }
}

} // namespace reco::tests
