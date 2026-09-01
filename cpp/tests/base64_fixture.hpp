#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace reco::tests {

inline std::filesystem::path find_runfile(std::string_view filename) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || runfiles[0] == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    if (entry.path().filename() == filename) {
      return entry.path();
    }
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
