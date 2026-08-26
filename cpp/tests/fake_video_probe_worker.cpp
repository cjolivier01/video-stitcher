#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <filesystem>
#include <unistd.h>
#endif

namespace {

constexpr std::size_t kMaximumFrameBytes = 256U * 1024U;
constexpr std::size_t kFrameHeaderBytes = sizeof(std::uint32_t);
using FrameHeader = std::array<char, kFrameHeaderBytes>;

FrameHeader frame_header(std::size_t size) {
  const auto value = static_cast<std::uint32_t>(size);
  return {static_cast<char>((value >> 24U) & 0xFFU), static_cast<char>((value >> 16U) & 0xFFU),
          static_cast<char>((value >> 8U) & 0xFFU), static_cast<char>(value & 0xFFU)};
}

bool read_exact(char* destination, std::size_t size) {
  std::cin.read(destination, static_cast<std::streamsize>(size));
  return std::cin.gcount() == static_cast<std::streamsize>(size);
}

bool read_request() {
  FrameHeader header{};
  if (!read_exact(header.data(), header.size())) {
    return false;
  }
  const auto byte = [](char value) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(value));
  };
  const auto size = (byte(header[0]) << 24U) | (byte(header[1]) << 16U) | (byte(header[2]) << 8U) |
                    byte(header[3]);
  if (size == 0 || size > kMaximumFrameBytes) {
    return false;
  }
  std::string request(size, '\0');
  return read_exact(request.data(), request.size());
}

void write_frame(std::string_view payload) {
  const auto header = frame_header(payload.size());
  std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::cout.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  constexpr int kExpectedArguments = 2;
#else
  constexpr int kExpectedArguments = 3;
#endif
  if (argc != kExpectedArguments || std::strcmp(argv[1], "--reco-video-probe-worker") != 0) {
    return EXIT_FAILURE;
  }
#if defined(_WIN32)
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
    return EXIT_FAILURE;
  }
#endif
  const char* scenario = std::getenv("RECO_FAKE_PROBE_WORKER_SCENARIO");
#if defined(__linux__)
  if (scenario != nullptr && std::strcmp(scenario, "descriptor-isolation") == 0) {
    const char* forbidden_path = std::getenv("RECO_FAKE_PROBE_FORBIDDEN_PATH");
    if (forbidden_path == nullptr || forbidden_path[0] == '\0') {
      return EXIT_FAILURE;
    }
    std::error_code directory_error;
    for (const auto& descriptor :
         std::filesystem::directory_iterator("/proc/self/fd", directory_error)) {
      std::error_code link_error;
      if (std::filesystem::read_symlink(descriptor.path(), link_error) == forbidden_path) {
        return 4;
      }
    }
  }
#endif
  if (scenario != nullptr && std::strcmp(scenario, "close-input") == 0) {
#if defined(_WIN32)
    CloseHandle(GetStdHandle(STD_INPUT_HANDLE));
#else
    close(STDIN_FILENO);
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return EXIT_SUCCESS;
  }
  if (scenario != nullptr && std::strcmp(scenario, "block-input") != 0) {
    if (!read_request()) {
      return EXIT_FAILURE;
    }
    if (std::strcmp(scenario, "crash") == 0) {
      return 3;
    }
    if (std::strcmp(scenario, "malformed-response") == 0) {
      write_frame("not CBOR");
      return EXIT_SUCCESS;
    }
    if (std::strcmp(scenario, "deep-response") == 0) {
      std::string nested(4'096, static_cast<char>(0x81));
      nested.push_back(static_cast<char>(0xF6));
      write_frame(nested);
      return EXIT_SUCCESS;
    }
    if (std::strcmp(scenario, "truncated-frame") == 0) {
      const auto header = frame_header(16);
      std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
      std::cout << "short";
      return EXIT_SUCCESS;
    }
    if (std::strcmp(scenario, "oversized-frame") == 0) {
      const auto header = frame_header(kMaximumFrameBytes + 1U);
      std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
      return EXIT_SUCCESS;
    }
    nlohmann::json response;
    if (std::strcmp(scenario, "wrong-version") == 0) {
      response = {{"protocol_version", 2}, {"ok", false}};
    } else if (std::strcmp(scenario, "wrapped-version") == 0) {
      response = {{"protocol_version", std::numeric_limits<std::uint64_t>::max()}, {"ok", false}};
    } else if (std::strcmp(scenario, "valid-metadata") == 0 ||
#if defined(__linux__)
               std::strcmp(scenario, "descriptor-isolation") == 0 ||
#endif
               std::strcmp(scenario, "invalid-metadata") == 0 ||
               std::strcmp(scenario, "negative-metadata") == 0 ||
               std::strcmp(scenario, "oversized-metadata") == 0) {
      const bool negative = std::strcmp(scenario, "negative-metadata") == 0;
      const bool oversized = std::strcmp(scenario, "oversized-metadata") == 0;
      response = {{"protocol_version", 1},
                  {"ok", true},
                  {"width", negative ? nlohmann::json(-2)
                            : oversized
                                ? nlohmann::json(std::numeric_limits<std::uint64_t>::max())
                                : nlohmann::json(
                                      std::strcmp(scenario, "valid-metadata") == 0 ||
#if defined(__linux__)
                                              std::strcmp(scenario, "descriptor-isolation") == 0 ||
#endif
                                              false
                                          ? 854
                                          : 853)},
                  {"height", 480},
                  {"fps_numerator", 30},
                  {"fps_denominator", 1},
                  {"duration_ns", negative ? nlohmann::json(-1) : nlohmann::json(1'000'000'000)},
                  {"total_frames", negative ? nlohmann::json(-1) : nlohmann::json(30)},
                  {"duration_is_estimated", false},
                  {"total_frames_is_estimated", false}};
    }
    const auto encoded = nlohmann::json::to_cbor(response);
    write_frame({reinterpret_cast<const char*>(encoded.data()), encoded.size()});
    if (std::strcmp(scenario, "trailing-response") == 0) {
      std::cout << "trailing";
    }
    return EXIT_SUCCESS;
  }
  std::this_thread::sleep_for(std::chrono::seconds(30));
  return EXIT_SUCCESS;
}
