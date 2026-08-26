#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

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
    const std::string ignored_request(std::istreambuf_iterator<char>(std::cin), {});
    (void)ignored_request;
    if (std::strcmp(scenario, "crash") == 0) {
      return 3;
    }
    if (std::strcmp(scenario, "malformed-response") == 0) {
      std::cout << "not CBOR";
      return EXIT_SUCCESS;
    }
    nlohmann::json response;
    if (std::strcmp(scenario, "wrong-version") == 0) {
      response = {{"protocol_version", 2}, {"ok", false}};
    } else if (std::strcmp(scenario, "valid-metadata") == 0 ||
               std::strcmp(scenario, "invalid-metadata") == 0) {
      response = {{"protocol_version", 1},
                  {"ok", true},
                  {"width", std::strcmp(scenario, "valid-metadata") == 0 ? 854 : 853},
                  {"height", 480},
                  {"fps_numerator", 30},
                  {"fps_denominator", 1},
                  {"duration_ns", 1'000'000'000},
                  {"total_frames", 30},
                  {"duration_is_estimated", false},
                  {"total_frames_is_estimated", false}};
    }
    const auto encoded = nlohmann::json::to_cbor(response);
    std::cout.write(reinterpret_cast<const char*>(encoded.data()),
                    static_cast<std::streamsize>(encoded.size()));
    return EXIT_SUCCESS;
  }
  std::this_thread::sleep_for(std::chrono::seconds(30));
  return EXIT_SUCCESS;
}
