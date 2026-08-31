#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#define RECO_FAKE_NVRTC_EXPORT extern "C" __declspec(dllexport)
#else
#define RECO_FAKE_NVRTC_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

using NvrtcProgram = void*;
using NvrtcResult = int;

constexpr NvrtcResult kSuccess = 0;
constexpr NvrtcResult kInvalidInput = 3;
constexpr NvrtcResult kCompilationError = 6;
constexpr NvrtcResult kInternalError = 11;
constexpr std::size_t kOversizedLogBytes = 5U * 1024U * 1024U;
constexpr std::size_t kOversizedPtxBytes = 65U * 1024U * 1024U;
constexpr std::size_t kLargeCachePtxBytes = 6U * 1024U * 1024U;

struct Program {
  std::string source;
  std::string source_name;
  std::string log;
  std::string ptx;
};

std::atomic<int> create_count{0};
std::atomic<int> destroy_count{0};
std::atomic<int> destroy_attempt_count{0};
std::atomic<int> log_read_count{0};
std::atomic<int> ptx_read_count{0};
std::atomic<int> compile_count{0};
std::atomic<int> active_compile_count{0};
std::atomic<int> maximum_active_compile_count{0};
std::atomic<int> last_architecture{-1};

std::string_view scenario() {
  const char* value = std::getenv("RECO_FAKE_NVRTC_SCENARIO");
  return value == nullptr ? std::string_view("success") : std::string_view(value);
}

Program* program_cast(NvrtcProgram program) { return static_cast<Program*>(program); }

void record_maximum_active(int active) {
  auto maximum = maximum_active_compile_count.load();
  while (active > maximum && !maximum_active_compile_count.compare_exchange_weak(maximum, active)) {
  }
}

void record_architecture(int option_count, const char* const* options) {
  constexpr std::string_view kPrefix = "--gpu-architecture=compute_";
  for (int index = 0; index < option_count; ++index) {
    if (options[index] == nullptr) {
      continue;
    }
    const std::string_view option(options[index]);
    if (!option.starts_with(kPrefix)) {
      continue;
    }
    int architecture = -1;
    const auto digits = option.substr(kPrefix.size());
    const auto conversion =
        std::from_chars(digits.data(), digits.data() + digits.size(), architecture);
    if (conversion.ec == std::errc{} && conversion.ptr == digits.data() + digits.size()) {
      last_architecture = architecture;
    }
  }
}

} // namespace

RECO_FAKE_NVRTC_EXPORT void recoFakeNvrtcReset() {
  create_count = 0;
  destroy_count = 0;
  destroy_attempt_count = 0;
  log_read_count = 0;
  ptx_read_count = 0;
  compile_count = 0;
  active_compile_count = 0;
  maximum_active_compile_count = 0;
  last_architecture = -1;
}

RECO_FAKE_NVRTC_EXPORT int recoFakeNvrtcCreateCount() { return create_count.load(); }
RECO_FAKE_NVRTC_EXPORT int recoFakeNvrtcDestroyCount() { return destroy_count.load(); }
RECO_FAKE_NVRTC_EXPORT int recoFakeNvrtcLogReadCount() { return log_read_count.load(); }
RECO_FAKE_NVRTC_EXPORT int recoFakeNvrtcPtxReadCount() { return ptx_read_count.load(); }
RECO_FAKE_NVRTC_EXPORT int recoFakeNvrtcCompileCount() { return compile_count.load(); }
RECO_FAKE_NVRTC_EXPORT int recoFakeNvrtcMaximumActiveCompileCount() {
  return maximum_active_compile_count.load();
}
RECO_FAKE_NVRTC_EXPORT int recoFakeNvrtcLastArchitecture() { return last_architecture.load(); }

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcVersion(int* major, int* minor) {
  if (scenario() == "version-error") {
    return kInternalError;
  }
  if (major == nullptr || minor == nullptr) {
    return kInvalidInput;
  }
  *major = 12;
  *minor = 8;
  return kSuccess;
}

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcGetNumSupportedArchs(int* count) {
  if (count == nullptr) {
    return kInvalidInput;
  }
  if (scenario() == "architecture-count-error") {
    return kInternalError;
  }
  if (scenario() == "architecture-count-zero") {
    *count = 0;
    return kSuccess;
  }
  if (scenario() == "architecture-count-oversized") {
    *count = 129;
    return kSuccess;
  }
  *count = 6;
  return kSuccess;
}

#if !defined(RECO_FAKE_NVRTC_INCOMPLETE_ARCHS)
RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcGetSupportedArchs(int* architectures) {
  if (architectures == nullptr) {
    return kInvalidInput;
  }
  if (scenario() == "architecture-list-error") {
    return kInternalError;
  }
  const int values[] = {90, 52, 86, 75, 80, 86};
  std::memcpy(architectures, values, sizeof(values));
  if (scenario() == "architecture-invalid") {
    architectures[2] = -1;
  }
  return kSuccess;
}
#endif

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcCreateProgram(NvrtcProgram* output, const char* source,
                                                      const char* source_name, int header_count,
                                                      const char* const*, const char* const*) {
  ++create_count;
  if (output == nullptr || source == nullptr || source_name == nullptr || header_count != 0) {
    return kInvalidInput;
  }
  auto* program = new Program;
  program->source = source;
  program->source_name = source_name;
  program->log = "fake warning\n";
  program->ptx =
      ".version 7.0\n.target sm_75\n.address_size 64\n.visible .entry fake_kernel() { ret; }\n";
  if (scenario() == "large-cache-ptx") {
    program->ptx.assign(kLargeCachePtxBytes, 'x');
  }
  *output = program;
  if (scenario() == "create-error") {
    return kInvalidInput;
  }
  if (scenario() == "create-null") {
    delete program;
    *output = nullptr;
  }
  return kSuccess;
}

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcDestroyProgram(NvrtcProgram* input) {
  if (input == nullptr || *input == nullptr) {
    return kInvalidInput;
  }
  if (++destroy_attempt_count == 1 && scenario() == "destroy-error") {
    return kInternalError;
  }
  delete program_cast(*input);
  *input = nullptr;
  ++destroy_count;
  return kSuccess;
}

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcCompileProgram(NvrtcProgram input, int option_count,
                                                       const char* const* options) {
  auto* program = program_cast(input);
  if (program == nullptr || option_count < 0 || (option_count != 0 && options == nullptr)) {
    return kInvalidInput;
  }
  ++compile_count;
  const auto active = ++active_compile_count;
  record_maximum_active(active);
  struct ActiveCompileGuard {
    ~ActiveCompileGuard() { --active_compile_count; }
  } active_guard;
  record_architecture(option_count, options);
  if (scenario() == "slow-success") {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  if (scenario() == "verify-request") {
    constexpr std::string_view kExpectedSource = "extern \"C\" __global__ void fake_kernel() {}";
    constexpr const char* kExpectedOptions[] = {
        "--std=c++17",
        "--gpu-architecture=compute_75",
        "--use_fast_math",
    };
    bool matches = program->source == kExpectedSource && program->source_name == "request.cu" &&
                   option_count == 3;
    for (int index = 0; matches && index < option_count; ++index) {
      matches =
          options[index] != nullptr && std::string_view(options[index]) == kExpectedOptions[index];
    }
    if (!matches) {
      program->log = "fake request mismatch";
      return kCompilationError;
    }
  }
  if (scenario() == "compile-error" || scenario() == "compile-error-oversized-log") {
    program->log = "synthetic compile failure";
    return kCompilationError;
  }
  return kSuccess;
}

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcGetProgramLogSize(NvrtcProgram input, std::size_t* size) {
  if (input == nullptr || size == nullptr) {
    return kInvalidInput;
  }
  if (scenario() == "log-size-error") {
    return kInternalError;
  }
  if (scenario() == "oversized-log" || scenario() == "compile-error-oversized-log") {
    *size = kOversizedLogBytes;
    return kSuccess;
  }
  *size = program_cast(input)->log.size() + 1;
  return kSuccess;
}

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcGetProgramLog(NvrtcProgram input, char* output) {
  ++log_read_count;
  if (input == nullptr || output == nullptr) {
    return kInvalidInput;
  }
  if (scenario() == "log-get-error") {
    return kInternalError;
  }
  const auto& log = program_cast(input)->log;
  std::memcpy(output, log.c_str(), log.size() + 1);
  return kSuccess;
}

RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcGetPTXSize(NvrtcProgram input, std::size_t* size) {
  if (input == nullptr || size == nullptr) {
    return kInvalidInput;
  }
  if (scenario() == "ptx-size-error") {
    return kInternalError;
  }
  if (scenario() == "oversized-ptx") {
    *size = kOversizedPtxBytes;
    return kSuccess;
  }
  if (scenario() == "empty-ptx") {
    *size = 0;
    return kSuccess;
  }
  if (scenario() == "interior-nul-ptx") {
    *size = 4;
    return kSuccess;
  }
  if (scenario() == "unterminated-ptx") {
    *size = 4;
    return kSuccess;
  }
  if (scenario() == "truncated-ptx") {
    *size = program_cast(input)->ptx.size();
    return kSuccess;
  }
  *size = program_cast(input)->ptx.size() + 1;
  return kSuccess;
}

#if !defined(RECO_FAKE_NVRTC_INCOMPLETE)
RECO_FAKE_NVRTC_EXPORT NvrtcResult nvrtcGetPTX(NvrtcProgram input, char* output) {
  ++ptx_read_count;
  if (input == nullptr || output == nullptr) {
    return kInvalidInput;
  }
  if (scenario() == "ptx-get-error") {
    return kInternalError;
  }
  if (scenario() == "interior-nul-ptx") {
    const char invalid_ptx[] = {'a', '\0', 'b', '\0'};
    std::memcpy(output, invalid_ptx, sizeof(invalid_ptx));
    return kSuccess;
  }
  if (scenario() == "unterminated-ptx") {
    const char invalid_ptx[] = {'a', 'b', 'c', 'd'};
    std::memcpy(output, invalid_ptx, sizeof(invalid_ptx));
    return kSuccess;
  }
  const auto& ptx = program_cast(input)->ptx;
  if (scenario() == "truncated-ptx") {
    std::memcpy(output, ptx.data(), ptx.size());
    return kSuccess;
  }
  std::memcpy(output, ptx.c_str(), ptx.size() + 1);
  return kSuccess;
}
#endif

RECO_FAKE_NVRTC_EXPORT const char* nvrtcGetErrorString(NvrtcResult result) {
  switch (result) {
  case kSuccess:
    return "NVRTC_SUCCESS";
  case kInvalidInput:
    return "NVRTC_ERROR_INVALID_INPUT";
  case kCompilationError:
    return "NVRTC_ERROR_COMPILATION";
  case kInternalError:
    return "NVRTC_ERROR_INTERNAL_ERROR";
  default:
    return "NVRTC_ERROR_UNKNOWN";
  }
}
