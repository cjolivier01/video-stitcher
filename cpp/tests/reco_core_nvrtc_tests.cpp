#include "reco/core/cuda_backend.hpp"
#include "reco/core/nvrtc_compiler.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__SANITIZE_ADDRESS__)
#define RECO_CORE_TEST_WITH_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RECO_CORE_TEST_WITH_ASAN 1
#endif
#endif

using namespace reco::core;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Actual, typename Expected>
void expect_eq(const Actual& actual, const Expected& expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function> void run_case(std::string_view name, Function&& function) {
  try {
    function();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << name << " threw: " << error.what() << '\n';
    ++failures;
  } catch (...) {
    std::cerr << "FAIL: " << name << " threw an unknown exception\n";
    ++failures;
  }
}

template <typename Function>
void expect_invalid_argument(Function&& function, std::string_view fragment,
                             std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_nvrtc_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const NvrtcError& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_logic_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::logic_error& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

void set_scenario(const char* value) {
#if defined(_WIN32)
  _putenv_s("RECO_FAKE_NVRTC_SCENARIO", value);
#else
  setenv("RECO_FAKE_NVRTC_SCENARIO", value, 1);
#endif
}

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

std::filesystem::path resolve_runfile(std::string_view path) {
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
  const auto logical_path = std::string(workspace) + "/" + std::string(path);
  const auto resolved = std::filesystem::path(runfiles->Rlocation(logical_path));
  if (resolved.empty() || !std::filesystem::is_regular_file(resolved)) {
    throw std::runtime_error(std::string(path) + " runfile not found");
  }
  return resolved;
}

class FakeRuntimeControl {
public:
  explicit FakeRuntimeControl(const std::filesystem::path& path) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(path.string().c_str());
#else
    handle_ = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw std::runtime_error("failed to load fake NVRTC control library");
    }
    reset_ = symbol<void (*)()>("recoFakeNvrtcReset");
    create_count_ = symbol<int (*)()>("recoFakeNvrtcCreateCount");
    destroy_count_ = symbol<int (*)()>("recoFakeNvrtcDestroyCount");
    log_read_count_ = symbol<int (*)()>("recoFakeNvrtcLogReadCount");
    ptx_read_count_ = symbol<int (*)()>("recoFakeNvrtcPtxReadCount");
    compile_count_ = symbol<int (*)()>("recoFakeNvrtcCompileCount");
    maximum_active_compile_count_ = symbol<int (*)()>("recoFakeNvrtcMaximumActiveCompileCount");
    last_architecture_ = symbol<int (*)()>("recoFakeNvrtcLastArchitecture");
  }

  FakeRuntimeControl(const FakeRuntimeControl&) = delete;
  FakeRuntimeControl& operator=(const FakeRuntimeControl&) = delete;

  ~FakeRuntimeControl() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      (void)FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      (void)dlclose(handle_);
    }
#endif
  }

  void reset() const { reset_(); }
  [[nodiscard]] int create_count() const { return create_count_(); }
  [[nodiscard]] int destroy_count() const { return destroy_count_(); }
  [[nodiscard]] int log_read_count() const { return log_read_count_(); }
  [[nodiscard]] int ptx_read_count() const { return ptx_read_count_(); }
  [[nodiscard]] int compile_count() const { return compile_count_(); }
  [[nodiscard]] int maximum_active_compile_count() const { return maximum_active_compile_count_(); }
  [[nodiscard]] int last_architecture() const { return last_architecture_(); }

private:
  template <typename Function> Function symbol(const char* name) {
#if defined(_WIN32)
    auto* value = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    void* value = dlsym(handle_, name);
#endif
    if (value == nullptr) {
      throw std::runtime_error(std::string("missing fake NVRTC control symbol ") + name);
    }
    return reinterpret_cast<Function>(value);
  }

  void* handle_ = nullptr;
  void (*reset_)() = nullptr;
  int (*create_count_)() = nullptr;
  int (*destroy_count_)() = nullptr;
  int (*log_read_count_)() = nullptr;
  int (*ptx_read_count_)() = nullptr;
  int (*compile_count_)() = nullptr;
  int (*maximum_active_compile_count_)() = nullptr;
  int (*last_architecture_)() = nullptr;
};

const std::string kSource = "extern \"C\" __global__ void fake_kernel() {}";

void exact_library_probe(const std::filesystem::path& fake_runtime,
                         const std::filesystem::path& incomplete_runtime,
                         const std::filesystem::path& incomplete_architecture_runtime) {
  set_scenario("success");
  expect_eq(NvrtcCompiler::availability_error(fake_runtime.string()), std::string(),
            "fake runtime availability");
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  expect_eq(compiler.version().major, 12, "NVRTC major version");
  expect_eq(compiler.version().minor, 8, "NVRTC minor version");
  expect_eq(std::string(compiler.library_path()), fake_runtime.string(), "NVRTC loaded path");

  const auto missing_error =
      NvrtcCompiler::availability_error((fake_runtime.string() + ".missing"));
  expect_true(missing_error.find("failed to load NVRTC library") != std::string::npos,
              "missing runtime diagnostic");
  const auto incomplete_error = NvrtcCompiler::availability_error(incomplete_runtime.string());
  expect_true(incomplete_error.find("missing NVRTC symbol nvrtcGetPTX") != std::string::npos,
              "missing symbol diagnostic");
  const auto incomplete_architecture_error =
      NvrtcCompiler::availability_error(incomplete_architecture_runtime.string());
  expect_true(incomplete_architecture_error.find("missing NVRTC symbol nvrtcGetSupportedArchs") !=
                  std::string::npos,
              "missing architecture symbol diagnostic");

  set_scenario("version-error");
  const auto version_error = NvrtcCompiler::availability_error(fake_runtime.string());
  expect_true(version_error.find("nvrtcVersion returned NVRTC error 11") != std::string::npos,
              "version failure diagnostic");
  set_scenario("success");

  expect_invalid_argument([] { (void)NvrtcCompiler::load({}); }, "library path",
                          "empty library path validation");
  expect_invalid_argument([] { (void)NvrtcCompiler::load(std::string("a\0b", 3)); }, "NUL",
                          "library path NUL validation");
}

void supported_architecture_selection(const std::filesystem::path& fake_runtime) {
  set_scenario("success");
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  expect_eq(compiler.supported_architectures(), std::vector<int>({52, 75, 80, 86, 90}),
            "supported architectures are sorted and deduplicated");
  expect_eq(compiler.select_architecture(52), 52, "exact architecture selected");
  expect_eq(compiler.select_architecture(89), 86, "newer device uses highest supported target");
  expect_eq(compiler.select_architecture(99), 90, "newest supported target selected");
  expect_nvrtc_error([&] { (void)compiler.select_architecture(50); }, "compute_50",
                     "device older than all supported targets rejected");
  expect_invalid_argument([&] { (void)compiler.select_architecture(0); }, "major * 10 + minor",
                          "invalid device architecture rejected");

  for (const auto scenario_name :
       {"architecture-count-error", "architecture-count-zero", "architecture-count-oversized",
        "architecture-list-error", "architecture-invalid"}) {
    set_scenario(scenario_name);
    expect_true(!NvrtcCompiler::availability_error(fake_runtime.string()).empty(),
                std::string(scenario_name) + " is rejected during discovery");
  }
  set_scenario("success");
}

void compile_success_and_cleanup(const std::filesystem::path& fake_runtime,
                                 const FakeRuntimeControl& control) {
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  NvrtcCompileOptions options;
  options.values = {
      "--std=c++17",
      "--gpu-architecture=compute_75",
      "--use_fast_math",
  };
  set_scenario("verify-request");
  control.reset();
  const auto result = compiler.compile(kSource, "request.cu", options);
  expect_true(result.ptx.find(".entry fake_kernel") != std::string::npos,
              "fake compilation returns PTX");
  expect_eq(result.log, std::string("fake warning\n"), "fake compilation preserves log");
  expect_true(result.ptx.find('\0') == std::string::npos, "returned PTX strips terminal NUL");
  expect_true(result.log.find('\0') == std::string::npos, "returned log strips terminal NUL");
  expect_eq(control.create_count(), 1, "successful compile creates one program");
  expect_eq(control.destroy_count(), 1, "successful compile destroys one program");
  expect_eq(control.log_read_count(), 1, "successful compile reads one log");
  expect_eq(control.ptx_read_count(), 1, "successful compile reads one PTX output");

  set_scenario("success");
  control.reset();
  const auto default_result = compiler.compile(kSource, "default.cu");
  expect_true(!default_result.ptx.empty(), "default options compile");
  expect_eq(control.destroy_count(), 1, "default compile cleans up");
}

void moved_from_compiler_is_diagnosed(const std::filesystem::path& fake_runtime) {
  set_scenario("success");
  auto source = NvrtcCompiler::load(fake_runtime.string());
  auto compiler = std::move(source);
  expect_eq(compiler.version().major, 12, "moved compiler remains usable");
  expect_logic_error([&] { (void)source.version(); }, "moved-from", "moved-from version access");
  expect_logic_error([&] { (void)source.library_path(); }, "moved-from", "moved-from path access");
  expect_logic_error([&] { (void)source.supported_architectures(); }, "moved-from",
                     "moved-from architecture access");
  expect_logic_error([&] { (void)source.select_architecture(75); }, "moved-from",
                     "moved-from architecture selection");
  expect_logic_error([&] { (void)source.compile(kSource, "moved.cu"); }, "moved-from",
                     "moved-from compile access");
}

void cache_key_covers_the_exact_request(const std::filesystem::path& fake_runtime,
                                        const FakeRuntimeControl& control) {
  set_scenario("success");
  control.reset();
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  (void)compiler.compile(kSource, "cache-key.cu");
  (void)compiler.compile(kSource, "cache-key.cu");
  expect_eq(control.compile_count(), 1, "identical sequential request is cached");

  NvrtcCompileOptions different_options;
  different_options.values = {"--std=c++17", "--use_fast_math"};
  (void)compiler.compile(kSource, "cache-key.cu", different_options);
  (void)compiler.compile(kSource + "\n", "cache-key.cu");
  (void)compiler.compile(kSource, "cache-key-other-name.cu");
  expect_eq(control.compile_count(), 4, "source, source name, and ordered options key the cache");
}

void shared_compiler_supports_concurrent_compiles(const std::filesystem::path& fake_runtime,
                                                  const FakeRuntimeControl& control) {
  constexpr std::size_t kThreadCount = 8;
  set_scenario("slow-success");
  control.reset();

  std::vector<NvrtcCompiler> compilers;
  compilers.reserve(kThreadCount);
  for (std::size_t index = 0; index < kThreadCount; ++index) {
    compilers.push_back(NvrtcCompiler::load(fake_runtime.string()));
  }

  std::array<std::exception_ptr, kThreadCount> errors{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (std::size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      try {
        const auto result = compilers[thread_index].compile(kSource, "single-flight.cu");
        if (result.ptx.empty() || result.log != "fake warning\n") {
          throw std::runtime_error("concurrent compile returned unexpected output");
        }
      } catch (...) {
        errors[thread_index] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    expect_true(error == nullptr, "shared compiler concurrent compile succeeds");
  }
  expect_eq(control.create_count(), 1, "identical concurrent requests create one program");
  expect_eq(control.compile_count(), 1, "identical concurrent requests compile once");
  expect_eq(control.destroy_count(), 1, "single-flight compile cleans up once");
  expect_eq(control.maximum_active_compile_count(), 1, "single-flight has one active compile");
  set_scenario("success");
}

void distinct_compiles_are_serialized(const std::filesystem::path& fake_runtime,
                                      const FakeRuntimeControl& control) {
  constexpr std::size_t kThreadCount = 8;
  set_scenario("slow-success");
  control.reset();
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  std::array<std::exception_ptr, kThreadCount> errors{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (std::size_t index = 0; index < kThreadCount; ++index) {
    threads.emplace_back([&, index] {
      try {
        const auto name = "serialized-" + std::to_string(index) + ".cu";
        if (compiler.compile(kSource, name).ptx.empty()) {
          throw std::runtime_error("serialized compile returned empty PTX");
        }
      } catch (...) {
        errors[index] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    expect_true(error == nullptr, "distinct concurrent compile succeeds");
  }
  expect_eq(control.compile_count(), static_cast<int>(kThreadCount),
            "distinct requests each compile");
  expect_eq(control.maximum_active_compile_count(), 1,
            "distinct NVRTC compilations are process-wide serialized");
  set_scenario("success");
}

void successful_cache_is_bounded(const std::filesystem::path& fake_runtime,
                                 const FakeRuntimeControl& control) {
  set_scenario("success");
  control.reset();
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  for (std::size_t index = 0; index <= kNvrtcCompileCacheMaximumEntries; ++index) {
    (void)compiler.compile(kSource, "entry-cap-" + std::to_string(index) + ".cu");
  }
  expect_eq(control.compile_count(), static_cast<int>(kNvrtcCompileCacheMaximumEntries + 1U),
            "unique requests fill and exceed the entry cap");
  (void)compiler.compile(kSource, "entry-cap-0.cu");
  expect_eq(control.compile_count(), static_cast<int>(kNvrtcCompileCacheMaximumEntries + 2U),
            "least-recent entry is evicted");

  set_scenario("large-cache-ptx");
  control.reset();
  for (std::size_t index = 0; index < 3; ++index) {
    (void)compiler.compile(kSource, "byte-cap-" + std::to_string(index) + ".cu");
  }
  (void)compiler.compile(kSource, "byte-cap-0.cu");
  expect_eq(control.compile_count(), 4, "accounted byte cap evicts the oldest large result");
  expect_eq(control.maximum_active_compile_count(), 1, "cache misses remain serialized");
  set_scenario("success");
}

void failures_are_retried(const std::filesystem::path& fake_runtime,
                          const FakeRuntimeControl& control) {
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  control.reset();
  set_scenario("compile-error");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "failure-retry.cu"); },
                     "synthetic compile failure", "first compile failure is reported");
  set_scenario("success");
  const auto retried = compiler.compile(kSource, "failure-retry.cu");
  expect_true(!retried.ptx.empty(), "failed request can be retried successfully");
  (void)compiler.compile(kSource, "failure-retry.cu");
  expect_eq(control.compile_count(), 2, "failure is not cached and later success is cached");
}

void input_validation_precedes_runtime(const std::filesystem::path& fake_runtime,
                                       const FakeRuntimeControl& control) {
  set_scenario("success");
  auto compiler = NvrtcCompiler::load(fake_runtime.string());
  control.reset();
  expect_invalid_argument([&] { (void)compiler.compile({}, "input.cu"); }, "source",
                          "empty source validation");
  expect_invalid_argument([&] { (void)compiler.compile(std::string("a\0b", 3), "input.cu"); },
                          "NUL", "source NUL validation");
  expect_invalid_argument(
      [&] { (void)compiler.compile(std::string(kNvrtcMaximumSourceBytes + 1, 'x'), "input.cu"); },
      "exceeds", "source size validation");
  expect_invalid_argument([&] { (void)compiler.compile(kSource, {}); }, "source name",
                          "empty source name validation");
  expect_invalid_argument([&] { (void)compiler.compile(kSource, std::string("a\0b", 3)); }, "NUL",
                          "source name NUL validation");
  expect_invalid_argument(
      [&] { (void)compiler.compile(kSource, std::string(kNvrtcMaximumSourceNameBytes + 1, 'n')); },
      "exceeds", "source name size validation");

  NvrtcCompileOptions empty_option;
  empty_option.values = {""};
  expect_invalid_argument([&] { (void)compiler.compile(kSource, "input.cu", empty_option); },
                          "option", "empty option validation");
  NvrtcCompileOptions nul_option;
  nul_option.values = {std::string("--a\0b", 5)};
  expect_invalid_argument([&] { (void)compiler.compile(kSource, "input.cu", nul_option); }, "NUL",
                          "option NUL validation");
  NvrtcCompileOptions long_option;
  long_option.values = {std::string(kNvrtcMaximumOptionBytes + 1, 'o')};
  expect_invalid_argument([&] { (void)compiler.compile(kSource, "input.cu", long_option); },
                          "exceeds", "individual option size validation");
  NvrtcCompileOptions many_options;
  many_options.values.assign(kNvrtcMaximumOptionCount + 1, "--x");
  expect_invalid_argument([&] { (void)compiler.compile(kSource, "input.cu", many_options); },
                          "count", "option count validation");
  NvrtcCompileOptions total_options;
  total_options.values.assign(kNvrtcMaximumTotalOptionBytes / kNvrtcMaximumOptionBytes + 1,
                              std::string(kNvrtcMaximumOptionBytes, 'o'));
  expect_invalid_argument([&] { (void)compiler.compile(kSource, "input.cu", total_options); },
                          "in total", "aggregate option size validation");
  expect_eq(control.create_count(), 0, "invalid input never reaches NVRTC");
}

void runtime_failures_are_bounded(const std::filesystem::path& fake_runtime,
                                  const FakeRuntimeControl& control) {
  auto compiler = NvrtcCompiler::load(fake_runtime.string());

  set_scenario("create-error");
  control.reset();
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "create-error.cu"); },
                     "nvrtcCreateProgram returned NVRTC error 3", "create error propagation");
  expect_eq(control.destroy_count(), 1, "partially created program is destroyed");

  set_scenario("create-null");
  control.reset();
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "create-null.cu"); },
                     "without returning a program", "null program validation");
  expect_eq(control.destroy_count(), 0, "null program is not destroyed");

  set_scenario("compile-error");
  control.reset();
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "compile-error.cu"); },
                     "synthetic compile failure", "compile log propagation");
  expect_eq(control.destroy_count(), 1, "failed compile destroys program");

  set_scenario("destroy-error");
  control.reset();
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "destroy-error.cu"); },
                     "nvrtcDestroyProgram returned NVRTC error 11", "destroy error propagation");
  expect_eq(control.destroy_count(), 1, "destroy guard retries cleanup after reported failure");

  set_scenario("compile-error-oversized-log");
  control.reset();
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "compile-oversized-log.cu"); },
                     "compiler log omitted", "failed compile bounds oversized log");
  expect_eq(control.log_read_count(), 0, "oversized failure log is not read");

  set_scenario("oversized-log");
  control.reset();
  const auto oversized_log = compiler.compile(kSource, "oversized-log.cu");
  expect_true(oversized_log.log.find("compiler log omitted") != std::string::npos,
              "successful compile reports omitted oversized log");
  expect_eq(control.log_read_count(), 0, "oversized success log is not read");

  set_scenario("log-size-error");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "log-size-error.cu"); },
                     "nvrtcGetProgramLogSize returned NVRTC error 11",
                     "log size error propagation");
  set_scenario("log-get-error");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "log-get-error.cu"); },
                     "nvrtcGetProgramLog returned NVRTC error 11", "log read error propagation");
  set_scenario("ptx-size-error");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "ptx-size-error.cu"); },
                     "nvrtcGetPTXSize returned NVRTC error 11", "PTX size error propagation");

  set_scenario("oversized-ptx");
  control.reset();
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "oversized-ptx.cu"); },
                     "exceeds 67108864 bytes", "oversized PTX validation");
  expect_eq(control.ptx_read_count(), 0, "oversized PTX is not read");
  set_scenario("empty-ptx");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "empty-ptx.cu"); }, "empty PTX",
                     "empty PTX validation");
  set_scenario("ptx-get-error");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "ptx-get-error.cu"); },
                     "nvrtcGetPTX returned NVRTC error 11", "PTX read error propagation");
  set_scenario("interior-nul-ptx");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "interior-nul-ptx.cu"); },
                     "interior NUL", "interior PTX NUL validation");
  set_scenario("unterminated-ptx");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "unterminated-ptx.cu"); },
                     "terminating NUL", "unterminated PTX validation");
  set_scenario("truncated-ptx");
  expect_nvrtc_error([&] { (void)compiler.compile(kSource, "truncated-ptx.cu"); },
                     "terminating NUL", "truncated PTX validation");
  set_scenario("success");
}

void hardware_smoke_if_available() {
#if defined(RECO_CORE_TEST_WITH_ASAN)
  if (!require_cuda()) {
    std::cout << "SKIP: hardware NVRTC smoke disabled under ASan unless explicitly required\n";
    return;
  }
#endif
  const auto nvrtc_error = NvrtcCompiler::availability_error();
  const auto cuda_error = CudaBackend::availability_error();
  if (!nvrtc_error.empty() || !cuda_error.empty()) {
    const auto diagnostic = std::string("NVRTC=") +
                            (nvrtc_error.empty() ? "available" : nvrtc_error) +
                            " CUDA=" + (cuda_error.empty() ? "available" : cuda_error);
    if (require_cuda()) {
      throw std::runtime_error("required hardware NVRTC smoke unavailable: " + diagnostic);
    }
    std::cout << "SKIP: hardware NVRTC smoke unavailable: " << diagnostic << '\n';
    return;
  }

  constexpr std::string_view kHardwareSource = R"cuda(
extern "C" __global__ void write_value(unsigned int* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *output = 0x1234abcdU;
  }
}
)cuda";
  auto backend = CudaBackend::create();
  auto compiler = NvrtcCompiler::create();
  const auto capability = backend.compute_capability();
  const auto architecture = compiler.select_architecture(capability.major * 10 + capability.minor);
  NvrtcCompileOptions options;
  options.values = {"--std=c++17", "--gpu-architecture=compute_" + std::to_string(architecture)};
  const auto compiled = compiler.compile(kHardwareSource, "reco_core_nvrtc_smoke.cu", options);
  auto output = backend.allocate(sizeof(std::uint32_t));
  backend.memset_d8(output, 0);
  auto module = backend.load_module_from_ptx(compiled.ptx);
  auto kernel = module.load_kernel("write_value");
  CudaDevicePtr output_ptr = output.ptr();
  std::array<void*, 1> arguments{&output_ptr};
  kernel.launch({.grid = {1, 1, 1}, .block = {1, 1, 1}}, arguments);
  kernel.synchronize();
  const auto bytes = backend.copy_to_host(output);
  std::uint32_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(value));
  expect_eq(value, std::uint32_t{0x1234abcdU}, "hardware NVRTC kernel result");
  std::cout << "hardware NVRTC smoke executed\n";
}

} // namespace

int main() {
  const auto fake_runtime = resolve_runfile("cpp/tests/libreco_core_fake_nvrtc_runtime.so");
  const auto incomplete_runtime =
      resolve_runfile("cpp/tests/libreco_core_fake_nvrtc_incomplete_runtime.so");
  const auto incomplete_architecture_runtime =
      resolve_runfile("cpp/tests/libreco_core_fake_nvrtc_incomplete_arch_runtime.so");
  FakeRuntimeControl control(fake_runtime);

  run_case("exact library probe", [&] {
    exact_library_probe(fake_runtime, incomplete_runtime, incomplete_architecture_runtime);
  });
  run_case("supported architecture selection",
           [&] { supported_architecture_selection(fake_runtime); });
  run_case("compile success and cleanup",
           [&] { compile_success_and_cleanup(fake_runtime, control); });
  run_case("moved-from compiler diagnostics",
           [&] { moved_from_compiler_is_diagnosed(fake_runtime); });
  run_case("exact compile cache key",
           [&] { cache_key_covers_the_exact_request(fake_runtime, control); });
  run_case("shared compiler concurrent compiles",
           [&] { shared_compiler_supports_concurrent_compiles(fake_runtime, control); });
  run_case("distinct compiles are serialized",
           [&] { distinct_compiles_are_serialized(fake_runtime, control); });
  run_case("successful cache is bounded",
           [&] { successful_cache_is_bounded(fake_runtime, control); });
  run_case("failures are retried", [&] { failures_are_retried(fake_runtime, control); });
  run_case("input validation", [&] { input_validation_precedes_runtime(fake_runtime, control); });
  run_case("bounded runtime failures",
           [&] { runtime_failures_are_bounded(fake_runtime, control); });
  run_case("hardware smoke", hardware_smoke_if_available);

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
