#include "nvrtc_compiler.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reco::calibrate::detail {
namespace {

enum class NvrtcResult : int {};
using NvrtcProgram = void*;

class DynamicLibrary {
public:
  explicit DynamicLibrary(const std::vector<const char*>& names) {
    std::string attempted;
    for (const char* name : names) {
#if defined(_WIN32)
      handle_ = LoadLibraryA(name);
#else
      handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
      if (handle_ != nullptr) {
        return;
      }
      if (!attempted.empty()) {
        attempted += ", ";
      }
      attempted += name;
    }
    throw std::runtime_error("failed to load NVRTC library; tried " + attempted);
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  DynamicLibrary(DynamicLibrary&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  ~DynamicLibrary() { reset(); }

  template <typename Function> [[nodiscard]] Function symbol(const char* name) const {
#if defined(_WIN32)
    auto* value = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* value = dlsym(handle_, name);
#endif
    if (value == nullptr) {
      throw std::runtime_error(std::string("missing NVRTC symbol ") + name);
    }
    return reinterpret_cast<Function>(value);
  }

private:
  void reset() noexcept {
    if (handle_ == nullptr) {
      return;
    }
#if defined(_WIN32)
    (void)FreeLibrary(static_cast<HMODULE>(handle_));
#else
    (void)dlclose(handle_);
#endif
    handle_ = nullptr;
  }

  void* handle_ = nullptr;
};

void validate_text(std::string_view value, const char* label) {
  if (value.empty()) {
    throw std::invalid_argument(std::string("NVRTC ") + label + " must be non-empty");
  }
  if (std::find(value.begin(), value.end(), '\0') != value.end()) {
    throw std::invalid_argument(std::string("NVRTC ") + label + " must not contain NUL bytes");
  }
}

} // namespace

struct NvrtcCompiler::Impl {
  Impl()
#if defined(_WIN32)
      : library({"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"}) {
#else
      : library({"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"}) {
#endif
    create_program = library.symbol<decltype(create_program)>("nvrtcCreateProgram");
    destroy_program = library.symbol<decltype(destroy_program)>("nvrtcDestroyProgram");
    compile_program = library.symbol<decltype(compile_program)>("nvrtcCompileProgram");
    get_ptx_size = library.symbol<decltype(get_ptx_size)>("nvrtcGetPTXSize");
    get_ptx = library.symbol<decltype(get_ptx)>("nvrtcGetPTX");
    get_program_log_size = library.symbol<decltype(get_program_log_size)>("nvrtcGetProgramLogSize");
    get_program_log = library.symbol<decltype(get_program_log)>("nvrtcGetProgramLog");
    get_error_string = library.symbol<decltype(get_error_string)>("nvrtcGetErrorString");
  }

  [[nodiscard]] std::string compile(std::string_view source, std::string_view source_name) const {
    validate_text(source, "source");
    validate_text(source_name, "source name");
    const std::string source_text(source);
    const std::string source_name_text(source_name);
    NvrtcProgram program = nullptr;
    check("nvrtcCreateProgram", create_program(&program, source_text.c_str(),
                                               source_name_text.c_str(), 0, nullptr, nullptr));
    struct ProgramGuard {
      NvrtcProgram value = nullptr;
      NvrtcResult (*destroy)(NvrtcProgram*) = nullptr;
      ~ProgramGuard() {
        if (value != nullptr) {
          (void)destroy(&value);
        }
      }
    } guard{program, destroy_program};

    const char* options[] = {"--std=c++11", "--fmad=false"};
    const auto result = compile_program(program, 2, options);
    if (static_cast<int>(result) != 0) {
      throw std::runtime_error("nvrtcCompileProgram failed for " + source_name_text + ": " +
                               program_log(program));
    }

    std::size_t ptx_size = 0;
    check("nvrtcGetPTXSize", get_ptx_size(program, &ptx_size));
    if (ptx_size == 0) {
      throw std::runtime_error("NVRTC produced empty PTX for " + source_name_text);
    }
    std::string ptx(ptx_size, '\0');
    check("nvrtcGetPTX", get_ptx(program, ptx.data()));
    return ptx;
  }

  void check(const char* function, NvrtcResult result) const {
    if (static_cast<int>(result) == 0) {
      return;
    }
    const char* detail = get_error_string != nullptr ? get_error_string(result) : "unknown";
    throw std::runtime_error(std::string(function) + " returned NVRTC error " + detail);
  }

  [[nodiscard]] std::string program_log(NvrtcProgram program) const {
    std::size_t size = 0;
    if (static_cast<int>(get_program_log_size(program, &size)) != 0 || size == 0) {
      return "no compiler log";
    }
    std::string log(size, '\0');
    if (static_cast<int>(get_program_log(program, log.data())) != 0) {
      return "failed to read compiler log";
    }
    while (!log.empty() && log.back() == '\0') {
      log.pop_back();
    }
    return log;
  }

  DynamicLibrary library;
  NvrtcResult (*create_program)(NvrtcProgram*, const char*, const char*, int, const char* const*,
                                const char* const*) = nullptr;
  NvrtcResult (*destroy_program)(NvrtcProgram*) = nullptr;
  NvrtcResult (*compile_program)(NvrtcProgram, int, const char* const*) = nullptr;
  NvrtcResult (*get_ptx_size)(NvrtcProgram, std::size_t*) = nullptr;
  NvrtcResult (*get_ptx)(NvrtcProgram, char*) = nullptr;
  NvrtcResult (*get_program_log_size)(NvrtcProgram, std::size_t*) = nullptr;
  NvrtcResult (*get_program_log)(NvrtcProgram, char*) = nullptr;
  const char* (*get_error_string)(NvrtcResult) = nullptr;
};

NvrtcCompiler::NvrtcCompiler() : impl_(new Impl()) {}
NvrtcCompiler::~NvrtcCompiler() { delete impl_; }
NvrtcCompiler::NvrtcCompiler(NvrtcCompiler&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}
NvrtcCompiler& NvrtcCompiler::operator=(NvrtcCompiler&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

std::string NvrtcCompiler::compile(std::string_view source, std::string_view source_name) const {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot use a moved-from NVRTC compiler");
  }
  return impl_->compile(source, source_name);
}

} // namespace reco::calibrate::detail
