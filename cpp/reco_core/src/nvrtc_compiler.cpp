#include "reco/core/nvrtc_compiler.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#if defined(_WIN32)
#include <filesystem>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reco::core {
namespace {

enum class NvrtcResult : int {};
using NvrtcProgram = void*;

constexpr int kNvrtcSuccess = 0;
constexpr std::size_t kMaximumLibraryPathBytes = 32U * 1024U;
constexpr std::size_t kMaximumErrorStringBytes = 1024U;

void validate_text(std::string_view value, std::size_t maximum_bytes, const char* label) {
  if (value.empty()) {
    throw std::invalid_argument(std::string("NVRTC ") + label + " must be non-empty");
  }
  if (value.size() > maximum_bytes) {
    throw std::invalid_argument(std::string("NVRTC ") + label + " exceeds " +
                                std::to_string(maximum_bytes) + " bytes");
  }
  if (std::find(value.begin(), value.end(), '\0') != value.end()) {
    throw std::invalid_argument(std::string("NVRTC ") + label + " must not contain NUL bytes");
  }
}

std::string bounded_c_string(const char* value) {
  if (value == nullptr) {
    return "unknown error";
  }
  std::size_t size = 0;
  while (size < kMaximumErrorStringBytes && value[size] != '\0') {
    ++size;
  }
  std::string result(value, size);
  if (size == kMaximumErrorStringBytes) {
    result += "...";
  }
  return result;
}

std::string strip_trailing_nul(std::string value) {
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

std::string sanitize_log(std::string log) {
  log = strip_trailing_nul(std::move(log));
  std::replace(log.begin(), log.end(), '\0', '?');
  return log;
}

#if defined(_WIN32)
std::wstring utf8_to_wide(std::string_view value) {
  const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                        static_cast<int>(value.size()), nullptr, 0);
  if (size <= 0) {
    throw NvrtcError("NVRTC library path is not valid UTF-8");
  }
  std::wstring result(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), result.data(), size) != size) {
    throw NvrtcError("failed to convert NVRTC library path to UTF-16");
  }
  return result;
}

std::wstring resolve_windows_library_path(const std::wstring& requested) {
  const std::filesystem::path requested_path(requested);
  if (requested_path.is_absolute()) {
    return requested_path.lexically_normal().native();
  }
  if (requested_path.has_parent_path()) {
    std::error_code error;
    const auto absolute = std::filesystem::absolute(requested_path, error);
    return error ? requested : absolute.lexically_normal().native();
  }

  constexpr unsigned long kMaximumPathEnvironmentCharacters = 1024U * 1024U;
  const auto required = GetEnvironmentVariableW(L"PATH", nullptr, 0);
  if (required == 0 || required > kMaximumPathEnvironmentCharacters) {
    return requested;
  }
  std::wstring path_value(static_cast<std::size_t>(required), L'\0');
  const auto written = GetEnvironmentVariableW(L"PATH", path_value.data(), required);
  if (written == 0 || written >= required) {
    return requested;
  }
  path_value.resize(written);
  std::size_t offset = 0;
  while (offset <= path_value.size()) {
    const auto separator = path_value.find(L';', offset);
    auto directory = path_value.substr(
        offset, separator == std::wstring::npos ? std::wstring::npos : separator - offset);
    if (directory.size() >= 2 && directory.front() == L'"' && directory.back() == L'"') {
      directory = directory.substr(1, directory.size() - 2);
    }
    const std::filesystem::path directory_path(directory);
    if (!directory.empty() && directory_path.is_absolute()) {
      const auto candidate = (directory_path / requested_path).lexically_normal();
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error) && !error) {
        return candidate.native();
      }
    }
    if (separator == std::wstring::npos) {
      break;
    }
    offset = separator + 1;
  }
  return requested;
}
#endif

class DynamicLibrary {
public:
  explicit DynamicLibrary(std::string path) : path_(std::move(path)) {
#if defined(_WIN32)
    const auto wide_path = resolve_windows_library_path(utf8_to_wide(path_));
    auto flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    if (std::filesystem::path(wide_path).is_absolute()) {
      flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
    }
    handle_ = LoadLibraryExW(wide_path.c_str(), nullptr, flags);
    if (handle_ == nullptr) {
      const auto error = GetLastError();
      throw NvrtcError("failed to load NVRTC library " + path_ + " (Windows error " +
                       std::to_string(error) + ")");
    }
#else
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw NvrtcError("failed to load NVRTC library " + path_ +
                       (error == nullptr ? "" : ": " + std::string(error)));
    }
#endif
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary() {
    if (handle_ == nullptr) {
      return;
    }
#if defined(_WIN32)
    (void)FreeLibrary(handle_);
#else
    (void)dlclose(handle_);
#endif
  }

  template <typename Function> [[nodiscard]] Function symbol(const char* name) const {
#if defined(_WIN32)
    auto* value = reinterpret_cast<void*>(GetProcAddress(handle_, name));
#else
    dlerror();
    void* value = dlsym(handle_, name);
#endif
    if (value == nullptr) {
      throw NvrtcError("missing NVRTC symbol " + std::string(name) + " in " + path_);
    }
    return reinterpret_cast<Function>(value);
  }

  [[nodiscard]] std::string_view path() const { return path_; }

private:
  std::string path_;
#if defined(_WIN32)
  HMODULE handle_ = nullptr;
#else
  void* handle_ = nullptr;
#endif
};

std::vector<std::string_view> default_library_names() {
#if defined(_WIN32)
  return {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"};
#elif defined(__APPLE__)
  return {"libnvrtc.dylib", "libnvrtc.12.dylib", "libnvrtc.11.2.dylib"};
#else
  return {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"};
#endif
}

void validate_options(const NvrtcCompileOptions& options) {
  if (options.values.size() > kNvrtcMaximumOptionCount) {
    throw std::invalid_argument("NVRTC option count exceeds " +
                                std::to_string(kNvrtcMaximumOptionCount));
  }
  std::size_t total_bytes = 0;
  for (const auto& option : options.values) {
    validate_text(option, kNvrtcMaximumOptionBytes, "option");
    if (option.size() > kNvrtcMaximumTotalOptionBytes - total_bytes) {
      throw std::invalid_argument("NVRTC options exceed " +
                                  std::to_string(kNvrtcMaximumTotalOptionBytes) +
                                  " bytes in total");
    }
    total_bytes += option.size();
  }
}

} // namespace

struct NvrtcCompiler::Impl {
  explicit Impl(std::string path) : library(std::move(path)) {
    version_function = library.symbol<decltype(version_function)>("nvrtcVersion");
    create_program = library.symbol<decltype(create_program)>("nvrtcCreateProgram");
    destroy_program = library.symbol<decltype(destroy_program)>("nvrtcDestroyProgram");
    compile_program = library.symbol<decltype(compile_program)>("nvrtcCompileProgram");
    get_ptx_size = library.symbol<decltype(get_ptx_size)>("nvrtcGetPTXSize");
    get_ptx = library.symbol<decltype(get_ptx)>("nvrtcGetPTX");
    get_program_log_size = library.symbol<decltype(get_program_log_size)>("nvrtcGetProgramLogSize");
    get_program_log = library.symbol<decltype(get_program_log)>("nvrtcGetProgramLog");
    get_error_string = library.symbol<decltype(get_error_string)>("nvrtcGetErrorString");

    int major = 0;
    int minor = 0;
    check("nvrtcVersion", version_function(&major, &minor));
    if (major < 0 || minor < 0) {
      throw NvrtcError("nvrtcVersion returned an invalid negative version");
    }
    loaded_version = {.major = major, .minor = minor};
  }

  [[nodiscard]] NvrtcCompileResult compile(std::string_view source, std::string_view source_name,
                                           const NvrtcCompileOptions& options) const {
    validate_text(source, kNvrtcMaximumSourceBytes, "source");
    validate_text(source_name, kNvrtcMaximumSourceNameBytes, "source name");
    validate_options(options);

    const std::string source_text(source);
    const std::string source_name_text(source_name);
    NvrtcProgram program = nullptr;
    const auto create_result = create_program(&program, source_text.c_str(),
                                              source_name_text.c_str(), 0, nullptr, nullptr);
    struct ProgramGuard {
      NvrtcProgram program = nullptr;
      NvrtcResult (*destroy)(NvrtcProgram*) = nullptr;
      ~ProgramGuard() {
        if (program != nullptr) {
          (void)destroy(&program);
        }
      }
      NvrtcResult close() {
        const auto result = destroy(&program);
        if (static_cast<int>(result) == kNvrtcSuccess) {
          program = nullptr;
        }
        return result;
      }
    } guard{program, destroy_program};
    check("nvrtcCreateProgram", create_result);
    if (program == nullptr) {
      throw NvrtcError("nvrtcCreateProgram succeeded without returning a program");
    }

    std::vector<const char*> option_values;
    option_values.reserve(options.values.size());
    for (const auto& option : options.values) {
      option_values.push_back(option.c_str());
    }
    const auto compile_result =
        compile_program(program, static_cast<int>(option_values.size()), option_values.data());
    if (static_cast<int>(compile_result) != kNvrtcSuccess) {
      throw NvrtcError(status_message("nvrtcCompileProgram", compile_result) + " for " +
                       source_name_text + ": " + program_log(program, false));
    }

    auto log = program_log(program, true);
    std::size_t ptx_size = 0;
    check("nvrtcGetPTXSize", get_ptx_size(program, &ptx_size));
    if (ptx_size == 0) {
      throw NvrtcError("NVRTC produced empty PTX for " + source_name_text);
    }
    if (ptx_size > kNvrtcMaximumPtxBytes) {
      throw NvrtcError("NVRTC PTX for " + source_name_text + " exceeds " +
                       std::to_string(kNvrtcMaximumPtxBytes) + " bytes");
    }
    std::string ptx(ptx_size, '\0');
    check("nvrtcGetPTX", get_ptx(program, ptx.data()));
    if (ptx.back() != '\0') {
      throw NvrtcError("NVRTC produced PTX without a terminating NUL byte for " + source_name_text);
    }
    ptx.pop_back();
    if (ptx.empty()) {
      throw NvrtcError("NVRTC produced empty PTX for " + source_name_text);
    }
    if (std::find(ptx.begin(), ptx.end(), '\0') != ptx.end()) {
      throw NvrtcError("NVRTC produced PTX containing an interior NUL byte for " +
                       source_name_text);
    }
    NvrtcCompileResult result{.ptx = std::move(ptx), .log = std::move(log)};
    check("nvrtcDestroyProgram", guard.close());
    return result;
  }

  void check(const char* function, NvrtcResult result) const {
    if (static_cast<int>(result) != kNvrtcSuccess) {
      throw NvrtcError(status_message(function, result));
    }
  }

  [[nodiscard]] std::string status_message(const char* function, NvrtcResult result) const {
    std::ostringstream message;
    message << function << " returned NVRTC error " << static_cast<int>(result) << " ("
            << bounded_c_string(get_error_string(result)) << ')';
    return message.str();
  }

  [[nodiscard]] std::string program_log(NvrtcProgram program, bool errors_are_fatal) const {
    std::size_t log_size = 0;
    const auto size_result = get_program_log_size(program, &log_size);
    if (static_cast<int>(size_result) != kNvrtcSuccess) {
      if (errors_are_fatal) {
        check("nvrtcGetProgramLogSize", size_result);
      }
      return "compiler log unavailable: " + status_message("nvrtcGetProgramLogSize", size_result);
    }
    if (log_size == 0) {
      return {};
    }
    if (log_size > kNvrtcMaximumLogBytes) {
      return "compiler log omitted: reported size " + std::to_string(log_size) + " exceeds " +
             std::to_string(kNvrtcMaximumLogBytes) + " bytes";
    }
    std::string log(log_size, '\0');
    const auto log_result = get_program_log(program, log.data());
    if (static_cast<int>(log_result) != kNvrtcSuccess) {
      if (errors_are_fatal) {
        check("nvrtcGetProgramLog", log_result);
      }
      return "compiler log unavailable: " + status_message("nvrtcGetProgramLog", log_result);
    }
    return sanitize_log(std::move(log));
  }

  DynamicLibrary library;
  NvrtcVersion loaded_version;
  NvrtcResult (*version_function)(int*, int*) = nullptr;
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

NvrtcCompiler::NvrtcCompiler(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

const NvrtcCompiler::Impl& NvrtcCompiler::checked_impl() const {
  if (impl_ == nullptr) {
    throw std::logic_error("cannot use a moved-from NVRTC compiler");
  }
  return *impl_;
}

bool NvrtcCompiler::is_available() { return availability_error().empty(); }

std::string NvrtcCompiler::availability_error() {
  try {
    (void)create();
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}

NvrtcCompiler NvrtcCompiler::create() {
  std::string errors;
  for (const auto name : default_library_names()) {
    try {
      return load(name);
    } catch (const std::exception& error) {
      if (!errors.empty()) {
        errors += "; ";
      }
      errors += error.what();
    }
  }
  throw NvrtcError("NVRTC is unavailable: " + errors);
}

NvrtcCompiler NvrtcCompiler::load(std::string_view library_path) {
  validate_text(library_path, kMaximumLibraryPathBytes, "library path");
  return NvrtcCompiler(std::make_shared<Impl>(std::string(library_path)));
}

std::string NvrtcCompiler::availability_error(std::string_view library_path) {
  try {
    (void)load(library_path);
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}

NvrtcVersion NvrtcCompiler::version() const { return checked_impl().loaded_version; }

std::string_view NvrtcCompiler::library_path() const { return checked_impl().library.path(); }

NvrtcCompileResult NvrtcCompiler::compile(std::string_view source, std::string_view source_name,
                                          const NvrtcCompileOptions& options) const {
  return checked_impl().compile(source, source_name, options);
}

} // namespace reco::core
