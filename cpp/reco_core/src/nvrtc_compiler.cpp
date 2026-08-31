#include "reco/core/nvrtc_compiler.hpp"

#include <algorithm>
#include <condition_variable>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <new>
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
constexpr int kMinimumEncodedArchitecture = 10;
constexpr int kMaximumEncodedArchitecture = 999;

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

struct CompileCacheKey {
  std::string library_path;
  NvrtcVersion version;
  std::string source;
  std::string source_name;
  std::vector<std::string> options;

  [[nodiscard]] bool operator==(const CompileCacheKey& other) const {
    return library_path == other.library_path && version.major == other.version.major &&
           version.minor == other.version.minor && source == other.source &&
           source_name == other.source_name && options == other.options;
  }
};

std::size_t checked_cache_bytes(const CompileCacheKey& key, const NvrtcCompileResult& result) {
  std::size_t bytes = sizeof(CompileCacheKey) + sizeof(NvrtcCompileResult) +
                      key.options.size() * sizeof(std::string);
  const auto add = [&](std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - bytes) {
      throw NvrtcError("NVRTC compile cache accounting overflow");
    }
    bytes += value;
  };
  add(key.library_path.size());
  add(key.source.size());
  add(key.source_name.size());
  for (const auto& option : key.options) {
    add(option.size());
  }
  add(result.ptx.size());
  add(result.log.size());
  return bytes;
}

class CompileCoordinator {
public:
  template <typename Compile>
  [[nodiscard]] NvrtcCompileResult compile(const CompileCacheKey& key, Compile&& compile_uncached) {
    std::unique_lock lock(mutex_);
    for (;;) {
      if (auto cached = find_locked(key); cached != cache_.end()) {
        cache_.splice(cache_.begin(), cache_, cached);
        return cached->result;
      }
      if (!compiling_) {
        compiling_ = true;
        break;
      }
      changed_.wait(lock);
    }
    lock.unlock();

    NvrtcCompileResult result;
    try {
      result = compile_uncached();
    } catch (...) {
      finish_compile();
      throw;
    }

    lock.lock();
    try {
      retain_locked(key, result);
    } catch (const std::bad_alloc&) {
      // A cache allocation failure must not discard an otherwise valid compilation.
    } catch (...) {
      compiling_ = false;
      lock.unlock();
      changed_.notify_all();
      throw;
    }
    compiling_ = false;
    lock.unlock();
    changed_.notify_all();
    return result;
  }

private:
  struct CachedCompile {
    CompileCacheKey key;
    NvrtcCompileResult result;
    std::size_t accounted_bytes = 0;
  };

  using Cache = std::list<CachedCompile>;

  [[nodiscard]] Cache::iterator find_locked(const CompileCacheKey& key) {
    return std::find_if(cache_.begin(), cache_.end(),
                        [&](const CachedCompile& entry) { return entry.key == key; });
  }

  void retain_locked(const CompileCacheKey& key, const NvrtcCompileResult& result) {
    const auto entry_bytes = checked_cache_bytes(key, result);
    if (entry_bytes > kNvrtcCompileCacheMaximumBytes) {
      return;
    }
    while (!cache_.empty() && (cache_.size() >= kNvrtcCompileCacheMaximumEntries ||
                               entry_bytes > kNvrtcCompileCacheMaximumBytes - cache_bytes_)) {
      cache_bytes_ -= cache_.back().accounted_bytes;
      cache_.pop_back();
    }
    cache_.push_front(CachedCompile{.key = key, .result = result, .accounted_bytes = entry_bytes});
    cache_bytes_ += entry_bytes;
  }

  void finish_compile() noexcept {
    {
      std::lock_guard lock(mutex_);
      compiling_ = false;
    }
    changed_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable changed_;
  Cache cache_;
  std::size_t cache_bytes_ = 0;
  bool compiling_ = false;
};

CompileCoordinator& compile_coordinator() {
  static CompileCoordinator coordinator;
  return coordinator;
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
    get_num_supported_architectures =
        library.symbol<decltype(get_num_supported_architectures)>("nvrtcGetNumSupportedArchs");
    get_supported_architectures =
        library.symbol<decltype(get_supported_architectures)>("nvrtcGetSupportedArchs");

    int major = 0;
    int minor = 0;
    check("nvrtcVersion", version_function(&major, &minor));
    if (major < 0 || minor < 0) {
      throw NvrtcError("nvrtcVersion returned an invalid negative version");
    }
    loaded_version = {.major = major, .minor = minor};
    supported_architecture_values = query_supported_architectures();
  }

  [[nodiscard]] NvrtcCompileResult compile(std::string_view source, std::string_view source_name,
                                           const NvrtcCompileOptions& options) const {
    validate_text(source, kNvrtcMaximumSourceBytes, "source");
    validate_text(source_name, kNvrtcMaximumSourceNameBytes, "source name");
    validate_options(options);

    CompileCacheKey key{
        .library_path = std::string(library.path()),
        .version = loaded_version,
        .source = std::string(source),
        .source_name = std::string(source_name),
        .options = options.values,
    };
    return compile_coordinator().compile(
        key, [&] { return compile_uncached(key.source, key.source_name, key.options); });
  }

  [[nodiscard]] NvrtcCompileResult compile_uncached(const std::string& source_text,
                                                    const std::string& source_name_text,
                                                    const std::vector<std::string>& options) const {
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
    option_values.reserve(options.size());
    for (const auto& option : options) {
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

  [[nodiscard]] std::vector<int> query_supported_architectures() const {
    int count = 0;
    check("nvrtcGetNumSupportedArchs", get_num_supported_architectures(&count));
    if (count <= 0) {
      throw NvrtcError("nvrtcGetNumSupportedArchs returned no supported architectures");
    }
    if (static_cast<std::size_t>(count) > kNvrtcMaximumSupportedArchitectureCount) {
      throw NvrtcError("nvrtcGetNumSupportedArchs returned " + std::to_string(count) +
                       " architectures, exceeding the limit of " +
                       std::to_string(kNvrtcMaximumSupportedArchitectureCount));
    }

    std::vector<int> architectures(static_cast<std::size_t>(count));
    check("nvrtcGetSupportedArchs", get_supported_architectures(architectures.data()));
    for (const auto architecture : architectures) {
      if (architecture < kMinimumEncodedArchitecture ||
          architecture > kMaximumEncodedArchitecture) {
        throw NvrtcError("nvrtcGetSupportedArchs returned invalid architecture " +
                         std::to_string(architecture));
      }
    }
    std::sort(architectures.begin(), architectures.end());
    architectures.erase(std::unique(architectures.begin(), architectures.end()),
                        architectures.end());
    return architectures;
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
  std::vector<int> supported_architecture_values;
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
  NvrtcResult (*get_num_supported_architectures)(int*) = nullptr;
  NvrtcResult (*get_supported_architectures)(int*) = nullptr;
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

std::vector<int> NvrtcCompiler::supported_architectures() const {
  return checked_impl().supported_architecture_values;
}

int NvrtcCompiler::select_architecture(int device_architecture) const {
  if (device_architecture < kMinimumEncodedArchitecture ||
      device_architecture > kMaximumEncodedArchitecture) {
    throw std::invalid_argument("CUDA device architecture must be encoded as major * 10 + minor");
  }
  const auto& impl = checked_impl();
  const auto selected =
      std::upper_bound(impl.supported_architecture_values.begin(),
                       impl.supported_architecture_values.end(), device_architecture);
  if (selected == impl.supported_architecture_values.begin()) {
    std::ostringstream message;
    message << "NVRTC " << impl.loaded_version.major << '.' << impl.loaded_version.minor
            << " does not support a virtual architecture compatible with compute_"
            << device_architecture << "; supported architectures:";
    for (const auto architecture : impl.supported_architecture_values) {
      message << " compute_" << architecture;
    }
    throw NvrtcError(message.str());
  }
  return *std::prev(selected);
}

NvrtcCompileResult NvrtcCompiler::compile(std::string_view source, std::string_view source_name,
                                          const NvrtcCompileOptions& options) const {
  return checked_impl().compile(source, source_name, options);
}

} // namespace reco::core
