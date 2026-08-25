#include "reco/detect/ort_session.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#include <mach-o/dyld.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace reco::detect {
namespace {

constexpr unsigned int kMinimumOrtMinorVersion = 23;

#if defined(_WIN32)
#define RECO_ORT_CALL __stdcall
#else
#define RECO_ORT_CALL
#endif

struct OrtApiBase {
  const void*(RECO_ORT_CALL* get_api)(std::uint32_t);
  const char*(RECO_ORT_CALL* get_version_string)();
};

using OrtGetApiBase = const OrtApiBase*(RECO_ORT_CALL*)();

class DynamicLibrary {
public:
  explicit DynamicLibrary(const std::filesystem::path& path) : path_(path.string()) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(path_.c_str());
#else
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw std::runtime_error("failed to load " + path_);
    }
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif
  }

  template <typename Fn> Fn symbol(const char* name) const {
#if defined(_WIN32)
    auto* sym = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* sym = dlsym(handle_, name);
#endif
    if (sym == nullptr) {
      throw std::runtime_error(path_ + " is not an ONNX Runtime library: missing " + name);
    }
    return reinterpret_cast<Fn>(sym);
  }

  [[nodiscard]] const std::string& path() const { return path_; }

  void release() noexcept { handle_ = nullptr; }

private:
  void* handle_ = nullptr;
  std::string path_;
};

std::filesystem::path current_executable() {
#if defined(_WIN32)
  std::string buffer(MAX_PATH, '\0');
  const DWORD len = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  if (len == 0) {
    return {};
  }
  buffer.resize(len);
  return buffer;
#elif defined(__APPLE__)
  std::uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
    return {};
  }
  buffer.resize(std::strlen(buffer.c_str()));
  return buffer;
#else
  std::string buffer(4096, '\0');
  const ssize_t len = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (len <= 0) {
    return {};
  }
  buffer.resize(static_cast<std::size_t>(len));
  return buffer;
#endif
}

std::filesystem::path default_soname() {
#if defined(_WIN32)
  return "onnxruntime.dll";
#elif defined(__APPLE__)
  return "libonnxruntime.dylib";
#else
  return "libonnxruntime.so";
#endif
}

std::filesystem::path getenv_path(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return {};
  }
  return value;
}

std::filesystem::path resolve_ort_library_path() {
  auto requested = getenv_path("ORT_DYLIB_PATH");
  if (requested.empty()) {
    requested = default_soname();
  }
  if (requested.is_absolute()) {
    return requested;
  }
  const auto exe = current_executable();
  if (!exe.empty()) {
    const auto beside_exe = exe.parent_path() / requested;
    std::error_code ec;
    if (std::filesystem::exists(beside_exe, ec)) {
      return beside_exe;
    }
  }
  return requested;
}

unsigned int parse_minor_version(std::string_view version) {
  const auto first_dot = version.find('.');
  if (first_dot == std::string_view::npos) {
    return 0;
  }
  const auto second_dot = version.find('.', first_dot + 1);
  const auto minor_text = version.substr(first_dot + 1, second_dot == std::string_view::npos
                                                            ? std::string_view::npos
                                                            : second_dot - first_dot - 1);
  unsigned long value = 0;
  for (char ch : minor_text) {
    if (ch < '0' || ch > '9') {
      return 0;
    }
    value = value * 10UL + static_cast<unsigned long>(ch - '0');
    if (value > std::numeric_limits<unsigned int>::max()) {
      return 0;
    }
  }
  return static_cast<unsigned int>(value);
}

OrtRuntimeProbe compute_ort_probe() {
  const auto path = resolve_ort_library_path();
  try {
    DynamicLibrary lib(path);
    const auto api_base_getter = lib.symbol<OrtGetApiBase>("OrtGetApiBase");
    const OrtApiBase* base = api_base_getter();
    if (base == nullptr || base->get_version_string == nullptr) {
      return {.available = false,
              .path = path.string(),
              .error = path.string() + ": OrtGetApiBase returned an invalid API base"};
    }

    const char* raw_version = base->get_version_string();
    const std::string version = raw_version == nullptr ? std::string{} : std::string(raw_version);
    const auto minor = parse_minor_version(version);
    if (minor < kMinimumOrtMinorVersion) {
      return {.available = false,
              .path = path.string(),
              .version = version,
              .error = "ONNX Runtime at `" + path.string() + "` is version " + version +
                       "; reco needs >= 1.23"};
    }
    lib.release();
    return {.available = true, .path = path.string(), .version = version};
  } catch (const std::exception& error) {
    return {.available = false,
            .path = path.string(),
            .error = "ONNX Runtime library not found (`" + path.string() + "`: " + error.what() +
                     "). Install onnxruntime or place the library next to the executable."};
  }
}

const OrtRuntimeProbe& cached_ort_probe() {
  static const OrtRuntimeProbe probe = compute_ort_probe();
  return probe;
}

std::filesystem::path platform_cache_base() {
#if defined(_WIN32)
  if (auto local_app_data = getenv_path("LOCALAPPDATA"); !local_app_data.empty()) {
    return local_app_data;
  }
#elif defined(__APPLE__)
  if (auto home = getenv_path("HOME"); !home.empty()) {
    return home / "Library" / "Caches";
  }
#else
  if (auto override_path = getenv_path("XDG_CACHE_HOME"); !override_path.empty()) {
    return override_path;
  }
  if (auto home = getenv_path("HOME"); !home.empty()) {
    return home / ".cache";
  }
#endif
  return std::filesystem::temp_directory_path();
}

} // namespace

OrtRuntimeProbe probe_ort_runtime() { return cached_ort_probe(); }

bool ort_runtime_available() { return cached_ort_probe().available; }

std::string ort_runtime_error() { return cached_ort_probe().error; }

std::filesystem::path reco_cache_dir(std::string_view subdir) {
  std::filesystem::path dir = platform_cache_base() / "reco" / std::filesystem::path(subdir);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
#if !defined(_WIN32)
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                               std::filesystem::perm_options::replace, ec);
#endif
  return dir;
}

} // namespace reco::detect
