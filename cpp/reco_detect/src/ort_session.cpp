#include "reco/detect/ort_session.hpp"

#include "reco/detect/detectors.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
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

constexpr std::uint32_t kOrtApiVersion = 23;
constexpr unsigned int kMinimumOrtMinorVersion = 23;
constexpr int kOrtLoggingLevelWarning = 2;
constexpr int kOrtEnableAllGraphOptimizations = 99;
constexpr int kOnnxTensorElementDataTypeFloat = 1;
constexpr int kOrtDeviceAllocator = 0;
constexpr int kOrtArenaAllocator = 1;
constexpr int kOrtMemTypeDefault = 0;
constexpr const char* kOrtSessionOptionsDisableCpuEpFallback =
    "session.disable_cpu_ep_fallback";

#if defined(_WIN32)
#define RECO_ORT_CALL __stdcall
#else
#define RECO_ORT_CALL
#endif

struct OrtCEnv;
struct OrtCStatus;
struct OrtCMemoryInfo;
struct OrtCSessionOptions;
struct OrtCSession;
struct OrtCValue;
struct OrtCTypeInfo;
struct OrtCTensorTypeAndShapeInfo;
struct OrtCAllocator;
struct OrtCModelMetadata;
struct OrtCRunOptions;
struct OrtCCustomOpDomain;
struct OrtCCustomOp;
struct OrtCThreadingOptions;
struct OrtCMapTypeInfo;
struct OrtCSequenceTypeInfo;

struct OrtApiBase {
  const void*(RECO_ORT_CALL* get_api)(std::uint32_t);
  const char*(RECO_ORT_CALL* get_version_string)();
};

using OrtGetApiBase = const OrtApiBase*(RECO_ORT_CALL*)();
using OrtStatusPtr = OrtCStatus*;
using OrtAppendCudaProvider = OrtCStatus*(RECO_ORT_CALL*)(OrtCSessionOptions*, int);

#if defined(_WIN32)
using OrtPathChar = wchar_t;
#else
using OrtPathChar = char;
#endif

struct OrtApi {
  OrtStatusPtr(RECO_ORT_CALL* CreateStatus)(int, const char*);
  int(RECO_ORT_CALL* GetErrorCode)(const OrtCStatus*);
  const char*(RECO_ORT_CALL* GetErrorMessage)(const OrtCStatus*);
  OrtStatusPtr(RECO_ORT_CALL* CreateEnv)(int, const char*, OrtCEnv**);
  void* CreateEnvWithCustomLogger;
  void* EnableTelemetryEvents;
  void* DisableTelemetryEvents;
  OrtStatusPtr(RECO_ORT_CALL* CreateSession)(const OrtCEnv*, const OrtPathChar*,
                                             const OrtCSessionOptions*, OrtCSession**);
  OrtStatusPtr(RECO_ORT_CALL* CreateSessionFromArray)(const OrtCEnv*, const void*, std::size_t,
                                                      const OrtCSessionOptions*, OrtCSession**);
  OrtStatusPtr(RECO_ORT_CALL* Run)(OrtCSession*, const OrtCRunOptions*, const char* const*,
                                   const OrtCValue* const*, std::size_t, const char* const*,
                                   std::size_t, OrtCValue**);
  OrtStatusPtr(RECO_ORT_CALL* CreateSessionOptions)(OrtCSessionOptions**);
  void* SetOptimizedModelFilePath;
  void* CloneSessionOptions;
  void* SetSessionExecutionMode;
  void* EnableProfiling;
  void* DisableProfiling;
  void* EnableMemPattern;
  void* DisableMemPattern;
  void* EnableCpuMemArena;
  void* DisableCpuMemArena;
  void* SetSessionLogId;
  void* SetSessionLogVerbosityLevel;
  OrtStatusPtr(RECO_ORT_CALL* SetSessionLogSeverityLevel)(OrtCSessionOptions*, int);
  OrtStatusPtr(RECO_ORT_CALL* SetSessionGraphOptimizationLevel)(OrtCSessionOptions*, int);
  void* SetIntraOpNumThreads;
  void* SetInterOpNumThreads;
  void* CreateCustomOpDomain;
  void* CustomOpDomain_Add;
  void* AddCustomOpDomain;
  void* RegisterCustomOpsLibrary;
  OrtStatusPtr(RECO_ORT_CALL* SessionGetInputCount)(const OrtCSession*, std::size_t*);
  OrtStatusPtr(RECO_ORT_CALL* SessionGetOutputCount)(const OrtCSession*, std::size_t*);
  void* SessionGetOverridableInitializerCount;
  OrtStatusPtr(RECO_ORT_CALL* SessionGetInputTypeInfo)(const OrtCSession*, std::size_t,
                                                       OrtCTypeInfo**);
  void* SessionGetOutputTypeInfo;
  void* SessionGetOverridableInitializerTypeInfo;
  OrtStatusPtr(RECO_ORT_CALL* SessionGetInputName)(const OrtCSession*, std::size_t, OrtCAllocator*,
                                                   char**);
  OrtStatusPtr(RECO_ORT_CALL* SessionGetOutputName)(const OrtCSession*, std::size_t, OrtCAllocator*,
                                                    char**);
  void* SessionGetOverridableInitializerName;
  void* CreateRunOptions;
  void* RunOptionsSetRunLogVerbosityLevel;
  void* RunOptionsSetRunLogSeverityLevel;
  void* RunOptionsSetRunTag;
  void* RunOptionsGetRunLogVerbosityLevel;
  void* RunOptionsGetRunLogSeverityLevel;
  void* RunOptionsGetRunTag;
  void* RunOptionsSetTerminate;
  void* RunOptionsUnsetTerminate;
  void* CreateTensorAsOrtValue;
  OrtStatusPtr(RECO_ORT_CALL* CreateTensorWithDataAsOrtValue)(const OrtCMemoryInfo*, void*,
                                                              std::size_t, const std::int64_t*,
                                                              std::size_t, int, OrtCValue**);
  void* IsTensor;
  OrtStatusPtr(RECO_ORT_CALL* GetTensorMutableData)(OrtCValue*, void**);
  void* FillStringTensor;
  void* GetStringTensorDataLength;
  void* GetStringTensorContent;
  OrtStatusPtr(RECO_ORT_CALL* CastTypeInfoToTensorInfo)(const OrtCTypeInfo*,
                                                        const OrtCTensorTypeAndShapeInfo**);
  void* GetOnnxTypeFromTypeInfo;
  void* CreateTensorTypeAndShapeInfo;
  void* SetTensorElementType;
  void* SetDimensions;
  OrtStatusPtr(RECO_ORT_CALL* GetTensorElementType)(const OrtCTensorTypeAndShapeInfo*, int*);
  OrtStatusPtr(RECO_ORT_CALL* GetDimensionsCount)(const OrtCTensorTypeAndShapeInfo*, std::size_t*);
  OrtStatusPtr(RECO_ORT_CALL* GetDimensions)(const OrtCTensorTypeAndShapeInfo*, std::int64_t*,
                                             std::size_t);
  void* GetSymbolicDimensions;
  void* GetTensorShapeElementCount;
  OrtStatusPtr(RECO_ORT_CALL* GetTensorTypeAndShape)(const OrtCValue*,
                                                     OrtCTensorTypeAndShapeInfo**);
  void* GetTypeInfo;
  void* GetValueType;
  OrtStatusPtr(RECO_ORT_CALL* CreateMemoryInfo)(const char*, int, int, int, OrtCMemoryInfo**);
  OrtStatusPtr(RECO_ORT_CALL* CreateCpuMemoryInfo)(int, int, OrtCMemoryInfo**);
  void* CompareMemoryInfo;
  void* MemoryInfoGetName;
  void* MemoryInfoGetId;
  void* MemoryInfoGetMemType;
  void* MemoryInfoGetType;
  void* AllocatorAlloc;
  OrtStatusPtr(RECO_ORT_CALL* AllocatorFree)(OrtCAllocator*, void*);
  void* AllocatorGetInfo;
  OrtStatusPtr(RECO_ORT_CALL* GetAllocatorWithDefaultOptions)(OrtCAllocator**);
  void* AddFreeDimensionOverride;
  void* GetValue;
  void* GetValueCount;
  void* CreateValue;
  void* CreateOpaqueValue;
  void* GetOpaqueValue;
  void* KernelInfoGetAttribute_float;
  void* KernelInfoGetAttribute_int64;
  void* KernelInfoGetAttribute_string;
  void* KernelContext_GetInputCount;
  void* KernelContext_GetOutputCount;
  void* KernelContext_GetInput;
  void* KernelContext_GetOutput;
  void(RECO_ORT_CALL* ReleaseEnv)(OrtCEnv*);
  void(RECO_ORT_CALL* ReleaseStatus)(OrtCStatus*);
  void(RECO_ORT_CALL* ReleaseMemoryInfo)(OrtCMemoryInfo*);
  void(RECO_ORT_CALL* ReleaseSession)(OrtCSession*);
  void(RECO_ORT_CALL* ReleaseValue)(OrtCValue*);
  void(RECO_ORT_CALL* ReleaseRunOptions)(OrtCRunOptions*);
  void(RECO_ORT_CALL* ReleaseTypeInfo)(OrtCTypeInfo*);
  void(RECO_ORT_CALL* ReleaseTensorTypeAndShapeInfo)(OrtCTensorTypeAndShapeInfo*);
  void(RECO_ORT_CALL* ReleaseSessionOptions)(OrtCSessionOptions*);
  void(RECO_ORT_CALL* ReleaseCustomOpDomain)(OrtCCustomOpDomain*);
  void* GetDenotationFromTypeInfo;
  void* CastTypeInfoToMapTypeInfo;
  void* CastTypeInfoToSequenceTypeInfo;
  void* GetMapKeyType;
  void* GetMapValueType;
  void* GetSequenceElementType;
  void* ReleaseMapTypeInfo;
  void* ReleaseSequenceTypeInfo;
  void* SessionEndProfiling;
  OrtStatusPtr(RECO_ORT_CALL* SessionGetModelMetadata)(const OrtCSession*, OrtCModelMetadata**);
  void* ModelMetadataGetProducerName;
  void* ModelMetadataGetGraphName;
  void* ModelMetadataGetDomain;
  void* ModelMetadataGetDescription;
  OrtStatusPtr(RECO_ORT_CALL* ModelMetadataLookupCustomMetadataMap)(const OrtCModelMetadata*,
                                                                    OrtCAllocator*, const char*,
                                                                    char**);
  void* ModelMetadataGetVersion;
  void(RECO_ORT_CALL* ReleaseModelMetadata)(OrtCModelMetadata*);
  void* CreateEnvWithGlobalThreadPools;
  void* DisablePerSessionThreads;
  void* CreateThreadingOptions;
  void* ReleaseThreadingOptions;
  void* ModelMetadataGetCustomMetadataMapKeys;
  void* AddFreeDimensionOverrideByName;
  void* GetAvailableProviders;
  void* ReleaseAvailableProviders;
  void* GetStringTensorElementLength;
  void* GetStringTensorElement;
  void* FillStringTensorElement;
  OrtStatusPtr(RECO_ORT_CALL* AddSessionConfigEntry)(OrtCSessionOptions*, const char*, const char*);
};

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

struct OrtRuntimeApi {
  const OrtApi* api = nullptr;
  OrtAppendCudaProvider append_cuda_provider = nullptr;
};

const OrtRuntimeApi& ort_runtime_api() {
  static const OrtRuntimeApi runtime = [] {
    const auto probe = cached_ort_probe();
    if (!probe.available) {
      throw std::runtime_error(probe.error);
    }
    const auto path = resolve_ort_library_path();
    static std::unique_ptr<DynamicLibrary> pinned_library;
    pinned_library = std::make_unique<DynamicLibrary>(path);
    const auto api_base_getter = pinned_library->symbol<OrtGetApiBase>("OrtGetApiBase");
    const OrtApiBase* base = api_base_getter();
    if (base == nullptr || base->get_api == nullptr) {
      throw std::runtime_error(path.string() + ": OrtGetApiBase returned an invalid API base");
    }
    const auto* api = static_cast<const OrtApi*>(base->get_api(kOrtApiVersion));
    if (api == nullptr) {
      throw std::runtime_error("ONNX Runtime at `" + path.string() +
                               "` does not support ORT C API version 23");
    }
    OrtAppendCudaProvider append_cuda_provider = nullptr;
    try {
      append_cuda_provider =
          pinned_library->symbol<OrtAppendCudaProvider>("OrtSessionOptionsAppendExecutionProvider_CUDA");
    } catch (const std::exception&) {
      append_cuda_provider = nullptr;
    }
    return OrtRuntimeApi{.api = api, .append_cuda_provider = append_cuda_provider};
  }();
  return runtime;
}

const OrtApi& ort_api() {
  return *ort_runtime_api().api;
}

std::string status_message(const OrtApi& api, OrtCStatus* status) {
  if (status == nullptr) {
    return {};
  }
  const char* message = api.GetErrorMessage(status);
  std::string result = message == nullptr ? "unknown ONNX Runtime error" : message;
  api.ReleaseStatus(status);
  return result;
}

void throw_if_error(const OrtApi& api, OrtCStatus* status, std::string_view op) {
  if (status != nullptr) {
    throw std::runtime_error(std::string(op) + ": " + status_message(api, status));
  }
}

template <typename T, auto Release> struct OrtHandle {
  const OrtApi* api = nullptr;
  T* ptr = nullptr;

  OrtHandle() = default;
  OrtHandle(const OrtApi& ort_api, T* value) : api(&ort_api), ptr(value) {}
  ~OrtHandle() { reset(); }
  OrtHandle(const OrtHandle&) = delete;
  OrtHandle& operator=(const OrtHandle&) = delete;
  OrtHandle(OrtHandle&& other) noexcept : api(other.api), ptr(other.ptr) {
    other.api = nullptr;
    other.ptr = nullptr;
  }
  OrtHandle& operator=(OrtHandle&& other) noexcept {
    if (this != &other) {
      reset();
      api = other.api;
      ptr = other.ptr;
      other.api = nullptr;
      other.ptr = nullptr;
    }
    return *this;
  }
  void reset() {
    if (api != nullptr && ptr != nullptr) {
      (api->*Release)(ptr);
    }
    ptr = nullptr;
  }
  [[nodiscard]] T* get() const { return ptr; }
  [[nodiscard]] T** out() {
    reset();
    return &ptr;
  }
  [[nodiscard]] T* release() {
    T* value = ptr;
    ptr = nullptr;
    return value;
  }
};

using EnvHandle = OrtHandle<OrtCEnv, &OrtApi::ReleaseEnv>;
using SessionOptionsHandle = OrtHandle<OrtCSessionOptions, &OrtApi::ReleaseSessionOptions>;
using SessionHandle = OrtHandle<OrtCSession, &OrtApi::ReleaseSession>;
using MemoryInfoHandle = OrtHandle<OrtCMemoryInfo, &OrtApi::ReleaseMemoryInfo>;
using ValueHandle = OrtHandle<OrtCValue, &OrtApi::ReleaseValue>;
using TypeInfoHandle = OrtHandle<OrtCTypeInfo, &OrtApi::ReleaseTypeInfo>;
using TensorInfoHandle = OrtHandle<OrtCTensorTypeAndShapeInfo, &OrtApi::ReleaseTensorTypeAndShapeInfo>;
using ModelMetadataHandle = OrtHandle<OrtCModelMetadata, &OrtApi::ReleaseModelMetadata>;

std::string take_allocator_string(const OrtApi& api, OrtCAllocator* allocator, char* value) {
  if (value == nullptr) {
    return {};
  }
  std::string result(value);
  throw_if_error(api, api.AllocatorFree(allocator, value), "AllocatorFree");
  return result;
}

std::vector<std::string> session_names(const OrtApi& api, OrtCSession* session, OrtCAllocator* allocator,
                                       bool input) {
  std::size_t count = 0;
  throw_if_error(api,
                 input ? api.SessionGetInputCount(session, &count)
                       : api.SessionGetOutputCount(session, &count),
                 input ? "SessionGetInputCount" : "SessionGetOutputCount");
  std::vector<std::string> names;
  names.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    char* raw = nullptr;
    throw_if_error(api,
                   input ? api.SessionGetInputName(session, i, allocator, &raw)
                         : api.SessionGetOutputName(session, i, allocator, &raw),
                   input ? "SessionGetInputName" : "SessionGetOutputName");
    names.push_back(take_allocator_string(api, allocator, raw));
  }
  return names;
}

std::vector<std::int64_t> input_shape(const OrtApi& api, OrtCSession* session) {
  TypeInfoHandle type_info(api, nullptr);
  throw_if_error(api, api.SessionGetInputTypeInfo(session, 0, type_info.out()),
                 "SessionGetInputTypeInfo");
  const OrtCTensorTypeAndShapeInfo* borrowed = nullptr;
  throw_if_error(api, api.CastTypeInfoToTensorInfo(type_info.get(), &borrowed),
                 "CastTypeInfoToTensorInfo");
  if (borrowed == nullptr) {
    return {};
  }
  std::size_t rank = 0;
  throw_if_error(api, api.GetDimensionsCount(borrowed, &rank), "GetDimensionsCount");
  std::vector<std::int64_t> dims(rank, 0);
  if (rank > 0) {
    throw_if_error(api, api.GetDimensions(borrowed, dims.data(), dims.size()), "GetDimensions");
  }
  return dims;
}

std::uint32_t input_size_from_shape(const std::vector<std::int64_t>& shape) {
  if (shape.size() >= 4 && shape[2] > 0 &&
      shape[2] <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
    return static_cast<std::uint32_t>(shape[2]);
  }
  return 1280;
}

std::vector<std::string> labels_from_metadata(const OrtApi& api, OrtCSession* session,
                                              OrtCAllocator* allocator,
                                              std::vector<std::string> fallback) {
  if (!fallback.empty()) {
    return fallback;
  }
  ModelMetadataHandle metadata(api, nullptr);
  if (OrtCStatus* status = api.SessionGetModelMetadata(session, metadata.out()); status != nullptr) {
    (void)status_message(api, status);
    return {"ball"};
  }
  char* raw_names = nullptr;
  if (OrtCStatus* status =
          api.ModelMetadataLookupCustomMetadataMap(metadata.get(), allocator, "names", &raw_names);
      status != nullptr) {
    (void)status_message(api, status);
    return {"ball"};
  }
  if (raw_names != nullptr) {
    const auto names = take_allocator_string(api, allocator, raw_names);
    if (auto parsed = parse_names_dict_string(names); parsed.has_value() && !parsed->empty()) {
      return *parsed;
    }
  }
  return {"ball"};
}

#if defined(_WIN32)
std::wstring ort_model_path(const std::filesystem::path& path) { return path.wstring(); }
#else
std::string ort_model_path(const std::filesystem::path& path) { return path.string(); }
#endif

OrtSessionConfig validate_ort_session_config(OrtSessionConfig config) {
  if (config.model_path.empty()) {
    throw std::invalid_argument("OrtSessionConfig.model_path is required");
  }
  if (!std::filesystem::exists(config.model_path)) {
    throw std::runtime_error("ONNX model not found: " + config.model_path.string());
  }
  if (config.providers.empty()) {
    throw std::invalid_argument("OrtSessionConfig.providers must not be empty");
  }
  const bool supported_provider =
      config.providers.size() == 1 && (config.providers.front() == OrtExecutionProvider::Cpu ||
                                       config.providers.front() == OrtExecutionProvider::Cuda);
  if (!supported_provider) {
    throw std::runtime_error("requested ORT execution provider stack is not registered in the C++ port yet");
  }
  return config;
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

struct OrtSession::Impl {
  explicit Impl(OrtSessionConfig config) : api(&ort_api()), env(*api, nullptr), options(*api, nullptr),
                                           session(*api, nullptr), cpu_memory_info(*api, nullptr),
                                           cuda_memory_info(*api, nullptr),
                                           provider(config.providers.front()) {
    throw_if_error(*api, api->CreateEnv(kOrtLoggingLevelWarning, "reco-detect", env.out()),
                   "CreateEnv");
    throw_if_error(*api, api->CreateSessionOptions(options.out()), "CreateSessionOptions");
    throw_if_error(*api,
                   api->SetSessionGraphOptimizationLevel(options.get(),
                                                         kOrtEnableAllGraphOptimizations),
                   "SetSessionGraphOptimizationLevel");
    throw_if_error(*api, api->SetSessionLogSeverityLevel(options.get(), kOrtLoggingLevelWarning),
                   "SetSessionLogSeverityLevel");
    if (provider == OrtExecutionProvider::Cuda) {
      const auto append_cuda_provider = ort_runtime_api().append_cuda_provider;
      if (append_cuda_provider == nullptr) {
        throw std::runtime_error("ONNX Runtime CUDA execution provider entry point is unavailable");
      }
      throw_if_error(*api,
                     api->AddSessionConfigEntry(options.get(),
                                                kOrtSessionOptionsDisableCpuEpFallback, "1"),
                     "AddSessionConfigEntry(session.disable_cpu_ep_fallback)");
      throw_if_error(*api, append_cuda_provider(options.get(), 0),
                     "OrtSessionOptionsAppendExecutionProvider_CUDA");
    }

    const auto model_path = ort_model_path(config.model_path);
    throw_if_error(*api, api->CreateSession(env.get(), model_path.c_str(), options.get(), session.out()),
                   "CreateSession");
    throw_if_error(*api,
                   api->CreateCpuMemoryInfo(kOrtArenaAllocator, kOrtMemTypeDefault,
                                            cpu_memory_info.out()),
                   "CreateCpuMemoryInfo");
    if (provider == OrtExecutionProvider::Cuda) {
      throw_if_error(*api,
                     api->CreateMemoryInfo("Cuda", kOrtDeviceAllocator, 0, kOrtMemTypeDefault,
                                           cuda_memory_info.out()),
                     "CreateMemoryInfo(Cuda)");
    }

    OrtCAllocator* allocator = nullptr;
    throw_if_error(*api, api->GetAllocatorWithDefaultOptions(&allocator),
                   "GetAllocatorWithDefaultOptions");
    metadata.input_names = session_names(*api, session.get(), allocator, true);
    metadata.output_names = session_names(*api, session.get(), allocator, false);
    metadata.input_size = input_size_from_shape(input_shape(*api, session.get()));
    metadata.labels = labels_from_metadata(*api, session.get(), allocator,
                                           std::move(config.fallback_labels));
    if (metadata.input_names.empty()) {
      throw std::runtime_error("ONNX model has no inputs");
    }
    if (metadata.output_names.empty()) {
      throw std::runtime_error("ONNX model has no outputs");
    }
  }

  const OrtApi* api;
  EnvHandle env;
  SessionOptionsHandle options;
  SessionHandle session;
  MemoryInfoHandle cpu_memory_info;
  MemoryInfoHandle cuda_memory_info;
  OrtExecutionProvider provider = OrtExecutionProvider::Cpu;
  OrtSessionMetadata metadata;
};

OrtSession::OrtSession(OrtSessionConfig config)
    : impl_(std::make_unique<Impl>(validate_ort_session_config(std::move(config)))) {}

OrtSession::~OrtSession() = default;

OrtSession::OrtSession(OrtSession&&) noexcept = default;

OrtSession& OrtSession::operator=(OrtSession&&) noexcept = default;

const OrtSessionMetadata& OrtSession::metadata() const { return impl_->metadata; }

std::vector<OrtTensorOutput> OrtSession::run_cpu_f32(std::span<const float> input,
                                                     std::span<const std::int64_t> shape) {
  if (impl_->provider != OrtExecutionProvider::Cpu) {
    throw std::invalid_argument("run_cpu_f32 requires a CPU ORT session");
  }
  if (shape.empty()) {
    throw std::invalid_argument("run_cpu_f32 requires a non-empty input shape");
  }
  std::size_t element_count = 1;
  for (const auto dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument("run_cpu_f32 requires concrete positive input dimensions");
    }
    const auto u_dim = static_cast<std::size_t>(dim);
    if (element_count > std::numeric_limits<std::size_t>::max() / u_dim) {
      throw std::overflow_error("run_cpu_f32 input shape overflows size_t");
    }
    element_count *= u_dim;
  }
  if (input.size() != element_count) {
    throw std::invalid_argument("run_cpu_f32 input length does not match shape");
  }
  const auto byte_count = input.size_bytes();
  ValueHandle input_value(*impl_->api, nullptr);
  throw_if_error(*impl_->api,
                 impl_->api->CreateTensorWithDataAsOrtValue(
                     impl_->cpu_memory_info.get(), const_cast<float*>(input.data()), byte_count,
                     shape.data(), shape.size(), kOnnxTensorElementDataTypeFloat,
                     input_value.out()),
                 "CreateTensorWithDataAsOrtValue");

  std::vector<const char*> input_names;
  input_names.reserve(impl_->metadata.input_names.size());
  for (const auto& name : impl_->metadata.input_names) {
    input_names.push_back(name.c_str());
  }
  std::vector<const char*> output_names;
  output_names.reserve(impl_->metadata.output_names.size());
  for (const auto& name : impl_->metadata.output_names) {
    output_names.push_back(name.c_str());
  }
  std::vector<const OrtCValue*> inputs = {input_value.get()};
  std::vector<OrtCValue*> raw_outputs(output_names.size(), nullptr);
  throw_if_error(*impl_->api,
                 impl_->api->Run(impl_->session.get(), nullptr, input_names.data(), inputs.data(),
                                 inputs.size(), output_names.data(), output_names.size(),
                                 raw_outputs.data()),
                 "Run");

  std::vector<ValueHandle> outputs;
  outputs.reserve(raw_outputs.size());
  for (auto* output : raw_outputs) {
    outputs.emplace_back(*impl_->api, output);
  }

  std::vector<OrtTensorOutput> result;
  result.reserve(outputs.size());
  for (const auto& output : outputs) {
    TensorInfoHandle tensor_info(*impl_->api, nullptr);
    throw_if_error(*impl_->api,
                   impl_->api->GetTensorTypeAndShape(output.get(), tensor_info.out()),
                   "GetTensorTypeAndShape");
    std::size_t rank = 0;
    throw_if_error(*impl_->api, impl_->api->GetDimensionsCount(tensor_info.get(), &rank),
                   "GetDimensionsCount");
    OrtTensorOutput tensor;
    tensor.shape.assign(rank, 0);
    if (rank > 0) {
      throw_if_error(*impl_->api,
                     impl_->api->GetDimensions(tensor_info.get(), tensor.shape.data(),
                                               tensor.shape.size()),
                     "GetDimensions");
    }
    int element_type = 0;
    throw_if_error(*impl_->api,
                   impl_->api->GetTensorElementType(tensor_info.get(), &element_type),
                   "GetTensorElementType");
    if (element_type != kOnnxTensorElementDataTypeFloat) {
      throw std::runtime_error("ORT output tensor is not float32");
    }
    std::size_t output_elements = 1;
    for (const auto dim : tensor.shape) {
      if (dim < 0) {
        throw std::runtime_error("ORT output has unresolved dynamic shape");
      }
      const auto u_dim = static_cast<std::size_t>(dim);
      if (u_dim == 0) {
        output_elements = 0;
        break;
      }
      if (output_elements > std::numeric_limits<std::size_t>::max() / u_dim) {
        throw std::overflow_error("ORT output shape overflows size_t");
      }
      output_elements *= u_dim;
    }
    void* data = nullptr;
    throw_if_error(*impl_->api, impl_->api->GetTensorMutableData(output.get(), &data),
                   "GetTensorMutableData");
    if (output_elements == 0) {
      result.push_back(std::move(tensor));
      continue;
    }
    if (data == nullptr) {
      throw std::runtime_error("ORT output tensor data pointer is null");
    }
    const auto* floats = static_cast<const float*>(data);
    tensor.data.assign(floats, floats + output_elements);
    result.push_back(std::move(tensor));
  }
  return result;
}

std::vector<OrtTensorOutput> OrtSession::run_cuda_f32(core::CudaDevicePtr input,
                                                      std::size_t byte_count,
                                                      std::span<const std::int64_t> shape) {
  if (impl_->provider != OrtExecutionProvider::Cuda) {
    throw std::invalid_argument("run_cuda_f32 requires a CUDA ORT session");
  }
  if (input == 0 || byte_count == 0) {
    throw std::invalid_argument("run_cuda_f32 requires a non-empty CUDA input buffer");
  }
  if (shape.empty()) {
    throw std::invalid_argument("run_cuda_f32 requires a non-empty input shape");
  }
  std::size_t element_count = 1;
  for (const auto dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument("run_cuda_f32 requires concrete positive input dimensions");
    }
    const auto u_dim = static_cast<std::size_t>(dim);
    if (element_count > std::numeric_limits<std::size_t>::max() / u_dim) {
      throw std::overflow_error("run_cuda_f32 input shape overflows size_t");
    }
    element_count *= u_dim;
  }
  if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      byte_count != element_count * sizeof(float)) {
    throw std::invalid_argument("run_cuda_f32 byte count does not match shape");
  }

  ValueHandle input_value(*impl_->api, nullptr);
  throw_if_error(*impl_->api,
                 impl_->api->CreateTensorWithDataAsOrtValue(
                     impl_->cuda_memory_info.get(), reinterpret_cast<void*>(input), byte_count,
                     shape.data(), shape.size(), kOnnxTensorElementDataTypeFloat,
                     input_value.out()),
                 "CreateTensorWithDataAsOrtValue(Cuda)");

  std::vector<const char*> input_names;
  input_names.reserve(impl_->metadata.input_names.size());
  for (const auto& name : impl_->metadata.input_names) {
    input_names.push_back(name.c_str());
  }
  std::vector<const char*> output_names;
  output_names.reserve(impl_->metadata.output_names.size());
  for (const auto& name : impl_->metadata.output_names) {
    output_names.push_back(name.c_str());
  }
  std::vector<const OrtCValue*> inputs = {input_value.get()};
  std::vector<OrtCValue*> raw_outputs(output_names.size(), nullptr);
  throw_if_error(*impl_->api,
                 impl_->api->Run(impl_->session.get(), nullptr, input_names.data(), inputs.data(),
                                 inputs.size(), output_names.data(), output_names.size(),
                                 raw_outputs.data()),
                 "Run");

  std::vector<ValueHandle> outputs;
  outputs.reserve(raw_outputs.size());
  for (auto* output : raw_outputs) {
    outputs.emplace_back(*impl_->api, output);
  }

  std::vector<OrtTensorOutput> result;
  result.reserve(outputs.size());
  for (const auto& output : outputs) {
    TensorInfoHandle tensor_info(*impl_->api, nullptr);
    throw_if_error(*impl_->api,
                   impl_->api->GetTensorTypeAndShape(output.get(), tensor_info.out()),
                   "GetTensorTypeAndShape");
    std::size_t rank = 0;
    throw_if_error(*impl_->api, impl_->api->GetDimensionsCount(tensor_info.get(), &rank),
                   "GetDimensionsCount");
    OrtTensorOutput tensor;
    tensor.shape.assign(rank, 0);
    if (rank > 0) {
      throw_if_error(*impl_->api,
                     impl_->api->GetDimensions(tensor_info.get(), tensor.shape.data(),
                                               tensor.shape.size()),
                     "GetDimensions");
    }
    int element_type = 0;
    throw_if_error(*impl_->api,
                   impl_->api->GetTensorElementType(tensor_info.get(), &element_type),
                   "GetTensorElementType");
    if (element_type != kOnnxTensorElementDataTypeFloat) {
      throw std::runtime_error("ORT output tensor is not float32");
    }
    std::size_t output_elements = 1;
    for (const auto dim : tensor.shape) {
      if (dim < 0) {
        throw std::runtime_error("ORT output has unresolved dynamic shape");
      }
      const auto u_dim = static_cast<std::size_t>(dim);
      if (u_dim == 0) {
        output_elements = 0;
        break;
      }
      if (output_elements > std::numeric_limits<std::size_t>::max() / u_dim) {
        throw std::overflow_error("ORT output shape overflows size_t");
      }
      output_elements *= u_dim;
    }
    void* data = nullptr;
    throw_if_error(*impl_->api, impl_->api->GetTensorMutableData(output.get(), &data),
                   "GetTensorMutableData");
    if (output_elements == 0) {
      result.push_back(std::move(tensor));
      continue;
    }
    if (data == nullptr) {
      throw std::runtime_error("ORT output tensor data pointer is null");
    }
    const auto* floats = static_cast<const float*>(data);
    tensor.data.assign(floats, floats + output_elements);
    result.push_back(std::move(tensor));
  }
  return result;
}

} // namespace reco::detect
