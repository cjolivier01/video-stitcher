#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#define RECO_ORT_CALL __stdcall
#define RECO_ORT_EXPORT extern "C" __declspec(dllexport)
#else
#define RECO_ORT_CALL
#define RECO_ORT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct OrtCEnv {};
struct OrtCSessionOptions {};
struct OrtCMemoryInfo {};
struct OrtCModelMetadata {};
struct OrtCTypeInfo {};

struct OrtCStatus {
  std::string message;
};

struct OrtCTensorTypeAndShapeInfo {
  std::vector<std::int64_t> shape;
  int element_type = 1;
};

struct OrtCSession {
  std::string model_path;
};

struct OrtCValue {
  std::vector<std::int64_t> shape;
  std::vector<float> f32;
  std::vector<std::int32_t> i32;
  int element_type = 1;
};

struct OrtCAllocator {};
struct OrtCRunOptions;
struct OrtCCustomOpDomain;
using OrtStatusPtr = OrtCStatus*;

struct OrtApiBase {
  const void*(RECO_ORT_CALL* get_api)(std::uint32_t);
  const char*(RECO_ORT_CALL* get_version_string)();
};

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
  void* CreateMemoryInfo;
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
};

OrtCStatus* ok() { return nullptr; }

OrtCStatus* fail(const char* message) { return new OrtCStatus{message}; }

char* copy_c_string(const char* text) {
  const auto len = std::strlen(text) + 1;
  auto* out = static_cast<char*>(std::malloc(len));
  if (out != nullptr) {
    std::memcpy(out, text, len);
  }
  return out;
}

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool near(float actual, float expected) { return std::abs(actual - expected) <= 1.0e-5F; }

std::size_t element_count(const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

OrtStatusPtr validate_nv12_2x2_preprocess(const OrtCValue* input) {
  if (input == nullptr || input->shape != std::vector<std::int64_t>({1, 3, 8, 8}) ||
      input->f32.size() != 192) {
    return fail("unexpected 2x2 preprocessed tensor shape");
  }
  constexpr std::size_t plane = 64;
  const auto at = [&](std::size_t channel, std::size_t x, std::size_t y) {
    return input->f32[channel * plane + y * 8 + x];
  };
  for (std::size_t channel = 0; channel < 3; ++channel) {
    if (!near(at(channel, 0, 0), 80.0F / 255.0F) ||
        !near(at(channel, 4, 0), 90.0F / 255.0F) ||
        !near(at(channel, 0, 4), 100.0F / 255.0F) ||
        !near(at(channel, 4, 4), 110.0F / 255.0F)) {
      return fail("2x2 NV12 preprocessing did not produce expected neutral RGB CHW samples");
    }
  }
  return ok();
}

OrtStatusPtr validate_yuv420p_4x2_letterbox(const OrtCValue* input) {
  if (input == nullptr || input->shape != std::vector<std::int64_t>({1, 3, 8, 8}) ||
      input->f32.size() != 192) {
    return fail("unexpected 4x2 preprocessed tensor shape");
  }
  constexpr std::size_t plane = 64;
  const auto at = [&](std::size_t channel, std::size_t x, std::size_t y) {
    return input->f32[channel * plane + y * 8 + x];
  };
  for (std::size_t channel = 0; channel < 3; ++channel) {
    if (!near(at(channel, 0, 0), 114.0F / 255.0F) ||
        !near(at(channel, 7, 1), 114.0F / 255.0F) ||
        !near(at(channel, 0, 2), 10.0F / 255.0F) ||
        !near(at(channel, 2, 2), 20.0F / 255.0F) ||
        !near(at(channel, 0, 4), 50.0F / 255.0F)) {
      return fail("4x2 YUV420p preprocessing did not preserve letterbox/CHW samples");
    }
  }
  return ok();
}

OrtStatusPtr RECO_ORT_CALL create_env(int, const char*, OrtCEnv** out) {
  *out = new OrtCEnv;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL create_session_options(OrtCSessionOptions** out) {
  *out = new OrtCSessionOptions;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL create_session(const OrtCEnv*, const OrtPathChar* model_path,
                                          const OrtCSessionOptions*, OrtCSession** out) {
  if (model_path == nullptr || !std::filesystem::exists(model_path)) {
    return fail("model not found");
  }
  *out = new OrtCSession{std::filesystem::path(model_path).string()};
  return ok();
}

OrtStatusPtr RECO_ORT_CALL set_session_int(OrtCSessionOptions*, int) { return ok(); }

OrtStatusPtr RECO_ORT_CALL create_cpu_memory_info(int, int, OrtCMemoryInfo** out) {
  *out = new OrtCMemoryInfo;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL get_default_allocator(OrtCAllocator** out) {
  static OrtCAllocator allocator;
  *out = &allocator;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL allocator_free(OrtCAllocator*, void* ptr) {
  std::free(ptr);
  return ok();
}

OrtStatusPtr RECO_ORT_CALL input_count(const OrtCSession*, std::size_t* out) {
  *out = 1;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL output_count(const OrtCSession*, std::size_t* out) {
  *out = 1;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL input_name(const OrtCSession*, std::size_t index, OrtCAllocator*,
                                      char** out) {
  if (index != 0) {
    return fail("bad input index");
  }
  *out = copy_c_string("images");
  return ok();
}

OrtStatusPtr RECO_ORT_CALL output_name(const OrtCSession*, std::size_t index, OrtCAllocator*,
                                       char** out) {
  if (index != 0) {
    return fail("bad output index");
  }
  *out = copy_c_string("detections");
  return ok();
}

OrtStatusPtr RECO_ORT_CALL input_type_info(const OrtCSession*, std::size_t index,
                                           OrtCTypeInfo** out) {
  if (index != 0) {
    return fail("bad input index");
  }
  *out = new OrtCTypeInfo;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL cast_type_info(const OrtCTypeInfo*,
                                          const OrtCTensorTypeAndShapeInfo** out) {
  static const OrtCTensorTypeAndShapeInfo input_info{{1, 3, 8, 8}, 1};
  *out = &input_info;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL dimensions_count(const OrtCTensorTypeAndShapeInfo* info,
                                            std::size_t* out) {
  *out = info->shape.size();
  return ok();
}

OrtStatusPtr RECO_ORT_CALL dimensions(const OrtCTensorTypeAndShapeInfo* info, std::int64_t* out,
                                      std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    out[i] = info->shape[i];
  }
  return ok();
}

OrtStatusPtr RECO_ORT_CALL tensor_element_type(const OrtCTensorTypeAndShapeInfo* info, int* out) {
  *out = info->element_type;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL model_metadata(const OrtCSession*, OrtCModelMetadata** out) {
  if (env_set("RECO_FAKE_ORT_METADATA_FAIL")) {
    return fail("metadata unavailable");
  }
  *out = new OrtCModelMetadata;
  return ok();
}

OrtStatusPtr RECO_ORT_CALL metadata_lookup(const OrtCModelMetadata*, OrtCAllocator*, const char* key,
                                           char** out) {
  if (env_set("RECO_FAKE_ORT_NAMES_MISSING")) {
    *out = nullptr;
    return ok();
  }
  if (std::strcmp(key, "names") != 0) {
    *out = nullptr;
    return ok();
  }
  *out = copy_c_string("{0: 'ball', 2: 'player'}");
  return ok();
}

OrtStatusPtr RECO_ORT_CALL create_tensor_with_data(const OrtCMemoryInfo*, void* data,
                                                   std::size_t byte_count,
                                                   const std::int64_t* shape, std::size_t rank,
                                                   int element_type, OrtCValue** out) {
  *out = new OrtCValue;
  (*out)->shape.assign(shape, shape + rank);
  (*out)->element_type = element_type;
  if (element_type == 1) {
    const auto count = element_count((*out)->shape);
    if (count > 0 && byte_count == count * sizeof(float) && data != nullptr) {
      const auto* floats = static_cast<const float*>(data);
      (*out)->f32.assign(floats, floats + count);
    }
  }
  return ok();
}

OrtStatusPtr RECO_ORT_CALL run(OrtCSession*, const OrtCRunOptions*, const char* const*,
                               const OrtCValue* const* inputs, std::size_t input_count,
                               const char* const*,
                               std::size_t output_count, OrtCValue** outputs) {
  if (input_count != 1) {
    return fail("unexpected input count");
  }
  if (env_set("RECO_FAKE_ORT_VALIDATE_NV12_2X2")) {
    if (auto* status = validate_nv12_2x2_preprocess(inputs[0]); status != nullptr) {
      return status;
    }
  }
  if (env_set("RECO_FAKE_ORT_VALIDATE_YUV420P_4X2")) {
    if (auto* status = validate_yuv420p_4x2_letterbox(inputs[0]); status != nullptr) {
      return status;
    }
  }
  for (std::size_t i = 0; i < output_count; ++i) {
    auto* value = new OrtCValue;
    if (env_set("RECO_FAKE_ORT_INT_OUTPUT")) {
      value->shape = {1, 1};
      value->element_type = 6;
      value->i32 = {7};
    } else if (env_set("RECO_FAKE_ORT_EMPTY_OUTPUT")) {
      value->shape = {0};
      value->element_type = 1;
    } else {
      value->shape = {1, 2, 6};
      value->element_type = 1;
      value->f32 = {1.0F, 2.0F, 3.0F, 4.0F, 0.9F, 0.0F,
                    5.0F, 6.0F, 7.0F, 8.0F, 0.8F, 2.0F};
    }
    outputs[i] = value;
  }
  return ok();
}

OrtStatusPtr RECO_ORT_CALL tensor_type_and_shape(const OrtCValue* value,
                                                 OrtCTensorTypeAndShapeInfo** out) {
  *out = new OrtCTensorTypeAndShapeInfo{value->shape, value->element_type};
  return ok();
}

OrtStatusPtr RECO_ORT_CALL tensor_data(OrtCValue* value, void** out) {
  if (value->element_type == 1) {
    *out = value->f32.empty() ? nullptr : value->f32.data();
  } else {
    *out = value->i32.empty() ? nullptr : value->i32.data();
  }
  return ok();
}

void RECO_ORT_CALL release_env(OrtCEnv* value) { delete value; }
void RECO_ORT_CALL release_status(OrtCStatus* value) { delete value; }
void RECO_ORT_CALL release_memory_info(OrtCMemoryInfo* value) { delete value; }
void RECO_ORT_CALL release_session(OrtCSession* value) { delete value; }
void RECO_ORT_CALL release_value(OrtCValue* value) { delete value; }
void RECO_ORT_CALL release_type_info(OrtCTypeInfo* value) { delete value; }
void RECO_ORT_CALL release_tensor_info(OrtCTensorTypeAndShapeInfo* value) { delete value; }
void RECO_ORT_CALL release_session_options(OrtCSessionOptions* value) { delete value; }
void RECO_ORT_CALL release_metadata(OrtCModelMetadata* value) { delete value; }

const void* RECO_ORT_CALL get_api(std::uint32_t version);
const char* RECO_ORT_CALL get_version_string() { return "1.23.2"; }

OrtApi make_api() {
  OrtApi api{};
  api.GetErrorMessage = [](const OrtCStatus* status) -> const char* {
    return status->message.c_str();
  };
  api.CreateEnv = create_env;
  api.CreateSession = create_session;
  api.Run = run;
  api.CreateSessionOptions = create_session_options;
  api.SetSessionLogSeverityLevel = set_session_int;
  api.SetSessionGraphOptimizationLevel = set_session_int;
  api.SessionGetInputCount = input_count;
  api.SessionGetOutputCount = output_count;
  api.SessionGetInputTypeInfo = input_type_info;
  api.SessionGetInputName = input_name;
  api.SessionGetOutputName = output_name;
  api.CreateTensorWithDataAsOrtValue = create_tensor_with_data;
  api.GetTensorMutableData = tensor_data;
  api.CastTypeInfoToTensorInfo = cast_type_info;
  api.GetTensorElementType = tensor_element_type;
  api.GetDimensionsCount = dimensions_count;
  api.GetDimensions = dimensions;
  api.GetTensorTypeAndShape = tensor_type_and_shape;
  api.CreateCpuMemoryInfo = create_cpu_memory_info;
  api.AllocatorFree = allocator_free;
  api.GetAllocatorWithDefaultOptions = get_default_allocator;
  api.ReleaseEnv = release_env;
  api.ReleaseStatus = release_status;
  api.ReleaseMemoryInfo = release_memory_info;
  api.ReleaseSession = release_session;
  api.ReleaseValue = release_value;
  api.ReleaseTypeInfo = release_type_info;
  api.ReleaseTensorTypeAndShapeInfo = release_tensor_info;
  api.ReleaseSessionOptions = release_session_options;
  api.SessionGetModelMetadata = model_metadata;
  api.ModelMetadataLookupCustomMetadataMap = metadata_lookup;
  api.ReleaseModelMetadata = release_metadata;
  return api;
}

const OrtApi api = make_api();

const OrtApiBase api_base = {get_api, get_version_string};

const void* RECO_ORT_CALL get_api(std::uint32_t version) {
  return version == 23 ? &api : nullptr;
}

} // namespace

RECO_ORT_EXPORT const OrtApiBase* RECO_ORT_CALL OrtGetApiBase() { return &api_base; }
