#include "reco/detect/trt_engine.hpp"

#include "reco/core/path.hpp"
#include "reco/core/windows_runtime_library.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#else
#include <dlfcn.h>
#endif

namespace reco::detect {
namespace {

#if defined(_WIN32)
#define RECO_TRT_CALL __stdcall
#else
#define RECO_TRT_CALL
#endif

using TrtCreateLogger = void*(RECO_TRT_CALL*)(int);
using TrtCreateRuntime = void*(RECO_TRT_CALL*)(void*);
using TrtDeserializeEngine = void*(RECO_TRT_CALL*)(void*, const void*, std::size_t);
using TrtCreateExecutionContext = void*(RECO_TRT_CALL*)(void*);
using TrtGetNbBindings = int(RECO_TRT_CALL*)(void*);
using TrtGetBindingName = const char*(RECO_TRT_CALL*)(void*, int);
using TrtBindingIsInput = int(RECO_TRT_CALL*)(void*, int);
using TrtGetBindingDims = void(RECO_TRT_CALL*)(void*, int, int*, std::int32_t*);
using TrtGetBindingDataType = int(RECO_TRT_CALL*)(void*, int);
using TrtEnqueueV2 = int(RECO_TRT_CALL*)(void*, void**, void*);
using TrtDestroy = void(RECO_TRT_CALL*)(void*);

class DynamicLibrary {
public:
  explicit DynamicLibrary(const std::filesystem::path& path) : path_(core::path_to_utf8(path)) {
#if defined(_WIN32)
    handle_ = static_cast<HMODULE>(core::detail::load_windows_runtime_library(path));
#else
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw TrtError("failed to load TensorRT wrapper `" + path_ + "`");
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
      throw TrtError("TensorRT wrapper `" + path_ + "` is missing " + name);
    }
    return reinterpret_cast<Fn>(sym);
  }

private:
  void* handle_ = nullptr;
  std::string path_;
};

struct TrtApi {
  TrtCreateLogger create_logger = nullptr;
  TrtCreateRuntime create_runtime = nullptr;
  TrtDeserializeEngine deserialize_engine = nullptr;
  TrtCreateExecutionContext create_execution_context = nullptr;
  TrtGetNbBindings get_nb_bindings = nullptr;
  TrtGetBindingName get_binding_name = nullptr;
  TrtBindingIsInput binding_is_input = nullptr;
  TrtGetBindingDims get_binding_dims = nullptr;
  TrtGetBindingDataType get_binding_data_type = nullptr;
  TrtEnqueueV2 enqueue_v2 = nullptr;
  TrtDestroy destroy_context = nullptr;
  TrtDestroy destroy_engine = nullptr;
  TrtDestroy destroy_runtime = nullptr;
  TrtDestroy destroy_logger = nullptr;
};

std::filesystem::path getenv_path(const char* name) {
  if (const char* value = std::getenv(name); value != nullptr && value[0] != '\0') {
    return std::filesystem::path(value);
  }
  return {};
}

std::vector<std::uint8_t> read_engine_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw TrtError("failed to open TensorRT engine `" + path.string() + "`");
  }
  file.seekg(0, std::ios::end);
  const auto size = file.tellg();
  if (size <= 0) {
    throw TrtError("TensorRT engine file is empty: `" + path.string() + "`");
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!file) {
    throw TrtError("failed to read TensorRT engine `" + path.string() + "`");
  }
  return data;
}

TrtDataType trt_data_type_from_raw(int value) {
  switch (value) {
  case 0:
    return TrtDataType::Float;
  case 1:
    return TrtDataType::Half;
  case 2:
    return TrtDataType::Int8;
  case 3:
    return TrtDataType::Int32;
  default:
    throw TrtError("unknown TensorRT data type: " + std::to_string(value));
  }
}

const TrtApi& trt_api() {
  static const TrtApi api = [] {
    static std::unique_ptr<DynamicLibrary> pinned_library;
    auto path = getenv_path("TRT_DYLIB_PATH");
    if (path.empty()) {
      path = getenv_path("RECO_TRT_DYLIB_PATH");
    }
    if (path.empty()) {
#if defined(_WIN32)
      path = "reco_tensorrt.dll";
#elif defined(__APPLE__)
      path = "libreco_tensorrt.dylib";
#else
      path = "libreco_tensorrt.so";
#endif
    }
    pinned_library = std::make_unique<DynamicLibrary>(path);
    return TrtApi{
        .create_logger = pinned_library->symbol<TrtCreateLogger>("trt_create_logger"),
        .create_runtime = pinned_library->symbol<TrtCreateRuntime>("trt_create_runtime"),
        .deserialize_engine =
            pinned_library->symbol<TrtDeserializeEngine>("trt_deserialize_engine"),
        .create_execution_context =
            pinned_library->symbol<TrtCreateExecutionContext>("trt_create_execution_context"),
        .get_nb_bindings = pinned_library->symbol<TrtGetNbBindings>("trt_get_nb_bindings"),
        .get_binding_name = pinned_library->symbol<TrtGetBindingName>("trt_get_binding_name"),
        .binding_is_input = pinned_library->symbol<TrtBindingIsInput>("trt_binding_is_input"),
        .get_binding_dims = pinned_library->symbol<TrtGetBindingDims>("trt_get_binding_dims"),
        .get_binding_data_type =
            pinned_library->symbol<TrtGetBindingDataType>("trt_get_binding_data_type"),
        .enqueue_v2 = pinned_library->symbol<TrtEnqueueV2>("trt_enqueue_v2"),
        .destroy_context = pinned_library->symbol<TrtDestroy>("trt_destroy_context"),
        .destroy_engine = pinned_library->symbol<TrtDestroy>("trt_destroy_engine"),
        .destroy_runtime = pinned_library->symbol<TrtDestroy>("trt_destroy_runtime"),
        .destroy_logger = pinned_library->symbol<TrtDestroy>("trt_destroy_logger"),
    };
  }();
  return api;
}

} // namespace

struct TrtEngineState {
  ~TrtEngineState();

  void* engine = nullptr;
  void* runtime = nullptr;
  void* logger = nullptr;
};

TrtEngineState::~TrtEngineState() {
  const auto& api = trt_api();
  if (engine != nullptr) {
    api.destroy_engine(engine);
  }
  if (runtime != nullptr) {
    api.destroy_runtime(runtime);
  }
  if (logger != nullptr) {
    api.destroy_logger(logger);
  }
}

std::size_t trt_data_type_byte_size(TrtDataType data_type) {
  switch (data_type) {
  case TrtDataType::Float:
  case TrtDataType::Int32:
    return 4;
  case TrtDataType::Half:
    return 2;
  case TrtDataType::Int8:
    return 1;
  }
  throw TrtError("unknown TensorRT data type");
}

TrtContext::TrtContext(void* context, std::shared_ptr<const TrtEngineState> engine_state,
                       std::size_t binding_count)
    : context_(context), engine_state_(std::move(engine_state)), binding_count_(binding_count) {}

TrtContext::~TrtContext() {
  if (context_ != nullptr) {
    trt_api().destroy_context(context_);
  }
}

TrtContext::TrtContext(TrtContext&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      engine_state_(std::move(other.engine_state_)),
      binding_count_(std::exchange(other.binding_count_, 0)) {}

TrtContext& TrtContext::operator=(TrtContext&& other) noexcept {
  if (this != &other) {
    if (context_ != nullptr) {
      trt_api().destroy_context(context_);
    }
    context_ = std::exchange(other.context_, nullptr);
    engine_state_ = std::move(other.engine_state_);
    binding_count_ = std::exchange(other.binding_count_, 0);
  }
  return *this;
}

void TrtContext::enqueue(std::span<void*> bindings, void* stream) const {
  if (context_ == nullptr) {
    throw TrtError("TensorRT context is null");
  }
  if (bindings.size() != binding_count_) {
    throw TrtError("TensorRT enqueue binding count does not match engine binding count");
  }
  if (trt_api().enqueue_v2(context_, bindings.data(), stream) != 0) {
    throw TrtError("TensorRT enqueue failed");
  }
}

TrtEngine::TrtEngine(const std::filesystem::path& engine_path) {
  const auto data = read_engine_file(engine_path);
  const auto& api = trt_api();
  auto state = std::make_shared<TrtEngineState>();
  state->logger = api.create_logger(2);
  if (state->logger == nullptr) {
    throw TrtError("failed to create TensorRT logger");
  }
  state->runtime = api.create_runtime(state->logger);
  if (state->runtime == nullptr) {
    throw TrtError("failed to create TensorRT runtime");
  }
  state->engine = api.deserialize_engine(state->runtime, data.data(), data.size());
  if (state->engine == nullptr) {
    throw TrtError("failed to deserialize TensorRT engine `" + engine_path.string() + "`");
  }
  state_ = std::move(state);
}

TrtEngine::~TrtEngine() = default;

TrtEngine::TrtEngine(TrtEngine&& other) noexcept : state_(std::move(other.state_)) {}

TrtEngine& TrtEngine::operator=(TrtEngine&& other) noexcept {
  if (this != &other) {
    state_ = std::move(other.state_);
  }
  return *this;
}

std::vector<TrtBindingInfo> TrtEngine::bindings() const {
  if (!state_ || state_->engine == nullptr) {
    throw TrtError("TensorRT engine is null");
  }
  const auto& api = trt_api();
  const int count = api.get_nb_bindings(state_->engine);
  if (count < 0) {
    throw TrtError("TensorRT returned a negative binding count");
  }
  std::vector<TrtBindingInfo> result;
  result.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const char* name_ptr = api.get_binding_name(state_->engine, i);
    int rank = 0;
    std::int32_t dims[8] = {};
    api.get_binding_dims(state_->engine, i, &rank, dims);
    if (rank < 0 || rank > 8) {
      throw TrtError("TensorRT binding `" + std::to_string(i) + "` returned invalid rank");
    }
    auto data_type = trt_data_type_from_raw(api.get_binding_data_type(state_->engine, i));
    std::size_t elements = 1;
    for (int d = 0; d < rank; ++d) {
      const auto dim = static_cast<std::size_t>(std::max<std::int32_t>(dims[d], 1));
      if (elements > std::numeric_limits<std::size_t>::max() / dim) {
        throw TrtError("TensorRT binding dimensions overflow element count");
      }
      elements *= dim;
    }
    const auto element_size = trt_data_type_byte_size(data_type);
    if (elements > std::numeric_limits<std::size_t>::max() / element_size) {
      throw TrtError("TensorRT binding byte size overflows size_t");
    }
    result.push_back(TrtBindingInfo{
        .name = name_ptr == nullptr ? "binding_" + std::to_string(i) : std::string(name_ptr),
        .is_input = api.binding_is_input(state_->engine, i) != 0,
        .dims = std::vector<std::int32_t>(dims, dims + rank),
        .data_type = data_type,
        .byte_size = elements * element_size,
    });
  }
  return result;
}

TrtContext TrtEngine::create_context() const {
  if (!state_ || state_->engine == nullptr) {
    throw TrtError("TensorRT engine is null");
  }
  const int count = trt_api().get_nb_bindings(state_->engine);
  if (count <= 0) {
    throw TrtError("TensorRT context requires at least one binding");
  }
  void* context = trt_api().create_execution_context(state_->engine);
  if (context == nullptr) {
    throw TrtError("failed to create TensorRT execution context");
  }
  return TrtContext(context, state_, static_cast<std::size_t>(count));
}

bool trt_runtime_available() {
  try {
    (void)trt_api();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

std::string trt_runtime_error() {
  try {
    (void)trt_api();
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}

} // namespace reco::detect
