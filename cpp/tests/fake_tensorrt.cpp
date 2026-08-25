#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#define RECO_TRT_CALL __stdcall
#define RECO_TRT_EXPORT extern "C" __declspec(dllexport)
#else
#define RECO_TRT_CALL
#define RECO_TRT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct Logger {
  int severity = 0;
};

struct Runtime {
  Logger* logger = nullptr;
};

struct Binding {
  std::string name;
  bool input = false;
  std::vector<std::int32_t> dims;
  int data_type = 0;
};

struct Engine {
  std::vector<Binding> bindings;
};

struct Context {
  Engine* engine = nullptr;
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "0") != 0;
}

std::vector<Binding> bindings_for_marker(const void* data, std::size_t size) {
  const std::string marker(static_cast<const char*>(data), static_cast<const char*>(data) + size);
  if (marker.find("overflow") != std::string::npos) {
    return {
        {"images", true, {std::numeric_limits<std::int32_t>::max(),
                          std::numeric_limits<std::int32_t>::max(),
                          std::numeric_limits<std::int32_t>::max(),
                          std::numeric_limits<std::int32_t>::max()},
         0},
    };
  }
  if (marker.find("bad_dtype") != std::string::npos) {
    return {{"images", true, {1, 3, 8, 8}, 99}};
  }
  if (marker.find("bad_rank") != std::string::npos) {
    return {{"images", true, {1, 3, 8, 8, 1, 1, 1, 1, 1}, 0}};
  }
  if (marker.find("batch2") != std::string::npos) {
    return {
        {"images", true, {2, 3, 8, 8}, 0},
        {"detections", false, {1, 300, 6}, 0},
    };
  }
  if (marker.find("negative_count") != std::string::npos) {
    return {};
  }
  return {
      {"images", true, {1, 3, 8, 8}, 0},
      {"detections", false, {1, 300, 6}, 0},
      {"shape_hint", false, {-1, 16}, 1},
  };
}

} // namespace

RECO_TRT_EXPORT void* RECO_TRT_CALL trt_create_logger(int severity) {
  if (env_set("RECO_FAKE_TRT_LOGGER_FAIL")) {
    return nullptr;
  }
  return new Logger{severity};
}

RECO_TRT_EXPORT void* RECO_TRT_CALL trt_create_runtime(void* logger) {
  if (env_set("RECO_FAKE_TRT_RUNTIME_FAIL")) {
    return nullptr;
  }
  return new Runtime{static_cast<Logger*>(logger)};
}

RECO_TRT_EXPORT void* RECO_TRT_CALL trt_deserialize_engine(void*, const void* data,
                                                          std::size_t size) {
  if (env_set("RECO_FAKE_TRT_DESERIALIZE_FAIL") || data == nullptr || size == 0) {
    return nullptr;
  }
  return new Engine{bindings_for_marker(data, size)};
}

RECO_TRT_EXPORT void* RECO_TRT_CALL trt_create_execution_context(void* engine) {
  if (env_set("RECO_FAKE_TRT_CONTEXT_FAIL")) {
    return nullptr;
  }
  return new Context{static_cast<Engine*>(engine)};
}

RECO_TRT_EXPORT int RECO_TRT_CALL trt_get_nb_bindings(void* engine) {
  if (env_set("RECO_FAKE_TRT_NEGATIVE_BINDING_COUNT")) {
    return -1;
  }
  return static_cast<int>(static_cast<Engine*>(engine)->bindings.size());
}

RECO_TRT_EXPORT const char* RECO_TRT_CALL trt_get_binding_name(void* engine, int index) {
  auto& bindings = static_cast<Engine*>(engine)->bindings;
  if (index < 0 || static_cast<std::size_t>(index) >= bindings.size()) {
    return nullptr;
  }
  return bindings[static_cast<std::size_t>(index)].name.c_str();
}

RECO_TRT_EXPORT int RECO_TRT_CALL trt_binding_is_input(void* engine, int index) {
  auto& bindings = static_cast<Engine*>(engine)->bindings;
  if (index < 0 || static_cast<std::size_t>(index) >= bindings.size()) {
    return 0;
  }
  return bindings[static_cast<std::size_t>(index)].input ? 1 : 0;
}

RECO_TRT_EXPORT void RECO_TRT_CALL trt_get_binding_dims(void* engine, int index, int* rank,
                                                       std::int32_t* dims) {
  auto& bindings = static_cast<Engine*>(engine)->bindings;
  if (index < 0 || static_cast<std::size_t>(index) >= bindings.size()) {
    *rank = 0;
    return;
  }
  const auto& src = bindings[static_cast<std::size_t>(index)].dims;
  *rank = static_cast<int>(src.size());
  for (std::size_t i = 0; i < src.size() && i < 8; ++i) {
    dims[i] = src[i];
  }
}

RECO_TRT_EXPORT int RECO_TRT_CALL trt_get_binding_data_type(void* engine, int index) {
  auto& bindings = static_cast<Engine*>(engine)->bindings;
  if (index < 0 || static_cast<std::size_t>(index) >= bindings.size()) {
    return 0;
  }
  return bindings[static_cast<std::size_t>(index)].data_type;
}

RECO_TRT_EXPORT int RECO_TRT_CALL trt_enqueue_v2(void*, void** bindings, void* stream) {
  if (env_set("RECO_FAKE_TRT_ENQUEUE_FAIL")) {
    return 1;
  }
  if (env_set("RECO_FAKE_TRT_VALIDATE_BINDINGS")) {
    if (bindings == nullptr || bindings[0] != reinterpret_cast<void*>(0x12340000U) ||
        bindings[1] != reinterpret_cast<void*>(0x56780000U) ||
        stream != reinterpret_cast<void*>(0x90120000U)) {
      return 2;
    }
  }
  if (env_set("RECO_FAKE_TRT_VALIDATE_DETECTOR_BINDINGS")) {
    if (bindings == nullptr || bindings[0] == nullptr || bindings[1] == nullptr ||
        bindings[2] != nullptr) {
      return 3;
    }
  }
  return 0;
}

RECO_TRT_EXPORT void RECO_TRT_CALL trt_destroy_context(void* value) {
  delete static_cast<Context*>(value);
}

RECO_TRT_EXPORT void RECO_TRT_CALL trt_destroy_engine(void* value) {
  delete static_cast<Engine*>(value);
}

RECO_TRT_EXPORT void RECO_TRT_CALL trt_destroy_runtime(void* value) {
  delete static_cast<Runtime*>(value);
}

RECO_TRT_EXPORT void RECO_TRT_CALL trt_destroy_logger(void* value) {
  delete static_cast<Logger*>(value);
}
