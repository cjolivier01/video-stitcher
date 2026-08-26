#include "reco/io/detail/nvbufsurface_9_1.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>

#if defined(_WIN32)
#define RECO_FAKE_EXPORT extern "C" __declspec(dllexport)
#else
#define RECO_FAKE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

namespace abi = reco::io::detail::nvbufsurface_9_1;

struct GErrorAbi {
  std::uint32_t domain = 0;
  std::int32_t code = 0;
  char* message = nullptr;
};

struct GstMapInfoAbi {
  void* memory = nullptr;
  std::uint32_t flags = 0;
  std::uint8_t* data = nullptr;
  std::size_t size = 0;
  std::size_t max_size = 0;
  std::array<void*, 4> user_data{};
  std::array<void*, 4> reserved{};
};

struct GstMiniObjectAbi {
  std::uintptr_t type = 0;
  std::int32_t ref_count = 1;
  std::int32_t lock_state = 0;
  std::uint32_t flags = 0;
  void* copy = nullptr;
  void* dispose = nullptr;
  void* free = nullptr;
  std::uint32_t private_uint = 0;
  void* private_pointer = nullptr;
};

struct GstBufferAbi {
  GstMiniObjectAbi mini_object;
  void* pool = nullptr;
  std::uint64_t pts = 1'000'000'000;
  std::uint64_t dts = 1'000'000'000;
  std::uint64_t duration = 33'333'333;
  std::uint64_t offset = 0;
  std::uint64_t offset_end = 0;
};

enum class ObjectKind {
  Pipeline,
  Sink,
  Bus,
};

struct FakeObject {
  explicit FakeObject(ObjectKind object_kind) : kind(object_kind) {}
  ObjectKind kind;
};

struct FakePipeline : FakeObject {
  FakePipeline() : FakeObject(ObjectKind::Pipeline) {}
};

struct FakeSink : FakeObject {
  FakeSink() : FakeObject(ObjectKind::Sink) {}
  std::uint32_t pull_count = 0;
};

struct FakeBus : FakeObject {
  FakeBus() : FakeObject(ObjectKind::Bus) {}
  bool emitted_error = false;
  std::uint32_t poll_count = 0;
};

struct FakeMessage {};

struct FakeSample {
  GstBufferAbi buffer;
  abi::SurfaceParams params;
  abi::Surface surface;
};

std::mutex event_mutex;

std::string scenario() {
  const char* value = std::getenv("RECO_FAKE_GST_SCENARIO");
  return value == nullptr ? "frame-eos" : value;
}

void record(std::string_view event) {
  const char* path = std::getenv("RECO_FAKE_GST_EVENT_PATH");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  std::lock_guard lock(event_mutex);
  std::ofstream output(path, std::ios::app);
  output << event << '\n';
}

char* duplicate(const char* value) {
  const auto size = std::strlen(value) + 1;
  auto* result = static_cast<char*>(std::malloc(size));
  if (result != nullptr) {
    std::memcpy(result, value, size);
  }
  return result;
}

GErrorAbi* make_error(const char* message) {
  auto* error = static_cast<GErrorAbi*>(std::malloc(sizeof(GErrorAbi)));
  if (error == nullptr) {
    return nullptr;
  }
  *error = {.domain = 1, .code = 2, .message = duplicate(message)};
  return error;
}

FakeSample* make_sample() {
  auto* sample = new FakeSample;
  if (scenario() == "unknown-time") {
    sample->buffer.pts = std::numeric_limits<std::uint64_t>::max();
    sample->buffer.duration = std::numeric_limits<std::uint64_t>::max();
  }

  sample->params.width = 1280;
  sample->params.height = 720;
  sample->params.pitch = 1280;
  sample->params.color_format = abi::kColorNv12_709;
  sample->params.layout = abi::kLayoutPitch;
  sample->params.data_size = 1280 * 1080;
  sample->params.data_ptr = reinterpret_cast<void*>(0x10000000);
  sample->params.plane_params.num_planes = 2;
  sample->params.plane_params.width[0] = 1280;
  sample->params.plane_params.height[0] = 720;
  sample->params.plane_params.pitch[0] = 1280;
  sample->params.plane_params.psize[0] = 1280 * 720;
  sample->params.plane_params.bytes_per_pix[0] = 1;
  sample->params.plane_params.width[1] = 640;
  sample->params.plane_params.height[1] = 360;
  sample->params.plane_params.pitch[1] = 1280;
  sample->params.plane_params.offset[1] = 1280 * 720;
  sample->params.plane_params.psize[1] = 1280 * 360;
  sample->params.plane_params.bytes_per_pix[1] = 2;
  sample->surface.gpu_id = 0;
  sample->surface.batch_size = 1;
  sample->surface.num_filled = scenario() == "invalid-surface" ? 0 : 1;
  sample->surface.mem_type = abi::kMemCudaDevice;
  sample->surface.surface_list = &sample->params;
  return sample;
}

} // namespace

RECO_FAKE_EXPORT void gst_version(std::uint32_t* major, std::uint32_t* minor, std::uint32_t* micro,
                                  std::uint32_t* nano) {
  *major = 1;
  *minor = scenario() == "old-version" ? 8 : 28;
  *micro = 2;
  *nano = 0;
}

RECO_FAKE_EXPORT int gst_init_check(int*, char***, GErrorAbi** error) {
  record("init");
  if (scenario() != "init-error") {
    return 1;
  }
  *error = make_error("fake initialization failure");
  return 0;
}

RECO_FAKE_EXPORT void* gst_parse_launch(const char*, GErrorAbi** error) {
  record("parse");
  if (scenario() == "parse-error") {
    *error = make_error("fake parse failure");
    return nullptr;
  }
  return new FakePipeline;
}

RECO_FAKE_EXPORT void* gst_bin_get_by_name(void*, const char*) {
  record("get-sink");
  if (scenario() == "missing-sink") {
    return nullptr;
  }
  return new FakeSink;
}

RECO_FAKE_EXPORT int gst_element_set_state(void*, int state) {
  record(state == 4 ? "state-playing" : "state-null");
  if (state == 4 && scenario() == "state-error") {
    return 0;
  }
  return 1;
}

RECO_FAKE_EXPORT void* gst_element_get_bus(void*) {
  record("get-bus");
  if (scenario() == "missing-bus") {
    return nullptr;
  }
  return new FakeBus;
}

RECO_FAKE_EXPORT void gst_object_unref(void* object) {
  if (object == nullptr) {
    return;
  }
  auto* base = static_cast<FakeObject*>(object);
  switch (base->kind) {
  case ObjectKind::Pipeline:
    record("unref-pipeline");
    delete static_cast<FakePipeline*>(object);
    break;
  case ObjectKind::Sink:
    record("unref-sink");
    delete static_cast<FakeSink*>(object);
    break;
  case ObjectKind::Bus:
    record("unref-bus");
    delete static_cast<FakeBus*>(object);
    break;
  }
}

RECO_FAKE_EXPORT void* gst_app_sink_try_pull_sample(void* sink_pointer, std::uint64_t) {
  record("pull");
  auto* sink = static_cast<FakeSink*>(sink_pointer);
  const auto current = sink->pull_count++;
  const auto current_scenario = scenario();
  if ((current_scenario == "frame-eos" || current_scenario == "unknown-time" ||
       current_scenario == "missing-buffer" || current_scenario == "map-error" ||
       current_scenario == "invalid-surface") &&
      current == 0) {
    return make_sample();
  }
  return nullptr;
}

RECO_FAKE_EXPORT int gst_app_sink_is_eos(void* sink_pointer) {
  const auto current_scenario = scenario();
  const auto* sink = static_cast<FakeSink*>(sink_pointer);
  return (current_scenario == "frame-eos" || current_scenario == "unknown-time") &&
         sink->pull_count >= 2;
}

RECO_FAKE_EXPORT void* gst_sample_get_buffer(void* sample) {
  if (scenario() == "missing-buffer") {
    return nullptr;
  }
  return &static_cast<FakeSample*>(sample)->buffer;
}

RECO_FAKE_EXPORT void gst_sample_unref(void* sample) {
  record("sample-unref");
  delete static_cast<FakeSample*>(sample);
}

RECO_FAKE_EXPORT int gst_buffer_map(void* buffer, GstMapInfoAbi* map, std::uint32_t) {
  record("map");
  if (scenario() == "map-error") {
    return 0;
  }
  auto* sample = reinterpret_cast<FakeSample*>(buffer);
  map->data = reinterpret_cast<std::uint8_t*>(&sample->surface);
  map->size = sizeof(sample->surface);
  map->max_size = sizeof(sample->surface);
  return 1;
}

RECO_FAKE_EXPORT void gst_buffer_unmap(void*, GstMapInfoAbi*) { record("unmap"); }

RECO_FAKE_EXPORT void* gst_bus_timed_pop_filtered(void* bus_pointer, std::uint64_t,
                                                  std::uint32_t types) {
  auto* bus = static_cast<FakeBus*>(bus_pointer);
  ++bus->poll_count;
  const bool ready = scenario() == "stream-error" ||
                     (scenario() == "delayed-stream-error" && bus->poll_count >= 2);
  if (!ready || bus->emitted_error || (types & (1U << 1U)) == 0) {
    return nullptr;
  }
  bus->emitted_error = true;
  return new FakeMessage;
}

RECO_FAKE_EXPORT void gst_message_parse_error(void*, GErrorAbi** error, char** debug) {
  *error = make_error("fake decoder failure");
  *debug = duplicate("fake debug detail");
}

RECO_FAKE_EXPORT void gst_message_unref(void* message) {
  delete static_cast<FakeMessage*>(message);
}

RECO_FAKE_EXPORT void g_error_free(GErrorAbi* error) {
  if (error != nullptr) {
    std::free(error->message);
    std::free(error);
  }
}

RECO_FAKE_EXPORT void g_free(void* pointer) { std::free(pointer); }
