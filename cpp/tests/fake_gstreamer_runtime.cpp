#include "reco/io/detail/nvbufsurface_9_1.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

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
  void* qdata = nullptr;
  void (*qdata_destroy)(void*) = nullptr;
};

enum class ObjectKind {
  Pipeline,
  Sink,
  Bus,
  DisplayInfo,
  OutputInfo,
  ProbeInfo,
  Pad,
};

struct FakeObject {
  explicit FakeObject(ObjectKind object_kind) : kind(object_kind) {}
  ObjectKind kind;
};

struct FakePipeline;

using FakePadProbeCallback = int (*)(void*, void*, void*);
using FakeDestroyNotify = void (*)(void*);

struct FakeSink : FakeObject {
  explicit FakeSink(FakePipeline* owner) : FakeObject(ObjectKind::Sink), pipeline(owner) {}
  FakePipeline* pipeline = nullptr;
  std::uint32_t pull_count = 0;
  bool probe_eos = false;
  std::uint64_t observed_seek_generation = 0;
  std::uint32_t probe_pulls_since_seek = 0;
  std::uint64_t probe_sequential_pull_count = 0;
};

struct FakeBus : FakeObject {
  FakeBus() : FakeObject(ObjectKind::Bus) {}
  bool emitted_error = false;
  std::uint32_t poll_count = 0;
};

struct FakeDisplayInfo : FakeObject {
  explicit FakeDisplayInfo(FakePipeline* owner)
      : FakeObject(ObjectKind::DisplayInfo), pipeline(owner) {}
  FakePipeline* pipeline = nullptr;
};

struct FakeOutputInfo : FakeObject {
  explicit FakeOutputInfo(FakePipeline* owner)
      : FakeObject(ObjectKind::OutputInfo), pipeline(owner) {}
  FakePipeline* pipeline = nullptr;
};

struct FakeProbeInfo : FakeObject {
  explicit FakeProbeInfo(FakePipeline* owner)
      : FakeObject(ObjectKind::ProbeInfo), pipeline(owner) {}
  FakePipeline* pipeline = nullptr;
};

struct FakePad : FakeObject {
  FakePad() : FakeObject(ObjectKind::Pad) {}
  FakePadProbeCallback callback = nullptr;
  void* callback_data = nullptr;
  FakeDestroyNotify destroy_notify = nullptr;
  unsigned long probe_id = 0;
  std::uint32_t current_width = 1280;
  std::uint32_t current_height = 720;
  bool output = false;
  bool probe = false;
};

struct FakePipeline : FakeObject {
  FakePipeline() : FakeObject(ObjectKind::Pipeline) {}
  FakePad* display_pad = nullptr;
  FakePad* output_pad = nullptr;
  bool parser_probe = false;
  bool elementary_probe = false;
  std::int64_t seek_target_ns = 0;
  bool has_seek = false;
  std::uint64_t seek_generation = 0;
};

constexpr std::uint64_t kFakeCapsMagic = 0x5245434f43415053ULL;

struct FakeCaps {
  std::uint64_t magic = kFakeCapsMagic;
  std::uint32_t width = 1280;
  std::uint32_t height = 720;
  std::int32_t fps_numerator = 30'000;
  std::int32_t fps_denominator = 1'001;
};

constexpr std::uint64_t kFakeMessageMagic = 0x5245434f4d534747ULL;

struct FakeMessage {
  std::uint64_t magic = kFakeMessageMagic;
};

struct FakePadProbeInfo {
  std::int32_t type = 0;
  unsigned long id = 0;
  void* data = nullptr;
};

struct FakeSample {
  GstBufferAbi buffer;
  abi::SurfaceParams params;
  abi::Surface surface;
  FakeCaps sample_caps;
  FakePipeline* pipeline = nullptr;
  bool post_pull_runahead_emitted = false;
  bool segment_outside = false;
  bool non_time_segment = false;
  std::uint64_t segment_stream_origin_ns = 0;
};

std::mutex event_mutex;

std::string scenario() {
  const char* value = std::getenv("RECO_FAKE_GST_SCENARIO");
  return value == nullptr ? "frame-eos" : value;
}

FakeCaps parser_sample_caps() {
  FakeCaps caps{.width = 3840, .height = 2160};
  if (scenario() == "probe-bad-dimensions") {
    caps.width = 0;
  } else if (scenario() == "probe-odd-dimensions") {
    caps.width = 853;
  } else if (scenario() == "probe-bad-fps") {
    caps.fps_denominator = 0;
  } else if (scenario() == "probe-high-fps") {
    caps.fps_numerator = 90'000;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-duration-mismatch" || scenario() == "probe-delayed-stream" ||
             scenario() == "probe-nonzero-origin" || scenario() == "probe-decode-order-origin" ||
             scenario() == "probe-unknown-pts" || scenario() == "probe-long-unknown-pts" ||
             scenario() == "probe-mixed-prefix-pts" || scenario() == "probe-one-frame-rounding") {
    caps.fps_numerator = 30;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-inexact-caps-fps") {
    caps.fps_numerator = 59;
    caps.fps_denominator = 2;
  } else if (scenario() == "probe-bframe-cutoff") {
    caps.fps_numerator = 119;
    caps.fps_denominator = 4;
  } else if (scenario() == "probe-quantized-timestamps") {
    caps.fps_numerator = 24'000;
    caps.fps_denominator = 1'001;
  } else if (scenario() == "probe-unset-fps-inferred" || scenario() == "probe-vfr-unset-fps") {
    caps.fps_numerator = 0;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-vfr-late-transition") {
    caps.fps_numerator = 45;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-dropped-frame-after-prefix") {
    caps.fps_numerator = 30;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-reduced-cadence-after-prefix") {
    caps.fps_numerator = 45;
    caps.fps_denominator = 2;
  } else if (scenario() == "probe-estimated-count-lower-bound" ||
             scenario() == "probe-long-untimed-elementary") {
    caps.fps_numerator = 1;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-retimed-constant-pts") {
    caps.fps_numerator = 15;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-vfr-missing-durations") {
    caps.fps_numerator = 45;
    caps.fps_denominator = 1;
  } else if (scenario() == "probe-caps-runahead") {
    caps.width = 854;
    caps.height = 480;
    caps.fps_numerator = 24;
    caps.fps_denominator = 1;
  }
  return caps;
}

std::uint64_t parser_frame_offset(std::uint64_t frame_index, const FakeCaps& caps) {
  const long double offset = static_cast<long double>(frame_index) * 1'000'000'000.0L *
                             caps.fps_denominator / caps.fps_numerator;
  return offset >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())
             ? std::numeric_limits<std::uint64_t>::max()
             : static_cast<std::uint64_t>(offset);
}

std::uint64_t parser_timing_offset(std::uint64_t frame_index, const FakeCaps& caps) {
  if (scenario() == "probe-vfr-unset-fps") {
    return (frame_index / 2U) * 70'000'000ULL + (frame_index % 2U) * 30'000'000ULL;
  }
  if (scenario() == "probe-vfr-late-transition" || scenario() == "probe-vfr-missing-durations") {
    return frame_index <= 90U ? frame_index * 1'000'000'000ULL / 30U
                              : 3'000'000'000ULL + (frame_index - 90U) * 1'000'000'000ULL / 60U;
  }
  if (scenario() == "probe-dropped-frame-after-prefix") {
    const auto presentation_index = frame_index < 100U ? frame_index : frame_index + 1U;
    return presentation_index * 1'000'000'000ULL / 30U;
  }
  if (scenario() == "probe-reduced-cadence-after-prefix") {
    return frame_index <= 90U ? frame_index * 1'000'000'000ULL / 30U
                              : 3'000'000'000ULL + (frame_index - 90U) * 1'000'000'000ULL / 15U;
  }
  return parser_frame_offset(frame_index, caps);
}

std::uint64_t snapped_parser_seek_target(std::uint64_t requested_target, std::uint64_t stream_start,
                                         const FakeCaps& caps) {
  if (requested_target <= stream_start) {
    return stream_start;
  }
  const auto relative_target = requested_target - stream_start;
  const long double frame_position = static_cast<long double>(relative_target) *
                                     caps.fps_numerator / (1'000'000'000.0L * caps.fps_denominator);
  const auto frame_index = static_cast<std::uint64_t>(std::ceil(frame_position - 1e-12L));
  return stream_start + parser_timing_offset(frame_index, caps);
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

FakeSample* make_sample(std::uint32_t sample_index = 0) {
  auto* sample = new FakeSample;
  if (scenario() == "unknown-time" || scenario() == "caps-runahead-unknown-time" ||
      scenario() == "dropped-unknown-transition") {
    sample->buffer.pts = std::numeric_limits<std::uint64_t>::max();
    sample->buffer.duration = std::numeric_limits<std::uint64_t>::max();
  } else if (scenario() == "caps-runahead" || scenario() == "caps-runahead-stale-caps" ||
             scenario() == "duplicate-transition-pts" || scenario() == "same-allocation-runahead") {
    sample->buffer.pts = (static_cast<std::uint64_t>(sample_index) + 1U) * 1'000'000'000ULL;
  } else if (scenario() == "drop-transition-first" ||
             scenario() == "drop-duplicate-transition-first" ||
             scenario() == "nonmonotonic-duplicate-transition" ||
             scenario() == "nonmonotonic-unique-transition") {
    sample->buffer.pts = 2'000'000'000ULL;
  } else if (scenario() == "retired-pts-reuse") {
    sample->buffer.pts = sample_index == 1 ? 2'000'000'000ULL : 1'000'000'000ULL;
  } else if (scenario() == "mixed-unknown-reorder") {
    sample->buffer.pts =
        sample_index == 0 ? std::numeric_limits<std::uint64_t>::max() : 1'000'000'000ULL;
  }

  sample->params.width =
      scenario() == "visible-crop" ||
              ((scenario() == "caps-runahead" || scenario() == "caps-runahead-unknown-time" ||
                scenario() == "caps-runahead-stale-caps") &&
               sample_index == 0)
          ? 864
          : 1280;
  sample->params.height = scenario() == "caps-runahead-stale-caps" && sample_index == 0 ? 480 : 720;
  sample->params.pitch = sample->params.width == 864 ? 1024 : 1280;
  sample->params.color_format = abi::kColorNv12_709;
  sample->params.layout = abi::kLayoutPitch;
  sample->params.data_size = sample->params.pitch * sample->params.height * 3U / 2U;
  sample->params.data_ptr = reinterpret_cast<void*>(0x10000000);
  sample->params.plane_params.num_planes = 2;
  sample->params.plane_params.width[0] = sample->params.width;
  sample->params.plane_params.height[0] = sample->params.height;
  sample->params.plane_params.pitch[0] = sample->params.pitch;
  sample->params.plane_params.psize[0] = sample->params.pitch * sample->params.height;
  sample->params.plane_params.bytes_per_pix[0] = 1;
  sample->params.plane_params.width[1] = sample->params.width / 2U;
  sample->params.plane_params.height[1] = sample->params.height / 2U;
  sample->params.plane_params.pitch[1] = sample->params.pitch;
  sample->params.plane_params.offset[1] = sample->params.pitch * sample->params.height;
  sample->params.plane_params.psize[1] = sample->params.pitch * sample->params.height / 2U;
  sample->params.plane_params.bytes_per_pix[1] = 2;
  sample->surface.gpu_id = 0;
  sample->surface.batch_size = 1;
  sample->surface.num_filled = scenario() == "invalid-surface" ? 0 : 1;
  sample->surface.mem_type = abi::kMemCudaDevice;
  sample->surface.surface_list = &sample->params;
  return sample;
}

std::uint32_t predecoder_width(std::uint32_t sample_index) {
  if (scenario() == "visible-crop" || scenario() == "caps-runahead-stale-caps" ||
      scenario() == "drop-stale-caps-first") {
    return 854;
  }
  if (scenario() == "oversized-caps") {
    return 1290;
  }
  if ((scenario() == "caps-runahead" || scenario() == "caps-runahead-unknown-time") &&
      sample_index == 0) {
    return 854;
  }
  if (scenario() == "same-allocation-runahead" || scenario() == "drop-transition-first") {
    return sample_index == 0 ? 1100 : 1200;
  }
  if (scenario() == "retired-pts-reuse") {
    constexpr std::array<std::uint32_t, 3> widths{1100, 1200, 1160};
    return widths.at(sample_index);
  }
  return 1280;
}

std::uint32_t predecoder_height() {
  if (scenario() == "caps-runahead-stale-caps" || scenario() == "drop-stale-caps-first") {
    return 480;
  }
  return 720;
}

void push_predecoder_buffer(FakePad* pad, GstBufferAbi& buffer, std::uint32_t width,
                            std::uint32_t height) {
  if (pad == nullptr || pad->callback == nullptr) {
    return;
  }
  pad->current_width = width;
  pad->current_height = height;
  FakePadProbeInfo info{.data = &buffer};
  (void)pad->callback(pad, &info, pad->callback_data);
}

void push_output_buffer(FakePad* pad, GstBufferAbi& buffer) {
  if (pad == nullptr || pad->callback == nullptr) {
    return;
  }
  FakePadProbeInfo info{.data = &buffer};
  (void)pad->callback(pad, &info, pad->callback_data);
}

void release_buffer_qdata(GstBufferAbi& buffer) {
  if (buffer.qdata_destroy != nullptr) {
    buffer.qdata_destroy(buffer.qdata);
  }
  buffer.qdata = nullptr;
  buffer.qdata_destroy = nullptr;
}

FakeSample* deliver_output(FakeSink* sink, FakeSample* sample) {
  sample->pipeline = sink->pipeline;
  push_output_buffer(sink->pipeline->output_pad, sample->buffer);
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

RECO_FAKE_EXPORT void* gst_parse_launch(const char* description, GErrorAbi** error) {
  const bool parser_probe =
      description != nullptr && std::strstr(description, "probe_info") != nullptr;
  record(parser_probe ? "parse-probe" : "parse-decoder");
  if (parser_probe && std::strstr(description, "video/x-h264;video/x-h265") != nullptr) {
    record("probe-codec-filter");
  }
  if (parser_probe && std::strstr(description, "parsebin") != nullptr) {
    record("probe-parsebin");
  }
  if (parser_probe &&
      std::strstr(description, "stream-format=byte-stream,alignment=au") != nullptr) {
    record("probe-decoder-caps");
  }
  if (parser_probe && std::strstr(description, "h264parse") != nullptr) {
    record("probe-h264-parser");
  }
  if (parser_probe && std::strstr(description, "h265parse") != nullptr) {
    record("probe-h265-parser");
  }
  if (parser_probe && std::strstr(description, "video/x-raw") != nullptr) {
    record("raw-video-caps");
  }
  if (description != nullptr && std::strstr(description, "nvv4l2decoder") != nullptr) {
    record("decoder-element");
  }
  if (scenario() == "parse-error" || scenario() == "probe-parse-error") {
    *error = make_error("fake parse failure");
    return nullptr;
  }
  auto* pipeline = new FakePipeline;
  pipeline->parser_probe = parser_probe;
  pipeline->elementary_probe = parser_probe && std::strstr(description, "parsebin") == nullptr;
  if (scenario() == "probe-parse-partial-error") {
    *error = make_error("fake partial parse failure");
  }
  return pipeline;
}

RECO_FAKE_EXPORT void* gst_bin_get_by_name(void* pipeline_pointer, const char* name) {
  auto* pipeline = static_cast<FakePipeline*>(pipeline_pointer);
  if (name != nullptr && std::strcmp(name, "sink") == 0) {
    record("get-sink");
    return scenario() == "missing-sink" ? nullptr : new FakeSink(pipeline);
  }
  if (name != nullptr && std::strcmp(name, "probe_sink") == 0) {
    record("get-probe-sink");
    return scenario() == "probe-missing-sink" ? nullptr : new FakeSink(pipeline);
  }
  if (name != nullptr && std::strcmp(name, "display_info") == 0) {
    record("get-display-info");
    return scenario() == "missing-display-info" ? nullptr : new FakeDisplayInfo(pipeline);
  }
  if (name != nullptr && std::strcmp(name, "output_info") == 0) {
    record("get-output-info");
    return scenario() == "missing-output-info" ? nullptr : new FakeOutputInfo(pipeline);
  }
  if (name != nullptr && std::strcmp(name, "probe_info") == 0) {
    record("get-probe-info");
    return scenario() == "probe-missing-info" ? nullptr : new FakeProbeInfo(pipeline);
  }
  return nullptr;
}

RECO_FAKE_EXPORT void* gst_element_get_static_pad(void* element_pointer, const char* name) {
  const auto* element = static_cast<FakeObject*>(element_pointer);
  const bool output = element->kind == ObjectKind::OutputInfo;
  const bool probe = element->kind == ObjectKind::ProbeInfo;
  record(output ? "get-output-pad" : probe ? "get-probe-pad" : "get-display-pad");
  if (name == nullptr || std::strcmp(name, "src") != 0 ||
      (!output && scenario() == "missing-display-pad") ||
      (output && scenario() == "missing-output-pad") ||
      (probe && scenario() == "probe-missing-pad")) {
    return nullptr;
  }
  auto* pad = new FakePad;
  pad->output = output;
  pad->probe = probe;
  if (probe) {
    pad->current_width = 3840;
    pad->current_height = 2160;
  }
  if (output) {
    static_cast<FakeOutputInfo*>(element_pointer)->pipeline->output_pad = pad;
  } else if (!probe) {
    static_cast<FakeDisplayInfo*>(element_pointer)->pipeline->display_pad = pad;
  }
  return pad;
}

RECO_FAKE_EXPORT int gst_element_set_state(void* pipeline_pointer, int state) {
  record(state == 4 ? "state-playing" : state == 3 ? "state-paused" : "state-null");
  const auto* pipeline = static_cast<FakePipeline*>(pipeline_pointer);
  if (state == 4 && ((!pipeline->parser_probe && scenario() == "state-error") ||
                     (pipeline->parser_probe &&
                      (scenario() == "probe-state-error" || scenario() == "probe-stream-error")))) {
    return 0;
  }
  return 1;
}

RECO_FAKE_EXPORT int gst_element_get_state(void*, int* state, int* pending, std::uint64_t) {
  record("get-state");
  if (pending != nullptr) {
    *pending = 0;
  }
  if (scenario() == "probe-stream-error" || scenario() == "probe-no-supported-video") {
    return 0;
  }
  if (scenario() == "probe-timeout") {
    if (state != nullptr) {
      *state = 2;
    }
    return 2;
  }
  if (state != nullptr) {
    *state = 3;
  }
  return 1;
}

RECO_FAKE_EXPORT int gst_element_query_duration(void*, int format, std::int64_t* duration) {
  record("query-duration");
  if (format != 3 || duration == nullptr || scenario() == "probe-duration-unknown" ||
      scenario() == "probe-long-unknown-pts") {
    return 0;
  }
  if (scenario() == "probe-duration-zero") {
    *duration = 0;
    return 1;
  }
  if (scenario() == "probe-exact-frame-count") {
    *duration = 100'100'000;
    return 1;
  }
  if (scenario() == "probe-integral-frame-count") {
    *duration = 1'001'000'000'000;
    return 1;
  }
  if (scenario() == "probe-frame-count-overflow") {
    *duration = std::numeric_limits<std::int64_t>::max();
    return 1;
  }
  if (scenario() == "probe-duration-mismatch" || scenario() == "probe-delayed-stream") {
    *duration = 5'000'000'000;
    return 1;
  }
  if (scenario() == "probe-nonzero-origin") {
    *duration = 1'978'082'185;
    return 1;
  }
  if (scenario() == "probe-decode-order-origin") {
    *duration = 1'000'000'000;
    return 1;
  }
  if (scenario() == "probe-seek-preroll") {
    *duration = 200'000'000'000;
    return 1;
  }
  if (scenario() == "probe-unset-fps-inferred" || scenario() == "probe-bframe-cutoff" ||
      scenario() == "probe-mixed-prefix-pts") {
    *duration = 4'000'000'000;
    return 1;
  }
  if (scenario() == "probe-quantized-timestamps") {
    *duration = 4'170'833'333;
    return 1;
  }
  if (scenario() == "probe-vfr-late-transition" ||
      scenario() == "probe-dropped-frame-after-prefix" ||
      scenario() == "probe-reduced-cadence-after-prefix" ||
      scenario() == "probe-vfr-missing-durations") {
    *duration = 6'000'000'000;
    return 1;
  }
  if (scenario() == "probe-estimated-count-lower-bound") {
    *duration = 10'000'000'000;
    return 1;
  }
  if (scenario() == "probe-retimed-constant-pts") {
    *duration = 5'000'000'000;
    return 1;
  }
  if (scenario() == "probe-long-untimed-elementary") {
    *duration = 575'935'624'115;
    return 1;
  }
  *duration = 10'000'000'000LL;
  return 1;
}

RECO_FAKE_EXPORT int gst_element_seek_simple(void* pipeline_pointer, int format, int,
                                             std::int64_t target) {
  record("seek-compressed");
  if (format != 3 || target < 0 || scenario() == "probe-seek-unsupported") {
    return 0;
  }
  static_cast<FakePipeline*>(pipeline_pointer)->seek_target_ns = target;
  static_cast<FakePipeline*>(pipeline_pointer)->has_seek = true;
  ++static_cast<FakePipeline*>(pipeline_pointer)->seek_generation;
  return 1;
}

RECO_FAKE_EXPORT void* gst_element_get_bus(void*) {
  record("get-bus");
  if (scenario() == "missing-bus" || scenario() == "probe-missing-bus") {
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
  case ObjectKind::DisplayInfo:
    record("unref-display-info");
    delete static_cast<FakeDisplayInfo*>(object);
    break;
  case ObjectKind::OutputInfo:
    record("unref-output-info");
    delete static_cast<FakeOutputInfo*>(object);
    break;
  case ObjectKind::ProbeInfo:
    record("unref-probe-info");
    delete static_cast<FakeProbeInfo*>(object);
    break;
  case ObjectKind::Pad:
    record("unref-display-pad");
    if (static_cast<FakePad*>(object)->probe_id != 0) {
      record("probe-leaked");
    }
    delete static_cast<FakePad*>(object);
    break;
  }
}

RECO_FAKE_EXPORT void* gst_pad_get_current_caps(void* pad_pointer) {
  record("pad-current-caps");
  const auto* pad = static_cast<FakePad*>(pad_pointer);
  if (scenario() == "missing-caps" || (pad->probe && scenario() == "probe-missing-current-caps")) {
    return nullptr;
  }
  if (pad->probe && scenario() == "probe-caps-runahead") {
    return new FakeCaps{.width = 1280, .height = 720, .fps_numerator = 30, .fps_denominator = 1};
  }
  return pad->probe ? new FakeCaps(parser_sample_caps())
                    : new FakeCaps{.width = pad->current_width, .height = pad->current_height};
}

RECO_FAKE_EXPORT unsigned long gst_pad_add_probe(void* pad_pointer, int,
                                                 FakePadProbeCallback callback, void* user_data,
                                                 FakeDestroyNotify destroy_notify) {
  auto* pad = static_cast<FakePad*>(pad_pointer);
  record(pad->output ? "add-output-probe" : "add-display-probe");
  if ((!pad->output && scenario() == "probe-install-error") ||
      (pad->output && scenario() == "output-probe-install-error")) {
    return 0;
  }
  pad->callback = callback;
  pad->callback_data = user_data;
  pad->destroy_notify = destroy_notify;
  pad->probe_id = 1;
  return pad->probe_id;
}

RECO_FAKE_EXPORT void gst_pad_remove_probe(void* pad_pointer, unsigned long probe_id) {
  auto* pad = static_cast<FakePad*>(pad_pointer);
  record(pad->output ? "remove-output-probe" : "remove-display-probe");
  if (pad->probe_id == probe_id) {
    auto* callback_data = pad->callback_data;
    const auto destroy_notify = pad->destroy_notify;
    pad->callback = nullptr;
    pad->callback_data = nullptr;
    pad->destroy_notify = nullptr;
    pad->probe_id = 0;
    if (destroy_notify != nullptr) {
      record(pad->output ? "destroy-output-probe-data" : "destroy-display-probe-data");
      destroy_notify(callback_data);
    }
  }
}

RECO_FAKE_EXPORT void gst_mini_object_set_qdata(void* object, std::uint32_t, void* data,
                                                FakeDestroyNotify destroy_notify) {
  auto* buffer = static_cast<GstBufferAbi*>(object);
  release_buffer_qdata(*buffer);
  buffer->qdata = data;
  buffer->qdata_destroy = destroy_notify;
}

RECO_FAKE_EXPORT void* gst_mini_object_get_qdata(void* object, std::uint32_t) {
  return static_cast<GstBufferAbi*>(object)->qdata;
}

RECO_FAKE_EXPORT std::uint32_t g_quark_from_static_string(const char*) { return 1; }

RECO_FAKE_EXPORT void gst_caps_unref(void* caps) {
  record("caps-unref");
  delete static_cast<FakeCaps*>(caps);
}

RECO_FAKE_EXPORT void* gst_app_sink_try_pull_sample(void* sink_pointer, std::uint64_t timeout_ns) {
  auto* sink = static_cast<FakeSink*>(sink_pointer);
  if (sink->pipeline->parser_probe) {
    record("pull-probe");
    sink->probe_eos = false;
    if (sink->observed_seek_generation != sink->pipeline->seek_generation) {
      sink->observed_seek_generation = sink->pipeline->seek_generation;
      sink->probe_pulls_since_seek = 0;
    }
    if (scenario() == "probe-no-supported-video") {
      sink->probe_eos = true;
      return nullptr;
    }
    if (scenario() == "probe-async-error") {
      return nullptr;
    }
    if (scenario() == "probe-timeout" ||
        (scenario() == "probe-pull-timeout" && sink->pipeline->has_seek)) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(timeout_ns));
      return nullptr;
    }
    const auto current_scenario = scenario();
    const auto sample_caps = parser_sample_caps();
    auto timing_caps = sample_caps;
    if (current_scenario == "probe-inexact-caps-fps") {
      timing_caps.fps_numerator = 30;
      timing_caps.fps_denominator = 1;
    } else if (current_scenario == "probe-bad-fps" ||
               current_scenario == "probe-unset-fps-inferred" ||
               current_scenario == "probe-vfr-unset-fps" ||
               current_scenario == "probe-bframe-cutoff") {
      timing_caps.fps_numerator = 30;
      timing_caps.fps_denominator = 1;
    } else if (current_scenario == "probe-estimated-count-lower-bound") {
      timing_caps.fps_numerator = 60;
      timing_caps.fps_denominator = 1;
    } else if (current_scenario == "probe-retimed-constant-pts") {
      timing_caps.fps_numerator = 30;
      timing_caps.fps_denominator = 1;
    }
    const std::uint64_t selected_stream_start_ns =
        current_scenario == "probe-delayed-stream"   ? 4'000'000'000ULL
        : current_scenario == "probe-nonzero-origin" ? 766'666'666ULL
                                                     : 0;
    const std::uint64_t selected_duration_ns =
        current_scenario == "probe-duration-unknown" || current_scenario == "probe-duration-zero" ||
                current_scenario == "probe-seek-unsupported" ||
                current_scenario == "probe-pull-timeout" || current_scenario == "probe-seek-preroll"
            ? 200'000'000'000ULL
        : current_scenario == "probe-delayed-stream"               ? 5'000'000'000ULL
        : current_scenario == "probe-nonzero-origin"               ? 2'766'666'666ULL
        : current_scenario == "probe-duration-mismatch"            ? 1'000'000'000ULL
        : current_scenario == "probe-decode-order-origin"          ? 1'000'000'000ULL
        : current_scenario == "probe-unknown-pts"                  ? 1'000'000'000ULL
        : current_scenario == "probe-long-unknown-pts"             ? 3'000'000'000ULL
        : current_scenario == "probe-mixed-prefix-pts"             ? 4'000'000'000ULL
        : current_scenario == "probe-one-frame-rounding"           ? 33'333'333ULL
        : current_scenario == "probe-inexact-caps-fps"             ? 2'000'000'000ULL
        : current_scenario == "probe-unset-fps-inferred"           ? 4'000'000'000ULL
        : current_scenario == "probe-vfr-unset-fps"                ? 4'720'000'000ULL
        : current_scenario == "probe-vfr-late-transition"          ? 6'000'000'000ULL
        : current_scenario == "probe-dropped-frame-after-prefix"   ? 6'000'000'000ULL
        : current_scenario == "probe-reduced-cadence-after-prefix" ? 6'000'000'000ULL
        : current_scenario == "probe-vfr-missing-durations"        ? 6'000'000'000ULL
        : current_scenario == "probe-estimated-count-lower-bound"  ? 10'000'000'000ULL
        : current_scenario == "probe-retimed-constant-pts"         ? 5'000'000'000ULL
        : current_scenario == "probe-long-untimed-elementary"      ? 600'000'000'000ULL
        : current_scenario == "probe-bframe-cutoff"                ? 4'000'000'000ULL
        : current_scenario == "probe-quantized-timestamps"         ? 4'170'833'333ULL
        : current_scenario == "probe-exact-frame-count"            ? 100'100'000ULL
        : current_scenario == "probe-integral-frame-count"         ? 1'001'000'000'000ULL
        : current_scenario == "probe-frame-count-overflow"
            ? static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
            : 10'000'000'000ULL;
    const auto sequential_index = sink->probe_sequential_pull_count;
    const auto seek_pull_index = sink->pipeline->has_seek ? sink->probe_pulls_since_seek++ : 0U;
    const auto seek_advance_index = current_scenario == "probe-seek-preroll" && seek_pull_index > 0
                                        ? seek_pull_index - 1U
                                        : seek_pull_index;
    const auto snapped_seek_target =
        sink->pipeline->has_seek
            ? snapped_parser_seek_target(static_cast<std::uint64_t>(sink->pipeline->seek_target_ns),
                                         selected_stream_start_ns, timing_caps)
            : 0;
    const auto seek_advance = parser_timing_offset(seek_advance_index, timing_caps);
    const auto target_ns =
        sink->pipeline->has_seek
            ? seek_advance > std::numeric_limits<std::uint64_t>::max() - snapped_seek_target
                  ? std::numeric_limits<std::uint64_t>::max()
                  : snapped_seek_target + seek_advance
            : selected_stream_start_ns + parser_timing_offset(sequential_index, timing_caps);
    if (target_ns >= selected_duration_ns) {
      sink->probe_eos = true;
      return nullptr;
    }
    if (!sink->pipeline->has_seek) {
      ++sink->probe_sequential_pull_count;
    }
    auto* sample = new FakeSample;
    sample->pipeline = sink->pipeline;
    sample->sample_caps = sample_caps;
    sample->non_time_segment = sink->pipeline->elementary_probe;
    sample->segment_stream_origin_ns = selected_stream_start_ns;
    sample->segment_outside = current_scenario == "probe-seek-preroll" &&
                              sink->pipeline->has_seek && seek_pull_index == 0;
    if (current_scenario == "probe-unknown-pts" || current_scenario == "probe-long-unknown-pts") {
      sample->buffer.pts = std::numeric_limits<std::uint64_t>::max();
    } else if (current_scenario == "probe-mixed-prefix-pts" && !sink->pipeline->has_seek &&
               sequential_index == 0) {
      sample->buffer.pts = std::numeric_limits<std::uint64_t>::max();
    } else if (current_scenario == "probe-decode-order-origin" && !sink->pipeline->has_seek) {
      const auto presentation_index = sequential_index == 0   ? 2ULL
                                      : sequential_index == 1 ? 0ULL
                                      : sequential_index == 2 ? 1ULL
                                                              : sequential_index;
      sample->buffer.pts =
          selected_stream_start_ns + parser_frame_offset(presentation_index, timing_caps);
    } else if (current_scenario == "probe-bframe-cutoff" && !sink->pipeline->has_seek) {
      const auto presentation_index = sequential_index == 63   ? 64ULL
                                      : sequential_index == 64 ? 63ULL
                                                               : sequential_index;
      sample->buffer.pts =
          selected_stream_start_ns + parser_frame_offset(presentation_index, timing_caps);
    } else if (current_scenario == "probe-quantized-timestamps") {
      sample->buffer.pts = target_ns / 1'000'000ULL * 1'000'000ULL;
    } else {
      sample->buffer.pts = target_ns;
    }
    const auto frame_duration_ns = sink->pipeline->has_seek
                                       ? parser_frame_offset(1, timing_caps)
                                       : parser_timing_offset(sequential_index + 1, timing_caps) -
                                             parser_timing_offset(sequential_index, timing_caps);
    const auto remaining_ns = selected_duration_ns - target_ns;
    sample->buffer.duration =
        remaining_ns > frame_duration_ns && remaining_ns - frame_duration_ns > 1 ? frame_duration_ns
                                                                                 : remaining_ns;
    if (current_scenario == "probe-vfr-missing-durations") {
      sample->buffer.duration = std::numeric_limits<std::uint64_t>::max();
    }
    return sample;
  }
  record("pull");
  const auto current = sink->pull_count++;
  const auto current_scenario = scenario();
  if (current_scenario == "duplicate-transition-pts" && current < 2) {
    auto* sample = make_sample(current);
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, 1280, 720);
    if (current == 1) {
      GstBufferAbi next_buffer;
      next_buffer.pts = sample->buffer.pts;
      push_predecoder_buffer(sink->pipeline->display_pad, next_buffer, 1100, 720);
    }
    return deliver_output(sink, sample);
  }
  if (current_scenario == "drop-runahead" && current == 0) {
    auto* sample = make_sample(current);
    for (std::uint64_t index = 0; index < 4'999; ++index) {
      GstBufferAbi dropped_buffer;
      dropped_buffer.pts = index * 33'333'333ULL;
      push_predecoder_buffer(sink->pipeline->display_pad, dropped_buffer, 1280, 720);
      push_output_buffer(sink->pipeline->output_pad, dropped_buffer);
      release_buffer_qdata(dropped_buffer);
    }
    sample->buffer.pts = 4'999ULL * 33'333'333ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, 1280, 720);
    return deliver_output(sink, sample);
  }
  if (current_scenario == "drop-transition-first" && current == 0) {
    auto* sample = make_sample(current);
    GstBufferAbi old_buffer;
    old_buffer.pts = 1'000'000'000ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, old_buffer, predecoder_width(0),
                           predecoder_height());
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, predecoder_width(1),
                           predecoder_height());
    return deliver_output(sink, sample);
  }
  if (current_scenario == "drop-duplicate-transition-first" && current == 0) {
    auto* sample = make_sample(current);
    GstBufferAbi earlier_old_buffer;
    earlier_old_buffer.pts = 1'000'000'000ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, earlier_old_buffer, 1280, 720);
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, 1280, 720);
    GstBufferAbi next_buffer;
    next_buffer.pts = sample->buffer.pts;
    push_predecoder_buffer(sink->pipeline->display_pad, next_buffer, 1100, 720);
    return deliver_output(sink, sample);
  }
  if (current_scenario == "nonmonotonic-duplicate-transition" && current == 0) {
    auto* sample = make_sample(current);
    GstBufferAbi first_old_buffer;
    first_old_buffer.pts = 500'000'000ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, first_old_buffer, 1280, 720);
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, 1280, 720);
    GstBufferAbi reordered_old_buffer;
    reordered_old_buffer.pts = 1'000'000'000ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, reordered_old_buffer, 1280, 720);
    GstBufferAbi next_buffer;
    next_buffer.pts = sample->buffer.pts;
    push_predecoder_buffer(sink->pipeline->display_pad, next_buffer, 1100, 720);
    return deliver_output(sink, sample);
  }
  if (current_scenario == "nonmonotonic-unique-transition" && current == 0) {
    auto* sample = make_sample(current);
    GstBufferAbi first_old_buffer;
    first_old_buffer.pts = 500'000'000ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, first_old_buffer, 1100, 720);
    GstBufferAbi reordered_old_buffer;
    reordered_old_buffer.pts = 3'000'000'000ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, reordered_old_buffer, 1100, 720);
    GstBufferAbi first_new_buffer;
    first_new_buffer.pts = 4'000'000'000ULL;
    push_predecoder_buffer(sink->pipeline->display_pad, first_new_buffer, 1200, 720);
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, 1200, 720);
    return deliver_output(sink, sample);
  }
  if (current_scenario == "retired-pts-reuse" && current < 3) {
    auto* sample = make_sample(current);
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, predecoder_width(current),
                           predecoder_height());
    return deliver_output(sink, sample);
  }
  if (current_scenario == "mixed-unknown-reorder" && current < 2) {
    auto* sample = make_sample(current);
    if (current == 0) {
      GstBufferAbi known_buffer;
      known_buffer.pts = 1'000'000'000ULL;
      push_predecoder_buffer(sink->pipeline->display_pad, known_buffer, 1280, 720);
      push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, 1280, 720);
    }
    return deliver_output(sink, sample);
  }
  if (current_scenario == "dropped-unknown-transition" && current == 0) {
    auto* sample = make_sample(current);
    GstBufferAbi dropped_buffer;
    dropped_buffer.pts = std::numeric_limits<std::uint64_t>::max();
    push_predecoder_buffer(sink->pipeline->display_pad, dropped_buffer, 1100, 720);
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, 1200, 720);
    push_output_buffer(sink->pipeline->output_pad, dropped_buffer);
    release_buffer_qdata(dropped_buffer);
    return deliver_output(sink, sample);
  }
  const bool caps_runahead = current_scenario == "caps-runahead" ||
                             current_scenario == "caps-runahead-unknown-time" ||
                             current_scenario == "caps-runahead-stale-caps" ||
                             current_scenario == "same-allocation-runahead";
  if (caps_runahead && current < 2) {
    auto* sample = make_sample(current);
    if (current == 0) {
      push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, predecoder_width(current),
                             predecoder_height());
      GstBufferAbi next_buffer;
      next_buffer.pts = current_scenario == "caps-runahead-unknown-time"
                            ? std::numeric_limits<std::uint64_t>::max()
                            : 2'000'000'000ULL;
      push_predecoder_buffer(sink->pipeline->display_pad, next_buffer, predecoder_width(1),
                             predecoder_height());
    }
    return deliver_output(sink, sample);
  }
  if ((current_scenario == "frame-eos" || current_scenario == "unknown-time" ||
       current_scenario == "missing-buffer" || current_scenario == "map-error" ||
       current_scenario == "invalid-surface" || current_scenario == "visible-crop" ||
       current_scenario == "missing-caps" || current_scenario == "missing-caps-structure" ||
       current_scenario == "invalid-caps" || current_scenario == "oversized-caps" ||
       current_scenario == "drop-stale-caps-first" || current_scenario == "post-pull-runahead") &&
      current == 0) {
    auto* sample = make_sample(current);
    push_predecoder_buffer(sink->pipeline->display_pad, sample->buffer, predecoder_width(current),
                           predecoder_height());
    return deliver_output(sink, sample);
  }
  return nullptr;
}

RECO_FAKE_EXPORT int gst_app_sink_is_eos(void* sink_pointer) {
  const auto current_scenario = scenario();
  const auto* sink = static_cast<FakeSink*>(sink_pointer);
  if (sink->pipeline->parser_probe) {
    return sink->probe_eos ? 1 : 0;
  }
  return (current_scenario == "frame-eos" || current_scenario == "unknown-time" ||
          current_scenario == "visible-crop" || current_scenario == "caps-runahead" ||
          current_scenario == "caps-runahead-unknown-time" ||
          current_scenario == "caps-runahead-stale-caps" ||
          current_scenario == "same-allocation-runahead" ||
          current_scenario == "drop-transition-first") &&
         sink->pull_count >= (current_scenario == "caps-runahead" ||
                                      current_scenario == "caps-runahead-unknown-time" ||
                                      current_scenario == "caps-runahead-stale-caps" ||
                                      current_scenario == "same-allocation-runahead"
                                  ? 3U
                                  : 2U);
}

RECO_FAKE_EXPORT void* gst_sample_get_buffer(void* sample) {
  if (scenario() == "missing-buffer") {
    return nullptr;
  }
  auto* fake_sample = static_cast<FakeSample*>(sample);
  if (scenario() == "post-pull-runahead" && !fake_sample->post_pull_runahead_emitted) {
    fake_sample->post_pull_runahead_emitted = true;
    for (std::uint64_t index = 0; index < 5'000; ++index) {
      GstBufferAbi dropped_buffer;
      dropped_buffer.pts = 2'000'000'000ULL + index * 33'333'333ULL;
      push_predecoder_buffer(fake_sample->pipeline->display_pad, dropped_buffer, 1280, 720);
      push_output_buffer(fake_sample->pipeline->output_pad, dropped_buffer);
      release_buffer_qdata(dropped_buffer);
    }
  }
  return &fake_sample->buffer;
}

RECO_FAKE_EXPORT const void* gst_sample_get_segment(void* sample) { return sample; }

RECO_FAKE_EXPORT std::uint64_t gst_segment_to_stream_time(const void* segment, int format,
                                                          std::uint64_t position) {
  return format == 3 && !static_cast<const FakeSample*>(segment)->segment_outside &&
                 !static_cast<const FakeSample*>(segment)->non_time_segment
             ? position
             : std::numeric_limits<std::uint64_t>::max();
}

RECO_FAKE_EXPORT std::uint64_t
gst_segment_position_from_stream_time(const void* segment, int format, std::uint64_t stream_time) {
  const auto* sample = static_cast<const FakeSample*>(segment);
  return format == 3 && !sample->segment_outside && !sample->non_time_segment &&
                 stream_time >= sample->segment_stream_origin_ns
             ? stream_time
             : std::numeric_limits<std::uint64_t>::max();
}

RECO_FAKE_EXPORT void* gst_sample_get_caps(void* sample) {
  record("sample-caps");
  if (scenario() == "missing-caps" || scenario() == "probe-missing-sample-caps") {
    return nullptr;
  }
  auto* fake_sample = static_cast<FakeSample*>(sample);
  return fake_sample->pipeline != nullptr && fake_sample->pipeline->parser_probe
             ? &fake_sample->sample_caps
             : sample;
}

RECO_FAKE_EXPORT void* gst_caps_get_structure(const void* caps, std::uint32_t index) {
  return index == 0 && scenario() != "missing-caps-structure" &&
                 scenario() != "probe-missing-caps-structure"
             ? const_cast<void*>(caps)
             : nullptr;
}

RECO_FAKE_EXPORT const char* gst_structure_get_name(const void*) {
  return scenario() == "probe-wrong-codec-caps" ? "video/x-vp9" : "video/x-h264";
}

RECO_FAKE_EXPORT int gst_structure_get_boolean(const void*, const char* field, int* value) {
  if (field == nullptr || value == nullptr || std::strcmp(field, "parsed") != 0) {
    return 0;
  }
  *value = scenario() == "probe-unparsed-caps" ? 0 : 1;
  return 1;
}

RECO_FAKE_EXPORT const char* gst_structure_get_string(const void*, const char* field) {
  if (field == nullptr) {
    return nullptr;
  }
  if (std::strcmp(field, "stream-format") == 0) {
    return scenario() == "probe-avc-caps" ? "avc" : "byte-stream";
  }
  if (std::strcmp(field, "alignment") == 0) {
    return scenario() == "probe-nal-caps" ? "nal" : "au";
  }
  return nullptr;
}

RECO_FAKE_EXPORT int gst_structure_get_int(const void* structure, const char* field, int* value) {
  if (value == nullptr || field == nullptr || scenario() == "invalid-caps") {
    return 0;
  }
  std::uint64_t magic = 0;
  if (structure != nullptr) {
    std::memcpy(&magic, structure, sizeof(magic));
  }
  if (std::strcmp(field, "width") == 0) {
    const bool predecoder_caps = magic == kFakeCapsMagic;
    *value = predecoder_caps                ? static_cast<const FakeCaps*>(structure)->width
             : scenario() == "visible-crop" ? 864
                                            : 1280;
    return 1;
  }
  if (std::strcmp(field, "height") == 0) {
    *value = magic == kFakeCapsMagic ? static_cast<const FakeCaps*>(structure)->height : 720;
    return 1;
  }
  return 0;
}

RECO_FAKE_EXPORT int gst_structure_get_fraction(const void* structure, const char* field,
                                                int* numerator, int* denominator) {
  if (structure == nullptr || field == nullptr || numerator == nullptr || denominator == nullptr ||
      std::strcmp(field, "framerate") != 0) {
    return 0;
  }
  const auto* caps = static_cast<const FakeCaps*>(structure);
  if (caps->magic != kFakeCapsMagic) {
    return 0;
  }
  *numerator = caps->fps_numerator;
  *denominator = caps->fps_denominator;
  return 1;
}

RECO_FAKE_EXPORT void gst_sample_unref(void* sample) {
  record("sample-unref");
  auto* fake_sample = static_cast<FakeSample*>(sample);
  release_buffer_qdata(fake_sample->buffer);
  delete fake_sample;
}

RECO_FAKE_EXPORT void gst_mini_object_unref(void* object) {
  std::uint64_t magic = 0;
  std::memcpy(&magic, object, sizeof(magic));
  if (magic == kFakeMessageMagic) {
    delete static_cast<FakeMessage*>(object);
    return;
  }
  gst_sample_unref(object);
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
  const bool ready = scenario() == "stream-error" || scenario() == "probe-async-error" ||
                     (scenario() == "delayed-stream-error" && bus->poll_count >= 2);
  if (!ready || bus->emitted_error || (types & (1U << 1U)) == 0) {
    return nullptr;
  }
  bus->emitted_error = true;
  return new FakeMessage;
}

RECO_FAKE_EXPORT void gst_message_parse_error(void*, GErrorAbi** error, char** debug) {
  *error = make_error(scenario() == "probe-async-error" ? "fake parser failure"
                                                        : "fake decoder failure");
  *debug = duplicate("fake debug detail");
}

RECO_FAKE_EXPORT void gst_message_unref(void* message) { gst_mini_object_unref(message); }

RECO_FAKE_EXPORT void g_error_free(GErrorAbi* error) {
  if (error != nullptr) {
    std::free(error->message);
    std::free(error);
  }
}

RECO_FAKE_EXPORT void g_free(void* pointer) { std::free(pointer); }
