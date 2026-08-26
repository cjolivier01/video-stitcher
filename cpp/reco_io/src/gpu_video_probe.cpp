#include "reco/io/gpu_video_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reco::io {
namespace {

constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr std::uint64_t kFallbackDurationSeconds = 60;
constexpr std::uint64_t kMinimumProbeTimeoutNs = kNanosecondsPerSecond;
constexpr std::uint64_t kMaximumProbeTimeoutNs = 3'600ULL * kNanosecondsPerSecond;
constexpr double kMaximumSaneFps = 1000.0;
constexpr int kGstFormatTime = 3;
constexpr int kGstStateNull = 1;
constexpr int kGstStatePlaying = 4;
constexpr int kGstStateChangeFailure = 0;
constexpr std::uint32_t kGstMessageError = 1U << 1U;
constexpr int kGstSeekFlagFlushAccurate = (1 << 0) | (1 << 1);
constexpr std::uint64_t kGstClockTimeNone = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kSamplePollTimeoutNs = 100'000'000ULL;

struct GErrorAbi {
  std::uint32_t domain = 0;
  std::int32_t code = 0;
  char* message = nullptr;
};

// GstBuffer is a public GStreamer 1.x ABI struct. The runtime major-version
// check below prevents this layout from being used with an incompatible ABI.
struct GstMiniObjectAbi {
  std::uintptr_t type = 0;
  std::int32_t ref_count = 0;
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
  std::uint64_t pts = kGstClockTimeNone;
  std::uint64_t dts = kGstClockTimeNone;
  std::uint64_t duration = kGstClockTimeNone;
  std::uint64_t offset = 0;
  std::uint64_t offset_end = 0;
};

static_assert(offsetof(GstBufferAbi, pts) == (sizeof(void*) == 8 ? 72 : 40));
static_assert(sizeof(GstBufferAbi) == (sizeof(void*) == 8 ? 112 : 80));

class DynamicLibrary {
public:
  explicit DynamicLibrary(std::string path) : path_(std::move(path)) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(path_.c_str());
    if (handle_ == nullptr) {
      throw GpuVideoProbeError("failed to load " + path_ + " (Windows error " +
                               std::to_string(GetLastError()) + ")");
    }
#else
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw GpuVideoProbeError("failed to load " + path_ +
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
    FreeLibrary(handle_);
#else
    dlclose(handle_);
#endif
  }

  template <typename Function> [[nodiscard]] Function symbol(const char* name) const {
#if defined(_WIN32)
    auto* value = GetProcAddress(handle_, name);
#else
    dlerror();
    void* value = dlsym(handle_, name);
#endif
    if (value == nullptr) {
      throw GpuVideoProbeError("missing video-probe runtime symbol " + std::string(name) + " in " +
                               path_);
    }
    return reinterpret_cast<Function>(value);
  }

private:
  std::string path_;
#if defined(_WIN32)
  HMODULE handle_ = nullptr;
#else
  void* handle_ = nullptr;
#endif
};

std::shared_ptr<DynamicLibrary> load_library(const char* environment_variable,
                                             std::initializer_list<const char*> names,
                                             std::string_view component) {
  if (const char* override_path = std::getenv(environment_variable);
      override_path != nullptr && override_path[0] != '\0') {
    return std::make_shared<DynamicLibrary>(override_path);
  }

  std::string errors;
  for (const char* name : names) {
    try {
      return std::make_shared<DynamicLibrary>(name);
    } catch (const GpuVideoProbeError& error) {
      if (!errors.empty()) {
        errors += "; ";
      }
      errors += error.what();
    }
  }
  throw GpuVideoProbeError("could not load " + std::string(component) + " runtime: " + errors);
}

class ProbeApi {
public:
  using InitCheck = int (*)(int*, char***, GErrorAbi**);
  using Version = void (*)(std::uint32_t*, std::uint32_t*, std::uint32_t*, std::uint32_t*);
  using ParseLaunch = void* (*)(const char*, GErrorAbi**);
  using BinGetByName = void* (*)(void*, const char*);
  using ElementGetStaticPad = void* (*)(void*, const char*);
  using ElementSetState = int (*)(void*, int);
  using ElementGetBus = void* (*)(void*);
  using ElementQueryDuration = int (*)(void*, int, std::int64_t*);
  using ElementSeekSimple = int (*)(void*, int, int, std::int64_t);
  using CapsGetStructure = void* (*)(const void*, std::uint32_t);
  using StructureGetName = const char* (*)(const void*);
  using StructureGetInt = int (*)(const void*, const char*, int*);
  using StructureGetBoolean = int (*)(const void*, const char*, int*);
  using StructureGetFraction = int (*)(const void*, const char*, int*, int*);
  using StructureGetString = const char* (*)(const void*, const char*);
  using AppSinkTryPullSample = void* (*)(void*, std::uint64_t);
  using AppSinkIsEos = int (*)(void*);
  using SampleGetBuffer = void* (*)(void*);
  using SampleGetCaps = void* (*)(void*);
  using SampleGetSegment = const void* (*)(void*);
  using SampleUnref = void (*)(void*);
  using SegmentToStreamTime = std::uint64_t (*)(const void*, int, std::uint64_t);
  using SegmentPositionFromStreamTime = std::uint64_t (*)(const void*, int, std::uint64_t);
  using BusTimedPopFiltered = void* (*)(void*, std::uint64_t, std::uint32_t);
  using MessageParseError = void (*)(void*, GErrorAbi**, char**);
  using MessageUnref = void (*)(void*);
  using ObjectUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);

  ProbeApi() {
#if defined(_WIN32)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH", {"gstreamer-1.0-0.dll"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"gstapp-1.0-0.dll"}, "GstApp");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0-0.dll", "glib-2.0-0.dll"}, "GLib");
#elif defined(__APPLE__)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"libgstapp-1.0.0.dylib", "libgstapp-1.0.dylib"},
                       "GstApp");
    glib =
        load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.0.dylib", "libglib-2.0.dylib"}, "GLib");
#else
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"libgstapp-1.0.so.0", "libgstapp-1.0.so"},
                       "GstApp");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.so.0", "libglib-2.0.so"}, "GLib");
#endif

    init_check = core->symbol<InitCheck>("gst_init_check");
    version = core->symbol<Version>("gst_version");
    parse_launch = core->symbol<ParseLaunch>("gst_parse_launch");
    bin_get_by_name = core->symbol<BinGetByName>("gst_bin_get_by_name");
    element_get_static_pad = core->symbol<ElementGetStaticPad>("gst_element_get_static_pad");
    element_set_state = core->symbol<ElementSetState>("gst_element_set_state");
    element_get_bus = core->symbol<ElementGetBus>("gst_element_get_bus");
    element_query_duration = core->symbol<ElementQueryDuration>("gst_element_query_duration");
    element_seek_simple = core->symbol<ElementSeekSimple>("gst_element_seek_simple");
    caps_get_structure = core->symbol<CapsGetStructure>("gst_caps_get_structure");
    structure_get_name = core->symbol<StructureGetName>("gst_structure_get_name");
    structure_get_int = core->symbol<StructureGetInt>("gst_structure_get_int");
    structure_get_boolean = core->symbol<StructureGetBoolean>("gst_structure_get_boolean");
    structure_get_fraction = core->symbol<StructureGetFraction>("gst_structure_get_fraction");
    structure_get_string = core->symbol<StructureGetString>("gst_structure_get_string");
    app_sink_try_pull_sample = app->symbol<AppSinkTryPullSample>("gst_app_sink_try_pull_sample");
    app_sink_is_eos = app->symbol<AppSinkIsEos>("gst_app_sink_is_eos");
    sample_get_buffer = core->symbol<SampleGetBuffer>("gst_sample_get_buffer");
    sample_get_caps = core->symbol<SampleGetCaps>("gst_sample_get_caps");
    sample_get_segment = core->symbol<SampleGetSegment>("gst_sample_get_segment");
    sample_unref = core->symbol<SampleUnref>("gst_sample_unref");
    segment_to_stream_time = core->symbol<SegmentToStreamTime>("gst_segment_to_stream_time");
    segment_position_from_stream_time =
        core->symbol<SegmentPositionFromStreamTime>("gst_segment_position_from_stream_time");
    bus_timed_pop_filtered = core->symbol<BusTimedPopFiltered>("gst_bus_timed_pop_filtered");
    message_parse_error = core->symbol<MessageParseError>("gst_message_parse_error");
    message_unref = core->symbol<MessageUnref>("gst_message_unref");
    object_unref = core->symbol<ObjectUnref>("gst_object_unref");
    error_free = glib->symbol<ErrorFree>("g_error_free");
    free = glib->symbol<Free>("g_free");
  }

  std::shared_ptr<DynamicLibrary> core;
  std::shared_ptr<DynamicLibrary> app;
  std::shared_ptr<DynamicLibrary> glib;
  InitCheck init_check = nullptr;
  Version version = nullptr;
  ParseLaunch parse_launch = nullptr;
  BinGetByName bin_get_by_name = nullptr;
  ElementGetStaticPad element_get_static_pad = nullptr;
  ElementSetState element_set_state = nullptr;
  ElementGetBus element_get_bus = nullptr;
  ElementQueryDuration element_query_duration = nullptr;
  ElementSeekSimple element_seek_simple = nullptr;
  CapsGetStructure caps_get_structure = nullptr;
  StructureGetName structure_get_name = nullptr;
  StructureGetInt structure_get_int = nullptr;
  StructureGetBoolean structure_get_boolean = nullptr;
  StructureGetFraction structure_get_fraction = nullptr;
  StructureGetString structure_get_string = nullptr;
  AppSinkTryPullSample app_sink_try_pull_sample = nullptr;
  AppSinkIsEos app_sink_is_eos = nullptr;
  SampleGetBuffer sample_get_buffer = nullptr;
  SampleGetCaps sample_get_caps = nullptr;
  SampleGetSegment sample_get_segment = nullptr;
  SampleUnref sample_unref = nullptr;
  SegmentToStreamTime segment_to_stream_time = nullptr;
  SegmentPositionFromStreamTime segment_position_from_stream_time = nullptr;
  BusTimedPopFiltered bus_timed_pop_filtered = nullptr;
  MessageParseError message_parse_error = nullptr;
  MessageUnref message_unref = nullptr;
  ObjectUnref object_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
};

std::string take_error(const ProbeApi& api, GErrorAbi*& error, std::string_view fallback) {
  std::unique_ptr<GErrorAbi, ProbeApi::ErrorFree> error_owner(std::exchange(error, nullptr),
                                                              api.error_free);
  std::string message(fallback);
  if (error_owner != nullptr) {
    if (error_owner->message != nullptr && error_owner->message[0] != '\0') {
      message = error_owner->message;
    }
  }
  return message;
}

std::string quote_gstreamer_property(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('"');
  for (const char ch : value) {
    if (ch == '\\' || ch == '"') {
      quoted.push_back('\\');
    }
    quoted.push_back(ch);
  }
  quoted.push_back('"');
  return quoted;
}

std::string_view parser_for_codec(GpuDecodeCodec codec) {
  switch (codec) {
  case GpuDecodeCodec::H264:
    return "h264parse";
  case GpuDecodeCodec::Hevc:
    return "h265parse";
  }
  return "h264parse";
}

std::string path_for_gstreamer(const std::filesystem::path& path) {
#if defined(_WIN32)
  const auto utf8 = path.u8string();
  return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
#else
  return path.string();
#endif
}

std::string build_probe_pipeline(const GpuFileDecodeConfig& config,
                                 const std::filesystem::path& absolute_path) {
  std::ostringstream pipeline;
  pipeline << "filesrc location=" << quote_gstreamer_property(path_for_gstreamer(absolute_path))
           << " ! ";
  if (config.elementary_stream) {
    pipeline << parser_for_codec(config.codec);
  } else {
    pipeline << gpu_decode_container_demuxer(*config.container)
             << " ! capsfilter caps=\"video/x-h264;video/x-h265\" ! parsebin";
  }
  pipeline << " ! capsfilter caps=\"video/x-h264,stream-format=byte-stream,alignment=au;"
              "video/x-h265,stream-format=byte-stream,alignment=au\""
           << " ! identity name=probe_info silent=true"
           << " ! appsink name=probe_sink emit-signals=false sync=false max-buffers=1 drop=false";
  return pipeline.str();
}

// Computes floor(a * b / divisor) with a portable 96-bit intermediate.
std::uint64_t multiply_divide_floor_saturating(std::uint64_t a, std::uint32_t b,
                                               std::uint64_t divisor) {
  const auto a_low = a & 0xffff'ffffULL;
  const auto a_high = a >> 32U;
  const auto low_product = a_low * b;
  const auto high_product = a_high * b;
  const auto shifted_high = high_product << 32U;
  const auto low = low_product + shifted_high;
  const auto carry = low < low_product ? 1ULL : 0ULL;
  const auto high = (high_product >> 32U) + carry;

  if (high >= divisor) {
    return std::numeric_limits<std::uint64_t>::max();
  }

  std::uint64_t quotient = 0;
  std::uint64_t remainder = high;
  for (int bit = 63; bit >= 0; --bit) {
    remainder = (remainder << 1U) | ((low >> bit) & 1ULL);
    if (remainder >= divisor) {
      remainder -= divisor;
      quotient |= 1ULL << bit;
    }
  }
  return quotient;
}

std::uint64_t frame_count_for_duration(std::uint64_t duration_ns, std::uint32_t fps_numerator,
                                       std::uint32_t fps_denominator) {
  const auto divisor = kNanosecondsPerSecond * fps_denominator;
  return multiply_divide_floor_saturating(duration_ns, fps_numerator, divisor);
}

std::uint64_t frame_timestamp_ns(std::uint64_t frame_index, std::uint32_t fps_numerator,
                                 std::uint32_t fps_denominator) {
  const auto period_numerator = kNanosecondsPerSecond * fps_denominator;
  const auto whole_periods = frame_index / fps_numerator;
  if (whole_periods > std::numeric_limits<std::uint64_t>::max() / period_numerator) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  const auto base = whole_periods * period_numerator;
  const auto remainder = static_cast<std::uint32_t>(frame_index % fps_numerator);
  const auto tail = multiply_divide_floor_saturating(period_numerator, remainder, fps_numerator);
  return tail > std::numeric_limits<std::uint64_t>::max() - base
             ? std::numeric_limits<std::uint64_t>::max()
             : base + tail;
}

std::uint64_t minimum_duration_for_frame_count(std::uint64_t frame_count,
                                               std::uint32_t fps_numerator,
                                               std::uint32_t fps_denominator) {
  auto duration_ns = frame_timestamp_ns(frame_count, fps_numerator, fps_denominator);
  if (duration_ns != std::numeric_limits<std::uint64_t>::max() &&
      frame_count_for_duration(duration_ns, fps_numerator, fps_denominator) < frame_count) {
    ++duration_ns;
  }
  return duration_ns;
}

std::uint64_t frame_count_ceiling_for_duration(std::uint64_t duration_ns,
                                               std::uint32_t fps_numerator,
                                               std::uint32_t fps_denominator) {
  const auto floor_count = frame_count_for_duration(duration_ns, fps_numerator, fps_denominator);
  if (floor_count == std::numeric_limits<std::uint64_t>::max()) {
    return floor_count;
  }
  return minimum_duration_for_frame_count(floor_count, fps_numerator, fps_denominator) < duration_ns
             ? floor_count + 1
             : floor_count;
}

std::uint64_t remaining_timeout_ns(std::chrono::steady_clock::time_point deadline,
                                   std::string_view timeout_message) {
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    throw GpuVideoProbeError(std::string(timeout_message));
  }
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count());
}

std::string pop_pipeline_error(const std::shared_ptr<ProbeApi>& api, void* bus) {
  void* message = api->bus_timed_pop_filtered(bus, 0, kGstMessageError);
  std::unique_ptr<void, ProbeApi::MessageUnref> message_owner(message, api->message_unref);
  if (message == nullptr) {
    return {};
  }

  GErrorAbi* error = nullptr;
  char* debug = nullptr;
  api->message_parse_error(message, &error, &debug);
  std::unique_ptr<void, ProbeApi::Free> debug_owner(debug, api->free);
  std::string result =
      "GStreamer parser-only probe failed: " + take_error(*api, error, "unknown streaming error");
  if (debug != nullptr && debug[0] != '\0') {
    result += " (" + std::string(debug) + ")";
  }
  return result;
}

void* pull_compressed_sample(const std::shared_ptr<ProbeApi>& api, void* probe_sink, void* bus,
                             std::chrono::steady_clock::time_point deadline,
                             std::string_view timeout_message) {
  while (true) {
    const auto poll_timeout =
        std::min(remaining_timeout_ns(deadline, timeout_message), kSamplePollTimeoutNs);
    if (void* sample = api->app_sink_try_pull_sample(probe_sink, poll_timeout); sample != nullptr) {
      return sample;
    }
    if (const auto error = pop_pipeline_error(api, bus); !error.empty()) {
      throw GpuVideoProbeError(error);
    }
    if (api->app_sink_is_eos(probe_sink) != 0) {
      return nullptr;
    }
  }
}

std::optional<std::uint64_t> segment_stream_origin(const std::shared_ptr<ProbeApi>& api,
                                                   const void* segment,
                                                   std::uint64_t sample_stream_time) {
  if (api->segment_position_from_stream_time(segment, kGstFormatTime, sample_stream_time) ==
      kGstClockTimeNone) {
    return std::nullopt;
  }

  std::uint64_t first = 0;
  std::uint64_t last = sample_stream_time;
  while (first < last) {
    const auto candidate = first + (last - first) / 2;
    if (api->segment_position_from_stream_time(segment, kGstFormatTime, candidate) ==
        kGstClockTimeNone) {
      first = candidate + 1;
    } else {
      last = candidate;
    }
  }
  return first;
}

struct FrameSeekResult {
  bool usable = true;
  bool available = false;
  std::uint64_t pts_ns = 0;
  std::uint64_t duration_ns = 0;
};

FrameSeekResult seek_compressed_frame(const std::shared_ptr<ProbeApi>& api, void* pipeline,
                                      void* probe_sink, void* bus, std::uint64_t stream_origin_ns,
                                      std::uint64_t frame_index, std::uint32_t fps_numerator,
                                      std::uint32_t fps_denominator,
                                      std::chrono::steady_clock::time_point deadline) {
  const auto relative_target_ns = frame_timestamp_ns(frame_index, fps_numerator, fps_denominator);
  if (relative_target_ns > std::numeric_limits<std::uint64_t>::max() - stream_origin_ns) {
    return {.usable = false};
  }
  const auto target_ns = stream_origin_ns + relative_target_ns;
  if (target_ns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      api->element_seek_simple(pipeline, kGstFormatTime, kGstSeekFlagFlushAccurate,
                               static_cast<std::int64_t>(target_ns)) == 0) {
    return {.usable = false};
  }

  const auto period_numerator = kNanosecondsPerSecond * fps_denominator;
  const auto half_frame_ns = period_numerator / (2ULL * fps_numerator);
  const auto earliest_matching_pts = target_ns > half_frame_ns ? target_ns - half_frame_ns : 0;
  const auto nominal_duration_ns = std::max<std::uint64_t>(period_numerator / fps_numerator, 1);
  while (true) {
    void* sample =
        pull_compressed_sample(api, probe_sink, bus, deadline,
                               "GStreamer parser-only probe timed out while seeking stream end");
    std::unique_ptr<void, ProbeApi::SampleUnref> sample_owner(sample, api->sample_unref);
    if (sample == nullptr) {
      return {.usable = true, .available = false};
    }

    const auto* buffer = static_cast<const GstBufferAbi*>(api->sample_get_buffer(sample));
    if (buffer == nullptr) {
      throw GpuVideoProbeError("GStreamer parser-only probe returned a sample without a buffer");
    }
    if (buffer->pts == kGstClockTimeNone) {
      return {.usable = false};
    }
    const void* segment = api->sample_get_segment(sample);
    if (segment == nullptr) {
      return {.usable = false};
    }
    const auto stream_pts = api->segment_to_stream_time(segment, kGstFormatTime, buffer->pts);
    if (stream_pts == kGstClockTimeNone) {
      continue;
    }
    if (stream_pts < earliest_matching_pts) {
      continue;
    }
    return {.usable = true,
            .available = true,
            .pts_ns = stream_pts,
            .duration_ns = buffer->duration == 0 || buffer->duration == kGstClockTimeNone
                               ? nominal_duration_ns
                               : buffer->duration};
  }
}

std::optional<std::uint64_t>
selected_stream_duration(const std::shared_ptr<ProbeApi>& api, void* pipeline, void* probe_sink,
                         void* bus, std::uint64_t container_duration_ns,
                         std::uint64_t stream_origin_ns, std::uint32_t fps_numerator,
                         std::uint32_t fps_denominator,
                         std::chrono::steady_clock::time_point deadline) {
  const auto duration_frame_ceiling =
      frame_count_ceiling_for_duration(container_duration_ns, fps_numerator, fps_denominator);
  if (duration_frame_ceiling == std::numeric_limits<std::uint64_t>::max()) {
    return std::nullopt;
  }
  const auto search_frame_limit = duration_frame_ceiling + 1;
  const auto maximum_stream_span_ns =
      minimum_duration_for_frame_count(search_frame_limit, fps_numerator, fps_denominator);
  if (maximum_stream_span_ns == std::numeric_limits<std::uint64_t>::max()) {
    return std::nullopt;
  }

  std::uint64_t first = 0;
  std::uint64_t last = search_frame_limit;
  std::optional<std::pair<std::uint64_t, FrameSeekResult>> final_available_frame;
  while (first < last) {
    const auto candidate = first + (last - first) / 2;
    const auto result = seek_compressed_frame(api, pipeline, probe_sink, bus, stream_origin_ns,
                                              candidate, fps_numerator, fps_denominator, deadline);
    if (!result.usable) {
      return std::nullopt;
    }
    if (result.available) {
      if (!final_available_frame.has_value() || candidate > final_available_frame->first) {
        final_available_frame = std::pair(candidate, result);
      }
      first = candidate + 1;
    } else {
      last = candidate;
    }
  }
  if (first == 0) {
    throw GpuVideoProbeError("video parser found no frames in the selected stream");
  }
  if (first == search_frame_limit) {
    return std::nullopt;
  }
  if (!final_available_frame.has_value() || final_available_frame->first != first - 1) {
    throw GpuVideoProbeError("video parser could not correlate the selected stream duration");
  }
  const auto& final_frame = final_available_frame->second;
  const auto frame_end =
      final_frame.duration_ns > std::numeric_limits<std::uint64_t>::max() - final_frame.pts_ns
          ? std::numeric_limits<std::uint64_t>::max()
          : final_frame.pts_ns + final_frame.duration_ns;
  if (frame_end <= stream_origin_ns) {
    return std::nullopt;
  }
  return std::min(frame_end - stream_origin_ns, maximum_stream_span_ns);
}

} // namespace

GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config, std::uint64_t timeout_ns) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  if (timeout_ns < kMinimumProbeTimeoutNs || timeout_ns > kMaximumProbeTimeoutNs) {
    throw std::invalid_argument("video probe timeout must be between one second and one hour");
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);

  std::error_code path_error;
  const auto absolute_path = std::filesystem::absolute(config.path, path_error);
  if (path_error) {
    throw GpuVideoProbeError("failed to resolve video path: " + path_error.message());
  }
  if (!std::filesystem::is_regular_file(absolute_path, path_error) || path_error) {
    throw GpuVideoProbeError("video probe path is not a readable regular file: " +
                             absolute_path.string());
  }

  auto api = std::make_shared<ProbeApi>();
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  std::uint32_t micro = 0;
  std::uint32_t nano = 0;
  api->version(&major, &minor, &micro, &nano);
  if (major != 1 || minor < 10) {
    throw GpuVideoProbeError("GStreamer 1.10 or newer is required, found " + std::to_string(major) +
                             "." + std::to_string(minor));
  }

  GErrorAbi* error = nullptr;
  if (api->init_check(nullptr, nullptr, &error) == 0) {
    throw GpuVideoProbeError("GStreamer initialization failed: " +
                             take_error(*api, error, "unknown initialization error"));
  }
  if (error != nullptr) {
    api->error_free(error);
    error = nullptr;
  }

  const auto pipeline_description = build_probe_pipeline(config, absolute_path);
  void* pipeline = api->parse_launch(pipeline_description.c_str(), &error);
  std::unique_ptr<void, ProbeApi::ObjectUnref> pipeline_owner(pipeline, api->object_unref);
  if (pipeline == nullptr || error != nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe construction failed: " +
                             take_error(*api, error, "pipeline construction returned no object"));
  }

  void* probe_info = api->bin_get_by_name(pipeline, "probe_info");
  std::unique_ptr<void, ProbeApi::ObjectUnref> probe_info_owner(probe_info, api->object_unref);
  if (probe_info == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no metadata identity");
  }
  void* probe_pad = api->element_get_static_pad(probe_info, "src");
  std::unique_ptr<void, ProbeApi::ObjectUnref> probe_pad_owner(probe_pad, api->object_unref);
  if (probe_pad == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no metadata pad");
  }
  void* probe_sink = api->bin_get_by_name(pipeline, "probe_sink");
  std::unique_ptr<void, ProbeApi::ObjectUnref> probe_sink_owner(probe_sink, api->object_unref);
  if (probe_sink == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no compressed-stream sink");
  }
  void* bus = api->element_get_bus(pipeline);
  std::unique_ptr<void, ProbeApi::ObjectUnref> bus_owner(bus, api->object_unref);
  if (bus == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no message bus");
  }

  struct ResetPipeline {
    std::shared_ptr<ProbeApi> api;
    void* pipeline = nullptr;
    ~ResetPipeline() { (void)api->element_set_state(pipeline, kGstStateNull); }
  } reset{api, pipeline};
  if (api->element_set_state(pipeline, kGstStatePlaying) == kGstStateChangeFailure) {
    throw GpuVideoProbeError("GStreamer parser-only probe failed to enter playing state");
  }
  void* initial_sample = nullptr;
  std::unique_ptr<void, ProbeApi::SampleUnref> initial_sample_owner(nullptr, api->sample_unref);
  std::uint64_t stream_origin_ns = 0;
  while (true) {
    initial_sample = pull_compressed_sample(
        api, probe_sink, bus, deadline,
        "GStreamer parser-only probe timed out while parsing stream metadata");
    initial_sample_owner.reset(initial_sample);
    if (initial_sample == nullptr) {
      throw GpuVideoProbeError(
          "video discovery found no H.264 or HEVC moving-video stream compatible with NVDEC");
    }
    const auto* initial_buffer =
        static_cast<const GstBufferAbi*>(api->sample_get_buffer(initial_sample));
    if (initial_buffer == nullptr) {
      throw GpuVideoProbeError("GStreamer parser-only probe returned a sample without a buffer");
    }
    if (config.elementary_stream) {
      break;
    }
    if (initial_buffer->pts == kGstClockTimeNone) {
      continue;
    }
    const void* initial_segment = api->sample_get_segment(initial_sample);
    if (initial_segment == nullptr) {
      continue;
    }
    const auto initial_stream_time =
        api->segment_to_stream_time(initial_segment, kGstFormatTime, initial_buffer->pts);
    if (initial_stream_time == kGstClockTimeNone) {
      continue;
    }
    const auto origin = segment_stream_origin(api, initial_segment, initial_stream_time);
    if (!origin.has_value()) {
      continue;
    }
    stream_origin_ns = *origin;
    break;
  }

  void* caps = api->sample_get_caps(initial_sample);
  if (caps == nullptr) {
    throw GpuVideoProbeError(
        "video discovery found no H.264 or HEVC moving-video stream compatible with NVDEC");
  }
  void* structure = api->caps_get_structure(caps, 0);
  if (structure == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe caps contain no structure");
  }

  const char* codec_caps = api->structure_get_name(structure);
  const char* stream_format = api->structure_get_string(structure, "stream-format");
  const char* alignment = api->structure_get_string(structure, "alignment");
  int parsed = 0;
  const bool supported_codec =
      codec_caps != nullptr && (std::string_view(codec_caps) == "video/x-h264" ||
                                std::string_view(codec_caps) == "video/x-h265");
  if (!supported_codec || api->structure_get_boolean(structure, "parsed", &parsed) == 0 ||
      parsed == 0 || stream_format == nullptr || std::string_view(stream_format) != "byte-stream" ||
      alignment == nullptr || std::string_view(alignment) != "au") {
    throw GpuVideoProbeError(
        "video parser output is not decoder-compatible H.264/HEVC byte-stream AU data");
  }

  int width = 0;
  int height = 0;
  int fps_numerator = 0;
  int fps_denominator = 0;
  if (api->structure_get_int(structure, "width", &width) == 0 ||
      api->structure_get_int(structure, "height", &height) == 0 || width <= 0 || height <= 0) {
    throw GpuVideoProbeError("video parser returned invalid visible frame dimensions");
  }
  if ((width % 2) != 0 || (height % 2) != 0) {
    throw GpuVideoProbeError("video parser returned visible dimensions incompatible with NV12");
  }
  if (api->structure_get_fraction(structure, "framerate", &fps_numerator, &fps_denominator) == 0 ||
      fps_numerator <= 0 || fps_denominator <= 0) {
    throw GpuVideoProbeError("video parser returned an invalid frame rate");
  }
  const double fps = static_cast<double>(fps_numerator) / fps_denominator;
  if (!std::isfinite(fps) || fps <= 0.0 || fps > kMaximumSaneFps) {
    throw GpuVideoProbeError("video parser returned an implausible frame rate");
  }
  initial_sample_owner.reset();

  std::int64_t queried_duration = 0;
  bool duration_is_estimated =
      api->element_query_duration(pipeline, kGstFormatTime, &queried_duration) == 0 ||
      queried_duration <= 0;
  auto duration_ns = duration_is_estimated ? kFallbackDurationSeconds * kNanosecondsPerSecond
                                           : static_cast<std::uint64_t>(queried_duration);
  if (!duration_is_estimated && !config.elementary_stream) {
    const auto selected_duration =
        selected_stream_duration(api, pipeline, probe_sink, bus, duration_ns, stream_origin_ns,
                                 static_cast<std::uint32_t>(fps_numerator),
                                 static_cast<std::uint32_t>(fps_denominator), deadline);
    if (selected_duration.has_value()) {
      duration_ns = *selected_duration;
    } else {
      duration_is_estimated = true;
    }
  }
  return {.width = static_cast<std::uint32_t>(width),
          .height = static_cast<std::uint32_t>(height),
          .fps_numerator = static_cast<std::uint32_t>(fps_numerator),
          .fps_denominator = static_cast<std::uint32_t>(fps_denominator),
          .fps = fps,
          .duration_ns = duration_ns,
          .total_frames =
              frame_count_for_duration(duration_ns, static_cast<std::uint32_t>(fps_numerator),
                                       static_cast<std::uint32_t>(fps_denominator)),
          .duration_is_estimated = duration_is_estimated};
}

} // namespace reco::io
