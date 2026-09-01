#include "reco/io/audio_passthrough.hpp"

#include "reco/io/gpu_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reco::io {
namespace {

constexpr int kGstStateNull = 1;
constexpr int kGstStatePlaying = 4;
constexpr int kGstStateChangeFailure = 0;
constexpr int kGstFormatTime = 3;
constexpr int kGstSeekFlush = 1 << 0;
constexpr int kGstSeekKeyUnit = 1 << 2;
constexpr std::uint32_t kGstMessageEos = 1U << 0U;
constexpr std::uint32_t kGstMessageError = 1U << 1U;
constexpr std::uint64_t kGstClockTimeNone = std::numeric_limits<std::uint64_t>::max();
constexpr std::size_t kMaximumCompressedAudioPacketBytes = 16U * 1024U * 1024U;

struct GErrorAbi {
  std::uint32_t domain = 0;
  std::int32_t code = 0;
  char* message = nullptr;
};

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
      throw AudioPassthroughError("failed to load " + path_ + " (Windows error " +
                                  std::to_string(GetLastError()) + ")");
    }
#else
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw AudioPassthroughError("failed to load " + path_ +
                                  (error == nullptr ? "" : ": " + std::string(error)));
    }
#endif
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  ~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      (void)FreeLibrary(handle_);
    }
#else
    if (handle_ != nullptr) {
      (void)dlclose(handle_);
    }
#endif
  }

  template <typename Function> Function symbol(const char* name) const {
#if defined(_WIN32)
    auto* value = GetProcAddress(handle_, name);
#else
    dlerror();
    void* value = dlsym(handle_, name);
#endif
    if (value == nullptr) {
      throw AudioPassthroughError("missing GStreamer runtime symbol " + std::string(name) + " in " +
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
    } catch (const AudioPassthroughError& error) {
      if (!errors.empty()) {
        errors += "; ";
      }
      errors += error.what();
    }
  }
  throw AudioPassthroughError("could not load " + std::string(component) + " runtime: " + errors);
}

class GstreamerAudioApi {
public:
  using InitCheck = int (*)(int*, char***, GErrorAbi**);
  using FilenameToUri = char* (*)(const char*, GErrorAbi**);
  using ParseLaunch = void* (*)(const char*, GErrorAbi**);
  using BinGetByName = void* (*)(void*, const char*);
  using ElementSetState = int (*)(void*, int);
  using ElementGetState = int (*)(void*, int*, int*, std::uint64_t);
  using ElementGetBus = void* (*)(void*);
  using ElementSeekSimple = int (*)(void*, int, int, std::int64_t);
  using ObjectUnref = void (*)(void*);
  using AppSinkTryPullSample = void* (*)(void*, std::uint64_t);
  using AppSinkIsEos = int (*)(void*);
  using SampleGetBuffer = void* (*)(void*);
  using SampleGetCaps = void* (*)(void*);
  using SampleUnref = void (*)(void*);
  using CapsToString = char* (*)(const void*);
  using BufferGetSize = std::size_t (*)(const void*);
  using BufferExtract = std::size_t (*)(const void*, std::size_t, void*, std::size_t);
  using BusTimedPopFiltered = void* (*)(void*, std::uint64_t, std::uint32_t);
  using MessageParseError = void (*)(void*, GErrorAbi**, char**);
  using MessageUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);
  using DiscovererNew = void* (*)(std::uint64_t, GErrorAbi**);
  using DiscovererDiscoverUri = void* (*)(void*, const char*, GErrorAbi**);
  using DiscovererInfoGetResult = int (*)(const void*);
  using DiscovererInfoGetAudioStreams = void* (*)(void*);
  using DiscovererStreamInfoListFree = void (*)(void*);
  using GObjectUnref = void (*)(void*);

  GstreamerAudioApi() {
#if defined(_WIN32)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH", {"gstreamer-1.0-0.dll"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"gstapp-1.0-0.dll"}, "GstApp");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0-0.dll", "glib-2.0-0.dll"}, "GLib");
    pbutils =
        load_library("RECO_GSTPBUTILS_DYLIB_PATH", {"gstpbutils-1.0-0.dll"}, "GStreamer PbUtils");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH", {"libgobject-2.0-0.dll", "gobject-2.0-0.dll"},
                           "GObject");
#elif defined(__APPLE__)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"libgstapp-1.0.0.dylib", "libgstapp-1.0.dylib"},
                       "GstApp");
    glib =
        load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.0.dylib", "libglib-2.0.dylib"}, "GLib");
    pbutils =
        load_library("RECO_GSTPBUTILS_DYLIB_PATH",
                     {"libgstpbutils-1.0.0.dylib", "libgstpbutils-1.0.dylib"}, "GStreamer PbUtils");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH",
                           {"libgobject-2.0.0.dylib", "libgobject-2.0.dylib"}, "GObject");
#else
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"libgstapp-1.0.so.0", "libgstapp-1.0.so"},
                       "GstApp");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.so.0", "libglib-2.0.so"}, "GLib");
    pbutils = load_library("RECO_GSTPBUTILS_DYLIB_PATH",
                           {"libgstpbutils-1.0.so.0", "libgstpbutils-1.0.so"}, "GStreamer PbUtils");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH", {"libgobject-2.0.so.0", "libgobject-2.0.so"},
                           "GObject");
#endif
    init_check = core->symbol<InitCheck>("gst_init_check");
    filename_to_uri = core->symbol<FilenameToUri>("gst_filename_to_uri");
    parse_launch = core->symbol<ParseLaunch>("gst_parse_launch");
    bin_get_by_name = core->symbol<BinGetByName>("gst_bin_get_by_name");
    element_set_state = core->symbol<ElementSetState>("gst_element_set_state");
    element_get_state = core->symbol<ElementGetState>("gst_element_get_state");
    element_get_bus = core->symbol<ElementGetBus>("gst_element_get_bus");
    element_seek_simple = core->symbol<ElementSeekSimple>("gst_element_seek_simple");
    object_unref = core->symbol<ObjectUnref>("gst_object_unref");
    sample_get_buffer = core->symbol<SampleGetBuffer>("gst_sample_get_buffer");
    sample_get_caps = core->symbol<SampleGetCaps>("gst_sample_get_caps");
    sample_unref = core->symbol<SampleUnref>("gst_sample_unref");
    caps_to_string = core->symbol<CapsToString>("gst_caps_to_string");
    buffer_get_size = core->symbol<BufferGetSize>("gst_buffer_get_size");
    buffer_extract = core->symbol<BufferExtract>("gst_buffer_extract");
    bus_timed_pop_filtered = core->symbol<BusTimedPopFiltered>("gst_bus_timed_pop_filtered");
    message_parse_error = core->symbol<MessageParseError>("gst_message_parse_error");
    message_unref = core->symbol<MessageUnref>("gst_message_unref");
    app_sink_try_pull_sample = app->symbol<AppSinkTryPullSample>("gst_app_sink_try_pull_sample");
    app_sink_is_eos = app->symbol<AppSinkIsEos>("gst_app_sink_is_eos");
    error_free = glib->symbol<ErrorFree>("g_error_free");
    free = glib->symbol<Free>("g_free");
    discoverer_new = pbutils->symbol<DiscovererNew>("gst_discoverer_new");
    discoverer_discover_uri = pbutils->symbol<DiscovererDiscoverUri>("gst_discoverer_discover_uri");
    discoverer_info_get_result =
        pbutils->symbol<DiscovererInfoGetResult>("gst_discoverer_info_get_result");
    discoverer_info_get_audio_streams =
        pbutils->symbol<DiscovererInfoGetAudioStreams>("gst_discoverer_info_get_audio_streams");
    discoverer_stream_info_list_free =
        pbutils->symbol<DiscovererStreamInfoListFree>("gst_discoverer_stream_info_list_free");
    g_object_unref = gobject->symbol<GObjectUnref>("g_object_unref");
  }

  std::shared_ptr<DynamicLibrary> core;
  std::shared_ptr<DynamicLibrary> app;
  std::shared_ptr<DynamicLibrary> glib;
  std::shared_ptr<DynamicLibrary> pbutils;
  std::shared_ptr<DynamicLibrary> gobject;
  InitCheck init_check = nullptr;
  FilenameToUri filename_to_uri = nullptr;
  ParseLaunch parse_launch = nullptr;
  BinGetByName bin_get_by_name = nullptr;
  ElementSetState element_set_state = nullptr;
  ElementGetState element_get_state = nullptr;
  ElementGetBus element_get_bus = nullptr;
  ElementSeekSimple element_seek_simple = nullptr;
  ObjectUnref object_unref = nullptr;
  AppSinkTryPullSample app_sink_try_pull_sample = nullptr;
  AppSinkIsEos app_sink_is_eos = nullptr;
  SampleGetBuffer sample_get_buffer = nullptr;
  SampleGetCaps sample_get_caps = nullptr;
  SampleUnref sample_unref = nullptr;
  CapsToString caps_to_string = nullptr;
  BufferGetSize buffer_get_size = nullptr;
  BufferExtract buffer_extract = nullptr;
  BusTimedPopFiltered bus_timed_pop_filtered = nullptr;
  MessageParseError message_parse_error = nullptr;
  MessageUnref message_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
  DiscovererNew discoverer_new = nullptr;
  DiscovererDiscoverUri discoverer_discover_uri = nullptr;
  DiscovererInfoGetResult discoverer_info_get_result = nullptr;
  DiscovererInfoGetAudioStreams discoverer_info_get_audio_streams = nullptr;
  DiscovererStreamInfoListFree discoverer_stream_info_list_free = nullptr;
  GObjectUnref g_object_unref = nullptr;
};

std::string quote_property(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 2U);
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

std::uint64_t timeout_ns(std::chrono::milliseconds timeout) {
  return static_cast<std::uint64_t>(timeout.count()) * 1'000'000ULL;
}

std::optional<std::uint64_t> finite_timestamp(std::uint64_t value) {
  return value == kGstClockTimeNone ? std::nullopt : std::optional(value);
}

std::uint64_t adjusted_timestamp(std::uint64_t value, std::uint64_t source_anchor,
                                 std::uint64_t output_anchor) {
  if (value <= source_anchor) {
    return output_anchor;
  }
  const auto delta = value - source_anchor;
  if (output_anchor > std::numeric_limits<std::uint64_t>::max() - delta) {
    throw AudioPassthroughError("compressed audio timestamp overflows the output timeline");
  }
  return output_anchor + delta;
}

} // namespace

std::optional<std::uint64_t> CompressedAudioPacket::timestamp_ns() const noexcept {
  if (pts_ns.has_value() && dts_ns.has_value()) {
    return std::min(*pts_ns, *dts_ns);
  }
  return pts_ns.has_value() ? pts_ns : dts_ns;
}

std::optional<std::string> validate_audio_passthrough_config(const AudioPassthroughConfig& config) {
  if (config.paths.empty()) {
    return "audio passthrough requires at least one input segment";
  }
  for (const auto& path : config.paths) {
    if (path.empty()) {
      return "audio passthrough input paths must not be empty";
    }
    if (path.find('\0') != std::string::npos) {
      return "audio passthrough input path contains an embedded NUL";
    }
    if (gpu_decode_path_is_elementary_stream(path) || !gpu_decode_container_for_path(path)) {
      return "audio passthrough requires a supported container input";
    }
  }
  if (config.read_timeout < std::chrono::milliseconds(1) ||
      config.read_timeout > std::chrono::hours(1)) {
    return "audio passthrough timeout must be between 1 millisecond and 1 hour";
  }
  return std::nullopt;
}

std::string build_gstreamer_audio_passthrough_pipeline(std::string_view path) {
  const auto container = gpu_decode_container_for_path(path);
  if (!container.has_value() || gpu_decode_path_is_elementary_stream(path)) {
    throw std::invalid_argument("audio passthrough requires a supported container input");
  }
  std::ostringstream pipeline;
  pipeline << "filesrc location=" << quote_property(path) << " ! "
           << gpu_decode_container_demuxer(*container)
           << " ! capsfilter caps=\"audio/mpeg;audio/x-opus;audio/x-vorbis;audio/x-flac;"
              "audio/x-alac;audio/x-ac3;audio/x-eac3\""
           << " ! parsebin ! appsink name=audio_sink sync=false emit-signals=false "
              "max-buffers=1 drop=false";
  return pipeline.str();
}

struct AudioPassthroughSource::Impl {
  explicit Impl(AudioPassthroughConfig config_value)
      : config(std::move(config_value)), api(std::make_shared<GstreamerAudioApi>()) {
    GErrorAbi* error = nullptr;
    if (api->init_check(nullptr, nullptr, &error) == 0) {
      std::string detail = "GStreamer initialization failed";
      if (error != nullptr) {
        if (error->message != nullptr) {
          detail = error->message;
        }
        api->error_free(error);
      }
      throw AudioPassthroughError(detail);
    }
    prime();
  }

  ~Impl() { stop(); }

  void close_segment() noexcept {
    if (pipeline != nullptr) {
      (void)api->element_set_state(pipeline, kGstStateNull);
    }
    if (sink != nullptr) {
      api->object_unref(sink);
      sink = nullptr;
    }
    if (bus != nullptr) {
      api->object_unref(bus);
      bus = nullptr;
    }
    if (pipeline != nullptr) {
      api->object_unref(pipeline);
      pipeline = nullptr;
    }
  }

  void stop() noexcept {
    stopped = true;
    pending.reset();
    close_segment();
  }

  std::string take_error(GErrorAbi*& error, std::string_view fallback) const {
    std::string detail(fallback);
    if (error != nullptr) {
      if (error->message != nullptr && error->message[0] != '\0') {
        detail = error->message;
      }
      api->error_free(error);
      error = nullptr;
    }
    return detail;
  }

  bool segment_has_audio(std::string_view path) const {
    GErrorAbi* error = nullptr;
    char* uri = api->filename_to_uri(std::string(path).c_str(), &error);
    if (uri == nullptr) {
      throw AudioPassthroughError(take_error(error, "failed to create an audio input URI"));
    }
    const std::unique_ptr<void, GstreamerAudioApi::Free> uri_owner(uri, api->free);

    void* discoverer = api->discoverer_new(timeout_ns(config.read_timeout), &error);
    if (discoverer == nullptr) {
      throw AudioPassthroughError(take_error(error, "failed to create GStreamer discoverer"));
    }
    const std::unique_ptr<void, GstreamerAudioApi::GObjectUnref> discoverer_owner(
        discoverer, api->g_object_unref);
    void* info = api->discoverer_discover_uri(discoverer, static_cast<const char*>(uri), &error);
    if (info == nullptr) {
      throw AudioPassthroughError(take_error(error, "failed to inspect audio input streams"));
    }
    const std::unique_ptr<void, GstreamerAudioApi::GObjectUnref> info_owner(info,
                                                                            api->g_object_unref);
    const int result = api->discoverer_info_get_result(info);
    constexpr int kDiscovererOk = 0;
    constexpr int kDiscovererMissingPlugins = 5;
    if (result != kDiscovererOk && result != kDiscovererMissingPlugins) {
      throw AudioPassthroughError(take_error(error, "GStreamer audio stream discovery failed"));
    }
    if (error != nullptr) {
      api->error_free(error);
    }
    void* streams = api->discoverer_info_get_audio_streams(info);
    if (streams == nullptr) {
      return false;
    }
    api->discoverer_stream_info_list_free(streams);
    return true;
  }

  std::optional<std::string> take_bus_error() {
    void* message = api->bus_timed_pop_filtered(bus, 0, kGstMessageError);
    if (message == nullptr) {
      return std::nullopt;
    }
    GErrorAbi* error = nullptr;
    char* debug = nullptr;
    api->message_parse_error(message, &error, &debug);
    std::string detail = "compressed audio pipeline failed";
    if (error != nullptr && error->message != nullptr) {
      detail = error->message;
    }
    if (error != nullptr) {
      api->error_free(error);
    }
    if (debug != nullptr) {
      api->free(debug);
    }
    api->message_unref(message);
    return detail;
  }

  bool open_next_segment() {
    close_segment();
    if (stopped) {
      return false;
    }
    while (next_path < config.paths.size() && !segment_has_audio(config.paths[next_path])) {
      ++next_path;
    }
    if (next_path >= config.paths.size()) {
      return false;
    }
    const bool first_supplied_segment = next_path == 0;
    const auto description = build_gstreamer_audio_passthrough_pipeline(config.paths[next_path++]);
    GErrorAbi* error = nullptr;
    pipeline = api->parse_launch(description.c_str(), &error);
    if (pipeline == nullptr || error != nullptr) {
      std::string detail = "failed to construct compressed audio pipeline";
      if (error != nullptr) {
        if (error->message != nullptr) {
          detail = error->message;
        }
        api->error_free(error);
      }
      close_segment();
      throw AudioPassthroughError(detail);
    }
    sink = api->bin_get_by_name(pipeline, "audio_sink");
    bus = api->element_get_bus(pipeline);
    if (sink == nullptr || bus == nullptr) {
      close_segment();
      throw AudioPassthroughError("compressed audio pipeline is missing appsink or bus resources");
    }
    if (api->element_set_state(pipeline, kGstStatePlaying) == kGstStateChangeFailure) {
      close_segment();
      throw AudioPassthroughError("compressed audio pipeline failed to enter PLAYING");
    }
    int current = 0;
    int pending_state = 0;
    if (api->element_get_state(pipeline, &current, &pending_state,
                               timeout_ns(config.read_timeout)) == kGstStateChangeFailure) {
      const auto detail =
          take_bus_error().value_or("compressed audio pipeline failed during startup");
      close_segment();
      throw AudioPassthroughError(detail);
    }
    segment_output_anchor = next_output_ns;
    segment_source_anchor.reset();
    trim_before_ns = first_supplied_segment ? config.start_time_ns : 0;
    if (trim_before_ns > 0) {
      (void)api->element_seek_simple(
          pipeline, kGstFormatTime, kGstSeekFlush | kGstSeekKeyUnit,
          static_cast<std::int64_t>(std::min<std::uint64_t>(
              trim_before_ns,
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))));
    }
    return true;
  }

  std::optional<CompressedAudioPacket> pull_current() {
    for (;;) {
      void* sample = api->app_sink_try_pull_sample(sink, timeout_ns(config.read_timeout));
      if (sample == nullptr) {
        if (const auto error = take_bus_error(); error.has_value()) {
          throw AudioPassthroughError(*error);
        }
        void* eos = api->bus_timed_pop_filtered(bus, 0, kGstMessageEos);
        if (api->app_sink_is_eos(sink) != 0 || eos != nullptr) {
          if (eos != nullptr) {
            api->message_unref(eos);
          }
          return std::nullopt;
        }
        throw AudioPassthroughError("timed out waiting for a compressed audio packet");
      }

      void* buffer = api->sample_get_buffer(sample);
      void* sample_caps = api->sample_get_caps(sample);
      if (buffer == nullptr || sample_caps == nullptr) {
        api->sample_unref(sample);
        throw AudioPassthroughError("compressed audio sample is missing buffer or caps metadata");
      }
      char* caps_text = api->caps_to_string(sample_caps);
      if (caps_text == nullptr || caps_text[0] == '\0') {
        if (caps_text != nullptr) {
          api->free(caps_text);
        }
        api->sample_unref(sample);
        throw AudioPassthroughError("compressed audio sample has empty negotiated caps");
      }
      const std::string negotiated_caps(caps_text);
      api->free(caps_text);
      if (!caps_value.has_value()) {
        caps_value = negotiated_caps;
      } else if (*caps_value != negotiated_caps) {
        api->sample_unref(sample);
        throw AudioPassthroughError("chained audio segments negotiated incompatible caps");
      }

      const auto* header = static_cast<const GstBufferAbi*>(buffer);
      const auto pts = finite_timestamp(header->pts);
      const auto dts = finite_timestamp(header->dts);
      const auto timestamp = pts.has_value() && dts.has_value()
                                 ? std::optional(std::min(*pts, *dts))
                                 : (pts.has_value() ? pts : dts);
      if (!timestamp.has_value()) {
        api->sample_unref(sample);
        throw AudioPassthroughError(
            "compressed audio packet has no presentation or decode timestamp");
      }
      if (*timestamp < trim_before_ns) {
        api->sample_unref(sample);
        continue;
      }
      if (!segment_source_anchor.has_value()) {
        segment_source_anchor = trim_before_ns > 0 ? trim_before_ns : *timestamp;
      }

      const auto packet_size = api->buffer_get_size(buffer);
      if (packet_size == 0 || packet_size > kMaximumCompressedAudioPacketBytes) {
        api->sample_unref(sample);
        throw AudioPassthroughError("compressed audio packet size is outside the supported bound");
      }
      CompressedAudioPacket packet;
      packet.bytes.resize(packet_size);
      if (api->buffer_extract(buffer, 0, packet.bytes.data(), packet.bytes.size()) !=
          packet.bytes.size()) {
        api->sample_unref(sample);
        throw AudioPassthroughError("failed to copy the bounded compressed audio packet");
      }
      const auto source_anchor = *segment_source_anchor;
      if (pts.has_value()) {
        packet.pts_ns = adjusted_timestamp(*pts, source_anchor, segment_output_anchor);
      }
      if (dts.has_value()) {
        packet.dts_ns = adjusted_timestamp(*dts, source_anchor, segment_output_anchor);
      }
      packet.duration_ns = header->duration == kGstClockTimeNone ? 0 : header->duration;
      packet.flags = header->mini_object.flags;
      api->sample_unref(sample);

      const auto output_timestamp = packet.pts_ns.value_or(packet.dts_ns.value_or(0));
      const auto duration = packet.duration_ns;
      next_output_ns = output_timestamp > std::numeric_limits<std::uint64_t>::max() - duration
                           ? std::numeric_limits<std::uint64_t>::max()
                           : std::max(next_output_ns, output_timestamp + duration);
      return packet;
    }
  }

  void prime() {
    while (open_next_segment()) {
      auto packet = pull_current();
      if (packet.has_value()) {
        pending = std::move(*packet);
        return;
      }
    }
  }

  AudioPassthroughReadResult read() {
    if (stopped) {
      return {};
    }
    if (pending.has_value()) {
      auto packet = std::move(pending);
      pending.reset();
      return {.status = AudioPassthroughStatus::Packet, .packet = std::move(packet)};
    }
    for (;;) {
      if (pipeline == nullptr && !open_next_segment()) {
        return {};
      }
      auto packet = pull_current();
      if (packet.has_value()) {
        return {.status = AudioPassthroughStatus::Packet, .packet = std::move(packet)};
      }
      close_segment();
      if (!open_next_segment()) {
        return {};
      }
    }
  }

  AudioPassthroughConfig config;
  std::shared_ptr<GstreamerAudioApi> api;
  std::size_t next_path = 0;
  void* pipeline = nullptr;
  void* sink = nullptr;
  void* bus = nullptr;
  std::optional<std::string> caps_value;
  std::optional<CompressedAudioPacket> pending;
  std::optional<std::uint64_t> segment_source_anchor;
  std::uint64_t segment_output_anchor = 0;
  std::uint64_t trim_before_ns = 0;
  std::uint64_t next_output_ns = 0;
  bool stopped = false;
};

AudioPassthroughSource AudioPassthroughSource::open(AudioPassthroughConfig config) {
  if (const auto error = validate_audio_passthrough_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  return AudioPassthroughSource(std::make_unique<Impl>(std::move(config)));
}

AudioPassthroughSource::AudioPassthroughSource(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AudioPassthroughSource::AudioPassthroughSource(AudioPassthroughSource&&) noexcept = default;
AudioPassthroughSource&
AudioPassthroughSource::operator=(AudioPassthroughSource&&) noexcept = default;
AudioPassthroughSource::~AudioPassthroughSource() = default;

std::optional<std::string> AudioPassthroughSource::caps() const {
  return impl_ == nullptr ? std::nullopt : impl_->caps_value;
}

AudioPassthroughReadResult AudioPassthroughSource::read() {
  if (impl_ == nullptr) {
    return {};
  }
  return impl_->read();
}

void AudioPassthroughSource::request_stop() noexcept {
  if (impl_ != nullptr) {
    impl_->stop();
  }
}

} // namespace reco::io
