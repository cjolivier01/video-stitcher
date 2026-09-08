#include "reco/io/audio_passthrough.hpp"

#include "reco/core/path.hpp"
#if defined(_WIN32)
#include "reco/core/windows_runtime_library.hpp"
#endif
#include "reco/io/gpu_decode.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
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
  explicit DynamicLibrary(const std::filesystem::path& path) : path_(core::path_to_utf8(path)) {
#if defined(_WIN32)
    handle_ = static_cast<HMODULE>(core::detail::load_windows_runtime_library(path));
    if (handle_ == nullptr) {
      throw AudioPassthroughError("failed to load " + path_ + " (Windows error " +
                                  std::to_string(GetLastError()) + ")");
    }
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
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
  if (const auto override_path = core::path_from_environment(environment_variable);
      override_path.has_value()) {
    return std::make_shared<DynamicLibrary>(*override_path);
  }
  std::string errors;
  for (const char* name : names) {
    try {
      return std::make_shared<DynamicLibrary>(std::filesystem::path(name));
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
  using SampleGetSegment = const void* (*)(void*);
  using SampleUnref = void (*)(void*);
  using SegmentToStreamTime = std::uint64_t (*)(const void*, int, std::uint64_t);
  using CapsToString = char* (*)(const void*);
  using BufferGetSize = std::size_t (*)(const void*);
  using BufferExtract = std::size_t (*)(const void*, std::size_t, void*, std::size_t);
  using BusTimedPopFiltered = void* (*)(void*, std::uint64_t, std::uint32_t);
  using MessageParseError = void (*)(void*, GErrorAbi**, char**);
  using MessageUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);
  using DiscovererNew = void* (*)(std::uint64_t, GErrorAbi**);
  using DiscovererStart = void (*)(void*);
  using DiscovererStop = void (*)(void*);
  using DiscovererDiscoverUriAsync = int (*)(void*, const char*);
  using DiscovererInfoGetResult = int (*)(const void*);
  using DiscovererInfoGetAudioStreams = void* (*)(void*);
  using DiscovererStreamInfoListFree = void (*)(void*);
  using GObjectUnref = void (*)(void*);
  using GenericCallback = void (*)();
  using DestroyNotify = void (*)(void*);
  using SignalConnectData = unsigned long (*)(void*, const char*, GenericCallback, void*,
                                              DestroyNotify, int);
  using SignalHandlerDisconnect = void (*)(void*, unsigned long);
  using MainContextNew = void* (*)();
  using MainContextPushThreadDefault = void (*)(void*);
  using MainContextPopThreadDefault = void (*)(void*);
  using MainContextIteration = int (*)(void*, int);
  using MainContextWakeup = void (*)(void*);
  using MainContextUnref = void (*)(void*);

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
    sample_get_segment = core->symbol<SampleGetSegment>("gst_sample_get_segment");
    sample_unref = core->symbol<SampleUnref>("gst_sample_unref");
    segment_to_stream_time = core->symbol<SegmentToStreamTime>("gst_segment_to_stream_time");
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
    main_context_new = glib->symbol<MainContextNew>("g_main_context_new");
    main_context_push_thread_default =
        glib->symbol<MainContextPushThreadDefault>("g_main_context_push_thread_default");
    main_context_pop_thread_default =
        glib->symbol<MainContextPopThreadDefault>("g_main_context_pop_thread_default");
    main_context_iteration = glib->symbol<MainContextIteration>("g_main_context_iteration");
    main_context_wakeup = glib->symbol<MainContextWakeup>("g_main_context_wakeup");
    main_context_unref = glib->symbol<MainContextUnref>("g_main_context_unref");
    discoverer_new = pbutils->symbol<DiscovererNew>("gst_discoverer_new");
    discoverer_start = pbutils->symbol<DiscovererStart>("gst_discoverer_start");
    discoverer_stop = pbutils->symbol<DiscovererStop>("gst_discoverer_stop");
    discoverer_discover_uri_async =
        pbutils->symbol<DiscovererDiscoverUriAsync>("gst_discoverer_discover_uri_async");
    discoverer_info_get_result =
        pbutils->symbol<DiscovererInfoGetResult>("gst_discoverer_info_get_result");
    discoverer_info_get_audio_streams =
        pbutils->symbol<DiscovererInfoGetAudioStreams>("gst_discoverer_info_get_audio_streams");
    discoverer_stream_info_list_free =
        pbutils->symbol<DiscovererStreamInfoListFree>("gst_discoverer_stream_info_list_free");
    g_object_unref = gobject->symbol<GObjectUnref>("g_object_unref");
    signal_connect_data = gobject->symbol<SignalConnectData>("g_signal_connect_data");
    signal_handler_disconnect =
        gobject->symbol<SignalHandlerDisconnect>("g_signal_handler_disconnect");
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
  SampleGetSegment sample_get_segment = nullptr;
  SampleUnref sample_unref = nullptr;
  SegmentToStreamTime segment_to_stream_time = nullptr;
  CapsToString caps_to_string = nullptr;
  BufferGetSize buffer_get_size = nullptr;
  BufferExtract buffer_extract = nullptr;
  BusTimedPopFiltered bus_timed_pop_filtered = nullptr;
  MessageParseError message_parse_error = nullptr;
  MessageUnref message_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
  MainContextNew main_context_new = nullptr;
  MainContextPushThreadDefault main_context_push_thread_default = nullptr;
  MainContextPopThreadDefault main_context_pop_thread_default = nullptr;
  MainContextIteration main_context_iteration = nullptr;
  MainContextWakeup main_context_wakeup = nullptr;
  MainContextUnref main_context_unref = nullptr;
  DiscovererNew discoverer_new = nullptr;
  DiscovererStart discoverer_start = nullptr;
  DiscovererStop discoverer_stop = nullptr;
  DiscovererDiscoverUriAsync discoverer_discover_uri_async = nullptr;
  DiscovererInfoGetResult discoverer_info_get_result = nullptr;
  DiscovererInfoGetAudioStreams discoverer_info_get_audio_streams = nullptr;
  DiscovererStreamInfoListFree discoverer_stream_info_list_free = nullptr;
  GObjectUnref g_object_unref = nullptr;
  SignalConnectData signal_connect_data = nullptr;
  SignalHandlerDisconnect signal_handler_disconnect = nullptr;
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

std::uint64_t adjusted_timestamp(std::uint64_t stream_time, std::uint64_t trim_before,
                                 std::uint64_t output_anchor) {
  if (stream_time < trim_before) {
    throw AudioPassthroughError("compressed audio timestamp precedes the selected video range");
  }
  const auto delta = stream_time - trim_before;
  if (output_anchor > std::numeric_limits<std::uint64_t>::max() - delta) {
    throw AudioPassthroughError("compressed audio timestamp overflows the output timeline");
  }
  return output_anchor + delta;
}

struct DiscoverySignalState {
  GstreamerAudioApi* api = nullptr;
  bool completed = false;
  bool has_audio = false;
  bool callback_failed = false;
  int result = -1;
  std::string error;
};

void discovery_completed(void*, void* info, GErrorAbi* error, void* user_data) noexcept {
  auto* state = static_cast<DiscoverySignalState*>(user_data);
  if (state == nullptr || state->api == nullptr) {
    return;
  }
  try {
    if (error != nullptr && error->message != nullptr && error->message[0] != '\0') {
      state->error = error->message;
    }
    if (info != nullptr) {
      state->result = state->api->discoverer_info_get_result(info);
      void* streams = state->api->discoverer_info_get_audio_streams(info);
      state->has_audio = streams != nullptr;
      if (streams != nullptr) {
        state->api->discoverer_stream_info_list_free(streams);
      }
    }
  } catch (...) {
    state->callback_failed = true;
  }
  state->completed = true;
}

} // namespace

std::optional<std::uint64_t> CompressedAudioPacket::timestamp_ns() const noexcept {
  if (pts_ns.has_value() && dts_ns.has_value()) {
    return std::min(*pts_ns, *dts_ns);
  }
  return pts_ns.has_value() ? pts_ns : dts_ns;
}

std::optional<std::string> validate_audio_passthrough_config(const AudioPassthroughConfig& config) {
  if (config.segments.empty()) {
    return "audio passthrough requires at least one input segment";
  }
  for (const auto& segment : config.segments) {
    if (segment.path.empty()) {
      return "audio passthrough input paths must not be empty";
    }
    if (segment.path.find('\0') != std::string::npos) {
      return "audio passthrough input path contains an embedded NUL";
    }
    if (!gpu_decode_path_is_elementary_stream(segment.path) &&
        !gpu_decode_container_for_path(segment.path)) {
      return "audio passthrough requires a supported video input";
    }
    if (segment.video_duration_ns == 0 || segment.video_duration_ns == kGstClockTimeNone) {
      return "audio passthrough video segment duration must be finite and non-zero";
    }
  }
  if (config.start_time_ns >= config.segments.front().video_duration_ns) {
    return "audio passthrough start time must fall within the first selected segment";
  }
  if (config.read_timeout < std::chrono::milliseconds(1) ||
      config.read_timeout > std::chrono::hours(1)) {
    return "audio passthrough timeout must be between 1 millisecond and 1 hour";
  }
  return std::nullopt;
}

std::string build_audio_passthrough_pipeline(const AudioPassthroughSegment& segment) {
  const auto container = gpu_decode_container_for_path(segment.path);
  if (!container.has_value() || gpu_decode_path_is_elementary_stream(segment.path)) {
    throw std::invalid_argument("audio passthrough requires a supported container input");
  }
  std::ostringstream pipeline;
  if (segment.stable_source) {
    pipeline << "fdsrc fd=" << segment.stable_source->descriptor() << " ! ";
  } else {
    pipeline << "filesrc location=" << quote_property(segment.path) << " ! ";
  }
  pipeline << gpu_decode_container_demuxer(*container)
           << " ! capsfilter caps=\"audio/mpeg;audio/x-opus;audio/x-vorbis;audio/x-flac;"
              "audio/x-alac;audio/x-ac3;audio/x-eac3\""
           << " ! parsebin ! appsink name=audio_sink sync=false emit-signals=false "
              "max-buffers=1 drop=false";
  return pipeline.str();
}

std::string build_gstreamer_audio_passthrough_pipeline(std::string_view path) {
  return build_audio_passthrough_pipeline({.path = std::string(path)});
}

struct AudioPassthroughSource::Impl {
  explicit Impl(AudioPassthroughConfig config_value)
      : config(std::move(config_value)), api(std::make_shared<GstreamerAudioApi>()) {
    GErrorAbi* error = nullptr;
    const int initialized = api->init_check(nullptr, nullptr, &error);
    const std::unique_ptr<GErrorAbi, GstreamerAudioApi::ErrorFree> error_owner(error,
                                                                               api->error_free);
    if (initialized == 0) {
      std::string detail = "GStreamer initialization failed";
      if (error_owner != nullptr && error_owner->message != nullptr) {
        detail = error_owner->message;
      }
      throw AudioPassthroughError(detail);
    }
    try {
      prime();
    } catch (...) {
      close_segment();
      throw;
    }
  }

  ~Impl() { stop(); }

  void close_segment_locked() noexcept {
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
    active_stable_source.reset();
  }

  void close_segment() noexcept {
    std::lock_guard lock(resources_mutex);
    close_segment_locked();
  }

  void stop() noexcept {
    stopped.store(true, std::memory_order_release);
    {
      std::lock_guard lock(resources_mutex);
      if (active_discoverer != nullptr) {
        api->discoverer_stop(active_discoverer);
      }
      if (active_discovery_context != nullptr) {
        api->main_context_wakeup(active_discovery_context);
      }
      if (pipeline != nullptr) {
        (void)api->element_set_state(pipeline, kGstStateNull);
      }
    }
    std::lock_guard operation_lock(operation_mutex);
    pending.reset();
    close_segment();
  }

  std::string take_error(GErrorAbi*& error, std::string_view fallback) const {
    const std::unique_ptr<GErrorAbi, GstreamerAudioApi::ErrorFree> error_owner(
        std::exchange(error, nullptr), api->error_free);
    std::string detail(fallback);
    if (error_owner != nullptr && error_owner->message != nullptr &&
        error_owner->message[0] != '\0') {
      detail = error_owner->message;
    }
    return detail;
  }

  bool segment_has_audio(const AudioPassthroughSegment& segment) {
    GErrorAbi* error = nullptr;
    char* allocated_uri = nullptr;
    std::string retained_uri;
    if (segment.stable_source) {
      retained_uri = "fd://" + std::to_string(segment.stable_source->descriptor());
    } else {
      allocated_uri = api->filename_to_uri(segment.path.c_str(), &error);
      if (allocated_uri == nullptr) {
        throw AudioPassthroughError(take_error(error, "failed to create an audio input URI"));
      }
    }
    const std::unique_ptr<void, GstreamerAudioApi::Free> uri_owner(allocated_uri, api->free);
    const char* uri = segment.stable_source ? retained_uri.c_str() : allocated_uri;
    if (error != nullptr) {
      api->error_free(std::exchange(error, nullptr));
    }
    struct RewindStableSource {
      std::shared_ptr<const StableMediaFile> source;
      bool armed = true;
      ~RewindStableSource() {
        if (source && armed) {
          try {
            source->rewind();
          } catch (...) {
          }
        }
      }
    } rewind{segment.stable_source};

    void* context = api->main_context_new();
    if (context == nullptr) {
      throw AudioPassthroughError("failed to create an audio discovery main context");
    }
    const std::unique_ptr<void, GstreamerAudioApi::MainContextUnref> context_owner(
        context, api->main_context_unref);
    api->main_context_push_thread_default(context);
    struct PopThreadDefault {
      std::shared_ptr<GstreamerAudioApi> api;
      void* context = nullptr;
      ~PopThreadDefault() { api->main_context_pop_thread_default(context); }
    } pop_thread_default{api, context};

    void* discoverer = api->discoverer_new(timeout_ns(config.read_timeout), &error);
    if (discoverer == nullptr) {
      throw AudioPassthroughError(take_error(error, "failed to create GStreamer discoverer"));
    }
    const std::unique_ptr<void, GstreamerAudioApi::GObjectUnref> discoverer_owner(
        discoverer, api->g_object_unref);
    if (error != nullptr) {
      api->error_free(error);
      error = nullptr;
    }

    DiscoverySignalState discovery{.api = api.get()};
    const auto signal_id = api->signal_connect_data(
        discoverer, "discovered",
        reinterpret_cast<GstreamerAudioApi::GenericCallback>(discovery_completed), &discovery,
        nullptr, 0);
    if (signal_id == 0) {
      throw AudioPassthroughError("failed to connect the audio discovery result handler");
    }
    struct DisconnectSignal {
      std::shared_ptr<GstreamerAudioApi> api;
      void* discoverer = nullptr;
      unsigned long signal_id = 0;
      ~DisconnectSignal() { api->signal_handler_disconnect(discoverer, signal_id); }
    } disconnect_signal{api, discoverer, signal_id};

    struct ActiveDiscovery {
      Impl* owner = nullptr;
      void* discoverer = nullptr;
      void* context = nullptr;
      bool started = false;
      ~ActiveDiscovery() {
        {
          std::lock_guard lock(owner->resources_mutex);
          if (owner->active_discoverer == discoverer) {
            owner->active_discoverer = nullptr;
            owner->active_discovery_context = nullptr;
          }
        }
        if (started) {
          owner->api->discoverer_stop(discoverer);
        }
      }
    } active{this, discoverer, context};

    int queued = 0;
    {
      std::lock_guard lock(resources_mutex);
      if (stopped.load(std::memory_order_acquire)) {
        return false;
      }
      active_discoverer = discoverer;
      active_discovery_context = context;
      api->discoverer_start(discoverer);
      active.started = true;
      queued = api->discoverer_discover_uri_async(discoverer, uri);
    }
    if (queued == 0) {
      throw AudioPassthroughError("failed to queue audio input stream discovery");
    }
    while (!discovery.completed && !stopped.load(std::memory_order_acquire)) {
      (void)api->main_context_iteration(context, 1);
    }
    if (stopped.load(std::memory_order_acquire)) {
      return false;
    }
    if (discovery.callback_failed) {
      throw AudioPassthroughError("audio discovery result handling failed");
    }
    constexpr int kDiscovererOk = 0;
    constexpr int kDiscovererMissingPlugins = 5;
    if (discovery.result != kDiscovererOk && discovery.result != kDiscovererMissingPlugins) {
      throw AudioPassthroughError(discovery.error.empty()
                                      ? "GStreamer audio stream discovery failed"
                                      : std::move(discovery.error));
    }
    const bool has_audio = discovery.has_audio;
    if (segment.stable_source) {
      segment.stable_source->rewind();
      rewind.armed = false;
    }
    return has_audio;
  }

  std::optional<std::string> take_bus_error() {
    void* message = api->bus_timed_pop_filtered(bus, 0, kGstMessageError);
    if (message == nullptr) {
      return std::nullopt;
    }
    const std::unique_ptr<void, GstreamerAudioApi::MessageUnref> message_owner(message,
                                                                               api->message_unref);
    GErrorAbi* error = nullptr;
    char* debug = nullptr;
    api->message_parse_error(message, &error, &debug);
    const std::unique_ptr<GErrorAbi, GstreamerAudioApi::ErrorFree> error_owner(error,
                                                                               api->error_free);
    const std::unique_ptr<void, GstreamerAudioApi::Free> debug_owner(debug, api->free);
    std::string detail = "compressed audio pipeline failed";
    if (error != nullptr && error->message != nullptr) {
      detail = error->message;
    }
    return detail;
  }

  bool open_next_segment() {
    if (segment_output_end.has_value()) {
      next_output_ns = std::max(next_output_ns, *segment_output_end);
      segment_output_end.reset();
    }
    close_segment();
    if (stopped.load(std::memory_order_acquire)) {
      return false;
    }
    const AudioPassthroughSegment* segment = nullptr;
    for (;;) {
      if (next_segment >= config.segments.size()) {
        return false;
      }
      const auto segment_index = next_segment++;
      segment = &config.segments[segment_index];
      const auto trim = segment_index == 0 ? config.start_time_ns : 0U;
      const auto selected_duration = segment->video_duration_ns - trim;
      if (next_output_ns > std::numeric_limits<std::uint64_t>::max() - selected_duration) {
        throw AudioPassthroughError("audio passthrough video timeline overflows");
      }
      segment_output_anchor = next_output_ns;
      segment_output_end = next_output_ns + selected_duration;
      trim_before_ns = trim;
      const bool supported_container = !gpu_decode_path_is_elementary_stream(segment->path) &&
                                       gpu_decode_container_for_path(segment->path).has_value();
      if (supported_container) {
        auto readable_segment = *segment;
        if (segment->stable_source) {
          readable_segment.stable_source = segment->stable_source->open_cursor();
        }
        if (segment_has_audio(readable_segment)) {
          active_stable_source = std::move(readable_segment.stable_source);
          break;
        }
      }
      next_output_ns = *segment_output_end;
      segment_output_end.reset();
      if (stopped.load(std::memory_order_acquire)) {
        return false;
      }
    }
    auto readable_segment = *segment;
    readable_segment.stable_source = active_stable_source;
    const auto description = build_audio_passthrough_pipeline(readable_segment);
    GErrorAbi* error = nullptr;
    void* candidate_pipeline = api->parse_launch(description.c_str(), &error);
    const std::unique_ptr<GErrorAbi, GstreamerAudioApi::ErrorFree> parse_error_owner(
        error, api->error_free);
    std::unique_ptr<void, GstreamerAudioApi::ObjectUnref> candidate_pipeline_owner(
        candidate_pipeline, api->object_unref);
    if (candidate_pipeline == nullptr || error != nullptr) {
      std::string detail = "failed to construct compressed audio pipeline";
      if (parse_error_owner != nullptr && parse_error_owner->message != nullptr) {
        detail = parse_error_owner->message;
      }
      throw AudioPassthroughError(detail);
    }
    void* candidate_sink = api->bin_get_by_name(candidate_pipeline, "audio_sink");
    void* candidate_bus = api->element_get_bus(candidate_pipeline);
    std::unique_ptr<void, GstreamerAudioApi::ObjectUnref> candidate_sink_owner(candidate_sink,
                                                                               api->object_unref);
    std::unique_ptr<void, GstreamerAudioApi::ObjectUnref> candidate_bus_owner(candidate_bus,
                                                                              api->object_unref);
    if (candidate_sink == nullptr || candidate_bus == nullptr) {
      throw AudioPassthroughError("compressed audio pipeline is missing appsink or bus resources");
    }
    {
      std::lock_guard lock(resources_mutex);
      if (stopped.load(std::memory_order_acquire)) {
        return false;
      }
      pipeline = candidate_pipeline_owner.release();
      sink = candidate_sink_owner.release();
      bus = candidate_bus_owner.release();
      if (api->element_set_state(pipeline, kGstStatePlaying) == kGstStateChangeFailure) {
        close_segment_locked();
        throw AudioPassthroughError("compressed audio pipeline failed to enter PLAYING");
      }
    }
    int current = 0;
    int pending_state = 0;
    if (api->element_get_state(pipeline, &current, &pending_state,
                               timeout_ns(config.read_timeout)) == kGstStateChangeFailure) {
      if (stopped.load(std::memory_order_acquire)) {
        close_segment();
        return false;
      }
      const auto detail =
          take_bus_error().value_or("compressed audio pipeline failed during startup");
      close_segment();
      throw AudioPassthroughError(detail);
    }
    if (trim_before_ns > 0) {
      if (api->element_seek_simple(
              pipeline, kGstFormatTime, kGstSeekFlush | kGstSeekKeyUnit,
              static_cast<std::int64_t>(std::min<std::uint64_t>(
                  trim_before_ns,
                  static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))) == 0) {
        close_segment();
        throw AudioPassthroughError("failed to seek compressed audio to the requested trim point");
      }
    }
    return true;
  }

  std::optional<CompressedAudioPacket> pull_current() {
    for (;;) {
      void* sample = api->app_sink_try_pull_sample(sink, timeout_ns(config.read_timeout));
      const std::unique_ptr<void, GstreamerAudioApi::SampleUnref> sample_owner(sample,
                                                                               api->sample_unref);
      if (stopped.load(std::memory_order_acquire)) {
        return std::nullopt;
      }
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
      const void* segment = api->sample_get_segment(sample);
      if (buffer == nullptr || sample_caps == nullptr || segment == nullptr) {
        throw AudioPassthroughError("compressed audio sample is missing buffer or caps metadata");
      }
      char* caps_text = api->caps_to_string(sample_caps);
      const std::unique_ptr<void, GstreamerAudioApi::Free> caps_text_owner(caps_text, api->free);
      if (caps_text == nullptr || caps_text[0] == '\0') {
        throw AudioPassthroughError("compressed audio sample has empty negotiated caps");
      }
      const std::string negotiated_caps(caps_text);
      if (!caps_value.has_value()) {
        caps_value = negotiated_caps;
      } else if (*caps_value != negotiated_caps) {
        throw AudioPassthroughError("chained audio segments negotiated incompatible caps");
      }

      const auto* header = static_cast<const GstBufferAbi*>(buffer);
      const auto to_stream_time = [&](std::uint64_t value,
                                      std::string_view name) -> std::optional<std::uint64_t> {
        if (!finite_timestamp(value).has_value()) {
          return std::nullopt;
        }
        const auto stream_time = api->segment_to_stream_time(segment, kGstFormatTime, value);
        if (stream_time == kGstClockTimeNone) {
          throw AudioPassthroughError("compressed audio " + std::string(name) +
                                      " cannot be converted to stream time");
        }
        return stream_time;
      };
      const auto pts = to_stream_time(header->pts, "PTS");
      const auto dts = to_stream_time(header->dts, "DTS");
      const auto timestamp = pts.has_value() && dts.has_value()
                                 ? std::optional(std::min(*pts, *dts))
                                 : (pts.has_value() ? pts : dts);
      if (!timestamp.has_value()) {
        throw AudioPassthroughError(
            "compressed audio packet has no presentation or decode timestamp");
      }
      if (*timestamp < trim_before_ns) {
        continue;
      }

      const auto packet_size = api->buffer_get_size(buffer);
      if (packet_size == 0 || packet_size > kMaximumCompressedAudioPacketBytes) {
        throw AudioPassthroughError("compressed audio packet size is outside the supported bound");
      }
      CompressedAudioPacket packet;
      packet.bytes.resize(packet_size);
      if (api->buffer_extract(buffer, 0, packet.bytes.data(), packet.bytes.size()) !=
          packet.bytes.size()) {
        throw AudioPassthroughError("failed to copy the bounded compressed audio packet");
      }
      if (pts.has_value()) {
        packet.pts_ns = adjusted_timestamp(*pts, trim_before_ns, segment_output_anchor);
      }
      if (dts.has_value()) {
        packet.dts_ns = adjusted_timestamp(*dts, trim_before_ns, segment_output_anchor);
      }
      packet.duration_ns = header->duration == kGstClockTimeNone ? 0 : header->duration;
      packet.flags = header->mini_object.flags;

      const auto output_timestamp = packet.timestamp_ns().value_or(0);
      if (!segment_output_end.has_value() || output_timestamp >= *segment_output_end) {
        return std::nullopt;
      }
      packet.duration_ns = std::min(packet.duration_ns, *segment_output_end - output_timestamp);
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
    std::lock_guard operation_lock(operation_mutex);
    if (stopped.load(std::memory_order_acquire)) {
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
      if (stopped.load(std::memory_order_acquire)) {
        return {};
      }
      close_segment();
      if (!open_next_segment()) {
        return {};
      }
    }
  }

  AudioPassthroughConfig config;
  std::shared_ptr<GstreamerAudioApi> api;
  std::size_t next_segment = 0;
  void* pipeline = nullptr;
  void* sink = nullptr;
  void* bus = nullptr;
  void* active_discoverer = nullptr;
  void* active_discovery_context = nullptr;
  std::shared_ptr<const StableMediaFile> active_stable_source;
  std::optional<std::string> caps_value;
  std::optional<CompressedAudioPacket> pending;
  std::optional<std::uint64_t> segment_output_end;
  std::uint64_t segment_output_anchor = 0;
  std::uint64_t trim_before_ns = 0;
  std::uint64_t next_output_ns = 0;
  std::mutex operation_mutex;
  std::mutex resources_mutex;
  std::atomic<bool> stopped{false};
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
