#include "reco/io/gpu_encode.hpp"

#include "reco/core/path.hpp"
#if defined(_WIN32)
#include "reco/core/windows_runtime_library.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
constexpr int kGstStateChangeSuccess = 1;
constexpr std::uint32_t kGstMessageEos = 1U << 0U;
constexpr std::uint32_t kGstMessageError = 1U << 1U;
constexpr std::uint64_t kGstClockTimeNone = std::numeric_limits<std::uint64_t>::max();
constexpr int kGstFlowOk = 0;
constexpr std::size_t kMaximumAudioPacketsInFlight = 32;
constexpr std::size_t kMaximumCompressedAudioPacketBytes = 16U * 1024U * 1024U;
constexpr std::size_t kMaximumCompressedAudioBytesInFlight = 32U * 1024U * 1024U;
constexpr std::uint32_t kGstMiniObjectFlagMask = (1U << 4U) - 1U;
constexpr std::uint32_t kGstBufferFlagTagMemory = 1U << 14U;

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

struct GstMessageAbi {
  GstMiniObjectAbi mini_object;
  std::uint32_t type = 0;
};

static_assert(offsetof(GstBufferAbi, pts) == (sizeof(void*) == 8 ? 72 : 40));
static_assert(sizeof(GstBufferAbi) == (sizeof(void*) == 8 ? 112 : 80));
static_assert(offsetof(GstMessageAbi, type) == (sizeof(void*) == 8 ? 64 : 36));

class DynamicLibrary {
public:
  explicit DynamicLibrary(const std::filesystem::path& path) : path_(core::path_to_utf8(path)) {
#if defined(_WIN32)
    handle_ = static_cast<HMODULE>(core::detail::load_windows_runtime_library(path));
    if (handle_ == nullptr) {
      throw GpuEncodeError("failed to load " + path_ + " (Windows error " +
                           std::to_string(GetLastError()) + ")");
    }
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw GpuEncodeError("failed to load " + path_ +
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
    (void)FreeLibrary(handle_);
#else
    (void)dlclose(handle_);
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
      throw GpuEncodeError("missing GStreamer runtime symbol " + std::string(name) + " in " +
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

std::shared_ptr<DynamicLibrary> load_runtime_library(const char* environment_variable,
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
    } catch (const GpuEncodeError& error) {
      if (!errors.empty()) {
        errors += "; ";
      }
      errors += error.what();
    }
  }
  throw GpuEncodeError("could not load " + std::string(component) + " runtime: " + errors);
}

class GstreamerEncodeApi {
public:
  using InitCheck = int (*)(int*, char***, GErrorAbi**);
  using Version = void (*)(std::uint32_t*, std::uint32_t*, std::uint32_t*, std::uint32_t*);
  using ParseLaunch = void* (*)(const char*, GErrorAbi**);
  using BinGetByName = void* (*)(void*, const char*);
  using ElementSetState = int (*)(void*, int);
  using ElementGetState = int (*)(void*, int*, int*, std::uint64_t);
  using ElementGetBus = void* (*)(void*);
  using ObjectUnref = void (*)(void*);
  using BufferNewWrappedFull = void* (*)(std::uint32_t, void*, std::size_t, std::size_t,
                                         std::size_t, void*, void (*)(void*));
  using AppSrcPushBuffer = int (*)(void*, void*);
  using AppSrcEndOfStream = int (*)(void*);
  using BusTimedPopFiltered = void* (*)(void*, std::uint64_t, std::uint32_t);
  using MessageParseError = void (*)(void*, GErrorAbi**, char**);
  using MessageUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);

  GstreamerEncodeApi() {
#if defined(_WIN32)
    core_library =
        load_runtime_library("RECO_GSTREAMER_DYLIB_PATH", {"gstreamer-1.0-0.dll"}, "GStreamer");
    app_library = load_runtime_library("RECO_GSTAPP_DYLIB_PATH", {"gstapp-1.0-0.dll"}, "GstApp");
    glib_library = load_runtime_library("RECO_GLIB_DYLIB_PATH",
                                        {"libglib-2.0-0.dll", "glib-2.0-0.dll"}, "GLib");
#elif defined(__APPLE__)
    core_library =
        load_runtime_library("RECO_GSTREAMER_DYLIB_PATH",
                             {"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib"}, "GStreamer");
    app_library = load_runtime_library("RECO_GSTAPP_DYLIB_PATH",
                                       {"libgstapp-1.0.0.dylib", "libgstapp-1.0.dylib"}, "GstApp");
    glib_library = load_runtime_library("RECO_GLIB_DYLIB_PATH",
                                        {"libglib-2.0.0.dylib", "libglib-2.0.dylib"}, "GLib");
#else
    core_library = load_runtime_library(
        "RECO_GSTREAMER_DYLIB_PATH", {"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"}, "GStreamer");
    app_library = load_runtime_library("RECO_GSTAPP_DYLIB_PATH",
                                       {"libgstapp-1.0.so.0", "libgstapp-1.0.so"}, "GstApp");
    glib_library = load_runtime_library("RECO_GLIB_DYLIB_PATH",
                                        {"libglib-2.0.so.0", "libglib-2.0.so"}, "GLib");
#endif
    init_check = core_library->symbol<InitCheck>("gst_init_check");
    version = core_library->symbol<Version>("gst_version");
    parse_launch = core_library->symbol<ParseLaunch>("gst_parse_launch");
    bin_get_by_name = core_library->symbol<BinGetByName>("gst_bin_get_by_name");
    element_set_state = core_library->symbol<ElementSetState>("gst_element_set_state");
    element_get_state = core_library->symbol<ElementGetState>("gst_element_get_state");
    element_get_bus = core_library->symbol<ElementGetBus>("gst_element_get_bus");
    object_unref = core_library->symbol<ObjectUnref>("gst_object_unref");
    buffer_new_wrapped_full =
        core_library->symbol<BufferNewWrappedFull>("gst_buffer_new_wrapped_full");
    bus_timed_pop_filtered =
        core_library->symbol<BusTimedPopFiltered>("gst_bus_timed_pop_filtered");
    message_parse_error = core_library->symbol<MessageParseError>("gst_message_parse_error");
    message_unref = core_library->symbol<MessageUnref>("gst_message_unref");
    app_src_push_buffer = app_library->symbol<AppSrcPushBuffer>("gst_app_src_push_buffer");
    app_src_end_of_stream = app_library->symbol<AppSrcEndOfStream>("gst_app_src_end_of_stream");
    error_free = glib_library->symbol<ErrorFree>("g_error_free");
    free = glib_library->symbol<Free>("g_free");
  }

  std::shared_ptr<DynamicLibrary> core_library;
  std::shared_ptr<DynamicLibrary> app_library;
  std::shared_ptr<DynamicLibrary> glib_library;
  InitCheck init_check = nullptr;
  Version version = nullptr;
  ParseLaunch parse_launch = nullptr;
  BinGetByName bin_get_by_name = nullptr;
  ElementSetState element_set_state = nullptr;
  ElementGetState element_get_state = nullptr;
  ElementGetBus element_get_bus = nullptr;
  ObjectUnref object_unref = nullptr;
  BufferNewWrappedFull buffer_new_wrapped_full = nullptr;
  AppSrcPushBuffer app_src_push_buffer = nullptr;
  AppSrcEndOfStream app_src_end_of_stream = nullptr;
  BusTimedPopFiltered bus_timed_pop_filtered = nullptr;
  MessageParseError message_parse_error = nullptr;
  MessageUnref message_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
};

class GstreamerDiscoverApi {
public:
  using InitCheck = int (*)(int*, char***, GErrorAbi**);
  using FilenameToUri = char* (*)(const char*, GErrorAbi**);
  using DiscovererNew = void* (*)(std::uint64_t, GErrorAbi**);
  using DiscovererDiscoverUri = void* (*)(void*, const char*, GErrorAbi**);
  using DiscovererInfoGetResult = int (*)(const void*);
  using DiscovererInfoGetVideoStreams = void* (*)(void*);
  using DiscovererStreamInfoListFree = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);
  using GObjectUnref = void (*)(void*);

  GstreamerDiscoverApi() {
#if defined(_WIN32)
    core = load_runtime_library("RECO_GSTREAMER_DYLIB_PATH", {"gstreamer-1.0-0.dll"}, "GStreamer");
    glib = load_runtime_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0-0.dll", "glib-2.0-0.dll"},
                                "GLib");
    pbutils = load_runtime_library("RECO_GSTPBUTILS_DYLIB_PATH", {"gstpbutils-1.0-0.dll"},
                                   "GStreamer PbUtils");
    gobject = load_runtime_library("RECO_GOBJECT_DYLIB_PATH",
                                   {"libgobject-2.0-0.dll", "gobject-2.0-0.dll"}, "GObject");
#elif defined(__APPLE__)
    core =
        load_runtime_library("RECO_GSTREAMER_DYLIB_PATH",
                             {"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib"}, "GStreamer");
    glib = load_runtime_library("RECO_GLIB_DYLIB_PATH",
                                {"libglib-2.0.0.dylib", "libglib-2.0.dylib"}, "GLib");
    pbutils = load_runtime_library("RECO_GSTPBUTILS_DYLIB_PATH",
                                   {"libgstpbutils-1.0.0.dylib", "libgstpbutils-1.0.dylib"},
                                   "GStreamer PbUtils");
    gobject = load_runtime_library("RECO_GOBJECT_DYLIB_PATH",
                                   {"libgobject-2.0.0.dylib", "libgobject-2.0.dylib"}, "GObject");
#else
    core = load_runtime_library("RECO_GSTREAMER_DYLIB_PATH",
                                {"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"}, "GStreamer");
    glib = load_runtime_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.so.0", "libglib-2.0.so"},
                                "GLib");
    pbutils = load_runtime_library("RECO_GSTPBUTILS_DYLIB_PATH",
                                   {"libgstpbutils-1.0.so.0", "libgstpbutils-1.0.so"},
                                   "GStreamer PbUtils");
    gobject = load_runtime_library("RECO_GOBJECT_DYLIB_PATH",
                                   {"libgobject-2.0.so.0", "libgobject-2.0.so"}, "GObject");
#endif
    init_check = core->symbol<InitCheck>("gst_init_check");
    filename_to_uri = core->symbol<FilenameToUri>("gst_filename_to_uri");
    discoverer_new = pbutils->symbol<DiscovererNew>("gst_discoverer_new");
    discoverer_discover_uri = pbutils->symbol<DiscovererDiscoverUri>("gst_discoverer_discover_uri");
    discoverer_info_get_result =
        pbutils->symbol<DiscovererInfoGetResult>("gst_discoverer_info_get_result");
    discoverer_info_get_video_streams =
        pbutils->symbol<DiscovererInfoGetVideoStreams>("gst_discoverer_info_get_video_streams");
    discoverer_stream_info_list_free =
        pbutils->symbol<DiscovererStreamInfoListFree>("gst_discoverer_stream_info_list_free");
    error_free = glib->symbol<ErrorFree>("g_error_free");
    free = glib->symbol<Free>("g_free");
    g_object_unref = gobject->symbol<GObjectUnref>("g_object_unref");
  }

  std::shared_ptr<DynamicLibrary> core;
  std::shared_ptr<DynamicLibrary> glib;
  std::shared_ptr<DynamicLibrary> pbutils;
  std::shared_ptr<DynamicLibrary> gobject;
  InitCheck init_check = nullptr;
  FilenameToUri filename_to_uri = nullptr;
  DiscovererNew discoverer_new = nullptr;
  DiscovererDiscoverUri discoverer_discover_uri = nullptr;
  DiscovererInfoGetResult discoverer_info_get_result = nullptr;
  DiscovererInfoGetVideoStreams discoverer_info_get_video_streams = nullptr;
  DiscovererStreamInfoListFree discoverer_stream_info_list_free = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
  GObjectUnref g_object_unref = nullptr;
};

template <typename Api>
std::string take_error(const std::shared_ptr<Api>& api, GErrorAbi*& error,
                       std::string_view fallback) {
  std::string message(fallback);
  if (error != nullptr) {
    if (error->message != nullptr && error->message[0] != '\0') {
      message = error->message;
    }
    api->error_free(error);
    error = nullptr;
  }
  return message;
}

enum class SlotStatus {
  Free,
  Acquired,
  InPipeline,
};

struct EncodeSlot {
  EncodeSlot(NvmmSurfaceAllocation allocation_value, NvmmCudaFrame mapping_value,
             core::CudaNv12FrameView view_value)
      : allocation(std::move(allocation_value)), mapping(std::move(mapping_value)),
        view(std::move(view_value)) {}

  NvmmSurfaceAllocation allocation;
  NvmmCudaFrame mapping;
  core::CudaNv12FrameView view;
  SlotStatus status = SlotStatus::Free;
};

struct EncodePoolState {
  std::mutex mutex;
  std::condition_variable available;
  std::vector<std::unique_ptr<EncodeSlot>> slots;
  std::shared_ptr<GpuEncodeTraceSink> trace;
  std::optional<std::string> sticky_error;
  bool accepting = true;

  void record_error(std::string message) noexcept {
    try {
      std::lock_guard lock(mutex);
      if (!sticky_error.has_value()) {
        sticky_error = std::move(message);
      }
      accepting = false;
    } catch (...) {
      accepting = false;
    }
    available.notify_all();
  }

  void release(std::size_t index, SlotStatus expected) noexcept {
    {
      std::lock_guard lock(mutex);
      if (index >= slots.size() || slots[index]->status != expected) {
        if (!sticky_error.has_value()) {
          sticky_error = "GPU encode surface ownership state is inconsistent";
        }
        accepting = false;
      } else {
        slots[index]->status = SlotStatus::Free;
      }
    }
    if (trace) {
      trace->surface_released();
    }
    available.notify_all();
  }
};

struct WrappedSurfaceOwner {
  std::shared_ptr<EncodePoolState> pool;
  std::shared_ptr<void> surface_owner;
  std::size_t slot_index = 0;
};

void release_wrapped_surface(void* raw_owner) {
  std::unique_ptr<WrappedSurfaceOwner> owner(static_cast<WrappedSurfaceOwner*>(raw_owner));
  owner->pool->release(owner->slot_index, SlotStatus::InPipeline);
}

struct AudioPacketPoolState {
  std::mutex mutex;
  std::condition_variable available;
  std::size_t in_flight = 0;
  std::size_t bytes_in_flight = 0;
};

struct WrappedAudioPacketOwner {
  std::shared_ptr<AudioPacketPoolState> pool;
  std::vector<std::byte> bytes;
};

void release_wrapped_audio_packet(void* raw_owner) {
  std::unique_ptr<WrappedAudioPacketOwner> owner(static_cast<WrappedAudioPacketOwner*>(raw_owner));
  {
    std::lock_guard lock(owner->pool->mutex);
    if (owner->pool->in_flight > 0) {
      --owner->pool->in_flight;
    }
    if (owner->bytes.size() <= owner->pool->bytes_in_flight) {
      owner->pool->bytes_in_flight -= owner->bytes.size();
    } else {
      owner->pool->bytes_in_flight = 0;
    }
  }
  owner->pool->available.notify_all();
}

core::CudaNv12FrameView make_nv12_view(const NvmmCudaFrame& mapping) {
  core::CudaPitchedPlaneView y_plane(mapping.y_validation, mapping.y_pitch, mapping.width,
                                     mapping.height);
  core::CudaPitchedPlaneView uv_plane(mapping.uv_validation, mapping.uv_pitch, mapping.width,
                                      mapping.height / 2U);
  return core::CudaNv12FrameView(std::move(y_plane), std::move(uv_plane), mapping.width,
                                 mapping.height, mapping.color_matrix, mapping.color_range);
}

std::uint64_t timeout_ns(std::chrono::milliseconds timeout) {
  return static_cast<std::uint64_t>(timeout.count()) * 1'000'000ULL;
}

} // namespace

struct GpuEncodeFrameLease::State {
  State(std::shared_ptr<EncodePoolState> pool_value, std::size_t slot_index_value)
      : pool(std::move(pool_value)), slot_index(slot_index_value) {}

  ~State() {
    if (acquired) {
      pool->release(slot_index, SlotStatus::Acquired);
    }
  }

  std::shared_ptr<EncodePoolState> pool;
  std::size_t slot_index = 0;
  bool acquired = true;
};

struct GpuVideoEncodeSession::Impl {
  Impl(GpuEncodeConfig config_value, std::shared_ptr<const NvbufSurfaceRuntime> runtime,
       std::shared_ptr<GpuEncodeTraceSink> trace)
      : config(std::move(config_value)), pipeline_text(build_gstreamer_gpu_encode_pipeline(config)),
        api(std::make_shared<GstreamerEncodeApi>()), pool(std::make_shared<EncodePoolState>()),
        audio_pool(std::make_shared<AudioPacketPoolState>()) {
    pool->trace = std::move(trace);
    GErrorAbi* error = nullptr;
    if (api->init_check(nullptr, nullptr, &error) == 0) {
      throw GpuEncodeError(take_error(api, error, "GStreamer initialization failed"));
    }
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t micro = 0;
    std::uint32_t nano = 0;
    api->version(&major, &minor, &micro, &nano);
    if (major != 1U) {
      throw GpuEncodeError("unsupported GStreamer runtime major version " + std::to_string(major));
    }

    for (std::uint32_t index = 0; index < config.pool_capacity; ++index) {
      auto allocation = allocate_nvmm_nv12_surface(
          config.width, config.height, config.device_ordinal, core::YuvColorMatrix::Bt709,
          core::YuvColorRange::Limited, runtime);
      auto mapping = map_nvmm_frame_to_cuda(allocation.frame, allocation.owner,
                                            core::CudaSpanAccess::ReadWrite);
      auto view = make_nv12_view(mapping);
      pool->slots.push_back(
          std::make_unique<EncodeSlot>(std::move(allocation), std::move(mapping), std::move(view)));
      if (pool->trace) {
        pool->trace->surface_allocated();
      }
    }

    error = nullptr;
    pipeline = api->parse_launch(pipeline_text.c_str(), &error);
    if (pipeline == nullptr || error != nullptr) {
      const auto detail = take_error(api, error, "failed to construct GPU encode pipeline");
      close_resources();
      throw GpuEncodeError(detail);
    }
    source = api->bin_get_by_name(pipeline, "source");
    if (config.audio_caps.has_value()) {
      audio_source = api->bin_get_by_name(pipeline, "audio_source");
    }
    bus = api->element_get_bus(pipeline);
    if (source == nullptr || bus == nullptr ||
        (config.audio_caps.has_value() && audio_source == nullptr)) {
      close_resources();
      throw GpuEncodeError("GPU encode pipeline is missing appsrc or bus resources");
    }
    if (api->element_set_state(pipeline, kGstStatePlaying) == kGstStateChangeFailure) {
      (void)api->element_set_state(pipeline, kGstStateNull);
      close_resources();
      throw GpuEncodeError("GStreamer GPU encode pipeline failed to enter PLAYING");
    }
    int current = 0;
    int pending = 0;
    const int startup =
        api->element_get_state(pipeline, &current, &pending, timeout_ns(config.startup_timeout));
    if (startup != kGstStateChangeSuccess || current != kGstStatePlaying || pending != 0) {
      (void)api->element_set_state(pipeline, kGstStateNull);
      close_resources();
      if (startup == kGstStateChangeFailure) {
        throw GpuEncodeError("GStreamer GPU encode pipeline failed during startup");
      }
      throw GpuEncodeError("GStreamer GPU encode pipeline timed out before reaching PLAYING");
    }
  }

  ~Impl() {
    abort();
    close_resources();
  }

  void close_resources() noexcept {
    if (audio_source != nullptr) {
      api->object_unref(audio_source);
      audio_source = nullptr;
    }
    if (source != nullptr) {
      api->object_unref(source);
      source = nullptr;
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

  std::optional<std::string> poll_bus(std::uint64_t wait_ns, bool eos_is_expected) {
    if (bus == nullptr) {
      return std::string("GPU encode pipeline is closed");
    }
    void* message = api->bus_timed_pop_filtered(bus, wait_ns, kGstMessageEos | kGstMessageError);
    if (message == nullptr) {
      return std::nullopt;
    }
    const std::unique_ptr<void, GstreamerEncodeApi::MessageUnref> owner(message,
                                                                        api->message_unref);
    const auto type = static_cast<GstMessageAbi*>(message)->type;
    if ((type & kGstMessageError) != 0U) {
      GErrorAbi* error = nullptr;
      char* debug = nullptr;
      api->message_parse_error(message, &error, &debug);
      const auto detail = take_error(api, error, "GStreamer GPU encode pipeline failed");
      if (debug != nullptr) {
        api->free(debug);
      }
      pool->record_error(detail);
      return detail;
    }
    if ((type & kGstMessageEos) != 0U) {
      if (eos_is_expected) {
        return std::string{};
      }
      const std::string detail = "GStreamer GPU encode pipeline reached EOS before finalization";
      pool->record_error(detail);
      return detail;
    }
    return std::nullopt;
  }

  GpuEncodeFrameLease acquire() {
    const auto deadline = std::chrono::steady_clock::now() + config.acquire_timeout;
    while (true) {
      std::unique_lock submission_lock(submission_mutex);
      if (const auto bus_result = poll_bus(0, false);
          bus_result.has_value() && !bus_result->empty()) {
        throw GpuEncodeError(*bus_result);
      }
      std::unique_lock pool_lock(pool->mutex);
      if (pool->sticky_error.has_value()) {
        throw GpuEncodeError(*pool->sticky_error);
      }
      if (!pool->accepting || finished || aborted) {
        throw GpuEncodeError("GPU encode session is no longer accepting frames");
      }
      for (std::size_t index = 0; index < pool->slots.size(); ++index) {
        if (pool->slots[index]->status == SlotStatus::Free) {
          pool->slots[index]->status = SlotStatus::Acquired;
          pool_lock.unlock();
          submission_lock.unlock();
          if (pool->trace) {
            pool->trace->surface_acquired();
          }
          return GpuEncodeFrameLease(std::make_unique<GpuEncodeFrameLease::State>(pool, index));
        }
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        throw GpuEncodeError("timed out waiting for a free GPU encode surface");
      }
      submission_lock.unlock();
      pool->available.wait_until(pool_lock,
                                 std::min(deadline, now + std::chrono::milliseconds(100)));
    }
  }

  void submit(GpuEncodeFrameLease&& lease, std::uint64_t pts_ns, std::uint64_t duration_ns) {
    std::lock_guard submission_lock(submission_mutex);
    if (!lease.state_ || !lease.state_->acquired || lease.state_->pool.get() != pool.get()) {
      throw GpuEncodeError("GPU encode frame lease does not belong to this session");
    }
    if (duration_ns == 0 || pts_ns == kGstClockTimeNone || duration_ns == kGstClockTimeNone) {
      throw GpuEncodeError("GPU encode timestamps must be finite and duration must be non-zero");
    }
    if (pts_ns > std::numeric_limits<std::uint64_t>::max() - duration_ns) {
      throw GpuEncodeError("GPU encode frame timestamp overflows");
    }
    if (const auto bus_result = poll_bus(0, false);
        bus_result.has_value() && !bus_result->empty()) {
      throw GpuEncodeError(*bus_result);
    }

    const auto index = lease.state_->slot_index;
    EncodeSlot* slot = nullptr;
    {
      std::lock_guard lock(pool->mutex);
      if (pool->sticky_error.has_value()) {
        throw GpuEncodeError(*pool->sticky_error);
      }
      if (!pool->accepting || finished || aborted || index >= pool->slots.size() ||
          pool->slots[index]->status != SlotStatus::Acquired) {
        throw GpuEncodeError("GPU encode frame cannot be submitted in the current state");
      }
      slot = pool->slots[index].get();
      slot->status = SlotStatus::InPipeline;
    }
    lease.state_->acquired = false;
    lease.state_.reset();

    std::unique_ptr<WrappedSurfaceOwner> wrapped_owner;
    void* buffer = nullptr;
    try {
      wrapped_owner = std::make_unique<WrappedSurfaceOwner>(
          WrappedSurfaceOwner{pool, slot->allocation.owner, index});
      buffer = api->buffer_new_wrapped_full(
          0, slot->allocation.frame.surface_ptr, slot->allocation.descriptor_size, 0,
          slot->allocation.descriptor_size, wrapped_owner.get(), release_wrapped_surface);
      if (buffer == nullptr) {
        throw GpuEncodeError("GStreamer failed to wrap an NVMM output surface");
      }
      wrapped_owner.release();
      auto* header = static_cast<GstBufferAbi*>(buffer);
      header->pts = pts_ns;
      header->dts = pts_ns;
      header->duration = duration_ns;
      if (pool->trace) {
        pool->trace->surface_submitted();
      }
      const int flow = api->app_src_push_buffer(source, buffer);
      if (flow != kGstFlowOk) {
        const std::string detail =
            "GStreamer appsrc rejected a GPU frame with flow status " + std::to_string(flow);
        pool->record_error(detail);
        throw GpuEncodeError(detail);
      }
    } catch (...) {
      if (buffer == nullptr) {
        pool->release(index, SlotStatus::InPipeline);
      }
      throw;
    }
  }

  void submit_audio(CompressedAudioPacket packet) {
    std::lock_guard submission_lock(submission_mutex);
    if (!config.audio_caps.has_value() || audio_source == nullptr) {
      throw GpuEncodeError("GPU encode session was not configured for audio passthrough");
    }
    if (packet.bytes.empty() || packet.bytes.size() > kMaximumCompressedAudioPacketBytes) {
      throw GpuEncodeError("compressed audio packet size is outside the supported bound");
    }
    if (!packet.pts_ns.has_value() && !packet.dts_ns.has_value()) {
      throw GpuEncodeError("compressed audio packet requires a presentation or decode timestamp");
    }
    if (packet.pts_ns == kGstClockTimeNone || packet.dts_ns == kGstClockTimeNone ||
        packet.duration_ns == kGstClockTimeNone) {
      throw GpuEncodeError("compressed audio packet timestamps must be finite");
    }

    const auto admitted_bytes = packet.bytes.size();
    const auto deadline = std::chrono::steady_clock::now() + config.acquire_timeout;
    for (;;) {
      if (const auto bus_result = poll_bus(0, false);
          bus_result.has_value() && !bus_result->empty()) {
        throw GpuEncodeError(*bus_result);
      }
      {
        std::lock_guard state_lock(pool->mutex);
        if (pool->sticky_error.has_value()) {
          throw GpuEncodeError(*pool->sticky_error);
        }
        if (!pool->accepting || finished || aborted) {
          throw GpuEncodeError("GPU encode session is no longer accepting audio packets");
        }
      }
      std::unique_lock audio_lock(audio_pool->mutex);
      if (audio_pool->in_flight < kMaximumAudioPacketsInFlight &&
          packet.bytes.size() <=
              kMaximumCompressedAudioBytesInFlight - audio_pool->bytes_in_flight) {
        ++audio_pool->in_flight;
        audio_pool->bytes_in_flight += packet.bytes.size();
        break;
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        throw GpuEncodeError("timed out waiting for compressed audio mux capacity");
      }
      audio_pool->available.wait_until(audio_lock,
                                       std::min(deadline, now + std::chrono::milliseconds(100)));
    }

    std::unique_ptr<WrappedAudioPacketOwner> wrapped_owner;
    void* buffer = nullptr;
    try {
      wrapped_owner = std::make_unique<WrappedAudioPacketOwner>(
          WrappedAudioPacketOwner{audio_pool, std::move(packet.bytes)});
      buffer = api->buffer_new_wrapped_full(
          0, wrapped_owner->bytes.data(), wrapped_owner->bytes.size(), 0,
          wrapped_owner->bytes.size(), wrapped_owner.get(), release_wrapped_audio_packet);
      if (buffer == nullptr) {
        throw GpuEncodeError("GStreamer failed to wrap a compressed audio packet");
      }
      wrapped_owner.release();
      auto* header = static_cast<GstBufferAbi*>(buffer);
      header->pts = packet.pts_ns.value_or(kGstClockTimeNone);
      header->dts = packet.dts_ns.value_or(kGstClockTimeNone);
      header->duration = packet.duration_ns == 0 ? kGstClockTimeNone : packet.duration_ns;
      header->mini_object.flags |=
          packet.flags & ~kGstMiniObjectFlagMask & ~kGstBufferFlagTagMemory;
      const int flow = api->app_src_push_buffer(audio_source, buffer);
      if (flow != kGstFlowOk) {
        const std::string detail =
            "GStreamer audio appsrc rejected a compressed packet with flow status " +
            std::to_string(flow);
        pool->record_error(detail);
        throw GpuEncodeError(detail);
      }
    } catch (...) {
      if (buffer == nullptr) {
        {
          std::lock_guard lock(audio_pool->mutex);
          if (audio_pool->in_flight > 0U) {
            --audio_pool->in_flight;
          }
          if (admitted_bytes <= audio_pool->bytes_in_flight) {
            audio_pool->bytes_in_flight -= admitted_bytes;
          } else {
            audio_pool->bytes_in_flight = 0U;
          }
        }
        audio_pool->available.notify_all();
      }
      throw;
    }
  }

  void finish_session() {
    std::lock_guard submission_lock(submission_mutex);
    {
      std::lock_guard lock(pool->mutex);
      if (finished) {
        return;
      }
      if (aborted) {
        throw GpuEncodeError("GPU encode session was aborted");
      }
      if (pool->sticky_error.has_value()) {
        throw GpuEncodeError(*pool->sticky_error);
      }
      for (const auto& slot : pool->slots) {
        if (slot->status == SlotStatus::Acquired) {
          throw GpuEncodeError("cannot finalize while a GPU encode frame lease is outstanding");
        }
      }
      pool->accepting = false;
    }
    if (audio_source != nullptr && api->app_src_end_of_stream(audio_source) != kGstFlowOk) {
      abort_session();
      throw GpuEncodeError("GStreamer audio appsrc rejected end-of-stream");
    }
    if (api->app_src_end_of_stream(source) != kGstFlowOk) {
      abort_session();
      throw GpuEncodeError("GStreamer appsrc rejected end-of-stream");
    }

    const auto deadline = std::chrono::steady_clock::now() + config.finalize_timeout;
    while (true) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        abort_session();
        throw GpuEncodeError("timed out while finalizing the GPU encoder and muxer");
      }
      {
        std::lock_guard lock(pool->mutex);
        if (aborted) {
          throw GpuEncodeError("GPU encode session was aborted");
        }
      }
      const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
      const auto poll_slice = std::min(
          remaining,
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(100)));
      const auto result = poll_bus(static_cast<std::uint64_t>(poll_slice.count()), true);
      if (!result.has_value()) {
        continue;
      }
      if (!result->empty()) {
        abort_session();
        throw GpuEncodeError(*result);
      }
      break;
    }
    if (api->element_set_state(pipeline, kGstStateNull) == kGstStateChangeFailure) {
      throw GpuEncodeError("GStreamer GPU encode pipeline failed to stop after finalization");
    }
    {
      std::lock_guard lock(pool->mutex);
      if (aborted) {
        throw GpuEncodeError("GPU encode session was aborted");
      }
      finished = true;
    }
    pool->available.notify_all();
    audio_pool->available.notify_all();
  }

  void abort() noexcept { abort_session(); }

  void abort_session() noexcept {
    bool should_stop_pipeline = false;
    {
      std::lock_guard lock(pool->mutex);
      if (aborted || finished) {
        return;
      }
      aborted = true;
      pool->accepting = false;
      should_stop_pipeline = true;
    }
    pool->available.notify_all();
    audio_pool->available.notify_all();
    if (should_stop_pipeline && pipeline != nullptr) {
      (void)api->element_set_state(pipeline, kGstStateNull);
    }
  }

  GpuEncodeConfig config;
  std::string pipeline_text;
  std::shared_ptr<GstreamerEncodeApi> api;
  std::shared_ptr<EncodePoolState> pool;
  std::shared_ptr<AudioPacketPoolState> audio_pool;
  // Serializes appsrc admission and bus polling with terminal EOS finalization.
  std::mutex submission_mutex;
  void* pipeline = nullptr;
  void* source = nullptr;
  void* audio_source = nullptr;
  void* bus = nullptr;
  bool finished = false;
  bool aborted = false;
};

GpuEncodeFrameLease::GpuEncodeFrameLease(std::unique_ptr<State> state) : state_(std::move(state)) {}

GpuEncodeFrameLease::GpuEncodeFrameLease(GpuEncodeFrameLease&&) noexcept = default;

GpuEncodeFrameLease& GpuEncodeFrameLease::operator=(GpuEncodeFrameLease&&) noexcept = default;

GpuEncodeFrameLease::~GpuEncodeFrameLease() = default;

const core::CudaNv12FrameView& GpuEncodeFrameLease::view() const {
  if (!state_ || !state_->acquired) {
    throw GpuEncodeError("GPU encode frame lease is empty");
  }
  return state_->pool->slots.at(state_->slot_index)->view;
}

GpuEncodeFrameLease::operator bool() const noexcept { return state_ && state_->acquired; }

GpuVideoEncodeSession GpuVideoEncodeSession::open(GpuEncodeConfig config,
                                                  std::shared_ptr<GpuEncodeTraceSink> trace_sink) {
  return open(std::move(config), discover_nvbufsurface_runtime(), std::move(trace_sink));
}

GpuVideoEncodeSession
GpuVideoEncodeSession::open(GpuEncodeConfig config,
                            std::shared_ptr<const NvbufSurfaceRuntime> runtime,
                            std::shared_ptr<GpuEncodeTraceSink> trace_sink) {
  if (const auto error = validate_gpu_encode_config(config); error.has_value()) {
    throw GpuEncodeError(*error);
  }
  return GpuVideoEncodeSession(
      std::make_unique<Impl>(std::move(config), std::move(runtime), std::move(trace_sink)));
}

GpuVideoEncodeSession::GpuVideoEncodeSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GpuVideoEncodeSession::GpuVideoEncodeSession(GpuVideoEncodeSession&&) noexcept = default;

GpuVideoEncodeSession& GpuVideoEncodeSession::operator=(GpuVideoEncodeSession&&) noexcept = default;

GpuVideoEncodeSession::~GpuVideoEncodeSession() = default;

GpuEncodeFrameLease GpuVideoEncodeSession::acquire_frame() {
  if (!impl_) {
    throw GpuEncodeError("GPU encode session is empty");
  }
  return impl_->acquire();
}

void GpuVideoEncodeSession::submit_frame(GpuEncodeFrameLease&& frame, std::uint64_t pts_ns,
                                         std::uint64_t duration_ns) {
  if (!impl_) {
    throw GpuEncodeError("GPU encode session is empty");
  }
  impl_->submit(std::move(frame), pts_ns, duration_ns);
}

void GpuVideoEncodeSession::submit_audio_packet(CompressedAudioPacket packet) {
  if (!impl_) {
    throw GpuEncodeError("GPU encode session is empty");
  }
  impl_->submit_audio(std::move(packet));
}

void GpuVideoEncodeSession::finish() {
  if (!impl_) {
    throw GpuEncodeError("GPU encode session is empty");
  }
  impl_->finish_session();
}

void GpuVideoEncodeSession::abort() noexcept {
  if (impl_) {
    impl_->abort();
  }
}

const GpuEncodeConfig& GpuVideoEncodeSession::config() const {
  if (!impl_) {
    throw GpuEncodeError("GPU encode session is empty");
  }
  return impl_->config;
}

std::string_view GpuVideoEncodeSession::pipeline() const {
  if (!impl_) {
    return {};
  }
  return impl_->pipeline_text;
}

void verify_muxed_gpu_video_output(std::string_view path, std::chrono::milliseconds timeout) {
  if (path.empty() || path.find('\0') != std::string_view::npos) {
    throw std::invalid_argument("GPU output verification requires a non-empty path without NUL");
  }
  if (timeout < std::chrono::milliseconds(1) || timeout > std::chrono::hours(1)) {
    throw std::invalid_argument("GPU output verification timeout is outside the supported bound");
  }

  auto api = std::make_shared<GstreamerDiscoverApi>();
  GErrorAbi* error = nullptr;
  if (api->init_check(nullptr, nullptr, &error) == 0) {
    throw GpuEncodeError(take_error(api, error, "GStreamer initialization failed"));
  }
  const std::string owned_path(path);
  char* uri = api->filename_to_uri(owned_path.c_str(), &error);
  if (uri == nullptr) {
    throw GpuEncodeError(take_error(api, error, "failed to create encoded output URI"));
  }
  const std::unique_ptr<void, GstreamerDiscoverApi::Free> uri_owner(uri, api->free);
  const auto timeout_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count();
  void* discoverer = api->discoverer_new(static_cast<std::uint64_t>(timeout_ns), &error);
  if (discoverer == nullptr) {
    throw GpuEncodeError(take_error(api, error, "failed to create encoded output discoverer"));
  }
  const std::unique_ptr<void, GstreamerDiscoverApi::GObjectUnref> discoverer_owner(
      discoverer, api->g_object_unref);
  void* info = api->discoverer_discover_uri(discoverer, static_cast<const char*>(uri), &error);
  if (info == nullptr) {
    throw GpuEncodeError(take_error(api, error, "failed to inspect encoded output streams"));
  }
  const std::unique_ptr<void, GstreamerDiscoverApi::GObjectUnref> info_owner(info,
                                                                             api->g_object_unref);
  constexpr int kDiscovererOk = 0;
  constexpr int kDiscovererMissingPlugins = 5;
  const int result = api->discoverer_info_get_result(info);
  if (result != kDiscovererOk && result != kDiscovererMissingPlugins) {
    throw GpuEncodeError(take_error(api, error, "encoded output stream discovery failed"));
  }
  if (error != nullptr) {
    api->error_free(error);
    error = nullptr;
  }
  void* streams = api->discoverer_info_get_video_streams(info);
  if (streams == nullptr) {
    throw GpuEncodeError("completed GPU output contains no video stream");
  }
  api->discoverer_stream_info_list_free(streams);
}

} // namespace reco::io
