#include "reco/io/gpu_decode.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
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

constexpr std::uint32_t kGstMapRead = 1;
constexpr int kGstStateNull = 1;
constexpr int kGstStatePaused = 3;
constexpr int kGstStatePlaying = 4;
constexpr int kGstStateChangeFailure = 0;
constexpr int kGstFormatTime = 3;
constexpr int kGstSeekFlagFlushAccurate = (1 << 0) | (1 << 1);
constexpr std::uint32_t kGstMessageError = 1U << 1U;
constexpr std::uint32_t kGstMessageTag = 1U << 4U;
constexpr int kGstPadProbeTypeBuffer = 1 << 4;
constexpr int kGstPadProbeOk = 1;
constexpr std::uint64_t kGstClockTimeNone = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kSamplePollTimeoutNs = 100'000'000;
constexpr std::size_t kMaximumGeometryObservations = 4096;
// Codec block alignment can pad a surface, but not by an entire 256-pixel block.
constexpr std::uint32_t kMaximumVisibleAllocationPadding = 255;

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

struct GstPadProbeInfoAbi {
  std::int32_t type = 0;
  unsigned long id = 0;
  void* data = nullptr;
};

struct GstMessageAbi {
  GstMiniObjectAbi mini_object;
  std::uint32_t type = 0;
};

// GstBuffer and GstMapInfo are public GStreamer 1.x ABI structs. The runtime
// major-version check below prevents these layouts from being used with a
// future incompatible ABI.
static_assert(offsetof(GstBufferAbi, pts) == (sizeof(void*) == 8 ? 72 : 40));
static_assert(sizeof(GstBufferAbi) == (sizeof(void*) == 8 ? 112 : 80));
static_assert(offsetof(GstMapInfoAbi, data) == (sizeof(void*) == 8 ? 16 : 8));
static_assert(offsetof(GstMapInfoAbi, size) == (sizeof(void*) == 8 ? 24 : 12));
static_assert(sizeof(GstMapInfoAbi) == (sizeof(void*) == 8 ? 104 : 52));
static_assert(offsetof(GstPadProbeInfoAbi, data) ==
              (sizeof(void*) == 8 && sizeof(unsigned long) == 8 ? 16 : 8));
static_assert(offsetof(GstMessageAbi, type) == (sizeof(void*) == 8 ? 64 : 36));

class DynamicLibrary {
public:
  explicit DynamicLibrary(std::string path) : path_(std::move(path)) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(path_.c_str());
    if (handle_ == nullptr) {
      throw GpuDecodeError("failed to load " + path_ + " (Windows error " +
                           std::to_string(GetLastError()) + ")");
    }
#else
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw GpuDecodeError("failed to load " + path_ +
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

  template <typename Function> Function symbol(const char* name) const {
#if defined(_WIN32)
    auto* value = GetProcAddress(handle_, name);
#else
    dlerror();
    void* value = dlsym(handle_, name);
#endif
    if (value == nullptr) {
      throw GpuDecodeError("missing GStreamer runtime symbol " + std::string(name) + " in " +
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
  if (const char* override_path = std::getenv(environment_variable);
      override_path != nullptr && override_path[0] != '\0') {
    return std::make_shared<DynamicLibrary>(override_path);
  }

  std::string errors;
  for (const char* name : names) {
    try {
      return std::make_shared<DynamicLibrary>(name);
    } catch (const GpuDecodeError& error) {
      if (!errors.empty()) {
        errors += "; ";
      }
      errors += error.what();
    }
  }
  throw GpuDecodeError("could not load " + std::string(component) + " runtime: " + errors);
}

class GstreamerApi {
public:
  using InitCheck = int (*)(int*, char***, GErrorAbi**);
  using Version = void (*)(std::uint32_t*, std::uint32_t*, std::uint32_t*, std::uint32_t*);
  using ParseLaunch = void* (*)(const char*, GErrorAbi**);
  using BinGetByName = void* (*)(void*, const char*);
  using ElementGetStaticPad = void* (*)(void*, const char*);
  using ElementSetState = int (*)(void*, int);
  using ElementGetState = int (*)(void*, int*, int*, std::uint64_t);
  using ElementSeekSimple = int (*)(void*, int, int, std::int64_t);
  using ElementGetBus = void* (*)(void*);
  using ObjectUnref = void (*)(void*);
  using PadGetCurrentCaps = void* (*)(void*);
  using PadProbeCallback = int (*)(void*, void*, void*);
  using DestroyNotify = void (*)(void*);
  using PadAddProbe = unsigned long (*)(void*, int, PadProbeCallback, void*, DestroyNotify);
  using PadRemoveProbe = void (*)(void*, unsigned long);
  using MiniObjectSetQdata = void (*)(void*, std::uint32_t, void*, DestroyNotify);
  using MiniObjectGetQdata = void* (*)(void*, std::uint32_t);
  using CapsUnref = void (*)(void*);
  using AppSinkTryPullSample = void* (*)(void*, std::uint64_t);
  using AppSinkIsEos = int (*)(void*);
  using SampleGetBuffer = void* (*)(void*);
  using SampleGetSegment = const void* (*)(void*);
  using SegmentToStreamTime = std::uint64_t (*)(const void*, int, std::uint64_t);
  using SampleUnref = void (*)(void*);
  using CapsGetStructure = void* (*)(const void*, std::uint32_t);
  using StructureGetInt = int (*)(const void*, const char*, int*);
  using BufferMap = int (*)(void*, GstMapInfoAbi*, std::uint32_t);
  using BufferUnmap = void (*)(void*, GstMapInfoAbi*);
  using BusTimedPopFiltered = void* (*)(void*, std::uint64_t, std::uint32_t);
  using MessageParseError = void (*)(void*, GErrorAbi**, char**);
  using MessageParseTag = void (*)(void*, void**);
  using MessageUnref = void (*)(void*);
  using TagListGetString = int (*)(const void*, const char*, char**);
  using TagListUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);
  using QuarkFromStaticString = std::uint32_t (*)(const char*);

  GstreamerApi() {
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
    element_get_static_pad =
        core_library->symbol<ElementGetStaticPad>("gst_element_get_static_pad");
    element_set_state = core_library->symbol<ElementSetState>("gst_element_set_state");
    element_get_state = core_library->symbol<ElementGetState>("gst_element_get_state");
    element_seek_simple = core_library->symbol<ElementSeekSimple>("gst_element_seek_simple");
    element_get_bus = core_library->symbol<ElementGetBus>("gst_element_get_bus");
    object_unref = core_library->symbol<ObjectUnref>("gst_object_unref");
    pad_get_current_caps = core_library->symbol<PadGetCurrentCaps>("gst_pad_get_current_caps");
    pad_add_probe = core_library->symbol<PadAddProbe>("gst_pad_add_probe");
    pad_remove_probe = core_library->symbol<PadRemoveProbe>("gst_pad_remove_probe");
    mini_object_set_qdata = core_library->symbol<MiniObjectSetQdata>("gst_mini_object_set_qdata");
    mini_object_get_qdata = core_library->symbol<MiniObjectGetQdata>("gst_mini_object_get_qdata");
    caps_unref = core_library->symbol<CapsUnref>("gst_caps_unref");
    sample_get_buffer = core_library->symbol<SampleGetBuffer>("gst_sample_get_buffer");
    sample_get_segment = core_library->symbol<SampleGetSegment>("gst_sample_get_segment");
    segment_to_stream_time =
        core_library->symbol<SegmentToStreamTime>("gst_segment_to_stream_time");
    sample_unref = core_library->symbol<SampleUnref>("gst_sample_unref");
    caps_get_structure = core_library->symbol<CapsGetStructure>("gst_caps_get_structure");
    structure_get_int = core_library->symbol<StructureGetInt>("gst_structure_get_int");
    buffer_map = core_library->symbol<BufferMap>("gst_buffer_map");
    buffer_unmap = core_library->symbol<BufferUnmap>("gst_buffer_unmap");
    bus_timed_pop_filtered =
        core_library->symbol<BusTimedPopFiltered>("gst_bus_timed_pop_filtered");
    message_parse_error = core_library->symbol<MessageParseError>("gst_message_parse_error");
    message_parse_tag = core_library->symbol<MessageParseTag>("gst_message_parse_tag");
    message_unref = core_library->symbol<MessageUnref>("gst_message_unref");
    tag_list_get_string = core_library->symbol<TagListGetString>("gst_tag_list_get_string");
    tag_list_unref = core_library->symbol<TagListUnref>("gst_tag_list_unref");
    app_sink_try_pull_sample =
        app_library->symbol<AppSinkTryPullSample>("gst_app_sink_try_pull_sample");
    app_sink_is_eos = app_library->symbol<AppSinkIsEos>("gst_app_sink_is_eos");
    error_free = glib_library->symbol<ErrorFree>("g_error_free");
    free = glib_library->symbol<Free>("g_free");
    quark_from_static_string =
        glib_library->symbol<QuarkFromStaticString>("g_quark_from_static_string");
    geometry_metadata_quark = quark_from_static_string("reco-gpu-visible-geometry");
    if (geometry_metadata_quark == 0) {
      throw GpuDecodeError("failed to create GStreamer geometry metadata key");
    }
  }

  std::shared_ptr<DynamicLibrary> core_library;
  std::shared_ptr<DynamicLibrary> app_library;
  std::shared_ptr<DynamicLibrary> glib_library;
  InitCheck init_check = nullptr;
  Version version = nullptr;
  ParseLaunch parse_launch = nullptr;
  BinGetByName bin_get_by_name = nullptr;
  ElementGetStaticPad element_get_static_pad = nullptr;
  ElementSetState element_set_state = nullptr;
  ElementGetState element_get_state = nullptr;
  ElementSeekSimple element_seek_simple = nullptr;
  ElementGetBus element_get_bus = nullptr;
  ObjectUnref object_unref = nullptr;
  PadGetCurrentCaps pad_get_current_caps = nullptr;
  PadAddProbe pad_add_probe = nullptr;
  PadRemoveProbe pad_remove_probe = nullptr;
  MiniObjectSetQdata mini_object_set_qdata = nullptr;
  MiniObjectGetQdata mini_object_get_qdata = nullptr;
  CapsUnref caps_unref = nullptr;
  AppSinkTryPullSample app_sink_try_pull_sample = nullptr;
  AppSinkIsEos app_sink_is_eos = nullptr;
  SampleGetBuffer sample_get_buffer = nullptr;
  SampleGetSegment sample_get_segment = nullptr;
  SegmentToStreamTime segment_to_stream_time = nullptr;
  SampleUnref sample_unref = nullptr;
  CapsGetStructure caps_get_structure = nullptr;
  StructureGetInt structure_get_int = nullptr;
  BufferMap buffer_map = nullptr;
  BufferUnmap buffer_unmap = nullptr;
  BusTimedPopFiltered bus_timed_pop_filtered = nullptr;
  MessageParseError message_parse_error = nullptr;
  MessageParseTag message_parse_tag = nullptr;
  MessageUnref message_unref = nullptr;
  TagListGetString tag_list_get_string = nullptr;
  TagListUnref tag_list_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
  QuarkFromStaticString quark_from_static_string = nullptr;
  std::uint32_t geometry_metadata_quark = 0;
};

std::string take_error(const std::shared_ptr<GstreamerApi>& api, GErrorAbi*& error,
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

std::pair<std::uint32_t, std::uint32_t> visible_dimensions(const std::shared_ptr<GstreamerApi>& api,
                                                           void* predecoder_pad) {
  void* caps = api->pad_get_current_caps(predecoder_pad);
  if (caps == nullptr) {
    throw GpuDecodeError("GStreamer pre-decoder pad does not contain negotiated caps");
  }
  const std::unique_ptr<void, GstreamerApi::CapsUnref> caps_owner(caps, api->caps_unref);
  void* structure = api->caps_get_structure(caps, 0);
  if (structure == nullptr) {
    throw GpuDecodeError("GStreamer pre-decoder caps do not contain a structure");
  }
  int width = 0;
  int height = 0;
  if (api->structure_get_int(structure, "width", &width) == 0 ||
      api->structure_get_int(structure, "height", &height) == 0 || width <= 0 || height <= 0) {
    throw GpuDecodeError("GStreamer pre-decoder caps do not contain valid visible dimensions");
  }
  return {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

class GstSampleOwner {
public:
  GstSampleOwner(std::shared_ptr<GstreamerApi> api, void* sample)
      : api_(std::move(api)), sample_(sample) {}

  GstSampleOwner(const GstSampleOwner&) = delete;
  GstSampleOwner& operator=(const GstSampleOwner&) = delete;

  ~GstSampleOwner() {
    if (mapped_) {
      api_->buffer_unmap(buffer_, &map_);
    }
    if (sample_ != nullptr) {
      api_->sample_unref(sample_);
    }
  }

  void map_buffer(void* buffer) {
    buffer_ = buffer;
    if (api_->buffer_map(buffer_, &map_, kGstMapRead) == 0) {
      buffer_ = nullptr;
      throw GpuDecodeError("failed to map GStreamer NVMM buffer");
    }
    mapped_ = true;
    if (map_.data == nullptr) {
      throw GpuDecodeError("GStreamer NVMM buffer map returned null data");
    }
  }

  [[nodiscard]] const void* mapped_data() const { return map_.data; }

private:
  std::shared_ptr<GstreamerApi> api_;
  void* sample_ = nullptr;
  void* buffer_ = nullptr;
  GstMapInfoAbi map_{};
  bool mapped_ = false;
};

enum class GeometryProbeFailure {
  None,
  MissingBuffer,
  MissingCaps,
  MissingCapsStructure,
  InvalidDimensions,
  QueueLimit,
  Unknown,
};

using GeometryDimensions = std::pair<std::uint32_t, std::uint32_t>;

enum class GeometryMetadataFailure {
  None,
  MissingCorrelation,
  UnknownTimestampAcrossChange,
  AmbiguousTimestamp,
};

struct GeometryMetadata {
  std::vector<GeometryDimensions> candidates;
  GeometryMetadataFailure failure = GeometryMetadataFailure::None;
};

struct GeometryObservation {
  std::optional<std::uint64_t> pts_ns;
  GeometryDimensions dimensions;
};

struct GeometryAmbiguityGroup {
  std::uint64_t pts_ns = 0;
};

class GeometryProbeState {
public:
  explicit GeometryProbeState(std::shared_ptr<GstreamerApi> api) : api_(std::move(api)) {}

  static int on_predecoder_buffer(void* pad, void* probe_info, void* user_data) noexcept {
    auto state = *static_cast<std::shared_ptr<GeometryProbeState>*>(user_data);
    try {
      state->record_predecoder_geometry(pad, probe_info);
    } catch (const std::exception& error) {
      state->record_error(error.what());
    } catch (...) {
      state->record_error(nullptr);
    }
    return kGstPadProbeOk;
  }

  static int on_output_buffer(void*, void* probe_info, void* user_data) noexcept {
    auto state = *static_cast<std::shared_ptr<GeometryProbeState>*>(user_data);
    try {
      state->attach_output_geometry(probe_info);
    } catch (const std::exception& error) {
      state->record_error(error.what());
    } catch (...) {
      state->record_error(nullptr);
    }
    return kGstPadProbeOk;
  }

  static void destroy_callback_data(void* user_data) noexcept {
    delete static_cast<std::shared_ptr<GeometryProbeState>*>(user_data);
  }

  static void destroy_metadata(void* metadata) noexcept {
    delete static_cast<GeometryMetadata*>(metadata);
  }

  void check_failure() const { throw_if_failed(); }

  [[nodiscard]] bool failed() const noexcept {
    return failure_.load(std::memory_order_acquire) != GeometryProbeFailure::None;
  }

  void reset_before_seek() {
    throw_if_failed();
    std::lock_guard lock(mutex_);
    observations_.clear();
    ambiguity_groups_.clear();
    unknown_ambiguity_active_ = false;
  }

  [[nodiscard]] GeometryDimensions
  take_dimensions(void* buffer, const GeometryDimensions& allocation_dimensions,
                  const std::optional<GeometryDimensions>& previous_allocation_dimensions,
                  const std::optional<GeometryDimensions>& previous_visible_dimensions) const {
    throw_if_failed();
    const auto* metadata = static_cast<const GeometryMetadata*>(
        api_->mini_object_get_qdata(buffer, api_->geometry_metadata_quark));
    if (metadata == nullptr) {
      throw GpuDecodeError("GStreamer decoded frame has no attached pre-decoder geometry");
    }
    switch (metadata->failure) {
    case GeometryMetadataFailure::None:
      break;
    case GeometryMetadataFailure::MissingCorrelation:
      throw GpuDecodeError(
          "GStreamer decoded frame timestamp has no correlated pre-decoder geometry");
    case GeometryMetadataFailure::UnknownTimestampAcrossChange:
      throw GpuDecodeError(
          "GStreamer decoded frame timestamp cannot be correlated across a geometry change");
    case GeometryMetadataFailure::AmbiguousTimestamp:
      throw GpuDecodeError(
          "GStreamer decoded frame timestamp is ambiguous at a geometry transition");
    }
    if (metadata->candidates.empty()) {
      throw GpuDecodeError("GStreamer decoded frame has no correlated pre-decoder geometry");
    }
    if (metadata->candidates.size() == 1) {
      return metadata->candidates.front();
    }

    const auto allocation_compatible = [&](const auto& dimensions) {
      return dimensions.first <= allocation_dimensions.first &&
             dimensions.second <= allocation_dimensions.second &&
             allocation_dimensions.first - dimensions.first <= kMaximumVisibleAllocationPadding &&
             allocation_dimensions.second - dimensions.second <= kMaximumVisibleAllocationPadding;
    };
    std::vector<GeometryDimensions> compatible_candidates;
    std::copy_if(metadata->candidates.begin(), metadata->candidates.end(),
                 std::back_inserter(compatible_candidates), allocation_compatible);
    if (compatible_candidates.size() == 1) {
      return compatible_candidates.front();
    }
    if (compatible_candidates.empty()) {
      throw GpuDecodeError(
          "GStreamer decoded frame allocation does not match timestamped pre-decoder geometry");
    }

    if (previous_allocation_dimensions.has_value() && previous_visible_dimensions.has_value() &&
        *previous_allocation_dimensions != allocation_dimensions) {
      std::optional<GeometryDimensions> changed_candidate;
      for (const auto& candidate : compatible_candidates) {
        if (candidate == *previous_visible_dimensions) {
          continue;
        }
        if (changed_candidate.has_value()) {
          changed_candidate.reset();
          break;
        }
        changed_candidate = candidate;
      }
      if (changed_candidate.has_value()) {
        return *changed_candidate;
      }
    }
    throw GpuDecodeError("GStreamer decoded frame timestamp is ambiguous at a geometry transition");
  }

private:
  static void add_candidate(std::vector<GeometryDimensions>& candidates,
                            const GeometryDimensions& dimensions) {
    if (std::find(candidates.begin(), candidates.end(), dimensions) == candidates.end()) {
      candidates.push_back(dimensions);
    }
  }

  void record_predecoder_geometry(void* pad, void* probe_info) {
    void* buffer = static_cast<GstPadProbeInfoAbi*>(probe_info)->data;
    if (buffer == nullptr) {
      throw GpuDecodeError("GStreamer pre-decoder probe did not contain a buffer");
    }
    const auto dimensions = visible_dimensions(api_, pad);
    const auto pts = static_cast<const GstBufferAbi*>(buffer)->pts;
    std::lock_guard lock(mutex_);
    if (observations_.size() >= kMaximumGeometryObservations) {
      throw GpuDecodeError("GStreamer pre-decoder geometry history exceeded its safety limit");
    }
    observations_.push_back(
        {.pts_ns = pts == kGstClockTimeNone ? std::nullopt : std::optional<std::uint64_t>(pts),
         .dimensions = dimensions});
  }

  void attach_output_geometry(void* probe_info) {
    void* buffer = static_cast<GstPadProbeInfoAbi*>(probe_info)->data;
    if (buffer == nullptr) {
      throw GpuDecodeError("GStreamer output probe did not contain a buffer");
    }
    const auto pts = static_cast<const GstBufferAbi*>(buffer)->pts;
    auto metadata = std::make_unique<GeometryMetadata>();
    {
      std::lock_guard lock(mutex_);
      if (pts == kGstClockTimeNone) {
        if (unknown_ambiguity_active_) {
          metadata->failure = GeometryMetadataFailure::UnknownTimestampAcrossChange;
        }
        for (const auto& observation : observations_) {
          if (!observation.pts_ns.has_value()) {
            add_candidate(metadata->candidates, observation.dimensions);
          }
        }
        const auto observation =
            std::find_if(observations_.begin(), observations_.end(),
                         [](const auto& value) { return !value.pts_ns.has_value(); });
        if (observation == observations_.end()) {
          metadata->failure = GeometryMetadataFailure::MissingCorrelation;
          unknown_ambiguity_active_ = false;
        } else {
          if (metadata->candidates.size() > 1) {
            metadata->failure = GeometryMetadataFailure::UnknownTimestampAcrossChange;
            unknown_ambiguity_active_ = true;
          }
          observations_.erase(observation);
          if (std::none_of(observations_.begin(), observations_.end(),
                           [](const auto& value) { return !value.pts_ns.has_value(); })) {
            unknown_ambiguity_active_ = false;
          }
        }
      } else {
        auto group = std::find_if(ambiguity_groups_.begin(), ambiguity_groups_.end(),
                                  [&](const auto& value) { return value.pts_ns == pts; });
        if (group != ambiguity_groups_.end()) {
          metadata->failure = GeometryMetadataFailure::AmbiguousTimestamp;
        }
        for (const auto& observation : observations_) {
          if (observation.pts_ns == pts) {
            add_candidate(metadata->candidates, observation.dimensions);
          }
        }
        const auto observation =
            std::find_if(observations_.begin(), observations_.end(),
                         [&](const auto& value) { return value.pts_ns == pts; });
        if (observation == observations_.end()) {
          metadata->failure = GeometryMetadataFailure::MissingCorrelation;
          if (group != ambiguity_groups_.end()) {
            ambiguity_groups_.erase(group);
          }
        } else {
          observations_.erase(observation);
          const bool same_pts_remain =
              std::any_of(observations_.begin(), observations_.end(),
                          [&](const auto& value) { return value.pts_ns == pts; });
          if (metadata->candidates.size() > 1 && same_pts_remain) {
            if (group == ambiguity_groups_.end()) {
              ambiguity_groups_.push_back({.pts_ns = pts});
            }
          } else if (group != ambiguity_groups_.end() && !same_pts_remain) {
            ambiguity_groups_.erase(group);
          }
        }
      }
    }
    api_->mini_object_set_qdata(buffer, api_->geometry_metadata_quark, metadata.release(),
                                &GeometryProbeState::destroy_metadata);
  }

  void throw_if_failed() const {
    switch (failure_.load(std::memory_order_acquire)) {
    case GeometryProbeFailure::None:
      return;
    case GeometryProbeFailure::MissingBuffer:
      throw GpuDecodeError("GStreamer geometry probe did not contain a buffer");
    case GeometryProbeFailure::MissingCaps:
      throw GpuDecodeError("GStreamer pre-decoder pad does not contain negotiated caps");
    case GeometryProbeFailure::MissingCapsStructure:
      throw GpuDecodeError("GStreamer pre-decoder caps do not contain a structure");
    case GeometryProbeFailure::InvalidDimensions:
      throw GpuDecodeError("GStreamer pre-decoder caps do not contain valid visible dimensions");
    case GeometryProbeFailure::QueueLimit:
      throw GpuDecodeError("GStreamer pre-decoder geometry history exceeded its safety limit");
    case GeometryProbeFailure::Unknown:
      throw GpuDecodeError("GStreamer geometry probe failed");
    }
    throw GpuDecodeError("GStreamer geometry probe failed");
  }

  void record_error(const char* message) noexcept {
    auto failure = GeometryProbeFailure::Unknown;
    if (message != nullptr && std::strstr(message, "did not contain a buffer") != nullptr) {
      failure = GeometryProbeFailure::MissingBuffer;
    } else if (message != nullptr &&
               std::strstr(message, "does not contain negotiated caps") != nullptr) {
      failure = GeometryProbeFailure::MissingCaps;
    } else if (message != nullptr &&
               std::strstr(message, "do not contain a structure") != nullptr) {
      failure = GeometryProbeFailure::MissingCapsStructure;
    } else if (message != nullptr && std::strstr(message, "valid visible dimensions") != nullptr) {
      failure = GeometryProbeFailure::InvalidDimensions;
    } else if (message != nullptr && std::strstr(message, "safety limit") != nullptr) {
      failure = GeometryProbeFailure::QueueLimit;
    }
    GeometryProbeFailure expected = GeometryProbeFailure::None;
    (void)failure_.compare_exchange_strong(expected, failure, std::memory_order_release,
                                           std::memory_order_relaxed);
  }

  std::shared_ptr<GstreamerApi> api_;
  std::mutex mutex_;
  std::deque<GeometryObservation> observations_;
  std::deque<GeometryAmbiguityGroup> ambiguity_groups_;
  bool unknown_ambiguity_active_ = false;
  std::atomic<GeometryProbeFailure> failure_{GeometryProbeFailure::None};
};

class GstreamerGpuFileDecodeSource final : public GpuFileDecodeSource {
public:
  GstreamerGpuFileDecodeSource(GpuFileDecodeConfig config, NvbufSurfaceAbi abi,
                               std::shared_ptr<GstreamerApi> api)
      : config_(std::move(config)),
        pipeline_description_(build_gstreamer_gpu_file_decode_pipeline(config_)), abi_(abi),
        api_(std::move(api)), next_frame_index_(config_.start_frame_index.value_or(0U)) {}

  GstreamerGpuFileDecodeSource(const GstreamerGpuFileDecodeSource&) = delete;
  GstreamerGpuFileDecodeSource& operator=(const GstreamerGpuFileDecodeSource&) = delete;

  ~GstreamerGpuFileDecodeSource() override { close(); }

  void start() {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t micro = 0;
    std::uint32_t nano = 0;
    api_->version(&major, &minor, &micro, &nano);
    if (major != 1 || minor < 10) {
      throw GpuDecodeError("GStreamer 1.10 or newer is required, found " + std::to_string(major) +
                           "." + std::to_string(minor));
    }

    GErrorAbi* error = nullptr;
    if (api_->init_check(nullptr, nullptr, &error) == 0) {
      throw GpuDecodeError("GStreamer initialization failed: " +
                           take_error(api_, error, "unknown initialization error"));
    }
    if (error != nullptr) {
      api_->error_free(error);
      error = nullptr;
    }

    pipeline_ = api_->parse_launch(pipeline_description_.c_str(), &error);
    if (pipeline_ == nullptr || error != nullptr) {
      throw GpuDecodeError("GStreamer pipeline parse failed: " +
                           take_error(api_, error, "parse returned no pipeline"));
    }
    sink_ = api_->bin_get_by_name(pipeline_, "sink");
    if (sink_ == nullptr) {
      throw GpuDecodeError("GStreamer pipeline does not contain appsink 'sink'");
    }
    display_info_ = api_->bin_get_by_name(pipeline_, "display_info");
    if (display_info_ == nullptr) {
      throw GpuDecodeError(
          "GStreamer pipeline does not contain pre-decoder identity 'display_info'");
    }
    display_info_pad_ = api_->element_get_static_pad(display_info_, "src");
    if (display_info_pad_ == nullptr) {
      throw GpuDecodeError("GStreamer pre-decoder identity does not provide a source pad");
    }
    geometry_probe_state_ = std::make_shared<GeometryProbeState>(api_);
    auto predecoder_callback_data =
        std::make_unique<std::shared_ptr<GeometryProbeState>>(geometry_probe_state_);
    display_info_probe_id_ = api_->pad_add_probe(
        display_info_pad_, kGstPadProbeTypeBuffer, &GeometryProbeState::on_predecoder_buffer,
        predecoder_callback_data.get(), &GeometryProbeState::destroy_callback_data);
    if (display_info_probe_id_ == 0) {
      geometry_probe_state_.reset();
      throw GpuDecodeError("failed to install GStreamer pre-decoder geometry probe");
    }
    (void)predecoder_callback_data.release();
    output_info_ = api_->bin_get_by_name(pipeline_, "output_info");
    if (output_info_ == nullptr) {
      throw GpuDecodeError("GStreamer pipeline does not contain output identity 'output_info'");
    }
    output_info_pad_ = api_->element_get_static_pad(output_info_, "src");
    if (output_info_pad_ == nullptr) {
      throw GpuDecodeError("GStreamer output identity does not provide a source pad");
    }
    auto output_callback_data =
        std::make_unique<std::shared_ptr<GeometryProbeState>>(geometry_probe_state_);
    output_info_probe_id_ = api_->pad_add_probe(
        output_info_pad_, kGstPadProbeTypeBuffer, &GeometryProbeState::on_output_buffer,
        output_callback_data.get(), &GeometryProbeState::destroy_callback_data);
    if (output_info_probe_id_ == 0) {
      throw GpuDecodeError("failed to install GStreamer output geometry probe");
    }
    (void)output_callback_data.release();
    bus_ = api_->element_get_bus(pipeline_);
    if (bus_ == nullptr) {
      throw GpuDecodeError("GStreamer pipeline does not provide a message bus");
    }
    if (config_.start_frame_index.has_value()) {
      seek_pipeline_to_frame(*config_.start_frame_index, true);
      return;
    }
    if (api_->element_set_state(pipeline_, kGstStatePlaying) == kGstStateChangeFailure) {
      throw GpuDecodeError("GStreamer pipeline rejected the PLAYING state");
    }
  }

  [[nodiscard]] const GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return pipeline_description_; }
  [[nodiscard]] bool gpu_resident() const override { return true; }

  void seek_to_frame(std::uint64_t frame_index) override {
    std::lock_guard lock(read_mutex_);
    if (terminal_error_.has_value()) {
      throw GpuDecodeError(*terminal_error_);
    }
    throw_if_geometry_probe_failed();
    drain_pipeline_messages();
    try {
      seek_pipeline_to_frame(frame_index, false);
    } catch (const GpuDecodeError& error) {
      terminal_error_ = error.what();
      throw GpuDecodeError(*terminal_error_);
    }
  }

  [[nodiscard]] GpuDecodeReadResult read() override {
    std::lock_guard lock(read_mutex_);
    if (terminal_error_.has_value()) {
      throw GpuDecodeError(*terminal_error_);
    }
    throw_if_geometry_probe_failed();
    if (ended_) {
      return make_gpu_decode_eos();
    }

    drain_pipeline_messages();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::nanoseconds(config_.read_timeout_ns);
    void* sample = nullptr;
    while (sample == nullptr) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        terminal_error_ = "GStreamer GPU decode read timed out";
        throw GpuDecodeError(*terminal_error_);
      }
      const auto remaining =
          std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count();
      const auto poll_timeout =
          std::min<std::uint64_t>(kSamplePollTimeoutNs, static_cast<std::uint64_t>(remaining));
      sample = api_->app_sink_try_pull_sample(sink_, poll_timeout);
      if (sample != nullptr) {
        break;
      }
      throw_if_geometry_probe_failed();
      drain_pipeline_messages();
      if (api_->app_sink_is_eos(sink_) != 0) {
        if (last_indexed_stream_time_ns_.has_value() &&
            timestamp_group_ordinal_ + 1U != config_.indexed_timestamp_multiplicity) {
          terminal_error_ = "indexed GStreamer GPU decode ended within a duplicate timestamp group";
          throw GpuDecodeError(*terminal_error_);
        }
        ended_ = true;
        return make_gpu_decode_eos();
      }
    }

    auto owner = std::make_shared<GstSampleOwner>(api_, sample);
    drain_pipeline_messages();
    throw_if_geometry_probe_failed();
    void* buffer = api_->sample_get_buffer(sample);
    if (buffer == nullptr) {
      throw GpuDecodeError("GStreamer sample does not contain a buffer");
    }
    owner->map_buffer(buffer);

    NvmmFrameInfo nvmm;
    try {
      nvmm = extract_nvmm_frame_info(owner->mapped_data(), abi_);
    } catch (const std::exception& error) {
      throw GpuDecodeError("invalid GStreamer NVMM frame: " + std::string(error.what()));
    }

    const auto& gst_buffer = *static_cast<const GstBufferAbi*>(buffer);
    const auto pts_ns = gst_buffer.pts == kGstClockTimeNone
                            ? std::nullopt
                            : std::optional<std::uint64_t>(gst_buffer.pts);
    const auto frame_index = index_for_sample(sample, pts_ns);
    const auto allocation_dimensions = std::pair(nvmm.width, nvmm.height);
    GeometryDimensions visible_dimensions;
    try {
      visible_dimensions = geometry_probe_state_->take_dimensions(buffer, allocation_dimensions,
                                                                  previous_allocation_dimensions_,
                                                                  previous_visible_dimensions_);
    } catch (const GpuDecodeError& error) {
      if (geometry_probe_state_->failed()) {
        terminal_error_ = error.what();
      }
      throw;
    }
    const auto [visible_width, visible_height] = visible_dimensions;
    GpuDecodedFrame frame{
        .nvmm = nvmm,
        .visible_width = visible_width,
        .visible_height = visible_height,
        .owner = owner,
        .frame_index = frame_index,
        .pts_ns = pts_ns,
        .duration_ns = gst_buffer.duration == kGstClockTimeNone
                           ? std::nullopt
                           : std::optional<std::uint64_t>(gst_buffer.duration),
        .rotation_degrees = rotation_degrees_,
    };
    if (const auto validation_error = validate_gpu_decoded_frame(frame);
        validation_error.has_value()) {
      throw GpuDecodeError("invalid GStreamer GPU frame: " + *validation_error);
    }
    if (nvmm.width - visible_width > kMaximumVisibleAllocationPadding ||
        nvmm.height - visible_height > kMaximumVisibleAllocationPadding) {
      throw GpuDecodeError(
          "GStreamer NVMM allocation dimensions are inconsistent with visible geometry");
    }
    const auto current_visible_dimensions = std::pair(visible_width, visible_height);
    if (previous_allocation_dimensions_.has_value() &&
        *previous_allocation_dimensions_ != allocation_dimensions &&
        previous_visible_dimensions_.has_value() &&
        *previous_visible_dimensions_ == current_visible_dimensions) {
      throw GpuDecodeError(
          "GStreamer NVMM allocation dimensions changed without a confirmed visible geometry "
          "change");
    }
    previous_allocation_dimensions_ = allocation_dimensions;
    previous_visible_dimensions_ = current_visible_dimensions;
    if (frame_index == std::numeric_limits<std::uint64_t>::max()) {
      throw GpuDecodeError("GStreamer GPU decode frame index overflow");
    }
    next_frame_index_ = frame_index + 1U;
    ++frames_emitted_;
    return make_gpu_decode_frame(std::move(frame));
  }

private:
  void seek_pipeline_to_frame(std::uint64_t frame_index, bool initial) {
    if (!config_.indexed_fps_numerator.has_value() ||
        !config_.indexed_stream_time_origin_ns.has_value()) {
      throw GpuDecodeError("GStreamer GPU decode indexed seek requires probed cadence and origin");
    }
    const auto multiplicity = static_cast<std::uint64_t>(config_.indexed_timestamp_multiplicity);
    const auto group_base_index = frame_index - frame_index % multiplicity;
    // An exact seek to a shared PTS may land on any member of that group. Seek
    // to the preceding empty cadence slot so every equal-PTS member is emitted.
    const auto seek_index =
        multiplicity > 1U && group_base_index > 0U ? group_base_index - 1U : group_base_index;
    const auto operation = initial ? "initial seek" : "indexed seek";
    if (api_->element_set_state(pipeline_, kGstStatePaused) == kGstStateChangeFailure) {
      throw GpuDecodeError("GStreamer pipeline rejected the PAUSED state for " +
                           std::string(operation));
    }
    wait_for_state(kGstStatePaused, "before " + std::string(operation));
    geometry_probe_state_->reset_before_seek();

    const long double target_ns = static_cast<long double>(*config_.indexed_stream_time_origin_ns) +
                                  static_cast<long double>(seek_index) * 1'000'000'000.0L *
                                      static_cast<long double>(*config_.indexed_fps_denominator) /
                                      static_cast<long double>(*config_.indexed_fps_numerator);
    if (!std::isfinite(target_ns) || target_ns < 0.0L ||
        target_ns > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
      throw GpuDecodeError("GPU decode " + std::string(operation) +
                           " target exceeds GStreamer time range");
    }
    const auto rounded_target_ns = static_cast<std::int64_t>(std::round(target_ns));
    if (api_->element_seek_simple(pipeline_, kGstFormatTime, kGstSeekFlagFlushAccurate,
                                  rounded_target_ns) == 0) {
      if (initial) {
        throw GpuDecodeError("GStreamer pipeline rejected the requested initial frame seek");
      }
      throw GpuDecodeError("GStreamer pipeline rejected the requested " + std::string(operation));
    }
    wait_for_state(kGstStatePaused, "after " + std::string(operation));

    next_frame_index_ = group_base_index;
    post_seek_expected_base_index_ = group_base_index;
    last_indexed_stream_time_ns_.reset();
    timestamp_group_ordinal_ = 0;
    ended_ = false;
    if (api_->element_set_state(pipeline_, kGstStatePlaying) == kGstStateChangeFailure) {
      throw GpuDecodeError("GStreamer pipeline rejected the PLAYING state after " +
                           std::string(operation));
    }
  }

  void wait_for_state(int expected, std::string_view operation) {
    int state = kGstStateNull;
    int pending = kGstStateNull;
    const auto result =
        api_->element_get_state(pipeline_, &state, &pending, config_.read_timeout_ns);
    if (result == kGstStateChangeFailure || state != expected) {
      throw GpuDecodeError("GStreamer pipeline did not reach the PAUSED state " +
                           std::string(operation));
    }
  }

  void throw_if_geometry_probe_failed() {
    try {
      geometry_probe_state_->check_failure();
    } catch (const GpuDecodeError& error) {
      terminal_error_ = error.what();
      throw GpuDecodeError(*terminal_error_);
    }
  }

  void drain_pipeline_messages() {
    while (void* message =
               api_->bus_timed_pop_filtered(bus_, 0, kGstMessageError | kGstMessageTag)) {
      const auto message_type = static_cast<const GstMessageAbi*>(message)->type;
      if (message_type == kGstMessageError) {
        GErrorAbi* error = nullptr;
        char* debug = nullptr;
        api_->message_parse_error(message, &error, &debug);
        std::string result =
            "GStreamer pipeline error: " + take_error(api_, error, "unknown streaming error");
        if (debug != nullptr && debug[0] != '\0') {
          result += " (" + std::string(debug) + ")";
        }
        if (debug != nullptr) {
          api_->free(debug);
        }
        api_->message_unref(message);
        terminal_error_ = std::move(result);
        throw GpuDecodeError(*terminal_error_);
      }
      if (message_type != kGstMessageTag) {
        api_->message_unref(message);
        terminal_error_ = "GStreamer bus returned an unexpected filtered message type";
        throw GpuDecodeError(*terminal_error_);
      }

      void* tags = nullptr;
      api_->message_parse_tag(message, &tags);
      api_->message_unref(message);
      if (tags == nullptr) {
        terminal_error_ = "GStreamer orientation tag message has no tag list";
        throw GpuDecodeError(*terminal_error_);
      }
      const std::unique_ptr<void, GstreamerApi::TagListUnref> tags_owner(tags,
                                                                         api_->tag_list_unref);
      char* orientation = nullptr;
      if (api_->tag_list_get_string(tags, "image-orientation", &orientation) == 0) {
        continue;
      }
      const std::unique_ptr<char, GstreamerApi::Free> orientation_owner(orientation, api_->free);
      if (orientation == nullptr) {
        terminal_error_ = "GStreamer image-orientation tag has no value";
        throw GpuDecodeError(*terminal_error_);
      }
      const std::string_view value(orientation);
      std::uint16_t parsed_rotation = 0;
      if (value == "rotate-0" || value == "normal") {
        parsed_rotation = 0;
      } else if (value == "rotate-90") {
        parsed_rotation = 90;
      } else if (value == "rotate-180") {
        parsed_rotation = 180;
      } else if (value == "rotate-270") {
        parsed_rotation = 270;
      } else {
        terminal_error_ = "unsupported GStreamer image orientation `" + std::string(value) + "`";
        throw GpuDecodeError(*terminal_error_);
      }
      if (frames_emitted_ != 0U && parsed_rotation != rotation_degrees_) {
        terminal_error_ = "GStreamer video orientation changed after decoding began";
        throw GpuDecodeError(*terminal_error_);
      }
      rotation_degrees_ = parsed_rotation;
    }
  }

  [[nodiscard]] std::uint64_t index_for_sample(void* sample,
                                               const std::optional<std::uint64_t>& pts_ns) {
    if (!config_.indexed_fps_numerator.has_value()) {
      return next_frame_index_;
    }
    if (!pts_ns.has_value()) {
      throw GpuDecodeError("indexed GStreamer GPU decode frame has no presentation timestamp");
    }
    const void* segment = api_->sample_get_segment(sample);
    if (segment == nullptr) {
      throw GpuDecodeError("indexed GStreamer GPU decode sample has no time segment");
    }
    const auto stream_time_ns = api_->segment_to_stream_time(segment, kGstFormatTime, *pts_ns);
    if (stream_time_ns == kGstClockTimeNone) {
      throw GpuDecodeError(
          "indexed GStreamer GPU decode PTS cannot be converted to presentation stream time");
    }
    if (!indexed_stream_time_origin_ns_.has_value()) {
      indexed_stream_time_origin_ns_ = stream_time_ns;
    }
    if (stream_time_ns < *indexed_stream_time_origin_ns_) {
      throw GpuDecodeError("indexed GStreamer GPU decode stream time moved before the origin");
    }
    const long double elapsed =
        static_cast<long double>(stream_time_ns - *indexed_stream_time_origin_ns_);
    const long double frames =
        elapsed * static_cast<long double>(*config_.indexed_fps_numerator) /
        (1'000'000'000.0L * static_cast<long double>(*config_.indexed_fps_denominator));
    const long double rounded = std::round(frames);
    const auto maximum_offset = static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
    if (!std::isfinite(frames) || std::abs(frames - rounded) > 0.125L || rounded < 0.0L ||
        rounded > maximum_offset) {
      throw GpuDecodeError("indexed GStreamer GPU decode stream time violates the probed cadence");
    }
    const auto group_base_index = static_cast<std::uint64_t>(rounded);
    const auto multiplicity = static_cast<std::uint64_t>(config_.indexed_timestamp_multiplicity);
    if (group_base_index % multiplicity != 0U) {
      throw GpuDecodeError(
          "indexed GStreamer GPU decode timestamp group violates the probed cadence");
    }

    if (last_indexed_stream_time_ns_.has_value()) {
      if (stream_time_ns < *last_indexed_stream_time_ns_) {
        throw GpuDecodeError("indexed GStreamer GPU decode stream time is not increasing");
      }
      if (stream_time_ns == *last_indexed_stream_time_ns_) {
        ++timestamp_group_ordinal_;
        if (timestamp_group_ordinal_ >= config_.indexed_timestamp_multiplicity) {
          throw GpuDecodeError(
              "indexed GStreamer GPU decode exceeds the probed timestamp multiplicity");
        }
      } else {
        if (timestamp_group_ordinal_ + 1U != config_.indexed_timestamp_multiplicity) {
          throw GpuDecodeError("indexed GStreamer GPU decode has an incomplete timestamp group");
        }
        timestamp_group_ordinal_ = 0;
      }
    } else {
      timestamp_group_ordinal_ = 0;
    }
    last_indexed_stream_time_ns_ = stream_time_ns;
    if (group_base_index > std::numeric_limits<std::uint64_t>::max() - timestamp_group_ordinal_) {
      throw GpuDecodeError("indexed GStreamer GPU decode frame index overflow");
    }
    const auto index = group_base_index + timestamp_group_ordinal_;
    if (post_seek_expected_base_index_.has_value() && index != *post_seek_expected_base_index_) {
      throw GpuDecodeError(
          "first post-seek GStreamer sample does not match the requested absolute frame index");
    }
    post_seek_expected_base_index_.reset();
    if (index < next_frame_index_) {
      throw GpuDecodeError("indexed GStreamer GPU decode frame indices are not increasing");
    }
    return index;
  }

  void close() noexcept {
    if (pipeline_ != nullptr) {
      (void)api_->element_set_state(pipeline_, kGstStateNull);
    }
    if (bus_ != nullptr) {
      api_->object_unref(bus_);
      bus_ = nullptr;
    }
    if (output_info_pad_ != nullptr) {
      if (output_info_probe_id_ != 0) {
        api_->pad_remove_probe(output_info_pad_, output_info_probe_id_);
        output_info_probe_id_ = 0;
      }
      api_->object_unref(output_info_pad_);
      output_info_pad_ = nullptr;
    }
    if (output_info_ != nullptr) {
      api_->object_unref(output_info_);
      output_info_ = nullptr;
    }
    if (display_info_pad_ != nullptr) {
      if (display_info_probe_id_ != 0) {
        api_->pad_remove_probe(display_info_pad_, display_info_probe_id_);
        display_info_probe_id_ = 0;
      }
      api_->object_unref(display_info_pad_);
      display_info_pad_ = nullptr;
    }
    if (display_info_ != nullptr) {
      api_->object_unref(display_info_);
      display_info_ = nullptr;
    }
    geometry_probe_state_.reset();
    if (sink_ != nullptr) {
      api_->object_unref(sink_);
      sink_ = nullptr;
    }
    if (pipeline_ != nullptr) {
      api_->object_unref(pipeline_);
      pipeline_ = nullptr;
    }
  }

  GpuFileDecodeConfig config_;
  std::string pipeline_description_;
  NvbufSurfaceAbi abi_;
  std::shared_ptr<GstreamerApi> api_;
  void* pipeline_ = nullptr;
  void* sink_ = nullptr;
  void* display_info_ = nullptr;
  void* display_info_pad_ = nullptr;
  unsigned long display_info_probe_id_ = 0;
  void* output_info_ = nullptr;
  void* output_info_pad_ = nullptr;
  unsigned long output_info_probe_id_ = 0;
  void* bus_ = nullptr;
  std::mutex read_mutex_;
  std::shared_ptr<GeometryProbeState> geometry_probe_state_;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> previous_allocation_dimensions_;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> previous_visible_dimensions_;
  std::uint64_t next_frame_index_ = 0;
  std::optional<std::uint64_t> indexed_stream_time_origin_ns_ =
      config_.indexed_stream_time_origin_ns;
  std::optional<std::uint64_t> last_indexed_stream_time_ns_;
  std::optional<std::uint64_t> post_seek_expected_base_index_;
  std::uint32_t timestamp_group_ordinal_ = 0;
  std::uint64_t frames_emitted_ = 0;
  std::uint16_t rotation_degrees_ = 0;
  std::optional<std::string> terminal_error_;
  bool ended_ = false;
};

} // namespace

std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_file_decode_source(GpuFileDecodeConfig config, NvbufSurfaceAbi abi) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  switch (abi) {
  case NvbufSurfaceAbi::DeepStream7_1:
  case NvbufSurfaceAbi::DeepStream9_1:
    break;
  default:
    throw std::invalid_argument("unsupported NvBufSurface ABI");
  }

  auto source = std::make_unique<GstreamerGpuFileDecodeSource>(std::move(config), abi,
                                                               std::make_shared<GstreamerApi>());
  source->start();
  return source;
}

} // namespace reco::io
