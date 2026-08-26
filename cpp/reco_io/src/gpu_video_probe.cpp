#include "reco/io/gpu_video_probe.hpp"

#include "gpu_video_probe_internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reco::io {
namespace {

constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr std::uint64_t kFallbackDurationSeconds = 60;
constexpr std::uint32_t kFallbackFpsNumerator = 30;
constexpr std::uint32_t kFallbackFpsDenominator = 1;
constexpr std::uint64_t kMinimumProbeTimeoutNs = kNanosecondsPerSecond;
constexpr std::uint64_t kMaximumProbeTimeoutNs = 3'600ULL * kNanosecondsPerSecond;
constexpr double kMaximumSaneFps = 1000.0;
constexpr int kGstFormatTime = 3;
constexpr int kGstStateNull = 1;
constexpr int kGstStatePlaying = 4;
constexpr int kGstStateChangeFailure = 0;
constexpr std::uint32_t kGstMessageError = 1U << 1U;
constexpr int kGstSeekFlagFlushAccurate = (1 << 0) | (1 << 1);
constexpr int kGstPadProbeTypeBuffer = 1 << 4;
constexpr int kGstPadProbeDrop = 0;
constexpr int kGstPadProbeOk = 1;
constexpr std::uint64_t kGstClockTimeNone = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kSamplePollTimeoutNs = 100'000'000ULL;
constexpr std::size_t kTimingAnalysisSamples = 64;
// The lookahead keeps the analysis prefix presentation-contiguous when the
// parser scan stops inside an H.264/HEVC picture-reorder group.
constexpr std::size_t kTimingReorderLookahead = 32;
// DTS can trail presentation time by the decoder picture-reorder depth.
constexpr std::uint64_t kMaximumDecodeReorderFrames = 32;
// Exact AU counting remains bounded for long recordings. Results beyond this
// prefix are explicitly marked estimated after duration correlation.
constexpr std::uint64_t kMaximumExactCountSamples = 512;
constexpr std::uint64_t kMaximumConflictingMetadataCountSamples = 4096;
// Keep metadata probing bounded. Longer streams use sparse timing windows and
// remain explicitly unverified for frame-index sampling.
constexpr std::uint64_t kMaximumEagerCadenceValidationSamples = 10'000;
constexpr std::uint64_t kTailTimingWindowNs = 5ULL * kNanosecondsPerSecond;
constexpr std::uint64_t kTimingWindowSeekPrerollNs = kNanosecondsPerSecond;
// Five seconds at the highest accepted frame rate plus a complete decoder
// reorder suffix. This also remains larger than the bounded prefix scan.
constexpr std::size_t kMaximumTimingWindowSamples = 5'000 + kTimingReorderLookahead + 1;
constexpr std::uint64_t kMaximumCompressedSamplePulls = 50'000;
constexpr std::uint64_t kMaximumBytesBeforeCompressedAccessUnit = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kStoredTimingSamples = kMaximumTimingWindowSamples;
static_assert(kStoredTimingSamples >= kTimingAnalysisSamples + kTimingReorderLookahead);
static_assert(kStoredTimingSamples > kMaximumConflictingMetadataCountSamples);
constexpr std::size_t kMinimumExactNoncanonicalRateSamples = 128;
constexpr std::uint32_t kMaximumFrameRateDenominator = 1001;
constexpr long double kFrameRatePreferenceTolerance = 0.05L;
constexpr long double kMaximumTimingDeltaVariation = 0.10L;
constexpr long double kCanonicalFrameRateTolerance = 0.0005L;
constexpr long double kMaterialGridResidualImprovementFrames = 0.0001L;
constexpr long double kMaximumTimestampPhaseResidualFrames = 0.05L;
constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 17> kCanonicalFrameRates{{
    {24'000, 1'001},
    {24, 1},
    {25, 1},
    {30'000, 1'001},
    {30, 1},
    {48, 1},
    {50, 1},
    {60'000, 1'001},
    {60, 1},
    {100, 1},
    {120'000, 1'001},
    {120, 1},
    {200, 1},
    {240'000, 1'001},
    {240, 1},
    {480, 1},
    {960, 1},
}};

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

struct GstPadProbeInfoAbi {
  std::int32_t type = 0;
  unsigned long id = 0;
  void* data = nullptr;
};

static_assert(offsetof(GstBufferAbi, pts) == (sizeof(void*) == 8 ? 72 : 40));
static_assert(sizeof(GstBufferAbi) == (sizeof(void*) == 8 ? 112 : 80));
static_assert(offsetof(GstPadProbeInfoAbi, data) ==
              (sizeof(void*) == 8 && sizeof(unsigned long) == 8 ? 16 : 8));

#if defined(_WIN32)
std::wstring utf8_to_wide(std::string_view value) {
  if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw GpuVideoProbeError("video-probe runtime path is too long");
  }
  const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                        static_cast<int>(value.size()), nullptr, 0);
  if (size <= 0) {
    throw GpuVideoProbeError("video-probe runtime path is not valid UTF-8");
  }
  std::wstring result(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), result.data(), size) != size) {
    throw GpuVideoProbeError("failed to convert video-probe runtime path");
  }
  return result;
}

std::wstring resolve_windows_library_path(const std::wstring& requested) {
  const std::filesystem::path requested_path(requested);
  if (requested_path.is_absolute()) {
    return requested_path.lexically_normal().native();
  }
  if (requested_path.has_parent_path()) {
    std::error_code error;
    const auto absolute = std::filesystem::absolute(requested_path, error);
    return error ? requested : absolute.lexically_normal().native();
  }

  const auto required = GetEnvironmentVariableW(L"PATH", nullptr, 0);
  if (required == 0) {
    return requested;
  }
  std::wstring path_value(static_cast<std::size_t>(required), L'\0');
  const auto written = GetEnvironmentVariableW(L"PATH", path_value.data(), required);
  if (written == 0 || written >= required) {
    return requested;
  }
  path_value.resize(written);
  std::size_t offset = 0;
  while (offset <= path_value.size()) {
    const auto separator = path_value.find(L';', offset);
    auto directory = path_value.substr(
        offset, separator == std::wstring::npos ? std::wstring::npos : separator - offset);
    if (directory.size() >= 2 && directory.front() == L'"' && directory.back() == L'"') {
      directory = directory.substr(1, directory.size() - 2);
    }
    const std::filesystem::path directory_path(directory);
    if (!directory.empty() && directory_path.is_absolute()) {
      const auto candidate = (directory_path / requested_path).lexically_normal();
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error) && !error) {
        return candidate.native();
      }
    }
    if (separator == std::wstring::npos) {
      break;
    }
    offset = separator + 1;
  }
  return requested;
}
#endif

class DynamicLibrary {
public:
  explicit DynamicLibrary(std::string path) : path_(std::move(path)) {
#if defined(_WIN32)
    const auto wide_path = resolve_windows_library_path(utf8_to_wide(path_));
    auto flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    if (std::filesystem::path(wide_path).is_absolute()) {
      flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
    }
    handle_ = LoadLibraryExW(wide_path.c_str(), nullptr, flags);
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
  using PadGetCurrentCaps = void* (*)(void*);
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
  using BufferGetSize = std::size_t (*)(const void*);
  using SampleGetCaps = void* (*)(void*);
  using SampleGetSegment = const void* (*)(void*);
  using SampleUnref = void (*)(void*);
  using SegmentToStreamTime = std::uint64_t (*)(const void*, int, std::uint64_t);
  using DestroyNotify = void (*)(void*);
  using PadProbeCallback = int (*)(void*, void*, void*);
  using PadAddProbe = unsigned long (*)(void*, int, PadProbeCallback, void*, DestroyNotify);
  using PadRemoveProbe = void (*)(void*, unsigned long);
  using BusTimedPopFiltered = void* (*)(void*, std::uint64_t, std::uint32_t);
  using MessageParseError = void (*)(void*, GErrorAbi**, char**);
  using MessageUnref = void (*)(void*);
  using ObjectUnref = void (*)(void*);
  using CapsUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);
  using GenericCallback = void (*)();
  using SignalConnectData = unsigned long (*)(void*, const char*, GenericCallback, void*,
                                              DestroyNotify, int);
  using SignalHandlerDisconnect = void (*)(void*, unsigned long);

  ProbeApi() {
#if defined(_WIN32)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH", {"gstreamer-1.0-0.dll"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"gstapp-1.0-0.dll"}, "GstApp");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0-0.dll", "glib-2.0-0.dll"}, "GLib");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH", {"libgobject-2.0-0.dll"}, "GObject");
#elif defined(__APPLE__)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib",
                         "/opt/homebrew/lib/libgstreamer-1.0.0.dylib",
                         "/usr/local/lib/libgstreamer-1.0.0.dylib",
                         "/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/"
                         "libgstreamer-1.0.0.dylib"},
                        "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH",
                       {"libgstapp-1.0.0.dylib", "libgstapp-1.0.dylib",
                        "/opt/homebrew/lib/libgstapp-1.0.0.dylib",
                        "/usr/local/lib/libgstapp-1.0.0.dylib",
                        "/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/"
                        "libgstapp-1.0.0.dylib"},
                       "GstApp");
    glib =
        load_library("RECO_GLIB_DYLIB_PATH",
                     {"libglib-2.0.0.dylib", "libglib-2.0.dylib",
                      "/opt/homebrew/lib/libglib-2.0.0.dylib", "/usr/local/lib/libglib-2.0.0.dylib",
                      "/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/"
                      "libglib-2.0.0.dylib"},
                     "GLib");
    gobject = load_library(
        "RECO_GOBJECT_DYLIB_PATH",
        {"libgobject-2.0.0.dylib", "libgobject-2.0.dylib",
         "/opt/homebrew/lib/libgobject-2.0.0.dylib", "/usr/local/lib/libgobject-2.0.0.dylib",
         "/Library/Frameworks/GStreamer.framework/Versions/1.0/lib/libgobject-2.0.0.dylib"},
        "GObject");
#else
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"}, "GStreamer");
    app = load_library("RECO_GSTAPP_DYLIB_PATH", {"libgstapp-1.0.so.0", "libgstapp-1.0.so"},
                       "GstApp");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.so.0", "libglib-2.0.so"}, "GLib");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH", {"libgobject-2.0.so.0", "libgobject-2.0.so"},
                           "GObject");
#endif

    init_check = core->symbol<InitCheck>("gst_init_check");
    version = core->symbol<Version>("gst_version");
    parse_launch = core->symbol<ParseLaunch>("gst_parse_launch");
    bin_get_by_name = core->symbol<BinGetByName>("gst_bin_get_by_name");
    element_get_static_pad = core->symbol<ElementGetStaticPad>("gst_element_get_static_pad");
    pad_get_current_caps = core->symbol<PadGetCurrentCaps>("gst_pad_get_current_caps");
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
    buffer_get_size = core->symbol<BufferGetSize>("gst_buffer_get_size");
    sample_get_caps = core->symbol<SampleGetCaps>("gst_sample_get_caps");
    sample_get_segment = core->symbol<SampleGetSegment>("gst_sample_get_segment");
    sample_unref = core->symbol<SampleUnref>("gst_mini_object_unref");
    segment_to_stream_time = core->symbol<SegmentToStreamTime>("gst_segment_to_stream_time");
    pad_add_probe = core->symbol<PadAddProbe>("gst_pad_add_probe");
    pad_remove_probe = core->symbol<PadRemoveProbe>("gst_pad_remove_probe");
    bus_timed_pop_filtered = core->symbol<BusTimedPopFiltered>("gst_bus_timed_pop_filtered");
    message_parse_error = core->symbol<MessageParseError>("gst_message_parse_error");
    message_unref = core->symbol<MessageUnref>("gst_mini_object_unref");
    object_unref = core->symbol<ObjectUnref>("gst_object_unref");
    caps_unref = core->symbol<CapsUnref>("gst_caps_unref");
    error_free = glib->symbol<ErrorFree>("g_error_free");
    free = glib->symbol<Free>("g_free");
    signal_connect_data = gobject->symbol<SignalConnectData>("g_signal_connect_data");
    signal_handler_disconnect =
        gobject->symbol<SignalHandlerDisconnect>("g_signal_handler_disconnect");
  }

  std::shared_ptr<DynamicLibrary> core;
  std::shared_ptr<DynamicLibrary> app;
  std::shared_ptr<DynamicLibrary> glib;
  std::shared_ptr<DynamicLibrary> gobject;
  InitCheck init_check = nullptr;
  Version version = nullptr;
  ParseLaunch parse_launch = nullptr;
  BinGetByName bin_get_by_name = nullptr;
  ElementGetStaticPad element_get_static_pad = nullptr;
  PadGetCurrentCaps pad_get_current_caps = nullptr;
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
  BufferGetSize buffer_get_size = nullptr;
  SampleGetCaps sample_get_caps = nullptr;
  SampleGetSegment sample_get_segment = nullptr;
  SampleUnref sample_unref = nullptr;
  SegmentToStreamTime segment_to_stream_time = nullptr;
  PadAddProbe pad_add_probe = nullptr;
  PadRemoveProbe pad_remove_probe = nullptr;
  BusTimedPopFiltered bus_timed_pop_filtered = nullptr;
  MessageParseError message_parse_error = nullptr;
  MessageUnref message_unref = nullptr;
  ObjectUnref object_unref = nullptr;
  CapsUnref caps_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
  SignalConnectData signal_connect_data = nullptr;
  SignalHandlerDisconnect signal_handler_disconnect = nullptr;
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

std::filesystem::path path_from_utf8(std::string_view value) {
#if defined(_WIN32)
  const auto* begin = reinterpret_cast<const char8_t*>(value.data());
  return std::filesystem::path(std::u8string(begin, begin + value.size()));
#else
  return std::filesystem::path(std::string(value));
#endif
}

std::string build_probe_pipeline(const GpuFileDecodeConfig& config,
                                 const std::filesystem::path& absolute_path) {
  std::ostringstream pipeline;
  pipeline << "filesrc location=" << quote_gstreamer_property(path_for_gstreamer(absolute_path))
           << " ! ";
  if (config.elementary_stream) {
    pipeline << "identity name=input_budget silent=true ! " << parser_for_codec(config.codec);
  } else {
    pipeline << gpu_decode_container_demuxer(*config.container)
             << " ! capsfilter caps=\"video/x-h264;video/x-h265\""
             << " ! identity name=container_info silent=true"
             << " ! identity name=input_budget silent=true ! parsebin";
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

std::uint64_t add_saturating(std::uint64_t first, std::uint64_t second) {
  return second > std::numeric_limits<std::uint64_t>::max() - first
             ? std::numeric_limits<std::uint64_t>::max()
             : first + second;
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

class CompressedSampleBudget {
public:
  [[nodiscard]] bool admit_input(std::size_t bytes) noexcept {
    if (input_limit_exceeded_.load(std::memory_order_acquire)) {
      return false;
    }
    auto current = input_bytes_.load(std::memory_order_relaxed);
    while (true) {
      const auto increment = static_cast<std::uint64_t>(bytes);
      const auto next = increment > std::numeric_limits<std::uint64_t>::max() - current
                            ? std::numeric_limits<std::uint64_t>::max()
                            : current + increment;
      if (input_bytes_.compare_exchange_weak(current, next, std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {
        if (next <= kMaximumBytesBeforeCompressedAccessUnit) {
          return true;
        }
        input_limit_exceeded_.store(true, std::memory_order_release);
        return false;
      }
    }
  }

  void require_input_within_limit() const {
    if (input_limit_exceeded_.load(std::memory_order_acquire) ||
        input_bytes_.load(std::memory_order_acquire) > kMaximumBytesBeforeCompressedAccessUnit) {
      throw GpuVideoProbeError("video parser exceeded the compressed bytes-per-access-unit limit");
    }
  }

  void consume() {
    const auto observed = input_bytes_.exchange(0, std::memory_order_acq_rel);
    if (input_limit_exceeded_.load(std::memory_order_acquire) ||
        observed > kMaximumBytesBeforeCompressedAccessUnit) {
      throw GpuVideoProbeError("video parser exceeded the compressed bytes-per-access-unit limit");
    }
    if (consumed_ == kMaximumCompressedSamplePulls) {
      throw GpuVideoProbeError(
          "video parser exceeded the compressed access-unit metadata work limit");
    }
    ++consumed_;
  }

private:
  std::atomic<std::uint64_t> input_bytes_{0};
  std::atomic<bool> input_limit_exceeded_{false};
  std::uint64_t consumed_ = 0;
};

struct InputBudgetCallbackData {
  const ProbeApi* api = nullptr;
  CompressedSampleBudget* budget = nullptr;
};

int input_budget_probe(void*, void* probe_info, void* user_data) noexcept {
  const auto* callback = static_cast<const InputBudgetCallbackData*>(user_data);
  if (callback == nullptr || callback->api == nullptr || callback->budget == nullptr ||
      probe_info == nullptr) {
    return kGstPadProbeDrop;
  }
  const auto* info = static_cast<const GstPadProbeInfoAbi*>(probe_info);
  if (info->data == nullptr) {
    return kGstPadProbeDrop;
  }
  return callback->budget->admit_input(callback->api->buffer_get_size(info->data))
             ? kGstPadProbeOk
             : kGstPadProbeDrop;
}

std::optional<std::pair<std::uint32_t, std::uint32_t>>
reduce_matroska_buffer_duration(std::uint64_t duration_ns) {
  // libavformat derives Matroska stream rates from DefaultDuration with this
  // bounded continued-fraction reduction.
  constexpr std::uint64_t kMaximumDurationNs = 1'000'000'000'000ULL;
  constexpr std::uint64_t kMaximumRationalComponent = 30'000;
  if (duration_ns == 0 || duration_ns > kMaximumDurationNs) {
    return std::nullopt;
  }

  std::uint64_t numerator = kNanosecondsPerSecond;
  std::uint64_t denominator = duration_ns;
  const auto divisor = std::gcd(numerator, denominator);
  numerator /= divisor;
  denominator /= divisor;

  std::uint64_t previous_numerator = 0;
  std::uint64_t previous_denominator = 1;
  std::uint64_t current_numerator = 1;
  std::uint64_t current_denominator = 0;
  if (numerator <= kMaximumRationalComponent && denominator <= kMaximumRationalComponent) {
    current_numerator = numerator;
    current_denominator = denominator;
    denominator = 0;
  }
  while (denominator != 0) {
    auto coefficient = numerator / denominator;
    const auto remainder = numerator - denominator * coefficient;
    const auto next_numerator = coefficient * current_numerator + previous_numerator;
    const auto next_denominator = coefficient * current_denominator + previous_denominator;
    if (next_numerator > kMaximumRationalComponent ||
        next_denominator > kMaximumRationalComponent) {
      if (current_numerator != 0) {
        coefficient = (kMaximumRationalComponent - previous_numerator) / current_numerator;
      }
      if (current_denominator != 0) {
        coefficient = std::min(coefficient, (kMaximumRationalComponent - previous_denominator) /
                                                current_denominator);
      }
      const auto candidate_denominator =
          2 * coefficient * current_denominator + previous_denominator;
      if (denominator * candidate_denominator > numerator * current_denominator) {
        current_numerator = coefficient * current_numerator + previous_numerator;
        current_denominator = coefficient * current_denominator + previous_denominator;
      }
      break;
    }
    previous_numerator = current_numerator;
    previous_denominator = current_denominator;
    current_numerator = next_numerator;
    current_denominator = next_denominator;
    numerator = denominator;
    denominator = remainder;
  }
  if (current_numerator == 0 || current_denominator == 0) {
    return std::nullopt;
  }
  const auto fps = static_cast<long double>(current_numerator) / current_denominator;
  if (!std::isfinite(fps) || fps <= 0.0L || fps > kMaximumSaneFps) {
    return std::nullopt;
  }
  return std::pair(static_cast<std::uint32_t>(current_numerator),
                   static_cast<std::uint32_t>(current_denominator));
}

void* pull_compressed_sample(const std::shared_ptr<ProbeApi>& api, void* probe_sink, void* bus,
                             CompressedSampleBudget& sample_budget,
                             std::chrono::steady_clock::time_point deadline,
                             std::string_view timeout_message) {
  while (true) {
    sample_budget.require_input_within_limit();
    const auto poll_timeout =
        std::min(remaining_timeout_ns(deadline, timeout_message), kSamplePollTimeoutNs);
    if (void* sample = api->app_sink_try_pull_sample(probe_sink, poll_timeout); sample != nullptr) {
      std::unique_ptr<void, ProbeApi::SampleUnref> sample_owner(sample, api->sample_unref);
      if (const auto error = pop_pipeline_error(api, bus); !error.empty()) {
        throw GpuVideoProbeError(error);
      }
      sample_budget.consume();
      return sample_owner.release();
    }
    sample_budget.require_input_within_limit();
    if (const auto error = pop_pipeline_error(api, bus); !error.empty()) {
      throw GpuVideoProbeError(error);
    }
    if (api->app_sink_is_eos(probe_sink) != 0) {
      return nullptr;
    }
  }
}

struct TimingScan {
  std::array<std::uint64_t, kStoredTimingSamples> stream_times{};
  std::array<std::uint64_t, kStoredTimingSamples> timed_sample_indices{};
  std::uint64_t sample_count = 0;
  std::uint64_t timed_sample_count = 0;
  std::size_t stored_timing_count = 0;
  std::optional<std::uint64_t> first_stream_time;
  std::optional<std::uint64_t> last_stream_time;
  std::optional<std::uint64_t> final_frame_end;
  std::optional<std::uint64_t> final_frame_duration;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> buffer_duration_frame_rate;
  bool all_durations_known = true;
  bool all_buffer_duration_rates_known = true;
  bool buffer_duration_rates_consistent = true;
  bool reached_eos = false;
};

struct ParserCapsIdentity {
  std::string codec;
  std::string stream_format;
  std::string alignment;
  int width = 0;
  int height = 0;
  std::optional<std::pair<int, int>> frame_rate;
};

void validate_parser_sample_caps(const std::shared_ptr<ProbeApi>& api, void* sample,
                                 const ParserCapsIdentity& expected) {
  void* caps = api->sample_get_caps(sample);
  void* structure = caps == nullptr ? nullptr : api->caps_get_structure(caps, 0);
  const char* codec = structure == nullptr ? nullptr : api->structure_get_name(structure);
  const char* stream_format =
      structure == nullptr ? nullptr : api->structure_get_string(structure, "stream-format");
  const char* alignment =
      structure == nullptr ? nullptr : api->structure_get_string(structure, "alignment");
  int parsed = 0;
  int width = 0;
  int height = 0;
  int fps_numerator = 0;
  int fps_denominator = 0;
  const bool has_frame_rate =
      structure != nullptr &&
      api->structure_get_fraction(structure, "framerate", &fps_numerator, &fps_denominator) != 0 &&
      fps_numerator > 0 && fps_denominator > 0;
  const bool frame_rate_matches =
      has_frame_rate == expected.frame_rate.has_value() &&
      (!has_frame_rate ||
       static_cast<std::int64_t>(fps_numerator) * expected.frame_rate->second ==
           static_cast<std::int64_t>(expected.frame_rate->first) * fps_denominator);
  if (structure == nullptr || codec == nullptr || stream_format == nullptr ||
      alignment == nullptr || api->structure_get_boolean(structure, "parsed", &parsed) == 0 ||
      parsed == 0 || api->structure_get_int(structure, "width", &width) == 0 ||
      api->structure_get_int(structure, "height", &height) == 0 || codec != expected.codec ||
      stream_format != expected.stream_format || alignment != expected.alignment ||
      width != expected.width || height != expected.height || !frame_rate_matches) {
    throw GpuVideoProbeError(
        "video parser changed codec, geometry, or timing caps during the selected stream");
  }
}

void observe_timing_sample(const std::shared_ptr<ProbeApi>& api, void* sample, TimingScan& scan,
                           const ParserCapsIdentity* expected_caps = nullptr) {
  if (expected_caps != nullptr) {
    validate_parser_sample_caps(api, sample, *expected_caps);
  }
  const auto* buffer = static_cast<const GstBufferAbi*>(api->sample_get_buffer(sample));
  if (buffer == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe returned a sample without a buffer");
  }
  ++scan.sample_count;
  const auto duration_rate = buffer->duration == kGstClockTimeNone
                                 ? std::nullopt
                                 : reduce_matroska_buffer_duration(buffer->duration);
  if (!duration_rate.has_value()) {
    scan.all_buffer_duration_rates_known = false;
  } else if (!scan.buffer_duration_frame_rate.has_value()) {
    scan.buffer_duration_frame_rate = duration_rate;
  } else if (*scan.buffer_duration_frame_rate != *duration_rate) {
    scan.buffer_duration_rates_consistent = false;
  }
  if (buffer->pts == kGstClockTimeNone) {
    scan.all_durations_known = false;
    return;
  }
  const void* segment = api->sample_get_segment(sample);
  if (segment == nullptr) {
    scan.all_durations_known = false;
    return;
  }
  const auto stream_time = api->segment_to_stream_time(segment, kGstFormatTime, buffer->pts);
  if (stream_time == kGstClockTimeNone) {
    scan.all_durations_known = false;
    return;
  }

  ++scan.timed_sample_count;
  if (scan.stored_timing_count < scan.stream_times.size()) {
    scan.stream_times[scan.stored_timing_count] = stream_time;
    scan.timed_sample_indices[scan.stored_timing_count] = scan.sample_count - 1;
    ++scan.stored_timing_count;
  }
  scan.first_stream_time = scan.first_stream_time.has_value()
                               ? std::min(*scan.first_stream_time, stream_time)
                               : stream_time;
  scan.last_stream_time = scan.last_stream_time.has_value()
                              ? std::max(*scan.last_stream_time, stream_time)
                              : stream_time;
  if (buffer->duration == 0 || buffer->duration == kGstClockTimeNone) {
    scan.all_durations_known = false;
    return;
  }
  const auto frame_end = buffer->duration > std::numeric_limits<std::uint64_t>::max() - stream_time
                             ? std::numeric_limits<std::uint64_t>::max()
                             : stream_time + buffer->duration;
  if (!scan.final_frame_end.has_value() || frame_end > *scan.final_frame_end) {
    scan.final_frame_end = frame_end;
    scan.final_frame_duration = buffer->duration;
  }
}

class DenseCadenceValidator {
public:
  DenseCadenceValidator(std::uint32_t fps_numerator, std::uint32_t fps_denominator)
      : fps_numerator_(fps_numerator), fps_denominator_(fps_denominator) {}

  void observe(std::uint64_t stream_time, std::uint64_t sample_index) {
    if (finalized_time_.has_value() && stream_time <= *finalized_time_) {
      fail("presentation timestamps exceed the supported decode reorder depth");
    }
    auto& group = pending_[stream_time];
    ++group.count;
    group.latest_sample_index = std::max(group.latest_sample_index, sample_index);
    while (!pending_.empty()) {
      const auto& first = pending_.begin()->second;
      if (sample_index <= first.latest_sample_index ||
          sample_index - first.latest_sample_index <= kTimingReorderLookahead) {
        break;
      }
      finalize_first();
    }
    if (pending_.size() > 2 * kTimingReorderLookahead + 1) {
      fail("presentation timestamps exceed the supported decode reorder depth");
    }
  }

  void finish() {
    while (!pending_.empty()) {
      finalize_first();
    }
    if (finalized_group_count_ < 3) {
      fail("fewer than three distinct presentation timestamps were observed");
    }
    if (previous_group_count_.has_value() && previous_grid_step_.has_value() &&
        *previous_group_count_ != *previous_grid_step_) {
      fail("the terminal timestamp group is incomplete");
    }
  }

private:
  struct PendingGroup {
    std::size_t count = 0;
    std::uint64_t latest_sample_index = 0;
  };

  [[noreturn]] static void fail(std::string_view detail) {
    throw GpuVideoProbeError(
        "variable frame-rate video is unsupported for indexed GPU calibration sampling (" +
        std::string(detail) + ")");
  }

  void finalize_first() {
    auto first = pending_.begin();
    const auto stream_time = first->first;
    const auto group_count = first->second.count;
    pending_.erase(first);
    ++finalized_group_count_;

    if (origin_time_.has_value()) {
      if (stream_time <= *previous_time_) {
        fail("presentation timestamps are not strictly ordered");
      }
      const auto offset_ns = stream_time - *origin_time_;
      const auto frame_position = static_cast<long double>(offset_ns) * fps_numerator_ /
                                  (kNanosecondsPerSecond * fps_denominator_);
      const auto rounded_position = std::round(frame_position);
      if (rounded_position < 1.0L ||
          std::abs(frame_position - rounded_position) > kMaximumTimestampPhaseResidualFrames ||
          rounded_position > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        fail("timestamps do not fit one constant frame grid");
      }
      const auto grid_position = static_cast<std::uint64_t>(rounded_position);
      if (!previous_grid_position_.has_value() || grid_position <= *previous_grid_position_) {
        fail("presentation timestamps do not advance on the frame grid");
      }
      const auto grid_step = grid_position - *previous_grid_position_;
      if (!previous_group_count_.has_value() || *previous_group_count_ != grid_step) {
        fail("timestamp cadence contains missing or duplicated frame periods");
      }
      previous_grid_position_ = grid_position;
      previous_grid_step_ = grid_step;
    } else {
      origin_time_ = stream_time;
      previous_grid_position_ = 0;
    }
    previous_time_ = stream_time;
    finalized_time_ = stream_time;
    previous_group_count_ = group_count;
  }

  std::uint32_t fps_numerator_ = 0;
  std::uint32_t fps_denominator_ = 0;
  std::map<std::uint64_t, PendingGroup> pending_;
  std::size_t finalized_group_count_ = 0;
  std::optional<std::uint64_t> origin_time_;
  std::optional<std::uint64_t> previous_time_;
  std::optional<std::uint64_t> finalized_time_;
  std::optional<std::size_t> previous_group_count_;
  std::optional<std::uint64_t> previous_grid_position_;
  std::optional<std::uint64_t> previous_grid_step_;
};

struct CadenceValidationCandidate {
  std::pair<std::uint32_t, std::uint32_t> rate;
  DenseCadenceValidator validator;
  std::optional<std::string> failure;
};

struct InferredFrameRate {
  std::uint32_t numerator = 0;
  std::uint32_t denominator = 0;
  std::size_t timestamp_multiplicity = 1;
  long double observed_fps = 0.0L;
  long double finite_span_fps_uncertainty = 0.0L;
  bool requires_complete_stream = false;
};

long double mean_timestamp_grid_residual(
    const std::array<std::uint64_t, kStoredTimingSamples>& sorted_unique_times,
    std::size_t unique_count, std::uint32_t numerator, std::uint32_t denominator) {
  long double total_residual = 0.0L;
  const auto candidate_fps = static_cast<long double>(numerator) / denominator;
  for (std::size_t index = 1; index < unique_count; ++index) {
    const auto delta_ns = sorted_unique_times[index] - sorted_unique_times[index - 1];
    const auto frame_steps =
        static_cast<long double>(delta_ns) * candidate_fps / kNanosecondsPerSecond;
    total_residual += std::abs(frame_steps - std::round(frame_steps));
  }
  return total_residual / static_cast<long double>(unique_count - 1);
}

bool timestamps_fit_rate_phase(
    const std::array<std::uint64_t, kStoredTimingSamples>& sorted_unique_times,
    std::size_t unique_count, std::uint32_t numerator, std::uint32_t denominator) {
  const auto fps = static_cast<long double>(numerator) / denominator;
  for (std::size_t index = 1; index < unique_count; ++index) {
    const auto offset_ns = sorted_unique_times[index] - sorted_unique_times[0];
    const auto frame_position = static_cast<long double>(offset_ns) * fps / kNanosecondsPerSecond;
    if (std::abs(frame_position - std::round(frame_position)) >
        kMaximumTimestampPhaseResidualFrames) {
      return false;
    }
  }
  return true;
}

std::size_t positional_timestamp_multiplicity(
    const TimingScan& scan,
    const std::array<std::uint64_t, kStoredTimingSamples>& timestamp_group_first_indices,
    const std::array<std::uint64_t, kStoredTimingSamples>& timestamp_group_last_indices,
    std::size_t timestamp_group_count, std::size_t duplicate_timestamp_multiplicity) {
  if (timestamp_group_count < 2 || scan.timed_sample_count == scan.sample_count ||
      duplicate_timestamp_multiplicity == 0 ||
      timestamp_group_first_indices[1] <= timestamp_group_first_indices[0]) {
    return 1;
  }
  const auto access_unit_stride =
      timestamp_group_first_indices[1] - timestamp_group_first_indices[0];
  if (access_unit_stride % duplicate_timestamp_multiplicity != 0) {
    return 1;
  }
  for (std::size_t index = 1; index < timestamp_group_count; ++index) {
    if (timestamp_group_first_indices[index] <= timestamp_group_first_indices[index - 1] ||
        timestamp_group_first_indices[index] - timestamp_group_first_indices[index - 1] !=
            access_unit_stride) {
      return 1;
    }
  }
  const auto first_index = timestamp_group_first_indices[0];
  const auto last_index = timestamp_group_last_indices[timestamp_group_count - 1];
  if (first_index >= access_unit_stride || last_index >= scan.sample_count ||
      scan.sample_count - 1 - last_index >= access_unit_stride) {
    return 1;
  }
  return static_cast<std::size_t>(access_unit_stride / duplicate_timestamp_multiplicity);
}

std::optional<InferredFrameRate> infer_sparse_reordered_frame_rate(
    const TimingScan& scan,
    const std::array<std::uint64_t, kStoredTimingSamples>& sorted_unique_times,
    const std::array<std::pair<std::uint64_t, std::uint64_t>, kStoredTimingSamples>& timing_points,
    std::size_t unique_count) {
  if (scan.stored_timing_count != scan.timed_sample_count ||
      scan.timed_sample_count == scan.sample_count || unique_count < 3 ||
      scan.sample_count <= 2 * kTimingReorderLookahead + 1) {
    return std::nullopt;
  }

  const auto point_end =
      timing_points.begin() + static_cast<std::ptrdiff_t>(scan.stored_timing_count);
  const auto first_group_end = std::find_if(
      timing_points.begin(), point_end, [first_time = sorted_unique_times[0]](const auto& point) {
        return point.first != first_time;
      });
  const auto last_group_begin =
      std::lower_bound(timing_points.begin(), point_end,
                       std::pair(sorted_unique_times[unique_count - 1], std::uint64_t{0}));
  if (first_group_end == timing_points.begin() || last_group_begin == point_end) {
    return std::nullopt;
  }
  const auto first_decode_index = std::min_element(timing_points.begin(), first_group_end,
                                                   [](const auto& left, const auto& right) {
                                                     return left.second < right.second;
                                                   })
                                      ->second;
  const auto last_decode_index =
      std::max_element(last_group_begin, point_end, [](const auto& left, const auto& right) {
        return left.second < right.second;
      })->second;
  if (first_decode_index > kTimingReorderLookahead || last_decode_index >= scan.sample_count ||
      scan.sample_count - 1 - last_decode_index > kTimingReorderLookahead) {
    return std::nullopt;
  }

  const auto span_ns = sorted_unique_times[unique_count - 1] - sorted_unique_times[0];
  if (span_ns == 0) {
    return std::nullopt;
  }
  const auto sample_intervals = static_cast<long double>(scan.sample_count - 1);
  const auto observed_fps = sample_intervals * kNanosecondsPerSecond / span_ns;
  constexpr long double kTimestampGridToleranceFrames = 0.01L;
  const auto maximum_reorder_interval_error =
      static_cast<long double>(2 * kTimingReorderLookahead + 1);

  std::optional<std::pair<std::uint32_t, std::uint32_t>> best_rate;
  auto best_error = std::numeric_limits<long double>::infinity();
  auto best_grid_error = std::numeric_limits<long double>::infinity();
  long double best_interval_count = 0.0L;
  for (const auto& [numerator, denominator] : kCanonicalFrameRates) {
    const auto candidate_fps = static_cast<long double>(numerator) / denominator;
    const auto candidate_intervals =
        static_cast<long double>(span_ns) * candidate_fps / kNanosecondsPerSecond;
    if (std::abs(candidate_intervals - sample_intervals) > maximum_reorder_interval_error) {
      continue;
    }
    bool fits_timestamp_grid = true;
    long double grid_error = 0.0L;
    for (std::size_t index = 1; index < unique_count; ++index) {
      const auto delta_ns = sorted_unique_times[index] - sorted_unique_times[index - 1];
      const auto frame_steps =
          static_cast<long double>(delta_ns) * candidate_fps / kNanosecondsPerSecond;
      const auto rounded_steps = std::round(frame_steps);
      if (rounded_steps < 1.0L ||
          std::abs(frame_steps - rounded_steps) > kTimestampGridToleranceFrames) {
        fits_timestamp_grid = false;
        break;
      }
      grid_error += std::abs(frame_steps - rounded_steps);
    }
    const auto error = std::abs(candidate_fps - observed_fps);
    constexpr long double kGridErrorTieTolerance = 1e-12L;
    if (fits_timestamp_grid && (grid_error + kGridErrorTieTolerance < best_grid_error ||
                                (std::abs(grid_error - best_grid_error) <= kGridErrorTieTolerance &&
                                 error < best_error))) {
      best_rate = std::pair(numerator, denominator);
      best_error = error;
      best_grid_error = grid_error;
      best_interval_count = candidate_intervals;
    }
  }
  if (!best_rate.has_value()) {
    return std::nullopt;
  }

  const auto average_multiplicity =
      best_interval_count / static_cast<long double>(unique_count - 1);
  const auto rounded_multiplicity = std::max<long long>(std::llround(average_multiplicity), 1);
  const auto timestamp_multiplicity =
      static_cast<std::size_t>(std::min<long long>(rounded_multiplicity, kTimingReorderLookahead));
  return InferredFrameRate{.numerator = best_rate->first,
                           .denominator = best_rate->second,
                           .timestamp_multiplicity = timestamp_multiplicity,
                           .observed_fps = observed_fps,
                           .finite_span_fps_uncertainty =
                               observed_fps * maximum_reorder_interval_error / sample_intervals};
}

std::optional<InferredFrameRate> infer_constant_frame_rate(const TimingScan& scan,
                                                           bool complete_stream,
                                                           bool complete_timing_span,
                                                           bool trim_reorder_suffix) {
  if (scan.stored_timing_count < 3) {
    return std::nullopt;
  }
  auto times = scan.stream_times;
  const auto stored_end = times.begin() + static_cast<std::ptrdiff_t>(scan.stored_timing_count);
  std::sort(times.begin(), stored_end);
  std::array<std::pair<std::uint64_t, std::uint64_t>, kStoredTimingSamples> timing_points{};
  for (std::size_t index = 0; index < scan.stored_timing_count; ++index) {
    timing_points[index] = {scan.stream_times[index], scan.timed_sample_indices[index]};
  }
  const auto point_end =
      timing_points.begin() + static_cast<std::ptrdiff_t>(scan.stored_timing_count);
  std::sort(timing_points.begin(), point_end);
  const auto reorder_suffix_start =
      scan.sample_count > kTimingReorderLookahead ? scan.sample_count - kTimingReorderLookahead : 0;
  std::size_t duplicate_timestamp_multiplicity = 0;
  std::array<std::uint64_t, kStoredTimingSamples> timestamp_group_first_indices{};
  std::array<std::uint64_t, kStoredTimingSamples> timestamp_group_last_indices{};
  std::size_t timestamp_group_count = 0;
  for (auto group_begin = timing_points.begin(); group_begin != point_end;) {
    const auto group_end =
        std::find_if(group_begin, point_end, [value = group_begin->first](const auto& item) {
          return item.first != value;
        });
    const auto group_size = static_cast<std::size_t>(group_end - group_begin);
    if (duplicate_timestamp_multiplicity == 0) {
      duplicate_timestamp_multiplicity = group_size;
    } else if (group_size != duplicate_timestamp_multiplicity) {
      const bool partial_group_is_in_reorder_suffix =
          !complete_stream && group_size < duplicate_timestamp_multiplicity &&
          std::all_of(
              group_begin, group_end,
              [reorder_suffix_start](const auto& item) {
                return item.second >= reorder_suffix_start;
              });
      if (!partial_group_is_in_reorder_suffix) {
        return std::nullopt;
      }
    }
    timestamp_group_first_indices[timestamp_group_count] = group_begin->second;
    timestamp_group_last_indices[timestamp_group_count] = std::prev(group_end)->second;
    ++timestamp_group_count;
    group_begin = group_end;
  }
  const auto unique_end = std::unique(times.begin(), stored_end);
  const auto stored_unique_count = static_cast<std::size_t>(unique_end - times.begin());
  // An incomplete decode-order prefix can end with future B-frame PTS values
  // whose intervening presentation timestamps arrive just after the pull
  // boundary. Analyze the presentation-contiguous core; doubled intervals in
  // that core, and every interval in an EOS-complete scan, still reject.
  const auto unique_count = trim_reorder_suffix && scan.timed_sample_count == scan.sample_count &&
                                    stored_unique_count > kTimingReorderLookahead + 2
                                ? stored_unique_count - kTimingReorderLookahead
                                : stored_unique_count;
  if (unique_count < 3 || duplicate_timestamp_multiplicity == 0) {
    return std::nullopt;
  }
  const auto positional_multiplicity = positional_timestamp_multiplicity(
      scan, timestamp_group_first_indices, timestamp_group_last_indices, timestamp_group_count,
      duplicate_timestamp_multiplicity);
  const auto timestamp_multiplicity = duplicate_timestamp_multiplicity * positional_multiplicity;

  auto minimum_delta = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t maximum_delta = 0;
  for (std::size_t index = 1; index < unique_count; ++index) {
    const auto delta = times[index] - times[index - 1];
    minimum_delta = std::min(minimum_delta, delta);
    maximum_delta = std::max(maximum_delta, delta);
  }
  if (minimum_delta == 0 || (maximum_delta - minimum_delta > 2 &&
                             static_cast<long double>(maximum_delta - minimum_delta) >
                                 minimum_delta * kMaximumTimingDeltaVariation)) {
    return infer_sparse_reordered_frame_rate(scan, times, timing_points, stored_unique_count);
  }

  const auto span_ns = times[unique_count - 1] - times[0];
  if (span_ns == 0) {
    return std::nullopt;
  }
  const long double observed_fps =
      static_cast<long double>((unique_count - 1) * timestamp_multiplicity) *
      kNanosecondsPerSecond / span_ns;
  // Require a rate correction to improve on caps by more than half a frame
  // over the finite observation span.
  const auto finite_span_fps_uncertainty =
      complete_timing_span && unique_count >= kTimingAnalysisSamples
          ? 0.0L
          : observed_fps /
                (2.0L * static_cast<long double>((unique_count - 1) * timestamp_multiplicity));
  std::optional<std::pair<std::uint32_t, std::uint32_t>> nearest_canonical_rate;
  auto nearest_canonical_error = std::numeric_limits<long double>::infinity();
  for (const auto& [numerator, denominator] : kCanonicalFrameRates) {
    const auto canonical_fps = static_cast<long double>(numerator) / denominator;
    const auto error = std::abs(canonical_fps - observed_fps);
    if (error < nearest_canonical_error) {
      nearest_canonical_rate = std::pair(numerator, denominator);
      nearest_canonical_error = error;
    }
  }
  long double best_error = std::numeric_limits<long double>::infinity();
  std::uint32_t best_numerator = 0;
  std::uint32_t best_denominator = 0;
  for (std::uint32_t denominator = 1; denominator <= kMaximumFrameRateDenominator; ++denominator) {
    const auto rounded_numerator = std::llround(observed_fps * denominator);
    if (rounded_numerator <= 0 || static_cast<unsigned long long>(rounded_numerator) >
                                      std::numeric_limits<std::uint32_t>::max()) {
      continue;
    }
    const auto numerator = static_cast<std::uint32_t>(rounded_numerator);
    const auto error = std::abs(static_cast<long double>(numerator) / denominator - observed_fps);
    if (error < best_error) {
      best_error = error;
      best_numerator = numerator;
      best_denominator = denominator;
    }
  }
  const auto tolerance = std::max(1e-6L, observed_fps * 1e-7L);
  const bool canonical_rate_is_close =
      nearest_canonical_rate.has_value() &&
      nearest_canonical_error <= (static_cast<long double>(nearest_canonical_rate->first) /
                                  nearest_canonical_rate->second) *
                                     kCanonicalFrameRateTolerance;
  const auto best_divisor = best_denominator == 0 ? 1U : std::gcd(best_numerator, best_denominator);
  const auto reduced_best_numerator = best_numerator / best_divisor;
  const auto reduced_best_denominator = best_denominator / best_divisor;
  const bool best_rate_is_canonical =
      best_denominator != 0 &&
      std::any_of(kCanonicalFrameRates.begin(), kCanonicalFrameRates.end(),
                  [reduced_best_numerator, reduced_best_denominator](const auto& rate) {
                    return static_cast<std::uint64_t>(reduced_best_numerator) * rate.second ==
                           static_cast<std::uint64_t>(rate.first) * reduced_best_denominator;
                  });
  bool noncanonical_rate_has_materially_better_grid_fit = false;
  if (best_denominator != 0 && best_error <= tolerance && !best_rate_is_canonical &&
      (complete_timing_span || unique_count >= kMinimumExactNoncanonicalRateSamples)) {
    if (!canonical_rate_is_close) {
      noncanonical_rate_has_materially_better_grid_fit = true;
    } else {
      const auto best_grid_residual = mean_timestamp_grid_residual(
          times, unique_count, reduced_best_numerator, reduced_best_denominator);
      const auto canonical_grid_residual = mean_timestamp_grid_residual(
          times, unique_count, nearest_canonical_rate->first, nearest_canonical_rate->second);
      noncanonical_rate_has_materially_better_grid_fit =
          best_grid_residual + kMaterialGridResidualImprovementFrames < canonical_grid_residual;
    }
  }
  if (noncanonical_rate_has_materially_better_grid_fit &&
      timestamps_fit_rate_phase(times, unique_count, reduced_best_numerator,
                                reduced_best_denominator)) {
    return InferredFrameRate{.numerator = reduced_best_numerator,
                             .denominator = reduced_best_denominator,
                             .timestamp_multiplicity = timestamp_multiplicity,
                             .observed_fps = observed_fps,
                             .finite_span_fps_uncertainty = finite_span_fps_uncertainty,
                             .requires_complete_stream = true};
  }
  if (canonical_rate_is_close &&
      timestamps_fit_rate_phase(times, unique_count, nearest_canonical_rate->first,
                                nearest_canonical_rate->second)) {
    return InferredFrameRate{.numerator = nearest_canonical_rate->first,
                             .denominator = nearest_canonical_rate->second,
                             .timestamp_multiplicity = timestamp_multiplicity,
                             .observed_fps = observed_fps,
                             .finite_span_fps_uncertainty = finite_span_fps_uncertainty};
  }
  return std::nullopt;
}

bool inferred_frame_rate_is_eligible(const InferredFrameRate& inferred, bool complete_stream,
                                     bool representative_tail_matches = false) {
  return !inferred.requires_complete_stream || complete_stream || representative_tail_matches;
}

bool inferred_frame_rate_improves_on_caps(const InferredFrameRate& inferred,
                                          bool caps_frame_rate_is_plausible, int caps_numerator,
                                          int caps_denominator) {
  if (!caps_frame_rate_is_plausible) {
    return true;
  }
  const auto inferred_fps = static_cast<long double>(inferred.numerator) / inferred.denominator;
  const auto caps_fps = static_cast<long double>(caps_numerator) / caps_denominator;
  return std::abs(inferred_fps - inferred.observed_fps) + inferred.finite_span_fps_uncertainty <
         std::abs(caps_fps - inferred.observed_fps);
}

bool frame_rates_are_identical(std::uint32_t first_numerator, std::uint32_t first_denominator,
                               std::uint32_t second_numerator, std::uint32_t second_denominator) {
  return static_cast<std::uint64_t>(first_numerator) * second_denominator ==
         static_cast<std::uint64_t>(second_numerator) * first_denominator;
}

struct UntimedPresentationPrefix {
  std::uint64_t frame_count = 0;
  std::uint64_t duration_ns = 0;
};

UntimedPresentationPrefix infer_untimed_presentation_prefix(const TimingScan& scan,
                                                            std::uint32_t fps_numerator,
                                                            std::uint32_t fps_denominator) {
  if (scan.stored_timing_count == 0 || scan.timed_sample_indices[0] == 0 ||
      !scan.first_stream_time.has_value()) {
    return {};
  }
  constexpr auto kOriginCandidateCount = kTimingAnalysisSamples + kTimingReorderLookahead;
  const auto candidate_count = std::min(scan.stored_timing_count, kOriginCandidateCount);
  // A signed mean over the analysis window plus reorder lookahead cancels
  // complete decode-order permutations without biasing the timeline origin.
  long double origin_sum = 0.0L;
  for (std::size_t index = 0; index < candidate_count; ++index) {
    const auto nominal_offset =
        frame_timestamp_ns(scan.timed_sample_indices[index], fps_numerator, fps_denominator);
    origin_sum += static_cast<long double>(scan.stream_times[index]) - nominal_offset;
  }
  const auto estimated_origin_value = origin_sum / static_cast<long double>(candidate_count);
  const auto estimated_origin =
      estimated_origin_value <= 0.0L ? 0
      : estimated_origin_value >=
              static_cast<long double>(std::numeric_limits<std::uint64_t>::max())
          ? std::numeric_limits<std::uint64_t>::max()
          : static_cast<std::uint64_t>(estimated_origin_value);
  if (*scan.first_stream_time <= estimated_origin) {
    return {};
  }
  const auto available_prefix_duration = *scan.first_stream_time - estimated_origin;
  const auto half_frame_duration = kNanosecondsPerSecond * fps_denominator / (2ULL * fps_numerator);
  const auto available_prefix_frames =
      frame_count_for_duration(add_saturating(available_prefix_duration, half_frame_duration),
                               fps_numerator, fps_denominator);
  const auto frame_count = std::min(scan.timed_sample_indices[0], available_prefix_frames);
  return {.frame_count = frame_count,
          .duration_ns =
              frame_count == available_prefix_frames
                  ? available_prefix_duration
                  : minimum_duration_for_frame_count(frame_count, fps_numerator, fps_denominator)};
}

bool frame_rates_are_close(std::uint32_t first_numerator, std::uint32_t first_denominator,
                           std::uint32_t second_numerator, std::uint32_t second_denominator,
                           long double relative_tolerance) {
  const auto first = static_cast<long double>(first_numerator) / first_denominator;
  const auto second = static_cast<long double>(second_numerator) / second_denominator;
  return std::abs(first - second) <= std::max(first, second) * relative_tolerance;
}

std::optional<std::uint64_t> observed_eos_duration(const TimingScan& scan,
                                                   std::uint64_t nominal_frame_duration,
                                                   bool clamp_final_period_to_nominal) {
  if (scan.timed_sample_count != scan.sample_count || !scan.first_stream_time.has_value() ||
      !scan.last_stream_time.has_value() || scan.stored_timing_count == 0) {
    return std::nullopt;
  }
  auto times = scan.stream_times;
  const auto stored_end = times.begin() + static_cast<std::ptrdiff_t>(scan.stored_timing_count);
  std::sort(times.begin(), stored_end);
  const auto unique_end = std::unique(times.begin(), stored_end);
  const auto unique_count = static_cast<std::size_t>(unique_end - times.begin());
  auto final_duration = nominal_frame_duration;
  if (unique_count >= 2) {
    final_duration = times[unique_count - 1] - times[unique_count - 2];
    if (clamp_final_period_to_nominal) {
      final_duration = std::max(final_duration, nominal_frame_duration);
    }
  }
  const auto span = *scan.last_stream_time - *scan.first_stream_time;
  return final_duration > std::numeric_limits<std::uint64_t>::max() - span
             ? std::numeric_limits<std::uint64_t>::max()
             : span + final_duration;
}

std::optional<std::uint64_t> sample_presentation_stream_time(const std::shared_ptr<ProbeApi>& api,
                                                             void* sample) {
  const auto* buffer = static_cast<const GstBufferAbi*>(api->sample_get_buffer(sample));
  if (buffer == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe returned a sample without a buffer");
  }
  if (buffer->pts == kGstClockTimeNone) {
    return std::nullopt;
  }
  const void* segment = api->sample_get_segment(sample);
  if (segment == nullptr) {
    return std::nullopt;
  }
  const auto stream_time = api->segment_to_stream_time(segment, kGstFormatTime, buffer->pts);
  return stream_time == kGstClockTimeNone ? std::nullopt
                                          : std::optional<std::uint64_t>(stream_time);
}

enum class TimingWindowStatus { Unavailable, Constant, Nonconstant };

struct TimingWindowEvidence {
  TimingWindowStatus status = TimingWindowStatus::Unavailable;
  std::optional<InferredFrameRate> frame_rate;
  std::optional<std::pair<std::uint32_t, std::uint32_t>> buffer_duration_frame_rate;
};

TimingWindowEvidence infer_timing_window(const std::shared_ptr<ProbeApi>& api, void* pipeline,
                                         void* probe_sink, void* bus, std::uint64_t start_ns,
                                         std::optional<std::uint64_t> end_ns, bool require_eos,
                                         CompressedSampleBudget& sample_budget,
                                         std::chrono::steady_clock::time_point deadline,
                                         const ParserCapsIdentity& expected_caps) {
  const auto seek_ns =
      start_ns > kTimingWindowSeekPrerollNs ? start_ns - kTimingWindowSeekPrerollNs : 0;
  if (seek_ns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      api->element_seek_simple(pipeline, kGstFormatTime, kGstSeekFlagFlushAccurate,
                               static_cast<std::int64_t>(seek_ns)) == 0) {
    return {};
  }

  TimingScan window_scan;
  bool window_started = false;
  bool covered_end = !end_ns.has_value();
  bool window_complete = false;
  std::size_t samples_after_end = 0;
  for (std::size_t sample_index = 0; sample_index < kMaximumTimingWindowSamples; ++sample_index) {
    void* sample = pull_compressed_sample(
        api, probe_sink, bus, sample_budget, deadline,
        require_eos
            ? "GStreamer parser-only probe timed out while sampling terminal stream timing"
            : "GStreamer parser-only probe timed out while sampling interior stream timing");
    std::unique_ptr<void, ProbeApi::SampleUnref> sample_owner(sample, api->sample_unref);
    if (sample == nullptr) {
      window_scan.reached_eos = true;
      break;
    }
    const auto stream_time = sample_presentation_stream_time(api, sample);
    if (!window_started) {
      if (!stream_time.has_value() || *stream_time < start_ns) {
        continue;
      }
      window_started = true;
    }
    observe_timing_sample(api, sample, window_scan, &expected_caps);
    if (!end_ns.has_value()) {
      continue;
    }
    if (!covered_end && stream_time.has_value() && *stream_time >= *end_ns) {
      covered_end = true;
      continue;
    }
    if (covered_end && ++samples_after_end >= kMaximumDecodeReorderFrames) {
      window_complete = true;
      break;
    }
  }
  if (!window_scan.reached_eos && !window_complete) {
    throw GpuVideoProbeError(
        "video parser exceeded the compressed access-unit timing-window work limit");
  }
  if (!window_started || (require_eos && !window_scan.reached_eos) ||
      (end_ns.has_value() && !covered_end)) {
    return {};
  }
  const auto buffer_duration_frame_rate =
      window_scan.all_buffer_duration_rates_known && window_scan.buffer_duration_rates_consistent
          ? window_scan.buffer_duration_frame_rate
          : std::nullopt;
  auto frame_rate = infer_constant_frame_rate(window_scan, require_eos, true, !require_eos);
  if (frame_rate.has_value()) {
    const auto maximum_start_delay =
        minimum_duration_for_frame_count(1, frame_rate->numerator, frame_rate->denominator);
    if (!window_scan.first_stream_time.has_value() ||
        *window_scan.first_stream_time > add_saturating(start_ns, maximum_start_delay)) {
      return {.status = window_scan.sample_count >= 3 &&
                                window_scan.timed_sample_count == window_scan.sample_count
                            ? TimingWindowStatus::Nonconstant
                            : TimingWindowStatus::Unavailable,
              .frame_rate = std::nullopt,
              .buffer_duration_frame_rate = buffer_duration_frame_rate};
    }
    return {.status = TimingWindowStatus::Constant,
            .frame_rate = std::move(frame_rate),
            .buffer_duration_frame_rate = buffer_duration_frame_rate};
  }
  return {
      .status = window_scan.sample_count >= kTimingAnalysisSamples &&
                        window_scan.timed_sample_count == window_scan.sample_count
                    ? TimingWindowStatus::Nonconstant
                    : TimingWindowStatus::Unavailable,
      .frame_rate = std::nullopt,
      .buffer_duration_frame_rate = buffer_duration_frame_rate,
  };
}

struct FrameSeekResult {
  bool usable = true;
  bool available = false;
  std::uint64_t pts_ns = 0;
  std::uint64_t duration_ns = 0;
};

struct SelectedStreamProbe {
  std::uint64_t duration_ns = 0;
  std::uint64_t frame_count = 0;
};

FrameSeekResult seek_compressed_frame(const std::shared_ptr<ProbeApi>& api, void* pipeline,
                                      void* probe_sink, void* bus, std::uint64_t stream_origin_ns,
                                      std::uint64_t frame_index, std::uint32_t fps_numerator,
                                      std::uint32_t fps_denominator,
                                      std::size_t timestamp_multiplicity,
                                      CompressedSampleBudget& sample_budget,
                                      std::chrono::steady_clock::time_point deadline,
                                      const ParserCapsIdentity& expected_caps) {
  const auto relative_target_ns = frame_timestamp_ns(frame_index, fps_numerator, fps_denominator);
  if (relative_target_ns > std::numeric_limits<std::uint64_t>::max() - stream_origin_ns) {
    return {.usable = false};
  }
  const auto target_ns = stream_origin_ns + relative_target_ns;
  const auto period_numerator = kNanosecondsPerSecond * fps_denominator;
  const auto half_frame_ns = period_numerator / (2ULL * fps_numerator);
  const auto nominal_frame_ns = period_numerator / fps_numerator;
  const auto duplicate_span_ns =
      timestamp_multiplicity - 1 > std::numeric_limits<std::uint64_t>::max() / nominal_frame_ns
          ? std::numeric_limits<std::uint64_t>::max()
          : nominal_frame_ns * (timestamp_multiplicity - 1);
  const auto matching_tolerance_ns = add_saturating(duplicate_span_ns, half_frame_ns);
  const auto earliest_matching_pts =
      target_ns > matching_tolerance_ns ? target_ns - matching_tolerance_ns : 0;
  if (earliest_matching_pts >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      api->element_seek_simple(pipeline, kGstFormatTime, kGstSeekFlagFlushAccurate,
                               static_cast<std::int64_t>(earliest_matching_pts)) == 0) {
    return {.usable = false};
  }

  const auto nominal_duration_ns = std::max<std::uint64_t>(period_numerator / fps_numerator, 1);
  const auto maximum_decode_reorder_ns =
      nominal_duration_ns > std::numeric_limits<std::uint64_t>::max() / kMaximumDecodeReorderFrames
          ? std::numeric_limits<std::uint64_t>::max()
          : nominal_duration_ns * kMaximumDecodeReorderFrames;
  for (std::size_t sample_index = 0; sample_index < kMaximumTimingWindowSamples; ++sample_index) {
    void* sample =
        pull_compressed_sample(api, probe_sink, bus, sample_budget, deadline,
                               "GStreamer parser-only probe timed out while seeking stream end");
    std::unique_ptr<void, ProbeApi::SampleUnref> sample_owner(sample, api->sample_unref);
    if (sample == nullptr) {
      return {.usable = true, .available = false};
    }
    validate_parser_sample_caps(api, sample, expected_caps);

    const auto* buffer = static_cast<const GstBufferAbi*>(api->sample_get_buffer(sample));
    if (buffer == nullptr) {
      throw GpuVideoProbeError("GStreamer parser-only probe returned a sample without a buffer");
    }
    const bool has_presentation_time = buffer->pts != kGstClockTimeNone;
    const auto timestamp = has_presentation_time ? buffer->pts : buffer->dts;
    if (timestamp == kGstClockTimeNone) {
      return {.usable = false};
    }
    const void* segment = api->sample_get_segment(sample);
    if (segment == nullptr) {
      return {.usable = false};
    }
    const auto stream_time = api->segment_to_stream_time(segment, kGstFormatTime, timestamp);
    if (stream_time == kGstClockTimeNone) {
      continue;
    }
    const auto earliest_matching_timestamp =
        has_presentation_time || earliest_matching_pts <= maximum_decode_reorder_ns
            ? (has_presentation_time ? earliest_matching_pts : 0)
            : earliest_matching_pts - maximum_decode_reorder_ns;
    if (stream_time < earliest_matching_timestamp) {
      continue;
    }
    return {.usable = true,
            .available = true,
            .pts_ns = has_presentation_time ? stream_time : target_ns,
            .duration_ns = buffer->duration == 0 || buffer->duration == kGstClockTimeNone
                               ? nominal_duration_ns
                               : buffer->duration};
  }
  throw GpuVideoProbeError(
      "video parser exceeded the compressed access-unit frame-seek work limit");
}

std::optional<SelectedStreamProbe> selected_stream_duration(
    const std::shared_ptr<ProbeApi>& api, void* pipeline, void* probe_sink, void* bus,
    std::uint64_t container_duration_ns, std::uint64_t stream_origin_ns,
    std::uint32_t fps_numerator, std::uint32_t fps_denominator, std::size_t timestamp_multiplicity,
    CompressedSampleBudget& sample_budget, std::chrono::steady_clock::time_point deadline,
    const ParserCapsIdentity& expected_caps) {
  const auto duration_frame_ceiling =
      frame_count_ceiling_for_duration(container_duration_ns, fps_numerator, fps_denominator);
  if (duration_frame_ceiling == std::numeric_limits<std::uint64_t>::max()) {
    return std::nullopt;
  }
  if (timestamp_multiplicity > std::numeric_limits<std::uint64_t>::max() - duration_frame_ceiling) {
    return std::nullopt;
  }
  const auto initial_search_frame_limit = duration_frame_ceiling + timestamp_multiplicity;

  std::uint64_t first = 0;
  std::uint64_t last = initial_search_frame_limit;
  std::optional<std::pair<std::uint64_t, FrameSeekResult>> final_available_frame;
  while (true) {
    const auto result = seek_compressed_frame(
        api, pipeline, probe_sink, bus, stream_origin_ns, last, fps_numerator, fps_denominator,
        timestamp_multiplicity, sample_budget, deadline, expected_caps);
    if (!result.usable) {
      return std::nullopt;
    }
    if (!result.available) {
      break;
    }
    final_available_frame = std::pair(last, result);
    if (last == std::numeric_limits<std::uint64_t>::max()) {
      return std::nullopt;
    }
    first = last + 1U;
    const auto expanded = last > std::numeric_limits<std::uint64_t>::max() / 2U
                              ? std::numeric_limits<std::uint64_t>::max()
                              : last * 2U;
    last = std::max(expanded, first);
  }
  while (first < last) {
    const auto candidate = first + (last - first) / 2;
    const auto result = seek_compressed_frame(
        api, pipeline, probe_sink, bus, stream_origin_ns, candidate, fps_numerator, fps_denominator,
        timestamp_multiplicity, sample_budget, deadline, expected_caps);
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
  const auto maximum_count = add_saturating(first, timestamp_multiplicity);
  const auto maximum_stream_span_ns =
      minimum_duration_for_frame_count(maximum_count, fps_numerator, fps_denominator);
  auto stream_duration_ns = std::min(frame_end - stream_origin_ns, maximum_stream_span_ns);
  const auto minimum_boundary_duration =
      minimum_duration_for_frame_count(first, fps_numerator, fps_denominator);
  const auto nominal_frame_duration =
      std::max<std::uint64_t>(kNanosecondsPerSecond * fps_denominator / fps_numerator, 1);
  if (final_frame.duration_ns >= nominal_frame_duration) {
    stream_duration_ns =
        std::min(std::max(stream_duration_ns, minimum_boundary_duration), maximum_stream_span_ns);
  }
  return SelectedStreamProbe{.duration_ns = stream_duration_ns, .frame_count = first};
}

} // namespace

GpuVideoProbe detail::probe_gpu_video_in_process(const GpuFileDecodeConfig& config,
                                                 std::uint64_t timeout_ns) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  if (timeout_ns < kMinimumProbeTimeoutNs || timeout_ns > kMaximumProbeTimeoutNs) {
    throw std::invalid_argument("video probe timeout must be between one second and one hour");
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
  CompressedSampleBudget sample_budget;

  std::error_code path_error;
  const auto absolute_path = std::filesystem::absolute(path_from_utf8(config.path), path_error);
  if (path_error) {
    throw GpuVideoProbeError("failed to resolve video path: " + path_error.message());
  }
  if (!std::filesystem::is_regular_file(absolute_path, path_error) || path_error) {
    throw GpuVideoProbeError("video probe path is not a readable regular file: " +
                             path_for_gstreamer(absolute_path));
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
  void* container_info =
      config.elementary_stream ? nullptr : api->bin_get_by_name(pipeline, "container_info");
  std::unique_ptr<void, ProbeApi::ObjectUnref> container_info_owner(container_info,
                                                                    api->object_unref);
  if (!config.elementary_stream && container_info == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no container metadata identity");
  }
  void* container_pad =
      container_info == nullptr ? nullptr : api->element_get_static_pad(container_info, "src");
  std::unique_ptr<void, ProbeApi::ObjectUnref> container_pad_owner(container_pad,
                                                                   api->object_unref);
  if (container_info != nullptr && container_pad == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no container metadata pad");
  }
  void* probe_sink = api->bin_get_by_name(pipeline, "probe_sink");
  std::unique_ptr<void, ProbeApi::ObjectUnref> probe_sink_owner(probe_sink, api->object_unref);
  if (probe_sink == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no compressed-stream sink");
  }
  void* input_budget_element = api->bin_get_by_name(pipeline, "input_budget");
  std::unique_ptr<void, ProbeApi::ObjectUnref> input_budget_owner(input_budget_element,
                                                                  api->object_unref);
  if (input_budget_element == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no compressed-input budget");
  }
  void* input_budget_pad = api->element_get_static_pad(input_budget_element, "src");
  std::unique_ptr<void, ProbeApi::ObjectUnref> input_budget_pad_owner(input_budget_pad,
                                                                      api->object_unref);
  if (input_budget_pad == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe has no compressed-input budget pad");
  }
  InputBudgetCallbackData input_budget_callback{.api = api.get(), .budget = &sample_budget};
  const auto input_budget_probe_id =
      api->pad_add_probe(input_budget_pad, kGstPadProbeTypeBuffer, &input_budget_probe,
                         &input_budget_callback, nullptr);
  if (input_budget_probe_id == 0) {
    throw GpuVideoProbeError("failed to attach the compressed-input byte budget");
  }
  struct RemoveInputBudgetProbe {
    std::shared_ptr<ProbeApi> api;
    void* pad = nullptr;
    unsigned long probe_id = 0;
    ~RemoveInputBudgetProbe() { api->pad_remove_probe(pad, probe_id); }
  } remove_input_budget_probe{api, input_budget_pad, input_budget_probe_id};
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
  initial_sample =
      pull_compressed_sample(api, probe_sink, bus, sample_budget, deadline,
                             "GStreamer parser-only probe timed out while parsing stream metadata");
  initial_sample_owner.reset(initial_sample);
  if (initial_sample == nullptr) {
    throw GpuVideoProbeError(
        "video discovery found no H.264 or HEVC moving-video stream compatible with NVDEC");
  }
  if (api->sample_get_buffer(initial_sample) == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe returned a sample without a buffer");
  }

  std::optional<std::pair<int, int>> container_frame_rate;
  if (container_pad != nullptr) {
    void* container_caps = api->pad_get_current_caps(container_pad);
    std::unique_ptr<void, ProbeApi::CapsUnref> container_caps_owner(container_caps,
                                                                    api->caps_unref);
    if (container_caps != nullptr) {
      void* container_structure = api->caps_get_structure(container_caps, 0);
      int container_fps_numerator = 0;
      int container_fps_denominator = 0;
      if (container_structure != nullptr &&
          api->structure_get_fraction(container_structure, "framerate", &container_fps_numerator,
                                      &container_fps_denominator) != 0 &&
          container_fps_numerator > 0 && container_fps_denominator > 0) {
        container_frame_rate = {container_fps_numerator, container_fps_denominator};
      }
    }
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
  const bool has_caps_frame_rate =
      api->structure_get_fraction(structure, "framerate", &fps_numerator, &fps_denominator) != 0 &&
      fps_numerator > 0 && fps_denominator > 0;
  const double caps_fps =
      has_caps_frame_rate ? static_cast<double>(fps_numerator) / fps_denominator : 0.0;
  const bool parser_caps_frame_rate_is_plausible =
      std::isfinite(caps_fps) && caps_fps > 0.0 && caps_fps <= kMaximumSaneFps;
  const auto parser_caps_frame_rate =
      parser_caps_frame_rate_is_plausible
          ? std::optional(std::pair(static_cast<std::uint32_t>(fps_numerator),
                                    static_cast<std::uint32_t>(fps_denominator)))
          : std::nullopt;
  const ParserCapsIdentity expected_caps{
      .codec = codec_caps,
      .stream_format = stream_format,
      .alignment = alignment,
      .width = width,
      .height = height,
      .frame_rate = has_caps_frame_rate ? std::optional(std::pair(fps_numerator, fps_denominator))
                                        : std::nullopt,
  };
  TimingScan timing_scan;
  observe_timing_sample(api, initial_sample, timing_scan, &expected_caps);
  initial_sample_owner.reset();
  while (timing_scan.sample_count <= kMaximumExactCountSamples) {
    void* sample = pull_compressed_sample(
        api, probe_sink, bus, sample_budget, deadline,
        "GStreamer parser-only probe timed out while counting compressed stream frames");
    initial_sample_owner.reset(sample);
    if (sample == nullptr) {
      timing_scan.reached_eos = true;
      break;
    }
    observe_timing_sample(api, sample, timing_scan, &expected_caps);
  }
  std::optional<std::pair<std::uint32_t, std::uint32_t>> matroska_duration_frame_rate;
  if (!config.elementary_stream && config.container == GpuDecodeContainer::Matroska &&
      timing_scan.all_buffer_duration_rates_known && timing_scan.buffer_duration_rates_consistent) {
    matroska_duration_frame_rate = timing_scan.buffer_duration_frame_rate;
    if (matroska_duration_frame_rate.has_value()) {
      container_frame_rate = {
          static_cast<int>(matroska_duration_frame_rate->first),
          static_cast<int>(matroska_duration_frame_rate->second),
      };
    }
  }
  bool container_frame_rate_is_plausible = false;
  if (container_frame_rate.has_value()) {
    const auto container_fps = static_cast<double>(container_frame_rate->first) /
                               static_cast<double>(container_frame_rate->second);
    if (std::isfinite(container_fps) && container_fps > 0.0 && container_fps <= kMaximumSaneFps) {
      container_frame_rate_is_plausible = true;
      fps_numerator = container_frame_rate->first;
      fps_denominator = container_frame_rate->second;
    }
  }
  const bool caps_frame_rate_is_plausible =
      container_frame_rate_is_plausible || parser_caps_frame_rate_is_plausible;
  std::optional<std::uint64_t> queried_container_duration;
  if (!timing_scan.reached_eos) {
    std::int64_t queried_duration = 0;
    if (api->element_query_duration(pipeline, kGstFormatTime, &queried_duration) != 0 &&
        queried_duration > 0) {
      queried_container_duration = static_cast<std::uint64_t>(queried_duration);
    }
  }
  const bool caps_rate_cannot_cover_observed_samples =
      queried_container_duration.has_value() && caps_frame_rate_is_plausible &&
      add_saturating(frame_count_ceiling_for_duration(*queried_container_duration,
                                                      static_cast<std::uint32_t>(fps_numerator),
                                                      static_cast<std::uint32_t>(fps_denominator)),
                     kTimingReorderLookahead) < timing_scan.sample_count;
  const auto preliminary_frame_rate = infer_constant_frame_rate(
      timing_scan, timing_scan.reached_eos, timing_scan.reached_eos, !timing_scan.reached_eos);
  const bool caps_rate_conflict_needs_more_evidence =
      preliminary_frame_rate.has_value() && caps_frame_rate_is_plausible &&
      !frame_rates_are_identical(
          preliminary_frame_rate->numerator, preliminary_frame_rate->denominator,
          static_cast<std::uint32_t>(fps_numerator), static_cast<std::uint32_t>(fps_denominator));
  const bool incomplete_noncanonical_rate_needs_more_evidence =
      preliminary_frame_rate.has_value() && preliminary_frame_rate->requires_complete_stream;
  if (!timing_scan.reached_eos &&
      (caps_rate_cannot_cover_observed_samples || caps_rate_conflict_needs_more_evidence ||
       incomplete_noncanonical_rate_needs_more_evidence)) {
    while (timing_scan.sample_count <= kMaximumConflictingMetadataCountSamples) {
      void* sample = pull_compressed_sample(
          api, probe_sink, bus, sample_budget, deadline,
          "GStreamer parser-only probe timed out while resolving conflicting metadata");
      initial_sample_owner.reset(sample);
      if (sample == nullptr) {
        timing_scan.reached_eos = true;
        break;
      }
      observe_timing_sample(api, sample, timing_scan, &expected_caps);
    }
  }
  if (matroska_duration_frame_rate.has_value() &&
      (!timing_scan.all_buffer_duration_rates_known ||
       !timing_scan.buffer_duration_rates_consistent ||
       timing_scan.buffer_duration_frame_rate != matroska_duration_frame_rate)) {
    throw GpuVideoProbeError(
        "variable frame-rate video is unsupported for indexed GPU calibration sampling "
        "(Matroska compressed-buffer durations disagree)");
  }
  const auto inferred_frame_rate = infer_constant_frame_rate(
      timing_scan, timing_scan.reached_eos, timing_scan.reached_eos, !timing_scan.reached_eos);
  if (inferred_frame_rate.has_value() &&
      static_cast<long double>(inferred_frame_rate->numerator) / inferred_frame_rate->denominator >
          kMaximumSaneFps) {
    throw GpuVideoProbeError("video parser returned an implausible frame rate");
  }
  const bool prefix_has_dense_nonconstant_timing =
      !inferred_frame_rate.has_value() &&
      timing_scan.timed_sample_count == timing_scan.sample_count && timing_scan.sample_count >= 3;
  if (prefix_has_dense_nonconstant_timing) {
    throw GpuVideoProbeError(
        "variable frame-rate video is unsupported for indexed GPU calibration sampling "
        "(dense prefix timing is nonconstant)");
  }

  std::vector<CadenceValidationCandidate> cadence_candidates;
  const auto add_cadence_candidate = [&](std::uint32_t numerator, std::uint32_t denominator) {
    if (std::none_of(cadence_candidates.begin(), cadence_candidates.end(),
                     [numerator, denominator](const auto& candidate) {
                       return frame_rates_are_identical(candidate.rate.first, candidate.rate.second,
                                                        numerator, denominator);
                     })) {
      cadence_candidates.push_back({.rate = {numerator, denominator},
                                    .validator = DenseCadenceValidator(numerator, denominator),
                                    .failure = std::nullopt});
    }
  };
  if (inferred_frame_rate.has_value()) {
    add_cadence_candidate(inferred_frame_rate->numerator, inferred_frame_rate->denominator);
  }
  if (caps_frame_rate_is_plausible) {
    add_cadence_candidate(static_cast<std::uint32_t>(fps_numerator),
                          static_cast<std::uint32_t>(fps_denominator));
  }
  if (parser_caps_frame_rate.has_value()) {
    add_cadence_candidate(parser_caps_frame_rate->first, parser_caps_frame_rate->second);
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> cadence_verified_rates;
  std::optional<std::uint64_t> full_cadence_sample_count;
  const bool eager_cadence_validation_is_bounded =
      timing_scan.reached_eos || !queried_container_duration.has_value() ||
      std::any_of(cadence_candidates.begin(), cadence_candidates.end(), [&](const auto& candidate) {
        return frame_count_ceiling_for_duration(*queried_container_duration, candidate.rate.first,
                                                candidate.rate.second) <=
               kMaximumEagerCadenceValidationSamples;
      });
  if (!cadence_candidates.empty() && timing_scan.timed_sample_count == timing_scan.sample_count &&
      eager_cadence_validation_is_bounded) {
    const auto validate_cadence_sample = [&](std::uint64_t stream_time,
                                             std::uint64_t sample_index) {
      for (auto& candidate : cadence_candidates) {
        if (candidate.failure.has_value()) {
          continue;
        }
        try {
          candidate.validator.observe(stream_time, sample_index);
        } catch (const GpuVideoProbeError& error) {
          candidate.failure = error.what();
        }
      }
      const auto first_live_candidate =
          std::find_if(cadence_candidates.begin(), cadence_candidates.end(),
                       [](const auto& candidate) { return !candidate.failure.has_value(); });
      if (first_live_candidate == cadence_candidates.end()) {
        throw GpuVideoProbeError(*cadence_candidates.front().failure);
      }
    };
    for (std::size_t index = 0; index < timing_scan.stored_timing_count; ++index) {
      validate_cadence_sample(timing_scan.stream_times[index],
                              timing_scan.timed_sample_indices[index]);
    }
    auto validation_sample_count = timing_scan.sample_count;
    if (!timing_scan.reached_eos) {
      while (validation_sample_count < kMaximumEagerCadenceValidationSamples) {
        void* sample = pull_compressed_sample(
            api, probe_sink, bus, sample_budget, deadline,
            "GStreamer parser-only probe timed out while verifying full-stream frame cadence");
        initial_sample_owner.reset(sample);
        if (sample == nullptr) {
          timing_scan.reached_eos = true;
          break;
        }
        observe_timing_sample(api, sample, timing_scan, &expected_caps);
        const auto stream_time = sample_presentation_stream_time(api, sample);
        if (!stream_time.has_value()) {
          throw GpuVideoProbeError(
              "variable frame-rate video is unsupported for indexed GPU calibration sampling "
              "(full-stream cadence becomes untimed)");
        }
        validate_cadence_sample(*stream_time, validation_sample_count++);
      }
    }
    if (timing_scan.reached_eos && validation_sample_count >= 3) {
      if (matroska_duration_frame_rate.has_value() &&
          (!timing_scan.all_buffer_duration_rates_known ||
           !timing_scan.buffer_duration_rates_consistent ||
           timing_scan.buffer_duration_frame_rate != matroska_duration_frame_rate)) {
        throw GpuVideoProbeError(
            "variable frame-rate video is unsupported for indexed GPU calibration sampling "
            "(Matroska compressed-buffer durations disagree)");
      }
      for (auto& candidate : cadence_candidates) {
        if (!candidate.failure.has_value()) {
          try {
            candidate.validator.finish();
          } catch (const GpuVideoProbeError& error) {
            candidate.failure = error.what();
          }
        }
        if (!candidate.failure.has_value()) {
          cadence_verified_rates.push_back(candidate.rate);
        }
      }
      if (cadence_verified_rates.empty()) {
        const auto failure =
            std::find_if(cadence_candidates.begin(), cadence_candidates.end(),
                         [](const auto& candidate) { return candidate.failure.has_value(); });
        throw GpuVideoProbeError(
            failure == cadence_candidates.end()
                ? "video parser could not verify constant full-stream frame timing"
                : *failure->failure);
      }
      full_cadence_sample_count = validation_sample_count;
    }
  }
  const auto cadence_rate_is_verified = [&](std::uint32_t numerator, std::uint32_t denominator) {
    return std::any_of(cadence_verified_rates.begin(), cadence_verified_rates.end(),
                       [numerator, denominator](const auto& rate) {
                         return frame_rates_are_identical(rate.first, rate.second, numerator,
                                                          denominator);
                       });
  };

  TimingWindowEvidence tail_timing;
  bool selected_stream_probe_attempted = false;
  std::optional<SelectedStreamProbe> correlated_selected_stream;
  std::uint32_t correlated_fps_numerator = 0;
  std::uint32_t correlated_fps_denominator = 0;
  std::size_t correlated_timestamp_multiplicity = 0;
  if (!config.elementary_stream && !timing_scan.reached_eos &&
      queried_container_duration.has_value()) {
    std::optional<std::pair<std::uint32_t, std::uint32_t>> representative_rate;
    if (inferred_frame_rate.has_value()) {
      representative_rate =
          std::pair(inferred_frame_rate->numerator, inferred_frame_rate->denominator);
    } else if (caps_frame_rate_is_plausible) {
      representative_rate = std::pair(static_cast<std::uint32_t>(fps_numerator),
                                      static_cast<std::uint32_t>(fps_denominator));
    }
    auto timing_range_start_ns = std::uint64_t{0};
    auto timing_range_end_ns = *queried_container_duration;
    if (representative_rate.has_value() && timing_scan.first_stream_time.has_value()) {
      correlated_fps_numerator = representative_rate->first;
      correlated_fps_denominator = representative_rate->second;
      correlated_timestamp_multiplicity =
          inferred_frame_rate.has_value() &&
                  frame_rates_are_identical(inferred_frame_rate->numerator,
                                            inferred_frame_rate->denominator,
                                            correlated_fps_numerator, correlated_fps_denominator)
              ? inferred_frame_rate->timestamp_multiplicity
              : 1U;
      selected_stream_probe_attempted = true;
      correlated_selected_stream = selected_stream_duration(
          api, pipeline, probe_sink, bus, *queried_container_duration,
          *timing_scan.first_stream_time, correlated_fps_numerator, correlated_fps_denominator,
          correlated_timestamp_multiplicity, sample_budget, deadline, expected_caps);
      if (correlated_selected_stream.has_value()) {
        timing_range_start_ns = *timing_scan.first_stream_time;
        timing_range_end_ns =
            add_saturating(timing_range_start_ns, correlated_selected_stream->duration_ns);
      }
    }
    if (timing_range_end_ns <= timing_range_start_ns) {
      throw GpuVideoProbeError("video parser returned an invalid selected stream timing range");
    }
    const auto timing_range_duration_ns = timing_range_end_ns - timing_range_start_ns;
    const auto validate_window = [&](const TimingWindowEvidence& evidence,
                                     std::uint64_t window_start_ns) {
      if (evidence.status != TimingWindowStatus::Constant || !evidence.frame_rate.has_value()) {
        throw GpuVideoProbeError(
            evidence.status == TimingWindowStatus::Nonconstant
                ? "variable frame-rate video is unsupported for indexed GPU calibration sampling "
                  "(timing window starting at " +
                      std::to_string(window_start_ns) + " ns is nonconstant)"
                : "video parser could not verify constant frame timing across the selected stream");
      }
      if (matroska_duration_frame_rate.has_value() &&
          evidence.buffer_duration_frame_rate != matroska_duration_frame_rate) {
        throw GpuVideoProbeError(
            "variable frame-rate video is unsupported for indexed GPU calibration sampling "
            "(Matroska compressed-buffer durations disagree)");
      }
      if (representative_rate.has_value() &&
          !frame_rates_are_close(representative_rate->first, representative_rate->second,
                                 evidence.frame_rate->numerator, evidence.frame_rate->denominator,
                                 kCanonicalFrameRateTolerance)) {
        throw GpuVideoProbeError(
            "variable frame-rate video is unsupported for indexed GPU calibration sampling "
            "(representative rates " +
            std::to_string(representative_rate->first) + "/" +
            std::to_string(representative_rate->second) + " and " +
            std::to_string(evidence.frame_rate->numerator) + "/" +
            std::to_string(evidence.frame_rate->denominator) + " disagree)");
      }
      representative_rate =
          std::pair(evidence.frame_rate->numerator, evidence.frame_rate->denominator);
    };

    constexpr std::array<std::uint64_t, 3> kInteriorWindowQuarterPositions{1, 2, 3};
    constexpr auto kHalfTimingWindowNs = kTailTimingWindowNs / 2U;
    for (const auto quarter : kInteriorWindowQuarterPositions) {
      const auto center_ns = add_saturating(
          timing_range_start_ns,
          multiply_divide_floor_saturating(timing_range_duration_ns, quarter,
                                           kInteriorWindowQuarterPositions.size() + 1U));
      const auto start_ns = center_ns > add_saturating(timing_range_start_ns, kHalfTimingWindowNs)
                                ? center_ns - kHalfTimingWindowNs
                                : timing_range_start_ns;
      const auto end_ns =
          std::min(add_saturating(start_ns, kTailTimingWindowNs), timing_range_end_ns);
      validate_window(infer_timing_window(api, pipeline, probe_sink, bus, start_ns, end_ns, false,
                                          sample_budget, deadline, expected_caps),
                      start_ns);
    }

    const auto tail_start_ns = timing_range_duration_ns > kTailTimingWindowNs
                                   ? timing_range_end_ns - kTailTimingWindowNs
                                   : timing_range_start_ns;
    tail_timing = infer_timing_window(api, pipeline, probe_sink, bus, tail_start_ns, std::nullopt,
                                      true, sample_budget, deadline, expected_caps);
    validate_window(tail_timing, tail_start_ns);
  }
  const bool representative_tail_matches =
      inferred_frame_rate.has_value() && tail_timing.frame_rate.has_value() &&
      frame_rates_are_close(inferred_frame_rate->numerator, inferred_frame_rate->denominator,
                            tail_timing.frame_rate->numerator, tail_timing.frame_rate->denominator,
                            kCanonicalFrameRateTolerance);
  if (inferred_frame_rate.has_value() && tail_timing.frame_rate.has_value() &&
      !representative_tail_matches) {
    throw GpuVideoProbeError(
        "variable frame-rate video is unsupported for indexed GPU calibration sampling");
  }
  const bool bounded_noncanonical_rate_is_conclusive =
      inferred_frame_rate.has_value() && inferred_frame_rate->requires_complete_stream &&
      !timing_scan.reached_eos && !caps_frame_rate_is_plausible &&
      timing_scan.sample_count > kMaximumConflictingMetadataCountSamples;
  const bool inferred_frame_rate_has_full_cadence_proof =
      inferred_frame_rate.has_value() &&
      cadence_rate_is_verified(inferred_frame_rate->numerator, inferred_frame_rate->denominator);
  const bool caps_frame_rate_has_full_cadence_proof =
      caps_frame_rate_is_plausible &&
      cadence_rate_is_verified(static_cast<std::uint32_t>(fps_numerator),
                               static_cast<std::uint32_t>(fps_denominator));
  const bool full_cadence_validation_completed = !cadence_verified_rates.empty();
  const bool inferred_frame_rate_is_no_worse_than_caps =
      inferred_frame_rate.has_value() &&
      (!full_cadence_validation_completed || inferred_frame_rate_has_full_cadence_proof) &&
      inferred_frame_rate_is_eligible(*inferred_frame_rate, timing_scan.reached_eos,
                                      representative_tail_matches ||
                                          bounded_noncanonical_rate_is_conclusive ||
                                          inferred_frame_rate_has_full_cadence_proof) &&
      (!caps_frame_rate_is_plausible || timing_scan.reached_eos || representative_tail_matches ||
       inferred_frame_rate_has_full_cadence_proof) &&
      (!caps_frame_rate_has_full_cadence_proof ||
       inferred_frame_rate_improves_on_caps(*inferred_frame_rate, caps_frame_rate_is_plausible,
                                            fps_numerator, fps_denominator));
  const bool prefer_inferred_frame_rate = inferred_frame_rate_is_no_worse_than_caps;
  const bool inferred_frame_rate_replaces_caps =
      prefer_inferred_frame_rate &&
      (!caps_frame_rate_is_plausible ||
       !frame_rates_are_identical(inferred_frame_rate->numerator, inferred_frame_rate->denominator,
                                  static_cast<std::uint32_t>(fps_numerator),
                                  static_cast<std::uint32_t>(fps_denominator)));
  if (prefer_inferred_frame_rate) {
    fps_numerator = static_cast<int>(inferred_frame_rate->numerator);
    fps_denominator = static_cast<int>(inferred_frame_rate->denominator);
  } else if (!caps_frame_rate_is_plausible) {
    fps_numerator = static_cast<int>(kFallbackFpsNumerator);
    fps_denominator = static_cast<int>(kFallbackFpsDenominator);
  }
  initial_sample_owner.reset();

  const double fps = static_cast<double>(fps_numerator) / fps_denominator;
  if (!std::isfinite(fps) || fps <= 0.0 || fps > kMaximumSaneFps) {
    throw GpuVideoProbeError("video parser returned an implausible frame rate");
  }

  const auto fps_num = static_cast<std::uint32_t>(fps_numerator);
  const auto fps_den = static_cast<std::uint32_t>(fps_denominator);
  const bool selected_stream_caps_verified = timing_scan.reached_eos;
  const bool indexed_sampling_cadence_verified = cadence_rate_is_verified(fps_num, fps_den);
  const auto timestamp_multiplicity =
      inferred_frame_rate.has_value() &&
              frame_rates_are_close(fps_num, fps_den, inferred_frame_rate->numerator,
                                    inferred_frame_rate->denominator, kFrameRatePreferenceTolerance)
          ? inferred_frame_rate->timestamp_multiplicity
          : 1U;
  std::uint64_t duration_ns = 0;
  std::uint64_t total_frames = 0;
  std::optional<std::uint64_t> correlated_frame_count;
  bool duration_is_estimated = false;
  bool total_frames_is_estimated = false;
  if (timing_scan.reached_eos) {
    total_frames = timing_scan.sample_count;
    const bool complete_timestamps = timing_scan.timed_sample_count == timing_scan.sample_count &&
                                     timing_scan.first_stream_time.has_value() &&
                                     timing_scan.last_stream_time.has_value();
    const auto nominal_frame_duration =
        std::max<std::uint64_t>(kNanosecondsPerSecond * fps_den / fps_num, 1);
    const auto inferred_final_frame_duration =
        minimum_duration_for_frame_count(1, fps_num, fps_den);
    if (complete_timestamps && !inferred_frame_rate_replaces_caps &&
        timing_scan.all_durations_known && timing_scan.final_frame_end.has_value() &&
        *timing_scan.final_frame_end >= *timing_scan.first_stream_time) {
      duration_ns = *timing_scan.final_frame_end - *timing_scan.first_stream_time;
      const auto minimum_boundary_duration =
          minimum_duration_for_frame_count(total_frames, fps_num, fps_den);
      if (timing_scan.final_frame_duration.has_value() &&
          *timing_scan.final_frame_duration >= nominal_frame_duration &&
          minimum_boundary_duration > duration_ns &&
          minimum_boundary_duration - duration_ns <= nominal_frame_duration / 2U) {
        duration_ns = minimum_boundary_duration;
      }
    } else if (complete_timestamps) {
      duration_ns = observed_eos_duration(timing_scan, inferred_final_frame_duration,
                                          inferred_frame_rate_replaces_caps)
                        .value_or(minimum_duration_for_frame_count(total_frames, fps_num, fps_den));
      duration_is_estimated = true;
    } else {
      duration_ns = minimum_duration_for_frame_count(total_frames, fps_num, fps_den);
      duration_is_estimated = true;
    }
    const auto cadence_boundary_duration =
        minimum_duration_for_frame_count(total_frames, fps_num, fps_den);
    if (indexed_sampling_cadence_verified && cadence_boundary_duration > duration_ns &&
        cadence_boundary_duration - duration_ns > nominal_frame_duration) {
      duration_ns = cadence_boundary_duration;
      duration_is_estimated = false;
    }
  } else {
    total_frames_is_estimated = !full_cadence_sample_count.has_value();
    duration_is_estimated = !queried_container_duration.has_value();
    duration_ns = duration_is_estimated ? kFallbackDurationSeconds * kNanosecondsPerSecond
                                        : *queried_container_duration;
    if (!config.elementary_stream && !duration_is_estimated &&
        timing_scan.first_stream_time.has_value()) {
      const auto selected_duration =
          selected_stream_probe_attempted && correlated_fps_numerator == fps_num &&
                  correlated_fps_denominator == fps_den &&
                  correlated_timestamp_multiplicity == timestamp_multiplicity
              ? correlated_selected_stream
              : selected_stream_duration(api, pipeline, probe_sink, bus, duration_ns,
                                         *timing_scan.first_stream_time, fps_num, fps_den,
                                         timestamp_multiplicity, sample_budget, deadline,
                                         expected_caps);
      if (selected_duration.has_value()) {
        const auto untimed_prefix =
            infer_untimed_presentation_prefix(timing_scan, fps_num, fps_den);
        duration_ns = add_saturating(selected_duration->duration_ns, untimed_prefix.duration_ns);
        correlated_frame_count =
            add_saturating(selected_duration->frame_count, untimed_prefix.frame_count);
        duration_is_estimated = true;
      } else {
        duration_is_estimated = true;
      }
    } else if (!config.elementary_stream && !timing_scan.first_stream_time.has_value()) {
      duration_is_estimated = true;
    } else if (config.elementary_stream) {
      duration_is_estimated = true;
    }
    total_frames = frame_count_for_duration(duration_ns, fps_num, fps_den);
    if (!correlated_frame_count.has_value() && timestamp_multiplicity > 1) {
      const auto remainder = total_frames % timestamp_multiplicity;
      const auto missing_group_frames = remainder == 0 ? 0 : timestamp_multiplicity - remainder;
      total_frames = add_saturating(total_frames, missing_group_frames);
      duration_ns =
          std::max(duration_ns, minimum_duration_for_frame_count(total_frames, fps_num, fps_den));
    }
    total_frames =
        full_cadence_sample_count.value_or(correlated_frame_count.value_or(total_frames));
    total_frames = std::max(total_frames, timing_scan.sample_count);
    duration_ns =
        std::max(duration_ns, minimum_duration_for_frame_count(total_frames, fps_num, fps_den));
  }
  return {.width = static_cast<std::uint32_t>(width),
          .height = static_cast<std::uint32_t>(height),
          .fps_numerator = fps_num,
          .fps_denominator = fps_den,
          .fps = fps,
          .duration_ns = duration_ns,
          .total_frames = total_frames,
          .duration_is_estimated = duration_is_estimated,
          .total_frames_is_estimated = total_frames_is_estimated,
          .selected_stream_caps_verified = selected_stream_caps_verified,
          .indexed_sampling_cadence_verified = indexed_sampling_cadence_verified};
}

} // namespace reco::io
