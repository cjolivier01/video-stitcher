#include "reco/io/gpu_video_probe.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
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
constexpr int kGstStatePaused = 3;
constexpr int kGstStateChangeFailure = 0;
constexpr int kGstStateChangeSuccess = 1;

struct GErrorAbi {
  std::uint32_t domain = 0;
  std::int32_t code = 0;
  char* message = nullptr;
};

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
  using ElementGetState = int (*)(void*, int*, int*, std::uint64_t);
  using ElementQueryDuration = int (*)(void*, int, std::int64_t*);
  using PadGetCurrentCaps = void* (*)(void*);
  using CapsGetStructure = void* (*)(const void*, std::uint32_t);
  using StructureGetInt = int (*)(const void*, const char*, int*);
  using StructureGetFraction = int (*)(const void*, const char*, int*, int*);
  using ObjectUnref = void (*)(void*);
  using CapsUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);

  ProbeApi() {
#if defined(_WIN32)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH", {"gstreamer-1.0-0.dll"}, "GStreamer");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0-0.dll", "glib-2.0-0.dll"}, "GLib");
#elif defined(__APPLE__)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib"}, "GStreamer");
    glib =
        load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.0.dylib", "libglib-2.0.dylib"}, "GLib");
#else
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"}, "GStreamer");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.so.0", "libglib-2.0.so"}, "GLib");
#endif

    init_check = core->symbol<InitCheck>("gst_init_check");
    version = core->symbol<Version>("gst_version");
    parse_launch = core->symbol<ParseLaunch>("gst_parse_launch");
    bin_get_by_name = core->symbol<BinGetByName>("gst_bin_get_by_name");
    element_get_static_pad = core->symbol<ElementGetStaticPad>("gst_element_get_static_pad");
    element_set_state = core->symbol<ElementSetState>("gst_element_set_state");
    element_get_state = core->symbol<ElementGetState>("gst_element_get_state");
    element_query_duration = core->symbol<ElementQueryDuration>("gst_element_query_duration");
    pad_get_current_caps = core->symbol<PadGetCurrentCaps>("gst_pad_get_current_caps");
    caps_get_structure = core->symbol<CapsGetStructure>("gst_caps_get_structure");
    structure_get_int = core->symbol<StructureGetInt>("gst_structure_get_int");
    structure_get_fraction = core->symbol<StructureGetFraction>("gst_structure_get_fraction");
    object_unref = core->symbol<ObjectUnref>("gst_object_unref");
    caps_unref = core->symbol<CapsUnref>("gst_caps_unref");
    error_free = glib->symbol<ErrorFree>("g_error_free");
  }

  std::shared_ptr<DynamicLibrary> core;
  std::shared_ptr<DynamicLibrary> glib;
  InitCheck init_check = nullptr;
  Version version = nullptr;
  ParseLaunch parse_launch = nullptr;
  BinGetByName bin_get_by_name = nullptr;
  ElementGetStaticPad element_get_static_pad = nullptr;
  ElementSetState element_set_state = nullptr;
  ElementGetState element_get_state = nullptr;
  ElementQueryDuration element_query_duration = nullptr;
  PadGetCurrentCaps pad_get_current_caps = nullptr;
  CapsGetStructure caps_get_structure = nullptr;
  StructureGetInt structure_get_int = nullptr;
  StructureGetFraction structure_get_fraction = nullptr;
  ObjectUnref object_unref = nullptr;
  CapsUnref caps_unref = nullptr;
  ErrorFree error_free = nullptr;
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
  pipeline << " ! identity name=probe_info silent=true"
           << " ! fakesink sync=false enable-last-sample=false";
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

} // namespace

GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config, std::uint64_t timeout_ns) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  if (timeout_ns < kMinimumProbeTimeoutNs || timeout_ns > kMaximumProbeTimeoutNs) {
    throw std::invalid_argument("video probe timeout must be between one second and one hour");
  }

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

  struct ResetPipeline {
    std::shared_ptr<ProbeApi> api;
    void* pipeline = nullptr;
    ~ResetPipeline() { (void)api->element_set_state(pipeline, kGstStateNull); }
  } reset{api, pipeline};
  if (api->element_set_state(pipeline, kGstStatePaused) == kGstStateChangeFailure) {
    throw GpuVideoProbeError("GStreamer parser-only probe failed to enter paused state");
  }

  int state = 0;
  int pending = 0;
  const int state_result = api->element_get_state(pipeline, &state, &pending, timeout_ns);
  if (state_result == kGstStateChangeFailure) {
    throw GpuVideoProbeError("GStreamer parser-only probe failed while parsing stream metadata");
  }
  if (state_result != kGstStateChangeSuccess || state != kGstStatePaused) {
    throw GpuVideoProbeError("GStreamer parser-only probe timed out while parsing stream metadata");
  }

  void* caps = api->pad_get_current_caps(probe_pad);
  std::unique_ptr<void, ProbeApi::CapsUnref> caps_owner(caps, api->caps_unref);
  if (caps == nullptr) {
    throw GpuVideoProbeError(
        "video discovery found no H.264 or HEVC moving-video stream compatible with NVDEC");
  }
  void* structure = api->caps_get_structure(caps, 0);
  if (structure == nullptr) {
    throw GpuVideoProbeError("GStreamer parser-only probe caps contain no structure");
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

  std::int64_t queried_duration = 0;
  const bool duration_is_estimated =
      api->element_query_duration(pipeline, kGstFormatTime, &queried_duration) == 0 ||
      queried_duration <= 0;
  const auto duration_ns = duration_is_estimated ? kFallbackDurationSeconds * kNanosecondsPerSecond
                                                 : static_cast<std::uint64_t>(queried_duration);
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
