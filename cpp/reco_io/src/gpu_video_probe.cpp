#include "reco/io/gpu_video_probe.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
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

constexpr std::uint64_t kClockTimeNone = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr std::uint64_t kFallbackDurationSeconds = 60;
constexpr int kDiscovererOk = 0;
constexpr int kDiscovererMissingPlugins = 5;

struct GErrorAbi {
  std::uint32_t domain = 0;
  std::int32_t code = 0;
  char* message = nullptr;
};

struct GListAbi {
  void* data = nullptr;
  GListAbi* next = nullptr;
  GListAbi* previous = nullptr;
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

class DiscovererApi {
public:
  using InitCheck = int (*)(int*, char***, GErrorAbi**);
  using Version = void (*)(std::uint32_t*, std::uint32_t*, std::uint32_t*, std::uint32_t*);
  using FilenameToUri = char* (*)(const char*, GErrorAbi**);
  using DiscovererNew = void* (*)(std::uint64_t, GErrorAbi**);
  using DiscoverUri = void* (*)(void*, const char*, GErrorAbi**);
  using InfoGetResult = int (*)(const void*);
  using InfoGetDuration = std::uint64_t (*)(const void*);
  using InfoGetVideoStreams = GListAbi* (*)(void*);
  using VideoInfoGetWidth = std::uint32_t (*)(const void*);
  using VideoInfoGetHeight = std::uint32_t (*)(const void*);
  using VideoInfoGetFramerate = std::uint32_t (*)(const void*);
  using VideoInfoIsImage = int (*)(const void*);
  using StreamInfoListFree = void (*)(GListAbi*);
  using ObjectUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);

  DiscovererApi() {
#if defined(_WIN32)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH", {"gstreamer-1.0-0.dll"}, "GStreamer");
    pbutils =
        load_library("RECO_GSTPBUTILS_DYLIB_PATH", {"gstpbutils-1.0-0.dll"}, "GStreamer PbUtils");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0-0.dll", "glib-2.0-0.dll"}, "GLib");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH", {"libgobject-2.0-0.dll", "gobject-2.0-0.dll"},
                           "GObject");
#elif defined(__APPLE__)
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib"}, "GStreamer");
    pbutils =
        load_library("RECO_GSTPBUTILS_DYLIB_PATH",
                     {"libgstpbutils-1.0.0.dylib", "libgstpbutils-1.0.dylib"}, "GStreamer PbUtils");
    glib =
        load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.0.dylib", "libglib-2.0.dylib"}, "GLib");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH",
                           {"libgobject-2.0.0.dylib", "libgobject-2.0.dylib"}, "GObject");
#else
    core = load_library("RECO_GSTREAMER_DYLIB_PATH",
                        {"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"}, "GStreamer");
    pbutils = load_library("RECO_GSTPBUTILS_DYLIB_PATH",
                           {"libgstpbutils-1.0.so.0", "libgstpbutils-1.0.so"}, "GStreamer PbUtils");
    glib = load_library("RECO_GLIB_DYLIB_PATH", {"libglib-2.0.so.0", "libglib-2.0.so"}, "GLib");
    gobject = load_library("RECO_GOBJECT_DYLIB_PATH", {"libgobject-2.0.so.0", "libgobject-2.0.so"},
                           "GObject");
#endif

    init_check = core->symbol<InitCheck>("gst_init_check");
    version = core->symbol<Version>("gst_version");
    filename_to_uri = core->symbol<FilenameToUri>("gst_filename_to_uri");
    discoverer_new = pbutils->symbol<DiscovererNew>("gst_discoverer_new");
    discover_uri = pbutils->symbol<DiscoverUri>("gst_discoverer_discover_uri");
    info_get_result = pbutils->symbol<InfoGetResult>("gst_discoverer_info_get_result");
    info_get_duration = pbutils->symbol<InfoGetDuration>("gst_discoverer_info_get_duration");
    info_get_video_streams =
        pbutils->symbol<InfoGetVideoStreams>("gst_discoverer_info_get_video_streams");
    video_info_get_width =
        pbutils->symbol<VideoInfoGetWidth>("gst_discoverer_video_info_get_width");
    video_info_get_height =
        pbutils->symbol<VideoInfoGetHeight>("gst_discoverer_video_info_get_height");
    video_info_get_framerate_numerator =
        pbutils->symbol<VideoInfoGetFramerate>("gst_discoverer_video_info_get_framerate_num");
    video_info_get_framerate_denominator =
        pbutils->symbol<VideoInfoGetFramerate>("gst_discoverer_video_info_get_framerate_denom");
    video_info_is_image = pbutils->symbol<VideoInfoIsImage>("gst_discoverer_video_info_is_image");
    stream_info_list_free =
        pbutils->symbol<StreamInfoListFree>("gst_discoverer_stream_info_list_free");
    object_unref = gobject->symbol<ObjectUnref>("g_object_unref");
    error_free = glib->symbol<ErrorFree>("g_error_free");
    free = glib->symbol<Free>("g_free");
  }

  std::shared_ptr<DynamicLibrary> core;
  std::shared_ptr<DynamicLibrary> pbutils;
  std::shared_ptr<DynamicLibrary> glib;
  std::shared_ptr<DynamicLibrary> gobject;
  InitCheck init_check = nullptr;
  Version version = nullptr;
  FilenameToUri filename_to_uri = nullptr;
  DiscovererNew discoverer_new = nullptr;
  DiscoverUri discover_uri = nullptr;
  InfoGetResult info_get_result = nullptr;
  InfoGetDuration info_get_duration = nullptr;
  InfoGetVideoStreams info_get_video_streams = nullptr;
  VideoInfoGetWidth video_info_get_width = nullptr;
  VideoInfoGetHeight video_info_get_height = nullptr;
  VideoInfoGetFramerate video_info_get_framerate_numerator = nullptr;
  VideoInfoGetFramerate video_info_get_framerate_denominator = nullptr;
  VideoInfoIsImage video_info_is_image = nullptr;
  StreamInfoListFree stream_info_list_free = nullptr;
  ObjectUnref object_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
};

std::string take_error(const DiscovererApi& api, GErrorAbi*& error, std::string_view fallback) {
  std::string message(fallback);
  if (error != nullptr) {
    if (error->message != nullptr && error->message[0] != '\0') {
      message = error->message;
    }
    api.error_free(error);
    error = nullptr;
  }
  return message;
}

std::string discoverer_result_name(int result) {
  switch (result) {
  case 1:
    return "URI is invalid";
  case 2:
    return "discovery failed";
  case 3:
    return "discovery timed out";
  case 4:
    return "discoverer is busy";
  case kDiscovererMissingPlugins:
    return "discovery is missing plugins";
  default:
    return "discovery returned result " + std::to_string(result);
  }
}

std::uint64_t frame_count_for_duration(std::uint64_t duration_ns, double fps) {
  const long double frames = static_cast<long double>(duration_ns) * fps / kNanosecondsPerSecond;
  if (frames >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(frames);
}

std::string path_for_gstreamer(const std::filesystem::path& path) {
#if defined(_WIN32)
  const auto utf8 = path.u8string();
  return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
#else
  return path.string();
#endif
}

} // namespace

GpuVideoProbe probe_gpu_video(const GpuFileDecodeConfig& config, std::uint64_t timeout_ns) {
  if (const auto error = validate_gpu_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  if (timeout_ns == 0 || timeout_ns == kClockTimeNone) {
    throw std::invalid_argument("video probe timeout must be finite and non-zero");
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

  auto api = std::make_shared<DiscovererApi>();
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

  const auto gstreamer_path = path_for_gstreamer(absolute_path);
  char* uri = api->filename_to_uri(gstreamer_path.c_str(), &error);
  if (uri == nullptr || error != nullptr) {
    const auto detail = take_error(*api, error, "URI conversion returned no value");
    if (uri != nullptr) {
      api->free(uri);
    }
    throw GpuVideoProbeError("video path URI conversion failed: " + detail);
  }
  std::unique_ptr<char, DiscovererApi::Free> uri_owner(uri, api->free);

  void* discoverer = api->discoverer_new(timeout_ns, &error);
  if (discoverer == nullptr || error != nullptr) {
    const auto detail = take_error(*api, error, "discoverer construction returned no object");
    if (discoverer != nullptr) {
      api->object_unref(discoverer);
    }
    throw GpuVideoProbeError("GStreamer Discoverer initialization failed: " + detail);
  }
  std::unique_ptr<void, DiscovererApi::ObjectUnref> discoverer_owner(discoverer, api->object_unref);

  void* info = api->discover_uri(discoverer, uri, &error);
  const auto discovery_error = take_error(*api, error, "");
  if (info == nullptr) {
    throw GpuVideoProbeError("video discovery failed" +
                             (discovery_error.empty() ? "" : ": " + discovery_error));
  }
  std::unique_ptr<void, DiscovererApi::ObjectUnref> info_owner(info, api->object_unref);

  const int result = api->info_get_result(info);
  if (result != kDiscovererOk && result != kDiscovererMissingPlugins) {
    throw GpuVideoProbeError(discoverer_result_name(result) +
                             (discovery_error.empty() ? "" : ": " + discovery_error));
  }

  GListAbi* streams = api->info_get_video_streams(info);
  std::unique_ptr<GListAbi, DiscovererApi::StreamInfoListFree> streams_owner(
      streams, api->stream_info_list_free);
  const void* selected = nullptr;
  std::size_t stream_count = 0;
  for (auto* node = streams; node != nullptr; node = node->next) {
    if (node->data != nullptr && api->video_info_is_image(node->data) == 0) {
      ++stream_count;
      if (selected == nullptr) {
        selected = node->data;
      }
    }
  }
  if (selected == nullptr) {
    throw GpuVideoProbeError("video discovery found no moving-video stream");
  }

  const auto width = api->video_info_get_width(selected);
  const auto height = api->video_info_get_height(selected);
  const auto fps_numerator = api->video_info_get_framerate_numerator(selected);
  const auto fps_denominator = api->video_info_get_framerate_denominator(selected);
  if (width == 0 || height == 0) {
    throw GpuVideoProbeError("video discovery returned zero frame dimensions");
  }
  if (fps_numerator == 0 || fps_denominator == 0) {
    throw GpuVideoProbeError("video discovery returned an invalid frame rate");
  }
  const double fps = static_cast<double>(fps_numerator) / fps_denominator;
  if (!std::isfinite(fps) || fps <= 0.0) {
    throw GpuVideoProbeError("video discovery returned a non-finite frame rate");
  }

  const auto discovered_duration = api->info_get_duration(info);
  const bool duration_is_estimated =
      discovered_duration == 0 || discovered_duration == kClockTimeNone;
  const auto duration_ns = duration_is_estimated ? kFallbackDurationSeconds * kNanosecondsPerSecond
                                                 : discovered_duration;
  return {.width = width,
          .height = height,
          .fps_numerator = fps_numerator,
          .fps_denominator = fps_denominator,
          .fps = fps,
          .duration_ns = duration_ns,
          .total_frames = frame_count_for_duration(duration_ns, fps),
          .video_stream_count = stream_count,
          .duration_is_estimated = duration_is_estimated,
          .discovery_complete = result == kDiscovererOk};
}

} // namespace reco::io
