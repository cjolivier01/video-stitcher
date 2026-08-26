#include "reco/io/gpu_decode.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace reco::io {
namespace {

constexpr std::uint32_t kGstMapRead = 1;
constexpr int kGstStateNull = 1;
constexpr int kGstStatePlaying = 4;
constexpr int kGstStateChangeFailure = 0;
constexpr std::uint32_t kGstMessageError = 1U << 1U;
constexpr std::uint64_t kGstClockTimeNone = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kSamplePollTimeoutNs = 100'000'000;

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

// GstBuffer and GstMapInfo are public GStreamer 1.x ABI structs. The runtime
// major-version check below prevents these layouts from being used with a
// future incompatible ABI.
static_assert(offsetof(GstBufferAbi, pts) == (sizeof(void*) == 8 ? 72 : 40));
static_assert(sizeof(GstBufferAbi) == (sizeof(void*) == 8 ? 112 : 80));
static_assert(offsetof(GstMapInfoAbi, data) == (sizeof(void*) == 8 ? 16 : 8));
static_assert(offsetof(GstMapInfoAbi, size) == (sizeof(void*) == 8 ? 24 : 12));
static_assert(sizeof(GstMapInfoAbi) == (sizeof(void*) == 8 ? 104 : 52));

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
  using ElementGetBus = void* (*)(void*);
  using ObjectUnref = void (*)(void*);
  using PadGetCurrentCaps = void* (*)(void*);
  using CapsUnref = void (*)(void*);
  using AppSinkTryPullSample = void* (*)(void*, std::uint64_t);
  using AppSinkIsEos = int (*)(void*);
  using SampleGetBuffer = void* (*)(void*);
  using SampleUnref = void (*)(void*);
  using CapsGetStructure = void* (*)(const void*, std::uint32_t);
  using StructureGetInt = int (*)(const void*, const char*, int*);
  using BufferMap = int (*)(void*, GstMapInfoAbi*, std::uint32_t);
  using BufferUnmap = void (*)(void*, GstMapInfoAbi*);
  using BusTimedPopFiltered = void* (*)(void*, std::uint64_t, std::uint32_t);
  using MessageParseError = void (*)(void*, GErrorAbi**, char**);
  using MessageUnref = void (*)(void*);
  using ErrorFree = void (*)(GErrorAbi*);
  using Free = void (*)(void*);

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
    element_get_bus = core_library->symbol<ElementGetBus>("gst_element_get_bus");
    object_unref = core_library->symbol<ObjectUnref>("gst_object_unref");
    pad_get_current_caps = core_library->symbol<PadGetCurrentCaps>("gst_pad_get_current_caps");
    caps_unref = core_library->symbol<CapsUnref>("gst_caps_unref");
    sample_get_buffer = core_library->symbol<SampleGetBuffer>("gst_sample_get_buffer");
    sample_unref = core_library->symbol<SampleUnref>("gst_sample_unref");
    caps_get_structure = core_library->symbol<CapsGetStructure>("gst_caps_get_structure");
    structure_get_int = core_library->symbol<StructureGetInt>("gst_structure_get_int");
    buffer_map = core_library->symbol<BufferMap>("gst_buffer_map");
    buffer_unmap = core_library->symbol<BufferUnmap>("gst_buffer_unmap");
    bus_timed_pop_filtered =
        core_library->symbol<BusTimedPopFiltered>("gst_bus_timed_pop_filtered");
    message_parse_error = core_library->symbol<MessageParseError>("gst_message_parse_error");
    message_unref = core_library->symbol<MessageUnref>("gst_message_unref");
    app_sink_try_pull_sample =
        app_library->symbol<AppSinkTryPullSample>("gst_app_sink_try_pull_sample");
    app_sink_is_eos = app_library->symbol<AppSinkIsEos>("gst_app_sink_is_eos");
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
  ElementGetStaticPad element_get_static_pad = nullptr;
  ElementSetState element_set_state = nullptr;
  ElementGetBus element_get_bus = nullptr;
  ObjectUnref object_unref = nullptr;
  PadGetCurrentCaps pad_get_current_caps = nullptr;
  CapsUnref caps_unref = nullptr;
  AppSinkTryPullSample app_sink_try_pull_sample = nullptr;
  AppSinkIsEos app_sink_is_eos = nullptr;
  SampleGetBuffer sample_get_buffer = nullptr;
  SampleUnref sample_unref = nullptr;
  CapsGetStructure caps_get_structure = nullptr;
  StructureGetInt structure_get_int = nullptr;
  BufferMap buffer_map = nullptr;
  BufferUnmap buffer_unmap = nullptr;
  BusTimedPopFiltered bus_timed_pop_filtered = nullptr;
  MessageParseError message_parse_error = nullptr;
  MessageUnref message_unref = nullptr;
  ErrorFree error_free = nullptr;
  Free free = nullptr;
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

class GstreamerGpuFileDecodeSource final : public GpuFileDecodeSource {
public:
  GstreamerGpuFileDecodeSource(GpuFileDecodeConfig config, NvbufSurfaceAbi abi,
                               std::shared_ptr<GstreamerApi> api)
      : config_(std::move(config)),
        pipeline_description_(build_gstreamer_gpu_file_decode_pipeline(config_)), abi_(abi),
        api_(std::move(api)) {}

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
    bus_ = api_->element_get_bus(pipeline_);
    if (bus_ == nullptr) {
      throw GpuDecodeError("GStreamer pipeline does not provide a message bus");
    }
    if (api_->element_set_state(pipeline_, kGstStatePlaying) == kGstStateChangeFailure) {
      throw GpuDecodeError("GStreamer pipeline rejected the PLAYING state");
    }
  }

  [[nodiscard]] const GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return pipeline_description_; }
  [[nodiscard]] bool gpu_resident() const override { return true; }

  [[nodiscard]] GpuDecodeReadResult read() override {
    std::lock_guard lock(read_mutex_);
    if (terminal_error_.has_value()) {
      throw GpuDecodeError(*terminal_error_);
    }
    if (ended_) {
      return make_gpu_decode_eos();
    }

    void* sample = nullptr;
    while (sample == nullptr) {
      sample = api_->app_sink_try_pull_sample(sink_, kSamplePollTimeoutNs);
      if (sample != nullptr) {
        break;
      }
      if (const auto error = pop_pipeline_error(); !error.empty()) {
        terminal_error_ = error;
        throw GpuDecodeError(*terminal_error_);
      }
      if (api_->app_sink_is_eos(sink_) != 0) {
        ended_ = true;
        return make_gpu_decode_eos();
      }
    }

    auto owner = std::make_shared<GstSampleOwner>(api_, sample);
    void* buffer = api_->sample_get_buffer(sample);
    if (buffer == nullptr) {
      throw GpuDecodeError("GStreamer sample does not contain a buffer");
    }
    owner->map_buffer(buffer);
    const auto [visible_width, visible_height] = visible_dimensions(api_, display_info_pad_);

    NvmmFrameInfo nvmm;
    try {
      nvmm = extract_nvmm_frame_info(owner->mapped_data(), abi_);
    } catch (const std::exception& error) {
      throw GpuDecodeError("invalid GStreamer NVMM frame: " + std::string(error.what()));
    }

    const auto& gst_buffer = *static_cast<const GstBufferAbi*>(buffer);
    GpuDecodedFrame frame{
        .nvmm = nvmm,
        .visible_width = visible_width,
        .visible_height = visible_height,
        .owner = owner,
        .frame_index = next_frame_index_,
        .pts_ns = gst_buffer.pts == kGstClockTimeNone
                      ? std::nullopt
                      : std::optional<std::uint64_t>(gst_buffer.pts),
        .duration_ns = gst_buffer.duration == kGstClockTimeNone
                           ? std::nullopt
                           : std::optional<std::uint64_t>(gst_buffer.duration),
    };
    if (const auto validation_error = validate_gpu_decoded_frame(frame);
        validation_error.has_value()) {
      throw GpuDecodeError("invalid GStreamer GPU frame: " + *validation_error);
    }
    ++next_frame_index_;
    return make_gpu_decode_frame(std::move(frame));
  }

private:
  [[nodiscard]] std::string pop_pipeline_error() {
    void* message = api_->bus_timed_pop_filtered(bus_, 0, kGstMessageError);
    if (message == nullptr) {
      return {};
    }

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
    return result;
  }

  void close() noexcept {
    if (pipeline_ != nullptr) {
      (void)api_->element_set_state(pipeline_, kGstStateNull);
    }
    if (bus_ != nullptr) {
      api_->object_unref(bus_);
      bus_ = nullptr;
    }
    if (display_info_pad_ != nullptr) {
      api_->object_unref(display_info_pad_);
      display_info_pad_ = nullptr;
    }
    if (display_info_ != nullptr) {
      api_->object_unref(display_info_);
      display_info_ = nullptr;
    }
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
  void* bus_ = nullptr;
  std::mutex read_mutex_;
  std::uint64_t next_frame_index_ = 0;
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
