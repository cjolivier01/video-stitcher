#include "reco/io/gstreamer.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#else
#include <dlfcn.h>
#endif

namespace reco::io {
namespace {

bool all_digits(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  for (const char ch : value) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }
  return true;
}

std::optional<std::string> loadable_library(std::initializer_list<const char*> names) {
  for (const char* name : names) {
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(name);
    if (handle != nullptr) {
      FreeLibrary(handle);
      return std::string(name);
    }
#else
    void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) {
      dlclose(handle);
      return std::string(name);
    }
#endif
  }
  return std::nullopt;
}

RuntimeProbe probe_library(std::initializer_list<const char*> names) {
  if (auto library = loadable_library(names); library.has_value()) {
    return RuntimeProbe{.available = true, .library = *library};
  }
  std::ostringstream message;
  message << "none of the expected runtime libraries could be loaded";
  return RuntimeProbe{.available = false, .library = {}, .error = message.str()};
}

RuntimeProbe probe_required_libraries(
    std::initializer_list<std::initializer_list<const char*>> groups) {
  std::vector<std::string> loaded;
  std::size_t index = 0;
  for (const auto group : groups) {
    if (auto library = loadable_library(group); library.has_value()) {
      loaded.push_back(*library);
    } else {
      std::ostringstream message;
      message << "required runtime library group " << index << " could not be loaded";
      return RuntimeProbe{.available = false, .library = {}, .error = message.str()};
    }
    ++index;
  }
  std::ostringstream joined;
  for (std::size_t i = 0; i < loaded.size(); ++i) {
    if (i != 0) {
      joined << ",";
    }
    joined << loaded[i];
  }
  return RuntimeProbe{.available = true, .library = joined.str()};
}

bool is_tegra() {
#if defined(__linux__)
  if (std::filesystem::exists("/etc/nv_tegra_release")) {
    return true;
  }
  std::error_code ec;
  const auto compatible = std::filesystem::path("/proc/device-tree/compatible");
  if (std::filesystem::exists(compatible, ec)) {
    std::ifstream file(compatible);
    std::string contents;
    std::getline(file, contents, '\0');
    return contents.find("nvidia,tegra") != std::string::npos;
  }
#endif
  return false;
}

} // namespace

std::string_view capture_format_name(CaptureFormat format) {
  switch (format) {
  case CaptureFormat::I420:
    return "I420";
  case CaptureFormat::Nv12:
    return "NV12";
  }
  return "I420";
}

CapturePlatform detect_capture_platform() {
  if (is_tegra()) {
    return CapturePlatform::Jetson;
  }
#if defined(_WIN32)
  return CapturePlatform::Windows;
#elif defined(__APPLE__)
  return CapturePlatform::Macos;
#else
  return CapturePlatform::LinuxV4l2;
#endif
}

std::optional<std::string> validate_capture_device(std::string_view device,
                                                   CapturePlatform platform) {
  if (platform == CapturePlatform::Auto) {
    platform = detect_capture_platform();
  }
  switch (platform) {
  case CapturePlatform::Jetson:
  case CapturePlatform::Macos:
  case CapturePlatform::Windows:
    if (all_digits(device)) {
      return std::nullopt;
    }
    return "expected a numeric camera index";
  case CapturePlatform::LinuxV4l2: {
    constexpr std::string_view prefix = "/dev/video";
    if (device.starts_with(prefix) && all_digits(device.substr(prefix.size()))) {
      return std::nullopt;
    }
    return "expected a V4L2 path like /dev/video0";
  }
  case CapturePlatform::Auto:
    break;
  }
  return "unknown capture platform";
}

std::string build_capture_pipeline_string(std::string_view device, std::uint32_t width,
                                          std::uint32_t height, std::uint32_t fps,
                                          CaptureFormat format, CapturePlatform platform) {
  if (width == 0 || height == 0 || fps == 0) {
    throw std::invalid_argument("capture dimensions and fps must be non-zero");
  }
  if (platform == CapturePlatform::Auto) {
    platform = detect_capture_platform();
  }
  if (const auto error = validate_capture_device(device, platform); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  const auto fmt = capture_format_name(format);
  std::ostringstream pipeline;
  switch (platform) {
  case CapturePlatform::Jetson:
    pipeline << "nvarguscamerasrc sensor-id=" << device
             << " ! video/x-raw(memory:NVMM),width=" << width << ",height=" << height
             << ",format=NV12,framerate=" << fps
             << "/1 ! nvvidconv ! video/x-raw,format=" << fmt
             << " ! appsink name=sink emit-signals=false sync=false";
    break;
  case CapturePlatform::Macos:
    pipeline << "avfvideosrc device-index=" << device << " ! video/x-raw,width=" << width
             << ",height=" << height << ",framerate=" << fps
             << "/1 ! videoconvert ! video/x-raw,format=" << fmt
             << " ! appsink name=sink emit-signals=false sync=false";
    break;
  case CapturePlatform::Windows:
    pipeline << "mfvideosrc device-index=" << device << " ! video/x-raw,width=" << width
             << ",height=" << height << ",framerate=" << fps
             << "/1 ! videoconvert ! video/x-raw,format=" << fmt
             << " ! appsink name=sink emit-signals=false sync=false";
    break;
  case CapturePlatform::LinuxV4l2:
    pipeline << "v4l2src device=" << device << " ! video/x-raw,width=" << width
             << ",height=" << height << ",framerate=" << fps
             << "/1 ! videoconvert ! video/x-raw,format=" << fmt
             << " ! appsink name=sink emit-signals=false sync=false";
    break;
  case CapturePlatform::Auto:
    break;
  }
  return pipeline.str();
}

RuntimeProbe probe_gstreamer_runtime() {
#if defined(_WIN32)
  return probe_required_libraries({{"gstreamer-1.0-0.dll"}, {"gstapp-1.0-0.dll"}});
#elif defined(__APPLE__)
  return probe_required_libraries(
      {{"libgstreamer-1.0.0.dylib", "libgstreamer-1.0.dylib"},
       {"libgstapp-1.0.0.dylib", "libgstapp-1.0.dylib"}});
#else
  return probe_required_libraries(
      {{"libgstreamer-1.0.so.0", "libgstreamer-1.0.so"},
       {"libgstapp-1.0.so.0", "libgstapp-1.0.so"}});
#endif
}

RuntimeProbe probe_deepstream_runtime() {
#if defined(_WIN32) || defined(__APPLE__)
  return RuntimeProbe{.available = false,
                      .library = {},
                      .error = "DeepStream runtime probing is only supported on Linux"};
#else
  return probe_library({"libnvds_meta.so", "libnvdsgst_meta.so"});
#endif
}

RuntimeProbe probe_nvbufsurface_runtime() {
#if defined(_WIN32) || defined(__APPLE__)
  return RuntimeProbe{.available = false,
                      .library = {},
                      .error = "NvBufSurface runtime probing is only supported on Linux"};
#else
  return probe_library({"libnvbufsurface.so"});
#endif
}

} // namespace reco::io
