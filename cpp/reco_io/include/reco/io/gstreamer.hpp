#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace reco::io {

enum class CaptureFormat {
  I420,
  Nv12,
};

enum class CapturePlatform {
  Auto,
  Jetson,
  LinuxV4l2,
  Macos,
  Windows,
};

struct CameraConfig {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps = 0;
  std::string left_device;
  std::string right_device;
};

struct RuntimeProbe {
  bool available = false;
  std::string library;
  std::string error;
};

[[nodiscard]] std::string_view capture_format_name(CaptureFormat format);
[[nodiscard]] CapturePlatform detect_capture_platform();
[[nodiscard]] std::optional<std::string> validate_capture_device(std::string_view device,
                                                                 CapturePlatform platform);
[[nodiscard]] std::string build_capture_pipeline_string(std::string_view device,
                                                        std::uint32_t width,
                                                        std::uint32_t height, std::uint32_t fps,
                                                        CaptureFormat format,
                                                        CapturePlatform platform);
[[nodiscard]] RuntimeProbe probe_gstreamer_runtime();
[[nodiscard]] RuntimeProbe probe_deepstream_runtime();
[[nodiscard]] RuntimeProbe probe_nvbufsurface_runtime();

} // namespace reco::io
