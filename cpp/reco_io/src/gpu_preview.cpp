#include "reco/io/gpu_preview.hpp"

#include <limits>
#include <sstream>

namespace reco::io {

std::optional<std::string> validate_gpu_preview_config(const GpuPreviewConfig& config) {
  if (config.width == 0 || config.height == 0 || (config.width & 1U) != 0U ||
      (config.height & 1U) != 0U || config.width > 8192U || config.height > 8192U) {
    return "GPU preview dimensions must be non-zero, even, and at most 8192";
  }
  if (config.fps_numerator == 0 || config.fps_denominator == 0 ||
      static_cast<std::uint64_t>(config.fps_numerator) >
          static_cast<std::uint64_t>(config.fps_denominator) * 240U) {
    return "GPU preview frame rate must be finite, positive, and at most 240 fps";
  }
  if (config.window_handle == 0) {
    return "GPU preview requires a non-zero native window handle";
  }
  if (config.device_ordinal > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return "GPU preview device ordinal exceeds the CUDA runtime range";
  }
  if (config.sink == GpuPreviewSink::Nvidia3d && config.device_ordinal != 0) {
    return "NVIDIA 3D preview only supports CUDA device ordinal zero";
  }
  if (config.pool_capacity < 2 || config.pool_capacity > kMaximumGpuEncodePoolCapacity) {
    return "GPU preview pool capacity must be between 2 and 16";
  }
  if (config.acquire_timeout < std::chrono::milliseconds(1) ||
      config.acquire_timeout > std::chrono::minutes(5)) {
    return "GPU preview acquire timeout must be between 1 millisecond and 5 minutes";
  }
  if (config.startup_timeout < std::chrono::milliseconds(1) ||
      config.startup_timeout > std::chrono::minutes(5)) {
    return "GPU preview startup timeout must be between 1 millisecond and 5 minutes";
  }
  return std::nullopt;
}

std::string_view gstreamer_gpu_preview_sink_factory(GpuPreviewSink sink) {
  switch (sink) {
  case GpuPreviewSink::NvidiaEgl:
    return "nveglglessink";
  case GpuPreviewSink::Nvidia3d:
    return "nv3dsink";
  }
  throw std::invalid_argument("unsupported GPU preview sink");
}

std::string build_gstreamer_gpu_preview_pipeline(const GpuPreviewConfig& config) {
  if (const auto error = validate_gpu_preview_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  std::ostringstream pipeline;
  pipeline << "appsrc name=source is-live=true format=time do-timestamp=false block=false "
              "max-buffers="
           << config.pool_capacity
           << " caps=\"video/x-raw(memory:NVMM),format=(string)NV12,width=(int)" << config.width
           << ",height=(int)" << config.height << ",framerate=(fraction)" << config.fps_numerator
           << '/' << config.fps_denominator
           << "\" ! queue leaky=downstream max-size-buffers=" << config.pool_capacity
           << " max-size-bytes=0 max-size-time=0 ! "
           << gstreamer_gpu_preview_sink_factory(config.sink)
           << " name=preview_sink sync=false async=false qos=false";
  if (config.sink == GpuPreviewSink::NvidiaEgl) {
    pipeline << " force-aspect-ratio=true create-window=false gpu-id=" << config.device_ordinal;
  }
  return pipeline.str();
}

} // namespace reco::io
