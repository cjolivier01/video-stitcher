#pragma once

#include "reco/io/gpu_encode.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace reco::io {

enum class GpuPreviewSink {
  NvidiaEgl,
  Nvidia3d,
};

/// Fixed geometry and native-window settings for GPU-resident preview presentation.
struct GpuPreviewConfig {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_numerator = 0;
  std::uint32_t fps_denominator = 0;
  std::uintptr_t window_handle = 0;
  GpuPreviewSink sink = GpuPreviewSink::NvidiaEgl;
  std::uint32_t device_ordinal = 0;
  std::uint32_t pool_capacity = 3;
  std::chrono::milliseconds acquire_timeout = std::chrono::seconds(1);
  std::chrono::milliseconds startup_timeout = std::chrono::seconds(10);
};

class GpuPreviewError : public std::runtime_error {
public:
  explicit GpuPreviewError(std::string message) : std::runtime_error(std::move(message)) {}
};

/// Bounded NVMM appsrc session presenting directly into a native Qt child window.
class GpuPreviewSession final {
public:
  [[nodiscard]] static GpuPreviewSession open(GpuPreviewConfig config,
                                              std::shared_ptr<GpuEncodeTraceSink> trace_sink = {});
  [[nodiscard]] static GpuPreviewSession open(GpuPreviewConfig config,
                                              std::shared_ptr<const NvbufSurfaceRuntime> runtime,
                                              std::shared_ptr<GpuEncodeTraceSink> trace_sink = {});

  GpuPreviewSession(const GpuPreviewSession&) = delete;
  GpuPreviewSession& operator=(const GpuPreviewSession&) = delete;
  GpuPreviewSession(GpuPreviewSession&&) noexcept;
  GpuPreviewSession& operator=(GpuPreviewSession&&) noexcept;
  ~GpuPreviewSession();

  [[nodiscard]] GpuEncodeFrameLease acquire_frame();
  void present(GpuEncodeFrameLease&& frame, std::uint64_t pts_ns, std::uint64_t duration_ns);
  void stop() noexcept;

  [[nodiscard]] const GpuPreviewConfig& config() const;
  [[nodiscard]] std::string_view pipeline() const;

private:
  struct Impl;
  explicit GpuPreviewSession(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::optional<std::string>
validate_gpu_preview_config(const GpuPreviewConfig& config);
[[nodiscard]] std::string_view gstreamer_gpu_preview_sink_factory(GpuPreviewSink sink);
[[nodiscard]] std::string build_gstreamer_gpu_preview_pipeline(const GpuPreviewConfig& config);

} // namespace reco::io
