#pragma once

#include "reco/io/gpu_preview.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace reco::gui {

/// Native media, presentation, and resource bounds for one GPU preview controller.
struct GpuPreviewControllerConfig {
  /// Left-eye local recording path.
  std::filesystem::path left_path;
  /// Right-eye local recording path.
  std::filesystem::path right_path;
  /// Match-calibration JSON path.
  std::filesystem::path calibration_path;
  /// Absolute path of the deployed `reco_video_probe_worker` executable.
  std::filesystem::path probe_worker_path;
  /// Native Qt child-window handle consumed by `GstVideoOverlay`.
  std::uintptr_t native_window_handle = 0;
  /// Even preview width in pixels.
  std::uint32_t output_width = 0;
  /// Even preview height in pixels.
  std::uint32_t output_height = 0;
  /// NVIDIA GStreamer sink selected for the target platform.
  io::GpuPreviewSink sink = io::GpuPreviewSink::NvidiaEgl;
  /// CUDA device and DeepStream GPU ordinal.
  std::uint32_t device_ordinal = 0;
  /// Number of bounded NVMM presentation surfaces.
  std::uint32_t surface_pool_capacity = 3;
  /// Number of retained NVDEC buffer owners on each stereo side.
  std::uint32_t decode_queue_capacity = 4;
  /// Optional seam blend override; absent preserves the calibration value.
  std::optional<float> blend_width_override;
  /// Per-input parser probe deadline.
  std::chrono::nanoseconds probe_timeout = std::chrono::seconds(30);
  /// Maximum wait for one free NVMM presentation surface.
  std::chrono::milliseconds surface_acquire_timeout = std::chrono::seconds(1);
  /// Maximum wait for the GStreamer presentation pipeline to start.
  std::chrono::milliseconds presentation_startup_timeout = std::chrono::seconds(10);
  /// Whether successful startup immediately begins playback.
  bool start_playing = true;
};

/// Virtual-camera controls consumed directly by the CUDA stitch renderer.
struct GpuPreviewViewport {
  /// Horizontal camera rotation in radians.
  float yaw_radians = 0.0F;
  /// Vertical camera rotation in radians.
  float pitch_radians = 0.0F;
  /// Vertical perspective field of view in degrees.
  float fov_degrees = 75.0F;
};

/// Stable lifecycle states exposed to GUI consumers.
enum class GpuPreviewControllerState {
  /// No worker or GPU resources are active.
  Stopped,
  /// Input probing or GPU resource creation is in progress.
  Starting,
  /// Frames are being decoded and presented.
  Playing,
  /// Persistent GPU resources are retained without frame production.
  Paused,
  /// The indexed NVDEC sessions are being rebuilt.
  Seeking,
  /// Both synchronized inputs reached EOS.
  EndOfStream,
  /// A sticky initialization or playback failure occurred.
  Error,
};

/// Immutable controller state copied across the GUI notification boundary.
struct GpuPreviewControllerSnapshot {
  /// Current lifecycle state.
  GpuPreviewControllerState state = GpuPreviewControllerState::Stopped;
  /// True after all GPU resources are initialized and until explicit stop or failure.
  bool ready = false;
  /// True only while the controller is actively producing frames.
  bool playing = false;
  /// True after both aligned inputs reach end of stream.
  bool end_of_stream = false;
  /// Last source-timeline frame presented, when one has been produced.
  std::optional<std::uint64_t> current_frame;
  /// Probe-derived number of aligned source frames.
  std::uint64_t total_frames = 0;
  /// Whether `total_frames` is parser-proven rather than estimated.
  bool total_frames_exact = false;
  /// Number of frames presented during this controller start.
  std::uint64_t presented_frames = 0;
  /// Monotonic presentation timestamp of the last frame.
  std::optional<std::uint64_t> presentation_timestamp_ns;
  /// Exact constant input frame-rate numerator.
  std::uint32_t fps_numerator = 0;
  /// Exact constant input frame-rate denominator.
  std::uint32_t fps_denominator = 0;
  /// Clamped viewport used for the next frame.
  GpuPreviewViewport viewport;
  /// Sticky failure detail while `state == Error`.
  std::string error;
};

/// Constant-cadence metadata returned by a controller backend.
struct GpuPreviewBackendStreamInfo {
  /// Input frame-rate numerator.
  std::uint32_t fps_numerator = 0;
  /// Input frame-rate denominator.
  std::uint32_t fps_denominator = 0;
  /// Number of aligned source frames available for seeking.
  std::uint64_t total_frames = 0;
  /// Whether `total_frames` was proven by full-stream parser evidence.
  bool total_frames_exact = false;
};

/// Result of one backend frame operation.
enum class GpuPreviewBackendFrameStatus {
  /// One GPU-resident frame reached the presentation sink.
  Presented,
  /// No further synchronized source frame exists.
  EndOfStream,
  /// A controller request interrupted the operation.
  Stopped,
};

/// Source identity accompanying one backend frame result.
struct GpuPreviewBackendFrameResult {
  /// Frame operation outcome.
  GpuPreviewBackendFrameStatus status = GpuPreviewBackendFrameStatus::EndOfStream;
  /// Aligned source-timeline frame when `status == Presented`.
  std::optional<std::uint64_t> source_frame;
};

/// GPU preview execution seam.
///
/// Production callers use the controller's normal constructor. This interface
/// permits deterministic lifecycle tests without loading CUDA; it transports no
/// pixels and cannot introduce a CPU frame path.
class GpuPreviewControllerBackend {
public:
  /// Releases backend-specific test or production state.
  virtual ~GpuPreviewControllerBackend() = default;

  /// Resets cancellation before a new worker is launched.
  virtual void prepare_start() {}
  /// Probes inputs and creates persistent rendering and presentation resources.
  [[nodiscard]] virtual GpuPreviewBackendStreamInfo
  initialize(const GpuPreviewControllerConfig& config) = 0;
  /// Rebuilds only the indexed NVDEC stereo session at `frame_index`.
  virtual void rebuild_decode(std::uint64_t frame_index) = 0;
  /// Renders, converts, and presents one frame without CPU pixel access.
  [[nodiscard]] virtual GpuPreviewBackendFrameResult
  present_next(const GpuPreviewViewport& viewport, std::uint64_t pts_ns,
               std::uint64_t duration_ns) = 0;
  /// Interrupts a blocked decode read while retaining presentation resources.
  virtual void interrupt_decode() noexcept = 0;
  /// Interrupts all blocking work and stops presentation.
  virtual void shutdown() noexcept = 0;
  /// Releases stopped decoder, presentation, and persistent GPU resources.
  virtual void release() noexcept = 0;
};

/// Thread-safe owner of the non-Qt GPU preview pipeline.
class GpuPreviewController final {
public:
  /// Callback invoked outside controller locks; it may run on a control or worker thread and may
  /// request that the controller stop. Qt consumers should forward the copied snapshot with a
  /// queued connection. A callback must not destroy its controller from the worker thread.
  using NotificationCallback = std::function<void(GpuPreviewControllerSnapshot)>;

  /// Creates a controller using the production NVDEC/CUDA/NVMM backend.
  explicit GpuPreviewController(GpuPreviewControllerConfig config,
                                NotificationCallback callback = {});
  /// Creates a controller with an explicit non-pixel backend, primarily for tests.
  GpuPreviewController(GpuPreviewControllerConfig config,
                       std::unique_ptr<GpuPreviewControllerBackend> backend,
                       NotificationCallback callback = {});
  /// Stops and joins any active worker before releasing GPU resources.
  ~GpuPreviewController();

  GpuPreviewController(const GpuPreviewController&) = delete;
  GpuPreviewController& operator=(const GpuPreviewController&) = delete;
  GpuPreviewController(GpuPreviewController&&) = delete;
  GpuPreviewController& operator=(GpuPreviewController&&) = delete;

  /// Starts asynchronous probing, GPU initialization, and optional playback.
  void start();
  /// Idempotently interrupts blocked work and joins the worker thread.
  void stop() noexcept;
  /// Resumes frame production when the initialized stream is not at EOS.
  void play();
  /// Pauses frame production without releasing persistent GPU resources.
  void pause();
  /// Interrupts initialized decode and rebuilds indexed NVDEC at `frame_index`.
  void seek(std::uint64_t frame_index);
  /// Applies finite, bounded controls to subsequent CUDA renders.
  void set_viewport(GpuPreviewViewport viewport);
  /// Replaces the callback and waits for invocations of the previous callback to finish.
  void set_notification_callback(NotificationCallback callback);

  /// Returns a lock-protected copy of the current state.
  [[nodiscard]] GpuPreviewControllerSnapshot snapshot() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Returns a validation error for invalid paths, dimensions, handles, or resource bounds.
[[nodiscard]] std::optional<std::string>
validate_gpu_preview_controller_config(const GpuPreviewControllerConfig& config);
/// Replaces non-finite controls and clamps them to renderer-safe GUI bounds.
[[nodiscard]] GpuPreviewViewport clamp_gpu_preview_viewport(GpuPreviewViewport viewport);
/// Computes `floor(frame_index * 1s * denominator / numerator)` without overflow.
[[nodiscard]] std::uint64_t gpu_preview_timestamp_for_frame(std::uint64_t frame_index,
                                                            std::uint32_t fps_numerator,
                                                            std::uint32_t fps_denominator);
/// Returns a stable textual lifecycle-state name.
[[nodiscard]] std::string_view
gpu_preview_controller_state_name(GpuPreviewControllerState state) noexcept;

} // namespace reco::gui
