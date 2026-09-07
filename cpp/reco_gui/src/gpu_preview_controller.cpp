#include "reco/gui/gpu_preview_controller.hpp"

#include "reco/core/calibration.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/core/cuda_frame.hpp"
#include "reco/core/cuda_rgba_to_nv12.hpp"
#include "reco/core/cuda_stitch_renderer.hpp"
#include "reco/core/nvrtc_compiler.hpp"
#include "reco/core/path.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gpu_video_probe.hpp"
#include "reco/io/nvmm.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace reco::gui {
namespace {

constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
constexpr float kPi = 3.14159265358979323846F;
constexpr float kMaximumPitchRadians = 89.0F * kPi / 180.0F;
constexpr float kMinimumFovDegrees = 20.0F;
constexpr float kMaximumFovDegrees = 150.0F;
thread_local const void* g_active_preview_callback_slot = nullptr;

template <typename Path> bool path_has_embedded_null(const Path& path) {
  for (const auto value : path.native()) {
    if (value == typename Path::value_type{}) {
      return true;
    }
  }
  return false;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, std::string_view error) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error(std::string(error));
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right, std::string_view error) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(error));
  }
  return left * right;
}

// Exact floor(a * b / divisor) using quotient/remainder accumulation, without
// relying on a compiler-specific double-width integer.
std::uint64_t multiply_divide_floor(std::uint64_t a, std::uint64_t b, std::uint64_t divisor) {
  if (divisor == 0U) {
    throw std::invalid_argument("GPU preview timestamp divisor must be non-zero");
  }
  const auto add_quotient = b / divisor;
  const auto add_remainder = b % divisor;
  std::uint64_t quotient = 0;
  std::uint64_t remainder = 0;

  for (int bit = 63; bit >= 0; --bit) {
    if (quotient > std::numeric_limits<std::uint64_t>::max() / 2U) {
      throw std::overflow_error("GPU preview timestamp exceeds the GStreamer time range");
    }
    quotient *= 2U;
    if (remainder >= divisor - remainder) {
      remainder -= divisor - remainder;
      quotient =
          checked_add(quotient, 1U, "GPU preview timestamp exceeds the GStreamer time range");
    } else {
      remainder += remainder;
    }

    if (((a >> bit) & 1U) == 0U) {
      continue;
    }
    quotient = checked_add(quotient, add_quotient,
                           "GPU preview timestamp exceeds the GStreamer time range");
    if (add_remainder != 0U && remainder >= divisor - add_remainder) {
      remainder -= divisor - add_remainder;
      quotient =
          checked_add(quotient, 1U, "GPU preview timestamp exceeds the GStreamer time range");
    } else {
      remainder += add_remainder;
    }
  }
  return quotient;
}

io::GpuFileDecodeConfig make_probe_config(const std::filesystem::path& path) {
  const auto utf8 = core::path_to_utf8(path);
  return {.path = utf8,
          .codec = io::gpu_decode_codec_for_path(utf8),
          .elementary_stream = io::gpu_decode_path_is_elementary_stream(utf8),
          .container = io::gpu_decode_container_for_path(utf8)};
}

void validate_indexed_probe(const io::GpuVideoProbe& probe, std::string_view label) {
  if (probe.width == 0U || probe.height == 0U || probe.fps_numerator == 0U ||
      probe.fps_denominator == 0U || probe.total_frames == 0U) {
    throw std::runtime_error(std::string(label) +
                             " video probe returned incomplete stream metadata");
  }
  if (!probe.indexed_sampling_cadence_verified || !probe.first_stream_time_ns.has_value()) {
    throw std::runtime_error(std::string(label) +
                             " video does not have parser-verified constant indexed cadence");
  }
  if (probe.timestamp_multiplicity == 0U ||
      probe.timestamp_multiplicity > io::kMaximumIndexedTimestampMultiplicity) {
    throw std::runtime_error(std::string(label) +
                             " video has unsupported presentation timestamp multiplicity");
  }
}

std::uint64_t aligned_frame_count(const io::GpuVideoProbe& left, const io::GpuVideoProbe& right,
                                  std::int64_t sync_offset) {
  if (sync_offset >= 0) {
    const auto offset = static_cast<std::uint64_t>(sync_offset);
    return offset >= right.total_frames ? 0U
                                        : std::min(left.total_frames, right.total_frames - offset);
  }
  const auto offset = static_cast<std::uint64_t>(-sync_offset);
  return offset >= left.total_frames ? 0U
                                     : std::min(right.total_frames, left.total_frames - offset);
}

void validate_runtime_rotations(const io::GpuDecodedFramePair& frames) {
  const auto supported = [](std::uint16_t rotation) { return rotation == 0U || rotation == 180U; };
  if (!supported(frames.left.rotation_degrees) || !supported(frames.right.rotation_degrees)) {
    throw std::runtime_error("90/270-degree stereo preview rotation is not supported");
  }
}

class ProductionGpuPreviewBackend final : public GpuPreviewControllerBackend {
public:
  void prepare_start() override { shutdown_requested_.store(false, std::memory_order_release); }

  GpuPreviewBackendStreamInfo initialize(const GpuPreviewControllerConfig& config) override {
    reset_resources();
    throw_if_shutdown();
    config_ = config;

    auto left_config = make_probe_config(config.left_path);
    auto right_config = make_probe_config(config.right_path);
    const auto timeout = static_cast<std::uint64_t>(config.probe_timeout.count());
    left_probe_ = io::probe_gpu_video(left_config, config.probe_worker_path, timeout);
    throw_if_shutdown();
    right_probe_ = io::probe_gpu_video(right_config, config.probe_worker_path, timeout);
    throw_if_shutdown();
    validate_indexed_probe(left_probe_, "left");
    validate_indexed_probe(right_probe_, "right");
    if (left_probe_.fps_numerator != right_probe_.fps_numerator ||
        left_probe_.fps_denominator != right_probe_.fps_denominator) {
      throw std::runtime_error("stereo preview inputs must have one matching constant frame rate");
    }

    std::string calibration_error;
    auto calibration = core::load_match_calibration_file(
        core::path_to_utf8(config.calibration_path), &calibration_error);
    if (!calibration.has_value()) {
      throw std::runtime_error(calibration_error.empty() ? "invalid preview calibration JSON"
                                                         : calibration_error);
    }
    if (config.blend_width_override.has_value()) {
      calibration->blend_width = *config.blend_width_override;
    }
    if (calibration->left.width != left_probe_.width ||
        calibration->left.height != left_probe_.height ||
        calibration->right.width != right_probe_.width ||
        calibration->right.height != right_probe_.height) {
      throw std::runtime_error("preview calibration dimensions do not match the probed inputs");
    }
    if (const auto error =
            io::validate_gpu_stereo_decode_config({.sync_offset = calibration->sync_offset,
                                                   .queue_capacity = config.decode_queue_capacity});
        error.has_value()) {
      throw std::runtime_error(*error);
    }
    sync_offset_ = calibration->sync_offset;
    const auto total_frames = aligned_frame_count(left_probe_, right_probe_, sync_offset_);
    if (total_frames == 0U) {
      throw std::runtime_error("preview inputs have no frames after calibration synchronization");
    }

    runtime_ = io::discover_nvbufsurface_runtime();
    throw_if_shutdown();
    cuda_.emplace(core::CudaBackend::create());
    const auto device = static_cast<int>(config.device_ordinal);
    if (device >= cuda_->device_count()) {
      throw std::runtime_error("GPU preview CUDA device ordinal is not installed");
    }
    cuda_->ensure_primary_context(device);
    auto compiler = core::NvrtcCompiler::create();
    renderer_.emplace(core::CudaStereoStitchRenderer::create({.calibration = *calibration,
                                                              .output_width = config.output_width,
                                                              .output_height = config.output_height,
                                                              .device_ordinal = device},
                                                             *cuda_, compiler));
    rgba_storage_ = cuda_->allocate_pitched(static_cast<std::size_t>(config.output_width) * 4U,
                                            config.output_height, 4U);
    rgba_view_.emplace(core::CudaPitchedPlaneView(
                           rgba_storage_.buffer.ptr(), rgba_storage_.buffer.size(),
                           rgba_storage_.pitch, static_cast<std::size_t>(config.output_width) * 4U,
                           config.output_height, cuda_->primary_context_id(device), device),
                       config.output_width, config.output_height);
    converter_.emplace(core::CudaRgbaToNv12Converter::create(
        {.width = config.output_width, .height = config.output_height, .device_ordinal = device},
        *cuda_, compiler));

    auto preview = std::make_unique<io::GpuPreviewSession>(
        io::GpuPreviewSession::open({.width = config.output_width,
                                     .height = config.output_height,
                                     .fps_numerator = left_probe_.fps_numerator,
                                     .fps_denominator = left_probe_.fps_denominator,
                                     .window_handle = config.native_window_handle,
                                     .sink = config.sink,
                                     .device_ordinal = config.device_ordinal,
                                     .pool_capacity = config.surface_pool_capacity,
                                     .acquire_timeout = config.surface_acquire_timeout,
                                     .startup_timeout = config.presentation_startup_timeout},
                                    runtime_));
    {
      std::lock_guard lock(access_mutex_);
      preview_ = std::move(preview);
      if (shutdown_requested_.load(std::memory_order_acquire)) {
        preview_->stop();
      }
    }
    throw_if_shutdown();

    stream_info_ = {.fps_numerator = left_probe_.fps_numerator,
                    .fps_denominator = left_probe_.fps_denominator,
                    .total_frames = total_frames,
                    .total_frames_exact = !left_probe_.total_frames_is_estimated &&
                                          !right_probe_.total_frames_is_estimated};
    rebuild_decode(0U);
    prefetch_first_pair();
    return stream_info_;
  }

  void rebuild_decode(std::uint64_t frame_index) override {
    if (frame_index >= stream_info_.total_frames) {
      throw std::out_of_range("GPU preview seek frame is outside the aligned input range");
    }
    prefetched_pair_.reset();
    const auto generation = interrupt_generation_.load(std::memory_order_acquire);
    auto left = io::open_gstreamer_gpu_file_decode_source(
        make_decode_config(config_.left_path, left_probe_, frame_index), runtime_);
    auto right = io::open_gstreamer_gpu_file_decode_source(
        make_decode_config(config_.right_path, right_probe_, frame_index), runtime_);
    auto decoder = std::make_unique<io::GpuStereoDecodeSession>(
        std::move(left), std::move(right),
        io::GpuStereoDecodeConfig{.sync_offset = sync_offset_,
                                  .queue_capacity = config_.decode_queue_capacity});

    std::unique_ptr<io::GpuStereoDecodeSession> previous;
    {
      std::lock_guard lock(access_mutex_);
      previous = std::move(decoder_);
      decoder_ = std::move(decoder);
      if (shutdown_requested_.load(std::memory_order_acquire) ||
          generation != interrupt_generation_.load(std::memory_order_acquire)) {
        decoder_->request_stop();
      }
    }
    previous.reset();
    throw_if_shutdown();
  }

  GpuPreviewBackendFrameResult present_next(const GpuPreviewViewport& viewport,
                                            std::uint64_t pts_ns,
                                            std::uint64_t duration_ns) override {
    const auto generation = interrupt_generation_.load(std::memory_order_acquire);
    io::GpuStereoDecodeSession* decoder = nullptr;
    io::GpuPreviewSession* preview = nullptr;
    {
      std::lock_guard lock(access_mutex_);
      decoder = decoder_.get();
      preview = preview_.get();
    }
    if (decoder == nullptr || preview == nullptr || !renderer_.has_value() ||
        !converter_.has_value() || !rgba_view_.has_value()) {
      throw std::runtime_error("GPU preview backend is not initialized");
    }

    std::optional<io::GpuDecodedFramePair> frames;
    if (prefetched_pair_.has_value()) {
      frames = std::move(prefetched_pair_);
      prefetched_pair_.reset();
    } else {
      auto decoded = decoder->read();
      if (generation != interrupt_generation_.load(std::memory_order_acquire) ||
          decoded.status == io::GpuStereoDecodeStatus::Stopped) {
        return {.status = GpuPreviewBackendFrameStatus::Stopped, .source_frame = std::nullopt};
      }
      if (decoded.status == io::GpuStereoDecodeStatus::EndOfStream) {
        return {.status = GpuPreviewBackendFrameStatus::EndOfStream, .source_frame = std::nullopt};
      }
      if (!decoded.frames.has_value()) {
        throw std::runtime_error("GPU stereo decoder returned an empty frame-pair result");
      }
      frames = std::move(decoded.frames);
    }
    validate_runtime_rotations(*frames);
    const auto left_rotation = frames->left.rotation_degrees;
    const auto right_rotation = frames->right.rotation_degrees;

    auto left = io::map_gpu_decoded_frame_to_cuda_lease(frames->left);
    auto right = io::map_gpu_decoded_frame_to_cuda_lease(frames->right);
    renderer_->render(left.view(), right.view(), *rgba_view_,
                      {.yaw = viewport.yaw_radians,
                       .pitch = viewport.pitch_radians,
                       .fov_degrees = viewport.fov_degrees,
                       .flip_left_180 = left_rotation == 180U,
                       .flip_right_180 = right_rotation == 180U});
    if (generation != interrupt_generation_.load(std::memory_order_acquire)) {
      return {.status = GpuPreviewBackendFrameStatus::Stopped, .source_frame = std::nullopt};
    }

    auto output = preview->acquire_frame();
    converter_->convert(*rgba_view_, output.view());
    if (generation != interrupt_generation_.load(std::memory_order_acquire)) {
      return {.status = GpuPreviewBackendFrameStatus::Stopped, .source_frame = std::nullopt};
    }
    preview->present(std::move(output), pts_ns, duration_ns);
    const auto source_frame =
        sync_offset_ >= 0 ? frames->left.frame_index : frames->right.frame_index;
    return {.status = GpuPreviewBackendFrameStatus::Presented, .source_frame = source_frame};
  }

  void interrupt_decode() noexcept override {
    interrupt_generation_.fetch_add(1U, std::memory_order_acq_rel);
    std::lock_guard lock(access_mutex_);
    if (decoder_) {
      decoder_->request_stop();
    }
  }

  void shutdown() noexcept override {
    shutdown_requested_.store(true, std::memory_order_release);
    interrupt_generation_.fetch_add(1U, std::memory_order_acq_rel);
    std::lock_guard lock(access_mutex_);
    if (decoder_) {
      decoder_->request_stop();
    }
    if (preview_) {
      preview_->stop();
    }
  }

  void release() noexcept override { reset_resources(); }

private:
  void prefetch_first_pair() {
    io::GpuStereoDecodeSession* decoder = nullptr;
    {
      std::lock_guard lock(access_mutex_);
      decoder = decoder_.get();
    }
    if (decoder == nullptr) {
      throw std::runtime_error("GPU preview decoder is not initialized");
    }
    auto decoded = decoder->read();
    throw_if_shutdown();
    if (decoded.status != io::GpuStereoDecodeStatus::FramePair || !decoded.frames.has_value()) {
      throw std::runtime_error("GPU preview inputs ended before the first aligned frame");
    }
    validate_runtime_rotations(*decoded.frames);
    prefetched_pair_ = std::move(decoded.frames);
  }

  io::GpuFileDecodeConfig make_decode_config(const std::filesystem::path& path,
                                             const io::GpuVideoProbe& probe,
                                             std::uint64_t start_frame) const {
    auto config = make_probe_config(path);
    config.max_buffers = config_.decode_queue_capacity;
    config.drop = false;
    config.read_timeout_ns = 30'000'000'000ULL;
    config.indexed_fps_numerator = probe.fps_numerator;
    config.indexed_fps_denominator = probe.fps_denominator;
    config.indexed_timestamp_multiplicity = probe.timestamp_multiplicity;
    config.indexed_stream_time_origin_ns = probe.first_stream_time_ns;
    config.start_frame_index = start_frame;
    return config;
  }

  void throw_if_shutdown() const {
    if (shutdown_requested_.load(std::memory_order_acquire)) {
      throw std::runtime_error("GPU preview initialization was stopped");
    }
  }

  void reset_resources() noexcept {
    std::unique_ptr<io::GpuStereoDecodeSession> decoder;
    std::unique_ptr<io::GpuPreviewSession> preview;
    {
      std::lock_guard lock(access_mutex_);
      decoder = std::move(decoder_);
      preview = std::move(preview_);
    }
    decoder.reset();
    preview.reset();
    prefetched_pair_.reset();
    rgba_view_.reset();
    rgba_storage_ = {};
    converter_.reset();
    renderer_.reset();
    cuda_.reset();
    runtime_.reset();
    stream_info_ = {};
  }

  GpuPreviewControllerConfig config_;
  io::GpuVideoProbe left_probe_;
  io::GpuVideoProbe right_probe_;
  std::int64_t sync_offset_ = 0;
  GpuPreviewBackendStreamInfo stream_info_;
  std::shared_ptr<const io::NvbufSurfaceRuntime> runtime_;
  std::optional<core::CudaBackend> cuda_;
  std::optional<core::CudaStereoStitchRenderer> renderer_;
  core::CudaPitchedAllocation rgba_storage_;
  std::optional<core::CudaRgbaFrameView> rgba_view_;
  std::optional<core::CudaRgbaToNv12Converter> converter_;
  std::mutex access_mutex_;
  std::unique_ptr<io::GpuStereoDecodeSession> decoder_;
  std::unique_ptr<io::GpuPreviewSession> preview_;
  std::optional<io::GpuDecodedFramePair> prefetched_pair_;
  std::atomic<std::uint64_t> interrupt_generation_{0};
  std::atomic<bool> shutdown_requested_{false};
};

void validate_backend_stream_info(const GpuPreviewBackendStreamInfo& info) {
  if (info.fps_numerator == 0U || info.fps_denominator == 0U ||
      static_cast<std::uint64_t>(info.fps_numerator) >
          static_cast<std::uint64_t>(info.fps_denominator) * 240U) {
    throw std::runtime_error("GPU preview backend returned an invalid frame rate");
  }
  if (info.total_frames == 0U) {
    throw std::runtime_error("GPU preview backend returned an empty aligned timeline");
  }
}

} // namespace

std::optional<std::string>
validate_gpu_preview_controller_config(const GpuPreviewControllerConfig& config) {
  if (config.left_path.empty() || path_has_embedded_null(config.left_path)) {
    return "GPU preview left path must be non-empty and contain no NUL";
  }
  if (config.right_path.empty() || path_has_embedded_null(config.right_path)) {
    return "GPU preview right path must be non-empty and contain no NUL";
  }
  if (config.calibration_path.empty() || path_has_embedded_null(config.calibration_path)) {
    return "GPU preview calibration path must be non-empty and contain no NUL";
  }
  if (config.probe_worker_path.empty() || path_has_embedded_null(config.probe_worker_path) ||
      !config.probe_worker_path.is_absolute()) {
    return "GPU preview probe worker path must be absolute and contain no NUL";
  }
  if (config.native_window_handle == 0U) {
    return "GPU preview requires a non-zero native window handle";
  }
  if (config.output_width == 0U || config.output_height == 0U || (config.output_width & 1U) != 0U ||
      (config.output_height & 1U) != 0U || config.output_width > core::kMaxCalibrationDimension ||
      config.output_height > core::kMaxCalibrationDimension) {
    return "GPU preview dimensions must be non-zero, even, and at most 8192";
  }
  switch (config.sink) {
  case io::GpuPreviewSink::NvidiaEgl:
  case io::GpuPreviewSink::Nvidia3d:
    break;
  default:
    return "GPU preview sink is unsupported";
  }
  if (config.device_ordinal != 0U) {
    return "GPU preview currently requires device 0 because GPU decode has no device selector";
  }
  if (config.surface_pool_capacity < 2U ||
      config.surface_pool_capacity > io::kMaximumGpuEncodePoolCapacity) {
    return "GPU preview surface pool capacity must be between 2 and 16";
  }
  if (config.decode_queue_capacity < io::kMinimumGpuStereoQueueCapacity ||
      config.decode_queue_capacity > io::kMaximumGpuStereoQueueCapacity) {
    return "GPU preview decode queue capacity must be between 1 and 16";
  }
  if (config.blend_width_override.has_value() &&
      (!std::isfinite(*config.blend_width_override) || *config.blend_width_override < 0.0F ||
       *config.blend_width_override > 1.0F)) {
    return "GPU preview blend width override must be finite and between zero and one";
  }
  if (config.probe_timeout < std::chrono::seconds(1) ||
      config.probe_timeout > std::chrono::hours(1)) {
    return "GPU preview probe timeout must be between one second and one hour";
  }
  io::GpuPreviewConfig preview_config{
      .width = config.output_width,
      .height = config.output_height,
      .fps_numerator = 1,
      .fps_denominator = 1,
      .window_handle = config.native_window_handle,
      .sink = config.sink,
      .device_ordinal = config.device_ordinal,
      .pool_capacity = config.surface_pool_capacity,
      .acquire_timeout = config.surface_acquire_timeout,
      .startup_timeout = config.presentation_startup_timeout,
  };
  return io::validate_gpu_preview_config(preview_config);
}

GpuPreviewViewport clamp_gpu_preview_viewport(GpuPreviewViewport viewport) {
  if (!std::isfinite(viewport.yaw_radians)) {
    viewport.yaw_radians = 0.0F;
  }
  if (!std::isfinite(viewport.pitch_radians)) {
    viewport.pitch_radians = 0.0F;
  }
  if (!std::isfinite(viewport.fov_degrees)) {
    viewport.fov_degrees = 75.0F;
  }
  viewport.yaw_radians = std::remainder(viewport.yaw_radians, 2.0F * kPi);
  viewport.pitch_radians =
      std::clamp(viewport.pitch_radians, -kMaximumPitchRadians, kMaximumPitchRadians);
  viewport.fov_degrees = std::clamp(viewport.fov_degrees, kMinimumFovDegrees, kMaximumFovDegrees);
  return viewport;
}

std::uint64_t gpu_preview_timestamp_for_frame(std::uint64_t frame_index,
                                              std::uint32_t fps_numerator,
                                              std::uint32_t fps_denominator) {
  if (fps_numerator == 0U || fps_denominator == 0U) {
    throw std::invalid_argument("GPU preview frame rate must be non-zero");
  }
  const auto scale = checked_multiply(kNanosecondsPerSecond, fps_denominator,
                                      "GPU preview frame-rate scale overflows");
  const auto whole_frames = frame_index / fps_numerator;
  const auto remaining_frames = frame_index % fps_numerator;
  const auto whole = checked_multiply(whole_frames, scale,
                                      "GPU preview timestamp exceeds the GStreamer time range");
  const auto fractional = multiply_divide_floor(remaining_frames, scale, fps_numerator);
  return checked_add(whole, fractional, "GPU preview timestamp exceeds the GStreamer time range");
}

std::string_view gpu_preview_controller_state_name(GpuPreviewControllerState state) noexcept {
  switch (state) {
  case GpuPreviewControllerState::Stopped:
    return "stopped";
  case GpuPreviewControllerState::Starting:
    return "starting";
  case GpuPreviewControllerState::Playing:
    return "playing";
  case GpuPreviewControllerState::Paused:
    return "paused";
  case GpuPreviewControllerState::Seeking:
    return "seeking";
  case GpuPreviewControllerState::EndOfStream:
    return "end_of_stream";
  case GpuPreviewControllerState::Error:
    return "error";
  }
  return "unknown";
}

class GpuPreviewController::Impl {
public:
  Impl(GpuPreviewControllerConfig config, std::unique_ptr<GpuPreviewControllerBackend> backend,
       NotificationCallback callback)
      : config_(std::move(config)), backend_(std::move(backend)) {
    if (const auto error = validate_gpu_preview_controller_config(config_); error.has_value()) {
      throw std::invalid_argument(*error);
    }
    if (!backend_) {
      throw std::invalid_argument("GPU preview controller backend must not be null");
    }
    state_.viewport = clamp_gpu_preview_viewport({});
    if (callback) {
      callback_slot_ = std::make_shared<CallbackSlot>(std::move(callback));
    }
  }

  ~Impl() { stop(); }

  void start() {
    {
      std::lock_guard lifecycle_lock(lifecycle_mutex_);
      {
        std::lock_guard lock(mutex_);
        if (worker_.joinable() || state_.state != GpuPreviewControllerState::Stopped) {
          throw std::logic_error("GPU preview controller is already active");
        }
        stopping_ = false;
        desired_playing_ = config_.start_playing;
        pending_seek_.reset();
        pause_rebuild_requested_ = false;
        pacing_reset_ = true;
        state_.state = GpuPreviewControllerState::Starting;
        state_.ready = false;
        state_.playing = false;
        state_.end_of_stream = false;
        state_.current_frame.reset();
        state_.total_frames = 0;
        state_.total_frames_exact = false;
        state_.presented_frames = 0;
        state_.presentation_timestamp_ns.reset();
        state_.fps_numerator = 0;
        state_.fps_denominator = 0;
        state_.error.clear();
      }
      try {
        backend_->prepare_start();
        worker_ = std::thread([this] { run(); });
      } catch (...) {
        std::lock_guard lock(mutex_);
        state_.state = GpuPreviewControllerState::Stopped;
        throw;
      }
    }
    notify();
  }

  void stop() noexcept {
    try {
      std::unique_lock lifecycle_lock(lifecycle_mutex_);
      {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        desired_playing_ = false;
      }
      backend_->shutdown();
      control_changed_.notify_all();
      if (worker_.joinable()) {
        if (worker_.get_id() == std::this_thread::get_id()) {
          return;
        }
        worker_.join();
      }
      backend_->release();
      {
        std::lock_guard lock(mutex_);
        set_stopped_locked();
      }
      lifecycle_lock.unlock();
      notify();
    } catch (...) {
    }
  }

  void play() {
    {
      std::lock_guard lock(mutex_);
      if (stopping_ || state_.state == GpuPreviewControllerState::Stopped ||
          state_.state == GpuPreviewControllerState::Error || state_.end_of_stream) {
        return;
      }
      desired_playing_ = true;
      pacing_reset_ = true;
      if (state_.ready && state_.state != GpuPreviewControllerState::Seeking) {
        state_.state = GpuPreviewControllerState::Playing;
        state_.playing = true;
      }
    }
    control_changed_.notify_all();
    notify();
  }

  void pause() {
    {
      std::lock_guard lock(mutex_);
      if (stopping_ || state_.state == GpuPreviewControllerState::Stopped ||
          state_.state == GpuPreviewControllerState::Error || state_.end_of_stream) {
        return;
      }
      desired_playing_ = false;
      pacing_reset_ = true;
      if (state_.ready && state_.state == GpuPreviewControllerState::Playing) {
        pause_rebuild_requested_ = true;
        state_.state = GpuPreviewControllerState::Seeking;
        state_.playing = false;
        backend_->interrupt_decode();
      }
    }
    control_changed_.notify_all();
    notify();
  }

  void seek(std::uint64_t frame_index) {
    {
      std::lock_guard lock(mutex_);
      if (stopping_ || state_.state == GpuPreviewControllerState::Stopped ||
          state_.state == GpuPreviewControllerState::Error || !state_.ready) {
        throw std::logic_error("GPU preview seek requires an initialized controller");
      }
      if (state_.total_frames != 0U && frame_index >= state_.total_frames) {
        throw std::out_of_range("GPU preview seek frame is outside the aligned input range");
      }
      pending_seek_ = frame_index;
      pause_rebuild_requested_ = false;
      pacing_reset_ = true;
      state_.end_of_stream = false;
      if (state_.ready) {
        state_.state = GpuPreviewControllerState::Seeking;
        state_.playing = false;
      }
      // Publish the seek to the worker only after the old decoder has observed
      // cancellation; otherwise a late interrupt can stop the rebuilt session.
      backend_->interrupt_decode();
    }
    control_changed_.notify_all();
    notify();
  }

  void set_viewport(GpuPreviewViewport viewport) {
    {
      std::lock_guard lock(mutex_);
      state_.viewport = clamp_gpu_preview_viewport(viewport);
    }
    notify();
  }

  void set_notification_callback(NotificationCallback callback) {
    auto replacement = callback ? std::make_shared<CallbackSlot>(std::move(callback)) : nullptr;
    std::shared_ptr<CallbackSlot> previous;
    {
      std::unique_lock lock(callback_mutex_);
      previous = std::exchange(callback_slot_, std::move(replacement));
      if (previous) {
        const std::size_t caller_invocations =
            g_active_preview_callback_slot == previous.get() ? 1U : 0U;
        callback_finished_.wait(lock, [&] { return previous->active <= caller_invocations; });
      }
    }
    notify();
  }

  GpuPreviewControllerSnapshot snapshot() const {
    std::lock_guard lock(mutex_);
    return state_;
  }

private:
  void run() noexcept {
    try {
      const auto info = backend_->initialize(config_);
      validate_backend_stream_info(info);
      bool stopped_during_initialization = false;
      {
        std::lock_guard lock(mutex_);
        if (stopping_) {
          set_stopped_locked();
          stopped_during_initialization = true;
        } else {
          state_.ready = true;
          state_.fps_numerator = info.fps_numerator;
          state_.fps_denominator = info.fps_denominator;
          state_.total_frames = info.total_frames;
          state_.total_frames_exact = info.total_frames_exact;
          state_.state = desired_playing_ ? GpuPreviewControllerState::Playing
                                          : GpuPreviewControllerState::Paused;
          state_.playing = desired_playing_;
        }
      }
      if (stopped_during_initialization) {
        backend_->shutdown();
        backend_->release();
        notify();
        return;
      }
      notify();

      std::uint64_t presentation_frame = 0U;
      auto deadline = std::chrono::steady_clock::now();
      for (;;) {
        std::optional<std::uint64_t> seek_frame;
        GpuPreviewViewport viewport;
        {
          std::unique_lock lock(mutex_);
          control_changed_.wait(lock, [&] {
            return stopping_ || pending_seek_.has_value() || pause_rebuild_requested_ ||
                   desired_playing_;
          });
          if (stopping_) {
            break;
          }
          if (pending_seek_.has_value()) {
            seek_frame = pending_seek_;
            pending_seek_.reset();
            pause_rebuild_requested_ = false;
          } else if (pause_rebuild_requested_) {
            if (state_.current_frame.has_value()) {
              seek_frame = *state_.current_frame < state_.total_frames - 1U
                               ? *state_.current_frame + 1U
                               : *state_.current_frame;
            } else {
              seek_frame = 0U;
            }
            pause_rebuild_requested_ = false;
          } else if (!desired_playing_) {
            continue;
          }
          if (pacing_reset_) {
            deadline = std::chrono::steady_clock::now();
            pacing_reset_ = false;
          }
          viewport = state_.viewport;
        }

        if (seek_frame.has_value()) {
          backend_->rebuild_decode(*seek_frame);
          deadline = std::chrono::steady_clock::now();
          {
            std::lock_guard lock(mutex_);
            if (stopping_) {
              break;
            }
            state_.end_of_stream = false;
            if (pending_seek_.has_value() || pause_rebuild_requested_) {
              state_.state = GpuPreviewControllerState::Seeking;
              state_.playing = false;
            } else {
              state_.state = desired_playing_ ? GpuPreviewControllerState::Playing
                                              : GpuPreviewControllerState::Paused;
              state_.playing = desired_playing_;
            }
          }
          notify();
          continue;
        }

        const auto pts = gpu_preview_timestamp_for_frame(presentation_frame, state_fps_numerator(),
                                                         state_fps_denominator());
        const auto next_pts = gpu_preview_timestamp_for_frame(
            checked_add(presentation_frame, 1U,
                        "GPU preview presentation frame counter overflowed"),
            state_fps_numerator(), state_fps_denominator());
        const auto duration = next_pts - pts;
        auto result = backend_->present_next(viewport, pts, duration);
        if (result.status == GpuPreviewBackendFrameStatus::Stopped) {
          std::lock_guard lock(mutex_);
          if (!stopping_ && !pending_seek_.has_value() && !pause_rebuild_requested_) {
            throw std::runtime_error("GPU preview decode stopped without a control request");
          }
          continue;
        }
        if (result.status == GpuPreviewBackendFrameStatus::EndOfStream) {
          bool control_superseded_eos = false;
          {
            std::lock_guard lock(mutex_);
            control_superseded_eos =
                stopping_ || pending_seek_.has_value() || pause_rebuild_requested_;
            if (!control_superseded_eos) {
              state_.state = GpuPreviewControllerState::EndOfStream;
              state_.ready = true;
              state_.playing = false;
              state_.end_of_stream = true;
              desired_playing_ = false;
            }
          }
          if (control_superseded_eos) {
            continue;
          }
          notify();
          continue;
        }
        if (!result.source_frame.has_value()) {
          throw std::runtime_error("GPU preview backend presented a frame without source identity");
        }

        {
          std::lock_guard lock(mutex_);
          state_.current_frame = result.source_frame;
          state_.presentation_timestamp_ns = pts;
          state_.presented_frames = checked_add(state_.presented_frames, 1U,
                                                "GPU preview presented-frame counter overflowed");
          if (pending_seek_.has_value() || pause_rebuild_requested_) {
            state_.state = GpuPreviewControllerState::Seeking;
            state_.playing = false;
          } else if (!desired_playing_) {
            state_.state = GpuPreviewControllerState::Paused;
            state_.playing = false;
          } else {
            state_.playing = true;
          }
        }
        presentation_frame = checked_add(presentation_frame, 1U,
                                         "GPU preview presentation frame counter overflowed");
        notify();

        if (duration > static_cast<std::uint64_t>(std::chrono::nanoseconds::max().count())) {
          throw std::overflow_error("GPU preview frame duration exceeds the steady clock range");
        }
        deadline += std::chrono::nanoseconds(duration);
        std::unique_lock lock(mutex_);
        control_changed_.wait_until(lock, deadline, [&] {
          return stopping_ || pending_seek_.has_value() || !desired_playing_ || pacing_reset_;
        });
      }

      backend_->shutdown();
      backend_->release();
      {
        std::lock_guard lock(mutex_);
        set_stopped_locked();
      }
      notify();
    } catch (const std::exception& error) {
      fail(error.what());
    } catch (...) {
      fail("unknown GPU preview controller failure");
    }
  }

  std::uint32_t state_fps_numerator() const {
    std::lock_guard lock(mutex_);
    return state_.fps_numerator;
  }

  std::uint32_t state_fps_denominator() const {
    std::lock_guard lock(mutex_);
    return state_.fps_denominator;
  }

  void fail(std::string error) noexcept {
    backend_->shutdown();
    backend_->release();
    {
      std::lock_guard lock(mutex_);
      if (stopping_) {
        set_stopped_locked();
      } else {
        state_.state = GpuPreviewControllerState::Error;
        state_.ready = false;
        state_.playing = false;
        state_.end_of_stream = false;
        state_.error = std::move(error);
      }
    }
    notify();
  }

  void set_stopped_locked() noexcept {
    state_.state = GpuPreviewControllerState::Stopped;
    state_.ready = false;
    state_.playing = false;
    state_.end_of_stream = false;
    state_.error.clear();
    stopping_ = true;
    desired_playing_ = false;
    pending_seek_.reset();
    pause_rebuild_requested_ = false;
  }

  void notify() noexcept {
    std::shared_ptr<CallbackSlot> slot;
    GpuPreviewControllerSnapshot snapshot;
    try {
      {
        std::lock_guard lock(callback_mutex_);
        slot = callback_slot_;
        if (slot) {
          ++slot->active;
        }
      }
      {
        std::lock_guard lock(mutex_);
        snapshot = state_;
      }
      if (slot) {
        const auto* previous_slot = g_active_preview_callback_slot;
        g_active_preview_callback_slot = slot.get();
        try {
          slot->callback(std::move(snapshot));
        } catch (...) {
        }
        g_active_preview_callback_slot = previous_slot;
        std::lock_guard lock(callback_mutex_);
        --slot->active;
        callback_finished_.notify_all();
      }
    } catch (...) {
      if (slot) {
        std::lock_guard lock(callback_mutex_);
        if (slot->active != 0U) {
          --slot->active;
          callback_finished_.notify_all();
        }
      }
    }
  }

  struct CallbackSlot {
    explicit CallbackSlot(NotificationCallback callback_value)
        : callback(std::move(callback_value)) {}

    NotificationCallback callback;
    std::size_t active = 0U;
  };

  GpuPreviewControllerConfig config_;
  std::unique_ptr<GpuPreviewControllerBackend> backend_;
  mutable std::mutex mutex_;
  std::mutex lifecycle_mutex_;
  std::mutex callback_mutex_;
  std::condition_variable callback_finished_;
  std::condition_variable control_changed_;
  GpuPreviewControllerSnapshot state_;
  std::shared_ptr<CallbackSlot> callback_slot_;
  std::thread worker_;
  std::optional<std::uint64_t> pending_seek_;
  bool pause_rebuild_requested_ = false;
  bool stopping_ = true;
  bool desired_playing_ = false;
  bool pacing_reset_ = true;
};

GpuPreviewController::GpuPreviewController(GpuPreviewControllerConfig config,
                                           NotificationCallback callback)
    : GpuPreviewController(std::move(config), std::make_unique<ProductionGpuPreviewBackend>(),
                           std::move(callback)) {}

GpuPreviewController::GpuPreviewController(GpuPreviewControllerConfig config,
                                           std::unique_ptr<GpuPreviewControllerBackend> backend,
                                           NotificationCallback callback)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(backend), std::move(callback))) {}

GpuPreviewController::~GpuPreviewController() = default;

void GpuPreviewController::start() { impl_->start(); }

void GpuPreviewController::stop() noexcept { impl_->stop(); }

void GpuPreviewController::play() { impl_->play(); }

void GpuPreviewController::pause() { impl_->pause(); }

void GpuPreviewController::seek(std::uint64_t frame_index) { impl_->seek(frame_index); }

void GpuPreviewController::set_viewport(GpuPreviewViewport viewport) {
  impl_->set_viewport(viewport);
}

void GpuPreviewController::set_notification_callback(NotificationCallback callback) {
  impl_->set_notification_callback(std::move(callback));
}

GpuPreviewControllerSnapshot GpuPreviewController::snapshot() const { return impl_->snapshot(); }

} // namespace reco::gui
