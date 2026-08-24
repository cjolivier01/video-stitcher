#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace reco::core {

struct FrameTiming {
  std::optional<std::chrono::nanoseconds> decode;
  std::optional<std::chrono::nanoseconds> upload;
  std::optional<std::chrono::nanoseconds> stitch;
  std::optional<std::chrono::nanoseconds> readback;
  std::optional<std::chrono::nanoseconds> submit;
  std::optional<std::chrono::nanoseconds> detection;
  std::optional<std::chrono::nanoseconds> tracking;
  std::optional<std::chrono::nanoseconds> total;
};

enum class PipelineStage {
  Decode,
  Upload,
  Stitch,
  Readback,
  Submit,
  Detection,
  Tracking,
};

[[nodiscard]] std::string pipeline_stage_name(PipelineStage stage);

struct TelemetrySnapshot {
  std::uint64_t frames_processed = 0;
  std::chrono::nanoseconds elapsed{};
  float fps_average = 0.0F;
  float fps_recent = 0.0F;
  float avg_decode_ms = 0.0F;
  float avg_upload_ms = 0.0F;
  float avg_stitch_ms = 0.0F;
  float avg_readback_ms = 0.0F;
  float avg_submit_ms = 0.0F;
  float avg_encode_worker_ms = 0.0F;
  std::uint64_t backpressure_stalls = 0;
  float backpressure_ms = 0.0F;
  float avg_detection_ms = 0.0F;
  float avg_total_ms = 0.0F;
  float p99_total_ms = 0.0F;
  float max_frame_ms = 0.0F;
  std::uint64_t total_detections = 0;
  float detections_per_frame = 0.0F;
  float ball_presence_pct = 0.0F;
  std::uint32_t active_tracks = 0;
  std::string gpu_name;
  std::optional<std::string> encoder_name;
  std::optional<std::string> decode_mode;
  std::optional<PipelineStage> bottleneck;
};

class TelemetrySink {
public:
  virtual ~TelemetrySink() = default;
  virtual void on_snapshot(const TelemetrySnapshot& snapshot) = 0;
};

class TelemetryCollector {
public:
  TelemetryCollector();

  void set_gpu_name(std::string name);
  void set_encoder_name(std::string name);
  void set_decode_mode(std::string mode);
  void set_sink(std::unique_ptr<TelemetrySink> sink, std::uint64_t interval_frames);

  void record_frame(const FrameTiming& timing);
  void record_detections(std::uint32_t count, std::uint32_t active_tracks, bool ball_present);
  [[nodiscard]] TelemetrySnapshot snapshot() const;

private:
  [[nodiscard]] TelemetrySnapshot build_snapshot() const;
  [[nodiscard]] float
  percentile_ms(const std::optional<std::chrono::nanoseconds> FrameTiming::* field,
                float percentile) const;

  FrameTiming ring_[256]{};
  std::uint64_t frame_count_ = 0;
  std::size_t ring_head_ = 0;
  std::size_t ring_len_ = 0;
  std::uint64_t total_detections_ = 0;
  std::uint64_t ball_present_frames_ = 0;
  std::uint32_t active_tracks_ = 0;
  std::chrono::nanoseconds max_frame_time_{};
  std::optional<std::chrono::steady_clock::time_point> session_start_;
  std::string gpu_name_;
  std::optional<std::string> encoder_name_;
  std::optional<std::string> decode_mode_;
  std::unique_ptr<TelemetrySink> sink_;
  std::uint64_t sink_interval_ = 30;
};

[[nodiscard]] std::string session_summary_text(const TelemetrySnapshot& snapshot);

} // namespace reco::core
