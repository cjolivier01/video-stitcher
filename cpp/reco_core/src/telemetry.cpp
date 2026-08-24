#include "reco/core/telemetry.hpp"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <utility>
#include <vector>

namespace reco::core {
namespace {

float millis(std::chrono::nanoseconds duration) {
  return static_cast<float>(
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count());
}

} // namespace

std::string pipeline_stage_name(PipelineStage stage) {
  switch (stage) {
  case PipelineStage::Decode:
    return "decode";
  case PipelineStage::Upload:
    return "upload";
  case PipelineStage::Stitch:
    return "stitch";
  case PipelineStage::Readback:
    return "readback";
  case PipelineStage::Submit:
    return "submit";
  case PipelineStage::Detection:
    return "detection";
  case PipelineStage::Tracking:
    return "tracking";
  }
  return "unknown";
}

TelemetryCollector::TelemetryCollector() = default;

void TelemetryCollector::set_gpu_name(std::string name) { gpu_name_ = std::move(name); }

void TelemetryCollector::set_encoder_name(std::string name) { encoder_name_ = std::move(name); }

void TelemetryCollector::set_decode_mode(std::string mode) { decode_mode_ = std::move(mode); }

void TelemetryCollector::set_sink(std::unique_ptr<TelemetrySink> sink,
                                  std::uint64_t interval_frames) {
  sink_ = std::move(sink);
  sink_interval_ = std::max<std::uint64_t>(interval_frames, 1);
}

void TelemetryCollector::record_frame(const FrameTiming& timing) {
  if (!session_start_.has_value()) {
    session_start_ = std::chrono::steady_clock::now();
  }
  ++frame_count_;
  if (timing.total.has_value() && *timing.total > max_frame_time_) {
    max_frame_time_ = *timing.total;
  }
  ring_[ring_head_] = timing;
  ring_head_ = (ring_head_ + 1) % std::size(ring_);
  if (ring_len_ < std::size(ring_)) {
    ++ring_len_;
  }
  if (sink_ != nullptr && frame_count_ % sink_interval_ == 0) {
    auto sink = std::move(sink_);
    sink->on_snapshot(build_snapshot());
    sink_ = std::move(sink);
  }
}

void TelemetryCollector::record_detections(std::uint32_t count, std::uint32_t active_tracks,
                                           bool ball_present) {
  total_detections_ += count;
  active_tracks_ = active_tracks;
  if (ball_present) {
    ++ball_present_frames_;
  }
}

TelemetrySnapshot TelemetryCollector::snapshot() const { return build_snapshot(); }

float TelemetryCollector::percentile_ms(
    const std::optional<std::chrono::nanoseconds> FrameTiming::* field, float percentile) const {
  std::vector<float> values;
  values.reserve(ring_len_);
  const std::size_t start = ring_len_ < std::size(ring_) ? 0 : ring_head_;
  for (std::size_t i = 0; i < ring_len_; ++i) {
    const auto& timing = ring_[(start + i) % std::size(ring_)];
    if ((timing.*field).has_value()) {
      values.push_back(millis(*(timing.*field)));
    }
  }
  if (values.empty()) {
    return 0.0F;
  }
  std::sort(values.begin(), values.end());
  const auto index = std::min<std::size_t>(static_cast<std::size_t>(values.size() * percentile),
                                           values.size() - 1);
  return values[index];
}

TelemetrySnapshot TelemetryCollector::build_snapshot() const {
  TelemetrySnapshot snapshot;
  snapshot.frames_processed = frame_count_;
  snapshot.elapsed = session_start_.has_value() ? std::chrono::steady_clock::now() - *session_start_
                                                : std::chrono::nanoseconds{};
  const auto elapsed_seconds = std::chrono::duration<float>(snapshot.elapsed).count();
  snapshot.fps_average =
      elapsed_seconds > 0.0F ? static_cast<float>(frame_count_) / elapsed_seconds : 0.0F;

  auto average_ms = [&](const std::optional<std::chrono::nanoseconds> FrameTiming::* field) {
    if (ring_len_ == 0) {
      return 0.0F;
    }
    float sum = 0.0F;
    std::uint32_t count = 0;
    const std::size_t start = ring_len_ < std::size(ring_) ? 0 : ring_head_;
    for (std::size_t i = 0; i < ring_len_; ++i) {
      const auto& timing = ring_[(start + i) % std::size(ring_)];
      if ((timing.*field).has_value()) {
        sum += millis(*(timing.*field));
        ++count;
      }
    }
    return count > 0 ? sum / static_cast<float>(count) : 0.0F;
  };

  snapshot.avg_decode_ms = average_ms(&FrameTiming::decode);
  snapshot.avg_upload_ms = average_ms(&FrameTiming::upload);
  snapshot.avg_stitch_ms = average_ms(&FrameTiming::stitch);
  snapshot.avg_readback_ms = average_ms(&FrameTiming::readback);
  snapshot.avg_submit_ms = average_ms(&FrameTiming::submit);
  snapshot.avg_detection_ms = average_ms(&FrameTiming::detection);
  snapshot.avg_total_ms = average_ms(&FrameTiming::total);
  snapshot.p99_total_ms = percentile_ms(&FrameTiming::total, 0.99F);
  snapshot.max_frame_ms = millis(max_frame_time_);

  float total_secs = 0.0F;
  std::uint32_t timed_frames = 0;
  const std::size_t start = ring_len_ < std::size(ring_) ? 0 : ring_head_;
  for (std::size_t i = 0; i < ring_len_; ++i) {
    const auto& timing = ring_[(start + i) % std::size(ring_)];
    if (timing.total.has_value()) {
      total_secs += std::chrono::duration<float>(*timing.total).count();
      ++timed_frames;
    }
  }
  snapshot.fps_recent = timed_frames > 1 && total_secs > 0.0F
                            ? static_cast<float>(timed_frames) / total_secs
                            : snapshot.fps_average;

  snapshot.total_detections = total_detections_;
  snapshot.detections_per_frame =
      frame_count_ > 0 ? static_cast<float>(total_detections_) / static_cast<float>(frame_count_)
                       : 0.0F;
  snapshot.ball_presence_pct = frame_count_ > 0 ? static_cast<float>(ball_present_frames_) /
                                                      static_cast<float>(frame_count_) * 100.0F
                                                : 0.0F;
  snapshot.active_tracks = active_tracks_;
  snapshot.gpu_name = gpu_name_;
  snapshot.encoder_name = encoder_name_;
  snapshot.decode_mode = decode_mode_;

  const std::pair<PipelineStage, float> stages[] = {
      {PipelineStage::Decode, snapshot.avg_decode_ms + snapshot.avg_upload_ms},
      {PipelineStage::Stitch, snapshot.avg_stitch_ms},
      {PipelineStage::Readback, snapshot.avg_readback_ms},
      {PipelineStage::Submit, snapshot.avg_submit_ms},
      {PipelineStage::Detection, snapshot.avg_detection_ms},
  };
  for (const auto& [stage, ms] : stages) {
    if (ms <= 0.1F) {
      continue;
    }
    if (!snapshot.bottleneck.has_value()) {
      snapshot.bottleneck = stage;
    } else {
      const auto current =
          std::find_if(std::begin(stages), std::end(stages),
                       [&](const auto& item) { return item.first == *snapshot.bottleneck; });
      if (current != std::end(stages) && ms >= current->second) {
        snapshot.bottleneck = stage;
      }
    }
  }

  return snapshot;
}

std::string session_summary_text(const TelemetrySnapshot& snapshot) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1);
  out << "--- Session Summary ---\n";
  out << "Frames:    " << snapshot.frames_processed << " processed\n";
  out << "Duration:  " << std::chrono::duration<double>(snapshot.elapsed).count() << "s\n";
  out << "FPS:       " << snapshot.fps_average << " avg, " << snapshot.fps_recent << " recent\n";
  out << "GPU:       " << snapshot.gpu_name << "\n";
  if (snapshot.encoder_name.has_value()) {
    out << "Encoder:   " << *snapshot.encoder_name << "\n";
  }
  if (snapshot.decode_mode.has_value()) {
    out << "Decode:    " << *snapshot.decode_mode << "\n";
  }
  out << "\nPer-frame timing (avg / p99):\n";
  const float total_decode = snapshot.avg_decode_ms + snapshot.avg_upload_ms;
  if (snapshot.avg_upload_ms > 0.05F) {
    out << "  Decode:    " << total_decode << " ms (wait " << snapshot.avg_decode_ms
        << " + staging " << snapshot.avg_upload_ms << ")\n";
  } else {
    out << "  Decode:    " << total_decode << " ms\n";
  }
  out << "  Stitch:    " << snapshot.avg_stitch_ms << " ms\n";
  out << "  Readback:  " << snapshot.avg_readback_ms << " ms\n";
  out << "  Submit:    " << snapshot.avg_submit_ms << " ms\n";
  if (snapshot.avg_encode_worker_ms > 0.0F || snapshot.backpressure_stalls > 0) {
    out << "  Encode:    " << snapshot.avg_encode_worker_ms << " ms (overlapped); backpressure "
        << snapshot.backpressure_stalls << " stalls / " << snapshot.backpressure_ms << " ms\n";
  }
  if (snapshot.avg_detection_ms > 0.0F) {
    out << "  Detection: " << snapshot.avg_detection_ms << " ms\n";
  }
  out << "  Total:     " << snapshot.avg_total_ms << " / " << snapshot.p99_total_ms << " ms\n";
  if (snapshot.bottleneck.has_value()) {
    out << "  Bottleneck: " << pipeline_stage_name(*snapshot.bottleneck) << "\n";
  }
  if (snapshot.total_detections > 0) {
    out << "\nDetection:\n";
    out << "  Ball present:  " << std::setprecision(0) << snapshot.ball_presence_pct
        << std::setprecision(1) << "% of frames\n";
    out << "  Detections:    " << snapshot.detections_per_frame << "/frame\n";
  }
  return out.str();
}

} // namespace reco::core
