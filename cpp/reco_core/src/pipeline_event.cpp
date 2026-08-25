#include "reco/core/pipeline_event.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <utility>

namespace reco::core {
namespace {

std::optional<std::uint32_t> to_micros(std::optional<std::chrono::nanoseconds> duration) {
  if (!duration.has_value()) {
    return std::nullopt;
  }
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(*duration).count();
  return static_cast<std::uint32_t>(micros);
}

} // namespace

std::string camera_id_name(CameraId camera) {
  switch (camera) {
  case CameraId::Left:
    return "L";
  case CameraId::Right:
    return "R";
  }
  return "?";
}

FrameTimingMicros FrameTimingMicros::from_frame_timing(const FrameTiming& timing) {
  return {
      to_micros(timing.decode),
      to_micros(timing.upload),
      to_micros(timing.stitch),
      to_micros(timing.readback),
      to_micros(timing.submit),
      to_micros(timing.detection),
      to_micros(timing.total).value_or(0),
  };
}

PipelineEvent::PipelineEvent(Variant event) : event_(std::move(event)) {}

std::uint64_t PipelineEvent::frame_index() const {
  return std::visit([](const auto& event) { return event.frame_index; }, event_);
}

const PipelineEvent::Variant& PipelineEvent::variant() const { return event_; }

} // namespace reco::core
