#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace reco::control {

enum class HotkeyIntent {
  YawLeft,
  YawRight,
  PitchUp,
  PitchDown,
  ZoomIn,
  ZoomOut,
  Reset,
  ToggleConstrained,
};

struct PoseSetYawRad {
  float value = 0.0F;
};
struct PoseSetPitchRad {
  float value = 0.0F;
};
struct PoseSetFovDeg {
  float value = 0.0F;
};
struct PoseDeltaYawRad {
  float value = 0.0F;
};
struct PoseDeltaPitchRad {
  float value = 0.0F;
};
struct PoseDeltaFovDeg {
  float value = 0.0F;
};
struct PoseReset {};

using PoseIntent = std::variant<PoseSetYawRad, PoseSetPitchRad, PoseSetFovDeg,
                                PoseDeltaYawRad, PoseDeltaPitchRad, PoseDeltaFovDeg, PoseReset>;

struct QualitySetCodec {
  std::string value;
};
struct QualitySetBitrate {
  std::uint64_t value = 0;
};
struct QualitySetResolution {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};
struct QualitySetPreset {
  std::string value;
};
struct QualitySetCrf {
  std::uint8_t value = 0;
};

using QualityIntent = std::variant<QualitySetCodec, QualitySetBitrate, QualitySetResolution,
                                   QualitySetPreset, QualitySetCrf>;

enum class CaptureIntent {
  StartRecord,
  StopRecord,
  Snapshot,
  ClearReplay,
  SaveReplay,
};

struct ModelSetDetectorModel {
  std::string value;
};
struct ModelSetDetectionInterval {
  std::uint64_t value = 1;
};
struct ModelDisableDetection {};

using ModelSelectIntent =
    std::variant<ModelSetDetectorModel, ModelSetDetectionInterval, ModelDisableDetection>;

struct HotkeyControlIntent {
  HotkeyIntent value = HotkeyIntent::Reset;
};
struct PoseControlIntent {
  PoseIntent value;
};
struct QualityControlIntent {
  QualityIntent value;
};
struct CaptureControlIntent {
  CaptureIntent value = CaptureIntent::Snapshot;
};
struct ModelSelectControlIntent {
  ModelSelectIntent value;
};

using ControlIntent = std::variant<HotkeyControlIntent, PoseControlIntent, QualityControlIntent,
                                   CaptureControlIntent, ModelSelectControlIntent>;

std::string to_json(const ControlIntent& intent);
std::string to_json(HotkeyIntent intent);
std::string to_json(CaptureIntent intent);

} // namespace reco::control

