#pragma once

#include <functional>
#include <span>

#include "reco/control/intents.hpp"
#include "reco/control/pose_control.hpp"

namespace reco::control {

class IntentTranslator {
public:
  using QualityHandler = std::function<void(const QualityIntent&)>;
  using CaptureHandler = std::function<void(CaptureIntent)>;
  using ModelSelectHandler = std::function<void(const ModelSelectIntent&)>;

  explicit IntentTranslator(PoseControl& pose);

  IntentTranslator& with_quality_handler(QualityHandler handler);
  IntentTranslator& with_capture_handler(CaptureHandler handler);
  IntentTranslator& with_model_select_handler(ModelSelectHandler handler);

  void dispatch(const ControlIntent& intent);
  void dispatch_all(std::span<const ControlIntent> intents);

private:
  void dispatch_pose(const PoseIntent& intent);

  PoseControl* pose_;
  QualityHandler on_quality_;
  CaptureHandler on_capture_;
  ModelSelectHandler on_model_select_;
};

} // namespace reco::control

