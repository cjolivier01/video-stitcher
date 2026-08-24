#include "reco/control/intent_translator.hpp"

#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace reco::control {

IntentTranslator::IntentTranslator(PoseControl& pose) : pose_(&pose) {}

IntentTranslator& IntentTranslator::with_quality_handler(QualityHandler handler) {
  on_quality_ = std::move(handler);
  return *this;
}

IntentTranslator& IntentTranslator::with_capture_handler(CaptureHandler handler) {
  on_capture_ = std::move(handler);
  return *this;
}

IntentTranslator& IntentTranslator::with_model_select_handler(ModelSelectHandler handler) {
  on_model_select_ = std::move(handler);
  return *this;
}

void IntentTranslator::dispatch(const ControlIntent& intent) {
  std::visit(
      [this](const auto& item) {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, HotkeyControlIntent>) {
          pose_->apply_hotkey(item.value);
        } else if constexpr (std::is_same_v<T, PoseControlIntent>) {
          dispatch_pose(item.value);
        } else if constexpr (std::is_same_v<T, QualityControlIntent>) {
          if (on_quality_) {
            on_quality_(item.value);
          }
        } else if constexpr (std::is_same_v<T, CaptureControlIntent>) {
          if (on_capture_) {
            on_capture_(item.value);
          }
        } else if constexpr (std::is_same_v<T, ModelSelectControlIntent>) {
          if (on_model_select_) {
            on_model_select_(item.value);
          }
        }
      },
      intent);
}

void IntentTranslator::dispatch_all(std::span<const ControlIntent> intents) {
  for (const auto& intent : intents) {
    dispatch(intent);
  }
}

void IntentTranslator::dispatch_pose(const PoseIntent& intent) {
  const auto current = pose_->target_pose();
  std::visit(
      [this, current](const auto& item) {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, PoseSetYawRad>) {
          pose_->set_target({item.value, current.pitch, std::nullopt});
        } else if constexpr (std::is_same_v<T, PoseSetPitchRad>) {
          pose_->set_target({current.yaw, item.value, std::nullopt});
        } else if constexpr (std::is_same_v<T, PoseSetFovDeg>) {
          pose_->set_target_fov(item.value);
        } else if constexpr (std::is_same_v<T, PoseDeltaYawRad>) {
          pose_->set_target({current.yaw + item.value, current.pitch, std::nullopt});
        } else if constexpr (std::is_same_v<T, PoseDeltaPitchRad>) {
          pose_->set_target({current.yaw, current.pitch + item.value, std::nullopt});
        } else if constexpr (std::is_same_v<T, PoseDeltaFovDeg>) {
          pose_->set_target_fov(current.fov_degrees.value_or(pose_->current_fov_deg()) + item.value);
        } else if constexpr (std::is_same_v<T, PoseReset>) {
          pose_->apply_hotkey(HotkeyIntent::Reset);
        }
      },
      intent);
}

} // namespace reco::control

