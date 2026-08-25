#include "reco/autocam/ball_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace reco::autocam {

BallTracker::BallTracker(std::uint16_t class_id) : class_id_(class_id) {}

BallTracker BallTracker::with_max_jump_rad(float rad) const {
  auto copy = *this;
  copy.max_jump_rad_ = std::isnan(rad) ? 0.0F : std::max(rad, 0.0F);
  return copy;
}

BallTracker BallTracker::with_max_coast_frames(std::uint32_t frames) const {
  auto copy = *this;
  copy.coaster_ = Coaster(frames);
  return copy;
}

BallTracker BallTracker::with_player_anchor_rad(float rad) const {
  auto copy = *this;
  copy.player_anchor_max_rad_ = std::isnan(rad) ? 0.0F : std::max(rad, 0.0F);
  return copy;
}

void BallTracker::set_players(const std::vector<reco::core::TrackedEntity>& players) {
  current_players_.clear();
  current_players_.reserve(players.size());
  for (const auto& player : players) {
    current_players_.push_back({player.yaw, player.pitch});
  }
}

void BallTracker::observe_world(const WorldState& world) { set_players(world.players); }

std::optional<float> BallTracker::score(const reco::core::MappedDetection& detection) const {
  if (!detection.position.has_value()) {
    return std::nullopt;
  }
  const auto& pos = *detection.position;
  if (!last_.has_value()) {
    return -detection.confidence;
  }
  const float dy = pos.yaw - last_->yaw;
  const float dp = pos.pitch - last_->pitch;
  const float dist = std::sqrt(dy * dy + dp * dp);
  if (dist > max_jump_rad_) {
    return std::nullopt;
  }
  return dist - 0.1F * detection.confidence;
}

bool BallTracker::passes_player_anchor(float yaw, float pitch) const {
  if (current_players_.empty()) {
    return true;
  }
  return std::any_of(current_players_.begin(), current_players_.end(), [&](const auto& player) {
    const float dy = yaw - player.first;
    const float dp = pitch - player.second;
    return std::sqrt(dy * dy + dp * dp) <= player_anchor_max_rad_;
  });
}

std::vector<reco::core::TrackedEntity>
BallTracker::update(const std::vector<reco::core::MappedDetection>& detections, double) {
  const reco::core::MappedDetection* best = nullptr;
  float best_score = std::numeric_limits<float>::infinity();
  for (const auto& detection : detections) {
    if (detection.class_id != class_id_ || !detection.position.has_value()) {
      continue;
    }
    if (!passes_player_anchor(detection.position->yaw, detection.position->pitch)) {
      continue;
    }
    const auto scored = score(detection);
    if (scored.has_value() &&
        (best == nullptr || (!std::isnan(best_score) && !std::isnan(*scored) &&
                             best_score > *scored))) {
      best_score = *scored;
      best = &detection;
    }
  }

  if (best != nullptr) {
    const auto& pos = *best->position;
    (void)coaster_.accept_fresh();
    last_ = LastKnown{pos.yaw, pos.pitch, best->camera};
    age_frames_ =
        age_frames_ == std::numeric_limits<std::uint64_t>::max() ? age_frames_ : age_frames_ + 1;
    return {{
        .id = 0,
        .class_id = class_id_,
        .yaw = pos.yaw,
        .pitch = pos.pitch,
        .confidence = best->confidence,
        .state = reco::core::TrackState::Tracking,
        .age_frames = age_frames_,
        .origin = best->camera,
    }};
  }

  switch (coaster_.step_without_fresh()) {
  case CoastStatus::Coasting:
    if (!last_.has_value()) {
      return {};
    }
    age_frames_ =
        age_frames_ == std::numeric_limits<std::uint64_t>::max() ? age_frames_ : age_frames_ + 1;
    return {{
        .id = 0,
        .class_id = class_id_,
        .yaw = last_->yaw,
        .pitch = last_->pitch,
        .confidence = 0.0F,
        .state = reco::core::TrackState::Coasting,
        .age_frames = age_frames_,
        .origin = last_->origin,
    }};
  case CoastStatus::Lost:
    if (!last_.has_value()) {
      return {};
    }
    {
      const auto last = *last_;
      last_.reset();
      age_frames_ = 0;
      return {{
          .id = 0,
          .class_id = class_id_,
          .yaw = last.yaw,
          .pitch = last.pitch,
          .confidence = 0.0F,
          .state = reco::core::TrackState::Lost,
          .age_frames = 0,
          .origin = last.origin,
      }};
    }
  case CoastStatus::Tracking:
    return {};
  }
  return {};
}

} // namespace reco::autocam
