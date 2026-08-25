#include "reco/autocam/field_panner.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <numeric>
#include <string>

namespace reco::autocam {
namespace {

constexpr float kPi = 3.14159265358979323846F;

float clamp(float value, float lo, float hi) { return std::clamp(value, lo, hi); }
float degrees(float rad) { return rad * 180.0F / kPi; }

float rust_min(float lhs, float rhs) {
  if (std::isnan(lhs)) {
    return rhs;
  }
  if (std::isnan(rhs)) {
    return lhs;
  }
  return std::min(lhs, rhs);
}

float rust_max(float lhs, float rhs) {
  if (std::isnan(lhs)) {
    return rhs;
  }
  if (std::isnan(rhs)) {
    return lhs;
  }
  return std::max(lhs, rhs);
}

float rust_clamp(float value, float lo, float hi) {
  if (std::isnan(value)) {
    return value;
  }
  return rust_min(rust_max(value, lo), hi);
}

float weight_for(float confidence, bool weighted) { return weighted ? confidence : 1.0F; }

std::optional<std::tuple<float, float, float, float>> bbox(const std::vector<FieldPanner::Point>& points) {
  float min_y = std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  float min_p = std::numeric_limits<float>::infinity();
  float max_p = -std::numeric_limits<float>::infinity();
  for (const auto& [yaw, pitch, confidence] : points) {
    (void)confidence;
    if (!std::isfinite(yaw) || !std::isfinite(pitch)) {
      continue;
    }
    min_y = std::min(min_y, yaw);
    max_y = std::max(max_y, yaw);
    min_p = std::min(min_p, pitch);
    max_p = std::max(max_p, pitch);
  }
  if (!std::isfinite(min_y) || !std::isfinite(max_y)) {
    return std::nullopt;
  }
  return std::tuple{0.5F * (min_y + max_y), 0.5F * (min_p + max_p), 0.5F * (max_y - min_y),
                    0.5F * (max_p - min_p)};
}

std::string lower(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

} // namespace

FieldPannerConfig FieldPannerConfig::broadcast() {
  FieldPannerConfig config;
  config.ball_weight = 0.20F;
  return config;
}

FieldPannerConfig FieldPannerConfig::action() {
  FieldPannerConfig config;
  config.dead_zone_rad = 0.12F;
  config.fov_tight = 20.0F;
  config.fov_wide = 48.0F;
  config.fov_default = 34.0F;
  config.lookahead_reactivity = 3.0F;
  config.ball_weight = 0.35F;
  config.edge_push = 0.20F;
  return config;
}

FieldPannerConfig FieldPannerConfig::frame_all() {
  FieldPannerConfig config;
  config.framing = FramingMode::FrameAll;
  config.confidence_weighted = false;
  config.frame_all_margin_deg = 10.0F;
  config.fov_wide = 70.0F;
  config.ball_weight = 0.0F;
  return config;
}

std::optional<FieldPannerConfig> FieldPannerConfig::from_preset_name(std::string_view name) {
  const auto normalized = lower(name);
  if (normalized == "broadcast") {
    return broadcast();
  }
  if (normalized == "action") {
    return action();
  }
  if (normalized == "frame_all") {
    return frame_all();
  }
  return std::nullopt;
}

FieldPannerConfig FieldPannerConfig::sanitized() const {
  auto out = *this;
  const float lo = rust_max(rust_min(out.fov_tight, out.fov_wide), 1.0F);
  const float hi = rust_max(rust_max(out.fov_tight, out.fov_wide), lo);
  out.fov_tight = lo;
  out.fov_wide = hi;
  out.fov_default = rust_clamp(out.fov_default, lo, hi);
  out.ball_weight = rust_clamp(out.ball_weight, 0.0F, 1.0F);
  out.keep_fraction = rust_clamp(out.keep_fraction, 0.01F, 1.0F);
  out.min_cluster = std::max<std::size_t>(out.min_cluster, 1);
  out.cluster_bandwidth_rad = rust_max(out.cluster_bandwidth_rad, 1.0e-3F);
  out.dead_zone_rad = rust_max(out.dead_zone_rad, 0.0F);
  out.ball_max_dist_from_cluster = rust_max(out.ball_max_dist_from_cluster, 0.0F);
  out.lead_gain = rust_max(out.lead_gain, 0.0F);
  out.lead_alpha = rust_clamp(out.lead_alpha, 1.0e-3F, 1.0F);
  out.cluster_alpha = rust_clamp(out.cluster_alpha, 1.0e-3F, 1.0F);
  out.fov_alpha = rust_clamp(out.fov_alpha, 1.0e-3F, 1.0F);
  out.velocity_alpha = rust_clamp(out.velocity_alpha, 1.0e-3F, 1.0F);
  out.lookahead_reactivity = rust_clamp(out.lookahead_reactivity, 1.0F, 10.0F);
  out.ball_presence_decay = rust_clamp(out.ball_presence_decay, 0.0F, 1.0F);
  out.ball_presence_attack = rust_clamp(out.ball_presence_attack, 0.0F, 1.0F);
  return out;
}

FieldPanner::FieldPanner(float fps) : FieldPanner(fps, FieldPannerConfig{}) {}

FieldPanner::FieldPanner(float fps, FieldPannerConfig config) : config_(config.sanitized()) {
  fps = clamp(fps, 1.0F, 1000.0F);
  current_fov_ = config_.fov_default;
  max_velocity_ = config_.max_velocity_rad_per_sec / fps;
}

FieldPanner FieldPanner::with_config(float fps, FieldPannerConfig config) {
  return FieldPanner(fps, config);
}

FieldPanner FieldPanner::with_ball_weight(float weight) const {
  auto copy = *this;
  copy.config_.ball_weight = clamp(weight, 0.0F, 1.0F);
  return copy;
}

FieldPanner FieldPanner::with_fov_range(float tight, float wide) const {
  auto copy = *this;
  copy.config_.fov_tight = rust_max(rust_min(tight, wide), 1.0F);
  copy.config_.fov_wide = rust_max(rust_max(tight, wide), copy.config_.fov_tight);
  return copy;
}

FieldPanner FieldPanner::with_cluster_alpha(float alpha) const {
  auto copy = *this;
  copy.config_.cluster_alpha = clamp(alpha, 0.001F, 1.0F);
  return copy;
}

void FieldPanner::set_pose_for_test(float yaw, float pitch) {
  yaw_ = yaw;
  pitch_ = pitch;
}

std::vector<FieldPanner::Point>
FieldPanner::to_points(const std::vector<reco::core::TrackedEntity>& players) const {
  std::vector<Point> points;
  points.reserve(players.size());
  for (const auto& player : players) {
    if (player.state == reco::core::TrackState::Lost) {
      continue;
    }
    points.emplace_back(player.yaw, player.pitch, player.confidence);
  }
  return points;
}

std::vector<FieldPanner::Point>
FieldPanner::cluster_and_trim(const std::vector<Point>& points) const {
  if (points.size() < config_.min_cluster) {
    return {};
  }
  float total_weight = 0.0F;
  float sum_yaw = 0.0F;
  float sum_pitch = 0.0F;
  for (const auto& [yaw, pitch, confidence] : points) {
    const float w = weight_for(confidence, config_.confidence_weighted);
    total_weight += w;
    sum_yaw += yaw * w;
    sum_pitch += pitch * w;
  }
  if (total_weight <= 0.0F) {
    return {};
  }
  const float cy = sum_yaw / total_weight;
  const float cp = sum_pitch / total_weight;
  if (!std::isfinite(cy) || !std::isfinite(cp)) {
    return {};
  }
  const float rounded_keep =
      std::round(static_cast<float>(points.size()) * rust_clamp(config_.keep_fraction, 0.0F, 1.0F));
  const auto raw_keep = std::isnan(rounded_keep) ? std::size_t{0} : static_cast<std::size_t>(rounded_keep);
  const auto keep_count = std::clamp<std::size_t>(raw_keep, config_.min_cluster, points.size());
  auto sorted = points;
  std::stable_sort(sorted.begin(), sorted.end(), [&](const auto& a, const auto& b) {
    const auto da =
        std::pow(std::get<0>(a) - cy, 2.0F) + std::pow(std::get<1>(a) - cp, 2.0F);
    const auto db =
        std::pow(std::get<0>(b) - cy, 2.0F) + std::pow(std::get<1>(b) - cp, 2.0F);
    if (std::isnan(da) || std::isnan(db)) {
      return false;
    }
    return da < db;
  });
  sorted.resize(keep_count);
  return sorted;
}

std::vector<FieldPanner::Point> FieldPanner::densest_cluster(const std::vector<Point>& points) const {
  if (points.size() < config_.min_cluster) {
    return {};
  }
  const float bandwidth = std::max(config_.cluster_bandwidth_rad, 1.0e-3F);
  const float bw_sq = bandwidth * bandwidth;
  const auto within = [&](const Point& a, const Point& b) {
    return std::pow(std::get<0>(a) - std::get<0>(b), 2.0F) +
               std::pow(std::get<1>(a) - std::get<1>(b), 2.0F) <=
           bw_sq;
  };
  const Point* center = nullptr;
  std::size_t best_count = 0;
  for (const auto& candidate : points) {
    if (!std::isfinite(std::get<0>(candidate)) || !std::isfinite(std::get<1>(candidate))) {
      continue;
    }
    const auto count = static_cast<std::size_t>(
        std::count_if(points.begin(), points.end(), [&](const auto& point) { return within(candidate, point); }));
    if (center == nullptr || count >= best_count) {
      center = &candidate;
      best_count = count;
    }
  }
  if (center == nullptr) {
    return {};
  }
  std::vector<Point> core;
  for (const auto& point : points) {
    if (within(*center, point)) {
      core.push_back(point);
    }
  }
  return core.size() < config_.min_cluster ? std::vector<Point>{} : core;
}

std::vector<FieldPanner::Point> FieldPanner::select_core(const std::vector<Point>& points) const {
  return config_.cluster_mode == ClusterMode::TrimmedMean ? cluster_and_trim(points)
                                                          : densest_cluster(points);
}

std::pair<float, float> FieldPanner::ema_step(float raw_yaw, float raw_pitch) {
  if (!ema_initialized_) {
    ema_yaw_ = raw_yaw;
    ema_pitch_ = raw_pitch;
    ema_initialized_ = true;
  } else {
    ema_yaw_ += config_.cluster_alpha * (raw_yaw - ema_yaw_);
    ema_pitch_ += config_.cluster_alpha * (raw_pitch - ema_pitch_);
  }
  return {ema_yaw_, ema_pitch_};
}

std::pair<float, float> FieldPanner::smooth_centroid(const std::vector<Point>& core) {
  float total_weight = 0.0F;
  float sum_yaw = 0.0F;
  float sum_pitch = 0.0F;
  for (const auto& [yaw, pitch, confidence] : core) {
    const float w = weight_for(confidence, config_.confidence_weighted);
    total_weight += w;
    sum_yaw += yaw * w;
    sum_pitch += pitch * w;
  }
  if (total_weight <= 0.0F) {
    return {ema_yaw_, ema_pitch_};
  }
  const float raw_yaw = sum_yaw / total_weight;
  const float raw_pitch = sum_pitch / total_weight;
  if (!std::isfinite(raw_yaw) || !std::isfinite(raw_pitch)) {
    return {ema_yaw_, ema_pitch_};
  }
  return ema_step(raw_yaw, raw_pitch);
}

std::optional<FieldCluster> FieldPanner::cluster_action(const std::vector<Point>& points) {
  const auto core = select_core(points);
  if (core.empty()) {
    return std::nullopt;
  }
  const auto [cy, cp] = smooth_centroid(core);
  float spread = 0.0F;
  for (const auto& [yaw, pitch, confidence] : core) {
    (void)confidence;
    const float dy = yaw - cy;
    const float dp = pitch - cp;
    spread = std::max(spread, std::sqrt(dy * dy + dp * dp));
  }
  return FieldCluster{.yaw = cy, .pitch = cp, .spread = spread, .count = core.size()};
}

std::optional<FieldCluster> FieldPanner::cluster_frame_all(const std::vector<Point>& points) {
  if (points.size() < config_.min_cluster) {
    return std::nullopt;
  }
  const auto box = bbox(points);
  if (!box.has_value()) {
    return std::nullopt;
  }
  const auto [cy, cp, half_yaw, half_pitch] = *box;
  const auto [yaw, pitch] = ema_step(cy, cp);
  return FieldCluster{
      .yaw = yaw, .pitch = pitch, .spread = std::max(half_yaw, half_pitch), .count = points.size()};
}

std::optional<FieldCluster>
FieldPanner::compute_cluster(const std::vector<reco::core::TrackedEntity>& players) {
  const auto points = to_points(players);
  return config_.framing == FramingMode::Action ? cluster_action(points) : cluster_frame_all(points);
}

std::optional<std::pair<float, float>> FieldPanner::raw_target(const WorldState& world) const {
  const auto points = to_points(world.players);
  if (config_.framing == FramingMode::Action) {
    const auto core = select_core(points);
    if (!core.empty()) {
      float total = 0.0F;
      float sum_yaw = 0.0F;
      float sum_pitch = 0.0F;
      for (const auto& [yaw, pitch, confidence] : core) {
        const float w = weight_for(confidence, config_.confidence_weighted);
        total += w;
        sum_yaw += yaw * w;
        sum_pitch += pitch * w;
      }
      if (total > 0.0F) {
        return std::pair{(sum_yaw / total) * (1.0F + config_.edge_push),
                         (sum_pitch / total) + config_.pitch_bias};
      }
    }
  } else if (points.size() >= config_.min_cluster) {
    if (const auto box = bbox(points); box.has_value()) {
      const auto [cy, cp, half_yaw, half_pitch] = *box;
      (void)half_yaw;
      (void)half_pitch;
      return std::pair{cy, cp};
    }
  }
  if (world.ball.has_value() && world.ball->state != reco::core::TrackState::Lost) {
    return std::pair{world.ball->yaw, world.ball->pitch};
  }
  return std::nullopt;
}

float FieldPanner::target_fov(float spread, float pitch, float velocity_mag) const {
  if (config_.framing == FramingMode::FrameAll) {
    return rust_clamp(degrees(2.0F * spread) + 2.0F * config_.frame_all_margin_deg, config_.fov_tight,
                      config_.fov_wide);
  }
  const float fov_from_spread = rust_max(degrees(2.0F * spread), config_.fov_tight);
  const float t_dist = rust_clamp((pitch - config_.pitch_near) / (config_.pitch_far - config_.pitch_near),
                                  0.0F, 1.0F);
  const float distance_bias = t_dist * config_.distance_bias_max;
  const float edge_bias = rust_min(std::abs(yaw_) * 5.0F, config_.edge_bias_max);
  const float vel_ratio = rust_clamp(velocity_mag / max_velocity_, 0.0F, 1.0F);
  const float velocity_bias = vel_ratio * config_.velocity_fov_bias_max;
  float fov = fov_from_spread + distance_bias + edge_bias + velocity_bias;
  if (ball_presence_ > 0.01F) {
    const float ball_offset =
        degrees(std::sqrt(std::pow(last_ball_yaw_ - yaw_, 2.0F) + std::pow(last_ball_pitch_ - pitch_, 2.0F)));
    fov = rust_max(fov, (ball_offset + config_.ball_frame_margin_deg) * 2.0F);
  }
  return rust_clamp(fov, config_.fov_tight, config_.fov_wide);
}

reco::core::ViewportPosition FieldPanner::decide(const WorldState& world, const PanContext& context) {
  return decide_with_lookahead(world, {}, context);
}

reco::core::ViewportPosition
FieldPanner::decide_with_lookahead(const WorldState& world, const std::vector<WorldState>& future,
                                   const PanContext&) {
  lookahead_active_ = lookahead_active_ || !future.empty();
  const float reactivity = lookahead_active_ ? rust_max(config_.lookahead_reactivity, 1.0F) : 1.0F;
  ++frame_index_;

  const auto cluster = compute_cluster(world.players);
  const bool ball_detected = config_.ball_weight > 0.0F && world.ball.has_value() &&
                             world.ball->state != reco::core::TrackState::Lost;
  const bool ball_near_cluster = ball_detected && cluster.has_value() && [&] {
    const float dy = world.ball->yaw - cluster->yaw;
    const float dp = world.ball->pitch - cluster->pitch;
    return std::sqrt(dy * dy + dp * dp) < config_.ball_max_dist_from_cluster;
  }();
  if (ball_near_cluster) {
    last_ball_yaw_ = world.ball->yaw;
    last_ball_pitch_ = world.ball->pitch;
    ball_presence_ += config_.ball_presence_attack * (1.0F - ball_presence_);
  } else {
    ball_presence_ *= config_.ball_presence_decay;
  }
  ball_presence_ = clamp(ball_presence_, 0.0F, 1.0F);

  std::optional<std::pair<float, float>> target;
  std::tuple<float, float, float> cluster_target{0.0F, 0.0F, 0.0F};
  if (cluster.has_value()) {
    if (config_.framing == FramingMode::Action) {
      float target_yaw = cluster->yaw * (1.0F + config_.edge_push);
      float target_pitch = cluster->pitch + config_.pitch_bias;
      const float effective_w = config_.ball_weight * ball_presence_;
      if (effective_w > 0.001F) {
        target_yaw = target_yaw * (1.0F - effective_w) + last_ball_yaw_ * effective_w;
        target_pitch = target_pitch * (1.0F - effective_w) + last_ball_pitch_ * effective_w;
      }
      cluster_target = {target_yaw, target_pitch, effective_w};
      target = std::pair{target_yaw, target_pitch};
    } else {
      cluster_target = {cluster->yaw, cluster->pitch, 0.0F};
      target = std::pair{cluster->yaw, cluster->pitch};
    }
  } else if (ball_detected) {
    target = std::pair{world.ball->yaw, world.ball->pitch};
  }

  if (!future.empty() && target.has_value()) {
    const auto current = raw_target(world);
    if (current.has_value()) {
      float sum_yaw = 0.0F;
      float sum_pitch = 0.0F;
      std::uint32_t n = 0;
      for (const auto& future_world : future) {
        if (const auto raw = raw_target(future_world); raw.has_value()) {
          sum_yaw += raw->first;
          sum_pitch += raw->second;
          ++n;
        }
      }
      if (n > 0) {
        const float raw_lead_yaw = (sum_yaw / static_cast<float>(n) - current->first) * config_.lead_gain;
        const float raw_lead_pitch = (sum_pitch / static_cast<float>(n) - current->second) * config_.lead_gain;
        lead_yaw_ += config_.lead_alpha * (raw_lead_yaw - lead_yaw_);
        lead_pitch_ += config_.lead_alpha * (raw_lead_pitch - lead_pitch_);
        target = std::pair{target->first + lead_yaw_, target->second + lead_pitch_};
      }
    }
  }

  if (config_.lock_pitch && target.has_value()) {
    target = std::pair{target->first, config_.locked_pitch_rad};
  }

  if (target.has_value() && std::isfinite(target->first) && std::isfinite(target->second)) {
    const float max_v = max_velocity_ * reactivity;
    const float v_alpha = rust_min(config_.velocity_alpha * reactivity, 1.0F);
    float err_yaw = target->first - yaw_;
    float err_pitch = target->second - pitch_;
    if (config_.dead_zone_rad > 0.0F) {
      const float mag = std::sqrt(err_yaw * err_yaw + err_pitch * err_pitch);
      if (mag > 1.0e-6F) {
        const float scale = std::max(mag - config_.dead_zone_rad, 0.0F) / mag;
        err_yaw *= scale;
        err_pitch *= scale;
      }
    }
    const float desired_yaw = clamp(err_yaw, -max_v, max_v);
    const float desired_pitch = clamp(err_pitch, -max_v, max_v);
    velocity_yaw_ += v_alpha * (desired_yaw - velocity_yaw_);
    velocity_pitch_ += v_alpha * (desired_pitch - velocity_pitch_);
    yaw_ += velocity_yaw_;
    pitch_ += velocity_pitch_;
  }

  if (cluster.has_value()) {
    const float vel_mag = std::sqrt(velocity_yaw_ * velocity_yaw_ + velocity_pitch_ * velocity_pitch_);
    const float target = target_fov(cluster->spread, cluster->pitch, vel_mag);
    if (std::isfinite(target)) {
      current_fov_ += config_.fov_alpha * (target - current_fov_);
    }
    const auto [target_yaw, target_pitch, effective_w] = cluster_target;
    last_debug_ = FieldPannerDebug{
        .cluster_yaw = cluster->yaw,
        .cluster_pitch = cluster->pitch,
        .cluster_spread = cluster->spread,
        .n_players = static_cast<std::uint32_t>(cluster->count),
        .ball_near_cluster = ball_near_cluster,
        .ball_presence = ball_presence_,
        .effective_ball_weight = effective_w,
        .target_yaw = target_yaw,
        .target_pitch = target_pitch,
        .fov_target = target,
    };
  } else {
    last_debug_ = std::nullopt;
  }
  return {.yaw = yaw_, .pitch = pitch_, .fov_degrees = current_fov_};
}

std::optional<reco::core::PipelineEvent> FieldPanner::debug_event(std::uint64_t frame_index) const {
  if (!last_debug_.has_value()) {
    return std::nullopt;
  }
  const auto& debug = *last_debug_;
  return reco::core::PipelineEvent(reco::core::PannerDebugEvent{
      .frame_index = frame_index,
      .cluster_yaw = debug.cluster_yaw,
      .cluster_pitch = debug.cluster_pitch,
      .cluster_spread = debug.cluster_spread,
      .n_players = debug.n_players,
      .ball_near_cluster = debug.ball_near_cluster,
      .ball_presence = debug.ball_presence,
      .effective_ball_weight = debug.effective_ball_weight,
      .target_yaw = debug.target_yaw,
      .target_pitch = debug.target_pitch,
      .fov_target = debug.fov_target,
  });
}

} // namespace reco::autocam
