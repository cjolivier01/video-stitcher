#pragma once

#include "reco/autocam/sweep_panner.hpp"
#include "reco/core/pipeline_event.hpp"
#include "reco/core/viewport_position.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

namespace reco::autocam {

enum class FramingMode {
  Action,
  FrameAll,
};

enum class ClusterMode {
  TrimmedMean,
  Density,
};

struct FieldPannerConfig {
  ClusterMode cluster_mode = ClusterMode::Density;
  float cluster_bandwidth_rad = 0.30F;
  float keep_fraction = 0.8F;
  std::size_t min_cluster = 2;
  float edge_push = 0.15F;
  float fov_alpha = 0.01F;
  float pitch_near = -0.05F;
  float pitch_far = 0.20F;
  float distance_bias_max = -12.0F;
  float edge_bias_max = 4.0F;
  float fov_tight = 22.0F;
  float fov_wide = 58.0F;
  float fov_default = 40.0F;
  float cluster_alpha = 0.012F;
  float max_velocity_rad_per_sec = 0.18F;
  float velocity_alpha = 0.06F;
  float pitch_bias = 0.05F;
  float ball_presence_decay = 0.90F;
  float ball_presence_attack = 0.15F;
  float velocity_fov_bias_max = 10.0F;
  float ball_frame_margin_deg = 3.0F;
  float ball_max_dist_from_cluster = 0.5F;
  float ball_weight = 0.5F;
  float lookahead_reactivity = 2.5F;
  float lead_gain = 1.3F;
  float lead_alpha = 0.1F;
  float dead_zone_rad = 0.20F;
  FramingMode framing = FramingMode::Action;
  bool confidence_weighted = true;
  float frame_all_margin_deg = 8.0F;
  bool lock_pitch = false;
  float locked_pitch_rad = 0.0F;

  [[nodiscard]] static FieldPannerConfig broadcast();
  [[nodiscard]] static FieldPannerConfig action();
  [[nodiscard]] static FieldPannerConfig frame_all();
  [[nodiscard]] static std::optional<FieldPannerConfig> from_preset_name(std::string_view name);
  [[nodiscard]] FieldPannerConfig sanitized() const;

  friend bool operator==(const FieldPannerConfig&, const FieldPannerConfig&) = default;
};

struct FieldCluster {
  float yaw = 0.0F;
  float pitch = 0.0F;
  float spread = 0.0F;
  std::size_t count = 0;
};

struct FieldPannerDebug {
  float cluster_yaw = 0.0F;
  float cluster_pitch = 0.0F;
  float cluster_spread = 0.0F;
  std::uint32_t n_players = 0;
  bool ball_near_cluster = false;
  float ball_presence = 0.0F;
  float effective_ball_weight = 0.0F;
  float target_yaw = 0.0F;
  float target_pitch = 0.0F;
  float fov_target = 0.0F;
};

class FieldPanner {
public:
  using Point = std::tuple<float, float, float>;

  explicit FieldPanner(float fps = 30.0F);
  FieldPanner(float fps, FieldPannerConfig config);

  [[nodiscard]] static FieldPanner with_config(float fps, FieldPannerConfig config);
  [[nodiscard]] FieldPanner with_ball_weight(float weight) const;
  [[nodiscard]] FieldPanner with_fov_range(float tight, float wide) const;
  [[nodiscard]] FieldPanner with_cluster_alpha(float alpha) const;

  [[nodiscard]] reco::core::ViewportPosition decide(const WorldState& world,
                                                    const PanContext& context);
  [[nodiscard]] reco::core::ViewportPosition
  decide_with_lookahead(const WorldState& world, const std::vector<WorldState>& future,
                        const PanContext& context);

  [[nodiscard]] std::vector<Point> cluster_and_trim(const std::vector<Point>& points) const;
  [[nodiscard]] std::vector<Point> densest_cluster(const std::vector<Point>& points) const;
  [[nodiscard]] float target_fov(float spread, float pitch, float velocity_mag) const;
  [[nodiscard]] std::optional<reco::core::PipelineEvent> debug_event(std::uint64_t frame_index) const;

  [[nodiscard]] float yaw() const { return yaw_; }
  [[nodiscard]] float pitch() const { return pitch_; }
  [[nodiscard]] float current_fov() const { return current_fov_; }
  void set_pose_for_test(float yaw, float pitch);

private:
  [[nodiscard]] std::vector<Point> to_points(const std::vector<reco::core::TrackedEntity>& players) const;
  [[nodiscard]] std::vector<Point> select_core(const std::vector<Point>& points) const;
  [[nodiscard]] std::optional<FieldCluster>
  compute_cluster(const std::vector<reco::core::TrackedEntity>& players);
  [[nodiscard]] std::optional<FieldCluster> cluster_action(const std::vector<Point>& points);
  [[nodiscard]] std::optional<FieldCluster> cluster_frame_all(const std::vector<Point>& points);
  [[nodiscard]] std::optional<std::pair<float, float>> raw_target(const WorldState& world) const;
  [[nodiscard]] std::pair<float, float> smooth_centroid(const std::vector<Point>& core);
  [[nodiscard]] std::pair<float, float> ema_step(float raw_yaw, float raw_pitch);

  FieldPannerConfig config_;
  float yaw_ = 0.0F;
  float pitch_ = 0.0F;
  float current_fov_ = 40.0F;
  float ema_yaw_ = 0.0F;
  float ema_pitch_ = 0.0F;
  bool ema_initialized_ = false;
  float velocity_yaw_ = 0.0F;
  float velocity_pitch_ = 0.0F;
  float max_velocity_ = 0.006F;
  float ball_presence_ = 0.0F;
  float last_ball_yaw_ = 0.0F;
  float last_ball_pitch_ = 0.0F;
  float lead_yaw_ = 0.0F;
  float lead_pitch_ = 0.0F;
  std::uint64_t frame_index_ = 0;
  bool lookahead_active_ = false;
  std::optional<FieldPannerDebug> last_debug_;
};

} // namespace reco::autocam
