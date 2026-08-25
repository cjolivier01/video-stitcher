#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

namespace reco::gui {

inline constexpr std::string_view kTelemetryEndpoint =
    "https://telemetry-ingestion-204135919265.us-central1.run.app/telemetry";
inline constexpr std::string_view kTelemetryAppName = "video-stitcher";
inline constexpr std::size_t kMaxReportBytes = 16 * 1024;

struct TelemetryEvent {
  std::uint32_t schema_version = 1;
  std::string ts;
  std::string name;
  std::string client_id;
  std::optional<nlohmann::json> props;
};

struct TelemetryBatch {
  std::uint32_t schema_version = 1;
  std::string client_id;
  std::string app_name = std::string(kTelemetryAppName);
  std::string app_version;
  std::string sent_at;
  std::string batch_id;
  std::vector<TelemetryEvent> events;
};

[[nodiscard]] std::tuple<std::int64_t, std::uint32_t, std::uint32_t> civil_from_days(
    std::int64_t days_since_epoch);
[[nodiscard]] std::string iso_from_unix_seconds(std::uint64_t seconds);
[[nodiscard]] std::string truncate_at_char_boundary(std::string_view text, std::size_t max);
[[nodiscard]] std::string fit_bug_report(std::string_view report);

[[nodiscard]] nlohmann::json context_props(std::string_view gpu, std::string_view os,
                                           std::string_view ai_status);
[[nodiscard]] nlohmann::json source_info_props(std::uint32_t width, std::uint32_t height,
                                               double fps, std::string_view decoder,
                                               std::int64_t sync_offset);
[[nodiscard]] nlohmann::json bug_report_props(std::string_view report);
[[nodiscard]] nlohmann::json export_complete_props(std::uint64_t frames, double duration_secs,
                                                   std::string_view codec);
[[nodiscard]] nlohmann::json export_error_props(std::string_view error, std::string_view codec);
[[nodiscard]] nlohmann::json calibration_complete_props(double confidence, std::size_t matches);
[[nodiscard]] nlohmann::json calibration_error_props(std::string_view error);

void to_json(nlohmann::json& json, const TelemetryEvent& event);
void to_json(nlohmann::json& json, const TelemetryBatch& batch);

} // namespace reco::gui
