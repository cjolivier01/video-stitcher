#include "reco/gui/telemetry.hpp"

#include <iomanip>
#include <sstream>

namespace reco::gui {
namespace {

bool is_utf8_continuation(unsigned char byte) { return (byte & 0b1100'0000U) == 0b1000'0000U; }

} // namespace

std::tuple<std::int64_t, std::uint32_t, std::uint32_t> civil_from_days(
    std::int64_t days_since_epoch) {
  auto z = days_since_epoch + 719468;
  const auto era = (z >= 0 ? z : z - 146096) / 146097;
  const auto doe = static_cast<std::uint32_t>(z - era * 146097);
  const auto yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  auto y = static_cast<std::int64_t>(yoe) + era * 400;
  const auto doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const auto mp = (5 * doy + 2) / 153;
  const auto d = doy - (153 * mp + 2) / 5 + 1;
  const auto m = mp < 10 ? mp + 3 : mp - 9;
  if (m <= 2) {
    ++y;
  }
  return {y, m, d};
}

std::string iso_from_unix_seconds(std::uint64_t seconds) {
  const auto h = (seconds / 3600) % 24;
  const auto m = (seconds / 60) % 60;
  const auto s = seconds % 60;
  const auto [year, month, day] = civil_from_days(static_cast<std::int64_t>(seconds / 86400));

  std::ostringstream out;
  out << year << '-' << std::setw(2) << std::setfill('0') << month << '-' << std::setw(2) << day
      << 'T' << std::setw(2) << h << ':' << std::setw(2) << m << ':' << std::setw(2) << s
      << ".000Z";
  return out.str();
}

std::string truncate_at_char_boundary(std::string_view text, std::size_t max) {
  if (text.size() <= max) {
    return std::string(text);
  }
  auto end = max;
  while (end > 0 && is_utf8_continuation(static_cast<unsigned char>(text[end]))) {
    --end;
  }
  return std::string(text.substr(0, end));
}

std::string fit_bug_report(std::string_view report) {
  if (report.size() <= kMaxReportBytes) {
    return std::string(report);
  }

  constexpr std::string_view marker = "\n## Log (last ";
  const auto marker_pos = report.find(marker);
  if (marker_pos == std::string_view::npos) {
    return truncate_at_char_boundary(report, kMaxReportBytes);
  }

  const auto head = report.substr(0, marker_pos);
  const auto log_rest = report.substr(marker_pos + marker.size());
  std::vector<std::string_view> lines;
  std::size_t start = 0;
  while (start < log_rest.size()) {
    const auto newline = log_rest.find('\n', start);
    lines.push_back(log_rest.substr(start, newline == std::string_view::npos
                                               ? std::string_view::npos
                                               : newline - start));
    if (newline == std::string_view::npos) {
      break;
    }
    start = newline + 1;
  }

  std::vector<std::string_view> content;
  if (lines.size() > 2) {
    content.assign(lines.begin() + 2, lines.end());
  }

  auto header = [](std::size_t dropped) {
    return "\n## Log (oldest " + std::to_string(dropped) + " lines dropped)\n```\n";
  };
  const auto worst_header = header(content.size());
  const auto overhead = head.size() + worst_header.size();
  if (overhead >= kMaxReportBytes) {
    return truncate_at_char_boundary(head, kMaxReportBytes);
  }

  const auto budget = kMaxReportBytes - overhead;
  std::size_t keep = 0;
  std::size_t used = 0;
  for (auto it = content.rbegin(); it != content.rend(); ++it) {
    if (used + it->size() + 1 > budget) {
      break;
    }
    used += it->size() + 1;
    ++keep;
  }

  const auto dropped = content.size() - keep;
  std::string out;
  out.reserve(overhead + used + 8);
  out.append(head);
  out.append(header(dropped));
  for (auto it = content.end() - static_cast<std::ptrdiff_t>(keep); it != content.end(); ++it) {
    out.append(*it);
    out.push_back('\n');
  }
  if (keep == 0) {
    out.append("```\n");
  }
  return out;
}

nlohmann::json context_props(std::string_view gpu, std::string_view os, std::string_view ai_status) {
  return nlohmann::json{{"os", os}, {"gpu", gpu}, {"ai", ai_status}};
}

nlohmann::json source_info_props(std::uint32_t width, std::uint32_t height, double fps,
                                 std::string_view decoder, std::int64_t sync_offset) {
  return nlohmann::json{{"width", width},
                        {"height", height},
                        {"fps", fps},
                        {"decoder", decoder},
                        {"sync_offset", sync_offset}};
}

nlohmann::json bug_report_props(std::string_view report) {
  return nlohmann::json{{"report", fit_bug_report(report)}};
}

nlohmann::json export_complete_props(std::uint64_t frames, double duration_secs,
                                     std::string_view codec) {
  return nlohmann::json{{"frames", frames}, {"duration_sec", duration_secs}, {"codec", codec}};
}

nlohmann::json export_error_props(std::string_view error, std::string_view codec) {
  return nlohmann::json{{"error_type", "export_failed"},
                        {"error_message", truncate_at_char_boundary(error, 500)},
                        {"codec", codec}};
}

nlohmann::json calibration_complete_props(double confidence, std::size_t matches) {
  return nlohmann::json{{"confidence", confidence}, {"matches", matches}};
}

nlohmann::json calibration_error_props(std::string_view error) {
  return nlohmann::json{{"error_message", truncate_at_char_boundary(error, 500)}};
}

void to_json(nlohmann::json& json, const TelemetryEvent& event) {
  json = nlohmann::json{{"schema_version", event.schema_version},
                        {"ts", event.ts},
                        {"name", event.name},
                        {"client_id", event.client_id},
                        {"props", event.props.has_value() ? *event.props : nlohmann::json()}};
}

void to_json(nlohmann::json& json, const TelemetryBatch& batch) {
  json = nlohmann::json{{"schema_version", batch.schema_version},
                        {"client_id", batch.client_id},
                        {"app",
                         nlohmann::json{{"name", batch.app_name}, {"version", batch.app_version}}},
                        {"sent_at", batch.sent_at},
                        {"batch_id", batch.batch_id},
                        {"events", batch.events}};
}

} // namespace reco::gui
