#include "reco/gui/telemetry.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

using namespace reco::gui;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

std::string synthetic_report(std::size_t lines) {
  std::string report =
      "## User description\ncrash on export\n\n## Contact\nuser@example.com\n\n## Environment\n- "
      "Reco v1.0\n- OS: linux x86_64\n";
  report += "\n## Log (last 200 lines)\n```\n";
  for (std::size_t i = 0; i < lines; ++i) {
    report += "[INFO] log line number " + std::to_string(i) +
              " padded with enough detail to take up realistic space\n";
  }
  report += "```\n";
  return report;
}

void timestamp_helpers_match_rust() {
  expect_eq(iso_from_unix_seconds(0), std::string("1970-01-01T00:00:00.000Z"),
            "epoch timestamp");
  expect_eq(iso_from_unix_seconds(86400), std::string("1970-01-02T00:00:00.000Z"),
            "day timestamp");
  expect_eq(iso_from_unix_seconds(1704067200), std::string("2024-01-01T00:00:00.000Z"),
            "modern timestamp");

  const auto [year, month, day] = civil_from_days(0);
  expect_eq(year, 1970, "civil year");
  expect_eq(month, 1U, "civil month");
  expect_eq(day, 1U, "civil day");
}

void payload_shapes_match_rust() {
  const auto context = context_props("RTX", "linux", "TensorRT");
  expect_eq(context.at("gpu").get<std::string>(), std::string("RTX"), "context gpu");
  expect_eq(context.at("os").get<std::string>(), std::string("linux"), "context os");
  expect_eq(context.at("ai").get<std::string>(), std::string("TensorRT"), "context ai");

  const auto source = source_info_props(3840, 2160, 59.94, "nvdec", -2);
  expect_eq(source.at("width").get<int>(), 3840, "source width");
  expect_eq(source.at("height").get<int>(), 2160, "source height");
  expect_eq(source.at("decoder").get<std::string>(), std::string("nvdec"), "source decoder");
  expect_eq(source.at("sync_offset").get<int>(), -2, "source sync");

  const auto complete = export_complete_props(300, 12.5, "h264");
  expect_eq(complete.at("frames").get<int>(), 300, "export frames");
  expect_eq(complete.at("codec").get<std::string>(), std::string("h264"), "export codec");

  const auto failed = export_error_props(std::string(600, 'x'), "hevc");
  expect_eq(failed.at("error_type").get<std::string>(), std::string("export_failed"),
            "export error type");
  expect_eq(failed.at("error_message").get<std::string>().size(), 500U,
            "export error truncated");

  const TelemetryEvent event{.ts = "2024-01-01T00:00:00.000Z",
                             .name = "context",
                             .client_id = "client",
                             .props = context};
  const TelemetryBatch batch{.client_id = "client",
                             .app_version = "0.5.4",
                             .sent_at = "2024-01-01T00:00:00.000Z",
                             .batch_id = "batch",
                             .events = {event}};
  const nlohmann::json json = batch;
  expect_eq(json.at("schema_version").get<int>(), 1, "batch schema");
  expect_eq(json.at("app").at("name").get<std::string>(), std::string("video-stitcher"),
            "batch app name");
  expect_eq(json.at("events").at(0).at("name").get<std::string>(), std::string("context"),
            "event name");
}

void bug_report_fitting_matches_rust() {
  const std::string small = "## User description\nit broke\n\n## Log (last 200 lines)\n```\nline\n```\n";
  expect_eq(fit_bug_report(small), small, "small report unchanged");

  const auto fitted = fit_bug_report(synthetic_report(400));
  expect_true(fitted.size() <= kMaxReportBytes, "oversized report under cap");
  expect_true(fitted.find("## User description") != std::string::npos, "keeps head");
  expect_true(fitted.find("user@example.com") != std::string::npos, "keeps contact");
  expect_true(fitted.find("log line number 399") != std::string::npos, "keeps newest");
  expect_true(fitted.find("log line number 0 ") == std::string::npos, "drops oldest");
  expect_true(fitted.find("lines dropped") != std::string::npos, "marks dropped lines");
  expect_true(fitted.ends_with("```\n"), "keeps closing fence");
  expect_true(!fitted.ends_with("```\n\n"), "no extra trailing blank line");

  const auto no_log = fit_bug_report(std::string(20000, 'x'));
  expect_true(no_log.size() <= kMaxReportBytes, "no log report capped");
  expect_true(no_log.starts_with("xxx"), "no log report head");

  const auto huge_head =
      fit_bug_report(std::string(20000, 'y') + "\n## Log (last 200 lines)\n```\nline\n```\n");
  expect_true(huge_head.size() <= kMaxReportBytes, "huge head capped");
  expect_true(huge_head.starts_with("yyy"), "huge head preserved");
}

void truncation_is_utf8_safe() {
  std::string text;
  for (int i = 0; i < 300; ++i) {
    text += "\xC3\xA9";
  }
  expect_eq(truncate_at_char_boundary(text, 499).size(), 498U, "utf8 boundary truncate");
  expect_eq(truncate_at_char_boundary("short", 500), std::string("short"), "short unchanged");
}

} // namespace

int main() {
  timestamp_helpers_match_rust();
  payload_shapes_match_rust();
  bug_report_fitting_matches_rust();
  truncation_is_utf8_safe();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
