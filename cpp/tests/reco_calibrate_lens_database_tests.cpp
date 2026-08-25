#include "reco/calibrate/lens_database.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

using namespace reco::calibrate;

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

void expect_near(double actual, double expected, double tolerance, std::string_view message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

std::string gyroflow_profile(std::string_view model, std::uint32_t width, std::uint32_t height,
                             std::string_view lens = "Wide") {
  return "{"
         "\"camera_brand\":\"GoPro\","
         "\"camera_model\":\"" +
         std::string(model) +
         "\","
         "\"lens_model\":\"" +
         std::string(lens) +
         "\","
         "\"camera_setting\":\"\","
         "\"resolution\":{\"w\":" +
         std::to_string(width) + ",\"h\":" + std::to_string(height) +
         "},"
         "\"camera_matrix\":{\"fx\":1000.0,\"fy\":1001.0,\"cx\":500.0,\"cy\":501.0},"
         "\"distortion_coeffs\":[0.1,0.2,0.3,0.4]"
         "}";
}

void load_json_formats_match_rust() {
  const auto flat =
      load_lens_from_json("{\"width\":1920,\"height\":1080,\"fx\":900.0,\"fy\":901.0,\"cx\":960.0,"
                          "\"cy\":540.0,\"d\":[0.01,0.02,0.03,0.04]}",
                          "flat");
  expect_eq(flat.width, 1920U, "flat width");
  expect_near(flat.d[3], 0.04, 1.0e-12, "flat distortion");

  const auto gyro = load_lens_from_json(gyroflow_profile("HERO10 Black", 3840, 2160), "gyro");
  expect_eq(gyro.width, 3840U, "gyroflow width");
  expect_near(gyro.fx, 1000.0, 1.0e-12, "gyroflow fx");

  const auto wrapped = load_lens_from_json(
      "{\"camera_brand\":\"DJI\",\"camera_model\":\"Action 4\","
      "\"calib_dimension\":{\"width\":1920,\"height\":1080},"
      "\"fisheye_params\":{\"camera_matrix\":[[800.0,0.0,400.0],[0.0,801.0,401.0],"
      "[0.0,0.0,1.0]],\"distortion_coeffs\":[1.0,2.0,3.0,4.0]}}",
      "wrapped");
  expect_eq(wrapped.height, 1080U, "wrapped height");
  expect_near(wrapped.cy, 401.0, 1.0e-12, "wrapped cy");

  bool invalid_threw = false;
  try {
    (void)load_lens_from_json("{\"not\":\"a profile\"}", "bad");
  } catch (const LensLoadError& err) {
    invalid_threw = err.kind == LensLoadErrorKind::UnrecognizedFormat;
  }
  expect_true(invalid_threw, "unrecognized JSON throws typed error");

  bool malformed_matrix_threw = false;
  try {
    (void)load_lens_from_json(
        "{\"camera_brand\":\"DJI\",\"camera_model\":\"Action 4\","
        "\"calib_dimension\":{\"width\":1920,\"height\":1080},"
        "\"fisheye_params\":{\"camera_matrix\":[[\"bad\",0.0,400.0],[0.0,801.0,401.0],"
        "[0.0,0.0,1.0]],\"distortion_coeffs\":[1.0,2.0,3.0,4.0]}}",
        "bad-matrix");
  } catch (const LensLoadError& err) {
    malformed_matrix_threw = err.kind == LensLoadErrorKind::UnrecognizedFormat;
  }
  expect_true(malformed_matrix_threw, "malformed matrix throws typed error");

  bool malformed_flat_distortion_threw = false;
  try {
    (void)load_lens_from_json(
        "{\"width\":1920,\"height\":1080,\"fx\":900.0,\"fy\":901.0,\"cx\":960.0,"
        "\"cy\":540.0,\"d\":[0.01,\"bad\",0.03,0.04]}",
        "bad-flat-d");
  } catch (const LensLoadError& err) {
    malformed_flat_distortion_threw = err.kind == LensLoadErrorKind::Parse;
  }
  expect_true(malformed_flat_distortion_threw, "malformed flat distortion throws typed error");

  bool oversized_flat_dimension_threw = false;
  try {
    (void)load_lens_from_json(
        "{\"width\":4294967296,\"height\":1080,\"fx\":900.0,\"fy\":901.0,\"cx\":960.0,"
        "\"cy\":540.0,\"d\":[0.01,0.02,0.03,0.04]}",
        "oversized-flat");
  } catch (const LensLoadError& err) {
    oversized_flat_dimension_threw = err.kind == LensLoadErrorKind::Parse;
  }
  expect_true(oversized_flat_dimension_threw, "oversized flat dimensions throw typed error");

  bool zero_flat_dimension_threw = false;
  try {
    (void)load_lens_from_json("{\"width\":0,\"height\":1080,\"fx\":900.0,\"fy\":901.0,\"cx\":960.0,"
                              "\"cy\":540.0,\"d\":[0.01,0.02,0.03,0.04]}",
                              "zero-flat");
  } catch (const LensLoadError& err) {
    zero_flat_dimension_threw = err.kind == LensLoadErrorKind::Parse;
  }
  expect_true(zero_flat_dimension_threw, "zero flat dimensions throw typed error");
}

void database_lookup_matches_rust_policy() {
  auto db = LensDatabase::load_empty();
  expect_true(db.is_empty(), "empty db");
  db.add_profile_from_json(gyroflow_profile("HERO10 Black", 3840, 2160, "Wide"), "hero10-wide");
  db.add_profile_from_json(gyroflow_profile("HERO10 Black", 2704, 1520, "Linear"), "hero10-linear");
  expect_eq(db.len(), 2U, "profile count");

  const auto exact = db.find("GoPro", "HERO10 Black", 3840, 2160, std::string("Wide"));
  expect_true(exact.has_value(), "exact fov match");
  expect_eq(exact->first.width, 3840U, "exact params width");
  expect_true(exact->second.source == ProfileSource::Database, "exact source");
  expect_eq(exact->second.camera, std::string("GoPro HERO10 Black"), "formatted camera");

  const auto scaled = db.find("GoPro", "HERO10 Black", 1920, 1080, std::string("Wide"));
  expect_true(scaled.has_value(), "same aspect scaled match");
  expect_eq(scaled->first.width, 1920U, "scaled width");
  expect_near(scaled->first.fx, 500.0, 1.0e-12, "scaled fx");

  const auto parent = db.find("GoPro", "HERO10 Black Mini", 3840, 2160, std::string("Wide"));
  expect_true(parent.has_value(), "variant lookup uses parent model");

  db.add_profile_from_json(gyroflow_profile("HERO11 Black", 3840, 2160, "Wide"), "hero11-wide");
  db.add_profile_from_json(gyroflow_profile("HERO11 Black Mini", 3840, 2160, "Linear"),
                           "hero11-mini-linear");
  const auto parent_fov = db.find("GoPro", "HERO11 Black Mini", 3840, 2160, std::string("Wide"));
  expect_true(parent_fov.has_value(), "variant with exact profiles uses parent fov fallback");
  expect_eq(parent_fov->second.camera, std::string("GoPro HERO11 Black"),
            "parent fov fallback camera");

  const auto telemetry =
      db.find_from_telemetry("GoPro", std::nullopt, 3840, 2160, std::string("Wide"));
  expect_true(!telemetry.has_value(), "telemetry without model falls back to camera type");

  const auto fallback = db.find_by_resolution(2704, 1520);
  expect_true(fallback.has_value(), "resolution fallback");
  expect_true(fallback->second.source == ProfileSource::Fallback, "fallback source");
}

void search_and_summary_match_rust_policy() {
  auto db = LensDatabase::load_empty();
  db.add_profile_from_json(gyroflow_profile("HERO10 Black", 3840, 2160, "Wide"), "hero10-wide");
  db.add_profile_from_json(gyroflow_profile("HERO11 Black", 1920, 1080, "Wide"), "hero11-wide");
  db.add_profile_from_json(gyroflow_profile("HERO11 Black", 1280, 720, "Narrow"), "hero11-narrow");

  const auto brands = db.brands();
  expect_eq(brands.size(), 1U, "brand count");
  expect_eq(brands[0], std::string("Gopro"), "brand title case matches Rust helper");

  const auto models = db.models_for_brand("GoPro");
  expect_eq(models.size(), 2U, "model count");
  expect_eq(models[0].first, std::string("Hero10 Black"), "model title case");
  expect_eq(models[1].second, 2U, "model profile count");

  const auto candidates = db.candidates(1920, 1080);
  expect_eq(candidates.size(), 1U, "candidate dimension filter");
  expect_eq(candidates[0].camera, std::string("GoPro HERO11 Black"), "candidate camera");

  const auto results = db.search("hero11 wide", 1920, 1080);
  expect_eq(results.size(), 1U, "multi-word search");
  expect_eq(results[0].width, 1920U, "search exact ranking");

  const auto params = db.load_by_summary(results[0]);
  expect_true(params.has_value(), "load by summary");
  expect_near(params->fy, 1001.0, 1.0e-12, "summary params fy");
}

} // namespace

int main() {
  load_json_formats_match_rust();
  database_lookup_matches_rust_policy();
  search_and_summary_match_rust_policy();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
