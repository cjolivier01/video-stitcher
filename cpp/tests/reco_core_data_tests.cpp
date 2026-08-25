#include "reco/core/calibration.hpp"
#include "reco/core/source.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace reco::core;

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
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

std::filesystem::path scratch_dir() {
  auto path = std::filesystem::temp_directory_path() / "reco_core_data_tests";
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

MatchCalibration valid_calibration() {
  MatchCalibration cal;
  cal.left = {3840, 2160, 1796.32, 1797.22, 1919.37, 1063.17, {0.0342, 0.0677, -0.0741, 0.0299}};
  cal.right = cal.left;
  cal.layout.camera_axis_offset = 0.2398;
  cal.layout.intersect = 0.5446;
  cal.layout.x_ty = 0.00476;
  cal.layout.x_rz = 0.00753;
  cal.layout.z_rx = -0.00431;
  cal.sync_offset = 0;
  return cal;
}

void source_path_validation_matches_rust_reasons() {
  const auto dir = scratch_dir();
  const auto valid = dir / "video.mp4";
  {
    std::ofstream output(valid, std::ios::binary);
    output << "not a real mp4 but non-empty";
  }
  expect_true(validate_input_path(valid).ok, "valid file accepted");

  const auto missing = validate_input_path(dir / "missing.mp4");
  expect_true(!missing.ok, "missing rejected");
  expect_true(missing.reason == InvalidPathReason::NotFound, "missing reason");

  const auto directory = validate_input_path(dir);
  expect_true(!directory.ok, "directory rejected");
  expect_true(directory.reason == InvalidPathReason::NotAFile, "directory reason");

  const auto empty_path = dir / "empty.mp4";
  std::ofstream(empty_path, std::ios::binary).close();
  const auto empty = validate_input_path(empty_path);
  expect_true(!empty.ok, "empty rejected");
  expect_true(empty.reason == InvalidPathReason::Empty, "empty reason");
  expect_eq(invalid_path_reason_name(InvalidPathReason::Empty), "file is empty", "empty label");
}

void frame_plane_validation_matches_rust_messages() {
  YuvFrame frame;
  frame.width = 4;
  frame.height = 2;
  frame.y.resize(8);
  frame.u.resize(2);
  frame.v.resize(2);
  expect_true(frame.validate().empty(), "valid yuv frame");
  frame.u.pop_back();
  expect_true(frame.validate().find("U plane size mismatch") != std::string::npos,
              "u plane mismatch");
}

void camera_input_contract_matches_rust() {
  StereoCameraInput stereo;
  MonoCameraInput mono;
  expect_eq(std::string_view(stereo.name()), "stereo-2camera", "stereo name");
  expect_eq(stereo.camera_count(), 2, "stereo count");
  expect_eq(std::string_view(mono.name()), "mono-1camera", "mono name");
  expect_eq(mono.camera_count(), 1, "mono count");
  std::vector<std::unique_ptr<CameraInput>> inputs;
  inputs.emplace_back(std::make_unique<StereoCameraInput>());
  inputs.emplace_back(std::make_unique<MonoCameraInput>());
  expect_eq(inputs[0]->camera_count(), 2, "dyn stereo count");
  expect_eq(inputs[1]->camera_count(), 1, "dyn mono count");
}

void calibration_validation_matches_rust_guards() {
  auto cal = valid_calibration();
  expect_true(cal.validate().empty(), "valid calibration");

  cal.left.fx = 0.0;
  expect_true(cal.validate().find("left.fx") != std::string::npos, "zero fx rejected");
  cal = valid_calibration();
  cal.right.height = kMaxCalibrationDimension + 1;
  expect_true(cal.validate().find("right camera height exceeds") != std::string::npos,
              "large dimension rejected");
  cal = valid_calibration();
  cal.layout.intersect = 1.001;
  expect_true(cal.validate().find("intersect") != std::string::npos, "bad intersect rejected");
  cal = valid_calibration();
  cal.sync_offset = std::numeric_limits<std::int64_t>::min();
  expect_true(cal.validate().find("sync_offset") != std::string::npos, "i64 min rejected");
  cal = valid_calibration();
  cal.blend_width = std::numeric_limits<float>::quiet_NaN();
  expect_true(cal.validate().find("blend_width") != std::string::npos, "nan blend rejected");
}

void calibration_json_parse_defaults_and_roundtrip() {
  const std::string json = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
  const auto parsed = parse_match_calibration_json(json);
  expect_true(parsed.has_value(), "v1 calibration parses");
  expect_eq(parsed->left.width, 3840U, "left width");
  expect_near(parsed->layout.camera_axis_offset, 0.2398, 1.0e-6, "axis offset");
  expect_near(parsed->lens_correction_amount, 1.0, 1.0e-6, "lens correction default");
  expect_near(parsed->blend_width, 0.05, 1.0e-6, "blend width default");

  const auto roundtrip = parse_match_calibration_json(calibration_to_json(*parsed));
  expect_true(roundtrip.has_value(), "roundtrip parses");
  expect_eq(roundtrip->right.height, 2160U, "roundtrip height");
  expect_near(roundtrip->left.fx, parsed->left.fx, 0.0, "roundtrip left fx precision");
  expect_near(roundtrip->layout.x_rz, parsed->layout.x_rz, 0.0, "roundtrip layout xRz precision");
}

void calibration_file_loader_rejects_large_files_before_parse() {
  const auto dir = scratch_dir();
  const auto path = dir / "too-large.json";
  {
    std::ofstream output(path, std::ios::binary);
    output << std::string(1024 * 1024 + 1, ' ');
  }
  std::string error;
  const auto parsed = load_match_calibration_file(path.string(), &error);
  expect_true(!parsed.has_value(), "large calibration file rejected");
  expect_eq(error, std::string("calibration file too large"), "large calibration error");
}

void calibration_json_parses_and_serializes_field_roi() {
  const std::string json = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    },
    "field_roi": {
      "left": [[0.49, 0.90], [0.33, 0.73], [0.42, 0.58]],
      "right": [[0.63, 0.85], [0.78, 0.68], [0.55, 0.60]]
    }
  })json";
  const auto parsed = parse_match_calibration_json(json);
  expect_true(parsed.has_value(), "roi calibration parses");
  expect_true(parsed->field_roi.has_value(), "roi present");
  expect_eq(parsed->field_roi->left.size(), 3U, "left roi count");
  expect_near(parsed->field_roi->right[1][1], 0.68, 1.0e-6, "right roi point");
  const auto roundtrip = parse_match_calibration_json(calibration_to_json(*parsed));
  expect_true(roundtrip.has_value(), "roi roundtrip parses");
  expect_true(roundtrip->field_roi.has_value(), "roi roundtrip present");
  expect_eq(roundtrip->field_roi->right.size(), 3U, "right roi roundtrip count");
}

void calibration_json_parses_top_level_non_defaults() {
  const std::string json = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    },
    "rig_tilt": 0.3,
    "rig_roll": -0.12,
    "sync_offset": 67,
    "lens_correction_amount": 0.25,
    "blend_width": 0.123
  })json";
  const auto parsed = parse_match_calibration_json(json);
  expect_true(parsed.has_value(), "top-level non-default calibration parses");
  expect_near(parsed->rig_tilt, 0.3, 1.0e-9, "rig tilt parsed");
  expect_near(parsed->rig_roll, -0.12, 1.0e-9, "rig roll parsed");
  expect_eq(parsed->sync_offset, 67, "sync offset parsed");
  expect_near(parsed->lens_correction_amount, 0.25, 1.0e-6, "lens correction parsed");
  expect_near(parsed->blend_width, 0.123, 1.0e-6, "blend width parsed");
}

void calibration_json_rejects_schema_that_rust_rejects() {
  auto valid_json = [] {
    return std::string{R"json({
      "left_uniforms": {
        "width": 3840, "height": 2160,
        "fx": 1796.32, "fy": 1797.22,
        "cx": 1919.37, "cy": 1063.17,
        "d": [0.0342, 0.0677, -0.0741, 0.0299]
      },
      "right_uniforms": {
        "width": 3840, "height": 2160,
        "fx": 1796.32, "fy": 1797.22,
        "cx": 1919.37, "cy": 1063.17,
        "d": [0.0342, 0.0677, -0.0741, 0.0299]
      },
      "params": {
        "cameraAxisOffset": 0.2398,
        "intersect": 0.5446,
        "xTy": 0.00476,
        "xRz": 0.00753,
        "zRx": -0.00431
      }
    })json"};
  };
  auto replace_once = [](std::string text, std::string_view from, std::string_view to) {
    const auto index = text.find(from);
    if (index != std::string::npos) {
      text.replace(index, from.size(), to);
    }
    return text;
  };
  auto append_root_member = [](std::string text, std::string_view member) {
    const auto index = text.rfind('}');
    if (index != std::string::npos) {
      text.insert(index, ",");
      text.insert(index + 1, member);
    }
    return text;
  };

  expect_true(parse_match_calibration_json(
                  replace_once(valid_json(), "\"left_uniforms\"", "\"left\\u005funiforms\""))
                  .has_value(),
              "escaped known key parses like serde_json");

  expect_true(!parse_match_calibration_json("x" + valid_json()).has_value(),
              "garbage before top-level object rejected");
  expect_true(!parse_match_calibration_json(valid_json() + "x").has_value(),
              "garbage after top-level object rejected");
  expect_true(
      !parse_match_calibration_json(append_root_member(valid_json(), R"json()json")).has_value(),
      "trailing comma rejected");
  expect_true(!parse_match_calibration_json(append_root_member(valid_json(), R"json("bad":[})json"))
                   .has_value(),
              "malformed ignored field rejected");
  expect_true(!parse_match_calibration_json(replace_once(valid_json(), "\"width\": 3840",
                                                         "\"width\": 3840, \"width\": 1920"))
                   .has_value(),
              "duplicate known camera field rejected");
  expect_true(
      !parse_match_calibration_json(replace_once(valid_json(), "\"cameraAxisOffset\": 0.2398",
                                                 "\"cameraAxisOffset\": 0.2398, "
                                                 "\"cameraAxisOffset\": 0.1"))
           .has_value(),
      "duplicate known params field rejected");
  expect_true(!parse_match_calibration_json(
                   append_root_member(valid_json(), R"json("rig_tilt":0.3,"rig_tilt":0.4)json"))
                   .has_value(),
              "duplicate known optional root field rejected");
  expect_true(!parse_match_calibration_json(
                   append_root_member(valid_json(),
                                      R"json("field_roi":{"left":[],"left":[[0.0,0.0]]})json"))
                   .has_value(),
              "duplicate known field_roi field rejected");
  expect_true(!parse_match_calibration_json(
                   replace_once(valid_json(), "\"width\": 3840", "\"meta\":{\"width\": 3840}"))
                   .has_value(),
              "nested camera width is not accepted as direct width");
  expect_true(
      !parse_match_calibration_json(replace_once(valid_json(), "\"cameraAxisOffset\": 0.2398",
                                                 "\"meta\":{\"cameraAxisOffset\": 0.2398}"))
           .has_value(),
      "nested layout axis offset is not accepted as direct params field");
  expect_true(
      !parse_match_calibration_json(replace_once(valid_json(), "\"fx\": 1796.32", "\"fx\": .5"))
           .has_value(),
      "leading-dot number rejected");
  expect_true(
      !parse_match_calibration_json(replace_once(valid_json(), "\"fx\": 1796.32", "\"fx\": 1."))
           .has_value(),
      "trailing-dot number rejected");
  expect_true(!parse_match_calibration_json(
                   replace_once(valid_json(), "\"width\": 3840", "\"width\": 03840"))
                   .has_value(),
              "leading-zero integer rejected");
  expect_true(
      !parse_match_calibration_json(replace_once(valid_json(), "\"cameraAxisOffset\": 0.2398",
                                                 "\"cameraAxisOffset\": 1e999999"))
           .has_value(),
      "out-of-range number rejected");

  const std::string missing_required_layout = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753
    }
  })json";
  expect_true(!parse_match_calibration_json(missing_required_layout).has_value(),
              "missing zRx rejected");

  const std::string fractional_width = R"json({
    "left_uniforms": {
      "width": 3840.9, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
  expect_true(!parse_match_calibration_json(fractional_width).has_value(),
              "fractional width rejected");

  const std::string invalid_width_suffix = R"json({
    "left_uniforms": {
      "width": 3840abc, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
  expect_true(!parse_match_calibration_json(invalid_width_suffix).has_value(),
              "invalid width suffix rejected");

  const std::string nested_sync_offset = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "sync_offset": 99,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
  const auto parsed = parse_match_calibration_json(nested_sync_offset);
  expect_true(parsed.has_value(), "nested unknown sync_offset does not break parse");
  expect_eq(parsed->sync_offset, 0, "nested sync_offset ignored");

  const std::string too_many_distortion_values = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299, 99.0]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
  expect_true(!parse_match_calibration_json(too_many_distortion_values).has_value(),
              "long distortion array rejected");

  const std::string malformed_distortion_separators = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342,, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
  expect_true(!parse_match_calibration_json(malformed_distortion_separators).has_value(),
              "malformed distortion separators rejected");

  const std::string invalid_float_suffix = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796abc, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    }
  })json";
  expect_true(!parse_match_calibration_json(invalid_float_suffix).has_value(),
              "invalid float suffix rejected");

  const std::string malformed_roi = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    },
    "field_roi": {
      "left": [[0.49]],
      "right": [[0.63, 0.85]]
    }
  })json";
  expect_true(!parse_match_calibration_json(malformed_roi).has_value(), "malformed roi rejected");

  const std::string non_object_roi = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    },
    "field_roi": 123
  })json";
  expect_true(!parse_match_calibration_json(non_object_roi).has_value(), "non-object roi rejected");

  const std::string invalid_optional_scalars = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    },
    "rig_tilt": "bad"
  })json";
  expect_true(!parse_match_calibration_json(invalid_optional_scalars).has_value(),
              "present invalid optional scalar rejected");

  const std::string invalid_sync_offset_suffix = R"json({
    "left_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "right_uniforms": {
      "width": 3840, "height": 2160,
      "fx": 1796.32, "fy": 1797.22,
      "cx": 1919.37, "cy": 1063.17,
      "d": [0.0342, 0.0677, -0.0741, 0.0299]
    },
    "params": {
      "cameraAxisOffset": 0.2398,
      "intersect": 0.5446,
      "xTy": 0.00476,
      "xRz": 0.00753,
      "zRx": -0.00431
    },
    "sync_offset": 67abc
  })json";
  expect_true(!parse_match_calibration_json(invalid_sync_offset_suffix).has_value(),
              "invalid sync offset suffix rejected");

  const std::string nested_required_objects_only = R"json({
    "unknown": {
      "left_uniforms": {
        "width": 3840, "height": 2160,
        "fx": 1796.32, "fy": 1797.22,
        "cx": 1919.37, "cy": 1063.17,
        "d": [0.0342, 0.0677, -0.0741, 0.0299]
      },
      "right_uniforms": {
        "width": 3840, "height": 2160,
        "fx": 1796.32, "fy": 1797.22,
        "cx": 1919.37, "cy": 1063.17,
        "d": [0.0342, 0.0677, -0.0741, 0.0299]
      },
      "params": {
        "cameraAxisOffset": 0.2398,
        "intersect": 0.5446,
        "xTy": 0.00476,
        "xRz": 0.00753,
        "zRx": -0.00431
      }
    }
  })json";
  expect_true(!parse_match_calibration_json(nested_required_objects_only).has_value(),
              "nested required objects ignored");
}

void calibration_file_size_cap_is_enforced() {
  std::string too_large(kMaxCalibrationFileSize + 1, ' ');
  expect_true(!parse_match_calibration_json(too_large).has_value(), "large calibration rejected");
}

} // namespace

int main() {
  source_path_validation_matches_rust_reasons();
  frame_plane_validation_matches_rust_messages();
  camera_input_contract_matches_rust();
  calibration_validation_matches_rust_guards();
  calibration_json_parse_defaults_and_roundtrip();
  calibration_file_loader_rejects_large_files_before_parse();
  calibration_json_parses_and_serializes_field_roi();
  calibration_json_parses_top_level_non_defaults();
  calibration_json_rejects_schema_that_rust_rejects();
  calibration_file_size_cap_is_enforced();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
