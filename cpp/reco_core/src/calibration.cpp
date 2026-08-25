#include "reco/core/calibration.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace reco::core {
namespace {

constexpr double kEpsilon = 1.0e-6;
using Json = nlohmann::json;

const std::unordered_set<std::string>& known_fields_for_path(const std::string& path) {
  static const std::unordered_set<std::string> root{
      "left_uniforms", "right_uniforms", "params",    "rig_tilt",
      "rig_roll",      "sync_offset",    "field_roi", "lens_correction_amount",
      "blend_width"};
  static const std::unordered_set<std::string> camera{"width", "height", "fx", "fy",
                                                      "cx",    "cy",     "d"};
  static const std::unordered_set<std::string> params{
      "cameraAxisOffset", "intersect", "xTy", "xRz", "zRx", "xRx", "zRz"};
  static const std::unordered_set<std::string> field_roi{"left", "right"};
  static const std::unordered_set<std::string> empty;

  if (path == "$") {
    return root;
  }
  if (path == "$.left_uniforms" || path == "$.right_uniforms") {
    return camera;
  }
  if (path == "$.params") {
    return params;
  }
  if (path == "$.field_roi") {
    return field_roi;
  }
  return empty;
}

struct DuplicateKnownFieldSax final : nlohmann::json_sax<Json> {
  struct ObjectContext {
    std::string path;
    std::unordered_set<std::string> seen_known_fields;
  };

  std::vector<ObjectContext> objects;
  std::optional<std::string> pending_key;
  bool duplicate = false;

  bool null() override {
    pending_key.reset();
    return true;
  }
  bool boolean(bool) override {
    pending_key.reset();
    return true;
  }
  bool number_integer(number_integer_t) override {
    pending_key.reset();
    return true;
  }
  bool number_unsigned(number_unsigned_t) override {
    pending_key.reset();
    return true;
  }
  bool number_float(number_float_t, const string_t&) override {
    pending_key.reset();
    return true;
  }
  bool string(string_t&) override {
    pending_key.reset();
    return true;
  }
  bool binary(binary_t&) override {
    pending_key.reset();
    return true;
  }
  bool start_object(std::size_t) override {
    std::string path = "$";
    if (!objects.empty()) {
      path = objects.back().path;
      if (pending_key.has_value()) {
        path += ".";
        path += *pending_key;
      } else {
        path += "[]";
      }
    }
    pending_key.reset();
    objects.push_back({std::move(path), {}});
    return true;
  }
  bool key(string_t& key_text) override {
    if (objects.empty()) {
      return false;
    }
    const auto& known_fields = known_fields_for_path(objects.back().path);
    if (known_fields.find(key_text) != known_fields.end() &&
        !objects.back().seen_known_fields.insert(key_text).second) {
      duplicate = true;
      return false;
    }
    pending_key = key_text;
    return true;
  }
  bool end_object() override {
    if (objects.empty()) {
      return false;
    }
    objects.pop_back();
    pending_key.reset();
    return true;
  }
  bool start_array(std::size_t) override {
    pending_key.reset();
    return true;
  }
  bool end_array() override {
    pending_key.reset();
    return true;
  }
  bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override {
    return false;
  }
};

bool has_duplicate_known_fields(std::string_view text) {
  DuplicateKnownFieldSax sax;
  const auto valid = Json::sax_parse(text, &sax);
  return !valid || sax.duplicate;
}

std::optional<std::uint32_t> get_u32(const Json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number_unsigned()) {
    return std::nullopt;
  }
  const auto value = it->get<std::uint64_t>();
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(value);
}

std::optional<double> get_number(const Json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number()) {
    return std::nullopt;
  }
  try {
    return it->get<double>();
  } catch (const nlohmann::json::exception&) {
    return std::nullopt;
  }
}

std::optional<std::int64_t> get_i64(const Json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end()) {
    return std::nullopt;
  }
  if (it->is_number_integer()) {
    return it->get<std::int64_t>();
  }
  if (it->is_number_unsigned()) {
    const auto value = it->get<std::uint64_t>();
    if (value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return static_cast<std::int64_t>(value);
    }
  }
  return std::nullopt;
}

std::optional<CameraParams> parse_camera(const Json& object) {
  if (!object.is_object()) {
    return std::nullopt;
  }
  CameraParams params;
  const auto width = get_u32(object, "width");
  const auto height = get_u32(object, "height");
  const auto fx = get_number(object, "fx");
  const auto fy = get_number(object, "fy");
  const auto cx = get_number(object, "cx");
  const auto cy = get_number(object, "cy");
  const auto d = object.find("d");
  if (!width || !height || !fx || !fy || !cx || !cy || d == object.end() || !d->is_array() ||
      d->size() != 4) {
    return std::nullopt;
  }
  params.width = *width;
  params.height = *height;
  params.fx = *fx;
  params.fy = *fy;
  params.cx = *cx;
  params.cy = *cy;
  for (std::size_t i = 0; i < params.d.size(); ++i) {
    if (!(*d)[i].is_number()) {
      return std::nullopt;
    }
    params.d[i] = (*d)[i].get<double>();
  }
  return params;
}

std::optional<std::vector<std::array<double, 2>>> parse_roi_points(const Json& points_json) {
  if (!points_json.is_array()) {
    return std::nullopt;
  }
  std::vector<std::array<double, 2>> points;
  points.reserve(points_json.size());
  for (const auto& point_json : points_json) {
    if (!point_json.is_array() || point_json.size() != 2 || !point_json[0].is_number() ||
        !point_json[1].is_number()) {
      return std::nullopt;
    }
    points.push_back({point_json[0].get<double>(), point_json[1].get<double>()});
  }
  return points;
}

bool parse_field_roi(const Json& root, std::optional<FieldRoi>& out) {
  const auto roi_value = root.find("field_roi");
  if (roi_value == root.end() || roi_value->is_null()) {
    out = std::nullopt;
    return true;
  }
  if (!roi_value->is_object()) {
    return false;
  }
  FieldRoi roi;
  if (const auto left = roi_value->find("left"); left != roi_value->end()) {
    auto parsed = parse_roi_points(*left);
    if (!parsed.has_value()) {
      return false;
    }
    roi.left = std::move(*parsed);
  }
  if (const auto right = roi_value->find("right"); right != roi_value->end()) {
    auto parsed = parse_roi_points(*right);
    if (!parsed.has_value()) {
      return false;
    }
    roi.right = std::move(*parsed);
  }
  out = std::move(roi);
  return true;
}

std::string validate_camera(const CameraParams& params, const char* camera) {
  auto dimension_error = [&](const char* field, std::uint32_t value) {
    std::ostringstream out;
    out << camera << " camera " << field << " must be > 0, got " << value;
    return out.str();
  };
  if (params.width == 0) {
    return dimension_error("width", params.width);
  }
  if (params.height == 0) {
    return dimension_error("height", params.height);
  }
  if (params.width > kMaxCalibrationDimension) {
    return std::string(camera) + " camera width exceeds maximum allowed value";
  }
  if (params.height > kMaxCalibrationDimension) {
    return std::string(camera) + " camera height exceeds maximum allowed value";
  }
  const std::string prefix(camera);
  for (const auto& [field, value] :
       {std::pair{prefix + ".fx", params.fx}, std::pair{prefix + ".fy", params.fy}}) {
    if (!std::isfinite(value)) {
      return "field '" + field + "' must be finite";
    }
    if (value <= kEpsilon) {
      return "field '" + field + "' must be > epsilon";
    }
  }
  for (const auto& [field, value] :
       {std::pair{prefix + ".cx", params.cx}, std::pair{prefix + ".cy", params.cy}}) {
    if (!std::isfinite(value)) {
      return "field '" + field + "' must be finite";
    }
  }
  for (std::size_t i = 0; i < params.d.size(); ++i) {
    if (!std::isfinite(params.d[i])) {
      return "field '" + prefix + ".d[" + std::to_string(i) + "]' must be finite";
    }
  }
  return {};
}

} // namespace

std::string MatchCalibration::validate() const {
  if (const auto error = validate_camera(left, "left"); !error.empty()) {
    return error;
  }
  if (const auto error = validate_camera(right, "right"); !error.empty()) {
    return error;
  }
  if (!std::isfinite(layout.camera_axis_offset)) {
    return "field 'params.cameraAxisOffset' must be finite";
  }
  if (layout.camera_axis_offset <= kEpsilon) {
    return "params.cameraAxisOffset must be > epsilon";
  }
  if (!std::isfinite(layout.intersect)) {
    return "field 'params.intersect' must be finite";
  }
  if (layout.intersect < 0.0 || layout.intersect > 1.0) {
    return "params.intersect must be in [0.0, 1.0]";
  }
  for (const auto& [field, value] : {
           std::pair{"params.xTy", layout.x_ty},
           std::pair{"params.xRz", layout.x_rz},
           std::pair{"params.xRx", layout.x_rx},
           std::pair{"params.zRx", layout.z_rx},
           std::pair{"params.zRz", layout.z_rz},
       }) {
    if (!std::isfinite(value)) {
      return std::string("field '") + field + "' must be finite";
    }
  }
  if (!std::isfinite(rig_tilt)) {
    return "field 'rig_tilt' must be finite";
  }
  if (!std::isfinite(rig_roll)) {
    return "field 'rig_roll' must be finite";
  }
  if (!std::isfinite(blend_width)) {
    return "field 'blend_width' must be finite";
  }
  if (!std::isfinite(lens_correction_amount)) {
    return "field 'lens_correction_amount' must be finite";
  }
  if (sync_offset < -kMaxSyncOffsetFrames || sync_offset > kMaxSyncOffsetFrames) {
    return "params.sync_offset must be in [-100000, 100000] frames";
  }
  return {};
}

std::optional<MatchCalibration> parse_match_calibration_json(std::string_view json) {
  if (json.size() > kMaxCalibrationFileSize) {
    return std::nullopt;
  }
  const std::string text(json);
  if (has_duplicate_known_fields(text)) {
    return std::nullopt;
  }
  Json root;
  try {
    root = Json::parse(text);
  } catch (const nlohmann::json::exception&) {
    return std::nullopt;
  }
  if (!root.is_object()) {
    return std::nullopt;
  }
  const auto left_object = root.find("left_uniforms");
  const auto right_object = root.find("right_uniforms");
  const auto params_object = root.find("params");
  if (left_object == root.end() || right_object == root.end() || params_object == root.end() ||
      !params_object->is_object()) {
    return std::nullopt;
  }
  auto left = parse_camera(*left_object);
  auto right = parse_camera(*right_object);
  if (!left || !right) {
    return std::nullopt;
  }
  MatchCalibration calibration;
  calibration.left = *left;
  calibration.right = *right;
  const auto camera_axis_offset = get_number(*params_object, "cameraAxisOffset");
  const auto intersect = get_number(*params_object, "intersect");
  const auto x_ty = get_number(*params_object, "xTy");
  const auto x_rz = get_number(*params_object, "xRz");
  const auto z_rx = get_number(*params_object, "zRx");
  if (!camera_axis_offset || !intersect || !x_ty || !x_rz || !z_rx) {
    return std::nullopt;
  }
  calibration.layout.camera_axis_offset = *camera_axis_offset;
  calibration.layout.intersect = *intersect;
  calibration.layout.x_ty = *x_ty;
  calibration.layout.x_rz = *x_rz;
  calibration.layout.z_rx = *z_rx;
  const auto has_x_rx = params_object->contains("xRx");
  const auto has_z_rz = params_object->contains("zRz");
  const auto x_rx = get_number(*params_object, "xRx");
  const auto z_rz = get_number(*params_object, "zRz");
  if ((has_x_rx && !x_rx.has_value()) || (has_z_rz && !z_rz.has_value())) {
    return std::nullopt;
  }
  calibration.layout.x_rx = x_rx.value_or(0.0);
  calibration.layout.z_rz = z_rz.value_or(0.0);

  auto parse_optional_number = [&](const char* key, double default_value, double& destination) {
    if (!root.contains(key)) {
      destination = default_value;
      return true;
    }
    auto parsed = get_number(root, key);
    if (!parsed.has_value()) {
      return false;
    }
    destination = *parsed;
    return true;
  };
  auto parse_optional_float = [&](const char* key, float default_value, float& destination) {
    if (!root.contains(key)) {
      destination = default_value;
      return true;
    }
    auto parsed = get_number(root, key);
    if (!parsed.has_value()) {
      return false;
    }
    destination = static_cast<float>(*parsed);
    return true;
  };
  if (!parse_optional_number("rig_tilt", 0.0, calibration.rig_tilt) ||
      !parse_optional_number("rig_roll", 0.0, calibration.rig_roll) ||
      !parse_optional_float("lens_correction_amount", 1.0F, calibration.lens_correction_amount) ||
      !parse_optional_float("blend_width", 0.05F, calibration.blend_width)) {
    return std::nullopt;
  }
  if (root.contains("sync_offset")) {
    const auto parsed = get_i64(root, "sync_offset");
    if (!parsed.has_value()) {
      return std::nullopt;
    }
    calibration.sync_offset = *parsed;
  }
  if (!parse_field_roi(root, calibration.field_roi)) {
    return std::nullopt;
  }
  if (!calibration.validate().empty()) {
    return std::nullopt;
  }
  return calibration;
}

std::optional<MatchCalibration> load_match_calibration_file(const std::string& path,
                                                            std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    if (error != nullptr) {
      *error = "cannot read calibration file";
    }
    return std::nullopt;
  }
  std::error_code status_error;
  const auto file_size = std::filesystem::file_size(path, status_error);
  if (status_error || file_size > kMaxCalibrationFileSize) {
    if (error != nullptr) {
      *error = status_error ? "cannot stat calibration file" : "calibration file too large";
    }
    return std::nullopt;
  }
  std::ostringstream contents;
  std::string text(kMaxCalibrationFileSize + 1, '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(input.gcount()));
  if (text.size() > kMaxCalibrationFileSize) {
    if (error != nullptr) {
      *error = "calibration file too large";
    }
    return std::nullopt;
  }
  auto parsed = parse_match_calibration_json(text);
  if (!parsed && error != nullptr) {
    *error = "invalid calibration JSON";
  }
  return parsed;
}

std::string calibration_to_json(const MatchCalibration& calibration) {
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\"left_uniforms\":{\"width\":" << calibration.left.width
      << ",\"height\":" << calibration.left.height << ",\"fx\":" << calibration.left.fx
      << ",\"fy\":" << calibration.left.fy << ",\"cx\":" << calibration.left.cx
      << ",\"cy\":" << calibration.left.cy << ",\"d\":[" << calibration.left.d[0] << ","
      << calibration.left.d[1] << "," << calibration.left.d[2] << "," << calibration.left.d[3]
      << "]},\"right_uniforms\":{\"width\":" << calibration.right.width
      << ",\"height\":" << calibration.right.height << ",\"fx\":" << calibration.right.fx
      << ",\"fy\":" << calibration.right.fy << ",\"cx\":" << calibration.right.cx
      << ",\"cy\":" << calibration.right.cy << ",\"d\":[" << calibration.right.d[0] << ","
      << calibration.right.d[1] << "," << calibration.right.d[2] << "," << calibration.right.d[3]
      << "]},\"params\":{\"cameraAxisOffset\":" << calibration.layout.camera_axis_offset
      << ",\"intersect\":" << calibration.layout.intersect << ",\"xTy\":" << calibration.layout.x_ty
      << ",\"xRz\":" << calibration.layout.x_rz << ",\"zRx\":" << calibration.layout.z_rx
      << ",\"xRx\":" << calibration.layout.x_rx << ",\"zRz\":" << calibration.layout.z_rz
      << "},\"rig_tilt\":" << calibration.rig_tilt << ",\"rig_roll\":" << calibration.rig_roll
      << ",\"sync_offset\":" << calibration.sync_offset
      << ",\"lens_correction_amount\":" << calibration.lens_correction_amount
      << ",\"blend_width\":" << calibration.blend_width;
  if (calibration.field_roi.has_value()) {
    auto write_points = [](std::ostringstream& stream,
                           const std::vector<std::array<double, 2>>& points) {
      stream << "[";
      for (std::size_t i = 0; i < points.size(); ++i) {
        if (i != 0) {
          stream << ",";
        }
        stream << "[" << points[i][0] << "," << points[i][1] << "]";
      }
      stream << "]";
    };
    out << ",\"field_roi\":{\"left\":";
    write_points(out, calibration.field_roi->left);
    out << ",\"right\":";
    write_points(out, calibration.field_roi->right);
    out << "}";
  }
  out << "}";
  return out.str();
}

} // namespace reco::core
