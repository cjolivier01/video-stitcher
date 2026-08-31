#include "reco/calibrate/lens_database.hpp"
#include "reco/core/path.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace reco::calibrate {
namespace {

using Json = nlohmann::json;

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string replace_spaces_with_dash(std::string value) {
  std::replace(value.begin(), value.end(), ' ', '-');
  for (;;) {
    const auto pos = value.find("--");
    if (pos == std::string::npos) {
      break;
    }
    value.replace(pos, 2, "-");
  }
  return value;
}

std::string normalize_camera_key(const std::string& brand, const std::string& model) {
  return replace_spaces_with_dash(lower_copy(brand)) + "/" +
         replace_spaces_with_dash(lower_copy(model));
}

std::optional<std::string> strip_model_variant(const std::string& key) {
  for (const std::string suffix : {"-mini", "-max", "-session", "-bones", "-creator-edition"}) {
    if (key.size() >= suffix.size() &&
        key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return key.substr(0, key.size() - suffix.size());
    }
  }
  return std::nullopt;
}

std::string title_case(const std::string& value) {
  std::stringstream input(value);
  std::string word;
  std::string out;
  while (std::getline(input, word, '-')) {
    if (!out.empty()) {
      out.push_back(' ');
    }
    if (!word.empty()) {
      word[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
    }
    out += word;
  }
  return out;
}

std::string format_words_title_case(const std::string& value) {
  std::stringstream input(value);
  std::string word;
  std::string out;
  while (input >> word) {
    if (!out.empty()) {
      out.push_back(' ');
    }
    word[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
    out += word;
  }
  return out;
}

std::string format_camera_name(const std::string& brand, const std::string& model) {
  return format_words_title_case(brand) + " " + format_words_title_case(model);
}

std::string format_lens_name(const std::string& lens_model, const std::string& camera_setting) {
  if (camera_setting.empty()) {
    return lens_model;
  }
  return lens_model + " (" + camera_setting + ")";
}

std::optional<std::uint32_t> json_u32(const Json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number_unsigned()) {
    return std::nullopt;
  }
  const auto value = it->get<std::uint64_t>();
  if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(value);
}

std::optional<double> json_number(const Json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number()) {
    return std::nullopt;
  }
  return it->get<double>();
}

std::optional<double> json_array_number(const Json& array, std::size_t index) {
  if (!array.is_array() || index >= array.size() || !array[index].is_number()) {
    return std::nullopt;
  }
  return array[index].get<double>();
}

struct ParsedProfile {
  std::string source;
  std::string brand;
  std::string model;
  std::string lens_model;
  std::string camera_setting;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  reco::core::CameraParams params;
};

std::optional<ParsedProfile> parse_profile_value(const Json& value, const std::string& source) {
  const auto brand_it = value.find("camera_brand");
  const auto model_it = value.find("camera_model");
  if (brand_it == value.end() || model_it == value.end() || !brand_it->is_string() ||
      !model_it->is_string()) {
    return std::nullopt;
  }

  const Json* res = nullptr;
  if (const auto it = value.find("resolution"); it != value.end() && it->is_object()) {
    res = &*it;
  } else if (const auto it = value.find("calib_dimension"); it != value.end() && it->is_object()) {
    res = &*it;
  }
  if (res == nullptr) {
    return std::nullopt;
  }
  auto width = json_u32(*res, "width");
  if (!width.has_value()) {
    width = json_u32(*res, "w");
  }
  auto height = json_u32(*res, "height");
  if (!height.has_value()) {
    height = json_u32(*res, "h");
  }
  if (!width.has_value() || !height.has_value()) {
    return std::nullopt;
  }

  const Json* camera_matrix = nullptr;
  if (const auto it = value.find("camera_matrix"); it != value.end()) {
    camera_matrix = &*it;
  } else if (const auto fp = value.find("fisheye_params"); fp != value.end() && fp->is_object()) {
    if (const auto it = fp->find("camera_matrix"); it != fp->end()) {
      camera_matrix = &*it;
    }
  }
  if (camera_matrix == nullptr) {
    return std::nullopt;
  }

  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  if (camera_matrix->is_object()) {
    const auto maybe_fx = json_number(*camera_matrix, "fx");
    const auto maybe_fy = json_number(*camera_matrix, "fy");
    const auto maybe_cx = json_number(*camera_matrix, "cx");
    const auto maybe_cy = json_number(*camera_matrix, "cy");
    if (!maybe_fx || !maybe_fy || !maybe_cx || !maybe_cy) {
      return std::nullopt;
    }
    fx = *maybe_fx;
    fy = *maybe_fy;
    cx = *maybe_cx;
    cy = *maybe_cy;
  } else if (camera_matrix->is_array() && camera_matrix->size() >= 3) {
    const auto& r0 = (*camera_matrix)[0];
    const auto& r1 = (*camera_matrix)[1];
    if (!r0.is_array() || !r1.is_array() || r0.size() < 3 || r1.size() < 3) {
      return std::nullopt;
    }
    const auto maybe_fx = json_array_number(r0, 0);
    const auto maybe_fy = json_array_number(r1, 1);
    const auto maybe_cx = json_array_number(r0, 2);
    const auto maybe_cy = json_array_number(r1, 2);
    if (!maybe_fx || !maybe_fy || !maybe_cx || !maybe_cy) {
      return std::nullopt;
    }
    fx = *maybe_fx;
    fy = *maybe_fy;
    cx = *maybe_cx;
    cy = *maybe_cy;
  } else {
    return std::nullopt;
  }

  const Json* distortion = nullptr;
  if (const auto it = value.find("distortion_coeffs"); it != value.end()) {
    distortion = &*it;
  } else if (const auto fp = value.find("fisheye_params"); fp != value.end() && fp->is_object()) {
    if (const auto it = fp->find("distortion_coeffs"); it != fp->end()) {
      distortion = &*it;
    }
  }
  if (distortion == nullptr || !distortion->is_array() || distortion->size() < 4) {
    return std::nullopt;
  }

  ParsedProfile out;
  out.source = source;
  out.brand = brand_it->get<std::string>();
  out.model = model_it->get<std::string>();
  out.lens_model = value.value("lens_model", "");
  out.camera_setting = value.value("camera_setting", "");
  out.width = *width;
  out.height = *height;
  out.params = {.width = *width,
                .height = *height,
                .fx = fx,
                .fy = fy,
                .cx = cx,
                .cy = cy,
                .d = {(*distortion)[0].is_number() ? (*distortion)[0].get<double>() : 0.0,
                      (*distortion)[1].is_number() ? (*distortion)[1].get<double>() : 0.0,
                      (*distortion)[2].is_number() ? (*distortion)[2].get<double>() : 0.0,
                      (*distortion)[3].is_number() ? (*distortion)[3].get<double>() : 0.0}};
  return out;
}

LensProfileSummary summary_from_profile(const LensDatabase::ProfileEntry& profile) {
  return {.camera = format_camera_name(profile.brand, profile.model),
          .lens = format_lens_name(profile.lens_model, profile.camera_setting),
          .width = profile.width,
          .height = profile.height};
}

LensProfileInfo info_from_profile(const LensDatabase::ProfileEntry& profile, ProfileSource source) {
  return {.camera = format_camera_name(profile.brand, profile.model),
          .lens = format_lens_name(profile.lens_model, profile.camera_setting),
          .source = source,
          .path = std::nullopt};
}

} // namespace

LensLoadError::LensLoadError(LensLoadErrorKind kind, const std::string& message)
    : std::runtime_error(message), kind(kind) {}

LensDatabase LensDatabase::load_empty() { return {}; }

void LensDatabase::add_profile_from_json(const std::string& json_text, const std::string& source) {
  Json value;
  try {
    value = Json::parse(json_text);
  } catch (const std::exception& e) {
    throw LensLoadError(LensLoadErrorKind::Parse, e.what());
  }
  const auto parsed = parse_profile_value(value, source);
  if (!parsed.has_value()) {
    throw LensLoadError(LensLoadErrorKind::UnrecognizedFormat, source);
  }
  profiles_.push_back({.source = parsed->source,
                       .brand = parsed->brand,
                       .model = parsed->model,
                       .lens_model = parsed->lens_model,
                       .camera_setting = parsed->camera_setting,
                       .width = parsed->width,
                       .height = parsed->height,
                       .params = parsed->params});
}

std::size_t LensDatabase::len() const { return profiles_.size(); }

bool LensDatabase::is_empty() const { return profiles_.empty(); }

std::optional<std::pair<reco::core::CameraParams, LensProfileInfo>>
LensDatabase::find(const std::string& brand, const std::string& model, std::uint32_t width,
                   std::uint32_t height, const std::optional<std::string>& lens_info) const {
  const auto key = normalize_camera_key(brand, model);
  std::vector<std::size_t> indices;
  std::vector<std::size_t> parent_indices;
  for (std::size_t i = 0; i < profiles_.size(); ++i) {
    if (normalize_camera_key(profiles_[i].brand, profiles_[i].model) == key) {
      indices.push_back(i);
    }
  }
  if (const auto parent = strip_model_variant(key); parent.has_value()) {
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
      if (normalize_camera_key(profiles_[i].brand, profiles_[i].model) == *parent) {
        parent_indices.push_back(i);
      }
    }
  }
  if (indices.empty()) {
    indices = parent_indices;
  }
  if (indices.empty()) {
    return std::nullopt;
  }

  std::vector<std::size_t> fov_filtered;
  std::vector<std::size_t> parent_fov_filtered;
  if (lens_info.has_value()) {
    const auto info_lower = lower_copy(*lens_info);
    for (const auto idx : indices) {
      const auto& p = profiles_[idx];
      if (lower_copy(p.lens_model) == info_lower && p.camera_setting.empty()) {
        fov_filtered.push_back(idx);
      }
    }
    if (fov_filtered.empty()) {
      for (const auto idx : parent_indices) {
        const auto& p = profiles_[idx];
        if (lower_copy(p.lens_model) == info_lower && p.camera_setting.empty()) {
          parent_fov_filtered.push_back(idx);
        }
      }
    }
  }
  const auto& candidates = !fov_filtered.empty()
                               ? fov_filtered
                               : (!parent_fov_filtered.empty() ? parent_fov_filtered : indices);

  for (const auto idx : candidates) {
    const auto& p = profiles_[idx];
    if (p.width == width && p.height == height) {
      return std::make_pair(p.params, info_from_profile(p, ProfileSource::Database));
    }
  }

  const double target_aspect = static_cast<double>(width) / static_cast<double>(height);
  std::optional<std::pair<std::size_t, double>> best;
  for (const auto idx : candidates) {
    const auto& p = profiles_[idx];
    const double aspect = static_cast<double>(p.width) / static_cast<double>(p.height);
    if (std::abs(aspect - target_aspect) < 0.05) {
      const double scale_diff = std::abs(static_cast<double>(p.width) - static_cast<double>(width));
      if (!best.has_value() || scale_diff < best->second) {
        best = {idx, scale_diff};
      }
    }
  }

  if (!best.has_value()) {
    return std::nullopt;
  }
  const auto& p = profiles_[best->first];
  const double scale = static_cast<double>(width) / static_cast<double>(p.width);
  auto params = p.params;
  params.width = width;
  params.height = height;
  params.fx *= scale;
  params.fy *= scale;
  params.cx *= scale;
  params.cy *= scale;
  return std::make_pair(params, info_from_profile(p, ProfileSource::Database));
}

std::optional<std::pair<reco::core::CameraParams, LensProfileInfo>>
LensDatabase::find_from_telemetry(const std::string& camera_type,
                                  const std::optional<std::string>& camera_model,
                                  std::uint32_t width, std::uint32_t height,
                                  const std::optional<std::string>& lens_info) const {
  return find(camera_type, camera_model.value_or(camera_type), width, height, lens_info);
}

std::optional<std::pair<reco::core::CameraParams, LensProfileInfo>>
LensDatabase::find_by_resolution(std::uint32_t width, std::uint32_t height) const {
  for (const auto& p : profiles_) {
    if (p.width == width && p.height == height) {
      return std::make_pair(p.params, info_from_profile(p, ProfileSource::Fallback));
    }
  }
  return std::nullopt;
}

std::vector<LensProfileSummary> LensDatabase::iter_profiles() const {
  std::vector<LensProfileSummary> out;
  out.reserve(profiles_.size());
  for (const auto& profile : profiles_) {
    out.push_back(summary_from_profile(profile));
  }
  return out;
}

std::vector<LensProfileSummary> LensDatabase::candidates(std::uint32_t width,
                                                         std::uint32_t height) const {
  std::vector<LensProfileSummary> out;
  for (const auto& p : profiles_) {
    if ((width == 0 || p.width == width) && (height == 0 || p.height == height)) {
      out.push_back(summary_from_profile(p));
    }
  }
  return out;
}

std::vector<LensProfileSummary> LensDatabase::search(const std::string& query, std::uint32_t width,
                                                     std::uint32_t height) const {
  std::vector<std::string> words;
  std::stringstream input(query);
  std::string word;
  while (input >> word) {
    words.push_back(lower_copy(word));
  }
  if (words.empty()) {
    return {};
  }

  const auto target_aspect =
      width > 0 && height > 0
          ? std::optional<double>(static_cast<double>(width) / static_cast<double>(height))
          : std::nullopt;
  std::vector<std::pair<std::size_t, int>> hits;
  for (std::size_t i = 0; i < profiles_.size(); ++i) {
    const auto& p = profiles_[i];
    const auto haystack =
        lower_copy(p.brand + " " + p.model + " " + p.lens_model + " " + p.camera_setting + " " +
                   std::to_string(p.width) + "x" + std::to_string(p.height));
    if (std::all_of(words.begin(), words.end(),
                    [&](const std::string& w) { return haystack.find(w) != std::string::npos; })) {
      int priority = 2;
      if (width > 0 && p.width == width && height > 0 && p.height == height) {
        priority = 0;
      } else if (target_aspect.has_value()) {
        const double aspect = static_cast<double>(p.width) / static_cast<double>(p.height);
        priority = std::abs(aspect - *target_aspect) < 0.05 ? 1 : 2;
      }
      hits.push_back({i, priority});
    }
  }
  std::stable_sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) {
      return a.second < b.second;
    }
    return a.first < b.first;
  });
  if (hits.size() > 100) {
    hits.resize(100);
  }
  std::vector<LensProfileSummary> out;
  out.reserve(hits.size());
  for (const auto [idx, priority] : hits) {
    (void)priority;
    out.push_back(summary_from_profile(profiles_[idx]));
  }
  return out;
}

std::vector<std::string> LensDatabase::brands() const {
  std::set<std::string> seen;
  for (const auto& p : profiles_) {
    const auto key = normalize_camera_key(p.brand, p.model);
    const auto slash = key.find('/');
    seen.insert(title_case(key.substr(0, slash)));
  }
  return {seen.begin(), seen.end()};
}

std::vector<std::pair<std::string, std::uint32_t>>
LensDatabase::models_for_brand(const std::string& brand) const {
  const auto prefix = replace_spaces_with_dash(lower_copy(brand));
  std::unordered_map<std::string, std::uint32_t> counts;
  for (const auto& p : profiles_) {
    const auto key = normalize_camera_key(p.brand, p.model);
    const auto slash = key.find('/');
    if (slash != std::string::npos && key.substr(0, slash) == prefix) {
      counts[title_case(key.substr(slash + 1))] += 1;
    }
  }
  std::vector<std::pair<std::string, std::uint32_t>> out(counts.begin(), counts.end());
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  return out;
}

std::optional<reco::core::CameraParams>
LensDatabase::load_by_summary(const LensProfileSummary& summary) const {
  for (const auto& p : profiles_) {
    if (summary_from_profile(p).camera == summary.camera &&
        summary_from_profile(p).lens == summary.lens && p.width == summary.width &&
        p.height == summary.height) {
      return p.params;
    }
  }
  return std::nullopt;
}

reco::core::CameraParams load_lens_from_json(const std::string& json_text,
                                             const std::string& source) {
  Json value;
  try {
    value = Json::parse(json_text);
  } catch (const std::exception& e) {
    throw LensLoadError(LensLoadErrorKind::Parse, e.what());
  }

  if (value.contains("fx") && value.contains("d")) {
    const auto width = json_u32(value, "width");
    const auto height = json_u32(value, "height");
    const auto fx = json_number(value, "fx");
    const auto fy = json_number(value, "fy");
    const auto cx = json_number(value, "cx");
    const auto cy = json_number(value, "cy");
    const auto d = value.find("d");
    if (!width || !height || !fx || !fy || !cx || !cy || d == value.end() || !d->is_array() ||
        d->size() < 4 || !json_array_number(*d, 0) || !json_array_number(*d, 1) ||
        !json_array_number(*d, 2) || !json_array_number(*d, 3)) {
      throw LensLoadError(LensLoadErrorKind::Parse, source);
    }
    return {.width = *width,
            .height = *height,
            .fx = *fx,
            .fy = *fy,
            .cx = *cx,
            .cy = *cy,
            .d = {*json_array_number(*d, 0), *json_array_number(*d, 1), *json_array_number(*d, 2),
                  *json_array_number(*d, 3)}};
  }

  if (const auto parsed = parse_profile_value(value, source); parsed.has_value()) {
    return parsed->params;
  }
  throw LensLoadError(LensLoadErrorKind::UnrecognizedFormat, source);
}

reco::core::CameraParams load_lens_from_file(const std::string& path) {
  std::ifstream file(reco::core::path_from_utf8(path));
  if (!file) {
    throw LensLoadError(LensLoadErrorKind::Io, path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return load_lens_from_json(buffer.str(), path);
}

} // namespace reco::calibrate
