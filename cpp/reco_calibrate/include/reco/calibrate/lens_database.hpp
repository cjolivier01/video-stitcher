#pragma once

#include "reco/calibrate/types.hpp"

#include "reco/core/calibration.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reco::calibrate {

enum class LensLoadErrorKind {
  Io,
  Parse,
  UnrecognizedFormat,
};

class LensLoadError final : public std::runtime_error {
public:
  LensLoadError(LensLoadErrorKind kind, const std::string& message);
  LensLoadErrorKind kind;
};

class LensDatabase {
public:
  struct ProfileEntry {
    std::string source;
    std::string brand;
    std::string model;
    std::string lens_model;
    std::string camera_setting;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    reco::core::CameraParams params;
  };

  [[nodiscard]] static LensDatabase load_empty();

  void add_profile_from_json(const std::string& json_text, const std::string& source);
  [[nodiscard]] std::size_t len() const;
  [[nodiscard]] bool is_empty() const;

  [[nodiscard]] std::optional<std::pair<reco::core::CameraParams, LensProfileInfo>>
  find(const std::string& brand, const std::string& model, std::uint32_t width,
       std::uint32_t height, const std::optional<std::string>& lens_info = std::nullopt) const;
  [[nodiscard]] std::optional<std::pair<reco::core::CameraParams, LensProfileInfo>>
  find_from_telemetry(const std::string& camera_type, const std::optional<std::string>& camera_model,
                      std::uint32_t width, std::uint32_t height,
                      const std::optional<std::string>& lens_info = std::nullopt) const;
  [[nodiscard]] std::optional<std::pair<reco::core::CameraParams, LensProfileInfo>>
  find_by_resolution(std::uint32_t width, std::uint32_t height) const;

  [[nodiscard]] std::vector<LensProfileSummary> iter_profiles() const;
  [[nodiscard]] std::vector<LensProfileSummary> candidates(std::uint32_t width,
                                                           std::uint32_t height) const;
  [[nodiscard]] std::vector<LensProfileSummary> search(const std::string& query,
                                                       std::uint32_t width,
                                                       std::uint32_t height) const;
  [[nodiscard]] std::vector<std::string> brands() const;
  [[nodiscard]] std::vector<std::pair<std::string, std::uint32_t>>
  models_for_brand(const std::string& brand) const;
  [[nodiscard]] std::optional<reco::core::CameraParams>
  load_by_summary(const LensProfileSummary& summary) const;

private:
  std::vector<ProfileEntry> profiles_;
};

[[nodiscard]] reco::core::CameraParams load_lens_from_json(const std::string& json_text,
                                                           const std::string& source);
[[nodiscard]] reco::core::CameraParams load_lens_from_file(const std::string& path);

} // namespace reco::calibrate
