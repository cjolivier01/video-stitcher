#include "reco/io/settings.hpp"

#include <algorithm>

namespace reco::io {
namespace {

std::variant<std::filesystem::path, SettingsError>
make_error(SettingsErrorKind kind, std::string message, std::string ns = {}) {
  return SettingsError{kind, std::errc{}, std::move(message), std::move(ns)};
}

std::optional<std::filesystem::path> getenv_path(const char* key) {
  const char* value = std::getenv(key);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  return std::filesystem::path(value);
}

std::filesystem::path home_config_dir() {
  if (auto xdg = getenv_path("XDG_CONFIG_HOME")) {
    return *xdg / "reco";
  }
  if (auto home = getenv_path("HOME")) {
    return *home / ".config" / "reco";
  }
  return {};
}

} // namespace

std::string_view settings_error_kind_name(SettingsErrorKind kind) {
  switch (kind) {
  case SettingsErrorKind::Io:
    return "io";
  case SettingsErrorKind::Serialize:
    return "serialize";
  case SettingsErrorKind::BadNamespace:
    return "bad_namespace";
  case SettingsErrorKind::NoConfigDir:
    return "no_config_dir";
  }
  return "io";
}

bool validate_settings_namespace(std::string_view ns) {
  return !ns.empty() && std::all_of(ns.begin(), ns.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
  });
}

std::variant<std::filesystem::path, SettingsError> config_dir() {
  if (auto override_path = getenv_path("RECO_CONFIG_DIR")) {
    return *override_path;
  }

#if defined(_WIN32)
  if (auto appdata = getenv_path("APPDATA")) {
    return *appdata / "reco";
  }
#elif defined(__APPLE__)
  if (auto home = getenv_path("HOME")) {
    return *home / "Library" / "Application Support" / "reco";
  }
#else
  if (auto path = home_config_dir(); !path.empty()) {
    return path;
  }
#endif

  return make_error(SettingsErrorKind::NoConfigDir, "cannot resolve config directory");
}

std::variant<std::filesystem::path, SettingsError> settings_path(std::string_view ns) {
  if (!validate_settings_namespace(ns)) {
    return SettingsError{SettingsErrorKind::BadNamespace, std::errc{}, "invalid namespace",
                         std::string(ns)};
  }
  auto dir_result = config_dir();
  if (auto* error = std::get_if<SettingsError>(&dir_result)) {
    return *error;
  }
  const auto& dir = std::get<std::filesystem::path>(dir_result);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                         std::string(ns)};
  }
  return dir / (std::string(ns) + ".json");
}

std::variant<bool, SettingsError> settings_exists(std::string_view ns) {
  auto path_result = settings_path(ns);
  if (auto* error = std::get_if<SettingsError>(&path_result)) {
    return *error;
  }
  std::error_code ec;
  const bool exists = std::filesystem::exists(std::get<std::filesystem::path>(path_result), ec);
  if (ec) {
    return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                         std::string(ns)};
  }
  return exists;
}

std::variant<bool, SettingsError> delete_settings(std::string_view ns) {
  auto path_result = settings_path(ns);
  if (auto* error = std::get_if<SettingsError>(&path_result)) {
    return *error;
  }
  std::error_code ec;
  const auto& path = std::get<std::filesystem::path>(path_result);
  if (!std::filesystem::exists(path, ec)) {
    if (ec) {
      return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                           std::string(ns)};
    }
    return false;
  }
  if (std::filesystem::is_directory(path, ec)) {
    return SettingsError{SettingsErrorKind::Io, std::errc::is_a_directory,
                         "settings path is a directory", std::string(ns)};
  }
  if (ec) {
    return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                         std::string(ns)};
  }
  const bool removed = std::filesystem::remove(path, ec);
  if (ec) {
    return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                         std::string(ns)};
  }
  return removed;
}

RecentFiles::RecentFiles() = default;

RecentFiles::RecentFiles(std::size_t max) : max_(std::max<std::size_t>(max, 1)) {}

void RecentFiles::push(std::filesystem::path path) {
  entries_.erase(std::remove(entries_.begin(), entries_.end(), path), entries_.end());
  entries_.insert(entries_.begin(), std::move(path));
  if (entries_.size() > max_) {
    entries_.resize(max_);
  }
}

void RecentFiles::remove(const std::filesystem::path& path) {
  entries_.erase(std::remove(entries_.begin(), entries_.end(), path), entries_.end());
}

void RecentFiles::clear() { entries_.clear(); }

void to_json(nlohmann::json& json, const RecentFiles& recent) {
  json = nlohmann::json{{"entries", nlohmann::json::array()}, {"max", recent.max_}};
  for (const auto& entry : recent.entries_) {
    json["entries"].push_back(entry.string());
  }
}

void from_json(const nlohmann::json& json, RecentFiles& recent) {
  recent.entries_.clear();
  recent.max_ = json.value("max", std::size_t{8});
  if (recent.max_ == 0) {
    recent.max_ = 1;
  }
  for (const auto& entry : json.value("entries", std::vector<std::string>{})) {
    recent.entries_.push_back(std::filesystem::path(entry));
  }
}

} // namespace reco::io
