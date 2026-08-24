#pragma once

#include <cerrno>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace reco::io {

enum class SettingsErrorKind {
  Io,
  Serialize,
  BadNamespace,
  NoConfigDir,
};

struct SettingsError {
  SettingsErrorKind kind = SettingsErrorKind::Io;
  std::errc io_kind = std::errc{};
  std::string message;
  std::string ns;
};

[[nodiscard]] std::string_view settings_error_kind_name(SettingsErrorKind kind);
[[nodiscard]] bool validate_settings_namespace(std::string_view ns);
[[nodiscard]] std::variant<std::filesystem::path, SettingsError> config_dir();
[[nodiscard]] std::variant<std::filesystem::path, SettingsError> settings_path(std::string_view ns);
[[nodiscard]] std::variant<bool, SettingsError> settings_exists(std::string_view ns);
[[nodiscard]] std::variant<bool, SettingsError> delete_settings(std::string_view ns);

namespace detail {

inline std::errc errc_from_error_code(const std::error_code& ec) {
  return static_cast<std::errc>(ec.default_error_condition().value());
}

inline std::errc errc_from_errno(int value) {
  if (value == 0) {
    return std::errc::io_error;
  }
  return static_cast<std::errc>(value);
}

} // namespace detail

template <typename T>
[[nodiscard]] std::variant<T, SettingsError> load_settings(std::string_view ns) {
  auto path_result = settings_path(ns);
  if (auto* error = std::get_if<SettingsError>(&path_result)) {
    return *error;
  }
  const auto& path = std::get<std::filesystem::path>(path_result);
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    if (ec) {
      return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                           std::string(ns)};
    }
    return SettingsError{
        SettingsErrorKind::Io,
        std::errc::no_such_file_or_directory,
        "failed to open settings file",
        std::string(ns),
    };
  }
  if (std::filesystem::is_directory(path, ec)) {
    return SettingsError{
        SettingsErrorKind::Io,
        std::errc::is_a_directory,
        "settings path is a directory",
        std::string(ns),
    };
  }
  (void)std::filesystem::file_size(path, ec);
  if (ec) {
    return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                         std::string(ns)};
  }
  errno = 0;
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return SettingsError{SettingsErrorKind::Io, detail::errc_from_errno(errno),
                         "failed to open settings file", std::string(ns)};
  }
  try {
    nlohmann::json json;
    input >> json;
    return json.get<T>();
  } catch (const nlohmann::json::exception& e) {
    return SettingsError{SettingsErrorKind::Serialize, std::errc{}, e.what(), std::string(ns)};
  } catch (const std::exception& e) {
    return SettingsError{SettingsErrorKind::Io, std::errc::io_error, e.what(), std::string(ns)};
  }
}

template <typename T> [[nodiscard]] T load_settings_or_default(std::string_view ns) {
  auto result = load_settings<T>(ns);
  if (auto* value = std::get_if<T>(&result)) {
    return *value;
  }
  return T{};
}

template <typename T>
[[nodiscard]] std::optional<SettingsError> save_settings(std::string_view ns, const T& value) {
  auto path_result = settings_path(ns);
  if (auto* error = std::get_if<SettingsError>(&path_result)) {
    return *error;
  }
  const auto& path = std::get<std::filesystem::path>(path_result);
  const auto tmp = path.parent_path() / (path.filename().string() + ".tmp");
  try {
    errno = 0;
    std::ofstream output(tmp, std::ios::binary | std::ios::trunc);
    if (!output) {
      return SettingsError{SettingsErrorKind::Io, detail::errc_from_errno(errno),
                           "failed to create settings temp file", std::string(ns)};
    }
    output << nlohmann::json(value).dump(2);
    output << '\n';
    output.close();
    if (!output) {
      return SettingsError{SettingsErrorKind::Io, detail::errc_from_errno(errno),
                           "failed to write settings temp file", std::string(ns)};
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
      return SettingsError{SettingsErrorKind::Io, detail::errc_from_error_code(ec), ec.message(),
                           std::string(ns)};
    }
    return std::nullopt;
  } catch (const nlohmann::json::exception& e) {
    return SettingsError{SettingsErrorKind::Serialize, std::errc{}, e.what(), std::string(ns)};
  }
}

class RecentFiles {
public:
  RecentFiles();
  explicit RecentFiles(std::size_t max);

  void push(std::filesystem::path path);
  void remove(const std::filesystem::path& path);
  void clear();

  [[nodiscard]] const std::vector<std::filesystem::path>& entries() const { return entries_; }
  [[nodiscard]] std::size_t max() const { return max_; }
  [[nodiscard]] std::size_t size() const { return entries_.size(); }
  [[nodiscard]] bool empty() const { return entries_.empty(); }

private:
  std::vector<std::filesystem::path> entries_;
  std::size_t max_ = 8;

  friend void to_json(nlohmann::json& json, const RecentFiles& recent);
  friend void from_json(const nlohmann::json& json, RecentFiles& recent);
};

void to_json(nlohmann::json& json, const RecentFiles& recent);
void from_json(const nlohmann::json& json, RecentFiles& recent);

} // namespace reco::io
