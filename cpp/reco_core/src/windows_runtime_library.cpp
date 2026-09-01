#include "reco/core/windows_runtime_library.hpp"

#if defined(_WIN32)

#include <filesystem>
#include <limits>
#include <optional>
#include <string>

#include <windows.h>

namespace reco::core::detail {
namespace {

std::optional<std::wstring> utf8_to_wide(std::string_view value) {
  if (value.empty() || value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                        static_cast<int>(value.size()), nullptr, 0);
  if (size <= 0) {
    return std::nullopt;
  }
  std::wstring result(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), result.data(), size) != size) {
    return std::nullopt;
  }
  return result;
}

std::optional<std::wstring> resolve_windows_runtime_library(std::wstring requested) {
  const std::filesystem::path requested_path(requested);
  if (requested_path.is_absolute()) {
    return requested_path.lexically_normal().native();
  }
  if (requested_path.has_parent_path()) {
    return std::nullopt;
  }

  const auto required = GetEnvironmentVariableW(L"PATH", nullptr, 0);
  if (required == 0) {
    return requested;
  }
  std::wstring path_value(static_cast<std::size_t>(required), L'\0');
  const auto written = GetEnvironmentVariableW(L"PATH", path_value.data(), required);
  if (written == 0 || written >= required) {
    return requested;
  }
  path_value.resize(written);
  std::size_t offset = 0;
  while (offset <= path_value.size()) {
    const auto separator = path_value.find(L';', offset);
    auto directory = path_value.substr(
        offset, separator == std::wstring::npos ? std::wstring::npos : separator - offset);
    if (directory.size() >= 2 && directory.front() == L'"' && directory.back() == L'"') {
      directory = directory.substr(1, directory.size() - 2);
    }
    const std::filesystem::path directory_path(directory);
    if (!directory.empty() && directory_path.is_absolute()) {
      const auto candidate = (directory_path / requested_path).lexically_normal();
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error) && !error) {
        return candidate.native();
      }
    }
    if (separator == std::wstring::npos) {
      break;
    }
    offset = separator + 1;
  }
  return requested;
}

} // namespace

void* load_windows_runtime_library(std::string_view requested) {
  const auto wide = utf8_to_wide(requested);
  const auto resolved = wide.has_value() ? resolve_windows_runtime_library(*wide) : std::nullopt;
  if (!resolved.has_value()) {
    SetLastError(ERROR_INVALID_NAME);
    return nullptr;
  }
  auto flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  if (std::filesystem::path(*resolved).is_absolute()) {
    flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
  }
  return LoadLibraryExW(resolved->c_str(), nullptr, flags);
}

} // namespace reco::core::detail

#endif
