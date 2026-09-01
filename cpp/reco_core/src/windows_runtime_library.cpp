#include "reco/core/windows_runtime_library.hpp"

#if defined(_WIN32)

#include <filesystem>
#include <optional>

#include <windows.h>

namespace reco::core::detail {
namespace {

constexpr DWORD kIsolatedRuntimeSearch =
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;

std::optional<std::filesystem::path>
find_windows_runtime_library_on_path(const std::filesystem::path& requested_path) {
  const auto required = GetEnvironmentVariableW(L"PATH", nullptr, 0);
  if (required == 0) {
    return std::nullopt;
  }
  std::wstring path_value(static_cast<std::size_t>(required), L'\0');
  const auto written = GetEnvironmentVariableW(L"PATH", path_value.data(), required);
  if (written == 0 || written >= required) {
    return std::nullopt;
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
      const auto normalized_directory = directory_path.lexically_normal();
      std::error_code error;
      if (std::filesystem::is_directory(normalized_directory, error) && !error) {
        const auto candidate = normalized_directory / requested_path;
        if (std::filesystem::is_regular_file(candidate, error) && !error) {
          return candidate;
        }
      }
    }
    if (separator == std::wstring::npos) {
      break;
    }
    offset = separator + 1;
  }
  return std::nullopt;
}

} // namespace

void* load_windows_runtime_library(const std::filesystem::path& requested) {
  if (requested.empty()) {
    SetLastError(ERROR_INVALID_NAME);
    return nullptr;
  }
  if (requested.has_parent_path() && !requested.is_absolute()) {
    SetLastError(ERROR_INVALID_NAME);
    return nullptr;
  }
  if (requested.is_absolute()) {
    const auto normalized = requested.lexically_normal();
    return LoadLibraryExW(normalized.c_str(), nullptr, kIsolatedRuntimeSearch);
  }

  if (auto* library = LoadLibraryExW(requested.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      library != nullptr) {
    return library;
  }
  const auto default_search_error = GetLastError();
  const auto path_library = find_windows_runtime_library_on_path(requested);
  if (!path_library.has_value()) {
    SetLastError(default_search_error);
    return nullptr;
  }
  return LoadLibraryExW(path_library->c_str(), nullptr, kIsolatedRuntimeSearch);
}

} // namespace reco::core::detail

#endif
