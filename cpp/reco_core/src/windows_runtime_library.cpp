#include "reco/core/windows_runtime_library.hpp"

#if defined(_WIN32)

#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

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

struct WindowsPathSearch {
  std::optional<std::wstring> library;
  std::vector<std::wstring> dependency_directories;
};

WindowsPathSearch
find_windows_runtime_library_on_path(const std::filesystem::path& requested_path) {
  WindowsPathSearch result;
  const auto required = GetEnvironmentVariableW(L"PATH", nullptr, 0);
  if (required == 0) {
    return result;
  }
  std::wstring path_value(static_cast<std::size_t>(required), L'\0');
  const auto written = GetEnvironmentVariableW(L"PATH", path_value.data(), required);
  if (written == 0 || written >= required) {
    return result;
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
        result.dependency_directories.push_back(normalized_directory.native());
        const auto candidate = normalized_directory / requested_path;
        if (!result.library.has_value() && std::filesystem::is_regular_file(candidate, error) &&
            !error) {
          result.library = candidate.native();
        }
      }
    }
    if (separator == std::wstring::npos) {
      break;
    }
    offset = separator + 1;
  }
  return result;
}

void* load_windows_path_runtime_library(const WindowsPathSearch& search) {
  if (!search.library.has_value()) {
    SetLastError(ERROR_FILE_NOT_FOUND);
    return nullptr;
  }
  std::vector<DLL_DIRECTORY_COOKIE> cookies;
  cookies.reserve(search.dependency_directories.size());
  for (const auto& directory : search.dependency_directories) {
    const auto cookie = AddDllDirectory(directory.c_str());
    if (cookie == nullptr) {
      const auto error = GetLastError();
      for (auto iterator = cookies.rbegin(); iterator != cookies.rend(); ++iterator) {
        RemoveDllDirectory(*iterator);
      }
      SetLastError(error);
      return nullptr;
    }
    cookies.push_back(cookie);
  }
  auto* library =
      LoadLibraryExW(search.library->c_str(), nullptr,
                     LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
  const auto load_error = library == nullptr ? GetLastError() : ERROR_SUCCESS;
  for (auto iterator = cookies.rbegin(); iterator != cookies.rend(); ++iterator) {
    RemoveDllDirectory(*iterator);
  }
  if (library == nullptr) {
    SetLastError(load_error);
  }
  return library;
}

} // namespace

void* load_windows_runtime_library(std::string_view requested) {
  const auto wide = utf8_to_wide(requested);
  if (!wide.has_value()) {
    SetLastError(ERROR_INVALID_NAME);
    return nullptr;
  }
  const std::filesystem::path requested_path(*wide);
  if (requested_path.has_parent_path() && !requested_path.is_absolute()) {
    SetLastError(ERROR_INVALID_NAME);
    return nullptr;
  }
  if (requested_path.is_absolute()) {
    const auto normalized = requested_path.lexically_normal().native();
    return LoadLibraryExW(normalized.c_str(), nullptr,
                          LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
  }

  if (auto* library = LoadLibraryExW(wide->c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      library != nullptr) {
    return library;
  }
  const auto default_search_error = GetLastError();
  const auto path_search = find_windows_runtime_library_on_path(requested_path);
  if (!path_search.library.has_value()) {
    SetLastError(default_search_error);
    return nullptr;
  }
  return load_windows_path_runtime_library(path_search);
}

} // namespace reco::core::detail

#endif
