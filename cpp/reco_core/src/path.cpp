#include "reco/core/path.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace reco::core {

std::filesystem::path path_from_utf8(std::string_view value) {
#if defined(_WIN32)
  return std::filesystem::u8path(value.begin(), value.end());
#else
  return std::filesystem::path(value);
#endif
}

std::string path_to_utf8(const std::filesystem::path& value) {
  const auto encoded = value.u8string();
  return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

std::optional<std::filesystem::path> path_from_environment(std::string_view name) {
  if (name.empty() || name.find('\0') != std::string_view::npos) {
    throw std::invalid_argument("environment variable name must be non-empty and contain no NUL");
  }
#if defined(_WIN32)
  std::wstring wide_name;
  wide_name.reserve(name.size());
  for (const unsigned char character : name) {
    if (character > 0x7fU) {
      throw std::invalid_argument("environment variable name must be ASCII");
    }
    wide_name.push_back(static_cast<wchar_t>(character));
  }

  DWORD capacity = GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
  if (capacity == 0) {
    return std::nullopt;
  }
  while (true) {
    if (capacity > static_cast<DWORD>(std::numeric_limits<int>::max())) {
      throw std::length_error("environment path is too long");
    }
    std::vector<wchar_t> value(capacity);
    const auto written = GetEnvironmentVariableW(wide_name.c_str(), value.data(), capacity);
    if (written == 0) {
      return std::nullopt;
    }
    if (written < capacity) {
      return std::filesystem::path(std::wstring_view(value.data(), written));
    }
    capacity = written + 1U;
  }
#else
  const std::string native_name(name);
  const char* value = std::getenv(native_name.c_str());
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::filesystem::path(value);
#endif
}

} // namespace reco::core
