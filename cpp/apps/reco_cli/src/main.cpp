#include "reco/cli/cli.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace {

#if defined(_WIN32)
std::string wide_to_utf8(std::wstring_view value) {
  if (value.empty()) {
    return {};
  }
  if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("Windows CLI argument is too long");
  }
  const auto input_size = static_cast<int>(value.size());
  const int output_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                                              input_size, nullptr, 0, nullptr, nullptr);
  if (output_size <= 0) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                            "cannot convert Windows CLI argument to UTF-8");
  }
  std::string result(static_cast<std::size_t>(output_size), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), input_size, result.data(),
                          output_size, nullptr, nullptr) != output_size) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                            "cannot convert Windows CLI argument to UTF-8");
  }
  return result;
}
#endif

std::optional<std::filesystem::path> current_executable_path() {
#if defined(_WIN32)
  for (DWORD capacity = 512; capacity <= 32768; capacity *= 2) {
    std::vector<wchar_t> buffer(capacity);
    const auto length = GetModuleFileNameW(nullptr, buffer.data(), capacity);
    if (length == 0) {
      return std::nullopt;
    }
    if (length + 1 < capacity) {
      return std::filesystem::path(std::wstring_view(buffer.data(), length));
    }
  }
#elif defined(__APPLE__)
  std::uint32_t size = 0;
  if (_NSGetExecutablePath(nullptr, &size) != -1 || size == 0) {
    return std::nullopt;
  }
  std::vector<char> buffer(static_cast<std::size_t>(size) + 1);
  if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
    std::error_code error;
    auto resolved = std::filesystem::weakly_canonical(buffer.data(), error);
    return error ? std::optional<std::filesystem::path>(std::filesystem::path(buffer.data()))
                 : std::optional<std::filesystem::path>(std::move(resolved));
  }
#elif defined(__linux__)
  std::vector<char> buffer(512);
  while (buffer.size() <= 1024U * 1024U) {
    const auto length = ::readlink("/proc/self/exe", buffer.data(), buffer.size());
    if (length < 0) {
      return std::nullopt;
    }
    if (static_cast<std::size_t>(length) < buffer.size()) {
      return std::filesystem::path(
          std::string_view(buffer.data(), static_cast<std::size_t>(length)));
    }
    buffer.resize(buffer.size() * 2);
  }
#endif
  return std::nullopt;
}

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  try {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc > 1 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) {
#if defined(_WIN32)
      args.push_back(wide_to_utf8(argv[i]));
#else
      args.emplace_back(argv[i]);
#endif
    }

    auto parsed = reco::cli::parse_args(args);
    if (const auto* error = std::get_if<reco::cli::ParseError>(&parsed)) {
      std::cerr << "error: " << error->message << "\n\n" << reco::cli::help_text() << '\n';
      return 2;
    }

    const auto command = std::get<reco::cli::Command>(std::move(parsed));
    auto executable = current_executable_path().value_or(argc > 0 ? std::filesystem::path(argv[0])
                                                                  : std::filesystem::path{});
    return reco::cli::run_command(command, std::cout, std::cerr, executable);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
