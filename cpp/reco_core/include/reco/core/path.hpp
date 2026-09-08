#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace reco::core {

/// Constructs a native filesystem path from a validated UTF-8 protocol or CLI string.
[[nodiscard]] std::filesystem::path path_from_utf8(std::string_view value);

/// Encodes a native filesystem path for UTF-8 protocol and CLI transport.
[[nodiscard]] std::string path_to_utf8(const std::filesystem::path& value);

/// Reads a native filesystem path from an environment variable without
/// passing Windows values through the active ANSI code page.
[[nodiscard]] std::optional<std::filesystem::path> path_from_environment(std::string_view name);

} // namespace reco::core
