#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace reco::core {

/// Constructs a native filesystem path from a validated UTF-8 protocol or CLI string.
[[nodiscard]] std::filesystem::path path_from_utf8(std::string_view value);

/// Encodes a native filesystem path for UTF-8 protocol and CLI transport.
[[nodiscard]] std::string path_to_utf8(const std::filesystem::path& value);

} // namespace reco::core
