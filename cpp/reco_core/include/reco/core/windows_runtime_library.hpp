#pragma once

#include <filesystem>

namespace reco::core::detail {

#if defined(_WIN32)
// Loads an explicit runtime DLL without consulting the process working directory.
[[nodiscard]] void* load_windows_runtime_library(const std::filesystem::path& requested);
#endif

} // namespace reco::core::detail
