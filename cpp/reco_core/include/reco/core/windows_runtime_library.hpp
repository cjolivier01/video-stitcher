#pragma once

#include <string_view>

namespace reco::core::detail {

#if defined(_WIN32)
// Loads an explicit runtime DLL without consulting the process working directory.
[[nodiscard]] void* load_windows_runtime_library(std::string_view requested);
#endif

} // namespace reco::core::detail
