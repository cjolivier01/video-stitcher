#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace reco::detect {

struct OrtRuntimeProbe {
  bool available = false;
  std::string path;
  std::string version;
  std::string error;
};

[[nodiscard]] OrtRuntimeProbe probe_ort_runtime();
[[nodiscard]] bool ort_runtime_available();
[[nodiscard]] std::string ort_runtime_error();
[[nodiscard]] std::filesystem::path reco_cache_dir(std::string_view subdir);

} // namespace reco::detect
