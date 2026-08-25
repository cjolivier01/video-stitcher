#pragma once

#include <string_view>
#include <vector>

namespace reco::core {

enum class ShaderStage {
  kVertex,
  kFragment,
  kCompute,
};

struct ShaderAsset {
  std::string_view name;
  std::string_view path;
  std::vector<ShaderStage> stages;
  std::vector<std::string_view> entry_points;
};

[[nodiscard]] const std::vector<ShaderAsset>& shader_catalog();
[[nodiscard]] const ShaderAsset* find_shader(std::string_view name);

} // namespace reco::core
