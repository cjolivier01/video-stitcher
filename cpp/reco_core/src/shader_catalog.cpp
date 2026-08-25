#include "reco/core/shader_catalog.hpp"

#include <algorithm>

namespace reco::core {

const std::vector<ShaderAsset>& shader_catalog() {
  static const std::vector<ShaderAsset> assets = {
      {.name = "bayer_demosaic",
       .path = "cpp/reco_core/shaders/bayer_demosaic.wgsl",
       .stages = {ShaderStage::kCompute},
       .entry_points = {"main"}},
      {.name = "color_grade",
       .path = "cpp/reco_core/shaders/color_grade.wgsl",
       .stages = {ShaderStage::kCompute},
       .entry_points = {"main"}},
      {.name = "cylindrical_mono",
       .path = "cpp/reco_core/shaders/cylindrical_mono.wgsl",
       .stages = {ShaderStage::kVertex, ShaderStage::kFragment},
       .entry_points = {"vs_fullscreen", "fs_cylindrical_mono"}},
      {.name = "fisheye",
       .path = "cpp/reco_core/shaders/fisheye.wgsl",
       .stages = {ShaderStage::kVertex, ShaderStage::kFragment},
       .entry_points = {"vs_main", "fs_main"}},
      {.name = "rgba_to_nv12",
       .path = "cpp/reco_core/shaders/rgba_to_nv12.wgsl",
       .stages = {ShaderStage::kCompute},
       .entry_points = {"main"}},
      {.name = "yuv420p_stack_pack",
       .path = "cpp/reco_core/shaders/yuv420p_stack_pack.wgsl",
       .stages = {ShaderStage::kCompute},
       .entry_points = {"pack_y", "pack_u", "pack_v", "pack_uv_from_nv12"}},
  };
  return assets;
}

const ShaderAsset* find_shader(std::string_view name) {
  const auto& assets = shader_catalog();
  const auto it = std::find_if(assets.begin(), assets.end(),
                               [name](const auto& asset) { return asset.name == name; });
  return it == assets.end() ? nullptr : &*it;
}

} // namespace reco::core
