#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace reco::core {

inline constexpr float kNearPlane = 0.01F;
inline constexpr float kFarPlane = 5.0F;
inline constexpr float kDeprecatedPlaneAspect = 16.0F / 9.0F;

struct GpuUniforms {
  std::array<float, 16> mvp{};
  std::array<float, 4> intrinsics{};
  std::array<float, 4> dist{};
  std::array<float, 4> color_scale{};
  std::array<float, 4> color_offset_blend{};
  std::array<std::uint32_t, 4> flags{};
  std::array<float, 4> lens_preview{};
};

struct Vertex {
  std::array<float, 3> position{};
  std::array<float, 2> uv{};
};

enum class InputFormat {
  kYuv420p,
  kNv12,
  kBgra,
};

enum class GpuPixelFormat {
  kNv12,
  kP010,
};

enum class TextureFormat {
  kR8Unorm,
  kRg8Unorm,
  kR16Unorm,
  kRg16Unorm,
  kRgba8Unorm,
  kNv12,
  kP010,
};

enum class BindingResourceKind {
  kTextureFloatFilterable2D,
  kFilteringSampler,
  kUniformBuffer,
};

struct BindGroupLayoutEntry {
  std::uint32_t group = 0;
  std::uint32_t binding = 0;
  BindingResourceKind kind = BindingResourceKind::kTextureFloatFilterable2D;
  bool vertex_visible = false;
  bool fragment_visible = false;
};

struct InputTextureBinding {
  std::uint32_t binding = 0;
  TextureFormat format = TextureFormat::kR8Unorm;
  bool full_resolution = true;
  bool dummy = false;
  std::string_view label;
};

struct VertexAttributeLayout {
  std::uint32_t shader_location = 0;
  std::size_t offset = 0;
  std::string_view format;
};

[[nodiscard]] std::size_t input_format_plane_count(InputFormat format);
[[nodiscard]] bool input_format_uses_dummy_v_texture(InputFormat format);
[[nodiscard]] std::vector<InputTextureBinding> input_format_texture_bindings(InputFormat format);
[[nodiscard]] TextureFormat gpu_pixel_y_format(GpuPixelFormat format);
[[nodiscard]] TextureFormat gpu_pixel_uv_format(GpuPixelFormat format);
[[nodiscard]] TextureFormat gpu_pixel_wgpu_format(GpuPixelFormat format);
[[nodiscard]] std::size_t gpu_pixel_bytes_per_sample(GpuPixelFormat format);
[[nodiscard]] std::vector<BindGroupLayoutEntry> stitch_texture_bindings();
[[nodiscard]] std::vector<BindGroupLayoutEntry> stitch_uniform_bindings();
[[nodiscard]] std::vector<VertexAttributeLayout> stitch_vertex_attributes();

} // namespace reco::core
