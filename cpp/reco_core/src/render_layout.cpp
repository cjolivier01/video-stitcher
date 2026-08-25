#include "reco/core/render_layout.hpp"

namespace reco::core {

std::size_t input_format_plane_count(InputFormat format) {
  switch (format) {
  case InputFormat::kYuv420p:
    return 3;
  case InputFormat::kNv12:
    return 2;
  case InputFormat::kBgra:
    return 1;
  }
  return 0;
}

bool input_format_uses_dummy_v_texture(InputFormat format) {
  return format == InputFormat::kNv12 || format == InputFormat::kBgra;
}

std::vector<InputTextureBinding> input_format_texture_bindings(InputFormat format) {
  switch (format) {
  case InputFormat::kYuv420p:
    return {
        {.binding = 0, .format = TextureFormat::kR8Unorm, .full_resolution = true, .label = "y"},
        {.binding = 1, .format = TextureFormat::kR8Unorm, .full_resolution = false, .label = "u"},
        {.binding = 2, .format = TextureFormat::kR8Unorm, .full_resolution = false, .label = "v"},
    };
  case InputFormat::kNv12:
    return {
        {.binding = 0, .format = TextureFormat::kR8Unorm, .full_resolution = true, .label = "y"},
        {.binding = 1, .format = TextureFormat::kRg8Unorm, .full_resolution = false, .label = "uv"},
        {.binding = 2,
         .format = TextureFormat::kR8Unorm,
         .full_resolution = false,
         .dummy = true,
         .label = "v_dummy"},
    };
  case InputFormat::kBgra:
    return {
        {.binding = 0,
         .format = TextureFormat::kRgba8Unorm,
         .full_resolution = true,
         .label = "rgba"},
        {.binding = 1,
         .format = TextureFormat::kR8Unorm,
         .full_resolution = false,
         .dummy = true,
         .label = "u_dummy"},
        {.binding = 2,
         .format = TextureFormat::kR8Unorm,
         .full_resolution = false,
         .dummy = true,
         .label = "v_dummy"},
    };
  }
  return {};
}

TextureFormat gpu_pixel_y_format(GpuPixelFormat format) {
  switch (format) {
  case GpuPixelFormat::kNv12:
    return TextureFormat::kR8Unorm;
  case GpuPixelFormat::kP010:
    return TextureFormat::kR16Unorm;
  }
  return TextureFormat::kR8Unorm;
}

TextureFormat gpu_pixel_uv_format(GpuPixelFormat format) {
  switch (format) {
  case GpuPixelFormat::kNv12:
    return TextureFormat::kRg8Unorm;
  case GpuPixelFormat::kP010:
    return TextureFormat::kRg16Unorm;
  }
  return TextureFormat::kRg8Unorm;
}

TextureFormat gpu_pixel_wgpu_format(GpuPixelFormat format) {
  switch (format) {
  case GpuPixelFormat::kNv12:
    return TextureFormat::kNv12;
  case GpuPixelFormat::kP010:
    return TextureFormat::kP010;
  }
  return TextureFormat::kNv12;
}

std::size_t gpu_pixel_bytes_per_sample(GpuPixelFormat format) {
  switch (format) {
  case GpuPixelFormat::kNv12:
    return 1;
  case GpuPixelFormat::kP010:
    return 2;
  }
  return 0;
}

std::vector<BindGroupLayoutEntry> stitch_texture_bindings() {
  return {
      {.group = 0,
       .binding = 0,
       .kind = BindingResourceKind::kTextureFloatFilterable2D,
       .fragment_visible = true},
      {.group = 0,
       .binding = 1,
       .kind = BindingResourceKind::kTextureFloatFilterable2D,
       .fragment_visible = true},
      {.group = 0,
       .binding = 2,
       .kind = BindingResourceKind::kTextureFloatFilterable2D,
       .fragment_visible = true},
      {.group = 0,
       .binding = 3,
       .kind = BindingResourceKind::kFilteringSampler,
       .fragment_visible = true},
  };
}

std::vector<BindGroupLayoutEntry> stitch_uniform_bindings() {
  return {{.group = 1,
           .binding = 0,
           .kind = BindingResourceKind::kUniformBuffer,
           .vertex_visible = true,
           .fragment_visible = true}};
}

std::vector<VertexAttributeLayout> stitch_vertex_attributes() {
  return {
      {.shader_location = 0, .offset = 0, .format = "Float32x3"},
      {.shader_location = 1, .offset = 12, .format = "Float32x2"},
  };
}

} // namespace reco::core
