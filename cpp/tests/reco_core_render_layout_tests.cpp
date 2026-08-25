#include "reco/core/render_layout.hpp"
#include "reco/core/shader_catalog.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

using namespace reco::core;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

std::filesystem::path runfile_path(std::string_view relative) {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (test_srcdir == nullptr || workspace == nullptr) {
    return std::filesystem::path(relative);
  }
  return std::filesystem::path(test_srcdir) / workspace / std::string(relative);
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "FAIL: unable to open " << path << '\n';
    ++failures;
    return {};
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void uniform_and_vertex_layout_match_rust_abi() {
  expect_eq(sizeof(GpuUniforms), 160U, "GpuUniforms size");
  expect_eq(alignof(GpuUniforms), 4U, "GpuUniforms alignment");
  expect_eq(offsetof(GpuUniforms, mvp), 0U, "GpuUniforms mvp offset");
  expect_eq(offsetof(GpuUniforms, intrinsics), 64U, "GpuUniforms intrinsics offset");
  expect_eq(offsetof(GpuUniforms, dist), 80U, "GpuUniforms dist offset");
  expect_eq(offsetof(GpuUniforms, color_scale), 96U, "GpuUniforms color scale offset");
  expect_eq(offsetof(GpuUniforms, color_offset_blend), 112U,
            "GpuUniforms color offset/blend offset");
  expect_eq(offsetof(GpuUniforms, flags), 128U, "GpuUniforms flags offset");
  expect_eq(offsetof(GpuUniforms, lens_preview), 144U, "GpuUniforms lens preview offset");

  expect_eq(sizeof(Vertex), 20U, "Vertex size");
  expect_eq(offsetof(Vertex, position), 0U, "Vertex position offset");
  expect_eq(offsetof(Vertex, uv), 12U, "Vertex uv offset");

  const auto attributes = stitch_vertex_attributes();
  expect_eq(attributes.size(), 2U, "vertex attribute count");
  if (attributes.size() == 2) {
    expect_eq(attributes[0].shader_location, 0U, "position shader location");
    expect_eq(attributes[0].offset, 0U, "position attribute offset");
    expect_eq(attributes[0].format, std::string_view("Float32x3"), "position attribute format");
    expect_eq(attributes[1].shader_location, 1U, "uv shader location");
    expect_eq(attributes[1].offset, 12U, "uv attribute offset");
    expect_eq(attributes[1].format, std::string_view("Float32x2"), "uv attribute format");
  }
}

void input_and_gpu_pixel_formats_match_rust() {
  expect_eq(input_format_plane_count(InputFormat::kYuv420p), 3U, "yuv420p plane count");
  expect_eq(input_format_plane_count(InputFormat::kNv12), 2U, "nv12 plane count");
  expect_eq(input_format_plane_count(InputFormat::kBgra), 1U, "bgra plane count");
  expect_true(!input_format_uses_dummy_v_texture(InputFormat::kYuv420p), "yuv420p v texture");
  expect_true(input_format_uses_dummy_v_texture(InputFormat::kNv12), "nv12 dummy v texture");
  expect_true(input_format_uses_dummy_v_texture(InputFormat::kBgra), "bgra dummy v texture");

  const auto yuv = input_format_texture_bindings(InputFormat::kYuv420p);
  expect_eq(yuv.size(), 3U, "yuv420p texture binding count");
  if (yuv.size() == 3) {
    expect_eq(yuv[0].binding, 0U, "yuv y binding");
    expect_true(yuv[0].format == TextureFormat::kR8Unorm, "yuv y format");
    expect_true(yuv[0].full_resolution, "yuv y full resolution");
    expect_true(!yuv[0].dummy, "yuv y not dummy");
    expect_eq(yuv[1].binding, 1U, "yuv u binding");
    expect_true(yuv[1].format == TextureFormat::kR8Unorm, "yuv u format");
    expect_true(!yuv[1].full_resolution, "yuv u half resolution");
    expect_true(!yuv[1].dummy, "yuv u not dummy");
    expect_eq(yuv[2].binding, 2U, "yuv v binding");
    expect_true(yuv[2].format == TextureFormat::kR8Unorm, "yuv v format");
    expect_true(!yuv[2].full_resolution, "yuv v half resolution");
    expect_true(!yuv[2].dummy, "yuv v not dummy");
  }

  const auto nv12 = input_format_texture_bindings(InputFormat::kNv12);
  expect_eq(nv12.size(), 3U, "nv12 texture binding count");
  if (nv12.size() == 3) {
    expect_true(nv12[0].format == TextureFormat::kR8Unorm, "nv12 y binding format");
    expect_true(nv12[1].format == TextureFormat::kRg8Unorm, "nv12 uv binding format");
    expect_true(!nv12[1].dummy, "nv12 uv not dummy");
    expect_true(nv12[2].format == TextureFormat::kR8Unorm, "nv12 dummy v format");
    expect_true(nv12[2].dummy, "nv12 dummy v");
  }

  const auto bgra = input_format_texture_bindings(InputFormat::kBgra);
  expect_eq(bgra.size(), 3U, "bgra texture binding count");
  if (bgra.size() == 3) {
    expect_true(bgra[0].format == TextureFormat::kRgba8Unorm, "bgra rgba binding format");
    expect_true(bgra[0].full_resolution, "bgra rgba full resolution");
    expect_true(!bgra[0].dummy, "bgra rgba not dummy");
    expect_true(bgra[1].format == TextureFormat::kR8Unorm, "bgra dummy u format");
    expect_true(bgra[1].dummy, "bgra dummy u");
    expect_true(bgra[2].format == TextureFormat::kR8Unorm, "bgra dummy v format");
    expect_true(bgra[2].dummy, "bgra dummy v");
  }

  expect_true(gpu_pixel_y_format(GpuPixelFormat::kNv12) == TextureFormat::kR8Unorm,
              "nv12 y format");
  expect_true(gpu_pixel_uv_format(GpuPixelFormat::kNv12) == TextureFormat::kRg8Unorm,
              "nv12 uv format");
  expect_true(gpu_pixel_wgpu_format(GpuPixelFormat::kNv12) == TextureFormat::kNv12,
              "nv12 full texture format");
  expect_eq(gpu_pixel_bytes_per_sample(GpuPixelFormat::kNv12), 1U, "nv12 bytes per sample");

  expect_true(gpu_pixel_y_format(GpuPixelFormat::kP010) == TextureFormat::kR16Unorm,
              "p010 y format");
  expect_true(gpu_pixel_uv_format(GpuPixelFormat::kP010) == TextureFormat::kRg16Unorm,
              "p010 uv format");
  expect_true(gpu_pixel_wgpu_format(GpuPixelFormat::kP010) == TextureFormat::kP010,
              "p010 full texture format");
  expect_eq(gpu_pixel_bytes_per_sample(GpuPixelFormat::kP010), 2U, "p010 bytes per sample");
}

void stitch_bindings_match_fisheye_wgsl() {
  const auto* fisheye = find_shader("fisheye");
  expect_true(fisheye != nullptr, "fisheye shader catalog entry");
  if (fisheye == nullptr) {
    return;
  }
  const auto source = read_text(runfile_path(fisheye->path));

  const auto texture_bindings = stitch_texture_bindings();
  expect_eq(texture_bindings.size(), 4U, "texture binding count");
  for (const auto& binding : texture_bindings) {
    expect_true(binding.group == 0, "texture binding group 0");
    expect_true(binding.fragment_visible, "texture binding fragment visibility");
    expect_true(source.find("@group(0) @binding(" + std::to_string(binding.binding) + ")") !=
                    std::string::npos,
                "texture binding appears in fisheye wgsl");
  }

  const auto uniform_bindings = stitch_uniform_bindings();
  expect_eq(uniform_bindings.size(), 1U, "uniform binding count");
  if (!uniform_bindings.empty()) {
    expect_eq(uniform_bindings[0].group, 1U, "uniform group");
    expect_eq(uniform_bindings[0].binding, 0U, "uniform binding");
    expect_true(uniform_bindings[0].vertex_visible, "uniform vertex visibility");
    expect_true(uniform_bindings[0].fragment_visible, "uniform fragment visibility");
    expect_true(source.find("@group(1) @binding(0) var<uniform> u: Uniforms;") !=
                    std::string::npos,
                "uniform binding appears in fisheye wgsl");
  }
}

} // namespace

int main() {
  uniform_and_vertex_layout_match_rust_abi();
  input_and_gpu_pixel_formats_match_rust();
  stitch_bindings_match_fisheye_wgsl();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
