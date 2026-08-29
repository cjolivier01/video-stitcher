#include "reco/core/cuda_stitch_renderer.hpp"

#include "reco/core/projection.hpp"
#include "reco/core/video_format.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace reco::core;

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kNear = 0.01F;
constexpr float kFar = 5.0F;
int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function> void run_case(std::string_view name, Function&& function) {
  try {
    function();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << name << " threw: " << error.what() << '\n';
    ++failures;
  } catch (...) {
    std::cerr << "FAIL: " << name << " threw an unknown exception\n";
    ++failures;
  }
}

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

bool address_sanitizer_build() {
#if defined(__SANITIZE_ADDRESS__)
  return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
  return true;
#else
  return false;
#endif
#else
  return false;
#endif
}

struct Float3 {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
};

Float3 add(Float3 lhs, Float3 rhs) { return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z}; }
Float3 subtract(Float3 lhs, Float3 rhs) { return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z}; }
Float3 multiply(Float3 value, float scalar) {
  return {value.x * scalar, value.y * scalar, value.z * scalar};
}
float dot(Float3 lhs, Float3 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
Float3 cross(Float3 lhs, Float3 rhs) {
  return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z,
          lhs.x * rhs.y - lhs.y * rhs.x};
}
Float3 normalize(Float3 value) {
  const float length = std::sqrt(dot(value, value));
  if (!(length > 1.0e-12F)) {
    throw std::runtime_error("reference geometry contains a degenerate vector");
  }
  return multiply(value, 1.0F / length);
}
Float3 rotate(Float3 value, Float3 axis, float angle) {
  axis = normalize(axis);
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  return add(add(multiply(value, cosine), multiply(cross(axis, value), sine)),
             multiply(axis, dot(axis, value) * (1.0F - cosine)));
}
Float3 from_vec3(Vec3 value) { return {value.x, value.y, value.z}; }

struct ReferenceBasis {
  Float3 origin;
  Float3 right;
  Float3 up;
  Float3 normal;
};

ReferenceBasis reference_plane_basis(CameraId camera, const SceneGeometry& scene) {
  const Float3 origin = from_vec3(plane_uv_to_world({0.5, 0.5}, camera, scene));
  const Float3 right = subtract(from_vec3(plane_uv_to_world({2.5, 0.5}, camera, scene)), origin);
  const Float3 up = subtract(
      from_vec3(plane_uv_to_world({0.5, 0.5 - 2.0 * scene.plane_aspect}, camera, scene)), origin);
  return {.origin = origin, .right = right, .up = up, .normal = normalize(cross(right, up))};
}

struct ReferenceView {
  Float3 eye;
  Float3 forward;
  Float3 right;
  Float3 up;
  float tan_half_fov = 0.0F;
  float output_aspect = 1.0F;
};

ReferenceView reference_view(const CudaStitchRendererConfig& config,
                             const CudaStitchViewport& viewport) {
  const auto aspect = static_cast<float>(config.calibration.left.width) /
                      static_cast<float>(config.calibration.left.height);
  const auto scene = SceneGeometry::from_layout_with_aspect(config.calibration.layout, aspect);
  const Float3 eye{scene.camera_position[0], scene.camera_position[1], scene.camera_position[2]};
  Float3 forward = normalize(multiply(eye, -1.0F));
  const Float3 world_up{0.0F, 1.0F, 0.0F};
  const Float3 base_right = normalize(cross(forward, world_up));
  Float3 up = world_up;
  if (std::abs(config.calibration.rig_tilt) > 1.0e-6) {
    forward = rotate(forward, base_right, static_cast<float>(config.calibration.rig_tilt));
    up = rotate(up, base_right, static_cast<float>(config.calibration.rig_tilt));
  }
  if (std::abs(config.calibration.rig_roll) > 1.0e-6) {
    up = rotate(up, forward, -static_cast<float>(config.calibration.rig_roll));
  }
  forward = rotate(forward, up, viewport.yaw);
  const Float3 pitch_axis = rotate(base_right, up, viewport.yaw);
  forward = rotate(forward, pitch_axis, viewport.pitch);
  up = rotate(up, pitch_axis, viewport.pitch);
  const Float3 screen_right = normalize(cross(forward, up));
  return {.eye = eye,
          .forward = normalize(forward),
          .right = screen_right,
          .up = normalize(cross(screen_right, forward)),
          .tan_half_fov = std::tan(viewport.fov_degrees * kPi / 360.0F),
          .output_aspect =
              static_cast<float>(config.output_width) / static_cast<float>(config.output_height)};
}

struct HostNv12 {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> uv;
  YuvColorMatrix matrix = YuvColorMatrix::Bt709;
  YuvColorRange range = YuvColorRange::Limited;
  bool flip_180 = false;
};

float bilinear_plane(const std::vector<std::uint8_t>& data, std::uint32_t width,
                     std::uint32_t height, std::uint32_t channels, std::uint32_t channel, float u,
                     float v) {
  const float sample_x = u * static_cast<float>(width) - 0.5F;
  const float sample_y = v * static_cast<float>(height) - 0.5F;
  const int floor_x = static_cast<int>(std::floor(sample_x));
  const int floor_y = static_cast<int>(std::floor(sample_y));
  const float tx = sample_x - static_cast<float>(floor_x);
  const float ty = sample_y - static_cast<float>(floor_y);
  const auto x0 = std::clamp(floor_x, 0, static_cast<int>(width) - 1);
  const auto x1 = std::clamp(floor_x + 1, 0, static_cast<int>(width) - 1);
  const auto y0 = std::clamp(floor_y, 0, static_cast<int>(height) - 1);
  const auto y1 = std::clamp(floor_y + 1, 0, static_cast<int>(height) - 1);
  const auto sample = [&](int x, int y) {
    return static_cast<float>(
        data[(static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)) * channels +
             channel]);
  };
  const float top = sample(x0, y0) + (sample(x1, y0) - sample(x0, y0)) * tx;
  const float bottom = sample(x0, y1) + (sample(x1, y1) - sample(x0, y1)) * tx;
  return top + (bottom - top) * ty;
}

struct ReferencePixel {
  float r = 0.0F;
  float g = 0.0F;
  float b = 0.0F;
  float a = 0.0F;
};

ReferencePixel sample_reference_nv12(const HostNv12& frame, float u, float v) {
  if (frame.flip_180) {
    u = 1.0F - u;
    v = 1.0F - v;
  }
  const float y_raw = bilinear_plane(frame.y, frame.width, frame.height, 1, 0, u, v);
  const float u_raw = bilinear_plane(frame.uv, frame.width / 2U, frame.height / 2U, 2, 0, u, v);
  const float v_raw = bilinear_plane(frame.uv, frame.width / 2U, frame.height / 2U, 2, 1, u, v);
  const auto coefficients = yuv_to_rgb_coefficients(frame.matrix, frame.range);
  const float center = frame.range == YuvColorRange::Full ? 127.5F : 128.0F;
  const float y_value = (y_raw + coefficients.y_offset) * coefficients.y_scale;
  const float cb = u_raw - center;
  const float cr = v_raw - center;
  return {.r = std::clamp(y_value + coefficients.red_from_v * cr, 0.0F, 255.0F),
          .g = std::clamp(y_value + coefficients.green_from_u * cb + coefficients.green_from_v * cr,
                          0.0F, 255.0F),
          .b = std::clamp(y_value + coefficients.blue_from_u * cb, 0.0F, 255.0F),
          .a = 1.0F};
}

float smoothstep(float width, float value) {
  const float t = std::clamp(value / width, 0.0F, 1.0F);
  return t * t * (3.0F - 2.0F * t);
}

bool shade_reference(const HostNv12& frame, const CameraParams& camera, const ReferenceBasis& basis,
                     const ReferenceView& view, Float3 ray, float plane_aspect, float correction,
                     float blend_width, bool is_right, ReferencePixel& output) {
  const float denominator = dot(basis.normal, ray);
  if (std::abs(denominator) < 1.0e-6F) {
    return false;
  }
  const float distance = dot(subtract(basis.origin, view.eye), basis.normal) / denominator;
  if (distance <= 0.0F) {
    return false;
  }
  const Float3 hit = add(view.eye, multiply(ray, distance));
  const float depth = dot(subtract(hit, view.eye), view.forward);
  if (depth < kNear || depth > kFar) {
    return false;
  }
  const Float3 relative = subtract(hit, basis.origin);
  const float local_x = dot(relative, basis.right) / dot(basis.right, basis.right);
  const float local_y = dot(relative, basis.up) / dot(basis.up, basis.up);
  if (local_x < -0.5F || local_x > 0.5F || local_y < -0.5F / plane_aspect ||
      local_y > 0.5F / plane_aspect) {
    return false;
  }
  const float plane_u = local_x * 2.0F + 0.5F;
  const float plane_v = 0.5F - local_y * plane_aspect * 2.0F;
  const float fx = static_cast<float>(camera.fx / camera.width);
  const float fy = static_cast<float>(camera.fy / camera.height);
  const float cx = static_cast<float>(camera.cx / camera.width);
  const float cy = static_cast<float>(camera.cy / camera.height);
  const float x = (plane_u - cx) / fx;
  const float y = (plane_v - cy) / fy;
  const float radius = std::sqrt(x * x + y * y);
  const float theta = std::atan(radius);
  const float theta2 = theta * theta;
  const float theta4 = theta2 * theta2;
  const float theta6 = theta4 * theta2;
  const float theta8 = theta4 * theta4;
  const float full_theta_d =
      theta *
      (1.0F + static_cast<float>(camera.d[0]) * theta2 + static_cast<float>(camera.d[1]) * theta4 +
       static_cast<float>(camera.d[2]) * theta6 + static_cast<float>(camera.d[3]) * theta8);
  const float theta_d = theta + (full_theta_d - theta) * correction;
  const float scale = radius > 0.0F ? theta_d / radius : 1.0F;
  const float sample_u = fx * x * scale + cx;
  const float sample_v = fy * y * scale + cy;
  if (sample_u < 0.0F || sample_u > 1.0F || sample_v < 0.0F || sample_v > 1.0F) {
    return false;
  }
  output = sample_reference_nv12(frame, sample_u, sample_v);
  output.a = is_right && blend_width > 0.0F ? smoothstep(blend_width, plane_u) : 1.0F;
  return true;
}

void composite(ReferencePixel source, ReferencePixel& destination) {
  const float inverse_alpha = 1.0F - source.a;
  destination.r = source.r * source.a + destination.r * inverse_alpha;
  destination.g = source.g * source.a + destination.g * inverse_alpha;
  destination.b = source.b * source.a + destination.b * inverse_alpha;
  destination.a = source.a + destination.a * inverse_alpha;
}

std::uint8_t quantize(float value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0.0F, 255.0F) + 0.5F);
}

struct ReferenceCoverage {
  std::size_t left_hits = 0;
  std::size_t right_hits = 0;
  std::size_t fractional_blends = 0;
  std::size_t uncovered = 0;
};

std::vector<std::uint8_t> reference_render(const CudaStitchRendererConfig& config,
                                           const CudaStitchViewport& viewport, const HostNv12& left,
                                           const HostNv12& right,
                                           ReferenceCoverage* coverage = nullptr) {
  const float plane_aspect = static_cast<float>(config.calibration.left.width) /
                             static_cast<float>(config.calibration.left.height);
  const auto scene =
      SceneGeometry::from_layout_with_aspect(config.calibration.layout, plane_aspect);
  const auto left_basis = reference_plane_basis(CameraId::Left, scene);
  const auto right_basis = reference_plane_basis(CameraId::Right, scene);
  const auto view = reference_view(config, viewport);
  std::vector<std::uint8_t> result(static_cast<std::size_t>(config.output_width) *
                                   config.output_height * 4U);
  for (std::uint32_t pixel_y = 0; pixel_y < config.output_height; ++pixel_y) {
    for (std::uint32_t pixel_x = 0; pixel_x < config.output_width; ++pixel_x) {
      const float ndc_x =
          ((static_cast<float>(pixel_x) + 0.5F) / static_cast<float>(config.output_width)) * 2.0F -
          1.0F;
      const float ndc_y =
          1.0F -
          ((static_cast<float>(pixel_y) + 0.5F) / static_cast<float>(config.output_height)) * 2.0F;
      Float3 ray =
          add(view.forward, multiply(view.right, ndc_x * view.tan_half_fov * view.output_aspect));
      ray = normalize(add(ray, multiply(view.up, ndc_y * view.tan_half_fov)));
      ReferencePixel destination;
      ReferencePixel source;
      const bool left_hit = shade_reference(left, config.calibration.left, left_basis, view, ray,
                                            plane_aspect, config.calibration.lens_correction_amount,
                                            config.calibration.blend_width, false, source);
      if (left_hit) {
        composite(source, destination);
      }
      const bool right_hit = shade_reference(
          right, config.calibration.right, right_basis, view, ray, plane_aspect,
          config.calibration.lens_correction_amount, config.calibration.blend_width, true, source);
      const float right_alpha = right_hit ? source.a : 0.0F;
      if (right_hit) {
        composite(source, destination);
      }
      if (coverage != nullptr) {
        coverage->left_hits += left_hit ? 1U : 0U;
        coverage->right_hits += right_hit ? 1U : 0U;
        coverage->fractional_blends +=
            right_hit && right_alpha > 0.0F && right_alpha < 1.0F ? 1U : 0U;
        coverage->uncovered += !left_hit && !right_hit ? 1U : 0U;
      }
      const auto offset = (static_cast<std::size_t>(pixel_y) * config.output_width + pixel_x) * 4U;
      result[offset] = quantize(destination.r);
      result[offset + 1U] = quantize(destination.g);
      result[offset + 2U] = quantize(destination.b);
      result[offset + 3U] = quantize(destination.a * 255.0F);
    }
  }
  return result;
}

MatchCalibration parity_calibration();

std::optional<std::pair<float, float>> reference_project_ray(Float3 ray, const CameraParams& camera,
                                                             const ReferenceBasis& basis,
                                                             const ReferenceView& view,
                                                             float plane_aspect) {
  const float denominator = dot(basis.normal, ray);
  if (std::abs(denominator) < 1.0e-6F) {
    return std::nullopt;
  }
  const float distance = dot(subtract(basis.origin, view.eye), basis.normal) / denominator;
  if (distance <= 0.0F) {
    return std::nullopt;
  }
  const Float3 hit = add(view.eye, multiply(ray, distance));
  const Float3 relative = subtract(hit, basis.origin);
  const float local_x = dot(relative, basis.right) / dot(basis.right, basis.right);
  const float local_y = dot(relative, basis.up) / dot(basis.up, basis.up);
  if (local_x < -0.5F || local_x > 0.5F || local_y < -0.5F / plane_aspect ||
      local_y > 0.5F / plane_aspect) {
    return std::nullopt;
  }
  const auto sample =
      forward_fisheye(local_x * 2.0F + 0.5F, 0.5F - local_y * plane_aspect * 2.0F, camera);
  if (sample.first < 0.0 || sample.first > 1.0 || sample.second < 0.0 || sample.second > 1.0) {
    return std::nullopt;
  }
  return std::pair{static_cast<float>(sample.first), static_cast<float>(sample.second)};
}

void representative_rays_match_projection_api() {
  CudaStitchRendererConfig config{
      .calibration = parity_calibration(), .output_width = 48, .output_height = 28};
  config.calibration.rig_tilt = 0.0;
  config.calibration.rig_roll = 0.0;
  config.calibration.lens_correction_amount = 1.0F;
  const CudaStitchViewport viewport{.yaw = 0.08F, .pitch = -0.03F, .fov_degrees = 76.0F};
  const float plane_aspect = static_cast<float>(config.calibration.left.width) /
                             static_cast<float>(config.calibration.left.height);
  const auto scene =
      SceneGeometry::from_layout_with_aspect(config.calibration.layout, plane_aspect);
  const auto view = reference_view(config, viewport);
  const VirtualCamera virtual_camera(scene.camera_position);
  std::size_t comparisons = 0;
  for (const auto [pixel_x, pixel_y] :
       {std::pair{24U, 14U}, std::pair{12U, 14U}, std::pair{36U, 14U}, std::pair{24U, 8U},
        std::pair{24U, 20U}}) {
    const float ndc_x =
        ((static_cast<float>(pixel_x) + 0.5F) / static_cast<float>(config.output_width)) * 2.0F -
        1.0F;
    const float ndc_y =
        1.0F -
        ((static_cast<float>(pixel_y) + 0.5F) / static_cast<float>(config.output_height)) * 2.0F;
    Float3 ray =
        add(view.forward, multiply(view.right, ndc_x * view.tan_half_fov * view.output_aspect));
    ray = normalize(add(ray, multiply(view.up, ndc_y * view.tan_half_fov)));
    const auto panorama = virtual_camera.direction_to_yaw_pitch({ray.x, ray.y, ray.z});
    for (const auto camera_id : {CameraId::Left, CameraId::Right}) {
      const auto& camera =
          camera_id == CameraId::Left ? config.calibration.left : config.calibration.right;
      const auto basis = reference_plane_basis(camera_id, scene);
      const auto direct = reference_project_ray(ray, camera, basis, view, plane_aspect);
      const auto projected =
          panorama_to_camera(panorama.yaw, panorama.pitch, camera_id, config.calibration, scene);
      expect_true(direct.has_value() == projected.has_value(),
                  "representative ray coverage matches panorama_to_camera");
      if (direct.has_value() && projected.has_value()) {
        expect_true(std::abs(direct->first - projected->first) < 2.0e-5F &&
                        std::abs(direct->second - projected->second) < 2.0e-5F,
                    "representative ray UV matches panorama_to_camera");
        ++comparisons;
      }
    }
  }
  expect_true(comparisons >= 3, "representative projection rays cover both scene planes");
}

MatchCalibration parity_calibration() {
  MatchCalibration calibration;
  calibration.left = {.width = 64,
                      .height = 36,
                      .fx = 31.2,
                      .fy = 30.8,
                      .cx = 31.4,
                      .cy = 17.7,
                      .d = {0.032, 0.011, -0.007, 0.002}};
  calibration.right = {.width = 64,
                       .height = 36,
                       .fx = 30.7,
                       .fy = 31.1,
                       .cx = 32.2,
                       .cy = 18.1,
                       .d = {0.027, 0.015, -0.006, 0.001}};
  calibration.layout.camera_axis_offset = 0.25;
  calibration.layout.intersect = 0.48;
  calibration.layout.x_ty = 0.006;
  calibration.layout.x_rz = 0.009;
  calibration.layout.z_rx = -0.005;
  calibration.layout.x_rx = 0.004;
  calibration.layout.z_rz = -0.003;
  calibration.rig_tilt = 0.025;
  calibration.rig_roll = -0.018;
  calibration.blend_width = 0.18F;
  calibration.lens_correction_amount = 0.8F;
  return calibration;
}

HostNv12 patterned_frame(std::uint32_t seed, YuvColorMatrix matrix, YuvColorRange range,
                         bool flip) {
  HostNv12 frame{.width = 64,
                 .height = 36,
                 .y = std::vector<std::uint8_t>(64U * 36U),
                 .uv = std::vector<std::uint8_t>(64U * 18U),
                 .matrix = matrix,
                 .range = range,
                 .flip_180 = flip};
  for (std::uint32_t y = 0; y < frame.height; ++y) {
    for (std::uint32_t x = 0; x < frame.width; ++x) {
      const std::uint32_t value = seed + x * 5U + y * 9U + ((x * y) % 29U);
      frame.y[static_cast<std::size_t>(y) * frame.width + x] =
          range == YuvColorRange::Limited ? static_cast<std::uint8_t>(16U + value % 220U)
                                          : static_cast<std::uint8_t>(value % 256U);
    }
  }
  for (std::uint32_t y = 0; y < frame.height / 2U; ++y) {
    for (std::uint32_t x = 0; x < frame.width / 2U; ++x) {
      const auto offset = (static_cast<std::size_t>(y) * (frame.width / 2U) + x) * 2U;
      frame.uv[offset] = static_cast<std::uint8_t>(32U + (seed + x * 7U + y * 3U) % 192U);
      frame.uv[offset + 1U] =
          static_cast<std::uint8_t>(32U + (seed * 3U + x * 2U + y * 11U) % 192U);
    }
  }
  return frame;
}

struct DeviceFrameStorage {
  CudaPitchedAllocation y;
  CudaPitchedAllocation uv;
};

DeviceFrameStorage upload_frame(const CudaBackend& backend, const HostNv12& frame) {
  const std::size_t y_pitch = static_cast<std::size_t>(frame.width) + 19U;
  const std::size_t uv_pitch = static_cast<std::size_t>(frame.width) + 23U;
  DeviceFrameStorage storage{
      .y = {.buffer = backend.allocate(y_pitch * frame.height), .pitch = y_pitch},
      .uv = {.buffer = backend.allocate(uv_pitch * (frame.height / 2U)), .pitch = uv_pitch}};
  backend.copy_host_to_device_2d({.src = frame.y.data(),
                                  .src_pitch = frame.width,
                                  .dst = storage.y.buffer.ptr(),
                                  .dst_pitch = storage.y.pitch,
                                  .width_bytes = frame.width,
                                  .height = frame.height});
  backend.copy_host_to_device_2d({.src = frame.uv.data(),
                                  .src_pitch = frame.width,
                                  .dst = storage.uv.buffer.ptr(),
                                  .dst_pitch = storage.uv.pitch,
                                  .width_bytes = frame.width,
                                  .height = frame.height / 2U});
  return storage;
}

CudaNv12FrameView frame_view(const DeviceFrameStorage& storage, const HostNv12& frame,
                             CudaContextId context) {
  return CudaNv12FrameView(
      CudaPitchedPlaneView(storage.y.buffer.ptr(), storage.y.buffer.size(), storage.y.pitch,
                           frame.width, frame.height, context),
      CudaPitchedPlaneView(storage.uv.buffer.ptr(), storage.uv.buffer.size(), storage.uv.pitch,
                           frame.width, frame.height / 2U, context),
      frame.width, frame.height, frame.matrix, frame.range);
}

void compare_pixels(const std::vector<std::uint8_t>& actual,
                    const std::vector<std::uint8_t>& expected, std::string_view label) {
  expect_true(actual.size() == expected.size(), std::string(label) + " output size");
  if (actual.size() != expected.size()) {
    return;
  }
  std::size_t mismatches = 0;
  int maximum_error = 0;
  std::size_t first_mismatch = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const int error = std::abs(static_cast<int>(actual[index]) - static_cast<int>(expected[index]));
    maximum_error = std::max(maximum_error, error);
    if (error > 2) {
      if (mismatches == 0) {
        first_mismatch = index;
      }
      ++mismatches;
    }
  }
  if (mismatches != 0) {
    std::cerr << "FAIL: " << label << " pixel parity mismatches=" << mismatches
              << " max_error=" << maximum_error << " first_byte=" << first_mismatch
              << " expected=" << static_cast<int>(expected[first_mismatch])
              << " actual=" << static_cast<int>(actual[first_mismatch]) << '\n';
    ++failures;
  }
}

void run_hardware_parity_case(const CudaBackend& backend, const CudaStereoStitchRenderer& renderer,
                              const CudaStitchRendererConfig& config,
                              const CudaStitchViewport& viewport, const HostNv12& left,
                              const HostNv12& right) {
  const auto context = renderer.context_id();
  auto left_storage = upload_frame(backend, left);
  auto right_storage = upload_frame(backend, right);
  const std::size_t output_row_bytes = static_cast<std::size_t>(config.output_width) * 4U;
  const std::size_t output_pitch = output_row_bytes + 29U;
  CudaPitchedAllocation output{.buffer = backend.allocate(output_pitch * config.output_height),
                               .pitch = output_pitch};
  const auto left_view = frame_view(left_storage, left, context);
  const auto right_view = frame_view(right_storage, right, context);
  const CudaRgbaFrameView output_view(
      CudaPitchedPlaneView(output.buffer.ptr(), output.buffer.size(), output.pitch,
                           static_cast<std::size_t>(config.output_width) * 4U, config.output_height,
                           context),
      config.output_width, config.output_height);
  renderer.render(left_view, right_view, output_view, viewport);
  std::vector<std::uint8_t> actual(static_cast<std::size_t>(config.output_width) *
                                   config.output_height * 4U);
  backend.copy_device_to_host_2d({.dst = actual.data(),
                                  .dst_pitch = static_cast<std::size_t>(config.output_width) * 4U,
                                  .src = output.buffer.ptr(),
                                  .src_pitch = output.pitch,
                                  .width_bytes = static_cast<std::size_t>(config.output_width) * 4U,
                                  .height = config.output_height});
  ReferenceCoverage coverage;
  const auto expected = reference_render(config, viewport, left, right, &coverage);
  expect_true(coverage.left_hits != 0, "parity fixture exercises left-plane coverage");
  expect_true(coverage.right_hits != 0, "parity fixture exercises right-plane coverage");
  expect_true(coverage.fractional_blends != 0,
              "parity fixture exercises fractional right-plane smoothstep blending");
  expect_true(coverage.uncovered != 0, "parity fixture exercises uncovered output pixels");
  const auto center = (static_cast<std::size_t>(config.output_height / 2U) * config.output_width +
                       config.output_width / 2U) *
                      4U;
  expect_true(expected[center + 3U] != 0U, "parity fixture center ray intersects the scene");
  compare_pixels(actual, expected, "CUDA stitch");
}

void hardware_parity_if_available() {
  if (address_sanitizer_build() && !require_cuda()) {
    std::cout << "SKIP: CUDA stitch parity disabled under ASan\n";
    return;
  }
  const auto cuda_error = CudaBackend::availability_error();
  const auto nvrtc_error = NvrtcCompiler::availability_error();
  if (!cuda_error.empty() || !nvrtc_error.empty()) {
    const auto diagnostic =
        "CUDA=" + (cuda_error.empty() ? std::string("available") : cuda_error) +
        " NVRTC=" + (nvrtc_error.empty() ? std::string("available") : nvrtc_error);
    if (require_cuda()) {
      throw std::runtime_error("required CUDA stitch parity unavailable: " + diagnostic);
    }
    std::cout << "SKIP: CUDA stitch parity unavailable: " << diagnostic << '\n';
    return;
  }

  CudaStitchRendererConfig config{
      .calibration = parity_calibration(), .output_width = 48, .output_height = 28};
  auto backend = CudaBackend::create();
  auto renderer = CudaStereoStitchRenderer::create(config, backend, NvrtcCompiler::create());
  const CudaStitchViewport first_viewport{.yaw = 0.07F, .pitch = -0.04F, .fov_degrees = 82.0F};
  run_hardware_parity_case(backend, renderer, config, first_viewport,
                           patterned_frame(7, YuvColorMatrix::Bt601, YuvColorRange::Limited, false),
                           patterned_frame(19, YuvColorMatrix::Bt2020, YuvColorRange::Full, false));
  const CudaStitchViewport second_viewport{.yaw = -0.11F,
                                           .pitch = 0.06F,
                                           .fov_degrees = 68.0F,
                                           .flip_left_180 = true,
                                           .flip_right_180 = true};
  run_hardware_parity_case(
      backend, renderer, config, second_viewport,
      patterned_frame(31, YuvColorMatrix::Bt709, YuvColorRange::Full, true),
      patterned_frame(47, YuvColorMatrix::Bt709, YuvColorRange::Limited, true));
  std::cout << "hardware CUDA stitch parity executed\n";
}

} // namespace

int main() {
  run_case("representative projection rays", representative_rays_match_projection_api);
  run_case("hardware CUDA stitch parity", hardware_parity_if_available);
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "all tests passed\n";
  return EXIT_SUCCESS;
}
