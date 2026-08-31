#include "reco/core/cuda_stitch_renderer.hpp"

#include "reco/core/projection.hpp"
#include "reco/core/render_layout.hpp"
#include "reco/core/video_format.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace reco::core {
namespace {

constexpr std::string_view kKernelName = "reco_stitch_nv12_rgba";
constexpr std::uint32_t kBlockWidth = 16;
constexpr std::uint32_t kBlockHeight = 16;
constexpr float kPi = 3.14159265358979323846F;

struct alignas(8) KernelPlaneParams {
  std::uint64_t y_ptr = 0;
  std::uint64_t uv_ptr = 0;
  std::uint64_t y_pitch = 0;
  std::uint64_t uv_pitch = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t flip_180 = 0;
  std::uint32_t reserved = 0;
  std::array<float, 4> intrinsics{};
  std::array<float, 4> distortion{};
  std::array<float, 8> color{};
  std::array<float, 4> origin{};
  std::array<float, 4> right{};
  std::array<float, 4> up{};
  std::array<float, 4> normal{};
};

struct alignas(8) KernelViewParams {
  std::array<float, 4> eye{};
  std::array<float, 4> forward{};
  std::array<float, 4> right{};
  std::array<float, 4> up{};
  std::array<float, 4> projection{};
  std::array<float, 4> blend_clip{};
};

static_assert(sizeof(KernelPlaneParams) == 176);
static_assert(offsetof(KernelPlaneParams, intrinsics) == 48);
static_assert(offsetof(KernelPlaneParams, color) == 80);
static_assert(offsetof(KernelPlaneParams, origin) == 112);
static_assert(sizeof(KernelViewParams) == 96);

constexpr std::string_view kCudaSource = R"cuda(
struct __align__(8) PlaneParams {
  unsigned long long y_ptr;
  unsigned long long uv_ptr;
  unsigned long long y_pitch;
  unsigned long long uv_pitch;
  unsigned int width;
  unsigned int height;
  unsigned int flip_180;
  unsigned int reserved;
  float intrinsics[4];
  float distortion[4];
  float color[8];
  float origin[4];
  float right[4];
  float up[4];
  float normal[4];
};

struct __align__(8) ViewParams {
  float eye[4];
  float forward[4];
  float right[4];
  float up[4];
  float projection[4];
  float blend_clip[4];
};

struct Vec3 { float x; float y; float z; };
struct Pixel { float r; float g; float b; float a; };

static_assert(sizeof(PlaneParams) == 176, "PlaneParams ABI mismatch");
static_assert(sizeof(ViewParams) == 96, "ViewParams ABI mismatch");

__device__ __forceinline__ Vec3 make_vec3(const float* value) {
  return {value[0], value[1], value[2]};
}

__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ Vec3 sub(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ Vec3 mul(Vec3 value, float scalar) {
  return {value.x * scalar, value.y * scalar, value.z * scalar};
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 normalize(Vec3 value) {
  const float length = sqrtf(dot(value, value));
  return length > 1.0e-20f ? mul(value, 1.0f / length) : Vec3{0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ int clamp_index(int value, int maximum) {
  return value < 0 ? 0 : (value > maximum ? maximum : value);
}

__device__ __forceinline__ float bilinear_y(const PlaneParams& plane, float u, float v) {
  const float sample_x = u * static_cast<float>(plane.width) - 0.5f;
  const float sample_y = v * static_cast<float>(plane.height) - 0.5f;
  const int floor_x = static_cast<int>(floorf(sample_x));
  const int floor_y = static_cast<int>(floorf(sample_y));
  const float tx = sample_x - static_cast<float>(floor_x);
  const float ty = sample_y - static_cast<float>(floor_y);
  const int x0 = clamp_index(floor_x, static_cast<int>(plane.width) - 1);
  const int x1 = clamp_index(floor_x + 1, static_cast<int>(plane.width) - 1);
  const int y0 = clamp_index(floor_y, static_cast<int>(plane.height) - 1);
  const int y1 = clamp_index(floor_y + 1, static_cast<int>(plane.height) - 1);
  const auto* base = reinterpret_cast<const unsigned char*>(plane.y_ptr);
  const float p00 = static_cast<float>(base[static_cast<unsigned long long>(y0) * plane.y_pitch + x0]);
  const float p10 = static_cast<float>(base[static_cast<unsigned long long>(y0) * plane.y_pitch + x1]);
  const float p01 = static_cast<float>(base[static_cast<unsigned long long>(y1) * plane.y_pitch + x0]);
  const float p11 = static_cast<float>(base[static_cast<unsigned long long>(y1) * plane.y_pitch + x1]);
  const float top = p00 + (p10 - p00) * tx;
  const float bottom = p01 + (p11 - p01) * tx;
  return top + (bottom - top) * ty;
}

__device__ __forceinline__ void bilinear_uv(const PlaneParams& plane, float u, float v,
                                             float* out_u, float* out_v) {
  const int width = static_cast<int>(plane.width / 2U);
  const int height = static_cast<int>(plane.height / 2U);
  const float sample_x = u * static_cast<float>(width) - 0.5f;
  const float sample_y = v * static_cast<float>(height) - 0.5f;
  const int floor_x = static_cast<int>(floorf(sample_x));
  const int floor_y = static_cast<int>(floorf(sample_y));
  const float tx = sample_x - static_cast<float>(floor_x);
  const float ty = sample_y - static_cast<float>(floor_y);
  const int x0 = clamp_index(floor_x, width - 1);
  const int x1 = clamp_index(floor_x + 1, width - 1);
  const int y0 = clamp_index(floor_y, height - 1);
  const int y1 = clamp_index(floor_y + 1, height - 1);
  const auto* base = reinterpret_cast<const unsigned char*>(plane.uv_ptr);
  const unsigned long long row0 = static_cast<unsigned long long>(y0) * plane.uv_pitch;
  const unsigned long long row1 = static_cast<unsigned long long>(y1) * plane.uv_pitch;
  const int byte_x0 = x0 * 2;
  const int byte_x1 = x1 * 2;
  const float u00 = static_cast<float>(base[row0 + byte_x0]);
  const float u10 = static_cast<float>(base[row0 + byte_x1]);
  const float u01 = static_cast<float>(base[row1 + byte_x0]);
  const float u11 = static_cast<float>(base[row1 + byte_x1]);
  const float v00 = static_cast<float>(base[row0 + byte_x0 + 1]);
  const float v10 = static_cast<float>(base[row0 + byte_x1 + 1]);
  const float v01 = static_cast<float>(base[row1 + byte_x0 + 1]);
  const float v11 = static_cast<float>(base[row1 + byte_x1 + 1]);
  const float u_top = u00 + (u10 - u00) * tx;
  const float u_bottom = u01 + (u11 - u01) * tx;
  const float v_top = v00 + (v10 - v00) * tx;
  const float v_bottom = v01 + (v11 - v01) * tx;
  *out_u = u_top + (u_bottom - u_top) * ty;
  *out_v = v_top + (v_bottom - v_top) * ty;
}

__device__ __forceinline__ float clamp_byte(float value) {
  return fminf(255.0f, fmaxf(0.0f, value));
}

__device__ __forceinline__ Pixel sample_nv12(const PlaneParams& plane, float u, float v) {
  if (plane.flip_180 != 0U) {
    u = 1.0f - u;
    v = 1.0f - v;
  }
  const float y_raw = bilinear_y(plane, u, v);
  float u_raw = 0.0f;
  float v_raw = 0.0f;
  bilinear_uv(plane, u, v, &u_raw, &v_raw);
  const float y = (y_raw + plane.color[0]) * plane.color[1];
  const float cb = u_raw - plane.color[2];
  const float cr = v_raw - plane.color[2];
  return {
      clamp_byte(y + plane.color[3] * cr),
      clamp_byte(y + plane.color[4] * cb + plane.color[5] * cr),
      clamp_byte(y + plane.color[6] * cb),
      1.0f,
  };
}

__device__ __forceinline__ float smoothstep(float edge0, float edge1, float value) {
  const float t = fminf(1.0f, fmaxf(0.0f, (value - edge0) / (edge1 - edge0)));
  return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ bool shade_plane(const PlaneParams& plane, const ViewParams& view,
                                             Vec3 ray, bool is_right, Pixel* pixel) {
  const Vec3 eye = make_vec3(view.eye);
  const Vec3 origin = make_vec3(plane.origin);
  const Vec3 normal = make_vec3(plane.normal);
  const float denominator = dot(normal, ray);
  if (fabsf(denominator) < 1.0e-6f) {
    return false;
  }
  const float distance = dot(sub(origin, eye), normal) / denominator;
  if (distance <= 0.0f) {
    return false;
  }
  const Vec3 hit = add(eye, mul(ray, distance));
  const float view_depth = dot(sub(hit, eye), make_vec3(view.forward));
  if (view_depth < view.blend_clip[1] || view_depth > view.blend_clip[2]) {
    return false;
  }
  const Vec3 relative = sub(hit, origin);
  const Vec3 plane_right = make_vec3(plane.right);
  const Vec3 plane_up = make_vec3(plane.up);
  const float local_x = dot(relative, plane_right) / dot(plane_right, plane_right);
  const float local_y = dot(relative, plane_up) / dot(plane_up, plane_up);
  const float plane_aspect = view.projection[2];
  if (local_x < -0.5f || local_x > 0.5f ||
      local_y < -0.5f / plane_aspect || local_y > 0.5f / plane_aspect) {
    return false;
  }

  const float plane_u = local_x * 2.0f + 0.5f;
  const float plane_v = 0.5f - local_y * plane_aspect * 2.0f;
  const float x = (plane_u - plane.intrinsics[2]) / plane.intrinsics[0];
  const float y = (plane_v - plane.intrinsics[3]) / plane.intrinsics[1];
  const float radius = sqrtf(x * x + y * y);
  const float theta = atanf(radius);
  const float theta2 = theta * theta;
  const float theta4 = theta2 * theta2;
  const float theta6 = theta4 * theta2;
  const float theta8 = theta4 * theta4;
  const float full_theta_d = theta * (1.0f + plane.distortion[0] * theta2 +
      plane.distortion[1] * theta4 + plane.distortion[2] * theta6 +
      plane.distortion[3] * theta8);
  const float theta_d = theta + (full_theta_d - theta) * view.projection[3];
  const float scale = radius > 0.0f ? theta_d / radius : 1.0f;
  const float sample_u = plane.intrinsics[0] * x * scale + plane.intrinsics[2];
  const float sample_v = plane.intrinsics[1] * y * scale + plane.intrinsics[3];
  if (sample_u < 0.0f || sample_u > 1.0f || sample_v < 0.0f || sample_v > 1.0f) {
    return false;
  }

  *pixel = sample_nv12(plane, sample_u, sample_v);
  pixel->a = is_right && view.blend_clip[0] > 0.0f
                 ? smoothstep(0.0f, view.blend_clip[0], plane_u)
                 : 1.0f;
  return true;
}

__device__ __forceinline__ void composite(Pixel source, Pixel* destination) {
  const float inverse_alpha = 1.0f - source.a;
  destination->r = source.r * source.a + destination->r * inverse_alpha;
  destination->g = source.g * source.a + destination->g * inverse_alpha;
  destination->b = source.b * source.a + destination->b * inverse_alpha;
  destination->a = source.a + destination->a * inverse_alpha;
}

__device__ __forceinline__ unsigned char quantize(float value) {
  return static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, value)) + 0.5f);
}

extern "C" __global__ void reco_stitch_nv12_rgba(
    PlaneParams left, PlaneParams right, ViewParams view,
    unsigned long long output_ptr, unsigned long long output_pitch,
    unsigned int output_width, unsigned int output_height) {
  const unsigned int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (pixel_x >= output_width || pixel_y >= output_height) {
    return;
  }

  const float ndc_x = ((static_cast<float>(pixel_x) + 0.5f) /
                       static_cast<float>(output_width)) * 2.0f - 1.0f;
  const float ndc_y = 1.0f - ((static_cast<float>(pixel_y) + 0.5f) /
                              static_cast<float>(output_height)) * 2.0f;
  Vec3 ray = add(make_vec3(view.forward),
                 mul(make_vec3(view.right), ndc_x * view.projection[0] * view.projection[1]));
  ray = normalize(add(ray, mul(make_vec3(view.up), ndc_y * view.projection[0])));

  Pixel destination{0.0f, 0.0f, 0.0f, 0.0f};
  Pixel source{};
  if (shade_plane(left, view, ray, false, &source)) {
    composite(source, &destination);
  }
  if (shade_plane(right, view, ray, true, &source)) {
    composite(source, &destination);
  }

  auto* output = reinterpret_cast<unsigned char*>(output_ptr) +
                 static_cast<unsigned long long>(pixel_y) * output_pitch + pixel_x * 4U;
  output[0] = quantize(destination.r);
  output[1] = quantize(destination.g);
  output[2] = quantize(destination.b);
  output[3] = quantize(destination.a * 255.0f);
}
)cuda";

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
  if (!(length > 1.0e-12F) || !std::isfinite(length)) {
    throw std::invalid_argument("CUDA stitch geometry contains a degenerate vector");
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

std::array<float, 4> padded(Float3 value) { return {value.x, value.y, value.z, 0.0F}; }

Float3 from_scene(Vec3 value) { return {value.x, value.y, value.z}; }

struct PlaneBasis {
  Float3 origin;
  Float3 right;
  Float3 up;
  Float3 normal;
};

PlaneBasis plane_basis(CameraId camera, const SceneGeometry& scene) {
  const Float3 origin = from_scene(plane_uv_to_world({0.5, 0.5}, camera, scene));
  const Float3 right_point = from_scene(plane_uv_to_world({2.5, 0.5}, camera, scene));
  const Float3 up_point =
      from_scene(plane_uv_to_world({0.5, 0.5 - 2.0 * scene.plane_aspect}, camera, scene));
  const Float3 right = subtract(right_point, origin);
  const Float3 up = subtract(up_point, origin);
  return {.origin = origin, .right = right, .up = up, .normal = normalize(cross(right, up))};
}

void validate_config(const CudaStitchRendererConfig& config, const CudaBackend& backend) {
  if (const auto error = config.calibration.validate(); !error.empty()) {
    throw std::invalid_argument("invalid CUDA stitch calibration: " + error);
  }
  if (config.output_width == 0 || config.output_height == 0) {
    throw std::invalid_argument("CUDA stitch output dimensions must be non-zero");
  }
  if (config.output_width > kMaxCalibrationDimension ||
      config.output_height > kMaxCalibrationDimension) {
    throw std::invalid_argument("CUDA stitch output dimensions exceed the supported maximum");
  }
  const auto& left = config.calibration.left;
  const auto& right = config.calibration.right;
  if ((left.width % 2U) != 0U || (left.height % 2U) != 0U || (right.width % 2U) != 0U ||
      (right.height % 2U) != 0U) {
    throw std::invalid_argument("CUDA stitch calibration dimensions must be even for NV12");
  }
  if (config.calibration.blend_width < 0.0F || config.calibration.blend_width > 1.0F) {
    throw std::invalid_argument("CUDA stitch blend width must be in [0, 1]");
  }
  if (config.calibration.lens_correction_amount < 0.0F ||
      config.calibration.lens_correction_amount > 1.0F) {
    throw std::invalid_argument("CUDA stitch lens correction amount must be in [0, 1]");
  }
  if (config.device_ordinal < 0 || config.device_ordinal >= backend.device_count()) {
    throw std::invalid_argument("CUDA stitch device ordinal is out of range");
  }
}

KernelPlaneParams make_plane_params(const CudaNv12FrameView& frame, const CameraParams& camera,
                                    const PlaneBasis& basis, bool flip_180) {
  KernelPlaneParams params;
  params.y_ptr = frame.y_plane().ptr();
  params.uv_ptr = frame.uv_plane().ptr();
  params.y_pitch = frame.y_plane().pitch_bytes();
  params.uv_pitch = frame.uv_plane().pitch_bytes();
  params.width = frame.width();
  params.height = frame.height();
  params.flip_180 = flip_180 ? 1U : 0U;
  const float width = static_cast<float>(camera.width);
  const float height = static_cast<float>(camera.height);
  params.intrinsics = {
      static_cast<float>(camera.fx) / width, static_cast<float>(camera.fy) / height,
      static_cast<float>(camera.cx) / width, static_cast<float>(camera.cy) / height};
  params.distortion = {static_cast<float>(camera.d[0]), static_cast<float>(camera.d[1]),
                       static_cast<float>(camera.d[2]), static_cast<float>(camera.d[3])};
  const auto coefficients = yuv_to_rgb_coefficients(frame.color_matrix(), frame.color_range());
  params.color = {coefficients.y_offset,
                  coefficients.y_scale,
                  frame.color_range() == YuvColorRange::Full ? 127.5F : 128.0F,
                  coefficients.red_from_v,
                  coefficients.green_from_u,
                  coefficients.green_from_v,
                  coefficients.blue_from_u,
                  0.0F};
  params.origin = padded(basis.origin);
  params.right = padded(basis.right);
  params.up = padded(basis.up);
  params.normal = padded(basis.normal);
  return params;
}

KernelViewParams make_view_params(const CudaStitchRendererConfig& config,
                                  const SceneGeometry& scene, const CudaStitchViewport& viewport) {
  if (!std::isfinite(viewport.yaw) || !std::isfinite(viewport.pitch)) {
    throw std::invalid_argument("CUDA stitch viewport yaw and pitch must be finite");
  }
  if (!std::isfinite(viewport.fov_degrees) || viewport.fov_degrees <= 1.0F ||
      viewport.fov_degrees >= 179.0F) {
    throw std::invalid_argument("CUDA stitch vertical FOV must be in (1, 179) degrees");
  }

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
  const Float3 screen_up = normalize(cross(screen_right, forward));

  KernelViewParams params;
  params.eye = padded(eye);
  params.forward = padded(normalize(forward));
  params.right = padded(screen_right);
  params.up = padded(screen_up);
  params.projection = {
      std::tan(viewport.fov_degrees * kPi / 360.0F),
      static_cast<float>(config.output_width) / static_cast<float>(config.output_height),
      scene.plane_aspect,
      config.calibration.lens_correction_amount,
  };
  params.blend_clip = {config.calibration.blend_width, kNearPlane, kFarPlane, 0.0F};
  return params;
}

bool spans_overlap(const CudaPitchedPlaneView& lhs, const CudaPitchedPlaneView& rhs) {
  const auto lhs_last = lhs.ptr() + lhs.address_span_bytes() - 1U;
  const auto rhs_last = rhs.ptr() + rhs.address_span_bytes() - 1U;
  return lhs.ptr() <= rhs_last && rhs.ptr() <= lhs_last;
}

void validate_frame_provenance(const CudaNv12FrameView& frame, const CameraParams& camera,
                               CudaContextId context_id, int device_ordinal,
                               std::string_view label) {
  if (frame.width() != camera.width || frame.height() != camera.height) {
    throw std::invalid_argument("CUDA stitch " + std::string(label) +
                                " frame dimensions do not match calibration");
  }
  if (frame.context_id() != context_id) {
    throw std::invalid_argument("CUDA stitch " + std::string(label) +
                                " frame belongs to a different CUDA context");
  }
  if (frame.device_ordinal() != device_ordinal) {
    throw std::invalid_argument("CUDA stitch " + std::string(label) +
                                " frame belongs to a different CUDA device");
  }
}

} // namespace

struct CudaStereoStitchRenderer::Impl {
  Impl(CudaStitchRendererConfig config_in, CudaContextId context_id_in, CudaKernel kernel_in)
      : config(std::move(config_in)),
        scene(SceneGeometry::from_layout_with_aspect(
            config.calibration.layout, static_cast<float>(config.calibration.left.width) /
                                           static_cast<float>(config.calibration.left.height))),
        left_basis(plane_basis(CameraId::Left, scene)),
        right_basis(plane_basis(CameraId::Right, scene)), context_id(context_id_in),
        kernel(std::move(kernel_in)) {}

  CudaStitchRendererConfig config;
  SceneGeometry scene;
  PlaneBasis left_basis;
  PlaneBasis right_basis;
  CudaContextId context_id = 0;
  CudaKernel kernel;
  mutable std::mutex render_mutex;
};

CudaStereoStitchRenderer CudaStereoStitchRenderer::create(CudaStitchRendererConfig config) {
  return create(std::move(config), CudaBackend::create(), NvrtcCompiler::create());
}

CudaStereoStitchRenderer CudaStereoStitchRenderer::create(CudaStitchRendererConfig config,
                                                          CudaBackend backend,
                                                          NvrtcCompiler compiler) {
  validate_config(config, backend);
  const auto capability = backend.compute_capability(config.device_ordinal);
  const auto device_architecture = capability.major * 10 + capability.minor;
  const auto target_architecture = compiler.select_architecture(device_architecture);
  const auto context_id = backend.primary_context_id(config.device_ordinal);
  NvrtcCompileOptions options;
  options.values = {"--std=c++17",
                    "--gpu-architecture=compute_" + std::to_string(target_architecture)};
  const auto compiled = compiler.compile(kCudaSource, "reco_cuda_stitch_renderer.cu", options);
  auto module = backend.load_module_from_ptx(compiled.ptx, config.device_ordinal);
  auto kernel = module.load_kernel(kKernelName);
  return CudaStereoStitchRenderer(
      std::make_unique<Impl>(std::move(config), context_id, std::move(kernel)));
}

CudaStereoStitchRenderer::CudaStereoStitchRenderer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CudaStereoStitchRenderer::CudaStereoStitchRenderer(CudaStereoStitchRenderer&&) noexcept = default;
CudaStereoStitchRenderer&
CudaStereoStitchRenderer::operator=(CudaStereoStitchRenderer&&) noexcept = default;
CudaStereoStitchRenderer::~CudaStereoStitchRenderer() = default;

void CudaStereoStitchRenderer::render(const CudaNv12FrameView& left, const CudaNv12FrameView& right,
                                      const CudaRgbaFrameView& output,
                                      const CudaStitchViewport& viewport) const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA stitch renderer");
  }
  const auto& state = *impl_;
  std::lock_guard<std::mutex> lock(state.render_mutex);
  validate_frame_provenance(left, state.config.calibration.left, state.context_id,
                            state.config.device_ordinal, "left");
  validate_frame_provenance(right, state.config.calibration.right, state.context_id,
                            state.config.device_ordinal, "right");
  if (output.width() != state.config.output_width ||
      output.height() != state.config.output_height) {
    throw std::invalid_argument("CUDA stitch RGBA output dimensions do not match the renderer");
  }
  if (output.context_id() != state.context_id) {
    throw std::invalid_argument("CUDA stitch RGBA output belongs to a different CUDA context");
  }
  if (output.device_ordinal() != state.config.device_ordinal) {
    throw std::invalid_argument("CUDA stitch RGBA output belongs to a different CUDA device");
  }
  for (const auto* input_plane :
       {&left.y_plane(), &left.uv_plane(), &right.y_plane(), &right.uv_plane()}) {
    if (spans_overlap(*input_plane, output.plane())) {
      throw std::invalid_argument("CUDA stitch input and RGBA output memory must not overlap");
    }
  }

  auto left_params = make_plane_params(left, state.config.calibration.left, state.left_basis,
                                       viewport.flip_left_180);
  auto right_params = make_plane_params(right, state.config.calibration.right, state.right_basis,
                                        viewport.flip_right_180);
  auto view_params = make_view_params(state.config, state.scene, viewport);
  auto output_ptr = output.plane().ptr();
  auto output_pitch = static_cast<std::uint64_t>(output.plane().pitch_bytes());
  auto output_width = output.width();
  auto output_height = output.height();
  std::array<void*, 7> arguments{&left_params,  &right_params, &view_params,  &output_ptr,
                                 &output_pitch, &output_width, &output_height};
  const auto grid_x = (output_width + kBlockWidth - 1U) / kBlockWidth;
  const auto grid_y = (output_height + kBlockHeight - 1U) / kBlockHeight;
  state.kernel.launch({.grid = {grid_x, grid_y, 1},
                       .block = {kBlockWidth, kBlockHeight, 1},
                       .shared_memory_bytes = 0},
                      std::span<void*>(arguments));
  state.kernel.synchronize();
}

CudaContextId CudaStereoStitchRenderer::context_id() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA stitch renderer");
  }
  return impl_->context_id;
}

int CudaStereoStitchRenderer::device_ordinal() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA stitch renderer");
  }
  return impl_->config.device_ordinal;
}

std::uint32_t CudaStereoStitchRenderer::output_width() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA stitch renderer");
  }
  return impl_->config.output_width;
}

std::uint32_t CudaStereoStitchRenderer::output_height() const {
  if (!impl_) {
    throw std::logic_error("cannot use a moved-from CUDA stitch renderer");
  }
  return impl_->config.output_height;
}

} // namespace reco::core
