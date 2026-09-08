#include "reco/core/cuda_stitch_renderer.hpp"
#include "reco/core/path.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace reco::core;

namespace {

static_assert(!std::is_copy_constructible_v<CudaStereoStitchRenderer>);
static_assert(!std::is_copy_assignable_v<CudaStereoStitchRenderer>);
static_assert(std::is_nothrow_move_constructible_v<CudaStereoStitchRenderer>);
static_assert(std::is_nothrow_move_assignable_v<CudaStereoStitchRenderer>);

int failures = 0;

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

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Actual, typename Expected>
void expect_eq(const Actual& actual, const Expected& expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_near(float actual, float expected, float tolerance, std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
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

template <typename Function>
void expect_invalid_argument(Function&& function, std::string_view fragment,
                             std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_logic_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::logic_error& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

std::filesystem::path resolve_runfile(std::string_view path) {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
  const auto logical = std::string(workspace) + "/" + std::string(path);
  const auto resolved = std::filesystem::path(runfiles->Rlocation(logical));
  if (resolved.empty() || !std::filesystem::is_regular_file(resolved)) {
    throw std::runtime_error(std::string(path) + " runfile not found");
  }
  return resolved;
}

class DynamicControl {
public:
  explicit DynamicControl(const std::filesystem::path& path) {
#if defined(_WIN32)
    handle_ = LoadLibraryW(path.c_str());
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw std::runtime_error("failed to load fake runtime controls");
    }
  }

  DynamicControl(const DynamicControl&) = delete;
  DynamicControl& operator=(const DynamicControl&) = delete;

  ~DynamicControl() {
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
  }

  template <typename Function> Function symbol(const char* name) const {
#if defined(_WIN32)
    auto* value = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* value = dlsym(handle_, name);
#endif
    if (value == nullptr) {
      throw std::runtime_error(std::string("missing fake runtime control ") + name);
    }
    return reinterpret_cast<Function>(value);
  }

private:
  void* handle_ = nullptr;
};

void set_hardware_vmm_access(const CudaBackend& backend, CudaDevicePtr pointer, std::size_t bytes,
                             unsigned int flags) {
  struct Location {
    unsigned int type;
    int id;
  };
  struct AccessDescription {
    Location location;
    unsigned int flags;
  };
#if defined(_WIN32)
  DynamicControl driver("nvcuda.dll");
#else
  DynamicControl driver("libcuda.so.1");
#endif
  const auto set_access =
      driver.symbol<int (*)(CudaDevicePtr, std::size_t, const AccessDescription*, std::size_t)>(
          "cuMemSetAccess");
  backend.ensure_primary_context(0);
  const AccessDescription access{{1U, 0}, flags};
  if (set_access(pointer, bytes, &access, 1) != 0) {
    throw std::runtime_error("failed to set CUDA VMM access in hardware test");
  }
}

struct FakeCudaControl {
  explicit FakeCudaControl(const std::filesystem::path& path) : library(path) {
    reset_fn = library.symbol<void (*)()>("recoFakeCudaStitchReset");
    launch_count_fn = library.symbol<int (*)()>("recoFakeCudaStitchLaunchCount");
    synchronize_count_fn = library.symbol<int (*)()>("recoFakeCudaStitchSynchronizeCount");
    launch_sequence_fn = library.symbol<int (*)()>("recoFakeCudaStitchLaunchSequence");
    synchronize_sequence_fn = library.symbol<int (*)()>("recoFakeCudaStitchSynchronizeSequence");
    pointer_attribute_count_fn =
        library.symbol<int (*)()>("recoFakeCudaStitchPointerAttributeCount");
    memory_access_count_fn = library.symbol<int (*)()>("recoFakeCudaStitchMemoryAccessCount");
    retain_count_fn = library.symbol<int (*)()>("recoFakeCudaStitchRetainCount");
    release_count_fn = library.symbol<int (*)()>("recoFakeCudaStitchReleaseCount");
    set_current_context_fn =
        library.symbol<void (*)(std::uintptr_t)>("recoFakeCudaStitchSetCurrentContext");
    current_context_fn = library.symbol<std::uintptr_t (*)()>("recoFakeCudaStitchCurrentContext");
    fail_module_load_fn = library.symbol<void (*)(int)>("recoFakeCudaStitchFailModuleLoad");
    captured_u64_fn = library.symbol<std::uint64_t (*)(int)>("recoFakeCudaStitchCapturedU64");
    captured_u32_fn = library.symbol<std::uint32_t (*)(int)>("recoFakeCudaStitchCapturedU32");
    captured_float_fn = library.symbol<float (*)(int)>("recoFakeCudaStitchCapturedFloat");
  }

  void reset() const { reset_fn(); }
  int launch_count() const { return launch_count_fn(); }
  int synchronize_count() const { return synchronize_count_fn(); }
  int launch_sequence() const { return launch_sequence_fn(); }
  int synchronize_sequence() const { return synchronize_sequence_fn(); }
  int pointer_attribute_count() const { return pointer_attribute_count_fn(); }
  int memory_access_count() const { return memory_access_count_fn(); }
  int retain_count() const { return retain_count_fn(); }
  int release_count() const { return release_count_fn(); }
  void set_current_context(std::uintptr_t context) const { set_current_context_fn(context); }
  std::uintptr_t current_context() const { return current_context_fn(); }
  void fail_module_load(bool fail) const { fail_module_load_fn(fail ? 1 : 0); }
  std::uint64_t captured_u64(int index) const { return captured_u64_fn(index); }
  std::uint32_t captured_u32(int index) const { return captured_u32_fn(index); }
  float captured_float(int index) const { return captured_float_fn(index); }

  DynamicControl library;
  void (*reset_fn)() = nullptr;
  int (*launch_count_fn)() = nullptr;
  int (*synchronize_count_fn)() = nullptr;
  int (*launch_sequence_fn)() = nullptr;
  int (*synchronize_sequence_fn)() = nullptr;
  int (*pointer_attribute_count_fn)() = nullptr;
  int (*memory_access_count_fn)() = nullptr;
  int (*retain_count_fn)() = nullptr;
  int (*release_count_fn)() = nullptr;
  void (*set_current_context_fn)(std::uintptr_t) = nullptr;
  std::uintptr_t (*current_context_fn)() = nullptr;
  void (*fail_module_load_fn)(int) = nullptr;
  std::uint64_t (*captured_u64_fn)(int) = nullptr;
  std::uint32_t (*captured_u32_fn)(int) = nullptr;
  float (*captured_float_fn)(int) = nullptr;
};

struct FakeNvrtcControl {
  explicit FakeNvrtcControl(const std::filesystem::path& path) : library(path) {
    reset_fn = library.symbol<void (*)()>("recoFakeNvrtcReset");
    create_count_fn = library.symbol<int (*)()>("recoFakeNvrtcCreateCount");
    destroy_count_fn = library.symbol<int (*)()>("recoFakeNvrtcDestroyCount");
    last_architecture_fn = library.symbol<int (*)()>("recoFakeNvrtcLastArchitecture");
  }

  void reset() const { reset_fn(); }
  int create_count() const { return create_count_fn(); }
  int destroy_count() const { return destroy_count_fn(); }
  int last_architecture() const { return last_architecture_fn(); }

  DynamicControl library;
  void (*reset_fn)() = nullptr;
  int (*create_count_fn)() = nullptr;
  int (*destroy_count_fn)() = nullptr;
  int (*last_architecture_fn)() = nullptr;
};

MatchCalibration calibration() {
  MatchCalibration value;
  value.left = {.width = 4, .height = 2, .fx = 2.0, .fy = 2.0, .cx = 2.0, .cy = 1.0};
  value.right = value.left;
  value.layout.camera_axis_offset = 0.25;
  value.layout.intersect = 0.5;
  value.blend_width = 0.2F;
  value.lens_correction_amount = 1.0F;
  return value;
}

CudaStitchRendererConfig config() {
  return {.calibration = calibration(), .output_width = 4, .output_height = 2};
}

CudaNv12FrameView nv12_frame(CudaDevicePtr base, CudaContextId context,
                             YuvColorMatrix matrix = YuvColorMatrix::Bt709,
                             YuvColorRange range = YuvColorRange::Limited, int device = 0,
                             std::uint32_t width = 4, std::uint32_t height = 2) {
  const std::size_t y_pitch = 8;
  const std::size_t uv_pitch = 8;
  return CudaNv12FrameView(
      CudaPitchedPlaneView(base, y_pitch * height, y_pitch, width, height, context, device),
      CudaPitchedPlaneView(base + 0x1000U, uv_pitch * (height / 2U), uv_pitch, width, height / 2U,
                           context, device),
      width, height, matrix, range);
}

CudaRgbaFrameView rgba_frame(CudaDevicePtr base, CudaContextId context, int device = 0,
                             std::uint32_t width = 4, std::uint32_t height = 2) {
  const std::size_t row_bytes = static_cast<std::size_t>(width) * 4U;
  const std::size_t pitch = row_bytes + 16U;
  return CudaRgbaFrameView(
      CudaPitchedPlaneView(base, pitch * height, pitch, row_bytes, height, context, device), width,
      height);
}

CudaStereoStitchRenderer create_renderer(const CudaStitchRendererConfig& renderer_config,
                                         const std::filesystem::path& cuda_runtime,
                                         const std::filesystem::path& nvrtc_runtime) {
  return CudaStereoStitchRenderer::create(renderer_config, CudaBackend::load(cuda_runtime.string()),
                                          NvrtcCompiler::load(nvrtc_runtime.string()));
}

void exact_path_backend_retains_primary_context_for_process_lifetime(
    const std::filesystem::path& cuda_runtime, const FakeCudaControl& cuda_control) {
  expect_eq(cuda_control.retain_count(), 0, "fake CUDA context starts unretained");
  expect_eq(cuda_control.release_count(), 0, "fake CUDA context starts unreleased");
  CudaContextId context = 0;
  {
    auto backend = CudaBackend::load(cuda_runtime.string());
    context = backend.primary_context_id();
    expect_true(context != 0, "exact-path backend establishes a primary context");
  }
  expect_eq(cuda_control.retain_count(), 1,
            "exact-path backend cache keeps the primary context retained");
  expect_eq(cuda_control.release_count(), 0,
            "exact-path backend destruction does not invalidate a current context");
  auto reused = CudaBackend::load(cuda_runtime.string());
  expect_eq(reused.primary_context_id(), context,
            "repeated exact-path load reuses the retained primary context");
  expect_eq(cuda_control.retain_count(), 1, "repeated exact-path load does not retain again");
}

void exact_cuda_runtime_path_preserves_windows_unicode(const std::filesystem::path& cuda_runtime) {
#if defined(_WIN32)
  const auto directory = std::filesystem::temp_directory_path() /
                         std::filesystem::path(std::u8string(u8"reco-cuda-\u663e\u793a"));
  std::filesystem::create_directories(directory);
  const auto copied_runtime = directory / cuda_runtime.filename();
  std::filesystem::copy_file(cuda_runtime, copied_runtime,
                             std::filesystem::copy_options::overwrite_existing);
  auto backend = CudaBackend::load(path_to_utf8(copied_runtime));
  expect_eq(backend.device_count(), 1, "Unicode CUDA runtime path loads exact driver");
#else
  (void)cuda_runtime;
#endif
}

void compiles_once_and_synchronizes_each_render(const std::filesystem::path& cuda_runtime,
                                                const std::filesystem::path& nvrtc_runtime,
                                                const FakeCudaControl& cuda_control,
                                                const FakeNvrtcControl& nvrtc_control) {
  cuda_control.reset();
  nvrtc_control.reset();
  auto renderer = create_renderer(config(), cuda_runtime, nvrtc_runtime);
  expect_eq(nvrtc_control.create_count(), 1, "renderer performs one NVRTC compilation");
  expect_eq(nvrtc_control.destroy_count(), 1, "renderer releases the NVRTC program");
  expect_eq(nvrtc_control.last_architecture(), 86,
            "renderer selects the highest NVRTC target supported by the device");
  expect_eq(renderer.device_ordinal(), 0, "renderer device ordinal");
  expect_eq(renderer.output_width(), 4U, "renderer output width");
  expect_eq(renderer.output_height(), 2U, "renderer output height");

  const auto context = renderer.context_id();
  const auto left = nv12_frame(0x10000U, context, YuvColorMatrix::Bt601, YuvColorRange::Limited);
  const auto right = nv12_frame(0x30000U, context, YuvColorMatrix::Bt2020, YuvColorRange::Full);
  const auto output = rgba_frame(0x50000U, context);
  renderer.render(left, right, output,
                  {.yaw = 0.1F,
                   .pitch = -0.2F,
                   .fov_degrees = 80.0F,
                   .flip_left_180 = true,
                   .flip_right_180 = true});
  expect_eq(cuda_control.captured_u32(2), 1U, "left 180-degree flip propagated");
  expect_eq(cuda_control.captured_u32(5), 1U, "right 180-degree flip propagated");
  renderer.render(left, right, output);

  expect_eq(nvrtc_control.create_count(), 1, "render never recompiles the kernel");
  expect_eq(cuda_control.launch_count(), 2, "one fused launch per render");
  expect_eq(cuda_control.synchronize_count(), 2, "each render synchronizes before return");
  expect_eq(cuda_control.pointer_attribute_count(), 66,
            "each render validates legacy pointers and the VMM output provenance");
  expect_eq(cuda_control.memory_access_count(), 2,
            "each render validates the complete VMM output access range");
  expect_true(cuda_control.launch_sequence() < cuda_control.synchronize_sequence(),
              "launch precedes synchronization");
  expect_eq(cuda_control.captured_u64(0), left.y_plane().ptr(), "left Y pointer propagated");
  expect_eq(cuda_control.captured_u64(1), left.uv_plane().ptr(), "left UV pointer propagated");
  expect_eq(cuda_control.captured_u64(2), left.y_plane().pitch_bytes(), "left Y pitch propagated");
  expect_eq(cuda_control.captured_u64(4), right.y_plane().ptr(), "right Y pointer propagated");
  expect_eq(cuda_control.captured_u64(6), output.plane().ptr(), "output pointer propagated");
  expect_eq(cuda_control.captured_u64(7), output.plane().pitch_bytes(), "output pitch propagated");
  expect_eq(cuda_control.captured_u32(6), 1U, "bounded launch grid X");
  expect_eq(cuda_control.captured_u32(7), 1U, "bounded launch grid Y");
  expect_eq(cuda_control.captured_u32(9), 16U, "launch block X");
  expect_eq(cuda_control.captured_u32(10), 16U, "launch block Y");
  expect_near(cuda_control.captured_float(0), -16.0F, 1.0e-6F, "limited-range luma offset");
  expect_near(cuda_control.captured_float(1), 255.0F / 219.0F, 1.0e-6F, "limited-range luma scale");
  expect_near(cuda_control.captured_float(2), 128.0F, 1.0e-6F, "limited-range chroma center");
  expect_near(cuda_control.captured_float(13), 127.5F, 1.0e-6F, "full-range chroma center");
}

void enforces_device_access_permissions(const std::filesystem::path& cuda_runtime,
                                        const std::filesystem::path& nvrtc_runtime,
                                        const FakeCudaControl& cuda_control) {
  auto renderer = create_renderer(config(), cuda_runtime, nvrtc_runtime);
  const auto context = renderer.context_id();
  const auto right = nv12_frame(0x30000U, context);
  const auto output = rgba_frame(0x50000U, context);
  cuda_control.reset();

  renderer.render(nv12_frame(0xF0000U, context), right, output);
  expect_eq(cuda_control.launch_count(), 1, "read-only input is accepted for device reads");
  expect_invalid_argument([&] { renderer.render(nv12_frame(0xE0000U, context), right, output); },
                          "reads", "inaccessible input is rejected");
  expect_invalid_argument(
      [&] { renderer.render(nv12_frame(0x10000U, context), right, rgba_frame(0xF0000U, context)); },
      "writes", "read-only output is rejected");
  expect_eq(cuda_control.launch_count(), 1, "denied access never launches the kernel");
}

void rejects_split_vmm_permissions(const std::filesystem::path& cuda_runtime,
                                   const FakeCudaControl& cuda_control) {
  auto backend = CudaBackend::load(cuda_runtime.string());
  cuda_control.reset();
  expect_invalid_argument(
      [&] { (void)backend.retain_device_span(0x100000U, 0x2000U, CudaSpanAccess::Read); }, "reads",
      "inaccessible second VMM region is rejected");
  expect_eq(cuda_control.memory_access_count(), 2,
            "whole-span VMM validation reaches the inaccessible second region");
}

void rejects_physical_vmm_aliases(const std::filesystem::path& cuda_runtime,
                                  const std::filesystem::path& nvrtc_runtime,
                                  const FakeCudaControl& cuda_control) {
  auto renderer = create_renderer(config(), cuda_runtime, nvrtc_runtime);
  const auto context = renderer.context_id();
  cuda_control.reset();
  expect_invalid_argument(
      [&] {
        renderer.render(nv12_frame(0x60000U, context), nv12_frame(0x30000U, context),
                        rgba_frame(0x50000U, context));
      },
      "overlap", "distinct virtual mappings of one physical allocation are rejected");
  expect_eq(cuda_control.launch_count(), 0, "physical alias rejection prevents kernel launch");

  auto backend = CudaBackend::load(cuda_runtime.string());
  const auto known = backend.retain_device_span(0x50000U, 48U, CudaSpanAccess::ReadWrite);
  const auto distinct = backend.retain_device_span(0x100000U, 0x1000U, CudaSpanAccess::Read);
  expect_true(!known.aliases(distinct), "distinct physical VMM block identities do not alias");
  const auto unknown = backend.retain_device_span(0x110000U, 48U, CudaSpanAccess::Read);
  expect_true(known.aliases(unknown), "unknown VMM physical identity fails closed as an alias");
}

void retained_span_validation_avoids_render_time_driver_queries(
    const std::filesystem::path& cuda_runtime, const std::filesystem::path& nvrtc_runtime,
    const FakeCudaControl& cuda_control) {
  auto backend = CudaBackend::load(cuda_runtime.string());
  auto renderer = CudaStereoStitchRenderer::create(config(), backend,
                                                   NvrtcCompiler::load(nvrtc_runtime.string()));
  const CudaNv12FrameView left(
      CudaPitchedPlaneView(backend.retain_device_span(0x10000U, 12U, CudaSpanAccess::Read), 8, 4,
                           2),
      CudaPitchedPlaneView(backend.retain_device_span(0x11000U, 4U, CudaSpanAccess::Read), 8, 4, 1),
      4, 2, YuvColorMatrix::Bt709, YuvColorRange::Limited);
  const CudaNv12FrameView right(
      CudaPitchedPlaneView(backend.retain_device_span(0x30000U, 12U, CudaSpanAccess::Read), 8, 4,
                           2),
      CudaPitchedPlaneView(backend.retain_device_span(0x31000U, 4U, CudaSpanAccess::Read), 8, 4, 1),
      4, 2, YuvColorMatrix::Bt709, YuvColorRange::Limited);
  const CudaRgbaFrameView output(
      CudaPitchedPlaneView(backend.retain_device_span(0x50000U, 48U, CudaSpanAccess::ReadWrite), 32,
                           16, 2),
      4, 2);

  cuda_control.reset();
  renderer.render(left, right, output);
  expect_eq(cuda_control.pointer_attribute_count(), 0,
            "retained frame validation avoids render-time pointer queries");
  expect_eq(cuda_control.memory_access_count(), 0,
            "retained frame validation avoids render-time VMM granule queries");
  expect_eq(cuda_control.launch_count(), 1, "retained validated spans render normally");
}

void renderer_construction_restores_callers_context(const std::filesystem::path& cuda_runtime,
                                                    const std::filesystem::path& nvrtc_runtime,
                                                    const FakeCudaControl& cuda_control) {
  constexpr std::uintptr_t kCallerContext = 0xD00D0001U;
  cuda_control.set_current_context(kCallerContext);
  {
    auto renderer = create_renderer(config(), cuda_runtime, nvrtc_runtime);
    expect_eq(cuda_control.current_context(), kCallerContext,
              "successful renderer construction restores caller context");
  }
  expect_eq(cuda_control.current_context(), kCallerContext,
            "renderer destruction preserves caller context");

  auto backend = CudaBackend::load(cuda_runtime.string());
  cuda_control.fail_module_load(true);
  try {
    (void)backend.load_module_from_ptx("fake ptx");
    expect_true(false, "failed module load throws");
  } catch (const std::runtime_error&) {
  }
  cuda_control.fail_module_load(false);
  expect_eq(cuda_control.current_context(), kCallerContext,
            "failed module load restores caller context");
  cuda_control.set_current_context(0);
}

void rejects_invalid_configuration_before_compilation(const std::filesystem::path& cuda_runtime,
                                                      const std::filesystem::path& nvrtc_runtime,
                                                      const FakeNvrtcControl& nvrtc_control) {
  auto expect_config_error = [&](CudaStitchRendererConfig invalid, std::string_view fragment,
                                 std::string_view label) {
    nvrtc_control.reset();
    expect_invalid_argument([&] { (void)create_renderer(invalid, cuda_runtime, nvrtc_runtime); },
                            fragment, label);
    expect_eq(nvrtc_control.create_count(), 0, std::string(label) + " does not compile");
  };

  auto invalid = config();
  invalid.output_width = 0;
  expect_config_error(invalid, "non-zero", "zero output width");
  invalid = config();
  invalid.output_height = kMaxCalibrationDimension + 1U;
  expect_config_error(invalid, "maximum", "oversized output height");
  invalid = config();
  invalid.calibration.left.width = 3;
  expect_config_error(invalid, "even", "odd NV12 calibration width");
  invalid = config();
  invalid.calibration.blend_width = 1.1F;
  expect_config_error(invalid, "blend width", "invalid blend width");
  invalid = config();
  invalid.calibration.lens_correction_amount = -0.1F;
  expect_config_error(invalid, "lens correction", "invalid correction amount");
  invalid = config();
  invalid.device_ordinal = 1;
  expect_config_error(invalid, "out of range", "invalid device ordinal");
}

void rejects_unsafe_frame_contracts(const std::filesystem::path& cuda_runtime,
                                    const std::filesystem::path& nvrtc_runtime,
                                    const FakeCudaControl& cuda_control) {
  auto renderer = create_renderer(config(), cuda_runtime, nvrtc_runtime);
  const auto context = renderer.context_id();
  const auto left = nv12_frame(0x10000U, context);
  const auto right = nv12_frame(0x30000U, context);
  const auto output = rgba_frame(0x50000U, context);
  cuda_control.reset();

  expect_invalid_argument(
      [&] {
        renderer.render(
            nv12_frame(0x70000U, context, YuvColorMatrix::Bt709, YuvColorRange::Limited, 0, 6, 2),
            right, output);
      },
      "dimensions", "left dimensions must match calibration");
  expect_invalid_argument(
      [&] { renderer.render(nv12_frame(0x70000U, context + 1U), right, output); }, "context",
      "left context must match renderer");
  expect_invalid_argument(
      [&] {
        renderer.render(
            nv12_frame(0x70000U, context, YuvColorMatrix::Bt709, YuvColorRange::Limited, 1), right,
            output);
      },
      "device", "left device must match renderer");
  expect_invalid_argument(
      [&] { renderer.render(left, right, rgba_frame(0x70000U, context, 0, 6, 2)); }, "dimensions",
      "output dimensions must match renderer");
  expect_invalid_argument([&] { renderer.render(left, right, rgba_frame(0x70000U, context + 1U)); },
                          "context", "output context must match renderer");
  expect_invalid_argument([&] { renderer.render(left, right, rgba_frame(0x10000U, context)); },
                          "overlap", "output must not overlap an input plane");
  expect_invalid_argument([&] { renderer.render(left, right, output, {.fov_degrees = 1.0F}); },
                          "FOV", "minimum FOV is rejected");
  expect_invalid_argument(
      [&] {
        renderer.render(left, right, output, {.yaw = std::numeric_limits<float>::quiet_NaN()});
      },
      "finite", "non-finite yaw is rejected");
  expect_invalid_argument([&] { renderer.render(nv12_frame(0x80000U, context), right, output); },
                          "device memory", "host pointer is rejected");
  expect_invalid_argument([&] { renderer.render(nv12_frame(0x8F000U, context), right, output); },
                          "rejected", "freed UV pointer is rejected");
  expect_invalid_argument([&] { renderer.render(left, nv12_frame(0xA0000U, context), output); },
                          "exceeds", "undersized allocation is rejected");
  expect_invalid_argument([&] { renderer.render(left, right, rgba_frame(0xB0000U, context)); },
                          "different CUDA context", "foreign-context output is rejected");
  expect_invalid_argument([&] { renderer.render(left, nv12_frame(0xBF000U, context), output); },
                          "different CUDA device", "foreign-device UV pointer is rejected");
  expect_invalid_argument([&] { renderer.render(left, right, rgba_frame(0xD0000U, context)); },
                          "not mapped", "unmapped output pointer is rejected");
  expect_eq(cuda_control.launch_count(), 0, "invalid render requests do not launch");
  expect_eq(cuda_control.synchronize_count(), 0, "invalid render requests do not synchronize");
}

void moved_from_renderer_is_diagnosed(const std::filesystem::path& cuda_runtime,
                                      const std::filesystem::path& nvrtc_runtime) {
  auto source = create_renderer(config(), cuda_runtime, nvrtc_runtime);
  const auto context = source.context_id();
  auto renderer = std::move(source);
  expect_eq(renderer.context_id(), context, "moved renderer retains context");
  expect_logic_error([&] { (void)source.context_id(); }, "moved-from", "moved-from context access");
  expect_logic_error(
      [&] {
        source.render(nv12_frame(0x10000U, context), nv12_frame(0x30000U, context),
                      rgba_frame(0x50000U, context));
      },
      "moved-from", "moved-from render access");
}

void hardware_kernel_smoke_if_available() {
  if (address_sanitizer_build() && !require_cuda()) {
    std::cout << "SKIP: hardware CUDA stitch test disabled under ASan\n";
    return;
  }
  const auto cuda_error = CudaBackend::availability_error();
  const auto nvrtc_error = NvrtcCompiler::availability_error();
  if (!cuda_error.empty() || !nvrtc_error.empty()) {
    const auto diagnostic =
        "CUDA=" + (cuda_error.empty() ? std::string("available") : cuda_error) +
        " NVRTC=" + (nvrtc_error.empty() ? std::string("available") : nvrtc_error);
    if (require_cuda()) {
      throw std::runtime_error("required hardware CUDA stitch test unavailable: " + diagnostic);
    }
    std::cout << "SKIP: hardware CUDA stitch test unavailable: " << diagnostic << '\n';
    return;
  }

  auto backend = CudaBackend::create();
  const auto context = backend.primary_context_id();
  auto left_y = backend.allocate_pitched(4, 2, 4);
  auto left_uv = backend.allocate_pitched(4, 1, 4);
  auto right_y = backend.allocate_pitched(4, 2, 4);
  auto right_uv = backend.allocate_pitched(4, 1, 4);
  auto output_storage = backend.allocate_shared_memory(32);
  auto inaccessible_storage = backend.allocate_shared_memory(32);
  auto read_only_storage = backend.allocate_shared_memory(32);
  const std::vector<std::uint8_t> left_y_host(8, 82);
  const std::vector<std::uint8_t> left_uv_host{90, 240, 90, 240};
  const std::vector<std::uint8_t> right_y_host(8, 145);
  const std::vector<std::uint8_t> right_uv_host{200, 40, 200, 40};
  backend.copy_host_to_device_2d({.src = left_y_host.data(),
                                  .src_pitch = 4,
                                  .dst = left_y.buffer.ptr(),
                                  .dst_pitch = left_y.pitch,
                                  .width_bytes = 4,
                                  .height = 2});
  backend.copy_host_to_device_2d({.src = left_uv_host.data(),
                                  .src_pitch = 4,
                                  .dst = left_uv.buffer.ptr(),
                                  .dst_pitch = left_uv.pitch,
                                  .width_bytes = 4,
                                  .height = 1});
  backend.copy_host_to_device_2d({.src = right_y_host.data(),
                                  .src_pitch = 4,
                                  .dst = right_y.buffer.ptr(),
                                  .dst_pitch = right_y.pitch,
                                  .width_bytes = 4,
                                  .height = 2});
  backend.copy_host_to_device_2d({.src = right_uv_host.data(),
                                  .src_pitch = 4,
                                  .dst = right_uv.buffer.ptr(),
                                  .dst_pitch = right_uv.pitch,
                                  .width_bytes = 4,
                                  .height = 1});
  const std::vector<std::uint8_t> initial_output(32, 0xCD);
  backend.copy_host_to_device_2d({.src = initial_output.data(),
                                  .src_pitch = 16,
                                  .dst = output_storage.ptr(),
                                  .dst_pitch = 16,
                                  .width_bytes = 16,
                                  .height = 2});

  const CudaNv12FrameView left(
      CudaPitchedPlaneView(left_y.buffer.ptr(), left_y.buffer.size(), left_y.pitch, 4, 2, context),
      CudaPitchedPlaneView(left_uv.buffer.ptr(), left_uv.buffer.size(), left_uv.pitch, 4, 1,
                           context),
      4, 2, YuvColorMatrix::Bt601, YuvColorRange::Limited);
  const CudaNv12FrameView right(CudaPitchedPlaneView(right_y.buffer.ptr(), right_y.buffer.size(),
                                                     right_y.pitch, 4, 2, context),
                                CudaPitchedPlaneView(right_uv.buffer.ptr(), right_uv.buffer.size(),
                                                     right_uv.pitch, 4, 1, context),
                                4, 2, YuvColorMatrix::Bt2020, YuvColorRange::Full);
  const CudaRgbaFrameView output(
      CudaPitchedPlaneView(output_storage.ptr(), output_storage.size(), 16, 16, 2, context), 4, 2);
  auto renderer = CudaStereoStitchRenderer::create(config(), backend, NvrtcCompiler::create());
  set_hardware_vmm_access(backend, inaccessible_storage.ptr(), inaccessible_storage.size(), 0U);
  expect_invalid_argument(
      [&] {
        backend.validate_device_span(inaccessible_storage.ptr(), inaccessible_storage.size(),
                                     CudaSpanAccess::Read);
      },
      "reads", "hardware PROT_NONE mapping is rejected");
  set_hardware_vmm_access(backend, read_only_storage.ptr(), read_only_storage.size(), 1U);
  backend.validate_device_span(read_only_storage.ptr(), read_only_storage.size(),
                               CudaSpanAccess::Read);
  const CudaRgbaFrameView read_only_output(
      CudaPitchedPlaneView(read_only_storage.ptr(), read_only_storage.size(), 16, 16, 2, context),
      4, 2);
  expect_invalid_argument([&] { renderer.render(left, right, read_only_output); }, "writes",
                          "hardware read-only output is rejected");
  renderer.render(left, right, output);
  std::vector<std::uint8_t> pixels(32, 0);
  backend.copy_device_to_host_2d({.dst = pixels.data(),
                                  .dst_pitch = 16,
                                  .src = output_storage.ptr(),
                                  .src_pitch = 16,
                                  .width_bytes = 16,
                                  .height = 2});
  expect_true(
      std::any_of(pixels.begin(), pixels.end(), [](std::uint8_t value) { return value != 0U; }),
      "hardware fused stitch produces output");
  expect_true(
      std::none_of(pixels.begin(), pixels.end(), [](std::uint8_t value) { return value == 0xCDU; }),
      "hardware fused stitch writes every output byte");
  std::cout << "hardware CUDA stitch kernel executed\n";
}

} // namespace

int main() {
  const auto cuda_runtime = resolve_runfile("cpp/tests/libreco_core_fake_cuda_stitch_driver.so");
  const auto nvrtc_runtime = resolve_runfile("cpp/tests/libreco_core_fake_nvrtc_runtime.so");
  FakeCudaControl cuda_control(cuda_runtime);
  FakeNvrtcControl nvrtc_control(nvrtc_runtime);

  run_case("exact CUDA runtime probe", [&] {
    expect_eq(CudaBackend::availability_error(cuda_runtime.string()), std::string(),
              "fake CUDA driver availability");
    expect_true(CudaBackend::availability_error(cuda_runtime.string() + ".missing")
                        .find("failed to load") != std::string::npos,
                "missing exact CUDA driver diagnostic");
    expect_invalid_argument([] { (void)CudaBackend::load({}); }, "path",
                            "empty exact CUDA driver path");
    expect_invalid_argument([] { (void)CudaBackend::load(std::string(32769U, 'x')); }, "exceeds",
                            "oversized exact CUDA driver path");
  });
  run_case("exact-path CUDA context lifetime", [&] {
    exact_path_backend_retains_primary_context_for_process_lifetime(cuda_runtime, cuda_control);
  });
  run_case("exact CUDA Windows Unicode path",
           [&] { exact_cuda_runtime_path_preserves_windows_unicode(cuda_runtime); });
  run_case("compile once and synchronize", [&] {
    compiles_once_and_synchronizes_each_render(cuda_runtime, nvrtc_runtime, cuda_control,
                                               nvrtc_control);
  });
  run_case("configuration validation", [&] {
    rejects_invalid_configuration_before_compilation(cuda_runtime, nvrtc_runtime, nvrtc_control);
  });
  run_case("frame contract validation",
           [&] { rejects_unsafe_frame_contracts(cuda_runtime, nvrtc_runtime, cuda_control); });
  run_case("device access validation",
           [&] { enforces_device_access_permissions(cuda_runtime, nvrtc_runtime, cuda_control); });
  run_case("split VMM access validation",
           [&] { rejects_split_vmm_permissions(cuda_runtime, cuda_control); });
  run_case("physical VMM alias validation",
           [&] { rejects_physical_vmm_aliases(cuda_runtime, nvrtc_runtime, cuda_control); });
  run_case("retained validation render path", [&] {
    retained_span_validation_avoids_render_time_driver_queries(cuda_runtime, nvrtc_runtime,
                                                               cuda_control);
  });
  run_case("renderer context restoration", [&] {
    renderer_construction_restores_callers_context(cuda_runtime, nvrtc_runtime, cuda_control);
  });
  run_case("moved-from renderer",
           [&] { moved_from_renderer_is_diagnosed(cuda_runtime, nvrtc_runtime); });
  run_case("hardware kernel smoke", hardware_kernel_smoke_if_available);

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "all tests passed\n";
  return EXIT_SUCCESS;
}
