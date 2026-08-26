#include "reco/io/detail/nvbufsurface_7_1.hpp"
#include "reco/io/detail/nvbufsurface_9_1.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/nvmm.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

using namespace reco::io;
namespace abi = reco::io::detail::nvbufsurface_9_1;
namespace abi7 = reco::io::detail::nvbufsurface_7_1;

namespace {

int failures = 0;

NvmmFrameInfo extract_info(const void* surface) {
  return extract_nvmm_frame_info(surface, NvbufSurfaceAbi::DeepStream9_1);
}

void expect_true(bool value, std::string_view message) {
  if (!value) {
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

template <typename Fn> void expect_nvmm_error(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const NvmmError&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::filesystem::path find_fake_runtime_runfile(std::string_view needle) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  if (runfiles == nullptr || *runfiles == '\0') {
    throw std::runtime_error("TEST_SRCDIR is not set");
  }
  for (const auto& entry : std::filesystem::recursive_directory_iterator(runfiles)) {
    const auto filename = entry.path().filename().string();
    if (filename.find(needle) != std::string::npos &&
        (ends_with(filename, ".so") || ends_with(filename, ".dylib") ||
         ends_with(filename, ".dll"))) {
      return entry.path();
    }
  }
  throw std::runtime_error("fake runtime runfile not found");
}

void set_runtime_path(const char* name, const std::filesystem::path& path) {
#if defined(_WIN32)
  _putenv_s(name, path.string().c_str());
#else
  setenv(name, path.string().c_str(), 1);
#endif
}

template <typename Params> Params make_params_as() {
  Params params;
  params.width = 1280;
  params.height = 720;
  params.pitch = 1280;
  params.color_format = abi::kColorNv12_709;
  params.layout = abi::kLayoutPitch;
  params.buffer_desc = 17;
  params.data_size = 1280 * 1080;
  params.data_ptr = nullptr;
  params.plane_params.num_planes = 2;
  params.plane_params.width[0] = 1280;
  params.plane_params.height[0] = 720;
  params.plane_params.pitch[0] = 1280;
  params.plane_params.psize[0] = 1280 * 720;
  params.plane_params.bytes_per_pix[0] = 1;
  params.plane_params.width[1] = 640;
  params.plane_params.height[1] = 360;
  params.plane_params.pitch[1] = 1280;
  params.plane_params.offset[1] = 1280 * 720;
  params.plane_params.psize[1] = 1280 * 360;
  params.plane_params.bytes_per_pix[1] = 2;
  return params;
}

abi::SurfaceParams make_params() { return make_params_as<abi::SurfaceParams>(); }

template <typename Surface, typename Params> Surface make_surface_as(Params& params) {
  Surface surface;
  surface.gpu_id = 0;
  surface.batch_size = 1;
  surface.num_filled = 1;
  surface.mem_type = abi::kMemSurfaceArray;
  surface.surface_list = &params;
  return surface;
}

abi::Surface make_surface(abi::SurfaceParams& params) {
  return make_surface_as<abi::Surface>(params);
}

void surface_array_mapping_retains_and_unmaps_owner() {
#if defined(__linux__)
  set_runtime_path("RECO_NVBUFSURFACE_DYLIB_PATH", find_fake_runtime_runfile("fake_nvbufsurface"));
  set_runtime_path("RECO_CUDA_DRIVER_DYLIB_PATH", find_fake_runtime_runfile("fake_cuda_driver"));
  expect_true(is_nvmm_cuda_interop_available(), "fake CUDA interop runtime available");
  expect_true(nvmm_cuda_interop_availability_error().empty(), "interop probe has no error");

  auto params = make_params();
  auto surface = make_surface(params);
  const auto info = extract_info(&surface);
  auto decoder_owner = std::make_shared<int>(9);
  std::weak_ptr<int> decoder_lifetime = decoder_owner;

  GpuDecodedFrame decoded{.nvmm = info,
                          .visible_width = info.width - 2,
                          .visible_height = info.height,
                          .owner = decoder_owner,
                          .frame_index = 3,
                          .pts_ns = 1'000'000,
                          .duration_ns = 33'333'333};
  auto mapped = map_gpu_decoded_frame_to_cuda(decoded);
  auto mapped_again = map_gpu_decoded_frame_to_cuda(decoded);
  expect_eq(mapped.y_ptr, 0x40000000U, "mapped Y pointer");
  expect_eq(mapped.uv_ptr, 0x40000000U + 1280U * 720U, "mapped UV pointer");
  expect_eq(mapped.y_pitch, 1280U, "mapped Y pitch");
  expect_eq(mapped.uv_pitch, 1280U, "mapped UV pitch");
  expect_eq(mapped.width, 1278U, "mapped CUDA view uses visible width");
  expect_eq(mapped.gpu_id, 0U, "mapped GPU id");
  expect_eq(mapped_again.y_ptr, mapped.y_ptr, "duplicate map shares CUDA mapping");
  expect_true(mapped.color_matrix == Nv12ColorMatrix::Bt709, "BT.709 metadata preserved");
  expect_true(mapped.color_range == Nv12ColorRange::Limited, "limited range preserved");
  expect_true(params.mapped_addr.cuda_ptr != nullptr, "runtime CUDA mapping remains live");
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(info, std::make_shared<int>(10)); },
                    "duplicate map with different owner rejected");

  decoder_owner.reset();
  decoded.owner.reset();
  expect_true(!decoder_lifetime.expired(), "mapping retains decoder buffer owner");
  mapped.owner.reset();
  expect_true(!decoder_lifetime.expired(), "duplicate mapping keeps decoder owner alive");
  expect_true(params.mapped_addr.cuda_ptr != nullptr,
              "duplicate mapping keeps runtime CUDA map live");
  mapped_again.owner.reset();
  expect_true(decoder_lifetime.expired(), "mapping releases decoder owner");
  expect_true(params.mapped_addr.cuda_ptr == nullptr, "mapping owner unmaps CUDA buffer");

  params = make_params();
  surface = make_surface(params);
  auto changed = extract_info(&surface);
  changed.uv_offset += 128;
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(changed, std::make_shared<int>(1)); },
                    "metadata mutation rejected before runtime mapping");

  params = make_params();
  params.buffer_desc = 99;
  surface = make_surface(params);
  const auto failing = extract_info(&surface);
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(failing, std::make_shared<int>(1)); },
                    "runtime mapping failure propagated");

  params = make_params();
  params.buffer_desc = 98;
  surface = make_surface(params);
  const auto partial_failure = extract_info(&surface);
  auto partial_owner = std::make_shared<int>(1);
  std::weak_ptr<int> partial_lifetime = partial_owner;
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(partial_failure, partial_owner); },
                    "partial CUDA mapping failure propagated");
  partial_owner.reset();
  expect_true(partial_lifetime.expired(), "clean partial CUDA rollback releases decoder owner");
  expect_true(params.mapped_addr.cuda_ptr == nullptr,
              "partial CUDA mapping failure rolls back runtime output");

  params = make_params();
  params.buffer_desc = 1;
  surface = make_surface(params);
  const auto incomplete = extract_info(&surface);
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(incomplete, std::make_shared<int>(1)); },
                    "post-map CUDA buffer validation failure propagated");
  expect_true(params.mapped_addr.cuda_ptr == nullptr,
              "post-map validation failure rolls back CUDA mapping");

  params = make_params();
  params.mapped_addr.cuda_ptr = reinterpret_cast<void*>(0xDEAD);
  surface = make_surface(params);
  const auto externally_mapped = extract_info(&surface);
  expect_nvmm_error(
      [&] { (void)map_nvmm_frame_to_cuda(externally_mapped, std::make_shared<int>(1)); },
      "external CUDA mapping rejected");
  expect_eq(params.mapped_addr.cuda_ptr, reinterpret_cast<void*>(0xDEAD),
            "external CUDA mapping remains owned by its creator");

  for (const std::uint64_t descriptor : {9U, 10U, 11U, 12U, 13U}) {
    params = make_params();
    params.buffer_desc = descriptor;
    surface = make_surface(params);
    const auto bad_pointer = extract_info(&surface);
    expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(bad_pointer, std::make_shared<int>(1)); },
                      "invalid CUDA pointer attributes rejected");
    expect_true(params.mapped_addr.cuda_ptr == nullptr,
                "invalid CUDA pointer mapping rolls back cleanly");
  }

  params = make_params();
  surface = make_surface(params);
  const auto valid = extract_info(&surface);
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(valid, {}); },
                    "missing decoder owner rejected");
#endif
}

void deepstream_7_1_egl_mapping_is_gpu_resident() {
#if defined(__linux__)
  auto params = make_params_as<abi7::SurfaceParams>();
  auto surface = make_surface_as<abi7::Surface>(params);
  const auto info = extract_nvmm_frame_info(&surface);
  auto mapped = map_nvmm_frame_to_cuda(info, std::make_shared<int>(2));
  expect_eq(mapped.y_ptr, 0x40000000U, "7.1 EGL CUDA Y pointer");
  expect_eq(mapped.uv_ptr, 0x40000000U + 1280U * 720U, "7.1 EGL CUDA UV pointer");
  expect_true(params.mapped_addr.egl_image != nullptr, "7.1 EGL image remains mapped");
  mapped.owner.reset();
  expect_true(params.mapped_addr.egl_image == nullptr, "7.1 EGL image unmaps with owner");

  params = make_params_as<abi7::SurfaceParams>();
  params.buffer_desc = 2;
  surface = make_surface_as<abi7::Surface>(params);
  const auto wrong_format = extract_nvmm_frame_info(&surface);
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(wrong_format, std::make_shared<int>(3)); },
                    "7.1 EGL mapping rejects a non-NV12 CUDA descriptor");
  expect_true(params.mapped_addr.egl_image == nullptr,
              "invalid EGL descriptor rolls back the EGL image mapping");

  params = make_params_as<abi7::SurfaceParams>();
  params.buffer_desc = 8;
  surface = make_surface_as<abi7::Surface>(params);
  const auto wrong_color = extract_nvmm_frame_info(&surface);
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(wrong_color, std::make_shared<int>(4)); },
                    "7.1 EGL mapping rejects mismatched CUDA colorimetry");
  expect_true(params.mapped_addr.egl_image == nullptr,
              "mismatched EGL colorimetry rolls back the EGL image mapping");

  params = make_params_as<abi7::SurfaceParams>();
  params.buffer_desc = 97;
  surface = make_surface_as<abi7::Surface>(params);
  const auto partial_failure = extract_nvmm_frame_info(&surface);
  auto partial_owner = std::make_shared<int>(4);
  std::weak_ptr<int> partial_lifetime = partial_owner;
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(partial_failure, partial_owner); },
                    "partial EGL mapping failure propagated");
  partial_owner.reset();
  expect_true(partial_lifetime.expired(), "clean partial EGL rollback releases decoder owner");
  expect_true(params.mapped_addr.egl_image == nullptr,
              "partial EGL mapping failure rolls back runtime output");

  params = make_params_as<abi7::SurfaceParams>();
  params.mapped_addr.egl_image = reinterpret_cast<void*>(0xBEEF);
  surface = make_surface_as<abi7::Surface>(params);
  const auto externally_mapped = extract_nvmm_frame_info(&surface);
  expect_nvmm_error(
      [&] { (void)map_nvmm_frame_to_cuda(externally_mapped, std::make_shared<int>(4)); },
      "external EGL mapping rejected");
  expect_eq(params.mapped_addr.egl_image, reinterpret_cast<void*>(0xBEEF),
            "external EGL mapping remains owned by its creator");
#endif
}

void cuda_device_mapping_retains_context_and_owner() {
#if defined(__linux__)
  auto params = make_params();
  params.data_ptr = reinterpret_cast<void*>(0x40000000);
  auto surface = make_surface(params);
  surface.mem_type = abi::kMemCudaDevice;
  const auto info = extract_info(&surface);
  auto decoder_owner = std::make_shared<int>(5);
  auto mapped = map_nvmm_frame_to_cuda(info, decoder_owner);
  expect_eq(mapped.y_ptr, 0x40000000U, "direct CUDA Y pointer");
  expect_eq(mapped.uv_ptr, 0x40000000U + 1280U * 720U, "direct CUDA UV pointer");
  expect_true(mapped.owner.use_count() == 1, "direct CUDA view owns context wrapper");

  params = make_params();
  params.data_ptr = reinterpret_cast<void*>(9);
  surface = make_surface(params);
  surface.mem_type = abi::kMemCudaDevice;
  const auto host_pointer = extract_info(&surface);
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(host_pointer, std::make_shared<int>(5)); },
                    "direct host-memory pointer rejected by CUDA attributes");
#endif
}

void concurrent_mapping_cleanup_is_serialized() {
#if defined(__linux__)
  auto params = make_params();
  auto surface = make_surface(params);
  const auto info = extract_info(&surface);
  auto decoder_owner = std::make_shared<int>(6);
  constexpr int thread_count = 8;
  constexpr int iteration_count = 250;
  std::atomic<bool> invalid_pointer = false;
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < iteration_count; ++iteration) {
        auto mapped = map_nvmm_frame_to_cuda(info, decoder_owner);
        if (mapped.y_ptr != 0x40000000U || mapped.uv_ptr != 0x40000000U + 1280U * 720U) {
          invalid_pointer = true;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  expect_true(!invalid_pointer, "concurrent mappings preserve CUDA plane pointers");
  expect_true(params.mapped_addr.cuda_ptr == nullptr,
              "last concurrent lease unmaps exactly after all users finish");
#endif
}

void independent_surfaces_keep_independent_runtime_mappings() {
#if defined(__linux__)
  auto first_params = make_params();
  auto second_params = make_params();
  second_params.buffer_desc = 18;
  auto first_surface = make_surface(first_params);
  auto second_surface = make_surface(second_params);
  auto first = map_nvmm_frame_to_cuda(extract_info(&first_surface), std::make_shared<int>(17));
  auto second = map_nvmm_frame_to_cuda(extract_info(&second_surface), std::make_shared<int>(18));
  expect_true(first_params.mapped_addr.cuda_ptr != second_params.mapped_addr.cuda_ptr,
              "runtime stores a distinct CUDA mapping object per surface");
  first.owner.reset();
  expect_true(first_params.mapped_addr.cuda_ptr == nullptr, "first surface unmaps independently");
  expect_true(second_params.mapped_addr.cuda_ptr != nullptr,
              "first surface cleanup does not invalidate second surface");
  second.owner.reset();
  expect_true(second_params.mapped_addr.cuda_ptr == nullptr, "second surface unmaps independently");
#endif
}

void failed_cleanup_poisoning_retains_surface_owners() {
#if defined(__linux__)
  auto cuda_params = make_params();
  cuda_params.buffer_desc = 3;
  auto cuda_surface = make_surface(cuda_params);
  const auto cuda_info = extract_info(&cuda_surface);
  auto cuda_owner = std::make_shared<int>(7);
  std::weak_ptr<int> cuda_lifetime = cuda_owner;
  auto cuda_mapping = map_nvmm_frame_to_cuda(cuda_info, cuda_owner);
  cuda_owner.reset();
  cuda_mapping.owner.reset();
  expect_true(!cuda_lifetime.expired(), "failed CUDA unmap retains decoder owner");
  expect_true(cuda_params.mapped_addr.cuda_ptr != nullptr, "failed CUDA unmap stays registered");
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(cuda_info, std::make_shared<int>(8)); },
                    "failed CUDA unmap poisons later remapping");

  auto egl_params = make_params_as<abi7::SurfaceParams>();
  egl_params.buffer_desc = 4;
  auto egl_surface = make_surface_as<abi7::Surface>(egl_params);
  const auto egl_info = extract_nvmm_frame_info(&egl_surface);
  auto egl_owner = std::make_shared<int>(9);
  std::weak_ptr<int> egl_lifetime = egl_owner;
  auto egl_mapping = map_nvmm_frame_to_cuda(egl_info, egl_owner);
  egl_owner.reset();
  egl_mapping.owner.reset();
  expect_true(!egl_lifetime.expired(), "failed EGL unregister retains decoder owner");
  expect_true(egl_params.mapped_addr.egl_image != nullptr,
              "failed EGL unregister keeps image mapping alive");
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(egl_info, std::make_shared<int>(10)); },
                    "failed EGL unregister poisons later remapping");

  auto egl_unmap_params = make_params_as<abi7::SurfaceParams>();
  egl_unmap_params.buffer_desc = 5;
  auto egl_unmap_surface = make_surface_as<abi7::Surface>(egl_unmap_params);
  const auto egl_unmap_info = extract_nvmm_frame_info(&egl_unmap_surface);
  auto egl_unmap_owner = std::make_shared<int>(11);
  std::weak_ptr<int> egl_unmap_lifetime = egl_unmap_owner;
  auto egl_unmap_mapping = map_nvmm_frame_to_cuda(egl_unmap_info, egl_unmap_owner);
  egl_unmap_owner.reset();
  egl_unmap_mapping.owner.reset();
  expect_true(!egl_unmap_lifetime.expired(), "failed EGL unmap retains decoder owner");
  expect_true(egl_unmap_params.mapped_addr.egl_image != nullptr,
              "failed EGL unmap keeps image mapping alive");
  expect_nvmm_error(
      [&] { (void)map_nvmm_frame_to_cuda(egl_unmap_info, std::make_shared<int>(12)); },
      "failed EGL unmap poisons later remapping");

  auto cuda_rollback_params = make_params();
  cuda_rollback_params.buffer_desc = 6;
  auto cuda_rollback_surface = make_surface(cuda_rollback_params);
  const auto cuda_rollback_info = extract_info(&cuda_rollback_surface);
  auto cuda_rollback_owner = std::make_shared<int>(13);
  std::weak_ptr<int> cuda_rollback_lifetime = cuda_rollback_owner;
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(cuda_rollback_info, cuda_rollback_owner); },
                    "failed CUDA setup rollback reports the validation failure");
  cuda_rollback_owner.reset();
  expect_true(!cuda_rollback_lifetime.expired(),
              "failed CUDA setup rollback retains decoder owner");
  expect_true(cuda_rollback_params.mapped_addr.cuda_ptr != nullptr,
              "failed CUDA setup rollback remains poisoned");
  expect_nvmm_error(
      [&] { (void)map_nvmm_frame_to_cuda(cuda_rollback_info, std::make_shared<int>(14)); },
      "failed CUDA setup rollback blocks remapping");

  auto egl_rollback_params = make_params_as<abi7::SurfaceParams>();
  egl_rollback_params.buffer_desc = 7;
  auto egl_rollback_surface = make_surface_as<abi7::Surface>(egl_rollback_params);
  const auto egl_rollback_info = extract_nvmm_frame_info(&egl_rollback_surface);
  auto egl_rollback_owner = std::make_shared<int>(15);
  std::weak_ptr<int> egl_rollback_lifetime = egl_rollback_owner;
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(egl_rollback_info, egl_rollback_owner); },
                    "failed EGL setup rollback reports descriptor validation failure");
  egl_rollback_owner.reset();
  expect_true(!egl_rollback_lifetime.expired(), "failed EGL setup rollback retains decoder owner");
  expect_true(egl_rollback_params.mapped_addr.egl_image != nullptr,
              "failed EGL setup rollback remains poisoned");
  expect_nvmm_error(
      [&] { (void)map_nvmm_frame_to_cuda(egl_rollback_info, std::make_shared<int>(16)); },
      "failed EGL setup rollback blocks remapping");

  auto cuda_partial_params = make_params();
  cuda_partial_params.buffer_desc = 96;
  auto cuda_partial_surface = make_surface(cuda_partial_params);
  const auto cuda_partial_info = extract_info(&cuda_partial_surface);
  auto cuda_partial_owner = std::make_shared<int>(19);
  std::weak_ptr<int> cuda_partial_lifetime = cuda_partial_owner;
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(cuda_partial_info, cuda_partial_owner); },
                    "partial CUDA map plus failed rollback reports map failure");
  cuda_partial_owner.reset();
  expect_true(!cuda_partial_lifetime.expired(),
              "failed partial CUDA rollback retains decoder owner");
  expect_true(cuda_partial_params.mapped_addr.cuda_ptr != nullptr,
              "failed partial CUDA rollback remains poisoned");
  expect_nvmm_error(
      [&] { (void)map_nvmm_frame_to_cuda(cuda_partial_info, std::make_shared<int>(20)); },
      "failed partial CUDA rollback blocks remapping");

  auto egl_partial_params = make_params_as<abi7::SurfaceParams>();
  egl_partial_params.buffer_desc = 95;
  auto egl_partial_surface = make_surface_as<abi7::Surface>(egl_partial_params);
  const auto egl_partial_info = extract_nvmm_frame_info(&egl_partial_surface);
  auto egl_partial_owner = std::make_shared<int>(21);
  std::weak_ptr<int> egl_partial_lifetime = egl_partial_owner;
  expect_nvmm_error([&] { (void)map_nvmm_frame_to_cuda(egl_partial_info, egl_partial_owner); },
                    "partial EGL map plus failed rollback reports map failure");
  egl_partial_owner.reset();
  expect_true(!egl_partial_lifetime.expired(), "failed partial EGL rollback retains decoder owner");
  expect_true(egl_partial_params.mapped_addr.egl_image != nullptr,
              "failed partial EGL rollback remains poisoned");
  expect_nvmm_error(
      [&] { (void)map_nvmm_frame_to_cuda(egl_partial_info, std::make_shared<int>(22)); },
      "failed partial EGL rollback blocks remapping");
#endif
}

} // namespace

int main() {
  surface_array_mapping_retains_and_unmaps_owner();
  deepstream_7_1_egl_mapping_is_gpu_resident();
  cuda_device_mapping_retains_context_and_owner();
  concurrent_mapping_cleanup_is_serialized();
  independent_surfaces_keep_independent_runtime_mappings();
  failed_cleanup_poisoning_retains_surface_owners();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
