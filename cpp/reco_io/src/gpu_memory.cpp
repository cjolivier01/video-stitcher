#include "reco/io/gpu_memory.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace reco::io {
namespace {

constexpr std::size_t kNvmmPitchAlignment = 256;
constexpr std::size_t kNvmmHeightAlignment = 64;
constexpr std::size_t kRgbaPitchAlignment = 512;
// Reserve 16 codec reference/DPB surfaces and four conversion/output surfaces per decoder.
constexpr std::size_t kDecoderInternalSurfaceReserve = 20;
constexpr std::size_t kMebibyte = 1024U * 1024U;
constexpr std::size_t kGibibyte = 1024U * kMebibyte;
constexpr std::size_t kDiscreteMinimumReserve = 512U * kMebibyte;
constexpr std::size_t kDiscreteMaximumReserve = 2U * kGibibyte;
constexpr std::size_t kIntegratedMinimumReserve = 1U * kGibibyte;
constexpr std::size_t kIntegratedMaximumReserve = 4U * kGibibyte;

std::size_t checked_add(std::size_t lhs, std::size_t rhs, std::string_view label) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw std::overflow_error(std::string(label) + " overflows size_t");
  }
  return lhs + rhs;
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, std::string_view label) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(std::string(label) + " overflows size_t");
  }
  return lhs * rhs;
}

std::size_t checked_align_up(std::size_t value, std::size_t alignment, std::string_view label) {
  const auto remainder = value % alignment;
  return remainder == 0 ? value : checked_add(value, alignment - remainder, label);
}

std::size_t estimate_rgba_surface_bytes(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("RGBA GPU memory estimate dimensions must be non-zero");
  }
  const auto row_bytes =
      checked_multiply(static_cast<std::size_t>(width), 4U, "RGBA row byte estimate");
  const auto pitch = checked_align_up(row_bytes, kRgbaPitchAlignment, "RGBA pitch estimate");
  return checked_multiply(pitch, static_cast<std::size_t>(height), "RGBA surface byte estimate");
}

std::string display_bytes(std::size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << static_cast<long double>(bytes) / static_cast<long double>(kGibibyte) << " GiB";
  return out.str();
}

} // namespace

std::size_t estimate_nvmm_nv12_surface_bytes(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0 || (width % 2U) != 0 || (height % 2U) != 0) {
    throw std::invalid_argument("NVMM memory estimate dimensions must be non-zero and even");
  }
  const auto pitch =
      checked_align_up(static_cast<std::size_t>(width), kNvmmPitchAlignment, "NVMM pitch estimate");
  const auto rows = checked_align_up(static_cast<std::size_t>(height), kNvmmHeightAlignment,
                                     "NVMM height estimate");
  const auto luma_bytes = checked_multiply(pitch, rows, "NVMM luma byte estimate");
  // Two bytes per aligned pixel conservatively covers NV12's 1.5 bytes plus block-linear
  // padding and allocation metadata without relying on one DeepStream generation's layout.
  return checked_multiply(luma_bytes, 2U, "NVMM surface byte estimate");
}

std::size_t estimate_gpu_encode_pool_bytes(std::uint32_t width, std::uint32_t height,
                                           std::size_t pool_capacity) {
  if (pool_capacity == 0) {
    throw std::invalid_argument("GPU encode pool memory estimate capacity must be non-zero");
  }
  return checked_multiply(estimate_nvmm_nv12_surface_bytes(width, height), pool_capacity,
                          "GPU encode pool byte estimate");
}

GpuStitchMemoryEstimate estimate_gpu_stitch_memory(const GpuStitchMemoryConfig& config) {
  if (config.decode_source_capacity == 0 || config.stereo_queue_capacity == 0 ||
      config.encode_pool_capacity == 0) {
    throw std::invalid_argument("GPU stitch memory estimate capacities must be non-zero");
  }

  GpuStitchMemoryEstimate estimate;
  estimate.encode_pool_bytes = estimate_gpu_encode_pool_bytes(
      config.output_width, config.output_height, config.encode_pool_capacity);
  estimate.rgba_output_bytes =
      estimate_rgba_surface_bytes(config.output_width, config.output_height);

  auto retained_per_input = checked_add(config.decode_source_capacity, config.stereo_queue_capacity,
                                        "GPU decode retained surface count");
  retained_per_input = checked_add(retained_per_input, 1U, "GPU decode in-flight surface count");
  retained_per_input = checked_add(retained_per_input, kDecoderInternalSurfaceReserve,
                                   "GPU decoder internal surface count");
  const auto left_bytes =
      checked_multiply(estimate_nvmm_nv12_surface_bytes(config.left_width, config.left_height),
                       retained_per_input, "left GPU decode surface byte estimate");
  const auto right_bytes =
      checked_multiply(estimate_nvmm_nv12_surface_bytes(config.right_width, config.right_height),
                       retained_per_input, "right GPU decode surface byte estimate");
  estimate.decode_surfaces_bytes =
      checked_add(left_bytes, right_bytes, "stereo GPU decode surface byte estimate");
  estimate.total_bytes =
      checked_add(checked_add(estimate.encode_pool_bytes, estimate.rgba_output_bytes,
                              "GPU stitch output working-set estimate"),
                  estimate.decode_surfaces_bytes, "GPU stitch total working-set estimate");
  return estimate;
}

GpuMemoryPreflight evaluate_gpu_memory_preflight(std::size_t working_set_bytes,
                                                 const core::CudaMemoryInfo& memory) {
  if (working_set_bytes == 0) {
    throw std::invalid_argument("GPU memory preflight working set must be non-zero");
  }
  if (memory.total_bytes == 0 || memory.free_bytes > memory.total_bytes) {
    throw std::invalid_argument("CUDA returned an invalid free/total memory snapshot");
  }

  const auto proportional_reserve =
      memory.integrated ? memory.total_bytes / 4U : memory.total_bytes / 10U;
  // Integrated GPUs compete with the host and media engines for physical pages, so retain a
  // larger bounded fraction there. Caps avoid needlessly stranding memory on large devices.
  const auto safety_reserve =
      memory.integrated
          ? std::clamp(proportional_reserve, kIntegratedMinimumReserve, kIntegratedMaximumReserve)
          : std::clamp(proportional_reserve, kDiscreteMinimumReserve, kDiscreteMaximumReserve);
  return {
      .working_set_bytes = working_set_bytes,
      .safety_reserve_bytes = safety_reserve,
      .required_free_bytes =
          checked_add(working_set_bytes, safety_reserve, "GPU memory preflight requirement"),
      .available_free_bytes = memory.free_bytes,
      .total_bytes = memory.total_bytes,
      .integrated = memory.integrated,
  };
}

void require_gpu_memory_preflight(const GpuMemoryPreflight& preflight, std::string_view operation) {
  if (operation.empty()) {
    throw std::invalid_argument("GPU memory preflight operation label must be non-empty");
  }
  if (preflight.working_set_bytes == 0 || preflight.safety_reserve_bytes == 0 ||
      preflight.total_bytes == 0 || preflight.available_free_bytes > preflight.total_bytes ||
      preflight.required_free_bytes != checked_add(preflight.working_set_bytes,
                                                   preflight.safety_reserve_bytes,
                                                   "GPU memory preflight result")) {
    throw std::invalid_argument("GPU memory preflight result is invalid");
  }
  if (preflight.available_free_bytes >= preflight.required_free_bytes) {
    return;
  }

  throw std::runtime_error(
      std::string(operation) + " needs at least " + display_bytes(preflight.required_free_bytes) +
      " free " + (preflight.integrated ? "shared system/GPU memory" : "device memory") +
      " (estimated GPU working set " + display_bytes(preflight.working_set_bytes) + " plus " +
      display_bytes(preflight.safety_reserve_bytes) + " safety reserve), but CUDA reports " +
      display_bytes(preflight.available_free_bytes) + " free of " +
      display_bytes(preflight.total_bytes) +
      (preflight.integrated ? " on an integrated CUDA device; " : "; ") +
      "reduce output dimensions or pool/queue capacities, or free GPU memory before retrying; "
      "CPU fallback is disabled");
}

} // namespace reco::io
