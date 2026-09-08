#include "reco/io/detail/nvbufsurface_7_1.hpp"
#include "reco/io/detail/nvbufsurface_9_1.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

namespace abi7 = reco::io::detail::nvbufsurface_7_1;
namespace abi = reco::io::detail::nvbufsurface_9_1;

namespace {

void* mapped_device_pointer(std::uint64_t descriptor) {
  switch (descriptor) {
  case 14:
    return reinterpret_cast<void*>(0x50000000);
  case 15:
    return reinterpret_cast<void*>(0x51000000);
  case 16:
    return reinterpret_cast<void*>(0x52000000);
  case 17:
    return reinterpret_cast<void*>(0x40000000);
  default:
    return reinterpret_cast<void*>(descriptor);
  }
}

#if !defined(RECO_FAKE_NVBUFSURFACE_7_1)
std::mutex cuda_buffers_mutex;
std::unordered_map<void*, std::unique_ptr<abi::CudaBuffer>> cuda_buffers;
std::atomic<std::uintptr_t> next_fake_device_pointer{0x60000000U};

struct FakeSurfaceAllocation {
  abi::Surface surface;
  abi::SurfaceParams params;
};
#endif

} // namespace

extern "C" void nvds_version(unsigned int* major, unsigned int* minor) {
  if (major == nullptr || minor == nullptr) {
    return;
  }
#if defined(RECO_FAKE_NVBUFSURFACE_7_1)
  *major = 7;
  *minor = 1;
#else
  const char* version = std::getenv("RECO_FAKE_DEEPSTREAM_VERSION");
  if (version != nullptr && std::strcmp(version, "7.1") == 0) {
    *major = 7;
    *minor = 1;
  } else if (version != nullptr && std::strcmp(version, "8.0") == 0) {
    *major = 8;
    *minor = 0;
  } else if (version != nullptr && std::strcmp(version, "0.0") == 0) {
    *major = 0;
    *minor = 0;
  } else {
    *major = 9;
    *minor = 1;
  }
#endif
}

#if !defined(RECO_FAKE_NVBUFSURFACE_7_1)
extern "C" int NvBufSurfaceCreate(void** output, std::uint32_t batch_size, void* raw_params) {
  if (output == nullptr || raw_params == nullptr || batch_size != 1U) {
    return -1;
  }
  const auto* create = static_cast<const abi::CreateParams*>(raw_params);
  if (create->width == 0 || create->height == 0 || (create->width % 2U) != 0 ||
      (create->height % 2U) != 0) {
    return -1;
  }
  auto allocation =
      std::unique_ptr<FakeSurfaceAllocation>(new (std::nothrow) FakeSurfaceAllocation);
  if (!allocation) {
    return -1;
  }
  const auto pitch = (create->width + 255U) & ~255U;
  const auto y_size = pitch * create->height;
  const auto uv_size = pitch * (create->height / 2U);
  auto& surface = allocation->surface;
  auto& params = allocation->params;
  surface.gpu_id = create->gpu_id;
  surface.batch_size = 1;
  surface.num_filled = 1;
  surface.is_contiguous = 1;
  surface.mem_type = abi::kMemCudaDevice;
  surface.surface_list = &params;
  params.width = create->width;
  params.height = create->height;
  params.pitch = pitch;
  params.color_format = create->color_format;
  params.layout = create->layout;
  params.data_size = y_size + uv_size;
  params.data_ptr = reinterpret_cast<void*>(
      next_fake_device_pointer.fetch_add(0x200000U, std::memory_order_relaxed));
  params.plane_params.num_planes = 2;
  params.plane_params.width[0] = create->width;
  params.plane_params.width[1] = create->width / 2U;
  params.plane_params.height[0] = create->height;
  params.plane_params.height[1] = create->height / 2U;
  params.plane_params.pitch[0] = pitch;
  params.plane_params.pitch[1] = pitch;
  params.plane_params.offset[0] = 0;
  params.plane_params.offset[1] = y_size;
  params.plane_params.psize[0] = y_size;
  params.plane_params.psize[1] = uv_size;
  params.plane_params.bytes_per_pix[0] = 1;
  params.plane_params.bytes_per_pix[1] = 2;
  *output = &allocation.release()->surface;
  return 0;
}

extern "C" int NvBufSurfaceDestroy(void* raw_surface) {
  if (raw_surface == nullptr) {
    return -1;
  }
  delete reinterpret_cast<FakeSurfaceAllocation*>(raw_surface);
  return 0;
}

extern "C" int NvBufSurfaceMapCudaBuffer(void* raw_surface, int index) {
  if (raw_surface == nullptr || index != 0) {
    return -1;
  }
  auto* surface = static_cast<abi::Surface*>(raw_surface);
  std::lock_guard<std::mutex> lock(cuda_buffers_mutex);
  if (surface->surface_list == nullptr || surface->surface_list[0].buffer_desc == 99 ||
      surface->surface_list[0].mapped_addr.cuda_ptr != nullptr ||
      cuda_buffers.find(surface) != cuda_buffers.end()) {
    return -1;
  }
  auto cuda_buffer = std::unique_ptr<abi::CudaBuffer>(new (std::nothrow) abi::CudaBuffer);
  if (!cuda_buffer) {
    return -1;
  }
  const auto descriptor = surface->surface_list[0].buffer_desc;
  cuda_buffer->base_ptr = mapped_device_pointer(descriptor);
  cuda_buffer->data_ptr =
      descriptor == 1 || descriptor == 6 ? nullptr : mapped_device_pointer(descriptor);
  auto* cuda_buffer_ptr = cuda_buffer.get();
  try {
    cuda_buffers.emplace(surface, std::move(cuda_buffer));
  } catch (...) {
    return -1;
  }
  surface->surface_list[0].mapped_addr.cuda_ptr = cuda_buffer_ptr;
  return descriptor == 96 || descriptor == 98 ? -1 : 0;
}

extern "C" int NvBufSurfaceUnMapCudaBuffer(void* raw_surface, int index) {
  if (raw_surface == nullptr || index != 0) {
    return -1;
  }
  auto* surface = static_cast<abi::Surface*>(raw_surface);
  std::lock_guard<std::mutex> lock(cuda_buffers_mutex);
  if (surface->surface_list == nullptr) {
    return -1;
  }
  if (surface->surface_list[0].buffer_desc == 3 || surface->surface_list[0].buffer_desc == 6 ||
      surface->surface_list[0].buffer_desc == 96) {
    return -1;
  }
  cuda_buffers.erase(surface);
  surface->surface_list[0].mapped_addr.cuda_ptr = nullptr;
  return 0;
}
#endif

extern "C" int NvBufSurfaceMapEglImage(void* raw_surface, int index) {
  if (raw_surface == nullptr || index != 0) {
    return -1;
  }
  auto* surface = static_cast<abi7::Surface*>(raw_surface);
  if (surface->surface_list == nullptr || surface->surface_list[0].buffer_desc == 99 ||
      surface->surface_list[0].mapped_addr.egl_image != nullptr) {
    return -1;
  }
  surface->surface_list[0].mapped_addr.egl_image =
      mapped_device_pointer(static_cast<std::uint64_t>(surface->surface_list[0].buffer_desc));
  return surface->surface_list[0].buffer_desc == 95 || surface->surface_list[0].buffer_desc == 97
             ? -1
             : 0;
}

extern "C" int NvBufSurfaceUnMapEglImage(void* raw_surface, int index) {
  if (raw_surface == nullptr || index != 0) {
    return -1;
  }
  auto* surface = static_cast<abi7::Surface*>(raw_surface);
  if (surface->surface_list == nullptr) {
    return -1;
  }
  if (surface->surface_list[0].buffer_desc == 5 || surface->surface_list[0].buffer_desc == 7 ||
      surface->surface_list[0].buffer_desc == 95) {
    return -1;
  }
  surface->surface_list[0].mapped_addr.egl_image = nullptr;
  return 0;
}
