#include "reco/io/detail/nvbufsurface_7_1.hpp"
#include "reco/io/detail/nvbufsurface_9_1.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

namespace abi7 = reco::io::detail::nvbufsurface_7_1;
namespace abi = reco::io::detail::nvbufsurface_9_1;

namespace {

void* mapped_device_pointer(std::uint64_t descriptor) {
  return descriptor == 17 ? reinterpret_cast<void*>(0x40000000)
                          : reinterpret_cast<void*>(descriptor);
}

std::mutex cuda_buffers_mutex;
std::unordered_map<void*, std::unique_ptr<abi::CudaBuffer>> cuda_buffers;

} // namespace

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
  return surface->surface_list[0].buffer_desc == 95 ||
                 surface->surface_list[0].buffer_desc == 97
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
