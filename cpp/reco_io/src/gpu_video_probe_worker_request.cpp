#include "gpu_video_probe_worker_request.hpp"

#include <utility>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace reco::io::detail {

#if defined(_WIN32)
ReceivedRequest::ReceivedRequest(std::string payload) noexcept : payload_(std::move(payload)) {}
#else
ReceivedRequest::ReceivedRequest(std::string payload, int descriptor) noexcept
    : payload_(std::move(payload)), descriptor_(descriptor) {}
#endif

ReceivedRequest::ReceivedRequest(ReceivedRequest&& other) noexcept
    : payload_(std::move(other.payload_))
#if !defined(_WIN32)
      ,
      descriptor_(std::exchange(other.descriptor_, -1))
#endif
{
}

ReceivedRequest& ReceivedRequest::operator=(ReceivedRequest&& other) noexcept {
  if (this == &other) {
    return *this;
  }
#if !defined(_WIN32)
  if (descriptor_ >= 0) {
    (void)::close(descriptor_);
  }
  descriptor_ = std::exchange(other.descriptor_, -1);
#endif
  payload_ = std::move(other.payload_);
  return *this;
}

ReceivedRequest::~ReceivedRequest() {
#if !defined(_WIN32)
  if (descriptor_ >= 0) {
    (void)::close(descriptor_);
  }
#endif
}

std::string& ReceivedRequest::payload() noexcept { return payload_; }

const std::string& ReceivedRequest::payload() const noexcept { return payload_; }

#if !defined(_WIN32)
int ReceivedRequest::descriptor() const noexcept { return descriptor_; }

int ReceivedRequest::release_descriptor() noexcept { return std::exchange(descriptor_, -1); }
#endif

} // namespace reco::io::detail
