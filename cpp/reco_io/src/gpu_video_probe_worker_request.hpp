#pragma once

#include <string>

namespace reco::io::detail {

class ReceivedRequest {
public:
#if defined(_WIN32)
  explicit ReceivedRequest(std::string payload) noexcept;
#else
  ReceivedRequest(std::string payload, int descriptor) noexcept;
#endif
  ReceivedRequest(const ReceivedRequest&) = delete;
  ReceivedRequest& operator=(const ReceivedRequest&) = delete;
  ReceivedRequest(ReceivedRequest&& other) noexcept;
  ReceivedRequest& operator=(ReceivedRequest&& other) noexcept;
  ~ReceivedRequest();

  [[nodiscard]] std::string& payload() noexcept;
  [[nodiscard]] const std::string& payload() const noexcept;

#if !defined(_WIN32)
  [[nodiscard]] int descriptor() const noexcept;
  [[nodiscard]] int release_descriptor() noexcept;
#endif

private:
  std::string payload_;
#if !defined(_WIN32)
  int descriptor_ = -1;
#endif
};

} // namespace reco::io::detail
