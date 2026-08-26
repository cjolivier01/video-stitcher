#pragma once

#include <string>
#include <string_view>

namespace reco::calibrate::detail {

class NvrtcCompiler {
public:
  NvrtcCompiler();
  ~NvrtcCompiler();

  NvrtcCompiler(const NvrtcCompiler&) = delete;
  NvrtcCompiler& operator=(const NvrtcCompiler&) = delete;
  NvrtcCompiler(NvrtcCompiler&&) noexcept;
  NvrtcCompiler& operator=(NvrtcCompiler&&) noexcept;

  [[nodiscard]] std::string compile(std::string_view source, std::string_view source_name) const;

private:
  struct Impl;
  Impl* impl_ = nullptr;
};

} // namespace reco::calibrate::detail
