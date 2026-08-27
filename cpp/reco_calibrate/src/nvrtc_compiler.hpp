#pragma once

#include <string>
#include <string_view>

namespace reco::calibrate::detail {

struct NvrtcCompileOptions {
  bool disable_fmad = false;
};

class NvrtcCompiler {
public:
  NvrtcCompiler();
  ~NvrtcCompiler();

  NvrtcCompiler(const NvrtcCompiler&) = delete;
  NvrtcCompiler& operator=(const NvrtcCompiler&) = delete;
  NvrtcCompiler(NvrtcCompiler&&) noexcept;
  NvrtcCompiler& operator=(NvrtcCompiler&&) noexcept;

  [[nodiscard]] std::string compile(std::string_view source, std::string_view source_name,
                                    NvrtcCompileOptions options = {}) const;

private:
  struct Impl;
  Impl* impl_ = nullptr;
};

} // namespace reco::calibrate::detail
