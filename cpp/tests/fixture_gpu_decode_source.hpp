#pragma once

#include "reco/io/gpu_decode.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace reco::tests {

class FixtureGpuFileDecodeSource final : public reco::io::GpuFileDecodeSource {
public:
  FixtureGpuFileDecodeSource(reco::io::GpuFileDecodeConfig config,
                             std::vector<reco::io::GpuDecodedFrame> frames)
      : config_(std::move(config)),
        pipeline_(reco::io::build_gstreamer_gpu_file_decode_pipeline(config_)),
        frames_(std::move(frames)) {
    for (const auto& frame : frames_) {
      if (const auto error = reco::io::validate_gpu_decoded_frame(frame); error.has_value()) {
        throw std::invalid_argument(*error);
      }
    }
  }

  [[nodiscard]] const reco::io::GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return pipeline_; }
  [[nodiscard]] bool gpu_resident() const override { return true; }

  [[nodiscard]] reco::io::GpuDecodeReadResult read() override {
    if (next_ >= frames_.size()) {
      return reco::io::make_gpu_decode_eos();
    }
    return reco::io::make_gpu_decode_frame(frames_[next_++]);
  }

private:
  reco::io::GpuFileDecodeConfig config_;
  std::string pipeline_;
  std::vector<reco::io::GpuDecodedFrame> frames_;
  std::size_t next_ = 0;
};

[[nodiscard]] inline std::unique_ptr<reco::io::GpuFileDecodeSource>
make_fixture_gpu_file_decode_source(reco::io::GpuFileDecodeConfig config,
                                    std::vector<reco::io::GpuDecodedFrame> frames) {
  return std::make_unique<FixtureGpuFileDecodeSource>(std::move(config), std::move(frames));
}

} // namespace reco::tests
