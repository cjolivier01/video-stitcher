#include "reco/io/gpu_decode.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace reco::io {
namespace {

constexpr std::size_t kMaximumGpuDecodeSegments = 4096;

using SourceOpener = std::function<std::unique_ptr<GpuFileDecodeSource>(GpuFileDecodeConfig)>;

std::uint64_t add_frame_counts(std::uint64_t left, std::uint64_t right) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw GpuDecodeError("chained GPU decode frame index overflow");
  }
  return left + right;
}

class ChainedGpuFileDecodeSource final : public GpuFileDecodeSource {
public:
  ChainedGpuFileDecodeSource(GpuChainedFileDecodeConfig config, SourceOpener opener)
      : config_(std::move(config)), opener_(std::move(opener)) {
    position_at(config_.start_frame_index.value_or(0));
  }

  ~ChainedGpuFileDecodeSource() override { request_stop(); }

  [[nodiscard]] const GpuFileDecodeConfig& config() const override {
    return config_.segments.front().config;
  }

  [[nodiscard]] std::string_view pipeline() const override {
    return "lazy-gpu-chain ! video/x-raw(memory:NVMM)";
  }
  [[nodiscard]] bool gpu_resident() const override { return true; }

  [[nodiscard]] GpuDecodeReadResult read() override {
    std::lock_guard operation_lock(operation_mutex_);
    for (;;) {
      if (stopped_.load(std::memory_order_acquire) || ended_) {
        return make_gpu_decode_eos();
      }
      ensure_source_open();
      if (ended_) {
        return make_gpu_decode_eos();
      }

      const auto source = current_source();
      if (!source) {
        throw GpuDecodeError("chained GPU decode lost its active segment");
      }
      auto result = source->read();
      if (result.status == GpuDecodeFrameStatus::Frame) {
        if (!result.frame.has_value()) {
          throw GpuDecodeError("chained GPU decode segment returned an empty frame");
        }
        const auto local_index = result.frame->frame_index;
        const auto& segment = config_.segments[segment_index_];
        if (segment.exact_frame_count.has_value() && local_index >= *segment.exact_frame_count) {
          throw GpuDecodeError("chained GPU decode exceeded the parser-proven segment length");
        }
        result.frame->frame_index = add_frame_counts(segment_base_index_, local_index);
        if (!last_local_index_.has_value() || local_index > *last_local_index_) {
          last_local_index_ = local_index;
        }
        return result;
      }
      if (result.frame.has_value()) {
        throw GpuDecodeError("chained GPU decode segment EOS contains a frame");
      }
      advance_segment(source);
    }
  }

  void request_stop() noexcept override {
    stopped_.store(true, std::memory_order_release);
    const auto source = current_source();
    if (source) {
      source->request_stop();
    }
  }

  void seek_to_frame(std::uint64_t frame_index) override {
    std::lock_guard operation_lock(operation_mutex_);
    if (stopped_.load(std::memory_order_acquire)) {
      throw GpuDecodeError("chained GPU decode source has been stopped");
    }
    if (const auto source = current_source(); source) {
      source->request_stop();
    }
    {
      std::lock_guard state_lock(state_mutex_);
      source_.reset();
    }
    ended_ = false;
    position_at(frame_index);
  }

private:
  [[nodiscard]] std::shared_ptr<GpuFileDecodeSource> current_source() const {
    std::lock_guard state_lock(state_mutex_);
    return source_;
  }

  void position_at(std::uint64_t frame_index) {
    segment_index_ = 0;
    segment_base_index_ = 0;
    local_start_frame_.reset();
    last_local_index_.reset();

    if (frame_index == 0) {
      return;
    }
    for (; segment_index_ < config_.segments.size(); ++segment_index_) {
      const auto count = config_.segments[segment_index_].exact_frame_count;
      if (!count.has_value()) {
        throw GpuDecodeError(
            "global chained GPU decode seeking requires exact counts for preceding segments");
      }
      const auto segment_end = add_frame_counts(segment_base_index_, *count);
      if (frame_index < segment_end) {
        local_start_frame_ = frame_index - segment_base_index_;
        return;
      }
      segment_base_index_ = segment_end;
      if (frame_index == segment_base_index_) {
        ++segment_index_;
        return;
      }
    }
    if (frame_index != segment_base_index_) {
      throw GpuDecodeError("chained GPU decode start frame exceeds the joined input length");
    }
    ended_ = true;
  }

  void ensure_source_open() {
    if (current_source()) {
      return;
    }
    if (segment_index_ >= config_.segments.size()) {
      ended_ = true;
      return;
    }
    auto segment_config = config_.segments[segment_index_].config;
    segment_config.start_frame_index = local_start_frame_;
    auto opened = std::shared_ptr<GpuFileDecodeSource>(opener_(std::move(segment_config)));
    if (!opened || !opened->gpu_resident()) {
      throw GpuDecodeError("chained GPU decode opener returned a non-GPU source");
    }
    bool reject_open = false;
    {
      std::lock_guard state_lock(state_mutex_);
      reject_open = stopped_.load(std::memory_order_acquire);
      if (!reject_open) {
        source_ = opened;
      }
    }
    if (reject_open) {
      opened->request_stop();
      ended_ = true;
    }
  }

  void advance_segment(const std::shared_ptr<GpuFileDecodeSource>& completed) {
    const auto& segment = config_.segments[segment_index_];
    const auto observed_count =
        last_local_index_.has_value() ? add_frame_counts(*last_local_index_, 1U) : 0U;
    const auto segment_count = segment.exact_frame_count.value_or(observed_count);
    segment_base_index_ = add_frame_counts(segment_base_index_, segment_count);
    {
      std::lock_guard state_lock(state_mutex_);
      if (source_ == completed) {
        source_.reset();
      }
    }
    ++segment_index_;
    local_start_frame_.reset();
    last_local_index_.reset();
    if (segment_index_ >= config_.segments.size()) {
      ended_ = true;
    }
  }

  GpuChainedFileDecodeConfig config_;
  SourceOpener opener_;
  mutable std::mutex state_mutex_;
  std::mutex operation_mutex_;
  std::shared_ptr<GpuFileDecodeSource> source_;
  std::atomic<bool> stopped_{false};
  std::size_t segment_index_ = 0;
  std::uint64_t segment_base_index_ = 0;
  std::optional<std::uint64_t> local_start_frame_;
  std::optional<std::uint64_t> last_local_index_;
  bool ended_ = false;
};

} // namespace

std::optional<std::string>
validate_gpu_chained_file_decode_config(const GpuChainedFileDecodeConfig& config) {
  if (config.segments.empty()) {
    return "chained GPU decode requires at least one segment";
  }
  if (config.segments.size() > kMaximumGpuDecodeSegments) {
    return "chained GPU decode exceeds the 4096-segment bound";
  }
  std::uint64_t exact_prefix = 0;
  bool prefix_is_exact = true;
  for (const auto& segment : config.segments) {
    if (const auto error = validate_gpu_file_decode_config(segment.config); error.has_value()) {
      return "invalid chained GPU decode segment: " + *error;
    }
    if (segment.config.start_frame_index.has_value()) {
      return "chained GPU decode segment starts are controlled by the joined source";
    }
    if (segment.exact_frame_count == 0U) {
      return "chained GPU decode exact frame counts must be positive";
    }
    if (prefix_is_exact && segment.exact_frame_count.has_value()) {
      if (*segment.exact_frame_count > std::numeric_limits<std::uint64_t>::max() - exact_prefix) {
        return "chained GPU decode exact frame counts overflow";
      }
      exact_prefix += *segment.exact_frame_count;
    } else {
      prefix_is_exact = false;
    }
  }
  if (config.start_frame_index.has_value()) {
    if (!prefix_is_exact) {
      return "chained GPU decode indexed start requires exact counts for every segment";
    }
    if (*config.start_frame_index > exact_prefix) {
      return "chained GPU decode start frame exceeds the joined input length";
    }
  }
  return std::nullopt;
}

std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_chained_file_decode_source(GpuChainedFileDecodeConfig config,
                                              NvbufSurfaceAbi abi) {
  if (const auto error = validate_gpu_chained_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  return std::make_unique<ChainedGpuFileDecodeSource>(
      std::move(config), [abi](GpuFileDecodeConfig segment) {
        return open_gstreamer_gpu_file_decode_source(std::move(segment), abi);
      });
}

std::unique_ptr<GpuFileDecodeSource>
open_gstreamer_gpu_chained_file_decode_source(GpuChainedFileDecodeConfig config,
                                              std::shared_ptr<const NvbufSurfaceRuntime> runtime) {
  if (const auto error = validate_gpu_chained_file_decode_config(config); error.has_value()) {
    throw std::invalid_argument(*error);
  }
  if (!runtime) {
    throw std::invalid_argument("chained GPU decode requires a retained NvBufSurface runtime");
  }
  return std::make_unique<ChainedGpuFileDecodeSource>(
      std::move(config), [runtime = std::move(runtime)](GpuFileDecodeConfig segment) {
        return open_gstreamer_gpu_file_decode_source(std::move(segment), runtime);
      });
}

} // namespace reco::io
