#include "reco/calibrate/sampling.hpp"
#include "reco/io/detail/nvbufsurface_9_1.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace reco::calibrate;
using namespace reco::core;
using namespace reco::io;
namespace abi = reco::io::detail::nvbufsurface_9_1;

namespace {

int failures = 0;

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

template <typename Error, typename Function>
void expect_error(Function&& function, std::string_view fragment, std::string_view message) {
  try {
    function();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const Error& error) {
    if (std::string_view(error.what()).find(fragment) == std::string_view::npos) {
      std::cerr << "FAIL: " << message << " missing fragment: " << error.what() << '\n';
      ++failures;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

GpuFileDecodeConfig fixture_config(bool drop = false) {
  return {.path = "fixture.mp4", .container = GpuDecodeContainer::QuickTime, .drop = drop};
}

GpuDecodedFrame metadata_frame(std::uint64_t frame_index) {
  return {.nvmm = {.dmabuf_fd = 7,
                   .width = 64,
                   .height = 32,
                   .y_offset = 0,
                   .y_pitch = 64,
                   .uv_offset = 64 * 32,
                   .uv_pitch = 64,
                   .total_size = 64 * 48,
                   .surface_ptr = reinterpret_cast<void*>(0x1000),
                   .abi = NvbufSurfaceAbi::DeepStream9_1,
                   .memory_type = NvmmMemoryType::SurfaceArray},
          .visible_width = 64,
          .visible_height = 32,
          .owner = std::make_shared<int>(1),
          .frame_index = frame_index};
}

class VectorSource final : public GpuFileDecodeSource {
public:
  explicit VectorSource(std::vector<GpuDecodedFrame> frames, bool gpu_resident = true,
                        bool drop = false)
      : config_(fixture_config(drop)), pipeline_(build_gstreamer_gpu_file_decode_pipeline(config_)),
        frames_(std::move(frames)), gpu_resident_(gpu_resident) {}

  [[nodiscard]] const GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return pipeline_; }
  [[nodiscard]] bool gpu_resident() const override { return gpu_resident_; }
  [[nodiscard]] GpuDecodeReadResult read() override {
    ++read_count_;
    if (next_ == frames_.size()) {
      return make_gpu_decode_eos();
    }
    return make_gpu_decode_frame(std::move(frames_[next_++]));
  }
  [[nodiscard]] std::size_t read_count() const { return read_count_; }

private:
  GpuFileDecodeConfig config_;
  std::string pipeline_;
  std::vector<GpuDecodedFrame> frames_;
  std::size_t next_ = 0;
  std::size_t read_count_ = 0;
  bool gpu_resident_ = true;
};

class MissingPayloadSource final : public GpuFileDecodeSource {
public:
  MissingPayloadSource() : config_(fixture_config()) {}
  [[nodiscard]] const GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return "fixture"; }
  [[nodiscard]] bool gpu_resident() const override { return true; }
  [[nodiscard]] GpuDecodeReadResult read() override {
    return {.status = GpuDecodeFrameStatus::Frame, .frame = std::nullopt};
  }

private:
  GpuFileDecodeConfig config_;
};

void extraction_contract_rejects_invalid_streams(CudaBackend& backend) {
  VectorSource non_gpu({}, false);
  expect_error<std::invalid_argument>(
      [&] { (void)extract_gpu_gray_frames(backend, non_gpu, std::vector<std::uint64_t>{0}); },
      "GPU-resident", "CPU source rejected");

  VectorSource dropping({}, true, true);
  expect_error<std::invalid_argument>(
      [&] { (void)extract_gpu_gray_frames(backend, dropping, std::vector<std::uint64_t>{0}); },
      "dropping to be disabled", "drop-enabled source rejected");
  expect_eq(dropping.read_count(), 0U, "drop-enabled source does not consume decoder frames");

  expect_error<std::invalid_argument>(
      [&] {
        (void)extract_gpu_gray_frames_from_file(
            backend, fixture_config(true), NvbufSurfaceAbi::DeepStream9_1,
            std::vector<std::uint64_t>{0});
      },
      "dropping to be disabled", "drop-enabled file config rejected before decoder startup");

  VectorSource unsorted({});
  expect_error<std::invalid_argument>(
      [&] { (void)extract_gpu_gray_frames(backend, unsorted, std::vector<std::uint64_t>{2, 1}); },
      "sorted and unique", "unsorted indices rejected");
  expect_eq(unsorted.read_count(), 0U, "invalid indices do not consume decoder frames");

  VectorSource duplicate({});
  expect_error<std::invalid_argument>(
      [&] { (void)extract_gpu_gray_frames(backend, duplicate, std::vector<std::uint64_t>{1, 1}); },
      "sorted and unique", "duplicate indices rejected");

  VectorSource eos({});
  expect_error<GpuFrameExtractionError>(
      [&] { (void)extract_gpu_gray_frames(backend, eos, std::vector<std::uint64_t>{4}); },
      "before calibration frame 4", "early EOS identifies missing frame");

  VectorSource skipped({metadata_frame(2)});
  expect_error<GpuFrameExtractionError>(
      [&] { (void)extract_gpu_gray_frames(backend, skipped, std::vector<std::uint64_t>{1}); },
      "skipped requested", "source frame gaps rejected");

  VectorSource duplicate_source({metadata_frame(1), metadata_frame(1)});
  expect_error<GpuFrameExtractionError>(
      [&] {
        (void)extract_gpu_gray_frames(backend, duplicate_source, std::vector<std::uint64_t>{2});
      },
      "not strictly increasing", "non-monotonic source indices rejected");

  MissingPayloadSource missing_payload;
  expect_error<GpuFrameExtractionError>(
      [&] {
        (void)extract_gpu_gray_frames(backend, missing_payload, std::vector<std::uint64_t>{0});
      },
      "without a frame payload", "frame status requires payload");

  VectorSource empty({});
  const auto result = extract_gpu_gray_frames(backend, empty, std::vector<std::uint64_t>{});
  expect_true(result.empty(), "empty selection returns no frames");
  expect_eq(empty.read_count(), 0U, "empty selection does not consume source");
}

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

bool address_sanitizer_build() {
#if defined(__SANITIZE_ADDRESS__)
  return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
  return true;
#else
  return false;
#endif
#else
  return false;
#endif
}

#if defined(__linux__)

enum class CopyTraceEvent {
  CopySubmitted,
  ContextSynchronized,
  DecoderOwnerReleased,
};

struct CopyOrderTrace final : CudaBackendTraceSink {
  void device_to_device_copy_submitted() noexcept override {
    record(CopyTraceEvent::CopySubmitted);
  }
  void context_synchronized() noexcept override { record(CopyTraceEvent::ContextSynchronized); }
  void decoder_owner_released() noexcept { record(CopyTraceEvent::DecoderOwnerReleased); }

  void record(CopyTraceEvent event) noexcept {
    if (event_count == events.size()) {
      overflow = true;
      return;
    }
    events[event_count++] = event;
  }

  std::array<CopyTraceEvent, 16> events{};
  std::size_t event_count = 0;
  bool overflow = false;
};

struct CudaNvmmOwner {
  explicit CudaNvmmOwner(CudaDeviceBuffer allocation, std::shared_ptr<CopyOrderTrace> trace = {})
      : allocation(std::move(allocation)), trace(std::move(trace)) {}
  ~CudaNvmmOwner() {
    if (trace) {
      trace->decoder_owner_released();
    }
  }

  CudaDeviceBuffer allocation;
  abi::SurfaceParams params;
  abi::Surface surface;
  std::shared_ptr<CopyOrderTrace> trace;
};

std::pair<GpuDecodedFrame, std::weak_ptr<CudaNvmmOwner>>
make_cuda_frame(CudaBackend& backend, std::uint64_t frame_index, std::uint8_t y_value,
                std::uint32_t width = 854, std::uint32_t height = 32,
                std::uint32_t allocation_width = 864, std::shared_ptr<CopyOrderTrace> trace = {}) {
  auto source_allocation = backend.allocate_pitched(allocation_width, height + height / 2U, 16);
  const std::size_t pitch = source_allocation.pitch;
  const std::size_t total_size = pitch * (height + height / 2U);
  auto owner =
      std::make_shared<CudaNvmmOwner>(std::move(source_allocation.buffer), std::move(trace));
  std::vector<std::uint8_t> pixels(total_size, 128);
  std::fill_n(pixels.begin(), pitch * height, y_value);
  backend.copy_host_to_device_2d({.src = pixels.data(),
                                  .src_pitch = pitch,
                                  .dst = owner->allocation.ptr(),
                                  .dst_pitch = pitch,
                                  .width_bytes = pitch,
                                  .height = height + height / 2U});

  auto& params = owner->params;
  params.width = allocation_width;
  params.height = height;
  params.pitch = pitch;
  params.color_format = abi::kColorNv12_709;
  params.layout = abi::kLayoutPitch;
  params.data_size = total_size;
  params.data_ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(owner->allocation.ptr()));
  params.plane_params.num_planes = 2;
  params.plane_params.width[0] = allocation_width;
  params.plane_params.height[0] = height;
  params.plane_params.pitch[0] = pitch;
  params.plane_params.psize[0] = pitch * height;
  params.plane_params.bytes_per_pix[0] = 1;
  params.plane_params.width[1] = allocation_width / 2U;
  params.plane_params.height[1] = height / 2U;
  params.plane_params.pitch[1] = pitch;
  params.plane_params.offset[1] = pitch * height;
  params.plane_params.psize[1] = pitch * (height / 2U);
  params.plane_params.bytes_per_pix[1] = 2;
  owner->surface.gpu_id = 0;
  owner->surface.batch_size = 1;
  owner->surface.num_filled = 1;
  owner->surface.mem_type = abi::kMemCudaDevice;
  owner->surface.surface_list = &params;

  auto info = extract_nvmm_frame_info(&owner->surface, NvbufSurfaceAbi::DeepStream9_1);
  std::weak_ptr<CudaNvmmOwner> lifetime = owner;
  return {GpuDecodedFrame{.nvmm = info,
                          .visible_width = width,
                          .visible_height = height,
                          .owner = std::move(owner),
                          .frame_index = frame_index,
                          .pts_ns = frame_index * 33'333'333U,
                          .duration_ns = 33'333'333U},
          lifetime};
}

void selected_y_planes_are_copied_device_to_device(CudaBackend& backend) {
  auto trace = std::make_shared<CopyOrderTrace>();
  auto extraction_backend = backend.with_trace_sink(trace);
  std::vector<GpuDecodedFrame> frames;
  std::vector<std::weak_ptr<CudaNvmmOwner>> lifetimes;
  for (std::uint64_t index = 0; index < 5; ++index) {
    auto [frame, lifetime] =
        make_cuda_frame(backend, index, static_cast<std::uint8_t>(20 + index), 854, 32, 864, trace);
    frames.push_back(std::move(frame));
    lifetimes.push_back(std::move(lifetime));
  }
  VectorSource source(std::move(frames));
  const auto selected =
      extract_gpu_gray_frames(extraction_backend, source, std::vector<std::uint64_t>{1, 3});
  expect_eq(selected.size(), 2U, "two indexed GPU frames extracted");
  constexpr std::array expected_events{
      CopyTraceEvent::DecoderOwnerReleased, CopyTraceEvent::CopySubmitted,
      CopyTraceEvent::ContextSynchronized,  CopyTraceEvent::DecoderOwnerReleased,
      CopyTraceEvent::DecoderOwnerReleased, CopyTraceEvent::CopySubmitted,
      CopyTraceEvent::ContextSynchronized,  CopyTraceEvent::DecoderOwnerReleased};
  expect_true(!trace->overflow, "CUDA copy ordering trace does not overflow");
  expect_eq(trace->event_count, expected_events.size(), "CUDA copy ordering event count");
  if (trace->event_count == expected_events.size()) {
    expect_true(std::equal(expected_events.begin(), expected_events.end(), trace->events.begin()),
                "D2D copies synchronize before decoder owners are released");
  }
  if (selected.size() != 2) {
    return;
  }

  expect_eq(selected[0].frame_index, 1U, "first selected index preserved");
  expect_eq(selected[1].frame_index, 3U, "second selected index preserved");
  expect_true(selected[0].pts_ns == 33'333'333U, "selected PTS preserved");
  expect_true(selected[1].duration_ns == 33'333'333U, "selected duration preserved");
  expect_eq(selected[0].width, 854U, "visible calibration frame width preserved");
  expect_true(selected[0].pitch >= selected[0].width,
              "calibration copy uses a pitch covering the frame width");
  expect_eq(selected[0].y_plane.size(), selected[0].pitch * selected[0].height,
            "calibration allocation retains its full pitched extent");
  expect_eq(selected[0].view().ptr, selected[0].y_plane.ptr(), "gray view references device copy");

  for (std::size_t i = 0; i < selected.size(); ++i) {
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(selected[i].width) *
                                     selected[i].height);
    backend.copy_device_to_host_2d({.dst = pixels.data(),
                                    .dst_pitch = selected[i].width,
                                    .src = selected[i].y_plane.ptr(),
                                    .src_pitch = selected[i].pitch,
                                    .width_bytes = selected[i].width,
                                    .height = selected[i].height});
    const auto expected = static_cast<std::uint8_t>(selected[i].frame_index + 20);
    expect_true(std::all_of(pixels.begin(), pixels.end(),
                            [expected](std::uint8_t value) { return value == expected; }),
                "selected CUDA Y plane matches source bytes");
  }

  for (std::size_t i = 0; i <= 3; ++i) {
    expect_true(lifetimes[i].expired(), "consumed decoder owner released after D2D copy");
  }
  expect_true(!lifetimes[4].expired(), "unread decoder frame remains owned by source");

  std::vector<GpuDecodedFrame> changing_frames;
  changing_frames.push_back(make_cuda_frame(backend, 0, 1, 64, 32, 64).first);
  changing_frames.push_back(make_cuda_frame(backend, 1, 2, 32, 16, 32).first);
  VectorSource changing_source(std::move(changing_frames));
  expect_error<GpuFrameExtractionError>(
      [&] {
        (void)extract_gpu_gray_frames(backend, changing_source, std::vector<std::uint64_t>{0, 1});
      },
      "dimensions changed", "mid-stream dimension changes are rejected");
}

#endif

} // namespace

int main() {
  if (address_sanitizer_build() && !require_cuda()) {
    std::cerr << "SKIP: CUDA sampling test is skipped under ASan unless explicitly required\n";
    return EXIT_SUCCESS;
  }
  if (!CudaBackend::is_available()) {
    if (require_cuda()) {
      std::cerr << "FAIL: CUDA unavailable: " << CudaBackend::availability_error() << '\n';
      return EXIT_FAILURE;
    }
    std::cerr << "SKIP: CUDA unavailable: " << CudaBackend::availability_error() << '\n';
    return EXIT_SUCCESS;
  }
  auto backend = CudaBackend::create();
  extraction_contract_rejects_invalid_streams(backend);
#if defined(__linux__)
  selected_y_planes_are_copied_device_to_device(backend);
#else
  std::cerr << "SKIP: NvBufSurface CUDA mapping tests require Linux\n";
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
