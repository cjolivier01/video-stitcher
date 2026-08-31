#include "reco/calibrate/pipeline.hpp"
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
  return {.path = "fixture.mp4",
          .container = GpuDecodeContainer::QuickTime,
          .drop = drop,
          .indexed_fps_numerator = std::nullopt,
          .indexed_fps_denominator = std::nullopt,
          .indexed_stream_time_origin_ns = std::nullopt,
          .start_frame_index = std::nullopt};
}

GpuDecodedFrame metadata_frame(std::uint64_t frame_index) {
  return {.nvmm = {.dmabuf_fd = 7,
                   .width = 64,
                   .height = 32,
                   .y_offset = 0,
                   .y_pitch = 64,
                   .y_size = 64 * 32,
                   .uv_offset = 64 * 32,
                   .uv_pitch = 64,
                   .uv_size = 64 * 16,
                   .total_size = 64 * 48,
                   .surface_ptr = reinterpret_cast<void*>(0x1000),
                   .abi = NvbufSurfaceAbi::DeepStream9_1,
                   .memory_type = NvmmMemoryType::SurfaceArray},
          .visible_width = 64,
          .visible_height = 32,
          .owner = std::make_shared<int>(1),
          .frame_index = frame_index,
          .pts_ns = std::nullopt,
          .duration_ns = std::nullopt,
          .rotation_degrees = 0};
}

class VectorSource final : public GpuFileDecodeSource {
public:
  explicit VectorSource(std::vector<GpuDecodedFrame> frames, bool gpu_resident = true,
                        bool drop = false)
      : config_(fixture_config(drop)), pipeline_(build_gstreamer_gpu_file_decode_pipeline(config_)),
        frames_(std::move(frames)), gpu_resident_(gpu_resident) {}

  VectorSource(GpuFileDecodeConfig config, std::vector<GpuDecodedFrame> frames,
               bool gpu_resident = true,
               std::shared_ptr<std::vector<std::uint64_t>> seek_trace = nullptr)
      : config_(std::move(config)), pipeline_(build_gstreamer_gpu_file_decode_pipeline(config_)),
        frames_(std::move(frames)), gpu_resident_(gpu_resident),
        seek_trace_(std::move(seek_trace)) {}

  [[nodiscard]] const GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return pipeline_; }
  [[nodiscard]] bool gpu_resident() const override { return gpu_resident_; }
  void request_stop() noexcept override {}
  [[nodiscard]] GpuDecodeReadResult read() override {
    ++read_count_;
    if (next_ == frames_.size()) {
      return make_gpu_decode_eos();
    }
    return make_gpu_decode_frame(std::move(frames_[next_++]));
  }
  void seek_to_frame(std::uint64_t frame_index) override {
    if (seek_trace_) {
      seek_trace_->push_back(frame_index);
    }
    const auto match = std::find_if(
        frames_.begin() + static_cast<std::ptrdiff_t>(next_), frames_.end(),
        [frame_index](const GpuDecodedFrame& frame) { return frame.frame_index == frame_index; });
    if (match == frames_.end()) {
      throw GpuDecodeError("fixture seek target is unavailable");
    }
    next_ = static_cast<std::size_t>(std::distance(frames_.begin(), match));
  }
  [[nodiscard]] std::size_t read_count() const { return read_count_; }

private:
  GpuFileDecodeConfig config_;
  std::string pipeline_;
  std::vector<GpuDecodedFrame> frames_;
  std::size_t next_ = 0;
  std::size_t read_count_ = 0;
  bool gpu_resident_ = true;
  std::shared_ptr<std::vector<std::uint64_t>> seek_trace_;
};

class MissingPayloadSource final : public GpuFileDecodeSource {
public:
  MissingPayloadSource() : config_(fixture_config()) {}
  [[nodiscard]] const GpuFileDecodeConfig& config() const override { return config_; }
  [[nodiscard]] std::string_view pipeline() const override { return "fixture"; }
  [[nodiscard]] bool gpu_resident() const override { return true; }
  void request_stop() noexcept override {}
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
        (void)extract_gpu_gray_frames_from_file(backend, fixture_config(true),
                                                NvbufSurfaceAbi::DeepStream9_1,
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
                std::uint32_t allocation_width = 864, std::shared_ptr<CopyOrderTrace> trace = {},
                std::uint16_t rotation_degrees = 0, bool patterned = false) {
  auto source_allocation = backend.allocate_pitched(allocation_width, height + height / 2U, 16);
  const std::size_t pitch = source_allocation.pitch;
  const std::size_t total_size = pitch * (height + height / 2U);
  auto owner =
      std::make_shared<CudaNvmmOwner>(std::move(source_allocation.buffer), std::move(trace));
  std::vector<std::uint8_t> pixels(total_size, 128);
  std::fill_n(pixels.begin(), pitch * height, y_value);
  if (patterned) {
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        pixels[static_cast<std::size_t>(y) * pitch + x] =
            static_cast<std::uint8_t>((static_cast<std::uint64_t>(y) * width + x) % 251U);
      }
    }
  }
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
                          .duration_ns = 33'333'333U,
                          .rotation_degrees = rotation_degrees},
          lifetime};
}

void stream_rotation_is_applied_on_device(CudaBackend& backend) {
  constexpr std::uint32_t width = 64;
  constexpr std::uint32_t height = 32;
  auto rotated_frame = make_cuda_frame(backend, 0, 0, width, height, width, {}, 180U, true).first;
  VectorSource rotated_source({std::move(rotated_frame)});
  const auto extracted =
      extract_gpu_gray_frames(backend, rotated_source, std::vector<std::uint64_t>{0});
  expect_eq(extracted.size(), 1U, "180-degree frame is extracted");
  if (!extracted.empty()) {
    expect_eq(extracted[0].applied_rotation_degrees, 180U,
              "extracted frame preserves the applied source rotation");
    expect_eq(extracted[0].view().applied_rotation_degrees, 180U,
              "calibration frame view exposes the applied source rotation");
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height);
    backend.copy_device_to_host_2d({.dst = pixels.data(),
                                    .dst_pitch = width,
                                    .src = extracted[0].y_plane.ptr(),
                                    .src_pitch = extracted[0].pitch,
                                    .width_bytes = width,
                                    .height = height});
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        const auto source_x = width - 1U - x;
        const auto source_y = height - 1U - y;
        const auto expected = static_cast<std::uint8_t>(
            (static_cast<std::uint64_t>(source_y) * width + source_x) % 251U);
        expect_eq(pixels[static_cast<std::size_t>(y) * width + x], expected,
                  "CUDA luma rotation pixel");
      }
    }
  }

  auto quarter_turn = make_cuda_frame(backend, 0, 0, width, height, width, {}, 90U).first;
  VectorSource quarter_turn_source({std::move(quarter_turn)});
  expect_error<GpuFrameExtractionError>(
      [&] {
        (void)extract_gpu_gray_frames(backend, quarter_turn_source, std::vector<std::uint64_t>{0});
      },
      "only 0- or 180-degree", "dimension-swapping rotation fails closed");
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
  expect_true(selected[0].color_range == YuvColorRange::Limited,
              "calibration copy preserves the decoder luma range");
  expect_true(selected[0].view().color_range == YuvColorRange::Limited,
              "gray view preserves the calibration luma range");
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

void sparse_file_sampling_reuses_one_seekable_source(CudaBackend& backend) {
  auto trace = std::make_shared<CopyOrderTrace>();
  auto extraction_backend = backend.with_trace_sink(trace);
  auto config = fixture_config();
  config.indexed_fps_numerator = 30;
  config.indexed_fps_denominator = 1;
  config.indexed_stream_time_origin_ns = 0;
  constexpr std::array<std::uint64_t, 3> indices{3, 300, 9'000};
  std::vector<std::uint64_t> opened_indices;
  auto seek_trace = std::make_shared<std::vector<std::uint64_t>>();

  const GpuFileDecodeSourceOpener opener = [&](GpuFileDecodeConfig sample_config,
                                               NvbufSurfaceAbi abi_value) {
    expect_true(abi_value == NvbufSurfaceAbi::DeepStream9_1,
                "sample opener receives the selected NvBufSurface ABI");
    expect_true(sample_config.start_frame_index.has_value(),
                "sample opener receives an indexed seek target");
    opened_indices.push_back(sample_config.start_frame_index.value_or(0));
    std::vector<GpuDecodedFrame> frames;
    for (const auto frame_index : indices) {
      frames.push_back(make_cuda_frame(backend, frame_index,
                                       static_cast<std::uint8_t>(frame_index % 251U), 64, 32, 64,
                                       trace)
                           .first);
    }
    return std::make_unique<VectorSource>(std::move(sample_config), std::move(frames), true,
                                          seek_trace);
  };

  const auto extracted = extract_gpu_gray_frames_from_file(
      extraction_backend, config, NvbufSurfaceAbi::DeepStream9_1, indices, opener);
  expect_true(opened_indices == std::vector<std::uint64_t>{indices.front()},
              "sparse sampling opens one source at the first absolute index");
  expect_true(*seek_trace == std::vector<std::uint64_t>({indices[1], indices[2]}),
              "one source seeks in place for later sparse samples");
  expect_eq(extracted.size(), indices.size(), "one reused GPU source yields every frame");
  for (std::size_t index = 0; index < extracted.size(); ++index) {
    expect_eq(extracted[index].frame_index, indices[index], "reused source preserves frame index");
  }
  expect_eq(static_cast<std::size_t>(std::count(trace->events.begin(),
                                                trace->events.begin() + trace->event_count,
                                                CopyTraceEvent::CopySubmitted)),
            indices.size(), "reused samples use one device-to-device copy each");

  const GpuFileDecodeSourceOpener ambiguous = [&](GpuFileDecodeConfig sample_config,
                                                  NvbufSurfaceAbi) {
    const auto requested = sample_config.start_frame_index.value_or(0);
    std::vector<GpuDecodedFrame> frames;
    frames.push_back(make_cuda_frame(backend, requested + 1U, 0, 64, 32, 64).first);
    return std::make_unique<VectorSource>(std::move(sample_config), std::move(frames));
  };
  constexpr std::array<std::uint64_t, 1> ambiguous_index{12'000};
  expect_error<GpuFrameExtractionError>(
      [&] {
        (void)extract_gpu_gray_frames_from_file(backend, config, NvbufSurfaceAbi::DeepStream9_1,
                                                ambiguous_index, ambiguous);
      },
      "skipped requested", "seek result that cannot prove the requested index fails closed");
}

void seekable_source_calibration_releases_each_pair(CudaBackend& backend) {
  std::vector<GpuDecodedFrame> left_frames;
  std::vector<GpuDecodedFrame> right_frames;
  std::vector<std::weak_ptr<CudaNvmmOwner>> lifetimes;
  constexpr std::array<std::uint64_t, 3> indices{3, 300, 9'000};
  for (const auto index : indices) {
    auto [left, left_lifetime] = make_cuda_frame(backend, index, 32, 854, 64, 864);
    auto [right, right_lifetime] = make_cuda_frame(backend, index, 224, 854, 64, 864);
    left_frames.push_back(std::move(left));
    right_frames.push_back(std::move(right));
    lifetimes.push_back(std::move(left_lifetime));
    lifetimes.push_back(std::move(right_lifetime));
  }
  auto source_config = fixture_config();
  source_config.indexed_fps_numerator = 30;
  source_config.indexed_fps_denominator = 1;
  source_config.indexed_stream_time_origin_ns = 0;
  source_config.start_frame_index = indices.front();
  auto left_seeks = std::make_shared<std::vector<std::uint64_t>>();
  auto right_seeks = std::make_shared<std::vector<std::uint64_t>>();
  VectorSource left_source(source_config, std::move(left_frames), true, left_seeks);
  VectorSource right_source(source_config, std::move(right_frames), true, right_seeks);
  const CameraParams params{.width = 854,
                            .height = 64,
                            .fx = 1.0e10,
                            .fy = 1.0e10,
                            .cx = 427.0,
                            .cy = 32.0,
                            .d = {0.0, 0.0, 0.0, 0.0}};
  CalibrationConfig config;
  config.num_frames = indices.size();
  config.akaze.max_keypoints = 32;
  config.optimizer.max_iters = 10;

  expect_error<CalibrationExecutionError>(
      [&] {
        (void)run_gpu_calibration_sources(backend, left_source, right_source, indices, indices,
                                          params, params, config);
      },
      "no usable frame pairs", "uniform seeked frames produce no calibration features");
  expect_eq(left_source.read_count(), indices.size(), "left source returns every selected frame");
  expect_eq(right_source.read_count(), indices.size(), "right source returns every selected frame");
  expect_true(*left_seeks == std::vector<std::uint64_t>({indices[1], indices[2]}),
              "left calibration source is reused across sparse samples");
  expect_true(*right_seeks == std::vector<std::uint64_t>({indices[1], indices[2]}),
              "right calibration source is reused across sparse samples");
  expect_true(std::all_of(lifetimes.begin(), lifetimes.end(),
                          [](const auto& lifetime) { return lifetime.expired(); }),
              "seeked calibration releases every decoder-owned surface");
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
  sparse_file_sampling_reuses_one_seekable_source(backend);
  stream_rotation_is_applied_on_device(backend);
  seekable_source_calibration_releases_each_pair(backend);
#else
  std::cerr << "SKIP: NvBufSurface CUDA mapping tests require Linux\n";
#endif
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
