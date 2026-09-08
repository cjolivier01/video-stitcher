#include "base64_fixture.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/core/cuda_frame.hpp"
#include "reco/core/cuda_rgba_to_nv12.hpp"
#include "reco/core/nvrtc_compiler.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gpu_encode.hpp"
#include "reco/io/gstreamer.hpp"
#include "reco/io/nvmm.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using namespace reco::core;
using namespace reco::io;

bool require_cuda() {
  const char* value = std::getenv("RECO_REQUIRE_CUDA_TEST");
  return value != nullptr && std::string_view(value) == "1";
}

std::filesystem::path probe_worker_runfile() {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (workspace == nullptr || workspace[0] == '\0') {
    throw std::runtime_error("TEST_WORKSPACE is not set");
  }
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    throw std::runtime_error("failed to initialize Bazel runfiles: " + error);
  }
#if defined(_WIN32)
  constexpr std::string_view executable = "cpp/reco_io/reco_video_probe_worker.exe";
#else
  constexpr std::string_view executable = "cpp/reco_io/reco_video_probe_worker";
#endif
  const auto logical_path = std::string(workspace) + "/" + std::string(executable);
  const auto resolved = std::filesystem::path(runfiles->Rlocation(logical_path));
  if (resolved.empty() || !std::filesystem::is_regular_file(resolved)) {
    throw std::runtime_error("video probe worker runfile not found");
  }
  return std::filesystem::absolute(resolved);
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 128; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("reco-gpu-encode-" + std::to_string(random()) + "-" +
               std::to_string(static_cast<unsigned int>(attempt)));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
    }
    throw std::runtime_error("cannot create GPU encode test directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

class Trace final : public GpuEncodeTraceSink {
public:
  void surface_allocated() noexcept override { ++allocated; }
  void surface_acquired() noexcept override { ++acquired; }
  void surface_submitted() noexcept override { ++submitted; }
  void surface_released() noexcept override { ++released; }

  std::atomic<std::uint32_t> allocated{0};
  std::atomic<std::uint32_t> acquired{0};
  std::atomic<std::uint32_t> submitted{0};
  std::atomic<std::uint32_t> released{0};
};

std::string availability_error() {
  if (const auto error = CudaBackend::availability_error(); !error.empty()) {
    return "CUDA: " + error;
  }
  if (const auto error = NvrtcCompiler::availability_error(); !error.empty()) {
    return "NVRTC: " + error;
  }
  const auto gstreamer = probe_gstreamer_runtime();
  if (!gstreamer.available) {
    return "GStreamer: " + gstreamer.error;
  }
  try {
    (void)discover_nvbufsurface_runtime();
  } catch (const std::exception& error) {
    return std::string("NvBufSurface: ") + error.what();
  }
  return {};
}

void verify_empty_video_track_is_rejected(const std::filesystem::path& worker) {
  TemporaryDirectory temporary;
  const auto fixture = temporary.path() / "empty-video-track-with-audio.mp4";
  reco::tests::materialize_base64_fixture(
      reco::tests::find_runfile("empty_video_track_with_audio_mp4.b64"), fixture);
  try {
    verify_muxed_gpu_video_output(fixture, Codec::H264, Format::Mp4, worker,
                                  std::chrono::seconds(10));
  } catch (const GpuEncodeError& error) {
    const std::string_view message(error.what());
    if (message.find("found no H.264, HEVC, or AV1 moving-video stream") !=
            std::string_view::npos ||
        message.find("reason not-linked") != std::string_view::npos) {
      return;
    }
    throw std::runtime_error("zero-sample video track returned the wrong verification error: " +
                             std::string(error.what()));
  }
  throw std::runtime_error(
      "video track with zero samples and nonzero audio duration passed output verification");
}

void run_round_trip(const std::filesystem::path& worker) {
  // NVENC rejects sub-minimum macroblock geometries on current discrete GPUs.
  constexpr std::uint32_t width = 320;
  constexpr std::uint32_t height = 180;
  constexpr std::uint32_t frame_count = 12;
  constexpr std::uint64_t duration_ns = 33'333'333ULL;

  TemporaryDirectory temporary;
  const auto output = temporary.path() / "round-trip.mp4";
  const auto audio_input = temporary.path() / "audio-only.mp4";
  reco::tests::materialize_base64_fixture(reco::tests::find_runfile("audio_only_mp4.b64"),
                                          audio_input);
  auto audio = AudioPassthroughSource::open(
      {.segments = {{.path = audio_input.string(), .video_duration_ns = 10'000'000'000ULL}}});
  if (!audio.caps().has_value()) {
    throw std::runtime_error("real AAC fixture did not expose compressed audio caps");
  }
  auto runtime = discover_nvbufsurface_runtime();
  auto trace = std::make_shared<Trace>();
  auto encoder = GpuVideoEncodeSession::open({.output_path = output.string(),
                                              .width = width,
                                              .height = height,
                                              .fps_numerator = 30,
                                              .fps_denominator = 1,
                                              .codec = Codec::H264,
                                              .quality = Quality::Fast,
                                              .format = Format::Mp4,
                                              .audio_caps = *audio.caps(),
                                              .pool_capacity = 8},
                                             runtime, trace);

  std::size_t submitted_audio_packets = 0;
  for (;;) {
    auto result = audio.read();
    if (result.status == AudioPassthroughStatus::EndOfStream) {
      break;
    }
    if (!result.packet.has_value()) {
      throw std::runtime_error("real AAC fixture returned an empty packet result");
    }
    encoder.submit_audio_packet(std::move(*result.packet));
    ++submitted_audio_packets;
  }
  if (submitted_audio_packets == 0) {
    throw std::runtime_error("real AAC fixture returned no compressed packets");
  }

  auto backend = CudaBackend::create();
  const auto context = backend.primary_context_id();
  auto rgba_storage = backend.allocate_pitched(width * 4U, height, 4);
  const CudaRgbaFrameView rgba_view(
      CudaPitchedPlaneView(rgba_storage.buffer.ptr(), rgba_storage.buffer.size(),
                           rgba_storage.pitch, width * 4U, height, context),
      width, height);
  auto converter = CudaRgbaToNv12Converter::create({.width = width, .height = height}, backend,
                                                   NvrtcCompiler::create());
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) * height * 4U);
  for (std::uint32_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        const auto offset = (static_cast<std::size_t>(y) * width + x) * 4U;
        rgba[offset] = static_cast<std::uint8_t>((x + frame_index * 7U) & 0xffU);
        rgba[offset + 1U] = static_cast<std::uint8_t>((y * 3U + frame_index * 5U) & 0xffU);
        rgba[offset + 2U] = static_cast<std::uint8_t>((x + y + frame_index * 11U) & 0xffU);
        rgba[offset + 3U] = 255;
      }
    }
    backend.copy_host_to_device_2d({.src = rgba.data(),
                                    .src_pitch = width * 4U,
                                    .dst = rgba_storage.buffer.ptr(),
                                    .dst_pitch = rgba_storage.pitch,
                                    .width_bytes = width * 4U,
                                    .height = height});
    auto output_frame = encoder.acquire_frame();
    converter.convert(rgba_view, output_frame.view());
    encoder.submit_frame(std::move(output_frame), frame_index * duration_ns, duration_ns);
  }
  encoder.finish();
  verify_muxed_gpu_video_output(output, Codec::H264, Format::Mp4, worker, std::chrono::seconds(10));

  if (!std::filesystem::is_regular_file(output) || std::filesystem::file_size(output) < 1024U) {
    throw std::runtime_error("GPU encoder did not produce a usable output file");
  }
  if (trace->allocated != 8U || trace->acquired != frame_count || trace->submitted != frame_count ||
      trace->released != frame_count) {
    throw std::runtime_error("GPU encode trace did not preserve bounded surface ownership");
  }

  auto decoder = open_gstreamer_gpu_file_decode_source({.path = output.string(),
                                                        .codec = GpuDecodeCodec::H264,
                                                        .elementary_stream = false,
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .max_buffers = 2,
                                                        .read_timeout_ns = 10'000'000'000ULL},
                                                       runtime);
  const auto decoded = decoder->read();
  if (decoded.status != GpuDecodeFrameStatus::Frame || !decoded.frame.has_value() ||
      decoded.frame->visible_width != width || decoded.frame->visible_height != height) {
    throw std::runtime_error("NVDEC did not return the encoded frame geometry");
  }
  const auto mapped = map_gpu_decoded_frame_to_cuda_lease(*decoded.frame);
  if (mapped.view().width() != width || mapped.view().height() != height) {
    throw std::runtime_error("round-trip decode did not remain CUDA/NVMM resident");
  }
  if (mapped.view().color_matrix() != YuvColorMatrix::Bt709 ||
      mapped.view().color_range() != YuvColorRange::Limited) {
    throw std::runtime_error("round-trip decode did not preserve BT.709 limited-range metadata");
  }
  decoder->request_stop();

  auto remuxed_audio = AudioPassthroughSource::open(
      {.segments = {{.path = output.string(), .video_duration_ns = 10'000'000'000ULL}}});
  if (!remuxed_audio.caps().has_value() ||
      remuxed_audio.read().status != AudioPassthroughStatus::Packet) {
    throw std::runtime_error("GPU encoder output did not retain passthrough audio");
  }
}

} // namespace

int main() {
  std::filesystem::path worker;
  try {
    worker = probe_worker_runfile();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: video probe worker resolution: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  const auto unavailable = availability_error();
  if (!unavailable.empty()) {
    if (require_cuda()) {
      std::cerr << "FAIL: required GPU encode runtime unavailable: " << unavailable << '\n';
      return EXIT_FAILURE;
    }
    std::cout << "SKIP: GPU encode runtime unavailable: " << unavailable << '\n';
    return EXIT_SUCCESS;
  }
  try {
    run_round_trip(worker);
    verify_empty_video_track_is_rejected(worker);
    std::cout << "GPU encode/NVDEC round trip passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
