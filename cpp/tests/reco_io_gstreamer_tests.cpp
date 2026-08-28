#include "fixture_gpu_decode_source.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gstreamer.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace reco::io;
using reco::tests::make_fixture_gpu_file_decode_source;

namespace {

int failures = 0;

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

void expect_true(bool value, std::string_view message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Fn> void expect_invalid_argument(Fn&& fn, std::string_view message) {
  try {
    fn();
    std::cerr << "FAIL: " << message << " did not throw\n";
    ++failures;
  } catch (const std::invalid_argument&) {
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " threw unexpected exception: " << error.what() << '\n';
    ++failures;
  }
}

void pipeline_builders_match_rust_policy() {
  expect_eq(capture_format_name(CaptureFormat::I420), std::string_view("I420"), "I420 name");
  expect_eq(capture_format_name(CaptureFormat::Nv12), std::string_view("NV12"), "NV12 name");

  expect_true(!validate_capture_device("0", CapturePlatform::Jetson).has_value(),
              "Jetson numeric device accepted");
  expect_true(validate_capture_device("/dev/video0", CapturePlatform::Jetson).has_value(),
              "Jetson path device rejected");
  expect_true(!validate_capture_device("/dev/video12", CapturePlatform::LinuxV4l2).has_value(),
              "Linux V4L2 path accepted");
  expect_true(
      validate_capture_device("/dev/video0 ! fakesink", CapturePlatform::LinuxV4l2).has_value(),
      "Linux pipeline injection rejected");
  expect_true(validate_capture_device("camera0", CapturePlatform::Macos).has_value(),
              "macOS nonnumeric device rejected");

  const auto jetson = build_capture_pipeline_string("1", 1920, 1080, 60, CaptureFormat::Nv12,
                                                    CapturePlatform::Jetson);
  expect_true(jetson.find("nvarguscamerasrc sensor-id=1") != std::string::npos,
              "Jetson source selected");
  expect_true(jetson.find("nvvideoconvert compute-hw=1 bl-output=false "
                          "disable-passthrough=true ! "
                          "video/x-raw(memory:NVMM),format=NV12 ! appsink") != std::string::npos,
              "Jetson NVMM NV12 caps reach appsink");
  expect_true(jetson.find("video/x-raw,format=") == std::string::npos,
              "Jetson pipeline never downloads before appsink");

  const auto linux = build_capture_pipeline_string("/dev/video2", 1280, 720, 30,
                                                   CaptureFormat::I420, CapturePlatform::LinuxV4l2);
  expect_true(linux.find("v4l2src device=/dev/video2") != std::string::npos,
              "Linux V4L2 source selected");
  expect_true(linux.find("format=I420") != std::string::npos, "Linux I420 output");

  const auto mac =
      build_capture_pipeline_string("0", 640, 480, 30, CaptureFormat::Nv12, CapturePlatform::Macos);
  expect_true(mac.find("avfvideosrc device-index=0") != std::string::npos, "macOS source selected");

  const auto windows = build_capture_pipeline_string("0", 640, 480, 30, CaptureFormat::Nv12,
                                                     CapturePlatform::Windows);
  expect_true(windows.find("mfvideosrc device-index=0") != std::string::npos,
              "Windows source selected");

  expect_invalid_argument(
      [] {
        (void)build_capture_pipeline_string("/dev/video0", 0, 480, 30, CaptureFormat::I420,
                                            CapturePlatform::LinuxV4l2);
      },
      "zero dimensions rejected");
  expect_invalid_argument(
      [] {
        (void)build_capture_pipeline_string("0 ! fakesink", 640, 480, 30, CaptureFormat::Nv12,
                                            CapturePlatform::Jetson);
      },
      "invalid device rejected before pipeline interpolation");
  expect_invalid_argument(
      [] {
        (void)build_capture_pipeline_string("0", 640, 480, 30, CaptureFormat::I420,
                                            CapturePlatform::Jetson);
      },
      "Jetson capture rejects non-NV12 GPU format");
}

void runtime_probes_are_stable() {
  const auto gst_first = probe_gstreamer_runtime();
  const auto gst_second = probe_gstreamer_runtime();
  expect_eq(gst_first.available, gst_second.available, "GStreamer probe stable");
  if (gst_first.available) {
    expect_true(!gst_first.library.empty(), "GStreamer available library named");
  } else {
    expect_true(!gst_first.error.empty(), "GStreamer unavailable error");
  }

  const auto ds_first = probe_deepstream_runtime();
  const auto ds_second = probe_deepstream_runtime();
  expect_eq(ds_first.available, ds_second.available, "DeepStream probe stable");
  if (ds_first.available) {
    expect_true(!ds_first.library.empty(), "DeepStream available library named");
  } else {
    expect_true(!ds_first.error.empty(), "DeepStream unavailable error");
  }

  const auto nvbuf_first = probe_nvbufsurface_runtime();
  const auto nvbuf_second = probe_nvbufsurface_runtime();
  expect_eq(nvbuf_first.available, nvbuf_second.available, "NvBufSurface probe stable");
  if (nvbuf_first.available) {
    expect_true(!nvbuf_first.library.empty(), "NvBufSurface available library named");
  } else {
    expect_true(!nvbuf_first.error.empty(), "NvBufSurface unavailable error");
  }
}

void gpu_file_decode_pipeline_preserves_nvmm() {
  GpuFileDecodeConfig config{.path = "/data/left video.mp4",
                             .container = gpu_decode_container_for_path("/data/left video.mp4")};
  expect_eq(gpu_decode_codec_name(config.codec), std::string_view("h264"), "GPU decode codec name");
  expect_true(gpu_decode_codec_for_path("clip.hevc") == GpuDecodeCodec::Hevc,
              "HEVC extension selects HEVC parser");
  expect_true(gpu_decode_path_is_elementary_stream("clip.hevc"),
              "HEVC extension selects elementary stream path");
  expect_true(gpu_decode_codec_for_path("clip.mp4") == GpuDecodeCodec::H264,
              "MP4 extension defaults to H264 parser");
  expect_true(!gpu_decode_path_is_elementary_stream("clip.mp4"),
              "MP4 extension selects demuxed path");
  expect_true(gpu_decode_container_for_path("clip.mp4") == GpuDecodeContainer::QuickTime,
              "MP4 extension selects qtdemux");
  expect_true(gpu_decode_container_for_path("clip.mkv") == GpuDecodeContainer::Matroska,
              "MKV extension selects matroskademux");
  expect_true(!gpu_decode_container_for_path("clip.webm").has_value(),
              "unsupported VP8/VP9 WebM container is rejected");
  expect_true(gpu_decode_container_for_path("clip.ts") == GpuDecodeContainer::MpegTs,
              "transport stream extension selects tsdemux");
  const auto pipeline = build_gstreamer_gpu_file_decode_pipeline(config);
  expect_true(pipeline.find("filesrc location=\"/data/left video.mp4\"") != std::string::npos,
              "GPU file source is quoted");
  expect_true(pipeline.find("qtdemux ! capsfilter caps=\"video/x-h264;video/x-h265\" ! "
                            "parsebin ! identity name=display_info silent=true ! "
                            "nvv4l2decoder") != std::string::npos,
              "containerized H264/HEVC video pad and hardware decode selected");
  expect_true(pipeline.find("identity name=display_info silent=true ! nvv4l2decoder") !=
                  std::string::npos,
              "pre-decoder display geometry is retained");
  expect_true(pipeline.find("nvvideoconvert compute-hw=1 bl-output=false "
                            "disable-passthrough=true") != std::string::npos,
              "GPU decode converts block-linear output to pitch-linear NVMM");
  expect_true(pipeline.find("video/x-raw(memory:NVMM),format=NV12") != std::string::npos,
              "NVMM NV12 caps preserved");
  expect_true(pipeline.find("appsink name=sink") != std::string::npos, "appsink selected");
  expect_true(pipeline.find("drop=false") != std::string::npos,
              "offline GPU decode is lossless by default");

  config.path = "/data/left.hevc";
  config.codec = gpu_decode_codec_for_path(config.path);
  config.elementary_stream = gpu_decode_path_is_elementary_stream(config.path);
  const auto hevc = build_gstreamer_gpu_file_decode_pipeline(config);
  expect_true(hevc.find("h265parse ! identity name=display_info silent=true ! nvv4l2decoder") !=
                  std::string::npos,
              "HEVC hardware decode selected");
  expect_true(hevc.find("qtdemux") == std::string::npos, "raw HEVC bypasses qtdemux");

  config.path = "/data/match.mkv";
  config.codec = GpuDecodeCodec::H264;
  config.elementary_stream = gpu_decode_path_is_elementary_stream(config.path);
  config.container = gpu_decode_container_for_path(config.path);
  const auto matroska = build_gstreamer_gpu_file_decode_pipeline(config);
  expect_true(matroska.find("matroskademux ! capsfilter "
                            "caps=\"video/x-h264;video/x-h265\" ! parsebin ! "
                            "identity name=display_info silent=true ! nvv4l2decoder") !=
                  std::string::npos,
              "Matroska H264/HEVC video pad and hardware decode selected");

  config.path = "/data/left-hevc.mp4";
  config.codec = GpuDecodeCodec::Hevc;
  config.container = GpuDecodeContainer::QuickTime;
  const auto hevc_mp4 = build_gstreamer_gpu_file_decode_pipeline(config);
  expect_true(hevc_mp4.find("qtdemux ! capsfilter caps=\"video/x-h264;video/x-h265\" ! "
                            "parsebin ! identity name=display_info silent=true ! "
                            "nvv4l2decoder") != std::string::npos,
              "containerized HEVC uses automatic video parser selection");

  config.path = "/data/left \"quoted\" video.mp4";
  config.elementary_stream = gpu_decode_path_is_elementary_stream(config.path);
  config.container = gpu_decode_container_for_path(config.path);
  const auto quoted = build_gstreamer_gpu_file_decode_pipeline(config);
  expect_true(quoted.find("left \\\"quoted\\\" video.mp4") != std::string::npos,
              "GPU file source quotes are escaped");

  config.path = "left num-buffers=1.mp4";
  config.elementary_stream = gpu_decode_path_is_elementary_stream(config.path);
  config.container = gpu_decode_container_for_path(config.path);
  const auto property_like = build_gstreamer_gpu_file_decode_pipeline(config);
  expect_true(property_like.find("location=\"left num-buffers=1.mp4\"") != std::string::npos,
              "property-looking path remains inside location value");

  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4 ! fakesink",
                                                        .codec = GpuDecodeCodec::H264,
                                                        .elementary_stream = false,
                                                        .container = std::nullopt,
                                                        .max_buffers = 4,
                                                        .drop = false});
      },
      "pipeline injection in file path rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .codec = GpuDecodeCodec::H264,
                                                        .elementary_stream = false,
                                                        .container = std::nullopt,
                                                        .max_buffers = 0,
                                                        .drop = false});
      },
      "zero max buffers rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .read_timeout_ns = 99'999'999ULL});
      },
      "sub-100ms read timeout rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .indexed_fps_numerator = 30U});
      },
      "partial indexed cadence rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .indexed_fps_numerator = 30U,
                                                        .indexed_fps_denominator = 0U});
      },
      "zero indexed cadence denominator rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .indexed_timestamp_multiplicity = 0U});
      },
      "zero indexed timestamp multiplicity rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .indexed_timestamp_multiplicity = 2U});
      },
      "timestamp multiplicity without indexed cadence rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .start_frame_index = 300U});
      },
      "start frame without indexed cadence rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.mp4",
                                                        .container = GpuDecodeContainer::QuickTime,
                                                        .indexed_fps_numerator = 30U,
                                                        .indexed_fps_denominator = 1U,
                                                        .start_frame_index = 300U});
      },
      "start frame without probed stream origin rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({
            .path = "left.mp4",
            .container = GpuDecodeContainer::QuickTime,
            .indexed_stream_time_origin_ns = 766'666'666ULL,
        });
      },
      "stream origin without indexed cadence rejected");
  expect_invalid_argument(
      [] {
        (void)build_gstreamer_gpu_file_decode_pipeline({.path = "left.avi",
                                                        .codec = GpuDecodeCodec::H264,
                                                        .elementary_stream = false,
                                                        .container = std::nullopt,
                                                        .max_buffers = 4,
                                                        .drop = false});
      },
      "unsupported default container rejected");
}

GpuDecodedFrame valid_gpu_frame() {
  return {.nvmm = {.dmabuf_fd = 12,
                   .width = 1920,
                   .height = 1080,
                   .y_offset = 0,
                   .y_pitch = 1920,
                   .uv_offset = 1920 * 1080,
                   .uv_pitch = 1920,
                   .total_size = 1920 * 1080 + 1920 * 540,
                   .surface_ptr = reinterpret_cast<void*>(0x1234),
                   .memory_type = NvmmMemoryType::SurfaceArray,
                   .gpu_id = 0},
          .visible_width = 1920,
          .visible_height = 1080,
          .owner = std::make_shared<int>(1),
          .frame_index = 7,
          .pts_ns = 33'333'333,
          .duration_ns = 33'333'333};
}

void gpu_decode_frame_source_preserves_gpu_residency() {
  auto frame = valid_gpu_frame();
  expect_true(!validate_gpu_decoded_frame(frame).has_value(), "valid NVMM frame accepted");

  auto source = make_fixture_gpu_file_decode_source(
      {.path = "/data/left.mp4", .container = gpu_decode_container_for_path("/data/left.mp4")},
      {frame});
  expect_true(source->gpu_resident(), "fixture source is GPU resident");
  expect_true(source->pipeline().find("video/x-raw(memory:NVMM),format=NV12") != std::string::npos,
              "fixture source exposes NVMM decode pipeline");

  const auto first = source->read();
  expect_true(first.status == GpuDecodeFrameStatus::Frame, "first read returns frame");
  expect_true(first.frame.has_value(), "frame payload present");
  expect_eq(first.frame->frame_index, 7U, "frame index preserved");
  expect_eq(first.frame->nvmm.dmabuf_fd, 12, "DMA-BUF fd preserved");

  const auto eos = source->read();
  expect_true(eos.status == GpuDecodeFrameStatus::EndOfStream, "second read returns EOS");
  expect_true(!eos.frame.has_value(), "EOS has no frame payload");

  frame.nvmm.dmabuf_fd = -1;
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "invalid surface-array fd rejected");
  expect_invalid_argument(
      [&] {
        (void)make_fixture_gpu_file_decode_source(
            {.path = "/data/left.mp4",
             .container = gpu_decode_container_for_path("/data/left.mp4")},
            {frame});
      },
      "fixture source rejects non-GPU frame");

  frame = valid_gpu_frame();
  frame.nvmm.total_size = frame.nvmm.uv_offset;
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "truncated NVMM frame rejected");

  frame = valid_gpu_frame();
  frame.nvmm.uv_offset = 1;
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "overlapping NV12 planes rejected");

  frame = valid_gpu_frame();
  frame.nvmm.surface_ptr = nullptr;
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "null NvBufSurface pointer rejected");

  frame = valid_gpu_frame();
  frame.owner.reset();
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "missing buffer owner rejected");

  frame = valid_gpu_frame();
  frame.visible_width = 0;
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "zero visible width rejected");

  frame = valid_gpu_frame();
  frame.visible_width = frame.nvmm.width + 2;
  expect_true(validate_gpu_decoded_frame(frame).has_value(),
              "visible width beyond NVMM allocation rejected");

  frame = valid_gpu_frame();
  frame.rotation_degrees = 45;
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "unsupported rotation rejected");

  frame = valid_gpu_frame();
  frame.nvmm.uv_pitch += 128;
  expect_true(validate_gpu_decoded_frame(frame).has_value(),
              "unequal CUDA/NPP plane pitches rejected at decode boundary");

  frame = valid_gpu_frame();
  frame.nvmm.width -= 1;
  expect_true(validate_gpu_decoded_frame(frame).has_value(), "odd NV12 width rejected");

  frame = valid_gpu_frame();
  frame.nvmm.memory_type = NvmmMemoryType::CudaDevice;
  frame.nvmm.dmabuf_fd = -1;
  frame.nvmm.cuda_base_ptr = 0x10000000;
  expect_true(!validate_gpu_decoded_frame(frame).has_value(),
              "dGPU CUDA-device frame accepted without DMA-BUF fd");

  frame = valid_gpu_frame();
  frame.nvmm.height = UINT32_MAX;
  frame.nvmm.y_pitch = UINT32_MAX;
  frame.nvmm.uv_pitch = UINT32_MAX;
  frame.nvmm.uv_offset = UINT32_MAX;
  frame.nvmm.total_size = UINT32_MAX;
  expect_true(validate_gpu_decoded_frame(frame).has_value(),
              "max odd height cannot wrap UV row validation");
}

} // namespace

int main() {
  pipeline_builders_match_rust_policy();
  runtime_probes_are_stable();
  gpu_file_decode_pipeline_preserves_nvmm();
  gpu_decode_frame_source_preserves_gpu_residency();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
