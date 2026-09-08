#include "reco/io/gpu_encode.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace reco::io;

int failures = 0;

void expect_true(bool value, std::string_view message) {
  if (!value) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

GpuEncodeConfig valid_config() {
  return {
      .output_path = "output.mp4",
      .width = 1920,
      .height = 1080,
      .fps_numerator = 30'000,
      .fps_denominator = 1'001,
  };
}

void validation_rejects_unsafe_contracts() {
  auto config = valid_config();
  expect_true(!validate_gpu_encode_config(config).has_value(), "valid config is accepted");

  config.output_path.clear();
  expect_true(validate_gpu_encode_config(config).has_value(), "empty output is rejected");
  config.output_descriptor = 7;
  expect_true(!validate_gpu_encode_config(config).has_value(), "descriptor output is accepted");
  auto descriptor_pipeline = build_gstreamer_gpu_encode_pipeline(config);
  expect_true(descriptor_pipeline.find("fdsink fd=7") != std::string::npos &&
                  descriptor_pipeline.find("filesink") == std::string::npos,
              "descriptor output uses fdsink without reopening a path");
  config.output_path = "output.mp4";
  expect_true(validate_gpu_encode_config(config).has_value(), "ambiguous dual output is rejected");
  config.output_path.clear();
  config.output_descriptor = -1;
  expect_true(validate_gpu_encode_config(config).has_value(), "negative descriptor is rejected");
  config = valid_config();
  config.width = 1919;
  expect_true(validate_gpu_encode_config(config).has_value(), "odd width is rejected");
  config = valid_config();
  config.height = 0;
  expect_true(validate_gpu_encode_config(config).has_value(), "zero height is rejected");
  config = valid_config();
  config.fps_denominator = 0;
  expect_true(validate_gpu_encode_config(config).has_value(),
              "zero frame-rate denominator is rejected");
  config = valid_config();
  config.fps_numerator = 1001;
  config.fps_denominator = 1;
  expect_true(validate_gpu_encode_config(config).has_value(), "excessive frame rate is rejected");
  config = valid_config();
  config.device_ordinal = 1;
  expect_true(validate_gpu_encode_config(config).has_value(),
              "nonzero device is rejected before NVMM allocation");
  config = valid_config();
  config.pool_capacity = 7;
  expect_true(validate_gpu_encode_config(config).has_value(), "undersized pool is rejected");
  config = valid_config();
  config.pool_capacity = 17;
  expect_true(validate_gpu_encode_config(config).has_value(), "oversized pool is rejected");
  config = valid_config();
  config.acquire_timeout = std::chrono::milliseconds(0);
  expect_true(validate_gpu_encode_config(config).has_value(), "zero timeout is rejected");
  config = valid_config();
  config.encoder = "libx264";
  expect_true(validate_gpu_encode_config(config).has_value(),
              "software encoder override is rejected");
  config = valid_config();
  config.codec = Codec::HEVC;
  config.encoder = "nvv4l2h264enc";
  expect_true(validate_gpu_encode_config(config).has_value(),
              "codec-mismatched hardware encoder is rejected");
  config = valid_config();
  config.audio_caps = "";
  expect_true(validate_gpu_encode_config(config).has_value(), "empty audio caps are rejected");
}

void pipeline_is_nvmm_and_hardware_only() {
  auto config = valid_config();
  config.output_path = "quoted \\\" output.mp4";
  config.pool_capacity = 8;
  config.quality = Quality::High;
  config.quality_value = 90;
  const auto pipeline = build_gstreamer_gpu_encode_pipeline(config);
  expect_true(pipeline.find("appsrc name=source") != std::string::npos,
              "pipeline has named appsrc");
  expect_true(pipeline.find("video/x-raw(memory:NVMM)") != std::string::npos,
              "pipeline requires NVMM caps");
  expect_true(pipeline.find("format=(string)NV12") != std::string::npos, "pipeline requires NV12");
  expect_true(pipeline.find("colorimetry=(string)bt709") != std::string::npos,
              "pipeline signals BT.709 limited-range colorimetry");
  expect_true(pipeline.find("chroma-site=(string)mpeg2") != std::string::npos,
              "pipeline signals NV12 chroma siting");
  expect_true(pipeline.find("max-size-buffers=8") != std::string::npos,
              "pipeline queue matches bounded pool");
  expect_true(pipeline.find("nvv4l2h264enc") != std::string::npos,
              "pipeline selects NVIDIA H.264 encoder");
  expect_true(pipeline.find("h264parse") != std::string::npos, "pipeline parses encoded H.264");
  expect_true(pipeline.find("mp4mux") != std::string::npos, "pipeline selects MP4 muxer");
  expect_true(pipeline.find("filesink") != std::string::npos, "pipeline terminates in filesink");
  expect_true(pipeline.find("libx264") == std::string::npos &&
                  pipeline.find("videoconvert") == std::string::npos,
              "pipeline contains no software pixel or encoder fallback");
  expect_true(pipeline.find("location=\"quoted \\\\\\\" output.mp4\"") != std::string::npos,
              "pipeline quotes the output property");
}

void codec_and_container_factories_are_explicit() {
  auto config = valid_config();
  config.codec = Codec::HEVC;
  config.format = Format::Mkv;
  auto pipeline = build_gstreamer_gpu_encode_pipeline(config);
  expect_true(pipeline.find("nvv4l2h265enc") != std::string::npos, "HEVC uses NVIDIA encoder");
  expect_true(pipeline.find("h265parse") != std::string::npos, "HEVC parser is present");
  expect_true(pipeline.find("matroskamux") != std::string::npos, "Matroska muxer is selected");

  config.codec = Codec::AV1;
  config.format = Format::Mp4Fragmented;
  pipeline = build_gstreamer_gpu_encode_pipeline(config);
  expect_true(pipeline.find("nvv4l2av1enc") != std::string::npos, "AV1 uses NVIDIA encoder");
  expect_true(pipeline.find("av1parse") != std::string::npos, "AV1 parser is present");
  expect_true(pipeline.find("fragment-duration=1000") != std::string::npos,
              "fragmented MP4 is explicit");
}

void compressed_audio_is_stream_copied_through_a_bounded_mux_branch() {
  auto config = valid_config();
  config.audio_caps = "audio/mpeg, mpegversion=(int)4, rate=(int)48000, channels=(int)2";
  const auto pipeline = build_gstreamer_gpu_encode_pipeline(config);
  expect_true(pipeline.find("appsrc name=audio_source") != std::string::npos,
              "audio passthrough uses a dedicated appsrc");
  expect_true(pipeline.find("max-buffers=32") != std::string::npos,
              "audio passthrough queue is bounded");
  expect_true(pipeline.find("max-bytes=33554432") != std::string::npos,
              "audio appsrc has a bounded aggregate byte budget");
  expect_true(pipeline.find("audio/mpeg") != std::string::npos,
              "negotiated compressed audio caps are retained");
  expect_true(pipeline.find("audioconvert") == std::string::npos &&
                  pipeline.find("audioresample") == std::string::npos &&
                  pipeline.find("voaacenc") == std::string::npos,
              "audio passthrough does not decode or re-encode packets");

  const auto source = build_gstreamer_audio_passthrough_pipeline("camera segment.mp4");
  expect_true(source.find("qtdemux") != std::string::npos,
              "audio source selects the container demuxer");
  expect_true(source.find("parsebin") != std::string::npos &&
                  source.find("appsink name=audio_sink") != std::string::npos,
              "audio source stops after compressed parsing");
  expect_true(source.find("decodebin") == std::string::npos,
              "audio source never decodes compressed packets");

  AudioPassthroughConfig passthrough{
      .segments = {{.path = "left.mp4", .video_duration_ns = 10'000'000'000ULL},
                   {.path = "left-2.mp4", .video_duration_ns = 10'000'000'000ULL}},
      .start_time_ns = 2'000'000'000ULL};
  expect_true(!validate_audio_passthrough_config(passthrough).has_value(),
              "chained container audio is accepted");
  passthrough.segments = {{.path = "left.h264", .video_duration_ns = 10'000'000'000ULL}};
  expect_true(!validate_audio_passthrough_config(passthrough).has_value(),
              "elementary video is retained as a silent timeline segment");
}

} // namespace

int main() {
  validation_rejects_unsafe_contracts();
  pipeline_is_nvmm_and_hardware_only();
  codec_and_container_factories_are_explicit();
  compressed_audio_is_stream_copied_through_a_bounded_mux_branch();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "all tests passed\n";
  return EXIT_SUCCESS;
}
