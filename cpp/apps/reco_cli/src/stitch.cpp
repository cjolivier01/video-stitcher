#include "stitch.hpp"

#include "reco/core/calibration.hpp"
#include "reco/core/cuda_backend.hpp"
#include "reco/core/cuda_frame.hpp"
#include "reco/core/cuda_rgba_to_nv12.hpp"
#include "reco/core/cuda_stitch_renderer.hpp"
#include "reco/core/nvrtc_compiler.hpp"
#include "reco/core/path.hpp"
#include "reco/io/audio_passthrough.hpp"
#include "reco/io/gpu_decode.hpp"
#include "reco/io/gpu_encode.hpp"
#include "reco/io/gpu_video_probe.hpp"
#include "reco/io/output.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace reco::cli::detail {
namespace {

using namespace reco::io;

constexpr std::uint64_t kProbeTimeoutNs = 120'000'000'000ULL;
constexpr std::size_t kMaximumInputSegments = 4096;

struct ProbedInput {
  std::vector<std::string> paths;
  std::vector<GpuVideoProbe> probes;
};

struct AudioSelection {
  std::vector<std::string> paths;
  std::uint64_t local_start_time_ns = 0;
};

std::vector<std::string> split_input_segments(std::string_view input, std::string_view label) {
  if (input.empty()) {
    throw std::runtime_error(std::string(label) + " input path is empty");
  }
  std::vector<std::string> paths;
  std::size_t begin = 0;
  while (begin <= input.size()) {
    const auto end = input.find(';', begin);
    const auto part =
        input.substr(begin, end == std::string_view::npos ? input.size() - begin : end - begin);
    if (part.empty()) {
      throw std::runtime_error(std::string(label) + " input contains an empty recording segment");
    }
    paths.emplace_back(part);
    if (paths.size() > kMaximumInputSegments) {
      throw std::runtime_error(std::string(label) + " input exceeds the 4096-segment bound");
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return paths;
}

class OutputTransaction {
public:
  explicit OutputTransaction(std::filesystem::path destination)
      : destination_(std::move(destination)) {
    if (destination_.filename().empty()) {
      throw std::runtime_error("stitch output must name a file");
    }
    auto parent = destination_.parent_path();
    if (parent.empty()) {
      parent = ".";
    }
    std::error_code error;
    if (!std::filesystem::is_directory(parent, error) || error) {
      throw std::runtime_error("stitch output parent is not an accessible directory");
    }
#if !defined(_WIN32)
    directory_descriptor_ = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (directory_descriptor_ < 0) {
      throw std::system_error(errno, std::system_category(),
                              "cannot retain stitch output directory");
    }
#endif
    std::random_device random;
    constexpr char hex[] = "0123456789abcdef";
    for (int attempt = 0; attempt < 128; ++attempt) {
      std::string token(32, '0');
      for (auto& digit : token) {
        digit = hex[random() & 0x0fU];
      }
      auto filename = destination_.filename();
      filename += ".tmp." + token;
      temporary_ = parent / filename;
#if defined(_WIN32)
      const HANDLE handle =
          CreateFileW(temporary_.c_str(), GENERIC_WRITE | FILE_READ_ATTRIBUTES,
                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
                      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
      if (handle != INVALID_HANDLE_VALUE) {
        (void)CloseHandle(handle);
        return;
      }
      if (GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                "cannot reserve stitch output temporary");
      }
#else
      temporary_name_ = filename.string();
      descriptor_ = ::openat(directory_descriptor_, temporary_name_.c_str(),
                             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
      if (descriptor_ >= 0) {
        return;
      }
      if (errno != EEXIST) {
        throw std::system_error(errno, std::system_category(),
                                "cannot reserve stitch output temporary");
      }
#endif
    }
    throw std::runtime_error("cannot reserve a unique stitch output temporary");
  }

  OutputTransaction(const OutputTransaction&) = delete;
  OutputTransaction& operator=(const OutputTransaction&) = delete;

  ~OutputTransaction() {
#if defined(_WIN32)
    if (!committed_) {
      std::error_code error;
      std::filesystem::remove(temporary_, error);
    }
#else
    if (!committed_ && directory_descriptor_ >= 0 && !temporary_name_.empty()) {
      (void)::unlinkat(directory_descriptor_, temporary_name_.c_str(), 0);
    }
    if (descriptor_ >= 0) {
      (void)::close(descriptor_);
    }
    if (directory_descriptor_ >= 0) {
      (void)::close(directory_descriptor_);
    }
#endif
  }

  [[nodiscard]] const std::filesystem::path& temporary() const { return temporary_; }
#if !defined(_WIN32)
  [[nodiscard]] int descriptor() const { return descriptor_; }
#endif

  void commit() {
#if defined(_WIN32)
    std::error_code error;
    const auto status = std::filesystem::symlink_status(temporary_, error);
    if (error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::file_size(temporary_, error) == 0 || error) {
      throw std::runtime_error("GPU encoder did not produce a regular non-empty output");
    }
    if (MoveFileExW(temporary_.c_str(), destination_.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
      throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                              "cannot publish completed stitch output");
    }
#else
    errno = 0;
    struct stat descriptor_identity{};
    struct stat path_identity{};
    if (descriptor_ < 0 || directory_descriptor_ < 0 ||
        ::fstat(descriptor_, &descriptor_identity) != 0 ||
        ::fstatat(directory_descriptor_, temporary_name_.c_str(), &path_identity,
                  AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISREG(descriptor_identity.st_mode) || !S_ISREG(path_identity.st_mode) ||
        descriptor_identity.st_size <= 0 || descriptor_identity.st_dev != path_identity.st_dev ||
        descriptor_identity.st_ino != path_identity.st_ino || ::fsync(descriptor_) != 0) {
      throw std::system_error(errno == 0 ? EIO : errno, std::system_category(),
                              "cannot validate completed stitch output");
    }
    const auto destination_name = destination_.filename().string();
    if (::renameat(directory_descriptor_, temporary_name_.c_str(), directory_descriptor_,
                   destination_name.c_str()) != 0) {
      throw std::system_error(errno, std::system_category(),
                              "cannot publish completed stitch output");
    }
    struct stat published_identity{};
    if (::fstatat(directory_descriptor_, destination_name.c_str(), &published_identity,
                  AT_SYMLINK_NOFOLLOW) != 0 ||
        published_identity.st_dev != descriptor_identity.st_dev ||
        published_identity.st_ino != descriptor_identity.st_ino ||
        ::fsync(directory_descriptor_) != 0) {
      throw std::runtime_error("published stitch output identity changed during publication");
    }
#endif
    committed_ = true;
  }

private:
  std::filesystem::path destination_;
  std::filesystem::path temporary_;
#if !defined(_WIN32)
  std::string temporary_name_;
  int directory_descriptor_ = -1;
  int descriptor_ = -1;
#endif
  bool committed_ = false;
};

void reject_output_alias(const std::filesystem::path& output, const std::filesystem::path& input,
                         std::string_view label) {
  std::error_code output_error;
  std::error_code input_error;
  if (std::filesystem::exists(output, output_error) && !output_error &&
      std::filesystem::exists(input, input_error) && !input_error) {
    std::error_code equivalent_error;
    if (std::filesystem::equivalent(output, input, equivalent_error) && !equivalent_error) {
      throw std::runtime_error("stitch output aliases " + std::string(label));
    }
  }
}

GpuFileDecodeConfig decode_config(const std::string& path, const GpuVideoProbe& probe,
                                  std::optional<std::uint64_t> start_frame) {
  GpuFileDecodeConfig config{
      .path = path,
      .codec = gpu_decode_codec_for_path(path),
      .elementary_stream = gpu_decode_path_is_elementary_stream(path),
      .container = gpu_decode_container_for_path(path),
      .max_buffers = 4,
      .drop = false,
      .read_timeout_ns = 30'000'000'000ULL,
  };
  if (probe.indexed_sampling_cadence_verified && probe.first_stream_time_ns.has_value()) {
    config.indexed_fps_numerator = probe.fps_numerator;
    config.indexed_fps_denominator = probe.fps_denominator;
    config.indexed_timestamp_multiplicity = probe.timestamp_multiplicity;
    config.indexed_stream_time_origin_ns = probe.first_stream_time_ns;
    config.start_frame_index = start_frame;
  } else if (start_frame.has_value()) {
    throw std::runtime_error("--start-time requires an input with verified constant cadence");
  }
  return config;
}

ProbedInput probe_input(std::vector<std::string> paths, const std::filesystem::path& worker,
                        std::string_view label) {
  ProbedInput input{.paths = std::move(paths)};
  input.probes.reserve(input.paths.size());
  for (const auto& path : input.paths) {
    input.probes.push_back(
        probe_gpu_video({.path = path,
                         .codec = gpu_decode_codec_for_path(path),
                         .elementary_stream = gpu_decode_path_is_elementary_stream(path),
                         .container = gpu_decode_container_for_path(path)},
                        worker, kProbeTimeoutNs));
  }
  const auto& first = input.probes.front();
  for (std::size_t index = 1; index < input.probes.size(); ++index) {
    const auto& probe = input.probes[index];
    if (probe.fps_numerator != first.fps_numerator ||
        probe.fps_denominator != first.fps_denominator) {
      throw std::runtime_error(std::string(label) +
                               " recording segments must have one constant frame rate");
    }
    if (probe.width != first.width || probe.height != first.height) {
      throw std::runtime_error(std::string(label) +
                               " recording segments must have one frame geometry");
    }
  }
  return input;
}

std::optional<std::uint64_t> exact_total_frames(const ProbedInput& input) {
  std::uint64_t total = 0;
  for (const auto& probe : input.probes) {
    if (!probe.indexed_sampling_cadence_verified || probe.total_frames_is_estimated) {
      return std::nullopt;
    }
    if (probe.total_frames > std::numeric_limits<std::uint64_t>::max() - total) {
      throw std::overflow_error("joined input frame count overflows");
    }
    total += probe.total_frames;
  }
  return total;
}

std::unique_ptr<GpuFileDecodeSource>
open_decode_source(const ProbedInput& input, std::optional<std::uint64_t> start_frame,
                   const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  if (input.paths.size() == 1U) {
    return open_gstreamer_gpu_file_decode_source(
        decode_config(input.paths.front(), input.probes.front(), start_frame), runtime);
  }

  GpuChainedFileDecodeConfig config{.start_frame_index = start_frame};
  config.segments.reserve(input.paths.size());
  for (std::size_t index = 0; index < input.paths.size(); ++index) {
    const auto& probe = input.probes[index];
    config.segments.push_back({.config = decode_config(input.paths[index], probe, std::nullopt),
                               .exact_frame_count = probe.indexed_sampling_cadence_verified &&
                                                            !probe.total_frames_is_estimated
                                                        ? std::optional(probe.total_frames)
                                                        : std::nullopt});
  }
  return open_gstreamer_gpu_chained_file_decode_source(std::move(config), runtime);
}

std::uint64_t timestamp_for_frame(std::uint64_t frame_index, std::uint32_t fps_numerator,
                                  std::uint32_t fps_denominator) {
  const auto whole_seconds = frame_index / fps_numerator;
  const auto remainder = frame_index % fps_numerator;
  constexpr std::uint64_t billion = 1'000'000'000ULL;
  if (whole_seconds > std::numeric_limits<std::uint64_t>::max() / billion / fps_denominator) {
    throw std::overflow_error("stitch output timestamp exceeds the GStreamer time range");
  }
  const auto whole = whole_seconds * billion * fps_denominator;
  const auto fractional_scale = billion * static_cast<std::uint64_t>(fps_denominator);
  return whole + (remainder * fractional_scale) / fps_numerator;
}

std::uint64_t nanoseconds_from_seconds(double seconds, std::string_view label) {
  if (!std::isfinite(seconds) || seconds < 0.0) {
    throw std::runtime_error(std::string(label) + " must be finite and non-negative");
  }
  const long double nanoseconds = static_cast<long double>(seconds) * 1'000'000'000.0L;
  if (nanoseconds > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    throw std::runtime_error(std::string(label) + " exceeds the GStreamer time range");
  }
  return static_cast<std::uint64_t>(std::llround(nanoseconds));
}

AudioSelection select_audio_segments(const ProbedInput& input, std::uint64_t start_time_ns) {
  std::size_t first_segment = 0;
  std::uint64_t local_start = start_time_ns;
  while (first_segment < input.paths.size() &&
         local_start >= input.probes[first_segment].duration_ns) {
    local_start -= input.probes[first_segment].duration_ns;
    ++first_segment;
  }
  if (first_segment >= input.paths.size()) {
    return {};
  }

  AudioSelection selection;
  bool selected_segment_has_audio_container = false;
  for (std::size_t index = first_segment; index < input.paths.size(); ++index) {
    const auto& path = input.paths[index];
    if (!gpu_decode_path_is_elementary_stream(path) && gpu_decode_container_for_path(path)) {
      if (selection.paths.empty()) {
        selected_segment_has_audio_container = index == first_segment;
      }
      selection.paths.push_back(path);
    }
  }
  selection.local_start_time_ns = selected_segment_has_audio_container ? local_start : 0U;
  return selection;
}

std::uint64_t audio_start_time_ns(const StitchCommand& command, std::int64_t sync_offset,
                                  std::uint32_t fps_numerator, std::uint32_t fps_denominator) {
  auto start = nanoseconds_from_seconds(command.start_time.value_or(0.0), "--start-time");
  if (sync_offset >= 0) {
    return start;
  }
  const auto skipped_frames = static_cast<std::uint64_t>(-(sync_offset + 1)) + 1U;
  const auto sync_time = timestamp_for_frame(skipped_frames, fps_numerator, fps_denominator);
  if (sync_time > std::numeric_limits<std::uint64_t>::max() - start) {
    throw std::runtime_error("audio synchronization offset exceeds the GStreamer time range");
  }
  return start + sync_time;
}

std::optional<std::uint64_t> start_frame_index(const StitchCommand& command,
                                               const ProbedInput& input) {
  if (!command.start_time.has_value()) {
    return std::nullopt;
  }
  const auto& probe = input.probes.front();
  const long double frame =
      static_cast<long double>(*command.start_time) * probe.fps_numerator / probe.fps_denominator;
  if (!std::isfinite(*command.start_time) || frame < 0.0L ||
      frame > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    throw std::runtime_error("--start-time is outside the input frame range");
  }
  const auto start = static_cast<std::uint64_t>(std::floor(frame));
  const auto total = exact_total_frames(input);
  if (!total.has_value()) {
    throw std::runtime_error(
        "--start-time requires exact constant-cadence metadata for every input segment");
  }
  if (start >= *total) {
    throw std::runtime_error("--start-time is outside the input frame range");
  }
  return start;
}

std::optional<std::uint64_t> output_frame_limit(const StitchCommand& command,
                                                const GpuVideoProbe& probe) {
  std::optional<std::uint64_t> limit = command.max_frames;
  if (command.end_time.has_value()) {
    const double start = command.start_time.value_or(0.0);
    if (!std::isfinite(*command.end_time) || !std::isfinite(start) || start < 0.0 ||
        *command.end_time <= start) {
      throw std::runtime_error("--end-time must be greater than --start-time");
    }
    const long double frames = static_cast<long double>(*command.end_time - start) *
                               probe.fps_numerator / probe.fps_denominator;
    const auto time_limit = static_cast<std::uint64_t>(std::ceil(frames));
    limit = limit.has_value() ? std::min(*limit, time_limit) : time_limit;
  }
  return limit;
}

void reject_unported_stitch_options(const StitchCommand& command) {
  if (command.no_zero_copy) {
    throw std::runtime_error("C++ stitch refuses --no-zero-copy because it requires CPU frames");
  }
  if (command.model.has_value() || command.events.has_value() || command.trajectory.has_value() ||
      command.panner_config.has_value() || command.panner_preset.has_value() ||
      command.replay.has_value() || command.replay_scale.has_value()) {
    throw std::runtime_error(
        "AI tracking, event output, trajectories, and replay are ported in a later GPU stage");
  }
  if (command.preset.has_value()) {
    throw std::runtime_error("explicit encoder presets are not yet portable across NVIDIA targets");
  }
}

} // namespace

int run_gpu_stitch(const StitchCommand& command, const std::filesystem::path& executable_path,
                   std::ostream& out, std::ostream& err) {
  try {
    reject_unported_stitch_options(command);
    auto left_paths = split_input_segments(command.left, "left");
    auto right_paths = split_input_segments(command.right, "right");
    const auto calibration_path = core::path_from_utf8(command.calibration);
    const auto output_path = core::path_from_utf8(command.output);
    for (const auto& path : left_paths) {
      reject_output_alias(output_path, core::path_from_utf8(path), "a left input segment");
    }
    for (const auto& path : right_paths) {
      reject_output_alias(output_path, core::path_from_utf8(path), "a right input segment");
    }
    reject_output_alias(output_path, calibration_path, "the calibration file");

    std::string calibration_error;
    auto calibration = core::load_match_calibration_file(command.calibration, &calibration_error);
    if (!calibration.has_value()) {
      throw std::runtime_error(calibration_error.empty() ? "invalid calibration JSON"
                                                         : calibration_error);
    }
    calibration->blend_width = command.blend;

    const auto worker = resolve_video_probe_worker(executable_path);
    if (!worker.has_value()) {
      throw std::runtime_error("cannot locate the deployed reco_video_probe_worker executable");
    }
    auto left_input = probe_input(std::move(left_paths), *worker, "left");
    auto right_input = probe_input(std::move(right_paths), *worker, "right");
    const auto& left_probe = left_input.probes.front();
    const auto& right_probe = right_input.probes.front();
    if (left_probe.fps_numerator != right_probe.fps_numerator ||
        left_probe.fps_denominator != right_probe.fps_denominator) {
      throw std::runtime_error("stereo inputs must have the same constant frame rate");
    }
    const auto start_frame = start_frame_index(command, left_input);
    if (start_frame.has_value()) {
      const auto right_total = exact_total_frames(right_input);
      if (!right_total.has_value()) {
        throw std::runtime_error(
            "--start-time requires exact constant-cadence metadata for every right input segment");
      }
      if (*start_frame >= *right_total) {
        throw std::runtime_error("--start-time is outside the right input frame range");
      }
    }
    const auto limit = output_frame_limit(command, left_probe);
    const std::int64_t sync_offset =
        command.sync_offset != 0 ? command.sync_offset : calibration->sync_offset;
    if (const auto sync_error =
            validate_gpu_stereo_decode_config({.sync_offset = sync_offset, .queue_capacity = 4});
        sync_error.has_value()) {
      throw std::runtime_error(*sync_error);
    }
    auto runtime = discover_nvbufsurface_runtime();

    auto backend = core::CudaBackend::create();
    auto renderer = core::CudaStereoStitchRenderer::create({.calibration = *calibration,
                                                            .output_width = command.width,
                                                            .output_height = command.height,
                                                            .device_ordinal = 0},
                                                           backend, core::NvrtcCompiler::create());
    auto rgba_storage =
        backend.allocate_pitched(static_cast<std::size_t>(command.width) * 4U, command.height, 4);
    const core::CudaRgbaFrameView rgba(
        core::CudaPitchedPlaneView(rgba_storage.buffer.ptr(), rgba_storage.buffer.size(),
                                   rgba_storage.pitch, static_cast<std::size_t>(command.width) * 4U,
                                   command.height, backend.primary_context_id(), 0),
        command.width, command.height);
    auto converter = core::CudaRgbaToNv12Converter::create(
        {.width = command.width, .height = command.height}, backend, core::NvrtcCompiler::create());

    const auto codec = parse_codec(command.codec);
    const auto quality = parse_quality(command.quality);
    if (!codec.has_value() || !quality.has_value()) {
      throw std::runtime_error("stitch codec or quality is invalid");
    }
    Format format = format_for_output(command.output);
    if (command.container.has_value()) {
      const auto parsed = parse_format(*command.container);
      if (!parsed.has_value()) {
        throw std::runtime_error("unsupported stitch output container");
      }
      format = *parsed;
    }
    std::optional<std::string> encoder_factory;
    if (command.encoder.has_value()) {
      const auto factory = std::string(gstreamer_hardware_encoder_factory(*codec));
      const auto nvenc_alias = std::string(codec_name(*codec)) + "_nvenc";
      if (*command.encoder != factory && *command.encoder != nvenc_alias) {
        throw std::runtime_error("only the selected NVIDIA hardware encoder is supported");
      }
      encoder_factory = factory;
    }

    std::optional<AudioPassthroughSource> audio;
    const auto audio_selection = select_audio_segments(
        left_input, audio_start_time_ns(command, sync_offset, left_probe.fps_numerator,
                                        left_probe.fps_denominator));
    if (!audio_selection.paths.empty()) {
      audio.emplace(AudioPassthroughSource::open(
          {.paths = audio_selection.paths, .start_time_ns = audio_selection.local_start_time_ns}));
      if (!audio->caps().has_value()) {
        audio.reset();
      }
    }

    OutputTransaction output(output_path);
    GpuEncodeConfig encode_config{
        .output_path = core::path_to_utf8(output.temporary()),
        .width = command.width,
        .height = command.height,
        .fps_numerator = left_probe.fps_numerator,
        .fps_denominator = left_probe.fps_denominator,
        .codec = *codec,
        .quality = *quality,
        .format = format,
        .encoder = std::move(encoder_factory),
        .audio_caps = audio.has_value() ? audio->caps() : std::nullopt,
        .quality_value = command.quality_value,
        .device_ordinal = 0,
        .pool_capacity = 8,
    };
#if !defined(_WIN32)
    encode_config.output_path.clear();
    encode_config.output_descriptor = output.descriptor();
#endif
    auto encoder = GpuVideoEncodeSession::open(std::move(encode_config), runtime);
    auto left = open_decode_source(left_input, start_frame, runtime);
    auto right = open_decode_source(right_input, start_frame, runtime);
    GpuStereoDecodeSession decoder(std::move(left), std::move(right),
                                   {.sync_offset = sync_offset, .queue_capacity = 4});

    std::optional<CompressedAudioPacket> pending_audio;
    bool audio_eos = false;
    const auto forward_audio_before = [&](std::uint64_t video_duration_ns) {
      while (audio.has_value() && !audio_eos) {
        if (!pending_audio.has_value()) {
          auto read = audio->read();
          if (read.status == AudioPassthroughStatus::EndOfStream) {
            audio_eos = true;
            break;
          }
          if (!read.packet.has_value()) {
            throw std::runtime_error("compressed audio source returned an empty packet result");
          }
          pending_audio = std::move(*read.packet);
        }
        const auto timestamp = pending_audio->timestamp_ns();
        if (!timestamp.has_value()) {
          throw std::runtime_error("compressed audio packet has no finite timestamp");
        }
        if (*timestamp >= video_duration_ns) {
          break;
        }
        encoder.submit_audio_packet(std::move(*pending_audio));
        pending_audio.reset();
      }
    };

    const auto started = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;
    while (!limit.has_value() || frames < *limit) {
      auto decoded = decoder.read();
      if (decoded.status == GpuStereoDecodeStatus::EndOfStream) {
        break;
      }
      if (decoded.status == GpuStereoDecodeStatus::Stopped || !decoded.frames.has_value()) {
        throw std::runtime_error("stereo decoder stopped before end-of-stream");
      }
      auto left_frame = map_gpu_decoded_frame_to_cuda_lease(decoded.frames->left);
      auto right_frame = map_gpu_decoded_frame_to_cuda_lease(decoded.frames->right);
      if ((decoded.frames->left.rotation_degrees != 0 &&
           decoded.frames->left.rotation_degrees != 180) ||
          (decoded.frames->right.rotation_degrees != 0 &&
           decoded.frames->right.rotation_degrees != 180)) {
        throw std::runtime_error("90/270-degree stereo input rotation is not supported");
      }
      renderer.render(left_frame.view(), right_frame.view(), rgba,
                      {.flip_left_180 = decoded.frames->left.rotation_degrees == 180,
                       .flip_right_180 = decoded.frames->right.rotation_degrees == 180});
      auto encoded = encoder.acquire_frame();
      converter.convert(rgba, encoded.view());
      const auto pts =
          timestamp_for_frame(frames, left_probe.fps_numerator, left_probe.fps_denominator);
      const auto next_pts =
          timestamp_for_frame(frames + 1U, left_probe.fps_numerator, left_probe.fps_denominator);
      encoder.submit_frame(std::move(encoded), pts, next_pts - pts);
      forward_audio_before(next_pts);
      ++frames;
    }
    decoder.request_stop();
    if (audio.has_value()) {
      audio->request_stop();
    }
    encoder.finish();
    output.commit();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started);
    const auto rate = elapsed.count() > 0.0 ? static_cast<double>(frames) / elapsed.count() : 0.0;
    out << "Stitched " << frames << " frames to " << command.output << " in " << elapsed.count()
        << "s (" << rate << " fps, CUDA/NVMM/NVENC)\n";
    return 0;
  } catch (const std::exception& error) {
    err << "error: " << error.what() << '\n';
    return 2;
  }
}

} // namespace reco::cli::detail
