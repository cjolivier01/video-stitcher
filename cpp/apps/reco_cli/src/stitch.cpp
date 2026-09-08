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
#include "reco/io/gpu_memory.hpp"
#include "reco/io/gpu_video_probe.hpp"
#include "reco/io/output.hpp"
#include "reco/io/stable_media_file.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <cerrno>
#include <climits>
#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace reco::cli::detail {

std::uint64_t nanoseconds_from_seconds(double seconds, std::string_view label) {
  if (!std::isfinite(seconds) || seconds < 0.0) {
    throw std::runtime_error(std::string(label) + " must be finite and non-negative");
  }
  const long double nanoseconds = static_cast<long double>(seconds) * 1'000'000'000.0L;
  const long double rounded = std::round(nanoseconds);
  constexpr long double kExclusiveUint64Limit = 18'446'744'073'709'551'616.0L;
  if (!std::isfinite(rounded) || rounded >= kExclusiveUint64Limit) {
    throw std::runtime_error(std::string(label) + " exceeds the GStreamer time range");
  }
  return static_cast<std::uint64_t>(rounded);
}

namespace {

using namespace reco::io;

constexpr auto kProbeTimeout = std::chrono::seconds(120);
constexpr std::uint64_t kProbeTimeoutNs =
    std::chrono::duration_cast<std::chrono::nanoseconds>(kProbeTimeout).count();
constexpr std::size_t kMaximumInputSegments = 4096;
constexpr std::size_t kStitchDecodeSourceCapacity = 4;
constexpr std::size_t kStitchStereoQueueCapacity = 4;
constexpr std::size_t kStitchEncodePoolCapacity = 8;

std::size_t open_descriptor_count() {
#if defined(_WIN32)
  const int limit = _getmaxstdio();
  std::size_t count = 0;
  for (int descriptor = 0; descriptor < limit; ++descriptor) {
    if (_get_osfhandle(descriptor) != -1) {
      ++count;
    }
  }
  return count;
#else
#if defined(__APPLE__)
  constexpr const char* descriptor_directory = "/dev/fd";
#else
  constexpr const char* descriptor_directory = "/proc/self/fd";
#endif
  if (auto* directory = ::opendir(descriptor_directory); directory != nullptr) {
    std::size_t count = 0;
    while (const auto* entry = ::readdir(directory)) {
      if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') {
        ++count;
      }
    }
    (void)::closedir(directory);
    return count;
  }

  struct rlimit limits{};
  if (::getrlimit(RLIMIT_NOFILE, &limits) != 0 || limits.rlim_cur == RLIM_INFINITY ||
      limits.rlim_cur > static_cast<rlim_t>(INT_MAX)) {
    throw std::runtime_error("cannot inspect the process descriptor budget");
  }
  std::size_t count = 0;
  for (int descriptor = 0; descriptor < static_cast<int>(limits.rlim_cur); ++descriptor) {
    if (::fcntl(descriptor, F_GETFD) >= 0 || errno != EBADF) {
      ++count;
    }
  }
  return count;
#endif
}

std::size_t descriptor_limit() {
#if defined(_WIN32)
  return static_cast<std::size_t>(_getmaxstdio());
#else
  struct rlimit limits{};
  if (::getrlimit(RLIMIT_NOFILE, &limits) != 0) {
    throw std::runtime_error("cannot inspect the process descriptor limit");
  }
  return limits.rlim_cur == RLIM_INFINITY
             ? std::numeric_limits<std::size_t>::max()
             : static_cast<std::size_t>(std::min<rlim_t>(
                   limits.rlim_cur, static_cast<rlim_t>(std::numeric_limits<std::size_t>::max())));
#endif
}

struct ProbedInput {
  std::vector<std::string> paths;
  std::vector<std::shared_ptr<const StableMediaFile>> stable_sources;
  std::vector<GpuVideoProbe> probes;
};

struct RetainedInputSources {
  std::vector<std::shared_ptr<const StableMediaFile>> decode;
};

struct AudioSelection {
  std::vector<AudioPassthroughSegment> segments;
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

GpuFileDecodeConfig decode_config(const std::string& path,
                                  std::shared_ptr<const StableMediaFile> stable_source,
                                  const GpuVideoProbe& probe,
                                  std::optional<std::uint64_t> start_frame) {
  GpuFileDecodeConfig config{
      .path = path,
      .stable_source = std::move(stable_source),
      .codec = gpu_decode_codec_for_path(path),
      .elementary_stream = gpu_decode_path_is_elementary_stream(path),
      .container = gpu_decode_container_for_path(path),
      .max_buffers = static_cast<std::uint32_t>(kStitchDecodeSourceCapacity),
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

ProbedInput probe_input(std::vector<std::string> paths, RetainedInputSources sources,
                        const std::filesystem::path& worker, std::string_view label,
                        const CancellationRequested& cancellation_requested) {
  if (paths.size() != sources.decode.size()) {
    throw std::logic_error("stable stitch input count does not match its paths");
  }
  ProbedInput input{.paths = std::move(paths), .stable_sources = std::move(sources.decode)};
  input.probes.reserve(input.paths.size());
  for (std::size_t index = 0; index < input.paths.size(); ++index) {
    const auto& path = input.paths[index];
    auto probe_source = input.stable_sources[index]->open_cursor();
    input.probes.push_back(
        probe_gpu_video({.path = path,
                         .stable_source = std::move(probe_source),
                         .codec = gpu_decode_codec_for_path(path),
                         .elementary_stream = gpu_decode_path_is_elementary_stream(path),
                         .container = gpu_decode_container_for_path(path)},
                        worker, kProbeTimeoutNs, cancellation_requested));
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

void require_exact_indexed_timeline(const ProbedInput& input, std::string_view label) {
  if (!exact_total_frames(input).has_value()) {
    throw std::runtime_error(std::string(label) +
                             " input requires exact verified constant-cadence metadata");
  }
  for (const auto& probe : input.probes) {
    if (!probe.first_stream_time_ns.has_value()) {
      throw std::runtime_error(std::string(label) +
                               " input is missing an indexed stream-time origin");
    }
  }
}

std::unique_ptr<GpuFileDecodeSource>
open_decode_source(const ProbedInput& input, std::optional<std::uint64_t> start_frame,
                   const std::shared_ptr<const NvbufSurfaceRuntime>& runtime) {
  if (input.paths.size() == 1U) {
    return open_gstreamer_gpu_file_decode_source(decode_config(input.paths.front(),
                                                               input.stable_sources.front(),
                                                               input.probes.front(), start_frame),
                                                 runtime);
  }

  GpuChainedFileDecodeConfig config{.start_frame_index = start_frame};
  config.segments.reserve(input.paths.size());
  for (std::size_t index = 0; index < input.paths.size(); ++index) {
    const auto& probe = input.probes[index];
    config.segments.push_back(
        {.config =
             decode_config(input.paths[index], input.stable_sources[index], probe, std::nullopt),
         .exact_frame_count =
             probe.indexed_sampling_cadence_verified && !probe.total_frames_is_estimated
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
  const auto scale = billion * static_cast<std::uint64_t>(fps_denominator);
  if (whole_seconds > std::numeric_limits<std::uint64_t>::max() / scale) {
    throw std::overflow_error("stitch output timestamp exceeds the GStreamer time range");
  }
  const auto whole = whole_seconds * scale;
  const auto fractional_whole = remainder * (scale / fps_numerator);
  const auto fractional_remainder = (remainder * (scale % fps_numerator)) / fps_numerator;
  if (fractional_whole > std::numeric_limits<std::uint64_t>::max() - fractional_remainder) {
    throw std::overflow_error("stitch output timestamp exceeds the GStreamer time range");
  }
  const auto fractional = fractional_whole + fractional_remainder;
  if (whole > std::numeric_limits<std::uint64_t>::max() - fractional) {
    throw std::overflow_error("stitch output timestamp exceeds the GStreamer time range");
  }
  return whole + fractional;
}

std::uint64_t rounded_frames_from_seconds(long double seconds, std::uint32_t fps_numerator,
                                          std::uint32_t fps_denominator, std::string_view label) {
  const long double frames = seconds * static_cast<long double>(fps_numerator) / fps_denominator;
  const long double rounded = std::round(frames);
  constexpr long double kExclusiveUint64Limit = 18'446'744'073'709'551'616.0L;
  if (!std::isfinite(rounded) || rounded < 0.0L || rounded >= kExclusiveUint64Limit) {
    throw std::runtime_error(std::string(label) + " is outside the input frame range");
  }
  return static_cast<std::uint64_t>(rounded);
}

AudioSelection select_audio_segments(const ProbedInput& input, std::uint64_t start_time_ns) {
  const auto segment_duration = [&](std::size_t index) {
    const auto& probe = input.probes[index];
    if (!probe.indexed_sampling_cadence_verified || probe.total_frames_is_estimated) {
      throw std::runtime_error("audio selection requires an exact indexed video timeline");
    }
    return stitch_timeline_duration_ns(probe.total_frames, probe.fps_numerator,
                                       probe.fps_denominator);
  };
  std::size_t first_segment = 0;
  std::uint64_t local_start = start_time_ns;
  while (first_segment < input.paths.size() && local_start >= segment_duration(first_segment)) {
    local_start -= segment_duration(first_segment);
    ++first_segment;
  }
  if (first_segment >= input.paths.size()) {
    return {};
  }

  AudioSelection selection;
  for (std::size_t index = first_segment; index < input.paths.size(); ++index) {
    selection.segments.push_back({.path = input.paths[index],
                                  .stable_source = input.stable_sources[index],
                                  .video_duration_ns = segment_duration(index)});
  }
  selection.local_start_time_ns = local_start;
  return selection;
}

std::uint64_t audio_start_time_ns(std::uint64_t frame_aligned_start_time_ns,
                                  std::int64_t sync_offset, std::uint32_t fps_numerator,
                                  std::uint32_t fps_denominator) {
  if (sync_offset >= 0) {
    return frame_aligned_start_time_ns;
  }
  const auto skipped_frames = static_cast<std::uint64_t>(-(sync_offset + 1)) + 1U;
  const auto sync_time = timestamp_for_frame(skipped_frames, fps_numerator, fps_denominator);
  if (sync_time > std::numeric_limits<std::uint64_t>::max() - frame_aligned_start_time_ns) {
    throw std::runtime_error("audio synchronization offset exceeds the GStreamer time range");
  }
  return frame_aligned_start_time_ns + sync_time;
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

RetainedInputSources retain_media_inputs(const std::vector<std::string>& paths) {
  RetainedInputSources retained;
  retained.decode.reserve(paths.size());
  for (const auto& path : paths) {
    auto source = StableMediaFile::open(core::path_from_utf8(path));
    retained.decode.push_back(std::move(source));
  }
  return retained;
}

void verify_retained_inputs(const ProbedInput& left, const ProbedInput& right,
                            const StableMediaFile& calibration) {
  for (const auto& input : left.stable_sources) {
    input->verify_unchanged();
  }
  for (const auto& input : right.stable_sources) {
    input->verify_unchanged();
  }
  calibration.verify_unchanged();
}

bool cancellation_is_requested(const CancellationRequested& requested) noexcept {
  if (!requested) {
    return false;
  }
  try {
    return requested();
  } catch (...) {
    return true;
  }
}

class StitchCancelled final : public std::runtime_error {
public:
  StitchCancelled() : std::runtime_error("stitch cancelled") {}
};

class StitchCancellationRelay final {
public:
  explicit StitchCancellationRelay(const CancellationRequested& requested) : requested_(requested) {
    if (!requested_) {
      return;
    }
    worker_ = std::jthread([this](std::stop_token stop) {
      while (!stop.stop_requested()) {
        if (cancellation_is_requested(requested_)) {
          observed_.store(true, std::memory_order_release);
          if (auto* decoder = decoder_.load(std::memory_order_acquire); decoder != nullptr) {
            decoder->request_stop();
          }
          if (auto* audio = audio_.load(std::memory_order_acquire); audio != nullptr) {
            audio->request_stop();
          }
          if (auto* encoder = encoder_.load(std::memory_order_acquire); encoder != nullptr) {
            encoder->abort();
          }
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }

  StitchCancellationRelay(const StitchCancellationRelay&) = delete;
  StitchCancellationRelay& operator=(const StitchCancellationRelay&) = delete;

  ~StitchCancellationRelay() {
    worker_.request_stop();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  [[nodiscard]] bool requested() const noexcept {
    return observed_.load(std::memory_order_acquire) || cancellation_is_requested(requested_);
  }

  void throw_if_requested() const {
    if (requested()) {
      throw StitchCancelled();
    }
  }

  void attach(GpuStereoDecodeSession& decoder) noexcept {
    decoder_.store(&decoder, std::memory_order_release);
    if (requested()) {
      decoder.request_stop();
    }
  }

  void attach(AudioPassthroughSource& audio) noexcept {
    audio_.store(&audio, std::memory_order_release);
    if (requested()) {
      audio.request_stop();
    }
  }

  void attach(GpuVideoEncodeSession& encoder) noexcept {
    encoder_.store(&encoder, std::memory_order_release);
    if (requested()) {
      encoder.abort();
    }
  }

private:
  const CancellationRequested& requested_;
  std::atomic<bool> observed_{false};
  std::atomic<GpuStereoDecodeSession*> decoder_{nullptr};
  std::atomic<AudioPassthroughSource*> audio_{nullptr};
  std::atomic<GpuVideoEncodeSession*> encoder_{nullptr};
  std::jthread worker_;
};

} // namespace

std::size_t stitch_descriptor_requirement(std::size_t input_segments) {
  constexpr std::size_t calibration_authority = 1;
  if (input_segments > std::numeric_limits<std::size_t>::max() - calibration_authority -
                           kStitchTransientDescriptorReserve) {
    throw std::overflow_error("stitch input descriptor requirement overflows");
  }
  return input_segments + calibration_authority + kStitchTransientDescriptorReserve;
}

bool stitch_descriptor_budget_fits(std::size_t open_descriptors, std::size_t limit,
                                   std::size_t input_segments) {
  const auto required = stitch_descriptor_requirement(input_segments);
  return open_descriptors <= limit && required <= limit - open_descriptors;
}

void require_stitch_descriptor_budget(std::size_t input_segments) {
  const auto required = stitch_descriptor_requirement(input_segments);
#if defined(_WIN32)
  const auto initial_open = open_descriptor_count();
  constexpr std::size_t kMaximumWindowsCrtDescriptors = 8192;
  if (required <=
      kMaximumWindowsCrtDescriptors - std::min(initial_open, kMaximumWindowsCrtDescriptors)) {
    const auto requested = std::min(kMaximumWindowsCrtDescriptors, initial_open + required);
    if (requested > descriptor_limit()) {
      (void)_setmaxstdio(static_cast<int>(requested));
    }
  }
#endif
  const auto current = open_descriptor_count();
  const auto limit = descriptor_limit();
  if (!stitch_descriptor_budget_fits(current, limit, input_segments)) {
    throw std::runtime_error("stitch requires " + std::to_string(required) +
                             " additional file descriptors for " + std::to_string(input_segments) +
                             " input segments, but only " +
                             std::to_string(current > limit ? 0 : limit - current) +
                             " are available; raise the process descriptor limit or use fewer "
                             "recording segments");
  }
}

std::uint64_t stitch_timeline_duration_ns(std::uint64_t frame_count, std::uint32_t fps_numerator,
                                          std::uint32_t fps_denominator) {
  if (fps_numerator == 0 || fps_denominator == 0) {
    throw std::invalid_argument("stitch source frame rate must be non-zero");
  }
  return timestamp_for_frame(frame_count, fps_numerator, fps_denominator);
}

StitchFrameWindow derive_stitch_frame_window(std::optional<double> start_time,
                                             std::optional<double> end_time,
                                             std::optional<std::uint64_t> max_frames,
                                             std::uint32_t fps_numerator,
                                             std::uint32_t fps_denominator) {
  if (fps_numerator == 0 || fps_denominator == 0) {
    throw std::invalid_argument("stitch source frame rate must be non-zero");
  }
  const double start = start_time.value_or(0.0);
  if (!std::isfinite(start) || start < 0.0) {
    throw std::runtime_error("--start-time must be finite and non-negative");
  }
  StitchFrameWindow window;
  window.start_frame = rounded_frames_from_seconds(static_cast<long double>(start), fps_numerator,
                                                   fps_denominator, "--start-time");
  window.start_time_ns = timestamp_for_frame(window.start_frame, fps_numerator, fps_denominator);
  window.frame_limit = max_frames;
  if (end_time.has_value()) {
    if (!std::isfinite(*end_time) || *end_time <= start) {
      throw std::runtime_error("--end-time must be greater than --start-time");
    }
    const auto time_limit = rounded_frames_from_seconds(
        static_cast<long double>(*end_time) - static_cast<long double>(start), fps_numerator,
        fps_denominator, "--end-time");
    if (time_limit == 0) {
      throw std::runtime_error("--end-time must select at least one output frame");
    }
    window.frame_limit =
        window.frame_limit.has_value() ? std::min(*window.frame_limit, time_limit) : time_limit;
  }
  return window;
}

StitchFrameTiming derive_stitch_frame_timing(std::uint64_t source_frame_index,
                                             std::uint64_t first_source_frame_index,
                                             std::uint32_t fps_numerator,
                                             std::uint32_t fps_denominator) {
  if (fps_numerator == 0 || fps_denominator == 0) {
    throw std::invalid_argument("stitch output frame rate must be non-zero");
  }
  if (source_frame_index < first_source_frame_index) {
    throw std::runtime_error("stitch source frame index moved backwards");
  }
  const auto relative_index = source_frame_index - first_source_frame_index;
  if (relative_index == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("stitch output frame index overflows");
  }
  const auto pts = timestamp_for_frame(relative_index, fps_numerator, fps_denominator);
  const auto next_pts = timestamp_for_frame(relative_index + 1U, fps_numerator, fps_denominator);
  return {.pts_ns = pts, .duration_ns = next_pts - pts};
}

std::optional<std::uint64_t> clip_stitch_audio_duration(std::optional<std::uint64_t> pts_ns,
                                                        std::optional<std::uint64_t> dts_ns,
                                                        std::uint64_t duration_ns,
                                                        std::uint64_t video_duration_ns) {
  const auto presentation_timestamp = pts_ns.has_value() ? pts_ns : dts_ns;
  if (!presentation_timestamp.has_value()) {
    throw std::runtime_error("compressed audio packet has no finite presentation timestamp");
  }
  if (*presentation_timestamp >= video_duration_ns) {
    return std::nullopt;
  }
  if (duration_ns == 0) {
    throw std::runtime_error(
        "compressed audio packet duration is unknown; cannot bound stream-copy output");
  }
  return std::min(duration_ns, video_duration_ns - *presentation_timestamp);
}

int run_gpu_stitch(const StitchCommand& command, const std::filesystem::path& executable_path,
                   std::ostream& out, std::ostream& err,
                   const CancellationRequested& cancellation_requested) {
  try {
    if (cancellation_is_requested(cancellation_requested)) {
      throw StitchCancelled();
    }
    reject_unported_stitch_options(command);
    auto left_paths = split_input_segments(command.left, "left");
    auto right_paths = split_input_segments(command.right, "right");
    if (left_paths.size() > std::numeric_limits<std::size_t>::max() - right_paths.size()) {
      throw std::overflow_error("combined stitch input segment count overflows");
    }
    require_stitch_descriptor_budget(left_paths.size() + right_paths.size());
    const auto calibration_path = core::path_from_utf8(command.calibration);
    const auto output_path = core::path_from_utf8(command.output);
    auto left_sources = retain_media_inputs(left_paths);
    auto right_sources = retain_media_inputs(right_paths);
    auto calibration_source = StableMediaFile::open(calibration_path);
    std::vector<AtomicOutputProtectedPath> protected_paths;
    protected_paths.reserve(left_paths.size() + right_paths.size() + 1U);
    for (const auto& path : left_paths) {
      const auto index = protected_paths.size();
      protected_paths.push_back({.path = core::path_from_utf8(path),
                                 .label = "a left input segment",
                                 .stable_source = left_sources.decode[index]});
    }
    const auto right_protected_offset = protected_paths.size();
    for (const auto& path : right_paths) {
      const auto index = protected_paths.size() - right_protected_offset;
      protected_paths.push_back({.path = core::path_from_utf8(path),
                                 .label = "a right input segment",
                                 .stable_source = right_sources.decode[index]});
    }
    protected_paths.push_back({.path = calibration_path,
                               .label = "the calibration file",
                               .stable_source = calibration_source});
    AtomicOutputFile output(output_path, {}, {}, protected_paths);
    std::optional<AudioPassthroughSource> audio;
    std::optional<GpuVideoEncodeSession> encoder;
    std::optional<GpuStereoDecodeSession> decoder;
    StitchCancellationRelay cancellation(cancellation_requested);
    cancellation.throw_if_requested();

    auto calibration = core::parse_match_calibration_json(
        calibration_source->read_all(core::kMaxCalibrationFileSize));
    if (!calibration.has_value()) {
      throw std::runtime_error("invalid calibration JSON");
    }
    calibration->blend_width = command.blend;

    const auto worker = resolve_video_probe_worker(executable_path);
    if (!worker.has_value()) {
      throw std::runtime_error("cannot locate the deployed reco_video_probe_worker executable");
    }
    auto left_input = probe_input(std::move(left_paths), std::move(left_sources), *worker, "left",
                                  cancellation_requested);
    auto right_input = probe_input(std::move(right_paths), std::move(right_sources), *worker,
                                   "right", cancellation_requested);
    cancellation.throw_if_requested();
    require_exact_indexed_timeline(left_input, "left");
    require_exact_indexed_timeline(right_input, "right");
    const auto& left_probe = left_input.probes.front();
    const auto& right_probe = right_input.probes.front();
    if (left_probe.fps_numerator != right_probe.fps_numerator ||
        left_probe.fps_denominator != right_probe.fps_denominator) {
      throw std::runtime_error("stereo inputs must have the same constant frame rate");
    }
    const auto window =
        derive_stitch_frame_window(command.start_time, command.end_time, command.max_frames,
                                   left_probe.fps_numerator, left_probe.fps_denominator);
    const auto start_frame =
        command.start_time.has_value() ? std::optional(window.start_frame) : std::nullopt;
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
    const auto limit = window.frame_limit;
    const std::int64_t sync_offset =
        command.sync_offset != 0 ? command.sync_offset : calibration->sync_offset;
    if (const auto sync_error = validate_gpu_stereo_decode_config(
            {.sync_offset = sync_offset, .queue_capacity = kStitchStereoQueueCapacity});
        sync_error.has_value()) {
      throw std::runtime_error(*sync_error);
    }
    auto runtime = discover_nvbufsurface_runtime();
    cancellation.throw_if_requested();

    auto backend = core::CudaBackend::create();
    cancellation.throw_if_requested();
    const auto memory_estimate = estimate_gpu_stitch_memory({
        .output_width = command.width,
        .output_height = command.height,
        .left_width = left_probe.width,
        .left_height = left_probe.height,
        .right_width = right_probe.width,
        .right_height = right_probe.height,
        .decode_source_capacity = kStitchDecodeSourceCapacity,
        .stereo_queue_capacity = kStitchStereoQueueCapacity,
        .encode_pool_capacity = kStitchEncodePoolCapacity,
    });
    require_gpu_memory_preflight(
        evaluate_gpu_memory_preflight(memory_estimate.total_bytes, backend.memory_info(0)),
        "GPU stitch pipeline allocation");
    auto renderer = core::CudaStereoStitchRenderer::create({.calibration = *calibration,
                                                            .output_width = command.width,
                                                            .output_height = command.height,
                                                            .device_ordinal = 0},
                                                           backend, core::NvrtcCompiler::create());
    cancellation.throw_if_requested();
    auto rgba_storage =
        backend.allocate_pitched(static_cast<std::size_t>(command.width) * 4U, command.height, 4);
    const core::CudaRgbaFrameView rgba(
        core::CudaPitchedPlaneView(
            backend.retain_device_span(rgba_storage.buffer.ptr(), rgba_storage.buffer.size(),
                                       core::CudaSpanAccess::ReadWrite),
            rgba_storage.pitch, static_cast<std::size_t>(command.width) * 4U, command.height),
        command.width, command.height);
    auto converter = core::CudaRgbaToNv12Converter::create(
        {.width = command.width, .height = command.height}, backend, core::NvrtcCompiler::create());
    cancellation.throw_if_requested();

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

    const auto audio_selection = select_audio_segments(
        left_input, audio_start_time_ns(window.start_time_ns, sync_offset, left_probe.fps_numerator,
                                        left_probe.fps_denominator));
    if (!audio_selection.segments.empty()) {
      audio.emplace(
          AudioPassthroughSource::open({.segments = audio_selection.segments,
                                        .start_time_ns = audio_selection.local_start_time_ns}));
      if (!audio->caps().has_value()) {
        audio.reset();
      } else {
        cancellation.attach(*audio);
      }
    }
    cancellation.throw_if_requested();

    GpuEncodeConfig encode_config{
        .output_path = {},
        .output_descriptor = output.descriptor(),
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
        .pool_capacity = kStitchEncodePoolCapacity,
    };
    encoder.emplace(GpuVideoEncodeSession::open(std::move(encode_config), runtime));
    cancellation.attach(*encoder);
    cancellation.throw_if_requested();
    auto left = open_decode_source(left_input, start_frame, runtime);
    cancellation.throw_if_requested();
    auto right = open_decode_source(right_input, start_frame, runtime);
    cancellation.throw_if_requested();
    decoder.emplace(std::move(left), std::move(right),
                    GpuStereoDecodeConfig{.sync_offset = sync_offset,
                                          .queue_capacity = kStitchStereoQueueCapacity});
    cancellation.attach(*decoder);
    cancellation.throw_if_requested();

    std::optional<CompressedAudioPacket> pending_audio;
    bool audio_eos = false;
    const auto forward_audio_before = [&](std::uint64_t video_duration_ns, bool final_boundary) {
      while (audio.has_value() && !audio_eos) {
        cancellation.throw_if_requested();
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
        if (!final_boundary && pending_audio->duration_ns == 0) {
          break;
        }
        const auto clipped_duration =
            clip_stitch_audio_duration(pending_audio->pts_ns, pending_audio->dts_ns,
                                       pending_audio->duration_ns, video_duration_ns);
        if (!clipped_duration.has_value()) {
          break;
        }
        if (!final_boundary && *clipped_duration != pending_audio->duration_ns) {
          break;
        }
        pending_audio->duration_ns = *clipped_duration;
        encoder->submit_audio_packet(std::move(*pending_audio));
        pending_audio.reset();
      }
    };

    const auto started = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;
    std::uint64_t video_duration_ns = 0;
    std::optional<std::uint64_t> first_source_frame_index;
    std::optional<std::uint64_t> previous_source_frame_index;
    while (!limit.has_value() || frames < *limit) {
      cancellation.throw_if_requested();
      auto decoded = decoder->read();
      if (decoded.status == GpuStereoDecodeStatus::EndOfStream) {
        break;
      }
      if (decoded.status == GpuStereoDecodeStatus::Stopped || !decoded.frames.has_value()) {
        cancellation.throw_if_requested();
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
      auto encoded = encoder->acquire_frame();
      auto batch = renderer.begin_batch();
      renderer.enqueue(batch, left_frame.view(), right_frame.view(), rgba,
                       {.flip_left_180 = decoded.frames->left.rotation_degrees == 180,
                        .flip_right_180 = decoded.frames->right.rotation_degrees == 180});
      converter.enqueue(batch, rgba, encoded.view());
      batch.wait();
      const auto source_frame_index = decoded.frames->left.frame_index;
      if (previous_source_frame_index.has_value() &&
          source_frame_index <= *previous_source_frame_index) {
        throw std::runtime_error("stereo decoder returned a non-increasing source frame index");
      }
      if (!first_source_frame_index.has_value()) {
        first_source_frame_index = source_frame_index;
      }
      const auto timing =
          derive_stitch_frame_timing(source_frame_index, *first_source_frame_index,
                                     left_probe.fps_numerator, left_probe.fps_denominator);
      encoder->submit_frame(std::move(encoded), timing.pts_ns, timing.duration_ns);
      video_duration_ns = timing.pts_ns + timing.duration_ns;
      forward_audio_before(video_duration_ns, false);
      previous_source_frame_index = source_frame_index;
      ++frames;
    }
    decoder->request_stop();
    if (frames == 0) {
      encoder->abort();
      if (audio.has_value()) {
        audio->request_stop();
      }
      throw std::runtime_error("stereo inputs produced no aligned video frames");
    }
    forward_audio_before(video_duration_ns, true);
    if (audio.has_value()) {
      audio->request_stop();
    }
    cancellation.throw_if_requested();
    encoder->finish();
    cancellation.throw_if_requested();
    verify_muxed_gpu_video_output(output.verification_source(), *codec, format, *worker,
                                  kProbeTimeout, cancellation_requested);
    cancellation.throw_if_requested();
    verify_retained_inputs(left_input, right_input, *calibration_source);
    cancellation.throw_if_requested();
    output.commit();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started);
    const auto rate = elapsed.count() > 0.0 ? static_cast<double>(frames) / elapsed.count() : 0.0;
    out << "Stitched " << frames << " frames to " << command.output << " in " << elapsed.count()
        << "s (" << rate << " fps, CUDA/NVMM/NVENC)\n";
    return 0;
  } catch (const StitchCancelled&) {
    err << "cancelled\n";
    return kCancelledExitCode;
  } catch (const std::exception& error) {
    if (cancellation_is_requested(cancellation_requested)) {
      err << "cancelled\n";
      return kCancelledExitCode;
    }
    err << "error: " << error.what() << '\n';
    return 2;
  }
}

} // namespace reco::cli::detail
