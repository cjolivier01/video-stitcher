#include "base64_fixture.hpp"
#include "reco/io/audio_passthrough.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string_view>

namespace {

using namespace reco::io;

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 128; ++attempt) {
      path_ =
          std::filesystem::temp_directory_path() /
          ("reco-audio-passthrough-" + std::to_string(random()) + "-" + std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
    }
    throw std::runtime_error("cannot create audio passthrough test directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

void verify_restored_path_or_fail_closed(const std::shared_ptr<const StableMediaFile>& source) {
  try {
    source->verify_unchanged();
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find("changed while it was retained") ==
        std::string_view::npos) {
      throw;
    }
  }
}

void run_real_runtime_checks() {
  TemporaryDirectory temporary;
  const auto video_only = temporary.path() / "video-only.mp4";
  const auto audio_only = temporary.path() / "audio-only.mp4";
  reco::tests::materialize_base64_fixture(reco::tests::find_runfile("video_only_mp4.b64"),
                                          video_only);
  reco::tests::materialize_base64_fixture(reco::tests::find_runfile("audio_only_mp4.b64"),
                                          audio_only);

  auto absent = AudioPassthroughSource::open(
      {.segments = {{.path = video_only.string(), .video_duration_ns = 1'000'000'000ULL}},
       .read_timeout = std::chrono::seconds(2)});
  if (absent.caps().has_value() || absent.read().status != AudioPassthroughStatus::EndOfStream) {
    throw std::runtime_error("video-only MP4 did not report clean audio EOS");
  }

  const auto consume_audio = [](AudioPassthroughSource& audio) {
    if (!audio.caps().has_value() || audio.caps()->find("audio/mpeg") == std::string::npos) {
      throw std::runtime_error("AAC MP4 did not retain parser-negotiated compressed caps");
    }
    std::size_t packet_count = 0;
    std::size_t payload_bytes = 0;
    for (;;) {
      const auto result = audio.read();
      if (result.status == AudioPassthroughStatus::EndOfStream) {
        break;
      }
      if (!result.packet.has_value() || result.packet->bytes.empty()) {
        throw std::runtime_error("real compressed audio packet is empty");
      }
      ++packet_count;
      payload_bytes += result.packet->bytes.size();
      if (packet_count > 128U) {
        throw std::runtime_error("real compressed audio source did not reach EOS");
      }
    }
    if (packet_count == 0 || payload_bytes == 0) {
      throw std::runtime_error("real compressed audio source returned no packets");
    }
  };
  {
    auto audio = AudioPassthroughSource::open(
        {.segments = {{.path = audio_only.string(), .video_duration_ns = 1'000'000'000ULL}},
         .read_timeout = std::chrono::seconds(2)});
    consume_audio(audio);
  }

  const auto retained_video_path = temporary.path() / "video-retained.mp4";
  const auto retained_video = StableMediaFile::open(video_only);
  std::filesystem::rename(video_only, retained_video_path);
  std::filesystem::copy_file(audio_only, video_only);
  {
    auto pinned_silence =
        AudioPassthroughSource::open({.segments = {{.path = video_only.string(),
                                                    .stable_source = retained_video,
                                                    .video_duration_ns = 1'000'000'000ULL}},
                                      .read_timeout = std::chrono::seconds(2)});
    if (pinned_silence.caps().has_value() ||
        pinned_silence.read().status != AudioPassthroughStatus::EndOfStream) {
      throw std::runtime_error("descriptor-pinned video-only MP4 did not remain silent");
    }
  }
  std::filesystem::remove(video_only);
  std::filesystem::rename(retained_video_path, video_only);
  verify_restored_path_or_fail_closed(retained_video);

  const auto retained_path = temporary.path() / "audio-retained.mp4";
  const auto retained = StableMediaFile::open(audio_only);
  std::filesystem::rename(audio_only, retained_path);
  std::filesystem::copy_file(video_only, audio_only);
  {
    auto pinned_audio =
        AudioPassthroughSource::open({.segments = {{.path = audio_only.string(),
                                                    .stable_source = retained,
                                                    .video_duration_ns = 1'000'000'000ULL}},
                                      .read_timeout = std::chrono::seconds(2)});
    consume_audio(pinned_audio);
  }
  std::filesystem::remove(audio_only);
  std::filesystem::rename(retained_path, audio_only);
  verify_restored_path_or_fail_closed(retained);
}

} // namespace

int main() {
  try {
    run_real_runtime_checks();
    std::cout << "real compressed audio passthrough checks passed\n";
    return EXIT_SUCCESS;
  } catch (const AudioPassthroughError& error) {
    if (std::string_view(error.what()).find("could not load") != std::string_view::npos) {
      std::cout << "SKIP: GStreamer runtime unavailable: " << error.what() << '\n';
      return EXIT_SUCCESS;
    }
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
