#include <charconv>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace reco::io::detail {
int run_gpu_video_probe_worker(int parent_liveness_descriptor);
}

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "--reco-video-probe-worker") != 0) {
    return 2;
  }
#if defined(_WIN32)
  if (argc != 2) {
    return 2;
  }
  if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
    return 2;
  }
  return reco::io::detail::run_gpu_video_probe_worker(-1);
#else
  if (argc != 3) {
    return 2;
  }
  const std::string_view value(argv[2]);
  int parent_liveness_descriptor = -1;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parent_liveness_descriptor);
  if (error != std::errc{} || end != value.data() + value.size() ||
      parent_liveness_descriptor < 0) {
    return 2;
  }
  return reco::io::detail::run_gpu_video_probe_worker(parent_liveness_descriptor);
#endif
}
