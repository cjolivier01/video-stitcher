#pragma once

#include <filesystem>
#include <optional>

namespace reco::io {

/// Resolves the deployed GPU video-probe worker for a consumer executable.
///
/// Resolution accepts the explicit `RECO_VIDEO_PROBE_WORKER` override, an
/// executable sibling in installed layouts, the Bazel runfiles tree, and the
/// Bazel output-tree sibling used by local builds. Only an existing absolute
/// executable is returned.
[[nodiscard]] std::optional<std::filesystem::path>
resolve_deployed_video_probe_worker(const std::filesystem::path& executable_path);

} // namespace reco::io
