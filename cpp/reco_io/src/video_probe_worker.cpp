#include "reco/io/video_probe_worker.hpp"

#include "reco/core/path.hpp"
#include "rules_cc/cc/runfiles/runfiles.h"

#include <memory>
#include <string>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace reco::io {
namespace {

using rules_cc::cc::runfiles::Runfiles;

#if defined(_WIN32)
constexpr std::string_view kProbeWorkerName = "reco_video_probe_worker.exe";
constexpr std::filesystem::path::value_type kPathSeparator = L';';
#else
constexpr std::string_view kProbeWorkerName = "reco_video_probe_worker";
constexpr std::filesystem::path::value_type kPathSeparator = ':';
#endif

std::optional<std::filesystem::path>
existing_absolute_executable(const std::filesystem::path& path) {
  std::error_code error;
  if (path.empty() || !std::filesystem::is_regular_file(path, error) || error) {
    return std::nullopt;
  }
#if !defined(_WIN32)
  if (::access(path.c_str(), X_OK) != 0) {
    return std::nullopt;
  }
#endif
  auto absolute = std::filesystem::absolute(path, error);
  if (error) {
    return std::nullopt;
  }
  return absolute.lexically_normal();
}

std::optional<std::filesystem::path>
resolve_path_invocation(const std::filesystem::path& executable_path) {
  if (executable_path.empty()) {
    return std::nullopt;
  }
  if (executable_path.is_absolute() || executable_path.has_parent_path()) {
    return existing_absolute_executable(executable_path);
  }

  const auto path_value = core::path_from_environment("PATH");
  if (!path_value.has_value()) {
    return std::nullopt;
  }
  const auto& path = path_value->native();
  std::size_t begin = 0;
  while (begin <= path.size()) {
    const auto end = path.find(kPathSeparator, begin);
    const auto component = path.substr(begin, end - begin);
    const auto directory =
        component.empty() ? std::filesystem::path(".") : std::filesystem::path(component);
    if (auto resolved = existing_absolute_executable(directory / executable_path);
        resolved.has_value()) {
      return resolved;
    }
#if defined(_WIN32)
    if (!executable_path.has_extension()) {
      auto with_extension = executable_path;
      with_extension += ".exe";
      if (auto resolved = existing_absolute_executable(directory / with_extension);
          resolved.has_value()) {
        return resolved;
      }
    }
#endif
    if (end == std::filesystem::path::string_type::npos) {
      break;
    }
    begin = end + 1U;
  }
  return std::nullopt;
}

} // namespace

std::optional<std::filesystem::path>
resolve_deployed_video_probe_worker(const std::filesystem::path& executable_path) {
  if (const auto configured = core::path_from_environment("RECO_VIDEO_PROBE_WORKER");
      configured.has_value()) {
    return existing_absolute_executable(*configured);
  }

  const auto resolved_executable = resolve_path_invocation(executable_path);
  if (resolved_executable.has_value()) {
    if (auto sibling =
            existing_absolute_executable(resolved_executable->parent_path() / kProbeWorkerName);
        sibling.has_value()) {
      return sibling;
    }
  }

  std::string runfiles_error;
  const auto runfiles_argv0 = core::path_to_utf8(resolved_executable.value_or(executable_path));
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(runfiles_argv0, BAZEL_CURRENT_REPOSITORY, &runfiles_error));
  if (runfiles != nullptr) {
    const auto logical_path =
        std::string("reco_video_stitcher/cpp/reco_io/") + std::string(kProbeWorkerName);
    if (auto resolved =
            existing_absolute_executable(core::path_from_utf8(runfiles->Rlocation(logical_path)));
        resolved.has_value()) {
      return resolved;
    }
  }

  if (resolved_executable.has_value()) {
    return existing_absolute_executable(resolved_executable->parent_path() / ".." / ".." /
                                        "reco_io" / kProbeWorkerName);
  }
  return std::nullopt;
}

} // namespace reco::io
