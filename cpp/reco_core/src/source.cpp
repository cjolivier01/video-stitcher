#include "reco/core/source.hpp"

#include <cerrno>
#include <cstdio>
#include <sstream>
#include <system_error>

namespace reco::core {

PathValidationResult validate_input_path(const std::filesystem::path& path) {
  PathValidationResult result;
  result.path = path.string();

  std::error_code ec;
  const auto status = std::filesystem::status(path, ec);
  if (ec) {
    result.reason = ec == std::errc::permission_denied ? InvalidPathReason::PermissionDenied
                                                       : InvalidPathReason::NotFound;
    return result;
  }
  if (!std::filesystem::exists(status)) {
    result.reason = InvalidPathReason::NotFound;
    return result;
  }
  if (!std::filesystem::is_regular_file(status)) {
    result.reason = InvalidPathReason::NotAFile;
    return result;
  }
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    result.reason = ec == std::errc::permission_denied ? InvalidPathReason::PermissionDenied
                                                       : InvalidPathReason::NotFound;
    return result;
  }
  if (size == 0) {
    result.reason = InvalidPathReason::Empty;
    return result;
  }
  errno = 0;
  FILE* input = std::fopen(path.string().c_str(), "rb");
  if (input == nullptr && (errno == EACCES || errno == EPERM)) {
    result.reason = InvalidPathReason::PermissionDenied;
    return result;
  }
  if (input != nullptr) {
    std::fclose(input);
  }
  result.ok = true;
  return result;
}

std::string invalid_path_reason_name(InvalidPathReason reason) {
  switch (reason) {
  case InvalidPathReason::NotFound:
    return "file not found";
  case InvalidPathReason::NotAFile:
    return "not a regular file";
  case InvalidPathReason::Empty:
    return "file is empty";
  case InvalidPathReason::PermissionDenied:
    return "permission denied";
  }
  return "unknown";
}

std::string YuvFrame::validate() const {
  const auto expected_y = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (y.size() != expected_y) {
    std::ostringstream out;
    out << "Y plane size mismatch: expected " << expected_y << " (" << width << "x" << height
        << "), got " << y.size();
    return out.str();
  }
  const auto expected_uv =
      (static_cast<std::size_t>(width) / 2) * (static_cast<std::size_t>(height) / 2);
  if (u.size() != expected_uv) {
    std::ostringstream out;
    out << "U plane size mismatch: expected " << expected_uv << " (" << (width / 2) << "x"
        << (height / 2) << "), got " << u.size();
    return out.str();
  }
  if (v.size() != expected_uv) {
    std::ostringstream out;
    out << "V plane size mismatch: expected " << expected_uv << " (" << (width / 2) << "x"
        << (height / 2) << "), got " << v.size();
    return out.str();
  }
  return {};
}

} // namespace reco::core
