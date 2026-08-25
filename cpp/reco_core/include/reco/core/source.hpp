#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace reco::core {

enum class InvalidPathReason {
  NotFound,
  NotAFile,
  Empty,
  PermissionDenied,
};

struct PathValidationResult {
  bool ok = false;
  InvalidPathReason reason = InvalidPathReason::NotFound;
  std::string path;
};

PathValidationResult validate_input_path(const std::filesystem::path& path);
std::string invalid_path_reason_name(InvalidPathReason reason);

struct YuvFrame {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::int64_t timestamp_us = 0;

  [[nodiscard]] std::string validate() const;
};

struct YuvData {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

struct Nv12Data {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> uv;
};

struct FramePair {
  YuvData left;
  YuvData right;
};

struct Nv12FramePair {
  Nv12Data left;
  Nv12Data right;
};

class CameraInput {
public:
  virtual ~CameraInput() = default;
  [[nodiscard]] virtual const char* name() const = 0;
  [[nodiscard]] virtual std::uint8_t camera_count() const = 0;
};

class StereoCameraInput final : public CameraInput {
public:
  [[nodiscard]] const char* name() const override { return "stereo-2camera"; }
  [[nodiscard]] std::uint8_t camera_count() const override { return 2; }
};

class MonoCameraInput final : public CameraInput {
public:
  [[nodiscard]] const char* name() const override { return "mono-1camera"; }
  [[nodiscard]] std::uint8_t camera_count() const override { return 1; }
};

} // namespace reco::core
