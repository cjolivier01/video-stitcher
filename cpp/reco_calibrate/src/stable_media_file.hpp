#pragma once

#include <filesystem>
#include <memory>

namespace reco::calibrate::detail {

/// Retains one immutable media identity across parser probing and GPU decode.
class StableMediaFile {
public:
  explicit StableMediaFile(const std::filesystem::path& path);
  ~StableMediaFile();

  StableMediaFile(const StableMediaFile&) = delete;
  StableMediaFile& operator=(const StableMediaFile&) = delete;
  StableMediaFile(StableMediaFile&&) noexcept;
  StableMediaFile& operator=(StableMediaFile&&) noexcept;

  /// Path that reopens the retained file description rather than the user path.
  [[nodiscard]] const std::filesystem::path& decode_path() const;
  /// Rejects in-place changes made after the retained handle was opened.
  void verify_unchanged() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Retains one immutable lens-profile identity across calibration.
class StableLensProfileFile {
public:
  explicit StableLensProfileFile(const std::filesystem::path& path);
  ~StableLensProfileFile();

  StableLensProfileFile(const StableLensProfileFile&) = delete;
  StableLensProfileFile& operator=(const StableLensProfileFile&) = delete;
  StableLensProfileFile(StableLensProfileFile&&) noexcept;
  StableLensProfileFile& operator=(StableLensProfileFile&&) noexcept;

  /// Path that reopens the retained file description rather than the user path.
  [[nodiscard]] const std::filesystem::path& retained_path() const;
  /// Rejects in-place changes or replacement of the selected profile path.
  void verify_unchanged() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace reco::calibrate::detail
