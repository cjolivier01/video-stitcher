#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace reco::io {

class StableMediaFile;

namespace detail {

[[nodiscard]] std::shared_ptr<const StableMediaFile>
adopt_stable_media_worker_descriptor(int descriptor, std::string display_path);
[[nodiscard]] std::intptr_t stable_media_native_handle(const StableMediaFile& file);

} // namespace detail

/// Descriptor-backed regular file whose selected identity cannot follow later pathname changes.
class StableMediaFile final {
public:
  /// Opens and retains the file currently selected by `path`.
  [[nodiscard]] static std::shared_ptr<const StableMediaFile>
  open(const std::filesystem::path& path);

  /// Opens an independent seek cursor and accepts it only when it matches this retained identity.
  [[nodiscard]] std::shared_ptr<const StableMediaFile> open_cursor() const;

  StableMediaFile(const StableMediaFile&) = delete;
  StableMediaFile& operator=(const StableMediaFile&) = delete;
  ~StableMediaFile();

  /// Original user-facing path. Consumers must not reopen it.
  [[nodiscard]] const std::filesystem::path& display_path() const noexcept;
  /// Borrowed C-runtime descriptor suitable for GStreamer's `fdsrc`.
  [[nodiscard]] int descriptor() const;
  /// Reads the retained file from offset zero without reopening its pathname.
  [[nodiscard]] std::string read_all(std::size_t maximum_bytes) const;
  /// Resets this descriptor's independent seek cursor to byte zero.
  void rewind() const;
  /// Fails when the retained file's identity or size/mtime/ctime snapshot changed, or when the
  /// original pathname no longer resolves to that unchanged snapshot.
  void verify_unchanged() const;

private:
  struct Impl;
  explicit StableMediaFile(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend std::shared_ptr<const StableMediaFile>
  detail::adopt_stable_media_worker_descriptor(int descriptor, std::string display_path);
  friend std::intptr_t detail::stable_media_native_handle(const StableMediaFile& file);
};

} // namespace reco::io
