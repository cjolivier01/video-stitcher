#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace reco::core {

/// Maximum accepted CUDA C++ source size.
inline constexpr std::size_t kNvrtcMaximumSourceBytes = 16U * 1024U * 1024U;
/// Maximum accepted virtual source filename size.
inline constexpr std::size_t kNvrtcMaximumSourceNameBytes = 4096U;
/// Maximum number of command-line options accepted for one compilation.
inline constexpr std::size_t kNvrtcMaximumOptionCount = 128U;
/// Maximum accepted size of one command-line option.
inline constexpr std::size_t kNvrtcMaximumOptionBytes = 4096U;
/// Maximum combined size of command-line options for one compilation.
inline constexpr std::size_t kNvrtcMaximumTotalOptionBytes = 64U * 1024U;
/// Maximum compiler log size that may be read from NVRTC.
inline constexpr std::size_t kNvrtcMaximumLogBytes = 4U * 1024U * 1024U;
/// Maximum PTX output size that may be read from NVRTC.
inline constexpr std::size_t kNvrtcMaximumPtxBytes = 64U * 1024U * 1024U;

/// Error reported by NVRTC discovery or execution.
class NvrtcError : public std::runtime_error {
public:
  explicit NvrtcError(std::string message) : std::runtime_error(std::move(message)) {}
};

/// Loaded NVRTC API version.
struct NvrtcVersion {
  /// Major NVRTC API version.
  int major = 0;
  /// Minor NVRTC API version.
  int minor = 0;
};

/// Options for one CUDA C++ compilation.
struct NvrtcCompileOptions {
  /// Options passed verbatim to `nvrtcCompileProgram` after validation.
  std::vector<std::string> values{"--std=c++17"};
};

/// Bounded outputs from one successful CUDA C++ compilation.
struct NvrtcCompileResult {
  /// Textual PTX without NVRTC's terminating NUL byte.
  std::string ptx;
  /// Bounded compiler diagnostics without a terminating NUL byte.
  std::string log;
};

/// Dynamically loaded NVRTC compiler with bounded inputs and outputs.
///
/// This type links neither the CUDA runtime nor NVRTC. PTX execution remains
/// the responsibility of `CudaBackend::load_module_from_ptx`.
class NvrtcCompiler {
public:
  /// Probes the platform's supported NVRTC library names.
  [[nodiscard]] static bool is_available();
  /// Returns an empty string when default NVRTC discovery succeeds.
  [[nodiscard]] static std::string availability_error();
  /// Loads the first supported platform NVRTC library.
  [[nodiscard]] static NvrtcCompiler create();
  /// Loads an exact NVRTC library path, primarily for explicit deployments and tests.
  [[nodiscard]] static NvrtcCompiler load(std::string_view library_path);
  /// Diagnoses an exact NVRTC library without retaining it.
  [[nodiscard]] static std::string availability_error(std::string_view library_path);

  /// Returns the loaded NVRTC API version.
  [[nodiscard]] NvrtcVersion version() const;
  /// Returns the exact loaded library name or path.
  [[nodiscard]] std::string_view library_path() const;
  /// Compiles CUDA C++ to PTX and returns any compiler diagnostics.
  [[nodiscard]] NvrtcCompileResult
  compile(std::string_view source, std::string_view source_name,
          const NvrtcCompileOptions& options = NvrtcCompileOptions{}) const;

private:
  struct Impl;
  explicit NvrtcCompiler(std::shared_ptr<Impl> impl);
  [[nodiscard]] const Impl& checked_impl() const;

  std::shared_ptr<Impl> impl_;
};

} // namespace reco::core
