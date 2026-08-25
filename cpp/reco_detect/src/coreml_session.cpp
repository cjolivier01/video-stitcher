#include "reco/detect/coreml_session.hpp"

#include <limits>
#include <string_view>

namespace reco::detect {
namespace {

bool contains_nul(std::string_view value) {
  return value.find('\0') != std::string_view::npos;
}

std::size_t checked_input_count(std::span<const std::int64_t> shape) {
  if (shape.size() != 4) {
    throw CoreMlError("CoreML input shape must be rank-4 NCHW");
  }
  std::size_t count = 1;
  for (const auto dim : shape) {
    if (dim <= 0) {
      throw CoreMlError("CoreML input shape dimensions must be positive");
    }
    const auto value = static_cast<std::size_t>(dim);
    if (count > std::numeric_limits<std::size_t>::max() / value) {
      throw CoreMlError("CoreML input shape element count overflows");
    }
    count *= value;
  }
  if (shape[0] != 1 || shape[1] != 3 || shape[2] != shape[3]) {
    throw CoreMlError("CoreML input shape must be [1,3,S,S]");
  }
  return count;
}

} // namespace

CoreMlRuntimeProbe probe_coreml_runtime() {
#if defined(__APPLE__)
  return CoreMlRuntimeProbe{.available = true, .provider = "CoreML"};
#else
  return CoreMlRuntimeProbe{
      .available = false,
      .provider = {},
      .error = "CoreML is available only on Apple platforms",
  };
#endif
}

bool coreml_runtime_available() { return probe_coreml_runtime().available; }

std::string coreml_runtime_error() { return probe_coreml_runtime().error; }

struct CoreMlSession::Impl {
  explicit Impl(CoreMlSessionConfig session_config)
      : config(std::move(session_config)), input_count(checked_input_count(config.input_shape)) {
    if (config.model_path.empty()) {
      throw CoreMlError("CoreML model path must not be empty");
    }
    if (contains_nul(config.model_path.string()) || contains_nul(config.input_name) ||
        contains_nul(config.output_name)) {
      throw CoreMlError("CoreML paths and tensor names must not contain NUL bytes");
    }
    if (!std::filesystem::exists(config.model_path)) {
      throw CoreMlError("CoreML model path not found: `" + config.model_path.string() + "`");
    }
    if (!std::filesystem::is_directory(config.model_path) ||
        config.model_path.extension() != ".mlmodelc") {
      throw CoreMlError("CoreML model path must be a compiled .mlmodelc directory");
    }
  }

  CoreMlSessionConfig config;
  std::size_t input_count = 0;
};

CoreMlSession::CoreMlSession(CoreMlSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
CoreMlSession::~CoreMlSession() = default;
CoreMlSession::CoreMlSession(CoreMlSession&&) noexcept = default;
CoreMlSession& CoreMlSession::operator=(CoreMlSession&&) noexcept = default;

const CoreMlSessionConfig& CoreMlSession::config() const { return impl_->config; }

CoreMlTensorOutput CoreMlSession::predict_shared_chw(std::span<float> input) {
  if (input.size() != impl_->input_count) {
    throw CoreMlError("CoreML input span length does not match input shape");
  }
#if defined(__APPLE__)
  throw CoreMlError("CoreML Objective-C++ prediction bridge is not ported yet");
#else
  throw CoreMlError(coreml_runtime_error());
#endif
}

} // namespace reco::detect
