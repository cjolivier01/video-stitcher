#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reco::detect {

class CoreMlError : public std::runtime_error {
public:
  explicit CoreMlError(std::string message) : std::runtime_error(std::move(message)) {}
};

struct CoreMlRuntimeProbe {
  bool available = false;
  std::string provider;
  std::string error;
};

struct CoreMlSessionConfig {
  std::filesystem::path model_path;
  std::string input_name = "images";
  std::string output_name = "output0";
  std::vector<std::int64_t> input_shape = {1, 3, 1280, 1280};
};

struct CoreMlTensorOutput {
  std::vector<std::int64_t> shape;
  std::vector<float> data;
};

[[nodiscard]] CoreMlRuntimeProbe probe_coreml_runtime();
[[nodiscard]] bool coreml_runtime_available();
[[nodiscard]] std::string coreml_runtime_error();

class CoreMlSession {
public:
  explicit CoreMlSession(CoreMlSessionConfig config);
  ~CoreMlSession();
  CoreMlSession(CoreMlSession&&) noexcept;
  CoreMlSession& operator=(CoreMlSession&&) noexcept;
  CoreMlSession(const CoreMlSession&) = delete;
  CoreMlSession& operator=(const CoreMlSession&) = delete;

  [[nodiscard]] const CoreMlSessionConfig& config() const;
  [[nodiscard]] CoreMlTensorOutput predict_shared_chw(std::span<float> input);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace reco::detect
