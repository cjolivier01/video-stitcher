#pragma once

#include <filesystem>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace reco::detect {

struct OrtRuntimeProbe {
  bool available = false;
  std::string path;
  std::string version;
  std::string error;
};

[[nodiscard]] OrtRuntimeProbe probe_ort_runtime();
[[nodiscard]] bool ort_runtime_available();
[[nodiscard]] std::string ort_runtime_error();
[[nodiscard]] std::filesystem::path reco_cache_dir(std::string_view subdir);

enum class OrtExecutionProvider {
  Cpu,
  Cuda,
  TensorRt,
  CoreMl,
  DirectMl,
};

struct OrtSessionConfig {
  std::filesystem::path model_path;
  std::vector<std::string> fallback_labels;
  std::vector<OrtExecutionProvider> providers = {OrtExecutionProvider::Cpu};
};

struct OrtSessionMetadata {
  std::uint32_t input_size = 1280;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> labels = {"ball"};
};

struct OrtTensorOutput {
  std::vector<std::int64_t> shape;
  std::vector<float> data;
};

class OrtSession {
public:
  explicit OrtSession(OrtSessionConfig config);
  ~OrtSession();

  OrtSession(const OrtSession&) = delete;
  OrtSession& operator=(const OrtSession&) = delete;
  OrtSession(OrtSession&&) noexcept;
  OrtSession& operator=(OrtSession&&) noexcept;

  [[nodiscard]] const OrtSessionMetadata& metadata() const;
  [[nodiscard]] std::vector<OrtTensorOutput> run_cpu_f32(std::span<const float> input,
                                                         std::span<const std::int64_t> shape);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace reco::detect
