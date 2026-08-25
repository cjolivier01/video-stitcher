#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace reco::detect {

class NcnnError : public std::runtime_error {
public:
  explicit NcnnError(std::string message) : std::runtime_error(std::move(message)) {}
};

struct NcnnTensorOutput {
  std::int32_t width = 0;
  std::int32_t height = 0;
  std::vector<float> data;
};

struct NcnnSessionConfig {
  std::filesystem::path model_dir;
  std::string input_name = "in0";
  std::string output_name = "out0";
  std::int32_t num_threads = 4;
};

class NcnnSession {
public:
  explicit NcnnSession(NcnnSessionConfig config);
  ~NcnnSession();
  NcnnSession(NcnnSession&&) noexcept;
  NcnnSession& operator=(NcnnSession&&) noexcept;
  NcnnSession(const NcnnSession&) = delete;
  NcnnSession& operator=(const NcnnSession&) = delete;

  [[nodiscard]] NcnnTensorOutput run_preprocessed_chw(std::span<const float> input,
                                                      std::uint32_t input_size);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] bool ncnn_runtime_available();
[[nodiscard]] std::string ncnn_runtime_error();

} // namespace reco::detect
