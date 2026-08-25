#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reco::detect {

enum class TrtDataType {
  Float,
  Half,
  Int8,
  Int32,
};

[[nodiscard]] std::size_t trt_data_type_byte_size(TrtDataType data_type);

struct TrtBindingInfo {
  std::string name;
  bool is_input = false;
  std::vector<std::int32_t> dims;
  TrtDataType data_type = TrtDataType::Float;
  std::size_t byte_size = 0;
};

class TrtError : public std::runtime_error {
public:
  explicit TrtError(std::string message) : std::runtime_error(std::move(message)) {}
};

struct TrtEngineState;

class TrtContext {
public:
  ~TrtContext();
  TrtContext(TrtContext&&) noexcept;
  TrtContext& operator=(TrtContext&&) noexcept;
  TrtContext(const TrtContext&) = delete;
  TrtContext& operator=(const TrtContext&) = delete;

  void enqueue(std::span<void*> bindings, void* stream) const;

private:
  friend class TrtEngine;
  TrtContext(void* context, std::shared_ptr<const TrtEngineState> engine_state,
             std::size_t binding_count);

  void* context_ = nullptr;
  std::shared_ptr<const TrtEngineState> engine_state_;
  std::size_t binding_count_ = 0;
};

class TrtEngine {
public:
  explicit TrtEngine(const std::filesystem::path& engine_path);
  ~TrtEngine();
  TrtEngine(TrtEngine&&) noexcept;
  TrtEngine& operator=(TrtEngine&&) noexcept;
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  [[nodiscard]] std::vector<TrtBindingInfo> bindings() const;
  [[nodiscard]] TrtContext create_context() const;

private:
  std::shared_ptr<TrtEngineState> state_;
};

[[nodiscard]] bool trt_runtime_available();
[[nodiscard]] std::string trt_runtime_error();

} // namespace reco::detect
