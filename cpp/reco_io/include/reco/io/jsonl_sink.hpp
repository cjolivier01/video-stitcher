#pragma once

#include "reco/core/pipeline_event.hpp"

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace reco::io {

[[nodiscard]] nlohmann::json pipeline_event_to_json(const reco::core::PipelineEvent& event);

class JsonlSink {
public:
  explicit JsonlSink(const std::filesystem::path& path);

  [[nodiscard]] bool ok() const { return writer_.good(); }
  void emit(const reco::core::PipelineEvent& event);
  void flush();

  [[nodiscard]] std::uint64_t write_failures() const { return write_failures_; }

private:
  std::ofstream writer_;
  std::uint64_t write_failures_ = 0;
};

} // namespace reco::io
