#include "reco/autocam/file_panner.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace reco::autocam {
namespace {

std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> cols;
  std::stringstream stream(line);
  std::string col;
  while (std::getline(stream, col, ',')) {
    cols.push_back(col);
  }
  return cols;
}

std::uint64_t parse_frame(const std::string& value) {
  std::size_t consumed = 0;
  const auto trimmed = trim(value);
  if (!trimmed.empty() && trimmed.front() == '-') {
    throw std::invalid_argument("invalid frame index");
  }
  const auto out = std::stoull(trimmed, &consumed);
  if (consumed != trimmed.size()) {
    throw std::invalid_argument("invalid frame index");
  }
  return out;
}

float parse_float(const std::string& value) {
  std::size_t consumed = 0;
  const auto trimmed = trim(value);
  const float out = std::stof(trimmed, &consumed);
  if (consumed != trimmed.size()) {
    throw std::invalid_argument("invalid float");
  }
  return out;
}

std::optional<float> parse_optional_float(const std::string& value) {
  const auto trimmed = trim(value);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  try {
    return parse_float(trimmed);
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  } catch (const std::out_of_range&) {
    return std::nullopt;
  }
}

} // namespace

FilePanner FilePanner::from_csv(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file panner CSV: " + path.string());
  }

  FilePanner panner;
  std::string line;
  std::uint64_t line_index = 0;
  while (std::getline(file, line)) {
    if (line_index++ == 0 || line.empty()) {
      continue;
    }
    const auto cols = split_csv_line(line);
    if (cols.size() < 3) {
      continue;
    }
    const auto frame = parse_frame(cols[0]);
    const auto yaw = parse_float(cols[1]);
    const auto pitch = parse_float(cols[2]);
    panner.poses_[frame] = reco::core::ViewportPosition{
        .yaw = yaw,
        .pitch = pitch,
        .fov_degrees = cols.size() > 3 ? parse_optional_float(cols[3]) : std::nullopt,
    };
  }
  return panner;
}

reco::core::ViewportPosition FilePanner::decide(const WorldState&, const PanContext& context) {
  if (const auto it = poses_.find(context.frame_index); it != poses_.end()) {
    last_ = it->second;
  }
  return last_;
}

} // namespace reco::autocam
