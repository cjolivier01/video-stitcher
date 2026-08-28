#pragma once

#include "reco/calibrate/pipeline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace reco::calibrate::detail {

inline constexpr std::uint16_t kCalibrationWorkerProtocolVersion = 4;
inline constexpr std::size_t kCalibrationWorkerFrameHeaderBytes = 12;
inline constexpr std::size_t kMaximumCalibrationWorkerFrameBytes = 64U * 1024U;
inline constexpr std::size_t kMaximumCalibrationWorkerPathBytes = 16U * 1024U;
inline constexpr std::size_t kMaximumCalibrationWorkerErrorBytes = 2U * 1024U;
inline constexpr std::size_t kMaximumCalibrationWorkerResultFrames = 256;
inline constexpr std::size_t kMaximumCalibrationWorkerRoiPoints = 256;
inline constexpr std::size_t kCalibrationWorkerMatchedPointBytes = 6U * sizeof(std::uint64_t);
inline constexpr std::size_t kMaximumCalibrationWorkerCorrespondences =
    (kMaximumCalibrationWorkerFrameBytes - kCalibrationWorkerFrameHeaderBytes) /
    kCalibrationWorkerMatchedPointBytes;

enum class CalibrationWorkerMessage : std::uint16_t {
  Request = 1,
  Success = 2,
  Failure = 3,
};

using CalibrationWorkerFrameHeader = std::array<char, kCalibrationWorkerFrameHeaderBytes>;

struct DecodedCalibrationWorkerHeader {
  CalibrationWorkerMessage message = CalibrationWorkerMessage::Failure;
  std::size_t payload_size = 0;
};

[[nodiscard]] DecodedCalibrationWorkerHeader
decode_calibration_worker_header(const CalibrationWorkerFrameHeader& header);
[[nodiscard]] std::string encode_calibration_worker_request(const GpuCalibrationRequest& request);
[[nodiscard]] GpuCalibrationRequest decode_calibration_worker_request(std::string_view frame);
[[nodiscard]] std::string encode_calibration_worker_success(const CalibrationResult& result);
[[nodiscard]] std::string encode_calibration_worker_failure(std::string_view message);
[[nodiscard]] CalibrationResult decode_calibration_worker_response(std::string_view frame);

} // namespace reco::calibrate::detail
