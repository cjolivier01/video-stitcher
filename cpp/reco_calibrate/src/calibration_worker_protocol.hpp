#pragma once

#include "reco/calibrate/gpu_features.hpp"
#include "reco/calibrate/pipeline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace reco::calibrate::detail {

inline constexpr std::uint16_t kCalibrationWorkerProtocolVersion = 4;
inline constexpr std::size_t kCalibrationWorkerFrameHeaderBytes = 12;
inline constexpr std::size_t kMaximumCalibrationWorkerRequestFrameBytes = 64U * 1024U;
inline constexpr std::size_t kMaximumCalibrationWorkerPathBytes = 16U * 1024U;
inline constexpr std::size_t kMaximumCalibrationWorkerErrorBytes = 2U * 1024U;
inline constexpr std::size_t kMaximumCalibrationWorkerResultFrames = 256;
inline constexpr std::size_t kMaximumCalibrationWorkerRoiPoints = 256;
inline constexpr std::size_t kMaximumCalibrationWorkerProfileTextBytes = 4U * 1024U;
inline constexpr std::size_t kCalibrationWorkerMatchedPointBytes = 6U * sizeof(std::uint64_t);
inline constexpr std::size_t kMaximumCalibrationWorkerCorrespondences =
    kMaximumCalibrationWorkerResultFrames * kMaxGpuAkazeFeatures;

inline constexpr std::size_t kCalibrationWorkerCameraBytes =
    2U * (2U * sizeof(std::uint32_t) + 8U * sizeof(std::uint64_t));
inline constexpr std::size_t kCalibrationWorkerRoiBytes =
    1U +
    2U * (sizeof(std::uint32_t) + kMaximumCalibrationWorkerRoiPoints * 2U * sizeof(std::uint64_t));
inline constexpr std::size_t kCalibrationWorkerProfileBytes =
    1U + 2U * (sizeof(std::uint32_t) + kMaximumCalibrationWorkerProfileTextBytes) + 1U + 1U +
    sizeof(std::uint32_t) + kMaximumCalibrationWorkerPathBytes;
inline constexpr std::size_t kCalibrationWorkerPerFrameMetadataBytes =
    6U * sizeof(std::uint64_t) + sizeof(std::uint32_t);
inline constexpr std::size_t kCalibrationWorkerFixedResultMetadataBytes =
    kCalibrationWorkerCameraBytes + 9U * sizeof(std::uint64_t) + sizeof(std::int64_t) +
    kCalibrationWorkerRoiBytes + 2U * sizeof(std::uint32_t) + 4U * sizeof(std::uint64_t) +
    sizeof(std::uint32_t) + 2U * kCalibrationWorkerProfileBytes + 1U + 3U * sizeof(std::uint64_t);
inline constexpr std::size_t kCalibrationWorkerResultMetadataBytes =
    kCalibrationWorkerFixedResultMetadataBytes +
    kMaximumCalibrationWorkerResultFrames * kCalibrationWorkerPerFrameMetadataBytes;
inline constexpr std::size_t kMaximumCalibrationWorkerSuccessFrameBytes =
    kCalibrationWorkerFrameHeaderBytes + kCalibrationWorkerResultMetadataBytes +
    kMaximumCalibrationWorkerCorrespondences * kCalibrationWorkerMatchedPointBytes;
inline constexpr std::size_t kMaximumCalibrationWorkerFailureFrameBytes =
    kCalibrationWorkerFrameHeaderBytes + sizeof(std::uint32_t) +
    kMaximumCalibrationWorkerErrorBytes;
inline constexpr std::size_t kMaximumCalibrationWorkerFrameBytes =
    kMaximumCalibrationWorkerSuccessFrameBytes;

static_assert(kMaximumCalibrationWorkerSuccessFrameBytes <=
              static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()));

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
[[nodiscard]] std::size_t
maximum_calibration_worker_success_frame_bytes(std::size_t maximum_frames,
                                               std::size_t maximum_keypoints);
[[nodiscard]] std::string encode_calibration_worker_request(const GpuCalibrationRequest& request);
[[nodiscard]] GpuCalibrationRequest decode_calibration_worker_request(std::string_view frame);
[[nodiscard]] std::string encode_calibration_worker_success(const CalibrationResult& result);
[[nodiscard]] std::string encode_calibration_worker_failure(std::string_view message);
[[nodiscard]] CalibrationResult decode_calibration_worker_response(std::string_view frame);

} // namespace reco::calibrate::detail
