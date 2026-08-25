#pragma once

#include "reco/calibrate/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reco::calibrate {

[[nodiscard]] std::vector<std::uint64_t> select_frame_indices(std::uint64_t total_frames,
                                                              double fps,
                                                              std::size_t num_samples,
                                                              double skip_start_secs,
                                                              double skip_end_secs);

[[nodiscard]] GrayFrame downscale_if_needed(const GrayFrame& frame, std::uint32_t target_width);

} // namespace reco::calibrate
