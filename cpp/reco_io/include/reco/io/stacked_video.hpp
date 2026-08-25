#pragma once

#include "reco/core/source.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <variant>
#include <vector>

namespace reco::io {

class GridLayout {
public:
  [[nodiscard]] static std::variant<GridLayout, std::string_view>
  vstack(std::uint32_t width, std::uint32_t height, std::uint32_t n);

  [[nodiscard]] static std::variant<GridLayout, std::string_view>
  hstack(std::uint32_t width, std::uint32_t height, std::uint32_t n);

  [[nodiscard]] static std::variant<GridLayout, std::string_view>
  grid(std::uint32_t width, std::uint32_t height, std::uint32_t rows, std::uint32_t cols);

  [[nodiscard]] std::uint32_t tile_width() const { return tile_width_; }
  [[nodiscard]] std::uint32_t tile_height() const { return tile_height_; }
  [[nodiscard]] std::uint32_t rows() const { return rows_; }
  [[nodiscard]] std::uint32_t cols() const { return cols_; }
  [[nodiscard]] std::uint32_t capacity() const { return rows_ * cols_; }
  [[nodiscard]] std::uint32_t packed_width() const { return cols_ * tile_width_; }
  [[nodiscard]] std::uint32_t packed_height() const { return rows_ * tile_height_; }

private:
  GridLayout(std::uint32_t width, std::uint32_t height, std::uint32_t rows, std::uint32_t cols)
      : tile_width_(width), tile_height_(height), rows_(rows), cols_(cols) {}

  std::uint32_t tile_width_ = 0;
  std::uint32_t tile_height_ = 0;
  std::uint32_t rows_ = 0;
  std::uint32_t cols_ = 0;
};

struct TooManyTiles {
  std::uint32_t capacity = 0;
  std::size_t got = 0;
};

struct TileDimensionMismatch {
  std::size_t index = 0;
  std::uint32_t got_w = 0;
  std::uint32_t got_h = 0;
  std::uint32_t expected_w = 0;
  std::uint32_t expected_h = 0;
};

struct PackedDimensionMismatch {
  std::uint32_t got_w = 0;
  std::uint32_t got_h = 0;
  std::uint32_t expected_w = 0;
  std::uint32_t expected_h = 0;
};

struct PlaneSizeMismatch {
  std::size_t index = 0;
  std::string_view plane;
  std::size_t got = 0;
  std::size_t expected = 0;
};

using StackError =
    std::variant<TooManyTiles, TileDimensionMismatch, PackedDimensionMismatch, PlaneSizeMismatch>;

using PackResult = std::variant<reco::core::YuvFrame, StackError>;
using UnpackResult = std::variant<std::vector<reco::core::YuvFrame>, StackError>;

inline constexpr std::size_t kPackedFrameIndex = static_cast<std::size_t>(-1);

[[nodiscard]] PackResult pack_yuv420p(const GridLayout& layout,
                                      const std::vector<const reco::core::YuvFrame*>& tiles);

[[nodiscard]] UnpackResult unpack_yuv420p(const GridLayout& layout,
                                          const reco::core::YuvFrame& packed);

} // namespace reco::io
