#include "reco/io/stacked_video.hpp"

#include <algorithm>
#include <limits>

namespace reco::io {
namespace {

bool even_nonzero(std::uint32_t value) { return value != 0 && value % 2 == 0; }

std::size_t y_len(std::uint32_t width, std::uint32_t height) {
  return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

std::size_t uv_len(std::uint32_t width, std::uint32_t height) {
  return static_cast<std::size_t>(width / 2) * static_cast<std::size_t>(height / 2);
}

void copy_tile_plane(const std::vector<std::uint8_t>& src, std::size_t tile_width,
                     std::size_t tile_height, std::uint32_t col, std::uint32_t row,
                     std::vector<std::uint8_t>& dst, std::size_t dst_stride) {
  const auto dst_x = static_cast<std::size_t>(col) * tile_width;
  const auto dst_y = static_cast<std::size_t>(row) * tile_height;
  for (std::size_t r = 0; r < tile_height; ++r) {
    const auto src_off = r * tile_width;
    const auto dst_off = (dst_y + r) * dst_stride + dst_x;
    std::copy_n(src.begin() + static_cast<std::ptrdiff_t>(src_off), tile_width,
                dst.begin() + static_cast<std::ptrdiff_t>(dst_off));
  }
}

void read_tile_plane(const std::vector<std::uint8_t>& packed, std::size_t packed_stride,
                     std::size_t tile_width, std::size_t tile_height, std::uint32_t col,
                     std::uint32_t row, std::vector<std::uint8_t>& dst) {
  const auto src_x = static_cast<std::size_t>(col) * tile_width;
  const auto src_y = static_cast<std::size_t>(row) * tile_height;
  for (std::size_t r = 0; r < tile_height; ++r) {
    const auto src_off = (src_y + r) * packed_stride + src_x;
    const auto dst_off = r * tile_width;
    std::copy_n(packed.begin() + static_cast<std::ptrdiff_t>(src_off), tile_width,
                dst.begin() + static_cast<std::ptrdiff_t>(dst_off));
  }
}

} // namespace

std::variant<GridLayout, std::string_view>
GridLayout::vstack(std::uint32_t width, std::uint32_t height, std::uint32_t n) {
  return grid(width, height, n, 1);
}

std::variant<GridLayout, std::string_view>
GridLayout::hstack(std::uint32_t width, std::uint32_t height, std::uint32_t n) {
  return grid(width, height, 1, n);
}

std::variant<GridLayout, std::string_view> GridLayout::grid(std::uint32_t width,
                                                            std::uint32_t height,
                                                            std::uint32_t rows,
                                                            std::uint32_t cols) {
  if (width == 0 || height == 0 || rows == 0 || cols == 0) {
    return std::string_view("width, height, rows, and cols must be non-zero");
  }
  if (!even_nonzero(width) || !even_nonzero(height)) {
    return std::string_view("YUV420P tile dimensions must be even");
  }
  if (rows > std::numeric_limits<std::uint32_t>::max() / cols ||
      cols > std::numeric_limits<std::uint32_t>::max() / width ||
      rows > std::numeric_limits<std::uint32_t>::max() / height) {
    return std::string_view("grid dimensions overflow");
  }
  return GridLayout(width, height, rows, cols);
}

PackResult pack_yuv420p(const GridLayout& layout,
                        const std::vector<const reco::core::YuvFrame*>& tiles) {
  if (tiles.size() > layout.capacity()) {
    return StackError{TooManyTiles{layout.capacity(), tiles.size()}};
  }

  const auto expected_y = y_len(layout.tile_width(), layout.tile_height());
  const auto expected_uv = uv_len(layout.tile_width(), layout.tile_height());
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    const auto* tile = tiles[i];
    if (tile == nullptr) {
      continue;
    }
    if (tile->width != layout.tile_width() || tile->height != layout.tile_height()) {
      return StackError{TileDimensionMismatch{i, tile->width, tile->height, layout.tile_width(),
                                              layout.tile_height()}};
    }
    if (tile->y.size() != expected_y) {
      return StackError{PlaneSizeMismatch{i, "y", tile->y.size(), expected_y}};
    }
    if (tile->u.size() != expected_uv) {
      return StackError{PlaneSizeMismatch{i, "u", tile->u.size(), expected_uv}};
    }
    if (tile->v.size() != expected_uv) {
      return StackError{PlaneSizeMismatch{i, "v", tile->v.size(), expected_uv}};
    }
  }

  const auto packed_width = layout.packed_width();
  const auto packed_height = layout.packed_height();
  reco::core::YuvFrame out;
  out.width = packed_width;
  out.height = packed_height;
  out.y.assign(y_len(packed_width, packed_height), 128);
  out.u.assign(uv_len(packed_width, packed_height), 128);
  out.v.assign(uv_len(packed_width, packed_height), 128);

  const auto tw = static_cast<std::size_t>(layout.tile_width());
  const auto th = static_cast<std::size_t>(layout.tile_height());
  const auto pw = static_cast<std::size_t>(packed_width);
  bool copied_timestamp = false;
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    const auto* tile = tiles[i];
    if (tile == nullptr) {
      continue;
    }
    const auto row = static_cast<std::uint32_t>(i / layout.cols());
    const auto col = static_cast<std::uint32_t>(i % layout.cols());
    copy_tile_plane(tile->y, tw, th, col, row, out.y, pw);
    copy_tile_plane(tile->u, tw / 2, th / 2, col, row, out.u, pw / 2);
    copy_tile_plane(tile->v, tw / 2, th / 2, col, row, out.v, pw / 2);
    if (!copied_timestamp) {
      out.timestamp_us = tile->timestamp_us;
      copied_timestamp = true;
    }
  }

  return out;
}

UnpackResult unpack_yuv420p(const GridLayout& layout, const reco::core::YuvFrame& packed) {
  if (packed.width != layout.packed_width() || packed.height != layout.packed_height()) {
    return StackError{PackedDimensionMismatch{packed.width, packed.height, layout.packed_width(),
                                              layout.packed_height()}};
  }

  const auto packed_y = y_len(layout.packed_width(), layout.packed_height());
  const auto packed_uv = uv_len(layout.packed_width(), layout.packed_height());
  if (packed.y.size() != packed_y) {
    return StackError{PlaneSizeMismatch{kPackedFrameIndex, "y", packed.y.size(), packed_y}};
  }
  if (packed.u.size() != packed_uv) {
    return StackError{PlaneSizeMismatch{kPackedFrameIndex, "u", packed.u.size(), packed_uv}};
  }
  if (packed.v.size() != packed_uv) {
    return StackError{PlaneSizeMismatch{kPackedFrameIndex, "v", packed.v.size(), packed_uv}};
  }

  const auto tw = static_cast<std::size_t>(layout.tile_width());
  const auto th = static_cast<std::size_t>(layout.tile_height());
  const auto pw = static_cast<std::size_t>(layout.packed_width());
  std::vector<reco::core::YuvFrame> out;
  out.reserve(layout.capacity());
  for (std::uint32_t i = 0; i < layout.capacity(); ++i) {
    const auto row = i / layout.cols();
    const auto col = i % layout.cols();
    reco::core::YuvFrame tile;
    tile.width = layout.tile_width();
    tile.height = layout.tile_height();
    tile.timestamp_us = packed.timestamp_us;
    tile.y.resize(y_len(layout.tile_width(), layout.tile_height()));
    tile.u.resize(uv_len(layout.tile_width(), layout.tile_height()));
    tile.v.resize(uv_len(layout.tile_width(), layout.tile_height()));
    read_tile_plane(packed.y, pw, tw, th, col, row, tile.y);
    read_tile_plane(packed.u, pw / 2, tw / 2, th / 2, col, row, tile.u);
    read_tile_plane(packed.v, pw / 2, tw / 2, th / 2, col, row, tile.v);
    out.push_back(std::move(tile));
  }
  return out;
}

} // namespace reco::io
