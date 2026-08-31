#include "reco/core/path.hpp"

namespace reco::core {

std::filesystem::path path_from_utf8(std::string_view value) {
#if defined(_WIN32)
  return std::filesystem::u8path(value.begin(), value.end());
#else
  return std::filesystem::path(value);
#endif
}

std::string path_to_utf8(const std::filesystem::path& value) {
  const auto encoded = value.u8string();
  return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

} // namespace reco::core
