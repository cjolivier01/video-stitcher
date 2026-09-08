#pragma once

#include "reco/cli/cli.hpp"

#include <filesystem>
#include <iosfwd>

namespace reco::cli::detail {

int run_gpu_stitch(const StitchCommand& command, const std::filesystem::path& executable_path,
                   std::ostream& out, std::ostream& err);

} // namespace reco::cli::detail
