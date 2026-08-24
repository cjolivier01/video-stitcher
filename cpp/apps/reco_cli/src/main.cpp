#include "reco/cli/cli.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc > 1 ? argc - 1 : 0));
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  auto parsed = reco::cli::parse_args(args);
  if (const auto* error = std::get_if<reco::cli::ParseError>(&parsed)) {
    std::cerr << "error: " << error->message << "\n\n" << reco::cli::help_text() << '\n';
    return 2;
  }

  const auto command = std::get<reco::cli::Command>(std::move(parsed));
  if (std::holds_alternative<reco::cli::HelpCommand>(command)) {
    std::cout << reco::cli::help_text() << '\n';
    return 0;
  }

  std::cerr << "error: C++ reco " << reco::cli::command_name(command)
            << " execution is not ported yet; GPU/runtime backend stages remain authoritative in "
               "Rust.\n";
  return 2;
}
