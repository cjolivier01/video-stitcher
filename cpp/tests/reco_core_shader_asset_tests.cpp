#include "reco/core/shader_catalog.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <string_view>

using namespace reco::core;

namespace {

int failures = 0;

void expect_true(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename T, typename U> void expect_eq(T actual, U expected, std::string_view message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    ++failures;
  }
}

std::filesystem::path runfile_path(std::string_view relative) {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (test_srcdir == nullptr || workspace == nullptr) {
    return std::filesystem::path(relative);
  }
  return std::filesystem::path(test_srcdir) / workspace / std::string(relative);
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "FAIL: unable to open " << path << '\n';
    ++failures;
    return {};
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::size_t count_stage_annotation(std::string_view source, ShaderStage stage) {
  const std::string_view annotation = [stage] {
    switch (stage) {
    case ShaderStage::kVertex:
      return std::string_view("@vertex");
    case ShaderStage::kFragment:
      return std::string_view("@fragment");
    case ShaderStage::kCompute:
      return std::string_view("@compute");
    }
    return std::string_view();
  }();

  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = source.find(annotation, pos)) != std::string_view::npos) {
    ++count;
    pos += annotation.size();
  }
  return count;
}

void catalog_matches_rust_shader_set() {
  const auto& assets = shader_catalog();
  expect_eq(assets.size(), 6U, "shader asset count");
  const std::set<std::string_view> names = {
      "bayer_demosaic", "color_grade", "cylindrical_mono",
      "fisheye",        "rgba_to_nv12", "yuv420p_stack_pack",
  };
  for (const auto& name : names) {
    expect_true(find_shader(name) != nullptr, std::string("find shader ") + std::string(name));
  }
}

void shader_files_have_declared_entry_points() {
  for (const auto& asset : shader_catalog()) {
    const auto source = read_text(runfile_path(asset.path));
    expect_true(!source.empty(), std::string("shader non-empty ") + std::string(asset.name));
    for (const auto stage : asset.stages) {
      expect_true(count_stage_annotation(source, stage) > 0,
                  std::string("stage annotation ") + std::string(asset.name));
    }
    for (const auto entry_point : asset.entry_points) {
      const std::string needle = "fn " + std::string(entry_point) + "(";
      expect_true(source.find(needle) != std::string::npos,
                  std::string("entry point ") + std::string(asset.name) + "::" +
                      std::string(entry_point));
    }
  }
}

void compute_catalog_matches_compute_shader_count() {
  std::size_t compute_assets = 0;
  for (const auto& asset : shader_catalog()) {
    if (asset.stages.size() == 1 && asset.stages[0] == ShaderStage::kCompute) {
      ++compute_assets;
    }
  }
  expect_eq(compute_assets, 4U, "compute shader asset count");
}

} // namespace

int main() {
  catalog_matches_rust_shader_set();
  shader_files_have_declared_entry_points();
  compute_catalog_matches_compute_shader_count();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
