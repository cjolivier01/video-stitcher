#include "reco/detect/ncnn_session.hpp"

#include "reco/core/path.hpp"
#include "reco/core/windows_runtime_library.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#else
#include <dlfcn.h>
#endif

namespace reco::detect {
namespace {

#if defined(_WIN32)
#define RECO_NCNN_CALL __stdcall
#else
#define RECO_NCNN_CALL
#endif

using NcnnNetCreate = void*(RECO_NCNN_CALL*)();
using NcnnNetDestroy = void(RECO_NCNN_CALL*)(void*);
using NcnnNetSetOption = void(RECO_NCNN_CALL*)(void*, void*);
using NcnnNetLoad = int(RECO_NCNN_CALL*)(void*, const char*);
using NcnnOptionCreate = void*(RECO_NCNN_CALL*)();
using NcnnOptionDestroy = void(RECO_NCNN_CALL*)(void*);
using NcnnOptionSetInt = void(RECO_NCNN_CALL*)(void*, int);
using NcnnExtractorCreate = void*(RECO_NCNN_CALL*)(void*);
using NcnnExtractorDestroy = void(RECO_NCNN_CALL*)(void*);
using NcnnExtractorInput = int(RECO_NCNN_CALL*)(void*, const char*, const void*);
using NcnnExtractorExtract = int(RECO_NCNN_CALL*)(void*, const char*, void**);
using NcnnMatCreate3d = void*(RECO_NCNN_CALL*)(int, int, int, void*);
using NcnnMatDestroy = void(RECO_NCNN_CALL*)(void*);
using NcnnMatGetInt = int(RECO_NCNN_CALL*)(const void*);
using NcnnMatGetSize = std::size_t(RECO_NCNN_CALL*)(const void*);
using NcnnMatGetData = void*(RECO_NCNN_CALL*)(const void*);
using NcnnMatGetChannelData = void*(RECO_NCNN_CALL*)(const void*, int);

class DynamicLibrary {
public:
  explicit DynamicLibrary(const std::filesystem::path& path) : path_(core::path_to_utf8(path)) {
#if defined(_WIN32)
    handle_ = static_cast<HMODULE>(core::detail::load_windows_runtime_library(path));
#else
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ == nullptr) {
      throw NcnnError("failed to load NCNN runtime `" + path_ + "`");
    }
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  ~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif
  }

  template <typename Fn> Fn symbol(const char* name) const {
#if defined(_WIN32)
    auto* sym = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    auto* sym = dlsym(handle_, name);
#endif
    if (sym == nullptr) {
      throw NcnnError("NCNN runtime `" + path_ + "` is missing " + name);
    }
    return reinterpret_cast<Fn>(sym);
  }

private:
  void* handle_ = nullptr;
  std::string path_;
};

struct NcnnApi {
  NcnnNetCreate net_create = nullptr;
  NcnnNetDestroy net_destroy = nullptr;
  NcnnNetSetOption net_set_option = nullptr;
  NcnnNetLoad net_load_param = nullptr;
  NcnnNetLoad net_load_model = nullptr;
  NcnnOptionCreate option_create = nullptr;
  NcnnOptionDestroy option_destroy = nullptr;
  NcnnOptionSetInt option_set_num_threads = nullptr;
  NcnnOptionSetInt option_set_use_vulkan_compute = nullptr;
  NcnnExtractorCreate extractor_create = nullptr;
  NcnnExtractorDestroy extractor_destroy = nullptr;
  NcnnExtractorInput extractor_input = nullptr;
  NcnnExtractorExtract extractor_extract = nullptr;
  NcnnMatCreate3d mat_create_3d = nullptr;
  NcnnMatDestroy mat_destroy = nullptr;
  NcnnMatGetInt mat_get_w = nullptr;
  NcnnMatGetInt mat_get_h = nullptr;
  NcnnMatGetSize mat_get_cstep = nullptr;
  NcnnMatGetData mat_get_data = nullptr;
  NcnnMatGetChannelData mat_get_channel_data = nullptr;
};

std::filesystem::path getenv_path(const char* name) {
  if (const char* value = std::getenv(name); value != nullptr && value[0] != '\0') {
    return std::filesystem::path(value);
  }
  return {};
}

bool contains_nul(std::string_view value) { return value.find('\0') != std::string_view::npos; }

const NcnnApi& ncnn_api() {
  static const NcnnApi api = [] {
    static std::unique_ptr<DynamicLibrary> pinned_library;
    auto path = getenv_path("NCNN_DYLIB_PATH");
    if (path.empty()) {
      path = getenv_path("RECO_NCNN_DYLIB_PATH");
    }
    if (path.empty()) {
#if defined(_WIN32)
      path = "ncnn.dll";
#elif defined(__APPLE__)
      path = "libncnn.dylib";
#else
      path = "libncnn.so";
#endif
    }
    pinned_library = std::make_unique<DynamicLibrary>(path);
    return NcnnApi{
        .net_create = pinned_library->symbol<NcnnNetCreate>("ncnn_net_create"),
        .net_destroy = pinned_library->symbol<NcnnNetDestroy>("ncnn_net_destroy"),
        .net_set_option = pinned_library->symbol<NcnnNetSetOption>("ncnn_net_set_option"),
        .net_load_param = pinned_library->symbol<NcnnNetLoad>("ncnn_net_load_param"),
        .net_load_model = pinned_library->symbol<NcnnNetLoad>("ncnn_net_load_model"),
        .option_create = pinned_library->symbol<NcnnOptionCreate>("ncnn_option_create"),
        .option_destroy = pinned_library->symbol<NcnnOptionDestroy>("ncnn_option_destroy"),
        .option_set_num_threads =
            pinned_library->symbol<NcnnOptionSetInt>("ncnn_option_set_num_threads"),
        .option_set_use_vulkan_compute =
            pinned_library->symbol<NcnnOptionSetInt>("ncnn_option_set_use_vulkan_compute"),
        .extractor_create = pinned_library->symbol<NcnnExtractorCreate>("ncnn_extractor_create"),
        .extractor_destroy = pinned_library->symbol<NcnnExtractorDestroy>("ncnn_extractor_destroy"),
        .extractor_input = pinned_library->symbol<NcnnExtractorInput>("ncnn_extractor_input"),
        .extractor_extract = pinned_library->symbol<NcnnExtractorExtract>("ncnn_extractor_extract"),
        .mat_create_3d = pinned_library->symbol<NcnnMatCreate3d>("ncnn_mat_create_3d"),
        .mat_destroy = pinned_library->symbol<NcnnMatDestroy>("ncnn_mat_destroy"),
        .mat_get_w = pinned_library->symbol<NcnnMatGetInt>("ncnn_mat_get_w"),
        .mat_get_h = pinned_library->symbol<NcnnMatGetInt>("ncnn_mat_get_h"),
        .mat_get_cstep = pinned_library->symbol<NcnnMatGetSize>("ncnn_mat_get_cstep"),
        .mat_get_data = pinned_library->symbol<NcnnMatGetData>("ncnn_mat_get_data"),
        .mat_get_channel_data =
            pinned_library->symbol<NcnnMatGetChannelData>("ncnn_mat_get_channel_data"),
    };
  }();
  return api;
}

class MatHandle {
public:
  explicit MatHandle(void* mat = nullptr) : mat_(mat) {}
  ~MatHandle() {
    if (mat_ != nullptr) {
      ncnn_api().mat_destroy(mat_);
    }
  }
  MatHandle(const MatHandle&) = delete;
  MatHandle& operator=(const MatHandle&) = delete;
  [[nodiscard]] void* get() const { return mat_; }
  [[nodiscard]] void** out() { return &mat_; }

private:
  void* mat_ = nullptr;
};

class ExtractorHandle {
public:
  explicit ExtractorHandle(void* extractor = nullptr) : extractor_(extractor) {}
  ~ExtractorHandle() {
    if (extractor_ != nullptr) {
      ncnn_api().extractor_destroy(extractor_);
    }
  }
  ExtractorHandle(const ExtractorHandle&) = delete;
  ExtractorHandle& operator=(const ExtractorHandle&) = delete;
  [[nodiscard]] void* get() const { return extractor_; }

private:
  void* extractor_ = nullptr;
};

} // namespace

struct NcnnSession::Impl {
  explicit Impl(NcnnSessionConfig config)
      : input_name(std::move(config.input_name)), output_name(std::move(config.output_name)) {
    if (config.num_threads <= 0) {
      throw NcnnError("NCNN num_threads must be positive");
    }
    if (contains_nul(input_name) || contains_nul(output_name)) {
      throw NcnnError("NCNN layer names must not contain NUL bytes");
    }
    const auto param_path = config.model_dir / "model.ncnn.param";
    const auto bin_path = config.model_dir / "model.ncnn.bin";
    if (!std::filesystem::exists(param_path)) {
      throw NcnnError("NCNN param not found: `" + param_path.string() + "`");
    }
    if (!std::filesystem::exists(bin_path)) {
      throw NcnnError("NCNN bin not found: `" + bin_path.string() + "`");
    }
    const auto param_string = param_path.string();
    const auto bin_string = bin_path.string();
    if (contains_nul(param_string) || contains_nul(bin_string)) {
      throw NcnnError("NCNN model paths must not contain NUL bytes");
    }
    const auto& api = ncnn_api();
    option = api.option_create();
    if (option == nullptr) {
      throw NcnnError("failed to create NCNN option");
    }
    api.option_set_num_threads(option, config.num_threads);
    api.option_set_use_vulkan_compute(option, 0);
    net = api.net_create();
    if (net == nullptr) {
      api.option_destroy(option);
      option = nullptr;
      throw NcnnError("failed to create NCNN net");
    }
    api.net_set_option(net, option);
    if (api.net_load_param(net, param_string.c_str()) != 0) {
      api.net_destroy(net);
      api.option_destroy(option);
      net = nullptr;
      option = nullptr;
      throw NcnnError("failed to load NCNN param");
    }
    if (api.net_load_model(net, bin_string.c_str()) != 0) {
      api.net_destroy(net);
      api.option_destroy(option);
      net = nullptr;
      option = nullptr;
      throw NcnnError("failed to load NCNN model");
    }
  }

  ~Impl() {
    const auto& api = ncnn_api();
    if (net != nullptr) {
      api.net_destroy(net);
    }
    if (option != nullptr) {
      api.option_destroy(option);
    }
  }

  void* net = nullptr;
  void* option = nullptr;
  std::string input_name;
  std::string output_name;
};

NcnnSession::NcnnSession(NcnnSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
NcnnSession::~NcnnSession() = default;
NcnnSession::NcnnSession(NcnnSession&&) noexcept = default;
NcnnSession& NcnnSession::operator=(NcnnSession&&) noexcept = default;

NcnnTensorOutput NcnnSession::run_preprocessed_chw(std::span<const float> input,
                                                   std::uint32_t input_size) {
  if (input_size == 0) {
    throw NcnnError("NCNN input size must be non-zero");
  }
  const auto sz = static_cast<std::size_t>(input_size);
  if (sz > std::numeric_limits<std::size_t>::max() / sz / 3U) {
    throw NcnnError("NCNN input dimensions overflow");
  }
  const auto element_count = 3U * sz * sz;
  if (input.size() != element_count) {
    throw NcnnError("NCNN input span size does not match [3,S,S]");
  }
  if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw NcnnError("NCNN input element count exceeds C API limit");
  }
  const auto& api = ncnn_api();
  MatHandle input_mat(
      api.mat_create_3d(static_cast<int>(input_size), static_cast<int>(input_size), 3, nullptr));
  if (input_mat.get() == nullptr) {
    throw NcnnError("failed to create NCNN input mat");
  }
  const auto plane_size = sz * sz;
  const auto cstep = api.mat_get_cstep(input_mat.get());
  if (cstep < plane_size) {
    throw NcnnError("NCNN input mat channel step is smaller than one plane");
  }
  for (int channel = 0; channel < 3; ++channel) {
    auto* channel_data = static_cast<float*>(api.mat_get_channel_data(input_mat.get(), channel));
    if (channel_data == nullptr) {
      throw NcnnError("NCNN input mat channel data pointer is null");
    }
    const auto offset = static_cast<std::size_t>(channel) * plane_size;
    std::memcpy(channel_data, input.data() + offset, plane_size * sizeof(float));
  }
  ExtractorHandle extractor(api.extractor_create(impl_->net));
  if (extractor.get() == nullptr) {
    throw NcnnError("failed to create NCNN extractor");
  }
  if (api.extractor_input(extractor.get(), impl_->input_name.c_str(), input_mat.get()) != 0) {
    throw NcnnError("NCNN extractor input failed");
  }
  MatHandle output_mat;
  if (api.extractor_extract(extractor.get(), impl_->output_name.c_str(), output_mat.out()) != 0) {
    throw NcnnError("NCNN extractor extract failed");
  }
  if (output_mat.get() == nullptr) {
    throw NcnnError("NCNN extractor returned a null output mat");
  }
  const auto width = api.mat_get_w(output_mat.get());
  const auto height = api.mat_get_h(output_mat.get());
  if (width < 0 || height < 0) {
    throw NcnnError("NCNN output shape is negative");
  }
  const auto w = static_cast<std::size_t>(width);
  const auto h = static_cast<std::size_t>(height);
  if (w != 0 && h > std::numeric_limits<std::size_t>::max() / w) {
    throw NcnnError("NCNN output dimensions overflow");
  }
  const auto* data = static_cast<const float*>(api.mat_get_data(output_mat.get()));
  const auto count = w * h;
  if (count != 0 && data == nullptr) {
    throw NcnnError("NCNN output data pointer is null");
  }
  NcnnTensorOutput output{.width = width, .height = height};
  if (count != 0) {
    output.data.assign(data, data + count);
  }
  return output;
}

bool ncnn_runtime_available() {
  try {
    (void)ncnn_api();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

std::string ncnn_runtime_error() {
  try {
    (void)ncnn_api();
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}

} // namespace reco::detect
