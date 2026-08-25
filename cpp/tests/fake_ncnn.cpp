#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#define RECO_NCNN_CALL __stdcall
#define RECO_NCNN_EXPORT extern "C" __declspec(dllexport)
#else
#define RECO_NCNN_CALL
#define RECO_NCNN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct Option {
  int threads = 0;
  int vulkan = 0;
};

struct Net {
  Option* option = nullptr;
  bool loaded_param = false;
  bool loaded_model = false;
};

struct Mat {
  int w = 0;
  int h = 0;
  int c = 0;
  std::size_t cstep = 0;
  std::vector<float> data;
};

struct Extractor {
  Net* net = nullptr;
  Mat* input = nullptr;
  std::string input_name;
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "0") != 0;
}

} // namespace

RECO_NCNN_EXPORT void* RECO_NCNN_CALL ncnn_net_create() {
  if (env_set("RECO_FAKE_NCNN_NET_CREATE_FAIL")) {
    return nullptr;
  }
  return new Net;
}

RECO_NCNN_EXPORT void RECO_NCNN_CALL ncnn_net_destroy(void* net) {
  delete static_cast<Net*>(net);
}

RECO_NCNN_EXPORT void RECO_NCNN_CALL ncnn_net_set_option(void* net, void* option) {
  static_cast<Net*>(net)->option = static_cast<Option*>(option);
}

RECO_NCNN_EXPORT int RECO_NCNN_CALL ncnn_net_load_param(void* net, const char* path) {
  if (env_set("RECO_FAKE_NCNN_PARAM_FAIL") || path == nullptr) {
    return 1;
  }
  static_cast<Net*>(net)->loaded_param = true;
  return 0;
}

RECO_NCNN_EXPORT int RECO_NCNN_CALL ncnn_net_load_model(void* net, const char* path) {
  if (env_set("RECO_FAKE_NCNN_MODEL_FAIL") || path == nullptr) {
    return 1;
  }
  static_cast<Net*>(net)->loaded_model = true;
  return 0;
}

RECO_NCNN_EXPORT void* RECO_NCNN_CALL ncnn_option_create() {
  if (env_set("RECO_FAKE_NCNN_OPTION_CREATE_FAIL")) {
    return nullptr;
  }
  return new Option;
}

RECO_NCNN_EXPORT void RECO_NCNN_CALL ncnn_option_destroy(void* option) {
  delete static_cast<Option*>(option);
}

RECO_NCNN_EXPORT void RECO_NCNN_CALL ncnn_option_set_num_threads(void* option, int num_threads) {
  static_cast<Option*>(option)->threads = num_threads;
}

RECO_NCNN_EXPORT void RECO_NCNN_CALL ncnn_option_set_use_vulkan_compute(void* option,
                                                                        int use_vulkan) {
  static_cast<Option*>(option)->vulkan = use_vulkan;
}

RECO_NCNN_EXPORT void* RECO_NCNN_CALL ncnn_extractor_create(void* net) {
  if (env_set("RECO_FAKE_NCNN_EXTRACTOR_CREATE_FAIL")) {
    return nullptr;
  }
  return new Extractor{static_cast<Net*>(net)};
}

RECO_NCNN_EXPORT void RECO_NCNN_CALL ncnn_extractor_destroy(void* extractor) {
  delete static_cast<Extractor*>(extractor);
}

RECO_NCNN_EXPORT int RECO_NCNN_CALL ncnn_extractor_input(void* extractor, const char* name,
                                                        const void* mat) {
  if (env_set("RECO_FAKE_NCNN_INPUT_FAIL") || name == nullptr || mat == nullptr) {
    return 1;
  }
  auto* ex = static_cast<Extractor*>(extractor);
  ex->input = const_cast<Mat*>(static_cast<const Mat*>(mat));
  ex->input_name = name;
  return 0;
}

RECO_NCNN_EXPORT int RECO_NCNN_CALL ncnn_extractor_extract(void* extractor, const char* name,
                                                          void** mat) {
  if (env_set("RECO_FAKE_NCNN_EXTRACT_FAIL") || name == nullptr || mat == nullptr) {
    return 1;
  }
  if (env_set("RECO_FAKE_NCNN_NULL_OUTPUT")) {
    *mat = nullptr;
    return 0;
  }
  auto* ex = static_cast<Extractor*>(extractor);
  if (ex->net == nullptr || !ex->net->loaded_param || !ex->net->loaded_model ||
      ex->input == nullptr || ex->input_name != "in0" || std::strcmp(name, "out0") != 0) {
    return 2;
  }
  if (ex->input->c != 3 || ex->input->cstep < static_cast<std::size_t>(ex->input->w) *
                                                     static_cast<std::size_t>(ex->input->h)) {
    return 3;
  }
  const auto plane = static_cast<std::size_t>(ex->input->w) * static_cast<std::size_t>(ex->input->h);
  const auto* ch0 = ex->input->data.data();
  const auto* ch1 = ex->input->data.data() + ex->input->cstep;
  const auto* ch2 = ex->input->data.data() + (2U * ex->input->cstep);
  auto* output = new Mat{2, 5, 1, 10};
  output->data = {
      ch0[0], ch1[0],
      ch2[0], ch1[plane - 1],
      0.10F, 0.20F,
      0.10F, 0.20F,
      0.90F, 0.80F,
  };
  *mat = output;
  return 0;
}

RECO_NCNN_EXPORT void* RECO_NCNN_CALL ncnn_mat_create_3d(int w, int h, int c, void*) {
  if (w <= 0 || h <= 0 || c <= 0 || env_set("RECO_FAKE_NCNN_MAT_CREATE_FAIL")) {
    return nullptr;
  }
  const auto plane = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  const auto cstep = ((plane + 7U) / 8U) * 8U;
  auto* mat = new Mat{w, h, c, cstep};
  mat->data.assign(cstep * static_cast<std::size_t>(c), 0.0F);
  return mat;
}

RECO_NCNN_EXPORT void RECO_NCNN_CALL ncnn_mat_destroy(void* mat) {
  delete static_cast<Mat*>(mat);
}

RECO_NCNN_EXPORT int RECO_NCNN_CALL ncnn_mat_get_w(const void* mat) {
  return static_cast<const Mat*>(mat)->w;
}

RECO_NCNN_EXPORT int RECO_NCNN_CALL ncnn_mat_get_h(const void* mat) {
  return static_cast<const Mat*>(mat)->h;
}

RECO_NCNN_EXPORT std::size_t RECO_NCNN_CALL ncnn_mat_get_cstep(const void* mat) {
  return static_cast<const Mat*>(mat)->cstep;
}

RECO_NCNN_EXPORT void* RECO_NCNN_CALL ncnn_mat_get_data(const void* mat) {
  return const_cast<float*>(static_cast<const Mat*>(mat)->data.data());
}

RECO_NCNN_EXPORT void* RECO_NCNN_CALL ncnn_mat_get_channel_data(const void* mat, int channel) {
  const auto* typed = static_cast<const Mat*>(mat);
  if (channel < 0 || channel >= typed->c) {
    return nullptr;
  }
  return const_cast<float*>(typed->data.data() + (static_cast<std::size_t>(channel) * typed->cstep));
}
