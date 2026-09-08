#if defined(_WIN32)
#define RECO_TEST_EXPORT __declspec(dllexport)
#else
#define RECO_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" int recoFakeWindowsRuntimeDependencyMarker();

extern "C" RECO_TEST_EXPORT int recoFakeWindowsRuntimeDependentMarker() {
  return recoFakeWindowsRuntimeDependencyMarker() + 1;
}
