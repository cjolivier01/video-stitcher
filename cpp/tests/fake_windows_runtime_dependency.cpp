#if defined(_WIN32)
#define RECO_TEST_EXPORT __declspec(dllexport)
#else
#define RECO_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" RECO_TEST_EXPORT int recoFakeWindowsRuntimeDependencyMarker() { return 41; }
