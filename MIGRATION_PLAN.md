# C++ Migration Plan

This plan ports the Rust workspace to C++ in stages while preserving release,
test, performance, and security behavior. The Rust implementation remains the
reference until the C++ implementation passes the parity gates below.

## Non-negotiable Gates

- No Rust crate, Cargo metadata, shader, fixture, or release path is removed
  until the matching C++ module passes parity tests and review.
- Every stage ports the module code, its tests, and any relevant fuzz or smoke
  coverage before the stage is considered complete.
- Each stage gets two independent review-agent rounds. A stage only advances
  after both reviewers report no findings or all findings are fixed and
  re-reviewed.
- Platform-specific skips must be listed in the parity matrix with a concrete
  reason and a replacement check.
- C++ builds use sanitizers in CI: ASan/UBSan for regular test targets, TSan
  for thread-heavy units where dependency support allows it, and libFuzzer for
  external-input parsers.

## Target C++ Tree

```text
cpp/
  reco_core/
  reco_io/
  reco_detect/
  reco_autocam/
  reco_calibrate/
  reco_control/
  apps/reco_cli/
  apps/reco_gui_qt/
  plugins/reco_obs/
  tests/fixtures/
  fuzz/
```

Root build files:

- `MODULE.bazel`, `.bazelrc`, and Bazel `BUILD.bazel` files
- Bazel test integration
- Bazel `cc_test` targets, with GoogleTest/Catch2 added only through
  Bazel-compatible wrappers when a stage needs richer assertions
- `clang-format` and `clang-tidy`
- Bazel config flags for FFmpeg, GStreamer, libcamera, V4L2, CUDA/NVCC,
  TensorRT, ONNX Runtime, NCNN, Qt, OBS, and profiling

## GPU Direction

The initial C++ renderer keeps the current WGSL shader assets and uses a
WebGPU-capable abstraction such as Dawn or wgpu-native so Vulkan, D3D12, Metal,
and Linux portability are retained. CUDA/NVCC is used where the current code
already relies on CUDA/NPP/TensorRT or where profiling proves a kernel should
move. The renderer must not become CUDA-only, and GPU work must not be ported
by replacing it with CPU processing. Any CPU fallback must be explicit,
feature-gated or error-reported, tested as a fallback path, and disallowed in
GPU parity/performance gates.

Renderer parity requires bounded pixel-diff or golden-frame tests for:

- fisheye stitch
- cylindrical mono
- Bayer demosaic
- color grade
- RGBA to NV12
- YUV420P stack packing
- RGBA readback

GPU smoke tests are not enough. The C++ output must be compared against Rust
fixtures on supported devices with explicit tolerances.

## Release And Packaging Parity

The Bazel release workflow must cover the existing release targets:

- Linux x86_64
- Linux arm64
- macOS arm64
- macOS x86_64
- Windows x86_64 MSVC

The release workflow must preserve dependency setup and packaging behavior:

- FFmpeg development libraries and runtime packaging
- rpath or loader path handling for bundled shared libraries
- ONNX Runtime 1.24.x packaging, including DirectML and `DirectML.dll` on
  Windows
- app/plugin resources, licenses, notices, shader assets, calibration data, and
  Qt runtime dependencies

## Security And Fuzz Parity

The current fuzz targets are blocker gates for Rust removal and must be ported
to C++ libFuzzer targets:

- `calibration_json`
- `onnx_names`
- `input_path`

The C++ targets must preserve input-size caps and validation intent. CI should
run bounded fuzz jobs and sanitizer builds. Any parser added during the C++
port gets a fuzz target if it accepts untrusted model metadata, calibration
data, paths, or network/device input.

## Module Stages

### Stage 0: Build, Fixtures, And Parity Matrix

- Add the C++/Bazel scaffold without changing Rust behavior.
- Add shared test fixtures generated from the Rust implementation.
- Add a parity matrix tracking APIs, tests, fuzz targets, platform support,
  packaging, performance gates, and known skips.
- Add a Bazel CI workflow that builds and tests scaffold targets.

### Stage 1: `reco-control`

- Port control intent vocabulary, JSON compatibility, pose control, keyboard
  transport, intent translator, and GoPro control helpers.
- Preserve JSON wire names and units.
- Add C++ tests for all existing Rust tests plus JSON golden vectors.
- Add GoPro HTTP/status parsing tests using mocked responses.

### Stage 2: Pure `reco-core` Data And Math

- Port calibration schema, validation, source path validation, projection math,
  lens preview/undistort/rig correction, telemetry, pipeline events, replay
  buffer policy, and non-GPU frame/source types.
- Add JSON round-trip and golden compatibility tests.
- Port `input_path` and `calibration_json` fuzz targets.

### Stage 3: `reco-io` Pure And File I/O

- Port settings, JSONL sink, stitch job parsing, output config, stacked-video
  pure pack/unpack, FFmpeg decode/encode, GStreamer ingest, libcamera, V4L2,
  smart source/adapters, I/O zero-copy, and stacked source/encoder.
- Use RAII wrappers for FFmpeg frames, packets, contexts, hardware frames, and
  error paths.
- Add round-trip tests using generated tiny video assets.

### Stage 4: Calibration Algorithms

- Port sampling, RANSAC, filters, geometry, optimizer, AKAZE, lens database,
  Gyroflow `profiles.cbor.gz` resource handling, telemetry parsing, video
  sampling, live calibration traits, and audio sync.
- Use Eigen/OpenCV/Ceres or NLopt only where they reduce risk without changing
  observable behavior.
- Add tolerance-based fixture tests for feature detection, optimization, and
  audio-sync outputs.

### Stage 5: Detection

- Port detector interfaces, YOLO preprocessing, postprocessing, class parsing,
  ONNX Runtime CPU/CUDA/CoreML/DirectML configuration, TensorRT native wrapper,
  NCNN backend, CUDA/NPP helpers, and Metal/CoreML guards.
- Port `onnx_names` fuzz target.
- Add backend availability probes that fail gracefully when dependencies are
  absent.

### Stage 6: Autocam

- Port tracking modes, ROI filtering, ball tracker, class provider, coaster
  filter, sweep/file/field panners, trajectory smoothing, density clustering,
  and WGPU detector adapter.
- Add fixture tests for panner decisions and lookahead behavior.

### Stage 7: GPU Core, Session, And Zero-Copy

- Port the C++ equivalent of `StitchCore`, `StitchSession`,
  `submit_frame_yuv`, `submit_frame_bgra`, replay management, frame buffers,
  VRAM pool, detection dispatch, run loop, async encode, and renderer pipeline.
- Port platform interop for CUDA, Vulkan, DMA-BUF, D3D11, Metal,
  VideoToolbox/CVPixelBuffer, Linux shared texture slots, backpressure,
  full-range YUV, rotation metadata, and texture lifetime rules.
- Add platform acceptance tests that detect CPU-copy fallback, wrong texture
  formats, lifetime bugs, and throughput regressions.
- Add performance gates for zero-copy paths and profiling trace generation.

### Stage 8: Apps And Plugins

- Port `reco-cli` with a C++ CLI parser. The command surface must match Rust
  help text and behavior for stitch, info, calibrate, preview, camera,
  replay, lookahead, trajectory CSV, event JSONL, RTMP, constrained look,
  V4L2 direct Bayer controls, live calibration, libcamera, and feature-gated
  paths.
- Add generated help-text parity tests and a golden CLI behavior matrix.
- Port `reco-gui` to Qt while consuming the C++ libraries. Preserve export,
  preview, settings, telemetry/client diagnostics, toast behavior, ROI editing,
  automation hooks, and asset packaging.
- Port `reco-obs` as a native C++ libobs plugin. Build against supported OBS
  header versions and smoke-test loading against matching OBS/libobs versions.
  Preserve BGRA source behavior, hotkeys, interactive pan/zoom, logging, and
  replay option.

### Stage 9: Profiling, Diagnostics, And Release Cutover

- Add C++ profiling equivalent to the Rust `profiling` feature, including
  per-frame spans/events and trace output usable for performance triage.
- Preserve log filtering and diagnostic bundle value.
- Run full Rust and C++ suites side by side.
- Compare release artifacts across all supported targets.
- Remove Rust only after the parity matrix has no unresolved blocker gaps and
  both review agents approve the cutover.

## Per-Stage Review Protocol

For every implementation stage:

1. Implement the C++ module slice.
2. Port existing tests and add edge tests for C++ ownership/lifetime failures.
3. Run Bazel build/test, sanitizers where applicable, fuzz targets where
   applicable, and the relevant Rust reference tests.
4. Ask two review agents to inspect the stage for correctness, parity gaps,
   missing tests, platform regressions, and performance/security risks.
5. Fix all findings.
6. Repeat review until both agents approve.
