# C++ Port Parity Matrix

This matrix tracks Rust-to-C++ cutover gates. `Blocked` means the Rust module
remains authoritative and must not be removed.

| Area | Rust Reference | C++ Target | Current Status | Required Gates |
|---|---|---|---|---|
| Build | Cargo workspace | Bazel modules/packages | In progress | Bazel build/test in CI, sanitizer configs |
| reco-control | `crates/reco-control` | `cpp/reco_control` | In progress | Intent JSON goldens, pose tests, keyboard tests, GoPro URL/status tests |
| reco-core data/math | `crates/reco-core` non-GPU types | `cpp/reco_core` | In progress for `ViewportPosition`, rig correction, source path validation, frame data, calibration validation/JSON, point-in-polygon, virtual-camera yaw/pitch, scene geometry, scalar KB4 fisheye projection, camera/panorama projection chains, viewport bounds, telemetry aggregation, frame timing micros, event frame-index vocabulary, replay buffer policy | Deferred: `CoverageBoundary`, GPU lens preview/undistort, WGSL/WebGPU renderer/session, replay management/recorder traits; add full lens/render GPU tests and fuzz parity runtime |
| reco-core GPU | wgpu/WGSL render/session/interop | WebGPU/WGSL C++ | Blocked | GPU golden pixel diffs, zero-copy tests, no CPU fallback in GPU gates |
| reco-io | FFmpeg/GStreamer/libcamera/V4L2/zero-copy | `cpp/reco_io` | Blocked | Decode/encode smoke tests, camera ingest tests, zero-copy perf gates |
| reco-calibrate | AKAZE/optimizer/audio/lens DB | `cpp/reco_calibrate` | Blocked | Fixtures, resource packaging, tolerance tests |
| reco-detect | ORT/TensorRT/NCNN/CoreML/CUDA/NPP | `cpp/reco_detect` | Blocked | Backend probes, postprocess tests, `onnx_names` fuzz |
| reco-autocam | panners/trackers/ROI | `cpp/reco_autocam` | Blocked | Panner/trajectory fixture tests |
| reco-cli | `reco` binary | `cpp/apps/reco_cli` | Blocked | Help-text parity, golden CLI behavior matrix |
| reco-gui | Slint app | Qt app | Blocked | UI workflow tests, asset/resource packaging |
| reco-obs | Rust cdylib + libobs bindings | C++ libobs plugin | Blocked | ABI/header-version smoke tests |
| Fuzz | `calibration_json`, `onnx_names`, `input_path` | C++ libFuzzer | In progress for `calibration_json` and `input_path` harness builds | Add `onnx_names`, run bounded libFuzzer jobs, sanitizer tested |
| Release | GitHub release workflow | Bazel release workflow | Blocked | Linux x86_64/arm64, macOS arm64/x86_64, Windows x86_64 packages |
