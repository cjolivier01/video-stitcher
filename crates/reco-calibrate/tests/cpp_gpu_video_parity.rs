//! Cross-language golden for the C++ CUDA decoded-video calibration test.

#![cfg(feature = "io")]

use std::path::{Path, PathBuf};

use reco_calibrate::{CalibrationConfig, CalibrationResult, calibrate};
use reco_core::calibration::CameraParams;
use reco_core::gpu::GpuContext;
use reco_io::ffmpeg::calibration_io;
use serde::{Deserialize, Serialize};

const FRAME_INDICES: [u64; 3] = [1, 3, 5];
const FRAME_PTS_NS: [u64; 3] = [100_000_000, 300_000_000, 500_000_000];

#[derive(Debug, Serialize, Deserialize)]
struct ParityGolden {
    schema_version: u32,
    provenance: String,
    frame_indices: Vec<u64>,
    frame_pts_ns: Vec<u64>,
    result: CalibrationResult,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../cpp/tests/fixtures/calibration_video_parity")
        .join(name)
}

fn camera() -> CameraParams {
    CameraParams {
        width: 640,
        height: 360,
        fx: 1.0e10,
        fy: 1.0e10,
        cx: 320.0,
        cy: 180.0,
        d: [0.0; 4],
    }
}

fn config() -> CalibrationConfig {
    let mut config = CalibrationConfig {
        num_frames: FRAME_INDICES.len(),
        ..CalibrationConfig::default()
    };
    config.akaze.threshold = 0.0001;
    config.akaze.max_keypoints = 128;
    config.akaze.detect_y_min = 0.15;
    config.akaze.detect_y_max = 0.85;
    config.matching.lowe_ratio = 1.0;
    config.matching.min_matches = 8;
    config.matching.spatial_x_threshold = 0.5;
    config.matching.spatial_x_inner = 0.0;
    config.matching.spatial_y_low = 0.15;
    config.matching.spatial_y_high = 0.85;
    config.matching.max_y_disparity = 0.02;
    config.matching.ransac_threshold = 1.0;
    config.optimizer.max_iters = 5_000;
    config
}

fn extract(path: &Path) -> Vec<reco_calibrate::YuvFrame> {
    calibration_io::extract_frames(path, &FRAME_INDICES).expect("decode parity fixture")
}

fn point_matches(
    actual: &reco_calibrate::types::MatchedPoint,
    expected: &reco_calibrate::types::MatchedPoint,
) -> bool {
    let near = |lhs: f64, rhs: f64| (lhs - rhs).abs() <= 5.0e-4;
    near(actual.left[0], expected.left[0])
        && near(actual.left[1], expected.left[1])
        && near(actual.right[0], expected.right[0])
        && near(actual.right[1], expected.right[1])
        && near(actual.left_pixel_nx, expected.left_pixel_nx)
        && near(actual.right_pixel_nx, expected.right_pixel_nx)
}

fn assert_result_matches_golden(actual: &CalibrationResult, expected: &CalibrationResult) {
    assert_eq!(
        serde_json::to_value(&actual.calibration).unwrap(),
        serde_json::to_value(&expected.calibration).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&actual.left_lens_profile).unwrap(),
        serde_json::to_value(&expected.left_lens_profile).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&actual.right_lens_profile).unwrap(),
        serde_json::to_value(&expected.right_lens_profile).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&actual.quality).unwrap(),
        serde_json::to_value(&expected.quality).unwrap()
    );
    assert_eq!(actual.total_matches, expected.total_matches);
    assert_eq!(actual.frames_used, expected.frames_used);
    assert!((actual.residual_error - expected.residual_error).abs() <= 1.0e-12);
    assert!((actual.confidence - expected.confidence).abs() <= 1.0e-12);
    assert_eq!(actual.per_frame.len(), expected.per_frame.len());

    let actual_layout = &actual.calibration.layout;
    let expected_layout = &expected.calibration.layout;
    for (actual, expected) in [
        (
            actual_layout.camera_axis_offset,
            expected_layout.camera_axis_offset,
        ),
        (actual_layout.intersect, expected_layout.intersect),
        (actual_layout.x_ty, expected_layout.x_ty),
        (actual_layout.x_rz, expected_layout.x_rz),
        (actual_layout.z_rx, expected_layout.z_rx),
        (actual_layout.x_rx, expected_layout.x_rx),
        (actual_layout.z_rz, expected_layout.z_rz),
    ] {
        assert!((actual - expected).abs() <= 1.0e-12);
    }

    for (frame_index, (actual, expected)) in
        actual.per_frame.iter().zip(&expected.per_frame).enumerate()
    {
        assert_eq!(actual.keypoints_left, expected.keypoints_left);
        assert_eq!(actual.keypoints_right, expected.keypoints_right);
        assert_eq!(actual.min_descriptors, expected.min_descriptors);
        assert_eq!(actual.post_ratio_test, expected.post_ratio_test);
        assert_eq!(actual.post_spatial_filter, expected.post_spatial_filter);
        assert_eq!(actual.post_ransac, expected.post_ransac);
        assert_eq!(actual.points.len(), expected.points.len());
        let mut used = vec![false; actual.points.len()];
        for expected_point in &expected.points {
            let Some(index) =
                actual.points.iter().enumerate().position(|(index, point)| {
                    !used[index] && point_matches(point, expected_point)
                })
            else {
                panic!("frame {frame_index} is missing a Rust-golden correspondence");
            };
            used[index] = true;
        }
    }
}

#[test]
fn decoded_video_calibration_matches_cpp_golden() {
    let left = extract(&fixture_path("left.mp4"));
    let right = extract(&fixture_path("right.mp4"));
    assert_eq!(left.len(), FRAME_INDICES.len());
    assert_eq!(right.len(), FRAME_INDICES.len());
    for (index, expected_pts_ns) in FRAME_PTS_NS.into_iter().enumerate() {
        assert_eq!(left[index].timestamp_us * 1_000, expected_pts_ns as i64);
        assert_eq!(right[index].timestamp_us * 1_000, expected_pts_ns as i64);
    }

    let pairs: Vec<_> = left.into_iter().zip(right).collect();
    let gpu = GpuContext::new_blocking().expect("GPU required for parity reference");
    let first = calibrate(&gpu, &pairs, &camera(), &camera(), &config())
        .expect("Rust calibration reference");
    let second = calibrate(&gpu, &pairs, &camera(), &camera(), &config())
        .expect("repeated Rust calibration reference");
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap(),
        "Rust reference must be byte-deterministic"
    );

    let actual = ParityGolden {
        schema_version: 1,
        provenance: "Rust reco-calibrate decoded-video reference".into(),
        frame_indices: FRAME_INDICES.to_vec(),
        frame_pts_ns: FRAME_PTS_NS.to_vec(),
        result: first,
    };
    let golden_path = fixture_path("rust_golden.json");
    let expected: ParityGolden =
        serde_json::from_slice(&std::fs::read(&golden_path).expect("read parity golden"))
            .expect("parse parity golden");
    assert_eq!(actual.schema_version, expected.schema_version);
    assert_eq!(actual.provenance, expected.provenance);
    assert_eq!(actual.frame_indices, expected.frame_indices);
    assert_eq!(actual.frame_pts_ns, expected.frame_pts_ns);
    assert_result_matches_golden(&actual.result, &expected.result);
}
