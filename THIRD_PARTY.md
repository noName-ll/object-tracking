# Third-party sources

- YOLOE visual-prompt models and the one-time exporter use Ultralytics 8.3.203: https://github.com/ultralytics/ultralytics . Ultralytics provides AGPL-3.0 and enterprise licensing options. The repository's Apache-2.0 license does not replace the licenses of Ultralytics or its model assets. YOLOE checkpoints and exported graphs are downloaded/generated separately and are not bundled in Git.
- ONNX Runtime is used via the Rust `ort` crate. The native runtime is loaded from `ORT_DYLIB_PATH`; it must support the v1.22 API or newer.
- Optional natural-image test fixture: https://github.com/ultralytics/assets/blob/main/im/bus.jpg . It is downloaded with SHA-256 verification, not bundled in Git, and is used only for an inference smoke test, not an accuracy benchmark.

- NanoTrack algorithm: HonglinChu, https://github.com/HonglinChu/NanoTrack . The upstream repository declares Apache-2.0. Its license is retained in LICENSE.
- NanoTrack v2 ONNX assets: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2 . Assets are downloaded separately, with SHA-256 verification; they are not bundled in Git. Review the upstream repository's terms for your intended use.
- Inference/postprocessing reference: OpenCV `modules/video/src/tracking/tracker_nano.cpp`, https://github.com/opencv/opencv/blob/4.x/modules/video/src/tracking/tracker_nano.cpp . Upstream authors: HongLinChu (1628464345@qq.com), OpenCV adaptation by ZihaoMu (zihaomu@outlook.com). OpenCV 4.5+ uses Apache-2.0.
- `src/nano.rs` is a Rust adaptation with explicit raw-probability gating, finite-value validation, affine crop sampling, and no automatic template adaptation. It differs from upstream C++ implementation and should not be assumed numerically identical.
- OpenCV, opencv-rust, r2r and other Cargo dependencies retain their respective upstream licenses. Cargo.lock records resolved Rust versions.

Repository code added for the Rust implementation is distributed under Apache-2.0; see LICENSE. No claim is made that the original reference image, a model prediction, or a tracking score establishes object identity.
