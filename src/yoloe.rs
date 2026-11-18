//! Target-independent visual-prompt YOLOE ONNX inference. No Python runtime.
use crate::contract::Bbox;
use anyhow::{ensure, Context, Result};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};
use ort::{session::Session, value::Tensor};

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub bbox: Bbox,
    pub score: f32,
}

#[derive(Debug)]
pub enum SearchResult {
    Found(Candidate),
    Missing,
    Ambiguous,
}

#[derive(Clone, Copy, Debug)]
pub struct Letterbox {
    pub size: usize,
    pub width: i32,
    pub height: i32,
    pub resized_width: i32,
    pub resized_height: i32,
    pub left: i32,
    pub top: i32,
}
impl Letterbox {
    pub fn new(width: i32, height: i32, size: usize) -> Self {
        let gain = (size as f64 / width as f64).min(size as f64 / height as f64);
        let resized_width = (width as f64 * gain).round_ties_even() as i32;
        let resized_height = (height as f64 * gain).round_ties_even() as i32;
        Self {
            size,
            width,
            height,
            resized_width,
            resized_height,
            left: (size as i32 - resized_width) / 2,
            top: (size as i32 - resized_height) / 2,
        }
    }
    pub fn unproject(&self, bbox: Bbox) -> Option<Bbox> {
        // Use actual rounded dimensions to avoid accumulating resize rounding errors.
        let sx = self.width as f32 / self.resized_width as f32;
        let sy = self.height as f32 / self.resized_height as f32;
        Bbox {
            x: (bbox.x - self.left as f32) * sx,
            y: (bbox.y - self.top as f32) * sy,
            width: bbox.width * sx,
            height: bbox.height * sy,
        }
        .clip(self.width, self.height)
    }
    /// Binary mask at P3 resolution, covering only the reference object (not padding).
    pub fn prompt_mask(&self) -> Vec<f32> {
        let side = self.size / 8;
        (0..side * side)
            .map(|i| {
                let x = (i % side) as f32;
                let y = (i / side) as f32;
                if x >= self.left as f32 / 8.0
                    && x < (self.left + self.resized_width) as f32 / 8.0
                    && y >= self.top as f32 / 8.0
                    && y < (self.top + self.resized_height) as f32 / 8.0
                {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }
}

fn preprocess(image: &Mat, size: usize) -> Result<(Vec<f32>, Letterbox)> {
    ensure!(
        !image.empty() && image.typ() == core::CV_8UC3,
        "YOLOE expects BGR8"
    );
    let transform = Letterbox::new(image.cols(), image.rows(), size);
    let mut resized = Mat::default();
    imgproc::resize(
        image,
        &mut resized,
        Size::new(transform.resized_width, transform.resized_height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let mut padded = Mat::new_rows_cols_with_default(
        size as i32,
        size as i32,
        core::CV_8UC3,
        Scalar::all(114.0),
    )?;
    resized.copy_to(&mut Mat::roi_mut(
        &mut padded,
        Rect::new(
            transform.left,
            transform.top,
            transform.resized_width,
            transform.resized_height,
        ),
    )?)?;
    let mut data = vec![0.0; 3 * size * size];
    for (i, pixel) in padded.data_bytes()?.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            data[c * size * size + i] = pixel[2 - c] as f32 / 255.0;
        }
    }
    Ok((data, transform))
}

pub struct YoloE {
    encoder: Session,
    detector: Session,
    size: usize,
    score: f32,
    iou: f32,
    margin: f32,
}
impl YoloE {
    pub fn load(
        encoder: &str,
        detector: &str,
        size: usize,
        score: f32,
        iou: f32,
        margin: f32,
    ) -> Result<Self> {
        ensure!(
            (64..=2048).contains(&size) && size.is_multiple_of(32),
            "YOLOE size must be 64..2048 and divisible by 32"
        );
        ensure!(
            [score, iou, margin]
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "invalid YOLOE thresholds"
        );
        // Dynamic loading keeps deployment independent of a Python installation.
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
            ensure!(
                std::path::Path::new(&path).is_file(),
                "ORT_DYLIB_PATH must point to libonnxruntime.so (or platform equivalent)"
            );
            ort::init_from(path)
                .with_name("mobile-object-tracker")
                .commit()?;
        } else {
            anyhow::bail!(
                "set ORT_DYLIB_PATH to an ONNX Runtime >=1.22 shared library; see README"
            );
        }
        let encoder = Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(encoder)
            .context("load YOLOE encoder; run scripts/export_yoloe.py")?;
        let detector = Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(detector)
            .context("load YOLOE detector; run scripts/export_yoloe.py")?;
        ensure!(
            encoder.inputs.len() == 2
                && encoder.inputs[0].name == "images"
                && encoder.inputs[1].name == "visuals",
            "encoder must use the two-input visual-prompt export contract"
        );
        ensure!(
            detector.inputs.len() == 2
                && detector.inputs[0].name == "images"
                && detector.inputs[1].name == "embeddings",
            "detector must accept runtime embeddings; fixed-class YOLO exports are not supported"
        );
        ensure!(
            encoder.outputs.len() == 1
                && encoder.outputs[0].name == "embeddings"
                && detector.outputs.len() == 1
                && detector.outputs[0].name == "detections",
            "unexpected YOLOE output contract; use scripts/export_yoloe.py"
        );
        let image_shape = [1, 3, size as i64, size as i64];
        let mask_shape = [1, 1, (size / 8) as i64, (size / 8) as i64];
        for (input, expected) in [
            (&encoder.inputs[0], &image_shape[..]),
            (&encoder.inputs[1], &mask_shape[..]),
            (&detector.inputs[0], &image_shape[..]),
            (&detector.inputs[1], &[1, 1, 512][..]),
        ] {
            ensure!(
                input
                    .input_type
                    .tensor_shape()
                    .is_some_and(|shape| shape.as_ref() == expected),
                "YOLOE input shape mismatch for {}; --yoloe-size must match the export size",
                input.name
            );
        }
        Ok(Self {
            encoder,
            detector,
            size,
            score,
            iou,
            margin,
        })
    }

    pub fn encode(&mut self, image: &Mat) -> Result<Vec<f32>> {
        let (pixels, transform) = preprocess(image, self.size)?;
        let mask = transform.prompt_mask();
        ensure!(
            mask.iter().any(|v| *v > 0.0),
            "reference is too thin for a visual prompt"
        );
        let outputs = self.encoder.run(ort::inputs![
            "images"=>Tensor::from_array(([1,3,self.size,self.size],pixels))?,
            "visuals"=>Tensor::from_array(([1,1,self.size/8,self.size/8],mask))?
        ])?;
        let (shape, embedding) = outputs["embeddings"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, 1, 512] && embedding.iter().all(|v| v.is_finite()),
            "invalid YOLOE embedding"
        );
        ensure!(
            embedding.iter().map(|v| v * v).sum::<f32>() > 1e-6,
            "empty YOLOE embedding"
        );
        Ok(embedding.to_vec())
    }

    pub fn search(&mut self, image: &Mat, embedding: &[f32]) -> Result<SearchResult> {
        let (pixels, transform) = preprocess(image, self.size)?;
        let outputs = self.detector.run(ort::inputs![
            "images"=>Tensor::from_array(([1,3,self.size,self.size],pixels))?,
            "embeddings"=>Tensor::from_array(([1,1,512],embedding.to_vec()))?
        ])?;
        let (shape, values) = outputs["detections"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[2] == 5,
            "expected YOLOE [1,N,5] decoded predictions"
        );
        Ok(select(
            values,
            &transform,
            self.score,
            self.iou,
            self.margin,
        ))
    }
}

pub fn select(
    values: &[f32],
    transform: &Letterbox,
    minimum: f32,
    iou: f32,
    margin: f32,
) -> SearchResult {
    let mut candidates = values
        .as_chunks::<5>()
        .0
        .iter()
        .filter_map(|row| {
            if !row.iter().all(|v| v.is_finite()) || row[4] < minimum || row[4] > 1.0 {
                return None;
            }
            let bbox = transform.unproject(Bbox {
                x: row[0] - row[2] / 2.0,
                y: row[1] - row[3] / 2.0,
                width: row[2],
                height: row[3],
            })?;
            (bbox.width >= 5.0 && bbox.height >= 5.0).then_some(Candidate {
                bbox,
                score: row[4],
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    // Class-agnostic NMS for one visual prompt; bounded to the top 300 proposals.
    candidates.truncate(300);
    let mut retained: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        if retained
            .iter()
            .all(|other| candidate.bbox.iou(other.bbox) <= iou)
        {
            retained.push(candidate);
        }
    }
    match retained.as_slice() {
        [] => SearchResult::Missing,
        [first, second, ..] if first.score - second.score < margin => SearchResult::Ambiguous,
        [first, ..] => SearchResult::Found(*first),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_is_excluded_from_prompt_and_coordinates_are_restored() {
        let t = Letterbox::new(800, 400, 640);
        assert_eq!(t.top, 160);
        assert_eq!(t.prompt_mask().iter().sum::<f32>(), 3200.0);
        let b = t
            .unproject(Bbox {
                x: 80.0,
                y: 200.0,
                width: 160.0,
                height: 80.0,
            })
            .unwrap();
        assert_eq!(
            b,
            Bbox {
                x: 100.0,
                y: 50.0,
                width: 200.0,
                height: 100.0
            }
        );
    }
    #[test]
    fn nms_filters_duplicates_and_ambiguous_instances() {
        let t = Letterbox::new(640, 640, 640);
        let duplicate = [
            100.0, 100.0, 50.0, 50.0, 0.9, 101.0, 101.0, 50.0, 50.0, 0.89,
        ];
        assert!(matches!(
            select(&duplicate, &t, 0.25, 0.5, 0.05),
            SearchResult::Found(_)
        ));
        let multiple = [
            100.0, 100.0, 50.0, 50.0, 0.9, 500.0, 500.0, 50.0, 50.0, 0.89,
        ];
        assert!(matches!(
            select(&multiple, &t, 0.25, 0.5, 0.05),
            SearchResult::Ambiguous
        ));
        assert!(matches!(
            select(&[f32::NAN; 5], &t, 0.25, 0.5, 0.05),
            SearchResult::Missing
        ));
    }
}
