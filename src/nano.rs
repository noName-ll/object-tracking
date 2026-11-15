//! NanoTrack v2 ONNX inference and decoding, implemented in Rust.
//! Based on the NanoTrack algorithm (HonglinChu) and OpenCV's tracker_nano.cpp.
//! Modified: explicit raw-score gate, finite checks, Rust postprocessing, no template updates.
use crate::contract::Bbox;
use anyhow::{ensure, Result};
use opencv::{
    core::{self, Mat, MatTraitConst, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
};

pub struct NanoTrack {
    backbone: dnn::Net,
    head: dnn::Net,
    template: Option<Mat>,
    bbox: Option<Bbox>,
}

impl NanoTrack {
    pub fn load(backbone: &str, head: &str) -> Result<Self> {
        let mut backbone = dnn::read_net_from_onnx(backbone)?;
        let mut head = dnn::read_net_from_onnx(head)?;
        for net in [&mut backbone, &mut head] {
            net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
            net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
        }
        Ok(Self {
            backbone,
            head,
            template: None,
            bbox: None,
        })
    }

    pub fn clear(&mut self) {
        self.bbox = None;
        self.template = None;
    }

    fn feature(&mut self, image: &Mat) -> Result<Mat> {
        let blob = dnn::blob_from_image(
            image,
            1.0,
            Size::default(),
            Scalar::default(),
            true,
            false,
            core::CV_32F,
        )?;
        self.backbone.set_input(&blob, "", 1.0, Scalar::default())?;
        Ok(self.backbone.forward_single_def()?)
    }

    pub fn init(&mut self, image: &Mat, bbox: Bbox) -> Result<()> {
        let bbox = bbox
            .clip(image.cols(), image.rows())
            .ok_or_else(|| anyhow::anyhow!("invalid NanoTrack init box"))?;
        let side = context_size(bbox.width, bbox.height);
        let crop = crop_square(
            image,
            bbox.x + bbox.width / 2.0,
            bbox.y + bbox.height / 2.0,
            side,
            127,
        )?;
        self.template = Some(self.feature(&crop)?);
        self.bbox = Some(bbox);
        Ok(())
    }

    /// A low-score candidate is not committed to internal tracking state.
    pub fn update(&mut self, image: &Mat, minimum_score: f32) -> Result<Option<(Bbox, f32)>> {
        let Some(old) = self.bbox else {
            return Ok(None);
        };
        let scale_z = 127.0 / context_size(old.width, old.height);
        let crop = crop_square(
            image,
            old.x + old.width / 2.0,
            old.y + old.height / 2.0,
            255.0 / scale_z,
            255,
        )?;
        let search = self.feature(&crop)?;
        self.head.set_input(
            self.template.as_ref().unwrap(),
            "input1",
            1.0,
            Scalar::default(),
        )?;
        self.head
            .set_input(&search, "input2", 1.0, Scalar::default())?;
        let names = Vector::<String>::from_iter(["output1", "output2"]);
        let mut outputs = Vector::<Mat>::new();
        self.head.forward(&mut outputs, &names)?;
        ensure!(
            outputs.len() == 2,
            "NanoTrack head must have output1 and output2"
        );
        let cls = outputs.get(0)?;
        let reg = outputs.get(1)?;
        ensure!(
            cls.mat_size().as_ref() == [1, 2, 16, 16] && reg.mat_size().as_ref() == [1, 4, 16, 16],
            "expected NanoTrack v2 output shapes [1,2,16,16] and [1,4,16,16]"
        );
        let candidate = decode(
            cls.data_typed::<f32>()?,
            reg.data_typed::<f32>()?,
            old,
            scale_z,
        );
        let Some((bbox, score)) = candidate else {
            return Ok(None);
        };
        if score < minimum_score {
            return Ok(None);
        }
        let Some(clipped) = bbox.clip(image.cols(), image.rows()) else {
            return Ok(None);
        };
        if clipped.width < 5.0 || clipped.height < 5.0 {
            return Ok(None);
        }
        self.bbox = Some(clipped);
        Ok(Some((clipped, score)))
    }
}

fn context_size(w: f32, h: f32) -> f32 {
    let pad = (w + h) * 0.5;
    ((w + pad) * (h + pad)).sqrt().max(2.0)
}

fn crop_square(image: &Mat, cx: f32, cy: f32, side: f32, output: i32) -> Result<Mat> {
    ensure!(side.is_finite() && side > 0.0, "invalid search crop");
    // Direct affine sampling avoids allocating huge border images for large targets.
    let ratio = side as f64 / output as f64;
    let transform = Mat::from_slice_2d(&[
        [ratio, 0.0, cx as f64 - side as f64 / 2.0],
        [0.0, ratio, cy as f64 - side as f64 / 2.0],
    ])?;
    let average = core::mean(image, &core::no_array())?;
    let mut crop = Mat::default();
    imgproc::warp_affine(
        image,
        &mut crop,
        &transform,
        Size::new(output, output),
        imgproc::INTER_LINEAR | imgproc::WARP_INVERSE_MAP,
        core::BORDER_CONSTANT,
        average,
    )?;
    Ok(crop)
}

fn change(v: f32) -> f32 {
    v.max(1.0 / v)
}

pub fn decode(cls: &[f32], reg: &[f32], old: Bbox, scale: f32) -> Option<(Bbox, f32)> {
    if cls.len() != 512 || reg.len() != 1024 || !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let mut best: Option<(f32, Bbox, f32)> = None;
    for i in 0..256 {
        let distances = [reg[i], reg[256 + i], reg[512 + i], reg[768 + i]];
        if !distances.iter().all(|v| v.is_finite() && *v >= 0.0)
            || !cls[i].is_finite()
            || !cls[256 + i].is_finite()
        {
            continue;
        }
        let w = distances[0] + distances[2];
        let h = distances[1] + distances[3];
        if w <= 1.0 || h <= 1.0 {
            continue;
        }
        let raw = 1.0 / (1.0 + (cls[i] - cls[256 + i]).clamp(-80.0, 80.0).exp());
        let sc = change(context_size(w, h) / context_size(old.width * scale, old.height * scale));
        let rc = change((old.width / old.height) / (w / h));
        let penalty = (-(sc * rc - 1.0) * 0.055).exp();
        let row = i / 16;
        let col = i % 16;
        let hann = |n: usize| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / 15.0).cos());
        let ranked = penalty * raw * (1.0 - 0.455) + (hann(row) * hann(col)).sqrt() * 0.455;
        let gx = (col as f32 - 8.0) * 16.0 + 127.0;
        let gy = (row as f32 - 8.0) * 16.0 + 127.0;
        let cx =
            old.x + old.width / 2.0 + (gx + (distances[2] - distances[0]) / 2.0 - 127.0) / scale;
        let cy =
            old.y + old.height / 2.0 + (gy + (distances[3] - distances[1]) / 2.0 - 127.0) / scale;
        let lr = penalty * raw * 0.37;
        let nw = (w / scale) * lr + old.width * (1.0 - lr);
        let nh = (h / scale) * lr + old.height * (1.0 - lr);
        let bbox = Bbox {
            x: cx - nw / 2.0,
            y: cy - nh / 2.0,
            width: nw,
            height: nh,
        };
        if best.as_ref().is_none_or(|(rank, _, _)| ranked > *rank) {
            best = Some((ranked, bbox, raw));
        }
    }
    best.map(|(_, bbox, score)| (bbox, score))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decoder_rejects_nan_and_uses_foreground_probability() {
        let old = Bbox {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 100.0,
        };
        assert!(decode(&[f32::NAN; 512], &[50.0; 1024], old, 1.0).is_none());
        let mut cls = vec![0.0; 512];
        cls[256..].fill(5.0);
        let (_, score) = decode(&cls, &[50.0; 1024], old, 1.0).unwrap();
        assert!(score > 0.99);
    }
}
