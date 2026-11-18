use crate::{
    contract::{Bbox, Header, Observation, TargetSlot},
    nano::NanoTrack,
    yoloe::{SearchResult, YoloE},
};
use anyhow::{ensure, Result};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size},
    imgcodecs, imgproc,
    prelude::*,
};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Config {
    pub backend: Backend,
    pub max_width: i32,
    pub yoloe_encoder: String,
    pub yoloe_detector: String,
    pub yoloe_size: usize,
    pub search_score: f32,
    pub search_iou: f32,
    pub search_margin: f32,
    pub model_score: f32,
    pub blur_threshold: f64,
    pub verify_every: u64,
    pub verification_ttl: Duration,
    pub gap_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Features,
            max_width: 960,
            yoloe_encoder: "models/yoloe/encoder.onnx".into(),
            yoloe_detector: "models/yoloe/detector.onnx".into(),
            yoloe_size: 640,
            search_score: 0.25,
            search_iou: 0.5,
            search_margin: 0.05,
            model_score: 0.75,
            blur_threshold: 25.0,
            verify_every: 15,
            verification_ttl: Duration::from_secs(5),
            gap_timeout: Duration::from_millis(1200),
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Backend {
    Features,
    Yoloe,
}
enum Prompt {
    Features(crate::features::Reference),
    Yoloe(Vec<f32>),
}
pub struct Engine {
    config: Config,
    reference: TargetSlot<Prompt>,
    search: Option<YoloE>,
    nano: NanoTrack,
    active: bool,
    frame: u64,
    last_frame: Option<Instant>,
    verified_at: Option<Instant>,
    work_size: Option<Size>,
}

pub fn read_image(path: &str) -> Result<Mat> {
    let image = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
    ensure!(!image.empty(), "cannot read image {path}");
    Ok(image)
}

fn resize(image: &Mat, maximum: i32) -> Result<Mat> {
    if image.cols() <= maximum {
        return Ok(image.try_clone()?);
    }
    let mut output = Mat::default();
    let height = (image.rows() as f64 * maximum as f64 / image.cols() as f64)
        .round()
        .max(1.0) as i32;
    imgproc::resize(
        image,
        &mut output,
        Size::new(maximum, height),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    Ok(output)
}

pub(crate) fn gray(image: &Mat) -> Result<Mat> {
    let mut output = Mat::default();
    imgproc::cvt_color(image, &mut output, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(output)
}

pub fn blur_variance(image: &Mat) -> Result<f64> {
    let mut lap = Mat::default();
    imgproc::laplacian(
        &gray(image)?,
        &mut lap,
        core::CV_64F,
        1,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    let mut mean = Scalar::default();
    let mut stddev = Scalar::default();
    core::mean_std_dev(&lap, &mut mean, &mut stddev, &core::no_array())?;
    Ok(stddev[0] * stddev[0])
}

impl Engine {
    pub fn new(config: Config, backbone: &str, head: &str) -> Result<Self> {
        let search = match config.backend {
            Backend::Features => None,
            Backend::Yoloe => Some(YoloE::load(
                &config.yoloe_encoder,
                &config.yoloe_detector,
                config.yoloe_size,
                config.search_score,
                config.search_iou,
                config.search_margin,
            )?),
        };
        Ok(Self {
            config,
            search,
            reference: TargetSlot {
                value: None,
                generation: 0,
            },
            nano: NanoTrack::load(backbone, head)?,
            active: false,
            frame: 0,
            last_frame: None,
            verified_at: None,
            work_size: None,
        })
    }

    /// One call per new screenshot. No new message is required on later frames.
    pub fn set_target(&mut self, image: &Mat, roi: Option<Rect>) -> Result<u64> {
        ensure!(
            !image.empty() && image.typ() == core::CV_8UC3,
            "reference must be nonempty BGR8"
        );
        let selected = if let Some(r) = roi {
            ensure!(
                r.x >= 0
                    && r.y >= 0
                    && r.width > 0
                    && r.height > 0
                    && i64::from(r.x) + i64::from(r.width) <= i64::from(image.cols())
                    && i64::from(r.y) + i64::from(r.height) <= i64::from(image.rows()),
                "ROI outside screenshot"
            );
            Mat::roi(image, r)?.try_clone()?
        } else {
            image.try_clone()?
        };
        ensure!(
            selected.cols() >= 16 && selected.rows() >= 16,
            "reference too small"
        );
        // Compute the candidate prompt before changing the retained target or NanoTrack state.
        let candidate = match self.search.as_mut() {
            Some(yolo) => Prompt::Yoloe(yolo.encode(&selected)?),
            None => Prompt::Features(crate::features::Reference::encode(&resize(
                &selected,
                self.config.max_width.min(640),
            )?)?),
        };
        let generation = self
            .reference
            .replace(Ok::<_, String>(candidate))
            .map_err(anyhow::Error::msg)?;
        self.reset_motion();
        Ok(generation)
    }

    pub fn reset_motion(&mut self) {
        self.nano.clear();
        self.active = false;
        self.verified_at = None;
        self.last_frame = None;
    }

    pub fn target_generation(&self) -> u64 {
        self.reference.generation
    }

    pub fn process(
        &mut self,
        image: &Mat,
        header: Header,
        sequence: u64,
        source_position_ms: Option<f64>,
    ) -> Result<Observation> {
        ensure!(
            !image.empty() && image.typ() == core::CV_8UC3,
            "frame must be BGR8"
        );
        let started = Instant::now();
        let work = resize(image, self.config.max_width)?;
        let size = work.size()?;
        if self.work_size.is_some_and(|old| old != size) {
            self.reset_motion();
        }
        self.work_size = Some(size);
        let sx = image.cols() as f32 / work.cols() as f32;
        let sy = image.rows() as f32 / work.rows() as f32;
        let blur = blur_variance(&work)?;
        if self
            .last_frame
            .is_some_and(|t| t.elapsed() > self.config.gap_timeout)
        {
            self.reset_motion();
        }
        self.last_frame = Some(Instant::now());
        self.frame += 1;
        let mut result = Observation {
            header,
            sequence,
            source_position_ms,
            target_generation: self.reference.generation,
            status: "waiting_target".into(),
            bbox: None,
            image_width: image.cols() as u32,
            image_height: image.rows() as u32,
            model_score: 0.0,
            search_score: 0.0,
            inliers: 0,
            inlier_ratio: 0.0,
            blur_variance: blur,
            processing_ms: 0.0,
        };
        if self.reference.value.is_none() {
            return Ok(result);
        }
        if blur < self.config.blur_threshold {
            // No guessed box and no contaminated template update on a blurred frame.
            result.status = "blurred".into();
            self.reset_motion();
        } else {
            let learned = if self.active {
                self.nano.update(&work, self.config.model_score)?
            } else {
                None
            };
            if self.active && learned.is_none() {
                self.reset_motion();
            }
            let verify = learned.is_none()
                || self.frame.is_multiple_of(self.config.verify_every)
                || self
                    .verified_at
                    .is_none_or(|t| t.elapsed() > self.config.verification_ttl);
            let global = if verify {
                match self.reference.value.as_ref().unwrap() {
                    Prompt::Yoloe(prompt) => self
                        .search
                        .as_mut()
                        .expect("configured YOLOE")
                        .search(&work, prompt)?,
                    Prompt::Features(prompt) => match prompt.locate(&work)? {
                        Some((bbox, count, ratio)) => {
                            result.inliers = count as u32;
                            result.inlier_ratio = ratio;
                            SearchResult::Found(crate::yoloe::Candidate { bbox, score: 0.0 })
                        }
                        None => SearchResult::Missing,
                    },
                }
            } else {
                SearchResult::Missing
            };
            let mut accepted = None;
            if let SearchResult::Found(candidate) = global {
                let bbox = candidate.bbox;
                result.search_score = candidate.score;
                if learned.is_some_and(|(b, _)| b.iou(bbox) < 0.1) {
                    // Disagreement may mean identity switch: do not silently jump.
                    self.reset_motion();
                    result.status = "ambiguous".into();
                } else {
                    self.nano.init(&work, bbox)?;
                    self.active = true;
                    self.verified_at = Some(Instant::now());
                    accepted = Some(bbox);
                    result.status = "tracking_verified".into();
                    result.model_score = learned.map_or(0.0, |(_, score)| score);
                }
            } else if matches!(global, SearchResult::Ambiguous) {
                self.reset_motion();
                result.status = "ambiguous".into();
            } else if let Some((bbox, score)) = learned {
                if self
                    .verified_at
                    .is_some_and(|t| t.elapsed() <= self.config.verification_ttl)
                {
                    accepted = Some(bbox);
                    result.model_score = score;
                    result.status = "tracking_model".into();
                } else {
                    self.reset_motion();
                    result.status = "lost".into();
                }
            } else {
                result.status = "lost".into();
            }
            result.bbox = accepted.and_then(|b| {
                Bbox {
                    x: b.x * sx,
                    y: b.y * sy,
                    width: b.width * sx,
                    height: b.height * sy,
                }
                .clip(image.cols(), image.rows())
            });
        }
        result.processing_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(result)
    }
}
