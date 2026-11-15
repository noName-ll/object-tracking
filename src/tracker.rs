use crate::{
    contract::{Bbox, Header, Observation, TargetSlot},
    nano::NanoTrack,
};
use anyhow::{ensure, Result};
use opencv::{
    calib3d,
    core::{self, DMatch, KeyPoint, Mat, Point2f, Rect, Scalar, Size, Vector},
    features2d, imgcodecs, imgproc,
    prelude::*,
};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Config {
    pub max_width: i32,
    pub min_inliers: usize,
    pub min_ratio: f32,
    pub model_score: f32,
    pub blur_threshold: f64,
    pub verify_every: u64,
    pub verification_ttl: Duration,
    pub gap_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_width: 960,
            min_inliers: 12,
            min_ratio: 0.4,
            model_score: 0.75,
            blur_threshold: 25.0,
            verify_every: 15,
            verification_ttl: Duration::from_secs(5),
            gap_timeout: Duration::from_millis(1200),
        }
    }
}

pub struct Reference {
    keypoints: Vector<KeyPoint>,
    descriptors: Mat,
    width: i32,
    height: i32,
}

pub struct Engine {
    config: Config,
    reference: TargetSlot<Reference>,
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

fn gray(image: &Mat) -> Result<Mat> {
    let mut output = Mat::default();
    imgproc::cvt_color(image, &mut output, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(output)
}

fn features(image: &Mat) -> Result<(Vector<KeyPoint>, Mat)> {
    let mut sift = features2d::SIFT::create_def()?;
    let mut points = Vector::new();
    let mut descriptors = Mat::default();
    sift.detect_and_compute(
        &gray(image)?,
        &core::no_array(),
        &mut points,
        &mut descriptors,
        false,
    )?;
    Ok((points, descriptors))
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
        Ok(Self {
            config,
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
        let selected = resize(&selected, self.config.max_width.min(640))?;
        let (keypoints, descriptors) = features(&selected)?;
        ensure!(keypoints.len()>=self.config.min_inliers && !descriptors.empty(),
            "reference has too few SIFT features for global acquisition; choose a textured view or a tighter ROI");
        let candidate = Reference {
            keypoints,
            descriptors,
            width: selected.cols(),
            height: selected.rows(),
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
            let geometry = if verify { self.locate(&work)? } else { None };
            let mut accepted = None;
            if let Some((bbox, inliers, ratio)) = geometry {
                result.inliers = inliers as u32;
                result.inlier_ratio = ratio;
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

    fn locate(&self, frame: &Mat) -> Result<Option<(Bbox, usize, f32)>> {
        let Some(reference) = self.reference.value.as_ref() else {
            return Ok(None);
        };
        let (points, desc) = features(frame)?;
        if desc.rows() < 2 {
            return Ok(None);
        }
        let matcher = features2d::BFMatcher::create(core::NORM_L2, false)?;
        let mut forward = Vector::<Vector<DMatch>>::new();
        let mut reverse = Vector::<DMatch>::new();
        matcher.knn_train_match_def(&reference.descriptors, &desc, &mut forward, 2)?;
        matcher.train_match_def(&desc, &reference.descriptors, &mut reverse)?;
        let mut src = Vector::<Point2f>::new();
        let mut dst = Vector::<Point2f>::new();
        for pair in forward {
            if pair.len() < 2 {
                continue;
            }
            let m = pair.get(0)?;
            let n = pair.get(1)?;
            if m.distance < 0.75 * n.distance
                && reverse.get(m.train_idx as usize)?.train_idx == m.query_idx
            {
                src.push(reference.keypoints.get(m.query_idx as usize)?.pt());
                dst.push(points.get(m.train_idx as usize)?.pt());
            }
        }
        if src.len() < self.config.min_inliers {
            return Ok(None);
        }
        let mut mask = Mat::default();
        let h = calib3d::find_homography_ext(
            &src,
            &dst,
            calib3d::USAC_MAGSAC,
            4.0,
            &mut mask,
            5000,
            0.999,
        )?;
        if h.empty() {
            return Ok(None);
        }
        let flags = mask.data_typed::<u8>()?;
        let count = flags.iter().filter(|v| **v != 0).count();
        let ratio = count as f32 / src.len() as f32;
        if count < self.config.min_inliers || ratio < self.config.min_ratio {
            return Ok(None);
        }
        let inside = Vector::<Point2f>::from_iter(
            src.iter()
                .zip(flags)
                .filter_map(|(p, v)| (*v != 0).then_some(p)),
        );
        let mut hull = Vector::<Point2f>::new();
        imgproc::convex_hull(&inside, &mut hull, false, true)?;
        let coverage =
            imgproc::contour_area(&hull, false)? / (reference.width * reference.height) as f64;
        if coverage < 0.035 {
            return Ok(None);
        }
        let corners = Vector::<Point2f>::from_iter([
            Point2f::new(0.0, 0.0),
            Point2f::new((reference.width - 1) as f32, 0.0),
            Point2f::new((reference.width - 1) as f32, (reference.height - 1) as f32),
            Point2f::new(0.0, (reference.height - 1) as f32),
        ]);
        let mut quad = Vector::<Point2f>::new();
        core::perspective_transform(&corners, &mut quad, &h)?;
        if !quad.iter().all(|p| p.x.is_finite() && p.y.is_finite())
            || !imgproc::is_contour_convex(&quad)?
        {
            return Ok(None);
        }
        let area = imgproc::contour_area(&quad, false)?.abs();
        if area < 100.0 || area > (frame.cols() * frame.rows()) as f64 * 1.5 {
            return Ok(None);
        }
        let screen = Vector::<Point2f>::from_iter([
            Point2f::new(0.0, 0.0),
            Point2f::new(frame.cols() as f32, 0.0),
            Point2f::new(frame.cols() as f32, frame.rows() as f32),
            Point2f::new(0.0, frame.rows() as f32),
        ]);
        let mut intersection = Vector::<Point2f>::new();
        let visible = imgproc::intersect_convex_convex(&quad, &screen, &mut intersection, true)?;
        if visible as f64 / area < 0.35 {
            return Ok(None);
        }
        let min_x = quad.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let min_y = quad.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let max_x = quad.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let max_y = quad.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        let bbox = Bbox {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        };
        Ok(bbox
            .clip(frame.cols(), frame.rows())
            .map(|b| (b, count, ratio)))
    }
}
