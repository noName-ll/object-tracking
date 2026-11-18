use crate::{contract::Bbox, tracker::gray};
use anyhow::{ensure, Result};
use opencv::{
    calib3d,
    core::{self, DMatch, KeyPoint, Mat, Point2f, Vector},
    features2d, imgproc,
    prelude::*,
};
pub struct Reference {
    keypoints: Vector<KeyPoint>,
    descriptors: Mat,
    width: i32,
    height: i32,
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

impl Reference {
    pub fn encode(image: &Mat) -> Result<Self> {
        let (keypoints, descriptors) = features(image)?;
        ensure!(
            keypoints.len() >= 12,
            "reference needs at least 12 SIFT features; choose a clearer view"
        );
        Ok(Self {
            keypoints,
            descriptors,
            width: image.cols(),
            height: image.rows(),
        })
    }
    pub fn locate(&self, frame: &Mat) -> Result<Option<(Bbox, usize, f32)>> {
        let reference = self;
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
        if src.len() < 12 {
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
        if count < 12 || ratio < 0.4 {
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
