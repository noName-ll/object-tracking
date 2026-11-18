//! One-shot foreground extraction and multi-scale colour template search.
use crate::contract::Bbox;
use crate::yoloe::{Candidate, SearchResult};
use anyhow::{ensure, Result};
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};

struct Template {
    image: Mat,
    mask: Mat,
    samples: f64,
}

pub struct Reference {
    templates: Vec<Template>,
    features: Option<crate::features::Reference>,
}

impl Reference {
    pub fn encode(image: &Mat, template_only: bool) -> Result<Self> {
        ensure!(image.cols().min(image.rows()) >= 6, "foreground too small");
        let mut labels = Mat::default();
        imgproc::grab_cut(
            image,
            &mut labels,
            Rect::new(1, 1, image.cols() - 2, image.rows() - 2),
            &mut Mat::default(),
            &mut Mat::default(),
            4,
            imgproc::GC_INIT_WITH_RECT,
        )?;
        let mut mask = Mat::default();
        core::bitwise_and(&labels, &Scalar::all(1.), &mut mask, &core::no_array())?;
        let rect = imgproc::bounding_rect(&mask)?;
        let has_foreground =
            rect.width >= 6 && rect.height >= 6 && core::count_non_zero(&mask)? >= 40;
        let crop = if has_foreground {
            rect
        } else {
            Rect::new(0, 0, image.cols(), image.rows())
        };
        let patch = image.roi(crop)?;
        let features = if template_only {
            None
        } else {
            let mut enlarged = Mat::default();
            let scale = 180. / f64::from(crop.width.max(crop.height));
            imgproc::resize(
                &patch,
                &mut enlarged,
                Size::default(),
                scale,
                scale,
                imgproc::INTER_CUBIC,
            )?;
            crate::features::Reference::encode_if_supported(&enlarged)?
        };
        let mut templates = Vec::new();
        if !has_foreground {
            ensure!(
                features.is_some(),
                "cannot separate foreground in reference; use a tighter crop"
            );
            return Ok(Self {
                templates,
                features,
            });
        }
        let foreground = mask.roi(rect)?;
        // Search on a half-resolution frame at five scales per octave.
        for step in -5..=11 {
            let factor = 0.5 * 2f64.powf(step as f64 / 5.);
            let size = Size::new(
                (rect.width as f64 * factor).round() as i32,
                (rect.height as f64 * factor).round() as i32,
            );
            if size.width.min(size.height) < 5 {
                continue;
            }
            let mut image = Mat::default();
            let mut mask = Mat::default();
            imgproc::resize(&patch, &mut image, size, 0., 0., imgproc::INTER_LINEAR)?;
            imgproc::resize(&foreground, &mut mask, size, 0., 0., imgproc::INTER_NEAREST)?;
            let samples = f64::from(core::count_non_zero(&mask)?) * 3.;
            if samples >= 60. {
                templates.push(Template {
                    image,
                    mask,
                    samples,
                });
            }
        }
        ensure!(
            !templates.is_empty() || features.is_some(),
            "foreground too small"
        );
        Ok(Self {
            templates,
            features,
        })
    }

    pub fn locate(&self, frame: &Mat) -> Result<SearchResult> {
        let mut small = Mat::default();
        let size = Size::new((frame.cols() + 1) / 2, (frame.rows() + 1) / 2);
        imgproc::resize(frame, &mut small, size, 0., 0., imgproc::INTER_AREA)?;
        let mut candidates = Vec::new();
        for template in &self.templates {
            let w = template.image.cols();
            let h = template.image.rows();
            if w > small.cols() || h > small.rows() {
                continue;
            }
            let mut errors = Mat::default();
            imgproc::match_template(
                &small,
                &template.image,
                &mut errors,
                imgproc::TM_SQDIFF,
                &template.mask,
            )?;
            // Keep the two best spatially distinct matches at this scale.
            for _ in 0..2 {
                let mut cost = 0.;
                let mut p = Point::default();
                core::min_max_loc(
                    &errors,
                    Some(&mut cost),
                    None,
                    Some(&mut p),
                    None,
                    &core::no_array(),
                )?;
                let rms = (cost.max(0.) / template.samples).sqrt();
                if rms > 35. {
                    break;
                }
                candidates.push((
                    rms,
                    Bbox {
                        x: (p.x * 2) as f32,
                        y: (p.y * 2) as f32,
                        width: (w * 2) as f32,
                        height: (h * 2) as f32,
                    },
                ));
                let left = (p.x - w / 2).max(0);
                let top = (p.y - h / 2).max(0);
                let right = (p.x + w / 2 + 1).min(errors.cols());
                let bottom = (p.y + h / 2 + 1).min(errors.rows());
                imgproc::rectangle(
                    &mut errors,
                    Rect::new(left, top, right - left, bottom - top),
                    Scalar::all(f32::MAX as f64),
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let Some(&(error, bbox)) = candidates.first() else {
            if let Some(features) = &self.features {
                if let Some((bbox, _, _)) = features.locate(frame)? {
                    return Ok(SearchResult::Found(Candidate { bbox, score: 0. }));
                }
            }
            return Ok(SearchResult::Missing);
        };
        if candidates
            .iter()
            .any(|(other, b)| bbox.iou(*b) < 0.15 && *other < error + 4.)
        {
            return Ok(SearchResult::Ambiguous);
        }
        Ok(SearchResult::Found(Candidate {
            bbox,
            score: (1. - error / 255.) as f32,
        }))
    }
}

/// Sample the middle half of the box to reduce background contribution.
pub fn sample(frame: &Mat, bbox: Bbox) -> Result<Mat> {
    let x = ((bbox.x + bbox.width * 0.25) as i32).clamp(0, frame.cols() - 1);
    let y = ((bbox.y + bbox.height * 0.25) as i32).clamp(0, frame.rows() - 1);
    let w = ((bbox.width * 0.5) as i32).clamp(1, frame.cols() - x);
    let h = ((bbox.height * 0.5) as i32).clamp(1, frame.rows() - y);
    let mut output = Mat::default();
    imgproc::resize(
        &frame.roi(Rect::new(x, y, w, h))?,
        &mut output,
        Size::new(8, 8),
        0.,
        0.,
        imgproc::INTER_AREA,
    )?;
    Ok(output)
}
