use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stamp {
    pub sec: i32,
    pub nanosec: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Header {
    pub stamp: Stamp,
    pub frame_id: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bbox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Bbox {
    pub fn clip(self, width: i32, height: i32) -> Option<Self> {
        if ![self.x, self.y, self.width, self.height]
            .iter()
            .all(|v| v.is_finite())
            || self.width <= 0.0
            || self.height <= 0.0
            || width <= 0
            || height <= 0
        {
            return None;
        }
        let x = self.x.max(0.0);
        let y = self.y.max(0.0);
        let right = (self.x + self.width).min(width as f32);
        let bottom = (self.y + self.height).min(height as f32);
        (right > x && bottom > y).then_some(Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        })
    }

    pub fn iou(self, other: Self) -> f32 {
        let w = ((self.x + self.width).min(other.x + other.width) - self.x.max(other.x)).max(0.0);
        let h = ((self.y + self.height).min(other.y + other.height) - self.y.max(other.y)).max(0.0);
        let intersection = w * h;
        intersection
            / (self.width * self.height + other.width * other.height - intersection).max(1e-6)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub header: Header,
    pub sequence: u64,
    pub source_position_ms: Option<f64>,
    pub target_generation: u64,
    pub status: String,
    pub bbox: Option<Bbox>,
    pub image_width: u32,
    pub image_height: u32,
    pub model_score: f32,
    pub search_score: f32,
    pub inliers: u32,
    pub inlier_ratio: f32,
    pub blur_variance: f64,
    pub processing_ms: f64,
}

impl Observation {
    /// Exactly the custom TrackingResult.msg schema, including explicit validity.
    pub fn ros_value(&self) -> serde_json::Value {
        let b = self.bbox;
        serde_json::json!({
            "header": self.header, "sequence": self.sequence,
            "target_generation": self.target_generation,
            "status": self.status, "found": b.is_some(),
            "x": b.map_or(0.0, |v|v.x), "y": b.map_or(0.0, |v|v.y),
            "width": b.map_or(0.0, |v|v.width), "height": b.map_or(0.0, |v|v.height),
            "image_width": self.image_width, "image_height": self.image_height,
            "model_score": self.model_score, "search_score": self.search_score, "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio, "blur_variance": self.blur_variance,
            "processing_ms": self.processing_ms,
            "has_source_position": self.source_position_ms.is_some(),
            "source_position_ms": self.source_position_ms.unwrap_or(0.0)
        })
    }
}

/// Commit only successfully prepared targets; callers prepare expensive features first.
#[derive(Default)]
pub struct TargetSlot<T> {
    pub value: Option<T>,
    pub generation: u64,
}
impl<T> TargetSlot<T> {
    pub fn replace(&mut self, candidate: Result<T, impl std::fmt::Display>) -> Result<u64, String> {
        let value = candidate.map_err(|e| e.to_string())?;
        self.value = Some(value);
        self.generation += 1;
        Ok(self.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn target_is_retained_and_invalid_replacement_is_atomic() {
        let mut slot = TargetSlot::<String>::default();
        slot.replace(Ok::<_, String>("first".into())).unwrap();
        assert!(slot.replace(Err::<String, _>("invalid image")).is_err());
        assert_eq!(slot.generation, 1);
        assert_eq!(slot.value.as_deref(), Some("first"));
        slot.replace(Ok::<_, String>("second".into())).unwrap();
        assert_eq!(slot.generation, 2);
        assert_eq!(slot.value.as_deref(), Some("second"));
    }
    #[test]
    fn clips_and_rejects_invalid_boxes() {
        let b = Bbox {
            x: -10.0,
            y: 5.0,
            width: 30.0,
            height: 100.0,
        };
        assert_eq!(
            b.clip(100, 50),
            Some(Bbox {
                x: 0.0,
                y: 5.0,
                width: 20.0,
                height: 45.0
            })
        );
        assert!(Bbox { x: f32::NAN, ..b }.clip(100, 50).is_none());
        assert!(Bbox { x: 101.0, ..b }.clip(100, 50).is_none());
    }

    #[test]
    fn invalid_observation_has_no_box_and_matches_ros_schema() {
        let observation = Observation {
            header: Header::default(),
            sequence: 3,
            source_position_ms: None,
            target_generation: 2,
            status: "lost".into(),
            bbox: None,
            image_width: 640,
            image_height: 480,
            model_score: 0.0,
            search_score: 0.0,
            inliers: 0,
            inlier_ratio: 0.0,
            blur_variance: 100.0,
            processing_ms: 10.0,
        };
        let value = observation.ros_value();
        assert_eq!(value["found"], false);
        for field in ["x", "y", "width", "height"] {
            assert_eq!(value[field], 0.0);
        }
        let message = include_str!("../ros2/mobile_tracker_interfaces/msg/TrackingResult.msg");
        let declared = message
            .lines()
            .filter_map(|line| {
                let text = line.split('#').next().unwrap().trim();
                text.split_whitespace().nth(1)
            })
            .collect::<std::collections::BTreeSet<_>>();
        let emitted = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(declared, emitted);
        assert!(serde_json::to_value(&observation).unwrap()["bbox"].is_null());
    }
}
