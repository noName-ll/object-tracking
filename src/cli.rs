use crate::tracker::Config;
use anyhow::{ensure, Result};
use clap::{ArgGroup, Parser, ValueEnum};
use opencv::core::Rect;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Durability {
    Volatile,
    TransientLocal,
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Rust NanoTrack + geometric reacquisition, with optional ROS 2 transport"
)]
#[command(group(ArgGroup::new("source").required(true).args(["video","image_topic"])))]
#[command(group(ArgGroup::new("reference").required(true).multiple(true).args(["target","target_topic"])))]
pub struct Args {
    #[arg(long, help = "Video file, RTSP/HTTP URL, or camera:N")]
    pub video: Option<String>,
    #[arg(long, help = "ROS sensor_msgs/msg/Image source")]
    pub image_topic: Option<String>,
    #[arg(
        long,
        requires = "image_topic",
        help = "Source topic uses sensor_msgs/msg/CompressedImage"
    )]
    pub image_compressed: bool,
    #[arg(long, help = "Initial reference image")]
    pub target: Option<String>,
    #[arg(
        long,
        help = "New reference images; each accepted message replaces the retained target"
    )]
    pub target_topic: Option<String>,
    #[arg(
        long,
        requires = "target_topic",
        help = "Reference topic uses sensor_msgs/msg/CompressedImage"
    )]
    pub target_compressed: bool,
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 4,
        allow_hyphen_values = true,
        help = "Reference-image crop X,Y,W,H; applies to each incoming reference"
    )]
    pub target_roi: Option<Vec<i32>>,
    #[arg(long, help = "Enable ROS publishers even with file-only inputs")]
    pub ros: bool,
    #[arg(long, default_value = "/tracker/bbox")]
    pub bbox_topic: String,
    #[arg(long, default_value = "volatile", value_enum)]
    pub target_durability: Durability,
    #[arg(long, default_value = "models/nanotrack_backbone_sim.onnx")]
    pub backbone: String,
    #[arg(long, default_value = "models/nanotrack_head_sim.onnx")]
    pub head: String,
    #[arg(long, default_value_t = 960)]
    pub max_width: i32,
    #[arg(long, default_value_t = 12)]
    pub min_inliers: usize,
    #[arg(long, default_value_t = 0.4)]
    pub min_ratio: f32,
    #[arg(long, default_value_t = 0.75)]
    pub model_score: f32,
    #[arg(long, default_value_t = 25.0)]
    pub blur_threshold: f64,
    #[arg(long, default_value_t = 15)]
    pub verify_every: u64,
    #[arg(long, default_value_t = 5000)]
    pub verification_ttl_ms: u64,
    #[arg(long, default_value_t = 1200)]
    pub gap_timeout_ms: u64,
    #[arg(
        long,
        default_value_t = 1000,
        help = "ROS source silence timeout; publishes source_timeout without a box"
    )]
    pub source_timeout_ms: u64,
    #[arg(
        long,
        default_value = "camera",
        help = "Frame ID for video sources only"
    )]
    pub frame_id: String,
    #[arg(
        long,
        help = "Also write JSONL observations; stdout always emits JSONL"
    )]
    pub jsonl: Option<PathBuf>,
}

impl Args {
    pub fn uses_ros(&self) -> bool {
        self.ros || self.image_topic.is_some() || self.target_topic.is_some()
    }
    pub fn roi(&self) -> Option<Rect> {
        self.target_roi
            .as_ref()
            .map(|v| Rect::new(v[0], v[1], v[2], v[3]))
    }
    pub fn config(&self) -> Result<Config> {
        ensure!(
            self.max_width >= 64 && self.min_inliers >= 8 && self.verify_every > 0,
            "invalid dimensions or feature thresholds"
        );
        ensure!(
            self.min_ratio.is_finite() && self.min_ratio > 0.0 && self.min_ratio <= 1.0,
            "min-ratio must be in (0,1]"
        );
        ensure!(
            self.model_score.is_finite() && (0.0..=1.0).contains(&self.model_score),
            "model-score must be in [0,1]"
        );
        ensure!(
            self.blur_threshold.is_finite() && self.blur_threshold >= 0.0,
            "blur-threshold must be nonnegative"
        );
        ensure!(
            self.verification_ttl_ms > 0 && self.gap_timeout_ms > 0 && self.source_timeout_ms > 0,
            "timeouts must be positive"
        );
        Ok(Config {
            max_width: self.max_width,
            min_inliers: self.min_inliers,
            min_ratio: self.min_ratio,
            model_score: self.model_score,
            blur_threshold: self.blur_threshold,
            verify_every: self.verify_every,
            verification_ttl: std::time::Duration::from_millis(self.verification_ttl_ms),
            gap_timeout: std::time::Duration::from_millis(self.gap_timeout_ms),
        })
    }
}
