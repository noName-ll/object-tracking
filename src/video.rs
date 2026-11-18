use crate::contract::{Header, Stamp};
use anyhow::{ensure, Context, Result};
use opencv::{core::Mat, prelude::*, videoio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Capture {
    capture: videoio::VideoCapture,
    live: bool,
    fps: f64,
    next_due: Instant,
    frame_id: String,
}

impl Capture {
    pub fn open(source: &str, frame_id: &str) -> Result<Self> {
        let camera = source
            .strip_prefix("camera:")
            .map(str::parse::<i32>)
            .transpose()?;
        let live = camera.is_some() || source.contains("://");
        let mut capture = if let Some(index) = camera {
            videoio::VideoCapture::new(index, videoio::CAP_ANY)?
        } else {
            videoio::VideoCapture::from_file(source, videoio::CAP_ANY)?
        };
        ensure!(capture.is_opened()?, "cannot open video source");
        let _ = capture.set(videoio::CAP_PROP_BUFFERSIZE, 1.0);
        let fps = capture.get(videoio::CAP_PROP_FPS)?;
        let fps = if fps.is_finite() && fps > 0.0 {
            fps.min(240.0)
        } else {
            30.0
        };
        Ok(Self {
            capture,
            live,
            fps,
            next_due: Instant::now(),
            frame_id: frame_id.into(),
        })
    }
    /// File frames are paced, preserving all frames; processing may run slower than media time.
    pub fn ready(&self) -> bool {
        self.live || Instant::now() >= self.next_due
    }
    pub fn read(&mut self) -> Result<Option<(Mat, Header, Option<f64>)>> {
        let mut frame = Mat::default();
        if !self.capture.read(&mut frame)? || frame.empty() {
            return Ok(None);
        }
        self.next_due = Instant::now() + Duration::from_secs_f64(1.0 / self.fps);
        let position = self.capture.get(videoio::CAP_PROP_POS_MSEC)?;
        let time = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let header = Header {
            stamp: Stamp {
                sec: time.as_secs().min(i32::MAX as u64) as i32,
                nanosec: time.subsec_nanos(),
            },
            frame_id: self.frame_id.clone(),
        };
        Ok(Some((
            frame,
            header,
            (!self.live && position.is_finite()).then_some(position),
        )))
    }
}

type Frame = (Mat, Header, Option<f64>);
type ReadResult = Result<Option<Frame>>;
#[derive(Default)]
struct LiveInput {
    frame: Option<Frame>,
    end: Option<ReadResult>,
}
enum Input {
    File(Capture),
    Live(std::sync::Arc<std::sync::Mutex<LiveInput>>),
}
pub struct Video(Input);
impl Video {
    pub fn open(source: &str, frame_id: &str) -> Result<Self> {
        let mut capture = Capture::open(source, frame_id)?;
        if !capture.live {
            return Ok(Self(Input::File(capture)));
        }
        let slot = std::sync::Arc::new(std::sync::Mutex::new(LiveInput::default()));
        let weak = std::sync::Arc::downgrade(&slot);
        std::thread::Builder::new()
            .name("camera-reader".into())
            .spawn(move || loop {
                let result = capture.read();
                let Some(shared) = weak.upgrade() else {
                    break;
                };
                let mut inbox = shared.lock().unwrap();
                match result {
                    Ok(Some(frame)) => inbox.frame = Some(frame),
                    terminal => {
                        inbox.end = Some(terminal);
                        break;
                    }
                }
            })?;
        Ok(Self(Input::Live(slot)))
    }
    pub fn ready(&self) -> bool {
        match &self.0 {
            Input::File(c) => c.ready(),
            Input::Live(slot) => {
                let s = slot.lock().unwrap();
                s.frame.is_some() || s.end.is_some()
            }
        }
    }
    pub fn read(&mut self) -> ReadResult {
        match &mut self.0 {
            Input::File(c) => c.read(),
            Input::Live(slot) => {
                let mut s = slot.lock().unwrap();
                if let Some(frame) = s.frame.take() {
                    return Ok(Some(frame));
                }
                s.end
                    .take()
                    .context("call ready before reading a live source")?
            }
        }
    }
}
