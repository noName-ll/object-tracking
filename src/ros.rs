use crate::{
    cli::{Args, Durability},
    contract::{Header, Observation},
    pixels,
    tracker::Engine,
    video::Video,
};
use anyhow::{Context, Result};
use futures::{stream::LocalBoxStream, FutureExt, StreamExt};
use opencv::{
    core::{Mat, Vector},
    imgcodecs,
    prelude::*,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[derive(Default)]
struct Inbox {
    epoch: u64,
    target: Option<Image>,
    frame: Option<(Image, Instant)>,
}

enum Image {
    Raw(r2r::sensor_msgs::msg::Image),
    Compressed(r2r::sensor_msgs::msg::CompressedImage),
}

impl Image {
    fn header(&self) -> Header {
        let header = match self {
            Self::Raw(v) => &v.header,
            Self::Compressed(v) => &v.header,
        };
        Header {
            frame_id: header.frame_id.clone(),
            stamp: crate::contract::Stamp {
                sec: header.stamp.sec,
                nanosec: header.stamp.nanosec,
            },
        }
    }
    fn decode(&self) -> Result<Mat> {
        match self {
            Self::Raw(v) => {
                let bytes = pixels::to_bgr(v.width, v.height, v.step, &v.encoding, &v.data)?;
                let flat = Mat::from_slice(&bytes)?;
                Ok(flat.reshape(3, v.height as i32)?.try_clone()?)
            }
            Self::Compressed(v) => {
                anyhow::ensure!(
                    v.data.len() <= 64 * 1024 * 1024,
                    "compressed image too large"
                );
                let image = imgcodecs::imdecode(
                    &Vector::<u8>::from_slice(&v.data),
                    imgcodecs::IMREAD_COLOR,
                )?;
                anyhow::ensure!(!image.empty(), "cannot decode compressed image");
                Ok(image)
            }
        }
    }
}

fn subscribe(
    node: &mut r2r::Node,
    topic: &str,
    compressed: bool,
    qos: r2r::QosProfile,
) -> Result<LocalBoxStream<'static, Image>> {
    if compressed {
        Ok(node
            .subscribe::<r2r::sensor_msgs::msg::CompressedImage>(topic, qos)?
            .map(Image::Compressed)
            .boxed_local())
    } else {
        Ok(node
            .subscribe::<r2r::sensor_msgs::msg::Image>(topic, qos)?
            .map(Image::Raw)
            .boxed_local())
    }
}

fn latest(stream: &mut Option<LocalBoxStream<'static, Image>>) -> Option<Image> {
    let mut result = None;
    if let Some(stream) = stream {
        // Bound draining so a busy camera cannot starve reference updates or shutdown.
        for _ in 0..32 {
            match stream.next().now_or_never() {
                Some(Some(image)) => result = Some(image),
                _ => break,
            }
        }
    }
    result
}

fn emit(
    observation: &Observation,
    publisher: &r2r::PublisherUntyped,
    file: &mut Option<BufWriter<File>>,
) -> Result<()> {
    publisher.publish(observation.ros_value())?;
    let line = serde_json::to_string(observation)?;
    println!("{line}");
    if let Some(file) = file {
        writeln!(file, "{line}")?;
        file.flush()?;
    }
    Ok(())
}

fn invalidate(last: &mut Observation, status: &str) {
    last.status = status.into();
    last.bbox = None;
    last.model_score = 0.0;
    last.search_score = 0.0;
    last.inliers = 0;
    last.inlier_ratio = 0.0;
    last.processing_ms = 0.0;
    last.blur_variance = 0.0;
}

pub fn run(
    args: &Args,
    engine: &mut Engine,
    running: Arc<AtomicBool>,
    file: &mut Option<BufWriter<File>>,
) -> Result<()> {
    let context = r2r::Context::create()?;
    let mut node = r2r::Node::create(context.clone(), "mobile_object_tracker", "")?;
    let mut camera_qos = r2r::QosProfile::sensor_data();
    camera_qos.depth = 1;
    let mut target_qos = r2r::QosProfile::default();
    target_qos.depth = 1;
    target_qos.durability = match args.target_durability {
        Durability::Volatile => r2r::qos::DurabilityPolicy::Volatile,
        Durability::TransientLocal => r2r::qos::DurabilityPolicy::TransientLocal,
    };
    // Keep DDS reception independent of synchronous detector and tracker inference.
    let mailbox = Arc::new(std::sync::Mutex::new(Inbox::default()));
    let inbox = Arc::clone(&mailbox);
    let image_topic = args.image_topic.clone();
    let target_topic = args.target_topic.clone();
    let image_compressed = args.image_compressed;
    let target_compressed = args.target_compressed;
    let receiver_context = context.clone();
    let stop = running.clone();
    let (ready, startup) = std::sync::mpsc::sync_channel(1);
    let receiver = std::thread::spawn(move || -> Result<()> {
        let mut reader = r2r::Node::create(receiver_context.clone(), "tracker_input", "")?;
        let mut target = target_topic
            .map(|topic| subscribe(&mut reader, &topic, target_compressed, target_qos))
            .transpose()?;
        let mut camera = image_topic
            .map(|topic| subscribe(&mut reader, &topic, image_compressed, camera_qos))
            .transpose()?;
        ready.send(())?;
        while stop.load(Ordering::Relaxed) && receiver_context.is_valid() {
            reader.spin_once(Duration::from_millis(2));
            let replacement = latest(&mut target);
            let frame = latest(&mut camera);
            let mut slot = inbox.lock().unwrap();
            if let Some(image) = replacement {
                slot.epoch += 1;
                slot.target = Some(image);
                slot.frame = None;
            }
            if let Some(image) = frame {
                slot.frame = Some((image, Instant::now()));
            }
        }
        Ok(())
    });
    if startup.recv().is_err() {
        return receiver.join().expect("input thread panicked");
    }
    let publisher=node.create_publisher_untyped(&args.bbox_topic,"mobile_tracker_interfaces/msg/TrackingResult",r2r::QosProfile{depth:1,..r2r::QosProfile::default()})
        .context("custom message unavailable: build ros2/mobile_tracker_interfaces, source its install/setup.bash, then rebuild r2r with IDL_PACKAGE_FILTER including mobile_tracker_interfaces")?;
    let mut video = args
        .video
        .as_ref()
        .map(|source| Video::open(source, &args.frame_id))
        .transpose()?;
    let mut last = Observation {
        header: Header::default(),
        sequence: 0,
        source_position_ms: None,
        target_generation: engine.target_generation(),
        status: "waiting_target".into(),
        bbox: None,
        image_width: 0,
        image_height: 0,
        model_score: 0.0,
        search_score: 0.0,
        inliers: 0,
        inlier_ratio: 0.0,
        blur_variance: 0.0,
        processing_ms: 0.0,
    };
    let mut last_received = Instant::now();
    let mut last_timeout = Instant::now();
    let mut sequence = 0;
    while running.load(Ordering::Relaxed) && context.is_valid() {
        node.spin_once(Duration::from_millis(2));
        let (epoch, replacement, frame) = {
            let mut slot = mailbox.lock().unwrap();
            (slot.epoch, slot.target.take(), slot.frame.take())
        };
        if let Some(image) = replacement {
            match image
                .decode()
                .and_then(|m| engine.set_target(&m, args.roi()))
            {
                Ok(generation) => {
                    eprintln!("reference accepted, generation={generation}");
                    invalidate(&mut last, "target_changed");
                    last.target_generation = generation;
                    emit(&last, &publisher, file)?;
                }
                Err(error) => eprintln!("reference rejected; keeping previous target: {error:#}"),
            }
        }
        let mut incoming = None;
        if let Some((image, received)) = frame {
            last_received = received;
            let header = image.header();
            match image.decode() {
                Ok(frame) => incoming = Some((frame, header, None)),
                Err(error) => {
                    eprintln!("frame rejected: {error:#}");
                    engine.reset_motion();
                    invalidate(&mut last, "invalid_frame");
                    last.header = header;
                    last.sequence = sequence;
                    emit(&last, &publisher, file)?;
                    sequence += 1;
                }
            }
        }
        if let Some(video) = &mut video {
            if video.ready() {
                incoming = match video.read() {
                    Ok(frame) => frame,
                    Err(error) => {
                        eprintln!("video read failed: {error:#}");
                        invalidate(&mut last, "source_ended");
                        emit(&last, &publisher, file)?;
                        return Err(error);
                    }
                };
                if incoming.is_none() {
                    invalidate(&mut last, "source_ended");
                    emit(&last, &publisher, file)?;
                    break;
                }
                last_received = Instant::now();
            }
        }
        if let Some((image, header, position)) = incoming {
            last = match engine.process(&image, header.clone(), sequence, position) {
                Ok(result) => result,
                Err(error) => {
                    eprintln!("frame processing failed: {error:#}");
                    engine.reset_motion();
                    invalidate(&mut last, "invalid_frame");
                    last.header = header;
                    last.sequence = sequence;
                    last.clone()
                }
            };
            // A prompt may arrive while the old target is in inference.
            if mailbox.lock().unwrap().epoch == epoch {
                emit(&last, &publisher, file)?;
            }
            sequence += 1;
        } else if args.image_topic.is_some()
            && last_received.elapsed() > Duration::from_millis(args.source_timeout_ms)
            && last_timeout.elapsed() > Duration::from_millis(args.source_timeout_ms)
        {
            engine.reset_motion();
            invalidate(&mut last, "source_timeout");
            emit(&last, &publisher, file)?;
            last_timeout = Instant::now();
        }
    }
    Ok(())
}
