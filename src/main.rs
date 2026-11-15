#[cfg(not(feature = "ros2"))]
use anyhow::bail;
use anyhow::{Context, Result};
use clap::Parser;
use mobile_object_tracker::{
    cli::Args,
    tracker::{read_image, Engine},
    video::Video,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

fn main() -> Result<()> {
    let args = Args::parse();
    let config = args.config()?;
    #[cfg(not(feature = "ros2"))]
    if args.uses_ros() {
        bail!("ROS support requires cargo build --release --features ros2");
    }
    let mut engine = Engine::new(config, &args.backbone, &args.head)
        .context("load NanoTrack v2 models; run scripts/download_models.sh first")?;
    if let Some(path) = &args.target {
        engine.set_target(&read_image(path)?, args.roi())?;
        eprintln!("initial reference accepted (generation 1)");
    }
    let running = Arc::new(AtomicBool::new(true));
    let flag = running.clone();
    ctrlc::set_handler(move || flag.store(false, Ordering::Relaxed))?;
    let mut file = args
        .jsonl
        .as_ref()
        .map(File::create)
        .transpose()?
        .map(BufWriter::new);
    #[cfg(feature = "ros2")]
    if args.uses_ros() {
        return mobile_object_tracker::ros::run(&args, &mut engine, running, &mut file);
    }
    let mut video = Video::open(args.video.as_deref().unwrap(), &args.frame_id)?;
    let mut sequence = 0;
    while running.load(Ordering::Relaxed) {
        if !video.ready() {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        let Some((image, header, position)) = video.read()? else {
            break;
        };
        let observation = engine.process(&image, header, sequence, position)?;
        emit(&observation, &mut file)?;
        sequence += 1;
    }
    Ok(())
}

fn emit(
    observation: &mobile_object_tracker::contract::Observation,
    file: &mut Option<BufWriter<File>>,
) -> Result<()> {
    let line = serde_json::to_string(observation)?;
    println!("{line}");
    if let Some(file) = file {
        writeln!(file, "{line}")?;
        file.flush()?;
    }
    Ok(())
}
