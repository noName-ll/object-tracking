#![cfg(feature = "vision")]
use mobile_object_tracker::{
    contract::{Bbox, Header},
    nano::NanoTrack,
    tracker::{Config, Engine},
};
use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
};
use std::time::Duration;

fn reference() -> anyhow::Result<Mat> {
    let mut image = Mat::new_rows_cols_with_default(180, 220, core::CV_8UC3, Scalar::all(0.0))?;
    let mut seed = 1234567u32;
    for byte in image.data_bytes_mut()? {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        *byte = (seed % 200 + 30) as u8;
    }
    for i in 0..12 {
        imgproc::put_text(
            &mut image,
            &format!("{i}"),
            core::Point::new(15 + (i % 4) * 48, 35 + (i / 4) * 55),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(image)
}

fn scene(reference: &Mat, x: i32, y: i32, seed: u32) -> anyhow::Result<Mat> {
    let mut image = Mat::new_rows_cols_with_default(540, 800, core::CV_8UC3, Scalar::all(80.0))?;
    let mut state = seed;
    for byte in image.data_bytes_mut()? {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *byte = (state % 70 + 50) as u8;
    }
    reference.copy_to(&mut Mat::roi_mut(
        &mut image,
        Rect::new(x, y, reference.cols(), reference.rows()),
    )?)?;
    Ok(image)
}

#[test]
#[ignore = "requires downloaded NanoTrack v2 models and native OpenCV DNN"]
fn real_onnx_model_forward_and_tracking() -> anyhow::Result<()> {
    let reference = reference()?;
    let image = scene(&reference, 90, 80, 111)?;
    let mut nano = NanoTrack::load(
        "models/nanotrack_backbone_sim.onnx",
        "models/nanotrack_head_sim.onnx",
    )?;
    let expected = Bbox {
        x: 90.0,
        y: 80.0,
        width: 220.0,
        height: 180.0,
    };
    nano.init(&image, expected)?;
    let (bbox, score) = nano
        .update(&image, 0.0)?
        .expect("model produced finite prediction");
    eprintln!(
        "same-frame NanoTrack score={score:.4}, IoU={:.4}",
        bbox.iou(expected)
    );
    assert!(
        score > 0.5,
        "foreground score should exceed random baseline"
    );
    assert!(bbox.iou(expected) > 0.5);
    Ok(())
}

#[test]
#[ignore = "requires downloaded NanoTrack v2 models and native OpenCV DNN"]
fn retained_reference_model_loss_reacquisition_and_target_swap() -> anyhow::Result<()> {
    let reference = reference()?;
    let config = Config {
        gap_timeout: Duration::from_secs(30),
        verification_ttl: Duration::from_secs(30),
        ..Default::default()
    };
    let mut engine = Engine::new(
        config,
        "models/nanotrack_backbone_sim.onnx",
        "models/nanotrack_head_sim.onnx",
    )?;
    let image = scene(&reference, 90, 80, 123)?;
    assert_eq!(
        engine.process(&image, Header::default(), 0, None)?.status,
        "waiting_target"
    );
    assert_eq!(engine.set_target(&reference, None)?, 1);
    let first = engine.process(&image, Header::default(), 1, None)?;
    assert_eq!(first.status, "tracking_verified");
    assert_eq!(first.target_generation, 1);
    assert!(
        first.bbox.unwrap().iou(Bbox {
            x: 90.0,
            y: 80.0,
            width: 220.0,
            height: 180.0
        }) > 0.9
    );
    // New background; the reference is not resent.
    let second = engine.process(&scene(&reference, 95, 82, 456)?, Header::default(), 2, None)?;
    eprintln!(
        "hybrid second frame: {} score={}",
        second.status, second.model_score
    );
    assert!(second.bbox.is_some());
    assert_eq!(second.target_generation, 1);
    let blank = Mat::new_rows_cols_with_default(540, 800, core::CV_8UC3, Scalar::all(0.0))?;
    let loss = engine.process(&blank, Header::default(), 3, None)?;
    assert_eq!(loss.status, "blurred");
    assert!(loss.bbox.is_none());
    let reacquired = engine.process(
        &scene(&reference, 500, 300, 789)?,
        Header::default(),
        4,
        None,
    )?;
    assert_eq!(reacquired.status, "tracking_verified");
    assert_eq!(reacquired.target_generation, 1);
    assert!(engine.set_target(&blank, None).is_err());
    let retained = engine.process(
        &scene(&reference, 500, 300, 999)?,
        Header::default(),
        5,
        None,
    )?;
    assert_eq!(retained.target_generation, 1);
    assert_eq!(engine.set_target(&reference, None)?, 2);
    assert_eq!(
        engine
            .process(&image, Header::default(), 6, None)?
            .target_generation,
        2
    );
    Ok(())
}

#[test]
#[ignore = "requires downloaded models and OpenCV video encoding"]
fn video_file_cli_emits_original_resolution_boxes_and_jsonl() -> anyhow::Result<()> {
    use opencv::{imgcodecs, videoio};
    let directory = std::env::temp_dir().join(format!("tracker_cli_{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let target = directory.join("target.png");
    let video = directory.join("input.avi");
    let output = directory.join("output.jsonl");
    let object = reference()?;
    imgcodecs::imwrite(target.to_str().unwrap(), &object, &core::Vector::new())?;
    let codec = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    let mut writer = videoio::VideoWriter::new(
        video.to_str().unwrap(),
        codec,
        15.0,
        core::Size::new(800, 540),
        true,
    )?;
    assert!(writer.is_opened()?);
    for i in 0..6 {
        let frame = if i == 3 {
            Mat::new_rows_cols_with_default(540, 800, core::CV_8UC3, Scalar::all(0.0))?
        } else {
            scene(
                &object,
                if i < 3 { 90 + i * 3 } else { 500 },
                if i < 3 { 80 } else { 300 },
                123 + i as u32,
            )?
        };
        writer.write(&frame)?;
    }
    writer.release()?;
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_mobile-object-tracker"))
        .args([
            "--video",
            video.to_str().unwrap(),
            "--target",
            target.to_str().unwrap(),
            "--jsonl",
            output.to_str().unwrap(),
            "--max-width",
            "640",
        ])
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout)?;
    let lines = stdout
        .lines()
        .map(serde_json::from_str::<serde_json::Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[0]["image_width"], 800);
    assert!((lines[0]["bbox"]["x"].as_f64().unwrap() - 90.0).abs() < 8.0);
    assert_eq!(lines[3]["status"], "blurred");
    assert!(lines[3]["bbox"].is_null());
    assert_eq!(lines[4]["status"], "tracking_verified");
    assert_eq!(std::fs::read_to_string(&output)?, stdout);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
