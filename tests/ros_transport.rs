//! Requires a sourced, built ROS 2 environment; intentionally not a doc-only test.
#![cfg(feature = "ros2")]
use anyhow::{ensure, Result};
use futures::{FutureExt, StreamExt};
use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
};
use std::{
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Stop(Child);
impl Drop for Stop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn pattern(inverted: bool) -> Result<Mat> {
    let mut image = Mat::new_rows_cols_with_default(160, 200, core::CV_8UC3, Scalar::all(90.0))?;
    let mut seed = if inverted { 987654u32 } else { 123456u32 };
    for byte in image.data_bytes_mut()? {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        *byte = (seed % 220) as u8;
    }
    imgproc::put_text(
        &mut image,
        if inverted { "SECOND" } else { "FIRST" },
        core::Point::new(10, 90),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        Scalar::all(255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(image)
}

fn message(image: &Mat) -> Result<r2r::sensor_msgs::msg::Image> {
    Ok(r2r::sensor_msgs::msg::Image {
        height: image.rows() as u32,
        width: image.cols() as u32,
        encoding: "bgr8".into(),
        step: image.cols() as u32 * 3,
        data: image.data_bytes()?.to_vec(),
        header: r2r::std_msgs::msg::Header {
            frame_id: "test_optical".into(),
            stamp: r2r::builtin_interfaces::msg::Time {
                sec: 123,
                nanosec: 456,
            },
        },
        ..Default::default()
    })
}

#[test]
#[ignore = "requires real ROS 2, built/sourced mobile_tracker_interfaces, and downloaded models; do not run with ros2-doc"]
fn one_shot_target_replacement_bbox_and_source_timeout() -> Result<()> {
    let base = format!("/tracker_smoke_{}", std::process::id());
    let image_topic = format!("{base}/image");
    let target_topic = format!("{base}/target");
    let result_topic = format!("{base}/bbox");
    let context = r2r::Context::create()?;
    let mut node = r2r::Node::create(context, "tracker_smoke", "")?;
    let source = node.create_publisher::<r2r::sensor_msgs::msg::Image>(
        &image_topic,
        r2r::QosProfile::sensor_data(),
    )?;
    let target = node.create_publisher::<r2r::sensor_msgs::msg::Image>(
        &target_topic,
        r2r::QosProfile::default(),
    )?;
    let mut results = node.subscribe_untyped(
        &result_topic,
        "mobile_tracker_interfaces/msg/TrackingResult",
        r2r::QosProfile::default(),
    )?;
    let _child = Stop(
        Command::new(env!("CARGO_BIN_EXE_mobile-object-tracker"))
            .args([
                "--image-topic",
                &image_topic,
                "--target-topic",
                &target_topic,
                "--bbox-topic",
                &result_topic,
                "--source-timeout-ms",
                "300",
                "--max-width",
                "640",
            ])
            .stdout(Stdio::null())
            .spawn()?,
    );
    let started = Instant::now();
    while (target.get_inter_process_subscription_count()? == 0
        || source.get_inter_process_subscription_count()? == 0)
        && started.elapsed() < Duration::from_secs(15)
    {
        node.spin_once(Duration::from_millis(20));
    }
    ensure!(
        target.get_inter_process_subscription_count()? > 0,
        "tracker subscriber not discovered"
    );
    for (generation, inverted) in [(1, false), (2, true)] {
        let object = pattern(inverted)?;
        // Exactly one reference publication per target.
        target.publish(&message(&object)?)?;
        let mut scene =
            Mat::new_rows_cols_with_default(480, 640, core::CV_8UC3, Scalar::all(80.0))?;
        object.copy_to(&mut Mat::roi_mut(
            &mut scene,
            Rect::new(100, 120, 200, 160),
        )?)?;
        let frame = message(&scene)?;
        let started = Instant::now();
        let mut consecutive = 0;
        while started.elapsed() < Duration::from_secs(15) && consecutive < 3 {
            source.publish(&frame)?;
            node.spin_once(Duration::from_millis(50));
            for _ in 0..32 {
                let Some(Some(result)) = results.next().now_or_never() else {
                    break;
                };
                let value = result?;
                if value["found"] == true && value["target_generation"] == generation {
                    ensure!(
                        value["header"]["frame_id"] == "test_optical",
                        "frame_id not preserved"
                    );
                    ensure!(
                        value["header"]["stamp"]["sec"] == 123,
                        "stamp not preserved"
                    );
                    ensure!(
                        (value["x"].as_f64().unwrap() - 100.0).abs() < 30.0,
                        "unexpected bbox x"
                    );
                    consecutive += 1;
                }
            }
        }
        ensure!(
            consecutive >= 3,
            "did not receive retained target generation {generation} on multiple frames"
        );
    }
    // Stop publishing frames and require an explicit invalid result.
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(5) {
        node.spin_once(Duration::from_millis(50));
        for _ in 0..32 {
            let Some(Some(result)) = results.next().now_or_never() else {
                break;
            };
            let value = result?;
            if value["status"] == "source_timeout" {
                ensure!(
                    value["found"] == false && value["width"] == 0.0,
                    "timeout leaked an old bbox"
                );
                return Ok(());
            }
        }
    }
    anyhow::bail!("did not receive source_timeout");
}
