//! Publish a reference image once, after DDS discovers a matching subscriber.
use anyhow::{ensure, Result};
use clap::Parser;
use mobile_object_tracker::tracker::read_image;
use opencv::prelude::*;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    file: String,
    #[arg(long, default_value = "/tracker/target_image")]
    topic: String,
    #[arg(
        long,
        help = "Keep reference available to late joiners while this publisher stays alive"
    )]
    transient_local: bool,
    #[arg(long, default_value_t = 10)]
    wait_seconds: u64,
    #[arg(
        long,
        default_value_t = 2,
        help = "Remain alive after the single publication; increase for late subscribers"
    )]
    keep_alive_seconds: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let image = read_image(&args.file)?;
    let context = r2r::Context::create()?;
    let mut node = r2r::Node::create(context, "send_tracker_target", "")?;
    let mut qos = r2r::QosProfile {
        depth: 1,
        ..r2r::QosProfile::default()
    };
    if args.transient_local {
        qos.durability = r2r::qos::DurabilityPolicy::TransientLocal;
    }
    let publisher = node.create_publisher::<r2r::sensor_msgs::msg::Image>(&args.topic, qos)?;
    let started = Instant::now();
    while publisher.get_inter_process_subscription_count()? == 0
        && started.elapsed() < Duration::from_secs(args.wait_seconds)
    {
        node.spin_once(Duration::from_millis(20));
    }
    ensure!(
        publisher.get_inter_process_subscription_count()? > 0,
        "no compatible subscriber discovered; start tracker and check ROS_DOMAIN_ID, topic and QoS"
    );
    let message = r2r::sensor_msgs::msg::Image {
        height: image.rows() as u32,
        width: image.cols() as u32,
        encoding: "bgr8".into(),
        is_bigendian: 0,
        step: image.cols() as u32 * 3,
        data: image.data_bytes()?.to_vec(),
        ..Default::default()
    };
    publisher.publish(&message)?;
    eprintln!("reference published ONCE to {}", args.topic);
    let sent = Instant::now();
    while sent.elapsed() < Duration::from_secs(args.keep_alive_seconds) {
        node.spin_once(Duration::from_millis(20));
    }
    Ok(())
}
