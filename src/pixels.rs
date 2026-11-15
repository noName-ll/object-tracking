use anyhow::{bail, ensure, Result};

/// Convert a padded ROS sensor_msgs/Image buffer to tightly packed BGR8.
pub fn to_bgr(width: u32, height: u32, step: u32, encoding: &str, data: &[u8]) -> Result<Vec<u8>> {
    let channels = match encoding {
        "bgr8" | "rgb8" => 3,
        "bgra8" | "rgba8" => 4,
        "mono8" => 1,
        _ => bail!(
            "unsupported encoding {encoding}; convert to bgr8/rgb8/mono8/bgra8/rgba8 upstream"
        ),
    };
    ensure!(width > 0 && height > 0, "zero image dimensions");
    let row = (width as usize)
        .checked_mul(channels)
        .ok_or_else(|| anyhow::anyhow!("image too large"))?;
    let total = (step as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| anyhow::anyhow!("image too large"))?;
    ensure!(
        step as usize >= row && data.len() >= total,
        "invalid step or truncated image data"
    );
    let output_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("image too large"))?;
    ensure!(
        output_len <= 256 * 1024 * 1024,
        "image exceeds 256 MiB safety limit"
    );
    let mut output = Vec::with_capacity(output_len);
    for y in 0..height as usize {
        for pixel in data[y * step as usize..y * step as usize + row].chunks_exact(channels) {
            match encoding {
                "mono8" => output.extend_from_slice(&[pixel[0]; 3]),
                "rgb8" | "rgba8" => output.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]),
                _ => output.extend_from_slice(&pixel[..3]),
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn respects_padding_and_rgb_order() {
        assert_eq!(
            to_bgr(1, 2, 4, "rgb8", &[1, 2, 3, 99, 4, 5, 6, 99]).unwrap(),
            vec![3, 2, 1, 6, 5, 4]
        );
    }
    #[test]
    fn rejects_bad_buffers_and_depth_images() {
        assert!(to_bgr(2, 2, 6, "bgr8", &[0; 6]).is_err());
        assert!(to_bgr(2, 2, 2, "bgr8", &[0; 12]).is_err());
        assert!(to_bgr(1, 1, 2, "16UC1", &[0; 2]).is_err());
    }
}
