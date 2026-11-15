#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo 'Source /opt/ros/humble/setup.bash or /opt/ros/jazzy/setup.bash first.' >&2
  exit 1
fi
colcon build --base-paths ros2 --build-base build/interfaces --install-base install/interfaces \
  --packages-select mobile_tracker_interfaces
set +u
source install/interfaces/setup.bash
set -u
export IDL_PACKAGE_FILTER='builtin_interfaces;std_msgs;sensor_msgs;geometry_msgs;unique_identifier_msgs;action_msgs;mobile_tracker_interfaces'
cargo build --release --locked --features ros2
echo 'Build complete. In every new terminal, source ROS and install/interfaces/setup.bash.'
