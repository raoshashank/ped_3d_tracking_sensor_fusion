"""Per-dataset robot/sensor setup for the pipeline scripts.

Values come from <bag>/slam_output/dataset.json (written by run_all.py --dataset ...). A missing file or
key falls back to the go2nus setup (Unitree Go2: Hesai XT-16 /rslidar_points, /utlidar/robot_odom), so
go2nus bags processed before this file existed behave exactly as before.

Everything height-dependent is derived from lidar_height (lidar above the ground):
  Patchwork++ / ERASOR2 sensor_height, ERASOR2 lift  = lidar_height
  ERASOR2 min_z_voi = -(lidar_height + 0.62)  (0.6 m below the ground, for slopes; go2nus -1.0)
  ERASOR2 max_z_voi = CEIL_Z - lidar_height   (go2nus 1.8)
  pedestrian mask band = PED_Z_MIN .. CEIL_Z above the ground (go2nus lidar z -0.28 .. 1.8)
  nominal ground when no Patchwork ground is near = lidar_height + 0.10 below the lidar (go2nus 0.48)
"""
import json
import os
from pathlib import Path

import numpy as np

GO2NUS = dict(
    dataset="go2nus",
    lidar_topic="/rslidar_points",
    lidar_offsets=[0, 4, 8, 16],          # byte offsets of x y z intensity (float32)
    lidar_time_offset=24,                 # per-point absolute stamp (float64, s); None: use the header stamp
    odom_topic="/utlidar/robot_odom",
    imu_topic=None,                       # gravity direction for the pose graph: None = odometry orientation
    floor_attitude=False,                 # gravity direction from the floor plane in each scan (IMU as fallback)
    gravity_sigma_deg=2.0,                # attitude factor noise (Go2 IMU tilt is noisy while trotting)
    floor_z_sigma=None,                   # m: prior that the floor plane seen in a scan sits at map z = 0
                                          # (level floors; VLP-16 ICP drifts in z). None: off
    base_T_lidar=[0.14543, 0.0, 0.13312, 0.0],   # x y z [m] yaw [rad]: SLAM base frame -> lidar (/tf_static)
    lidar_height=0.38,                    # lidar above the ground [m]
    ped3d_pkl="$P3D_PED3D_OUT/go2nus/{bag}_0.pkl",       # $P3D_* expanded from the environment
    camera_traj="{bag}_0",                # ped3d_lib trajectory name for the camera extrinsics
)

MIRAIKAN = dict(
    dataset="miraikan",
    lidar_topic="/velodyne_points",       # VLP-16, fields x y z intensity ring(u16@16) time(f32@18, all 0)
    lidar_offsets=[0, 4, 8, 12],
    lidar_time_offset=None,
    odom_topic="/odom",                   # odom -> base_footprint, planar (z = 0, no roll/pitch)
    imu_topic="/cabot/imu/data",          # imu_frame = base_control_shift, same orientation as base_footprint
    # The IMU's gravity direction is off by up to ~2 deg in some sessions (vs the floor seen by the lidar,
    # 6 bags / 6 sessions), which tilts the map (10 cm z drift over 10 m). Museum floors are level.
    floor_attitude=True,
    gravity_sigma_deg=0.5,
    floor_z_sigma=0.03,                   # ICP alone sank 11 cm over 10 m with the floor 0.55-0.57 m below throughout
    # base_footprint -> velodyne: x, y, yaw from ped3d_lib (= /tf chain); z = lidar height above the floor
    # from RANSAC floor planes (0.553, p10-p90 0.549-0.559 over 6 sessions; /tf_static chain 0.560,
    # dynamic /tf chain 0.522). Mount roll/pitch vs the base < 0.25 deg: not modelled.
    base_T_lidar=[0.248, -0.003, 0.553, 0.002],
    lidar_height=0.553,
    ped3d_pkl="$P3D_PED3D_OUT/miraikan/{bag}.pkl",
    camera_traj="{bag}",
)

PROFILES = {"go2nus": GO2NUS, "miraikan": MIRAIKAN}
CEIL_Z, PED_Z_MIN = 2.18, 0.10


def _find_json(start):
    for d in (start, start.parent):
        if (d / "dataset.json").exists():
            return d / "dataset.json"
    return None


def load(start=None):
    """Profile dict for the bag whose slam_output (or slam_output/erasor2) contains `start`."""
    start = Path(start or Path.cwd()).resolve()
    f = _find_json(start)
    p = dict(GO2NUS)
    if f is not None:
        p.update(json.load(open(f)))
    return p


def base_T_lidar(p):
    x, y, z, yaw = p["base_T_lidar"]
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def write(slam_output, name, bag):
    """Write slam_output/dataset.json for profile `name` with the bag-specific paths filled in."""
    p = dict(PROFILES[name])
    p["ped3d_pkl"] = os.path.expandvars(p["ped3d_pkl"]).format(bag=bag)   # needs P3D_PED3D_OUT
    p["camera_traj"] = p["camera_traj"].format(bag=bag)
    (Path(slam_output) / "dataset.json").write_text(json.dumps(p, indent=1))
    return p
