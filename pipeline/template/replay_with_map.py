#!/usr/bin/env python3
"""Show the offline SLAM result alongside a replayed rosbag in RViz2.

Publishes (use with `ros2 bag play <bag> --clock`):
  /slam_map   sensor_msgs/PointCloud2  global map in `map` frame (transient-local, published once)
  /slam_path  nav_msgs/Path            full SLAM trajectory of base_link (transient-local)
  /tf         map -> odom              correction so base_link in `map` follows the SLAM pose
  /tf_static  base_link -> rslidar / base_footprint / front_camera_link / camera_link

map -> odom is computed for every /utlidar/robot_odom message as
  T_map_odom(t) = T_map_base_SLAM(t) * inv(T_odom_base_ODOM(t)),
with the SLAM pose interpolated at t. The bag's odom -> base_link TF then lands the robot
(and its live /rslidar_points scans) exactly on the SLAM trajectory.

Usage:  python3 replay_with_map.py [--voxel 0.1] [--map erasor2/output_ped/static_map_slam_frame.pcd]
        (system python with ROS 2 Humble sourced)
"""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R, Slerp
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

HERE = Path(__file__).resolve().parent
STATIC_TFS = [  # (parent, child, xyz) from the bag's /tf_static, all identity rotations
    ("base_link", "base_footprint", (0.0, 0.0, -0.35)),
    ("base_link", "rslidar", (0.14543, 0.0, 0.13312)),
    ("base_link", "front_camera_link", (0.3215, 0.0, 0.04297)),
    ("front_camera_link", "camera_link", (0.01, 0.03, 0.06)),
]


def read_pcd(path):
    """Reader for binary or ascii PCDs with float32 x y z [intensity ...] (run_slam.py, open3d, PCL).
    Returns (N, 4) x y z intensity; intensity is 0 if the file has none."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode().strip()
            key, *vals = line.split()
            header[key] = vals
            if key == "DATA":
                break
        fields, n = header["FIELDS"], int(header["POINTS"][0])
        if header["DATA"][0] == "binary":
            dt = np.dtype([(fld, {"F": "f", "I": "i", "U": "u"}[ty] + sz)
                           for fld, ty, sz in zip(fields, header["TYPE"], header["SIZE"])])
            arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            cols = [arr[fld].astype(np.float32) for fld in ("x", "y", "z")]
            cols.append(arr["intensity"].astype(np.float32) if "intensity" in fields else np.zeros(n, np.float32))
            return np.stack(cols, 1)
        arr = np.loadtxt(f, dtype=np.float32, max_rows=n)
        pts = arr[:, [fields.index(c) for c in ("x", "y", "z")]]
        inten = arr[:, fields.index("intensity")] if "intensity" in fields else np.zeros(n, np.float32)
        return np.c_[pts, inten]


def voxel_downsample(pts, voxel):
    if voxel <= 0:
        return pts
    _, idx = np.unique(np.floor(pts[:, :3] / voxel).astype(np.int64), axis=0, return_index=True)
    return pts[idx]


def to_matrix(t, q):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def to_stamp(t):
    from builtin_interfaces.msg import Time
    return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)) % 1_000_000_000)


class SlamReplay(Node):
    def __init__(self, voxel, map_path=HERE / "map.pcd", name="slam_replay"):
        super().__init__(name, parameter_overrides=[Parameter("use_sim_time", value=True)])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)

        tum = np.loadtxt(HERE / "poses_base_link_tum.txt")
        self.t = tum[:, 0]
        self.pos = tum[:, 1:4]
        self.slerp = Slerp(self.t, R.from_quat(tum[:, 4:8]))

        # Map cloud
        pts = voxel_downsample(read_pcd(map_path), voxel)
        cloud = PointCloud2(header=Header(frame_id="map", stamp=to_stamp(self.t[0])), height=1, width=len(pts),
                            fields=[PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                                    for i, n in enumerate(["x", "y", "z", "intensity"])],
                            is_bigendian=False, point_step=16, row_step=16 * len(pts), is_dense=True,
                            data=pts.astype(np.float32).tobytes())
        self.create_publisher(PointCloud2, "/slam_map", latched).publish(cloud)
        self.get_logger().info(f"published /slam_map with {len(pts)} points (voxel {voxel} m)")

        # Trajectory
        path = PathMsg(header=Header(frame_id="map", stamp=to_stamp(self.t[0])))
        for row in tum:
            ps = PoseStamped(header=Header(frame_id="map", stamp=to_stamp(row[0])))
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = row[1:4]
            (ps.pose.orientation.x, ps.pose.orientation.y,
             ps.pose.orientation.z, ps.pose.orientation.w) = row[4:8]
            path.poses.append(ps)
        self.create_publisher(PathMsg, "/slam_path", latched).publish(path)

        # Static TFs (republished so RViz gets them even if it starts after the bag's /tf_static)
        self.static_br = StaticTransformBroadcaster(self)
        statics = []
        for parent, child, xyz in STATIC_TFS:
            tf = TransformStamped(header=Header(frame_id=parent), child_frame_id=child)
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = xyz
            tf.transform.rotation.w = 1.0
            statics.append(tf)
        self.static_br.sendTransform(statics)

        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.on_odom, 50)
        self.get_logger().info("waiting for /utlidar/robot_odom from `ros2 bag play ... --clock`")

    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        tc = float(np.clip(t, self.t[0], self.t[-1]))
        T_map_base = to_matrix([np.interp(tc, self.t, self.pos[:, k]) for k in range(3)],
                               self.slerp([tc]).as_quat()[0])
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        T_odom_base = to_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        T = T_map_base @ np.linalg.inv(T_odom_base)

        tf = TransformStamped(header=Header(frame_id="map", stamp=msg.header.stamp), child_frame_id="odom")
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = T[:3, 3]
        qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = qx, qy, qz, qw
        self.br.sendTransform(tf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.1, help="downsample map for RViz (m); 0 = full map.pcd")
    ap.add_argument("--map", default=str(HERE / "map.pcd"), help="PCD to show as /slam_map")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = SlamReplay(args.voxel, Path(args.map))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
