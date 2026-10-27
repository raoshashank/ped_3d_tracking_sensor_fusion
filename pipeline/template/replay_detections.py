#!/usr/bin/env python3
"""Replay the bag in RViz2 with the cleaned static map, the live residual (scan points that are not
in the map = moving things, 360 deg) and the lidar-vs-map pedestrian detections + ped3d tracks.

Everything replay_with_map.py publishes (map, path, map->odom TF), plus:
  /residual     sensor_msgs/PointCloud2  scan points > RES_DIST from the static map, `map` frame,
                                         intensity = height above ground
  /detections   visualization_msgs/MarkerArray  lidar-vs-map detections at the current scan:
                                         cylinders, blue = inside camera FOV, red = outside,
                                         label = track id, faint = gap filled by the smoother;
                                         green spheres = ped3d (camera) tracks, label p<ped3d id>
  /fused_tracks visualization_msgs/MarkerArray  fused persons (fuse_tracks.py), label P<person id>,
                                         green camera_verified, purple camera_only, orange
                                         lidar_only, grey unconfirmed_in_fov; faint = filled

Terminal 1:  ros2 bag play <bag_dir> --clock
Terminal 2:  python3 replay_detections.py            (system python, ROS 2 Humble sourced)
Terminal 3:  rviz2 -d detections_replay.rviz
"""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R, Slerp
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from replay_with_map import HERE, SlamReplay, read_pcd, to_stamp

E = HERE / "erasor2"
MAP = E / "output_ped" / "static_map_slam_frame.pcd"
LIDAR_TRACKS = E / "output_ped" / "lidar_tracks.csv"
LIDAR_SMOOTH = E / "output_ped" / "lidar_tracks_smooth.csv"   # lidar_tracker.py (preferred when present)
FUSED_TRACKS = E / "output_ped" / "fused_tracks.csv"          # fuse_tracks.py
LABELS = ["camera_verified", "camera_only", "lidar_only", "unconfirmed_in_fov"]
LABEL_RGB = [(0.1, 0.9, 0.2), (0.7, 0.3, 1.0), (1.0, 0.6, 0.0), (0.6, 0.6, 0.6)]   # green purple orange grey
PED3D_TRACKS = E / "output_ped" / "ped3d_tracks_map.csv"
RES_DIST = 0.15          # m, same as lidar_ped_detect.py
BASE_TO_GROUND = 0.35    # base_link_to_ground.yaml (go2nus)
MIN_H, MAX_RANGE = 0.1, 20.0


def cloud_msg(pts, stamp, frame="map"):
    return PointCloud2(header=Header(frame_id=frame, stamp=stamp), height=1, width=len(pts),
                       fields=[PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                               for i, n in enumerate(["x", "y", "z", "intensity"])],
                       is_bigendian=False, point_step=16, row_step=16 * len(pts), is_dense=True,
                       data=np.ascontiguousarray(pts, dtype=np.float32).tobytes())


class DetectionReplay(SlamReplay):
    def __init__(self, voxel, map_path):
        super().__init__(voxel, map_path, name="detection_replay")
        full = read_pcd(map_path)
        self.tree = cKDTree(full[:, :3])
        self.get_logger().info(f"KD-tree over {len(full)} map points ready")
        lid = np.loadtxt(HERE / "poses_rslidar_tum.txt")
        self.tl, self.pl = lid[:, 0], lid[:, 1:4]
        self.slerp_l = Slerp(self.tl, R.from_quat(lid[:, 4:8]))
        if LIDAR_SMOOTH.exists():   # v2 tracker: smoothed positions every scan, gaps filled
            s = np.loadtxt(LIDAR_SMOOTH, delimiter=",", skiprows=1)
            # -> track, scan, t, x, y, top_h, height, n, range, in_fov, detected (height 1.6 m when filled)
            top = np.where(np.isfinite(s[:, 11]), s[:, 11], 1.6)
            d = np.c_[s[:, 0:5], top, top, np.zeros(len(s)), s[:, 10], s[:, 9], s[:, 8]]
        else:
            d = np.loadtxt(LIDAR_TRACKS, delimiter=",", skiprows=1)
            d = np.c_[d, np.ones(len(d))]
        self.det = d[np.argsort(d[:, 2])]
        self.ped = {}
        for row in np.loadtxt(PED3D_TRACKS, delimiter=",", skiprows=1):
            self.ped.setdefault(int(row[0]), []).append(row[1:])
        self.ped = {k: np.array(v) for k, v in self.ped.items()}
        self.fused = None
        if FUSED_TRACKS.exists():   # fuse_tracks.py: one track set, person ids
            import csv
            rows = list(csv.DictReader(open(FUSED_TRACKS)))
            self.fused = np.array([(int(r["person"]), float(r["t"]), float(r["x"]), float(r["y"]),
                                    float(r["top_h"]) if r["top_h"] != "nan" else 1.6,
                                    r["source"] != "filled", LABELS.index(r["label"])) for r in rows])
            self.fused = self.fused[np.argsort(self.fused[:, 1])]
            self.pub_fused = self.create_publisher(MarkerArray, "/fused_tracks", 5)
        self.pub_res = self.create_publisher(PointCloud2, "/residual", 5)
        self.pub_mk = self.create_publisher(MarkerArray, "/detections", 5)
        self.create_subscription(PointCloud2, "/rslidar_points", self.on_scan,
                                 QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f"{len(self.det)} lidar detections, {len(self.ped)} ped3d tracks loaded")

    def on_scan(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t < self.tl[0] - 0.05 or t > self.tl[-1] + 0.05:
            return
        tc = float(np.clip(t, self.tl[0], self.tl[-1]))
        Rm = self.slerp_l([tc]).as_matrix()[0]
        pos = np.array([np.interp(tc, self.tl, self.pl[:, k]) for k in range(3)])
        dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3, "offsets": [0, 4, 8],
                       "itemsize": msg.point_step})
        arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
        s = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
        s = s[np.isfinite(s).all(1)]
        rng = np.linalg.norm(s, axis=1)
        s = s[(rng > 0.6) & (rng < MAX_RANGE)]
        w = s @ Rm.T + pos
        ground_z = np.interp(tc, self.t, self.pos[:, 2]) - BASE_TO_GROUND
        h = w[:, 2] - ground_z
        keep = (h > MIN_H) & np.isinf(self.tree.query(w, distance_upper_bound=RES_DIST, workers=4)[0])
        self.pub_res.publish(cloud_msg(np.c_[w[keep], h[keep]], msg.header.stamp))
        self.pub_mk.publish(self.markers(t, ground_z, msg.header.stamp))
        if self.fused is not None:
            self.pub_fused.publish(self.fused_markers(t, ground_z, msg.header.stamp))

    def fused_markers(self, t, ground_z, stamp):
        ma = MarkerArray()
        ma.markers.append(Marker(header=Header(frame_id="map", stamp=stamp), ns="", id=0, action=Marker.DELETEALL))
        lo, hi = np.searchsorted(self.fused[:, 1], [t - 0.06, t + 0.06])
        win = self.fused[lo:hi]
        best = {}   # one row per person: the one nearest in time
        for row in win:
            if row[0] not in best or abs(row[1] - t) < abs(best[row[0]][1] - t):
                best[row[0]] = row
        for k, (pid, _, x, y, top, measured, lab) in enumerate(best.values(), start=1):
            r, g, b = LABEL_RGB[int(lab)]
            m = Marker(header=Header(frame_id="map", stamp=stamp), ns="fused", id=k, type=Marker.CYLINDER,
                       action=Marker.ADD, color=ColorRGBA(r=r, g=g, b=b, a=0.7 if measured else 0.3))
            h = max(top, 0.3)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, ground_z + h / 2
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 0.5, 0.5, h
            m.lifetime.nanosec = 300_000_000
            ma.markers.append(m)
            txt = Marker(header=Header(frame_id="map", stamp=stamp), ns="fused_id", id=k, type=Marker.TEXT_VIEW_FACING,
                         action=Marker.ADD, text=f"P{int(pid)}", color=ColorRGBA(r=r, g=g, b=b, a=1.0))
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, ground_z + h + 0.35
            txt.scale.z = 0.35
            txt.lifetime.nanosec = 300_000_000
            ma.markers.append(txt)
        return ma

    def markers(self, t, ground_z, stamp):
        ma = MarkerArray()
        ma.markers.append(Marker(header=Header(frame_id="map", stamp=stamp), ns="", id=0, action=Marker.DELETEALL))
        k = 1
        lo, hi = np.searchsorted(self.det[:, 2], [t - 0.06, t + 0.06])
        for row in self.det[lo:hi]:
            tid, x, y, height, fov = int(row[0]), row[3], row[4], max(row[6], 0.3), row[9] > 0
            alpha = 0.6 if row[10] > 0 else 0.25   # faint: position filled by the smoother (no detection)
            color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=alpha) if fov else ColorRGBA(r=1.0, g=0.2, b=0.2, a=alpha)
            m = Marker(header=Header(frame_id="map", stamp=stamp), ns="lidar", id=k, type=Marker.CYLINDER,
                       action=Marker.ADD, color=color)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, ground_z + height / 2
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 0.6, 0.6, height
            m.lifetime.nanosec = 300_000_000
            ma.markers.append(m)
            txt = Marker(header=Header(frame_id="map", stamp=stamp), ns="lidar_id", id=k, type=Marker.TEXT_VIEW_FACING,
                         action=Marker.ADD, text=str(tid), color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, ground_z + height + 0.3
            txt.scale.z = 0.3
            txt.lifetime.nanosec = 300_000_000
            ma.markers.append(txt)
            k += 1
        for tid, tr in self.ped.items():
            if tr[0, 0] - 0.15 <= t <= tr[-1, 0] + 0.15:
                x, y = np.interp(t, tr[:, 0], tr[:, 1]), np.interp(t, tr[:, 0], tr[:, 2])
                m = Marker(header=Header(frame_id="map", stamp=stamp), ns="ped3d", id=k, type=Marker.SPHERE,
                           action=Marker.ADD, color=ColorRGBA(r=0.2, g=1.0, b=0.3, a=0.9))
                m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, ground_z + 1.0
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.35
                m.lifetime.nanosec = 300_000_000
                ma.markers.append(m)
                txt = Marker(header=Header(frame_id="map", stamp=stamp), ns="ped3d_id", id=k, type=Marker.TEXT_VIEW_FACING,
                             action=Marker.ADD, text=f"p{tid}", color=ColorRGBA(r=0.4, g=1.0, b=0.4, a=1.0))
                txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, ground_z + 0.6
                txt.scale.z = 0.25
                txt.lifetime.nanosec = 300_000_000
                ma.markers.append(txt)
                k += 1
        return ma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.1, help="downsample the displayed map (m)")
    ap.add_argument("--map", default=str(MAP))
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = DetectionReplay(args.voxel, Path(args.map))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
