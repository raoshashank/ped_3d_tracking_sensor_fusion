#!/usr/bin/env python3
"""Export the bag's lidar scans + SLAM poses in the layout ERASOR2's HeLiPR loader expects:

  data/Merged/velodyne/NNNNNN.bin   float32 x y z intensity, rslidar frame
  data/Merged/poses.txt             KITTI 3x4 row-major, map <- rslidar (one line per scan)

Scan i matches line i of ../poses_rslidar_tum.txt (both are in /rslidar_points order).
Run with the kiss_icp env:  python export_scans.py
"""
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # slam_output: dataset_profile.py
import dataset_profile as DP  # noqa: E402

HERE = Path(__file__).resolve().parent
BAG = HERE.parent.parent
SEQ = HERE / "data" / "Merged"
MIN_RANGE, MAX_RANGE, VOXEL = 0.6, 80.0, 0.05
PROF = DP.load(HERE)


def main():
    (SEQ / "velodyne").mkdir(parents=True, exist_ok=True)
    tum = np.loadtxt(HERE.parent / "poses_rslidar_tum.txt")
    with open(SEQ / "poses.txt", "w") as f:
        for row in tum:
            T = np.eye(4)
            T[:3, :3] = R.from_quat(row[4:8]).as_matrix()
            T[:3, 3] = row[1:4]
            f.write(" ".join(f"{v:.9f}" for v in T[:3].ravel()) + "\n")

    # corrupt scans that run_slam.py skipped (message indices): skip them too, so scan i <-> pose i
    sk = HERE.parent / "skipped_scans.txt"
    skipped = {int(l) for l in sk.read_text().split()} if sk.exists() else set()
    n = 0
    with AnyReader([BAG], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:   # some bags lack type definitions
        conns = [c for c in reader.connections if c.topic == PROF["lidar_topic"]]
        for k_msg, (conn, _, raw) in enumerate(reader.messages(connections=conns)):
            if k_msg in skipped:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            dt = np.dtype({"names": ["x", "y", "z", "intensity"], "formats": ["<f4"] * 4,
                           "offsets": list(PROF["lidar_offsets"]), "itemsize": msg.point_step})
            arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
            pts = np.stack([arr["x"], arr["y"], arr["z"], arr["intensity"]], axis=1)
            pts = pts[np.isfinite(pts).all(axis=1)]
            rng = np.linalg.norm(pts[:, :3], axis=1)
            pts = pts[(rng > MIN_RANGE) & (rng < MAX_RANGE)]
            pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(pts[:, :3]))
            pcd.point.intensity = o3d.core.Tensor(pts[:, 3:4])
            pcd = pcd.voxel_down_sample(VOXEL)
            out = np.hstack([pcd.point.positions.numpy(), pcd.point.intensity.numpy()]).astype(np.float32)
            out.tofile(SEQ / "velodyne" / f"{n:06d}.bin")
            n += 1
    assert n == len(tum), f"{n} scans but {len(tum)} poses"
    print(f"exported {n} scans + poses to {SEQ}")


if __name__ == "__main__":
    main()
