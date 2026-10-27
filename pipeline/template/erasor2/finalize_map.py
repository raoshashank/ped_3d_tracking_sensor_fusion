#!/usr/bin/env python3
"""Turn an ERASOR2 output dir into the deliverable: static_map_slam_frame.pcd (ERASOR2 lifts the map
by extrinsic.sensor_height = 0.38 m; undo it), plus stats in summary.json (point count, fraction
of ped3d-masked voxels left in the map, raw-map voxels removed).
Run with the erasor2 env from the erasor2/ dir:  python finalize_map.py output_ped [--rm-lifted]
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # slam_output: dataset_profile.py
import dataset_profile as DP  # noqa: E402

LIFT = DP.load(Path(__file__).resolve().parent)["lidar_height"]   # = ERASOR2 extrinsic.sensor_height
out = Path(sys.argv[1])
rm_lifted = "--rm-lifted" in sys.argv
D = Path("data/Merged")
lifted = sorted(glob.glob(str(out / "*_estimated.pcd")))
if lifted:
    st = np.asarray(o3d.io.read_point_cloud(lifted[-1]).points) - [0, 0, LIFT]
    o3d.io.write_point_cloud(str(out / "static_map_slam_frame.pcd"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(st)))
elif (out / "static_map_slam_frame.pcd").exists():   # stats only (e.g. ERASOR2-only map, re-scored once ped labels exist)
    st = np.asarray(o3d.io.read_point_cloud(str(out / "static_map_slam_frame.pcd")).points)
else:
    sys.exit(f"no map in {out}")
tree = cKDTree(st)


def vox(a, v=0.05):
    return np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a)).voxel_down_sample(v).points)


P = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
raw, ped = [], []
for i in range(0, len(P), 2):
    s = np.fromfile(D / "velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    w = s @ P[i, :, :3].T + P[i, :, 3]
    raw.append(w)
    pf = D / "ped" / f"{i:06d}.label"
    if pf.exists():
        ped.append(w[np.fromfile(pf, np.uint32).astype(bool)])
raw = vox(np.vstack(raw))
stats = {"points": int(len(st)), "raw_voxels_removed": float(np.isinf(tree.query(raw, distance_upper_bound=0.1)[0]).mean())}
if ped and len(np.vstack(ped)):
    ped = vox(np.vstack(ped))
    stats["ped3d_voxels"] = int(len(ped))
    stats["ped3d_residue"] = float(np.isfinite(tree.query(ped, distance_upper_bound=0.05)[0]).mean())
sfile = Path("summary.json")
summary = json.load(open(sfile)) if sfile.exists() else {}
summary[out.name] = stats
json.dump(summary, open(sfile, "w"), indent=1)
print(out.name, stats)
if rm_lifted:
    for f in lifted:
        os.remove(f)
