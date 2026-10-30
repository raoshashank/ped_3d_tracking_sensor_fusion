#!/usr/bin/env python3
"""Write slam_output/slam_qc.json for every bag that doesn't have one (runs made before the check
existed), using run_slam.slam_qc: SLAM vs onboard odometry over 1 s windows. Run with the kiss_icp env."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent / "template"))
from run_slam import slam_qc  # noqa: E402
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

ROOT = Path(p3d_env.P("P3D_BAGS_GO2NUS"))
n = flagged = 0
for d in sorted(ROOT.iterdir()):
    o = d / "slam_output"
    if not (o / "poses_base_link_tum.txt").exists() or (o / "slam_qc.json").exists() or not (o / "robot_odom_tum.txt").exists():
        continue
    s, od = np.loadtxt(o / "poses_base_link_tum.txt"), np.loadtxt(o / "robot_odom_tum.txt")
    def mats(a):
        T = np.tile(np.eye(4), (len(a), 1, 1)); T[:, :3, :3] = R.from_quat(a[:, 4:8]).as_matrix(); T[:, :3, 3] = a[:, 1:4]
        return list(T)
    qc = slam_qc(s[:, 0], mats(s), od[:, 0], mats(od))
    (o / "slam_qc.json").write_text(json.dumps(qc, indent=1))
    n += 1; flagged += qc["flag"]
print(f"wrote {n} slam_qc.json, flagged {flagged}")
