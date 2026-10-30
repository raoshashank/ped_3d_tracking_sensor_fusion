#!/usr/bin/env python3
"""Write erasor2/camera_model.npz (K, W, H, cam_T_base) for one bag (dataset from dataset.json) from ped3d's
per-trajectory extrinsics (camera pitch differs between recording sessions: -15..+1 deg).
Used only for the camera-FOV flag of the lidar detections; ped3d positions already had the right
extrinsics applied. Run with the calib env:  python make_camera_model.py <slam_output_dir>
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PED3D_DIR") or os.path.join(os.environ["P3D_REPO"], "ped3d"))   # repo ped3d/ (P3D_REPO: p3d_env.py)
import ped3d_lib as L  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_profile as DP  # noqa: E402

out = Path(sys.argv[1])
prof = DP.load(out)
ds, traj = prof["dataset"], prof["camera_traj"].format(bag=out.parent.name)
f = L.extrinsics_file(ds, traj)
if f is None:
    sys.exit(f"no extrinsics for {traj}")
W, H = L.DATASETS[ds]["img_wh"]
np.savez(out / "erasor2" / "camera_model.npz", K=L.load_intrinsics(ds), W=W, H=H,
         cam_T_base=np.loadtxt(f).reshape(4, 4))
print(f"camera_model.npz from {f}")
