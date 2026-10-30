"""Shared helpers for the camera-side (ped3d) blog figures 01-05.
Run the figure scripts with $P3D_CONDA_ENVS/calib/bin/python."""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ped3d"))
import ped3d_lib as L  # noqa: E402

DATASET = "go2nus"
TRAJ = "utown_1_2025-03-07_12-41-18_10_corrected_merged_0"
FRAME = 34                      # camera frame used in figures 01 and 03 (t ~ 8.6 s)
FIG = Path(p3d_env.P("P3D_FIG_OUT"))
PED3D_PKL = Path(p3d_env.P("P3D_PED3D_OUT")) / "go2nus" / f"{TRAJ}.pkl"

C = dict(ped3d="#2ca02c", lidar_in="#1f77b4", lidar_out="#d62728",
         lidar="#1f77b4", vggt="#ff7f0e", vggt_c="#e6ab02", vggt_a="#2ca02c", da="#9467bd")
SRC_LABEL = dict(lidar="lidar cluster", lidar_g="lidar cluster (VGGT-guided)", vggt="VGGT raw",
                 vggt_c="VGGT range-corrected", vggt_a="VGGT lidar-anchored", da="DepthAnything")

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 11,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 120})


def traj():
    return L.Traj(DATASET, TRAJ)


def ped3d():
    return pickle.load(open(PED3D_PKL, "rb"))


def lidar_in_base(T, k):
    p = T.lidar(k)
    return p @ T.base_T_lidar[:3, :3].T + T.base_T_lidar[:3, 3]


def project_lidar(T, k):
    """lidar points of frame k -> (u, v, range) of those in the image."""
    p = T.lidar(k)
    pc = p @ T.cam_T_lidar[:3, :3].T + T.cam_T_lidar[:3, 3]
    z = pc[:, 2]
    m = z > 0.3
    u = T.K[0, 0] * pc[m, 0] / z[m] + T.K[0, 2]
    v = T.K[1, 1] * pc[m, 1] / z[m] + T.K[1, 2]
    rng = np.linalg.norm(p[m], axis=1)
    inb = (u >= 0) & (u < T.W) & (v >= 0) & (v < T.H)
    return u[inb], v[inb], rng[inb]


def fov_half_angle_and_yaw(T):
    """Camera horizontal half-FOV and camera optical-axis yaw in base_link."""
    half = np.arctan2(T.W / 2, T.K[0, 0])
    z = T.base_T_cam[:3, 2]                      # optical axis in base
    return half, np.arctan2(z[1], z[0]), T.base_T_cam[:2, 3]


def draw_fov(ax, origin_xy, yaw, half, length, **kw):
    a0, a1 = yaw - half, yaw + half
    pts = np.array([origin_xy, origin_xy + length * np.array([np.cos(a0), np.sin(a0)]),
                    origin_xy + length * np.array([np.cos(a1), np.sin(a1)])])
    ax.fill(pts[:, 0], pts[:, 1], **kw)
