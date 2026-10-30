"""Shared paths and style for the blog figures (demo bag: utown_1 ... _10)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

BAG = Path(p3d_env.P("P3D_BAGS_GO2NUS")) / "utown_1_2025-03-07_12-41-18_10_corrected_merged"
S = BAG / "slam_output"
E = S / "erasor2"
PROC = Path(p3d_env.P("P3D_PROCESSED")) / "go2nus" / (BAG.name + "_0")
FIG = Path(p3d_env.P("P3D_FIG_OUT"))
GROUND_Z = -0.25          # map z of the ground (SLAM map frame: origin at base_link, 0.35 m above ground... lidar 0.13 above)
C_PED3D, C_IN, C_OUT = "#2ca02c", "#1f77b4", "#d62728"
MAP_GREY = "0.78"

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 120, "savefig.dpi": 120,
                     "savefig.facecolor": "white", "axes.facecolor": "white"})


def read_pcd(path):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def slam_poses():
    b = np.loadtxt(S / "poses_base_link_tum.txt")
    return b[:, 0], b[:, 1:4], b[:, 4:8]


def height_slice(p, lo=0.2, hi=2.0):
    h = p[:, 2] - GROUND_Z
    m = (h > lo) & (h < hi)
    return p[m], h[m]
