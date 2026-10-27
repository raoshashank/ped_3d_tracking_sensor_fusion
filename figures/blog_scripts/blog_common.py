"""Shared paths and style for the blog figures (demo bag: utown_1 ... _10)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BAG = Path("/media/dataset_access/go2nus_annotated/utown_1_2025-03-07_12-41-18_10_corrected_merged")
S = BAG / "slam_output"
E = S / "erasor2"
PROC = Path("/media/shashank/T1/processed/go2nus") / (BAG.name + "_0")
FIG = Path("/media/dataset_access/ped3d_blog/figures")
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
