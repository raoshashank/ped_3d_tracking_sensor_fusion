"""Shared paths, colours and loaders for the tracking / fusion / leg-detector blog figures (11-16).
Demo bag: utown_1_2025-03-07_12-41-18_10_corrected_merged. Run with the erasor2 conda env."""
import csv
from pathlib import Path

import numpy as np

BAG = Path("/media/dataset_access/go2nus_annotated/utown_1_2025-03-07_12-41-18_10_corrected_merged")
SO = BAG / "slam_output"
E = SO / "erasor2"
OP = E / "output_ped"
FIG = Path("/media/dataset_access/ped3d_blog/figures")

C_PED3D = "#2ca02c"
C_IN, C_OUT = "#1f77b4", "#d62728"
LABELS = ["camera_verified", "camera_only", "lidar_only", "unconfirmed_in_fov"]
C_LABEL = {"camera_verified": "#1a9e3f", "camera_only": "#9b4dca", "lidar_only": "#ff9900",
           "unconfirmed_in_fov": "#8c8c8c"}
C_SOURCE = {"both": "#1a9e3f", "lidar": "#1f77b4", "ped3d": "#9b4dca", "filled": "#d0d0d0"}
DPI = 120


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 11,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def static_map_slice(step=3, zmin=-0.05, zmax=1.75):
    """Static map points ~0.2-2 m above the ground (map ground ~ -0.25 m), subsampled."""
    import open3d as o3d
    st = np.asarray(o3d.io.read_point_cloud(str(OP / "static_map_slam_frame.pcd")).points)
    m = (st[:, 2] > zmin) & (st[:, 2] < zmax)
    return st[m][::step]


def robot_path():
    b = np.loadtxt(SO / "poses_base_link_tum.txt")
    return b[:, 0], b[:, 1:3]


def rows(path):
    return list(csv.DictReader(open(path)))


def arr(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def draw_map(ax, pts, path=True, s=0.15):
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c="#c8c8c8", lw=0, rasterized=True)
    if path:
        _, xy = robot_path()
        ax.plot(xy[:, 0], xy[:, 1], "k-", lw=1.8, label="robot path", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def track_colours(n, seed=0):
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [cmap(i % 20) for i in idx]
