#!/usr/bin/env python3
"""Paper figure: the reconstruction pipeline on one go2nus trajectory (utown_1 ... _10), three panels.

  (a) camera frame: 2D tracks (provided) + projected LiDAR -> 3D position per person (fused range shown)
  (b) static map: accumulated SLAM-registered scans, points removed as dynamic (ERASOR2 + pedestrian
      masking) in red, robot path
  (c) same instant, top-down: fused 360-deg pedestrian tracks (2 s trails) over the static map,
      coloured by evidence, camera field of view; persons numbered as in (a)

Run with the calib env:  python fig_pipeline.py   -> fig_pipeline.pdf / .png next to this script.
Reuses the blog helpers read-only (privacy blur cache, ped3d trajectory loader).
"""
import csv
import sys
from pathlib import Path

import numpy as np

BLOG = Path(__file__).resolve().parent.parent / "blog_scripts"   # figures/blog_scripts of this repo
sys.path.insert(0, str(BLOG))
from _blur import blur  # noqa: E402
from _camera_common import FRAME, ped3d, project_lidar, traj, fov_half_angle_and_yaw  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import open3d as o3d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

OUT = Path(__file__).resolve().parent
BAG = Path(p3d_env.P("P3D_BAGS_GO2NUS")) / "utown_1_2025-03-07_12-41-18_10_corrected_merged"
SO, E = BAG / "slam_output", BAG / "slam_output" / "erasor2"
OP = E / "output_ped"
GROUND_Z = -0.25            # map z of the ground for this bag (blog_common)
TRAIL_S = 2.0
X0, X1, Y0, Y1 = -2, 30, -12, 8
ZX0, ZX1, ZY0, ZY1 = -2, 18.5, -7, 6     # (c): around the robot at the camera frame
C_LABEL = {"camera_verified": "#1a9e3f", "camera_only": "#9b4dca", "lidar_only": "#ff9900",
           "unconfirmed_in_fov": "#8c8c8c"}
NAME = {"camera_verified": "camera + LiDAR", "camera_only": "camera only", "lidar_only": "LiDAR only",
        "unconfirmed_in_fov": "unconfirmed"}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "font.family": "serif",
                     "savefig.facecolor": "white", "pdf.fonttype": 42})


def pcd(p):
    return np.asarray(o3d.io.read_point_cloud(str(p)).points)


def crop(p, lo=0.35, hi=2.0):
    h = p[:, 2] - GROUND_Z
    m = (h > lo) & (h < hi) & (p[:, 0] > X0) & (p[:, 0] < X1) & (p[:, 1] > Y0) & (p[:, 1] < Y1)
    return p[m]


# ---------------------------------------------------------------- data
T = traj()
k = FRAME
d = ped3d()
t_cam = float(d["timestamps"][k])
img = blur(np.ascontiguousarray(T.image(k)[:, :, ::-1]), f"proc_{k}")
boxes, ids = T.boxes(k)
u, v, rng = project_lidar(T, k)

F = list(csv.DictReader(open(OP / "fused_tracks.csv")))
M = [r for r in csv.DictReader(open(OP / "fused_track_members.csv")) if r["kind"] == "ped3d"]
persons = {}
for r in F:
    persons.setdefault(int(r["person"]), {"label": r["label"], "rows": []})["rows"].append(
        (float(r["t"]), float(r["x"]), float(r["y"]), float(r["range"])))
for p in persons.values():
    p["rows"] = np.array(sorted(p["rows"]))


def person_of(bid, t):
    for r in M:
        if int(r["member_id"]) % 100000 == bid and float(r["t_start"]) - 0.15 <= t <= float(r["t_end"]) + 0.15:
            return int(r["person"])
    return None


def at(p, t, tol=0.1):
    a = persons[p]["rows"]
    i = np.argmin(np.abs(a[:, 0] - t))
    return a[i] if abs(a[i, 0] - t) <= tol else None


# number the persons visible in the camera frame left to right; same numbers in (c)
vis = []
for (x, y, w, h), bid in sorted(zip(boxes, ids), key=lambda b: b[0][0]):
    p = person_of(int(bid), t_cam)
    vis.append(((x, y, w, h), p))
num = {p: i + 1 for i, (_, p) in enumerate([v_ for v_ in vis if v_[1] is not None])}

# robot pose at t_cam (SLAM base_link)
B = np.loadtxt(SO / "poses_base_link_tum.txt")
i_r = np.argmin(np.abs(B[:, 0] - t_cam))
rxy = B[i_r, 1:3]
ryaw = R.from_quat(B[i_r, 4:8]).as_euler("zyx")[0]

# maps: raw accumulation of the scans ERASOR2 used vs the final static map
D = E / "data" / "Merged"
P3 = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
acc = []
for i in range(0, len(P3) - 1, 2):
    sc = np.fromfile(D / "velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    acc.append(sc @ P3[i, :, :3].T + P3[i, :, 3])
raw = crop(np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(acc))).voxel_down_sample(0.05).points))
static = pcd(OP / "static_map_slam_frame.pcd")
removed = ~np.isfinite(cKDTree(static).query(raw, distance_upper_bound=0.1)[0])
static_c = crop(static)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.16, 2.0))
gs = fig.add_gridspec(1, 3, width_ratios=[T.W / T.H, (X1 - X0) / (Y1 - Y0) * (T.H / T.H), (ZX1 - ZX0) / (ZY1 - ZY0)],
                      wspace=0.2)
ax_a, ax_b, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# (a)
ax_a.imshow(img)
sc_a = ax_a.scatter(u, v, c=rng, s=0.35, cmap="turbo", vmin=0, vmax=25, lw=0, alpha=0.7, rasterized=True)
for (x, y, w, h), p in vis:
    if p is None:
        ax_a.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec="white", lw=0.6, ls="--"))
        continue
    c = C_LABEL[persons[p]["label"]]
    ax_a.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec=c, lw=1.0))
    ax_a.text(x + 1, y - 2, str(num[p]), color="white", fontsize=5, va="bottom", fontweight="bold",
              bbox=dict(fc=c, ec="none", pad=0.5))
ax_a.set_xlim(0, T.W); ax_a.set_ylim(T.H, 0)
ax_a.set_xticks([]); ax_a.set_yticks([])
ax_a.set_title("(a) 2D tracks + projected LiDAR")
# LiDAR points coloured by range (turbo, 0-25 m): explained in the caption

# (b)
ax_b.scatter(raw[~removed, 0], raw[~removed, 1], s=0.05, c="0.55", lw=0, rasterized=True)
ax_b.scatter(raw[removed, 0], raw[removed, 1], s=0.05, c="#d62728", lw=0, rasterized=True)
ax_b.plot(B[:, 1], B[:, 2], "k-", lw=1.0)
ax_b.scatter([], [], s=6, c="0.55", label="static map")
ax_b.scatter([], [], s=6, c="#d62728", label="removed (dynamic)")
ax_b.plot([], [], "k-", lw=1.0, label="robot path (SLAM)")
ax_b.set_title("(b) static map (SLAM + ERASOR2)")

# (c)
ax_c.scatter(static_c[::2, 0], static_c[::2, 1], s=0.05, c="0.75", lw=0, rasterized=True)
ax_c.plot(B[: i_r + 1, 1], B[: i_r + 1, 2], "k-", lw=0.8, alpha=0.5)
half, cyaw, _ = fov_half_angle_and_yaw(T)
a0, a1 = ryaw + cyaw - half, ryaw + cyaw + half
L = 14.0
ax_c.fill([rxy[0], rxy[0] + L * np.cos(a0), rxy[0] + L * np.cos(a1)],
          [rxy[1], rxy[1] + L * np.sin(a0), rxy[1] + L * np.sin(a1)], color="#ffd84d", alpha=0.3, lw=0, zorder=1)
ax_c.add_patch(patches.RegularPolygon(rxy, 3, radius=0.6, orientation=ryaw - np.pi / 2, color="k", zorder=6))
for p, P in persons.items():
    a = P["rows"]
    m = (a[:, 0] <= t_cam + 0.05) & (a[:, 0] >= t_cam - TRAIL_S)
    if not m.any() or abs(a[m][-1, 0] - t_cam) > 0.1:
        continue
    c = C_LABEL[P["label"]]
    ax_c.plot(a[m, 1], a[m, 2], "-", color=c, lw=0.9, zorder=4)
    ax_c.plot(a[m][-1, 1], a[m][-1, 2], "o", color=c, ms=2.6, zorder=5)
    if p in num:
        ax_c.text(a[m][-1, 1] + 0.4, a[m][-1, 2] + 0.3, str(num[p]), fontsize=5, color=c, zorder=7, fontweight="bold")
for lab in ("camera_verified", "camera_only", "lidar_only", "unconfirmed_in_fov"):
    ax_c.plot([], [], "-o", color=C_LABEL[lab], lw=0.9, ms=2.6, label=NAME[lab])
ax_c.fill([], [], color="#ffd84d", alpha=0.3, label="camera FOV")
h, l = [], []
for ax in (ax_b, ax_c):
    hh, ll = ax.get_legend_handles_labels()
    h, l = h + hh, l + ll
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=len(l), frameon=False, handletextpad=0.3,
           columnspacing=1.0)
ax_c.set_title("(c) fused 360$^\\circ$ tracks (2 s trails)")
ax_b.add_patch(patches.Rectangle((ZX0, ZY0), ZX1 - ZX0, ZY1 - ZY0, fill=False, ec="k", lw=0.5, ls="--"))

for ax, (x0, x1, y0, y1) in ((ax_b, (X0, X1, Y0, Y1)), (ax_c, (ZX0, ZX1, ZY0, ZY1))):
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("x [m]", labelpad=1)
    ax.tick_params(length=2, pad=1)
ax_b.set_ylabel("y [m]", labelpad=0)

fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight", dpi=400)
fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight", dpi=300)
print("t_cam", t_cam, "persons in view", len(num), "removed fraction", removed.mean().round(3))
