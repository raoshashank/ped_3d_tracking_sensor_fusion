#!/usr/bin/env python3
"""Figure 00 (hero): the same moment before and after.
Both panels in the robot's own frame (x forward), so they line up.
Before: what we started with -- camera-only pedestrian positions (ped3d, taken into the robot frame
with ped3d's own odometry pose) over the raw lidar scan; nobody beside or behind the robot.
After: fused 360-degree pedestrian tracks with identities (SLAM frame -> robot frame) over the static
map with pedestrians removed.
Run with the erasor2 env."""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Wedge
from scipy.spatial.transform import Rotation as R
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

BAG = "utown_1_2025-03-07_12-41-18_10_corrected_merged"
SO = Path(p3d_env.P("P3D_BAGS_GO2NUS")) / BAG / "slam_output"
E = SO / "erasor2"
PED3D = Path(p3d_env.P("P3D_PED3D_OUT")) / "go2nus" / f"{BAG}_0.pkl"
FIG = Path(__file__).resolve().parent.parent / "figures"
T_REL = 21.5          # seconds into the recording
TAIL = 2.0            # s of history drawn behind each person
HALF_FOV = 34.9       # deg, camera half angle (camera_model.npz)
XLIM, YLIM = (-9.0, 20.0), (-9.0, 9.0)   # m, robot frame
VIEW = 20.0           # camera wedge length
LABELS = ["camera_verified", "camera_only", "lidar_only", "unconfirmed_in_fov"]
C_LABEL = {"camera_verified": "#1a9e3f", "camera_only": "#9b4dca", "lidar_only": "#ff9900",
           "unconfirmed_in_fov": "#8c8c8c"}
NICE = {"camera_verified": "camera + lidar", "camera_only": "camera only", "lidar_only": "lidar only",
        "unconfirmed_in_fov": "unconfirmed"}
C_PED3D = "#2ca02c"

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.dpi": 120})

# ---------------- timing
tl = np.loadtxt(SO / "poses_rslidar_tum.txt")
t0 = tl[0, 0]
t = t0 + T_REL
scan = int(np.argmin(np.abs(tl[:, 0] - t)))

# ---------------- before: ped3d in the odometry frame + raw scan placed with odometry
d = pickle.load(open(PED3D, "rb"))
ts, rpos, ryaw = d["timestamps"], d["robot_pos"], d["robot_yaw"]
k = int(np.argmin(np.abs(ts - t)))
ox, oy, oyaw = rpos[k, 0], rpos[k, 1], ryaw[k]
s = np.fromfile(E / "data/Merged/velodyne" / f"{scan:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
g = np.fromfile(E / "data/Merged/patchwork" / f"{scan:06d}.label", np.uint32)
s = s[(g == 0) & (np.linalg.norm(s, axis=1) < 25)]
scan_base = s[:, :2] + [0.14543, 0.0]                  # lidar -> base_link (pure translation)
c, sn = np.cos(oyaw), np.sin(oyaw)


def odom_to_base(xy):
    dx, dy = xy[:, 0] - ox, xy[:, 1] - oy
    return np.c_[c * dx + sn * dy, -sn * dx + c * dy]
ped_now, ped_tail = [], []
for tid, tr in d["tracks"].items():
    tt = ts[tr["frames"]]
    ok = np.isfinite(tr["fused"][:, 0])
    if not ok.any() or not (tt[ok][0] <= t <= tt[ok][-1]):
        continue
    xy = tr["fused"][ok, :2]
    tt = tt[ok]
    ped_now.append([np.interp(t, tt, xy[:, 0]), np.interp(t, tt, xy[:, 1])])
    m = (tt >= t - TAIL) & (tt <= t)
    ped_tail.append(odom_to_base(xy[m]) if m.any() else np.zeros((0, 2)))
ped_now = odom_to_base(np.array(ped_now).reshape(-1, 2))

# ---------------- after: fused tracks on the static map (SLAM frame)
b = np.loadtxt(SO / "poses_base_link_tum.txt")
j = int(np.argmin(np.abs(b[:, 0] - t)))
mx, my = b[j, 1], b[j, 2]
myaw = R.from_quat(b[j, 4:8]).as_euler("zyx")[0]
st = np.asarray(o3d.io.read_point_cloud(str(E / "output_ped/static_map_slam_frame.pcd")).points)
h = st[:, 2] + 0.25
st = st[(h > 0.2) & (h < 2.0) & (np.hypot(st[:, 0] - mx, st[:, 1] - my) < 30)]
cm, sm = np.cos(myaw), np.sin(myaw)


def map_to_base(xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    dx, dy = xy[:, 0] - mx, xy[:, 1] - my
    return np.c_[cm * dx + sm * dy, -sm * dx + cm * dy]


st = map_to_base(st[:, :2])
import csv
rows = list(csv.DictReader(open(E / "output_ped/fused_tracks.csv")))
per = {}
for r in rows:
    per.setdefault(int(r["person"]), []).append((float(r["t"]), float(r["x"]), float(r["y"]), r["label"]))
fused_now = []
for pid, v in per.items():
    v = np.array([(a, x, y) for a, x, y, _ in v])
    lab = per[pid][0][3]
    if not (v[0, 0] <= t <= v[-1, 0]):
        continue
    p = [np.interp(t, v[:, 0], v[:, 1]), np.interp(t, v[:, 0], v[:, 2])]
    m = (v[:, 0] >= t - TAIL) & (v[:, 0] <= t)
    fused_now.append((pid, lab, map_to_base(p)[0], map_to_base(v[m, 1:3]) if m.any() else np.zeros((0, 2))))

# ---------------- draw
fig, ax = plt.subplots(1, 2, figsize=(15, 7.4))


def robot(a):
    a.add_patch(Wedge((0, 0), VIEW, -HALF_FOV, HALF_FOV, color="#ffe680", alpha=0.45, zorder=1))
    a.plot([0], [0], marker=(3, 0, -90), markersize=16, color="black", zorder=6)


def inside(p):
    return XLIM[0] < p[0] < XLIM[1] and YLIM[0] < p[1] < YLIM[1]


a = ax[0]
a.scatter(scan_base[:, 0], scan_base[:, 1], s=1.5, c="0.55", zorder=2)
robot(a)
for tail in ped_tail:
    if len(tail) > 1:
        a.plot(tail[:, 0], tail[:, 1], "-", color=C_PED3D, lw=2, alpha=0.6, zorder=3)
a.scatter(ped_now[:, 0], ped_now[:, 1], s=110, c=C_PED3D, edgecolors="white", zorder=5)
n_before = sum(inside(p) for p in ped_now)
a.set_title(f"Before: camera detections + depth ({n_before} people)")
a.text(0.02, 0.02, "positions only inside the camera's view,\nin a drifting odometry frame,"
       "\nno identities beyond short tracks", transform=a.transAxes,
       fontsize=11, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7"))

a = ax[1]
a.scatter(st[:, 0], st[:, 1], s=0.4, c="0.7", zorder=2)
robot(a)
for pid, lab, p, tail in fused_now:
    col = C_LABEL[lab]
    if len(tail) > 1:
        a.plot(tail[:, 0], tail[:, 1], "-", color=col, lw=2, alpha=0.6, zorder=3)
    a.scatter([p[0]], [p[1]], s=110, c=col, edgecolors="white", zorder=5)
    if inside(p):
        a.text(p[0] + 0.3, p[1] + 0.3, f"P{pid}", fontsize=9, color=col, weight="bold", zorder=7)
for lab in LABELS:
    a.scatter([], [], s=80, c=C_LABEL[lab], label=NICE[lab])
a.legend(loc="lower right", fontsize=10, framealpha=0.9)
n_in = sum(inside(p) for _, _, p, _ in fused_now)
a.set_title(f"After: fused 360° tracks with identities ({n_in} people)")
for a in ax:
    a.set_xlim(*XLIM); a.set_ylim(*YLIM)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("0.6")
fig.suptitle(f"The same moment, {T_REL:.1f} s into a walk across campus", fontsize=15, y=0.98)
fig.tight_layout()
fig.savefig(FIG / "00_before_after.png", dpi=120)
print(f"t={T_REL}s scan {scan} ped3d frame {k}: before {n_before} people, after {n_in} people in view")
