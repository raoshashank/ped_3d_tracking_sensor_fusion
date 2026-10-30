#!/usr/bin/env python3
"""Figure 00 (hero), animated: before and after over the whole recording.
Both panels in the SLAM map frame (north-up, never rotated). The view window has a fixed size and
follows a SMOOTHED robot position (Savitzky-Golay over ~2 s of the SLAM base_link xy), leading ahead
along the smoothed heading; robot marker and camera wedge use the smoothed pose, so the gait wobble of
the raw pose does not shake the picture. Every lidar scan (10 Hz), 20 fps = 2x real time.
Before: raw lidar scan (non-ground, placed with the SLAM scan pose) + camera-only pedestrian positions
(ped3d, re-expressed in the SLAM frame per timestamp as in ped_tracks.py: odometry -> robot with
ped3d's pose, robot -> map with the SLAM pose at the same time), only while each track exists; 2 s
tails drawn from the tracks' own samples.
After: static map with pedestrians removed + fused 360-degree tracks with identities, coloured by
label; 2 s tails.
Outputs: figures/00_before_after.mp4, figures/00_before_after.gif, and figures/00_before_after.png
(the frame at t = 21.5 s, used as poster / static fallback).
Run with the erasor2 env:  python 00_before_after_anim.py [--workers 8]"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

# Other recordings (the extra videos before section 9): set BLOG_BAG, BLOG_NAME (output file stem),
# BLOG_POSTER_T (s) and BLOG_PLACE (poster caption) in the environment.
BAG = os.environ.get("BLOG_BAG", "utown_1_2025-03-07_12-41-18_10_corrected_merged")
NAME = os.environ.get("BLOG_NAME", "00_before_after")
PLACE = os.environ.get("BLOG_PLACE", "a walk across campus")
SO = Path(p3d_env.P("P3D_BAGS_GO2NUS")) / BAG / "slam_output"
E = SO / "erasor2"
sys.path.insert(0, str(E))
from ped_tracks import load_tracks  # noqa: E402  (ped3d tracks in the SLAM frame, per timestamp)

FIG = Path(__file__).resolve().parent.parent / "figures"
POSTER_T = float(os.environ.get("BLOG_POSTER_T", 21.5))   # s, frame saved as the static PNG
TAIL = 2.0            # s of history drawn behind each person
HALF_FOV = 34.9       # deg, camera half angle (camera_model.npz)
W, H = 29.0, 18.0     # m, view window
LEAD = 5.5            # m, window centre ahead of the robot along the smoothed heading
SMOOTH = 21           # scans (~2 s) Savitzky-Golay window for the view / robot marker
VIEW = 20.0           # camera wedge length
FPS = 20              # every scan at 20 fps = 2x real time
LABELS = ["camera_verified", "camera_only", "lidar_only", "unconfirmed_in_fov"]
C_LABEL = {"camera_verified": "#1a9e3f", "camera_only": "#9b4dca", "lidar_only": "#ff9900",
           "unconfirmed_in_fov": "#8c8c8c"}
NICE = {"camera_verified": "camera + lidar", "camera_only": "camera only", "lidar_only": "lidar only",
        "unconfirmed_in_fov": "unconfirmed"}
C_PED3D = "#1f5fbf"   # blue: distinct from the green "camera + lidar" of the after panel
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})

# ------------------------------------------------------------------ data (loaded once, shared by fork)
tl = np.loadtxt(SO / "poses_rslidar_tum.txt")
T0 = tl[0, 0]
SCAN_POSE = np.loadtxt(E / "data/Merged/poses.txt").reshape(-1, 3, 4)       # map <- rslidar per scan
B = np.loadtxt(SO / "poses_base_link_tum.txt")
BYAW = np.unwrap(R.from_quat(B[:, 4:8]).as_euler("zyx")[:, 0])
SX = savgol_filter(B[:, 1], SMOOTH, 2)
SY = savgol_filter(B[:, 2], SMOOTH, 2)
SYAW = savgol_filter(BYAW, SMOOTH, 2)
PED = [(t, xy) for t, xy, _ in load_tracks("fused").values()]                 # (t, map xy) per ped3d track
st = np.asarray(o3d.io.read_point_cloud(str(E / "output_ped/static_map_slam_frame.pcd")).points)
h = st[:, 2] + 0.25
st = st[(h > 0.2) & (h < 2.0)]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.c_[st[:, :2], np.zeros(len(st))]))
STATIC = np.asarray(pc.voxel_down_sample(0.12).points)[:, :2]          # 2D, thinned for speed/size
FUSED = {}
for r in csv.DictReader(open(E / "output_ped/fused_tracks.csv")):
    FUSED.setdefault(int(r["person"]), [r["label"], []])[1].append((float(r["t"]), float(r["x"]), float(r["y"])))
FUSED = {pid: (lab, np.array(v)) for pid, (lab, v) in FUSED.items()}


def smooth_pose(t):
    return (np.interp(t, B[:, 0], SX), np.interp(t, B[:, 0], SY), np.interp(t, B[:, 0], SYAW))


def window(t):
    x, y, yaw = smooth_pose(t)
    cx, cy = x + LEAD * np.cos(yaw), y + LEAD * np.sin(yaw)
    return (cx - W / 2, cx + W / 2), (cy - H / 2, cy + H / 2)


def track_at(tt, xy, t):
    """Position at t and the tail: the track's own samples in [t - TAIL, t] ending at the position."""
    p = np.array([np.interp(t, tt, xy[:, 0]), np.interp(t, tt, xy[:, 1])])
    m = (tt >= t - TAIL) & (tt < t)
    return p, (np.vstack([xy[m], p]) if m.any() else p[None])


def frame_data(scan):
    t = tl[scan, 0]
    xl, yl = window(t)
    s = np.fromfile(E / "data/Merged/velodyne" / f"{scan:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    g = np.fromfile(E / "data/Merged/patchwork" / f"{scan:06d}.label", np.uint32)
    s = s[(g == 0) & (np.linalg.norm(s, axis=1) < 30)].astype(np.float64)
    P = SCAN_POSE[scan]
    scan_map = (s @ P[:, :3].T + P[:, 3])[:, :2]
    before = [track_at(tt, xy, t) for tt, xy in PED if tt[0] <= t <= tt[-1]]
    m = (STATIC[:, 0] > xl[0] - 1) & (STATIC[:, 0] < xl[1] + 1) & (STATIC[:, 1] > yl[0] - 1) & (STATIC[:, 1] < yl[1] + 1)
    after = []
    for pid, (lab, v) in FUSED.items():
        if v[0, 0] - 0.05 <= t <= v[-1, 0] + 0.05:
            p, tail = track_at(v[:, 0], v[:, 1:3], t)
            after.append((pid, lab, p, tail))
    return t, (xl, yl), scan_map, before, STATIC[m], after


def inside(p, lims, margin=0.0):
    (xl, yl) = lims
    return xl[0] + margin < p[0] < xl[1] - margin and yl[0] + margin < p[1] < yl[1] - margin


def robot(a, t):
    x, y, yaw = smooth_pose(t)
    a.add_patch(Wedge((x, y), VIEW, np.degrees(yaw) - HALF_FOV, np.degrees(yaw) + HALF_FOV,
                      color="#ffe680", alpha=0.45, zorder=1))
    a.plot([x], [y], marker=(3, 0, np.degrees(yaw) - 90), markersize=16, color="black", zorder=6)


def render(scan, path, dpi, suptitle=None):
    t, lims, scan_map, before, static, after = frame_data(scan)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.1))
    a = ax[0]
    a.scatter(scan_map[:, 0], scan_map[:, 1], s=1.5, c="0.55", zorder=2, rasterized=True)
    robot(a, t)
    for p, tail in before:
        if len(tail) > 1:
            a.plot(tail[:, 0], tail[:, 1], "-", color=C_PED3D, lw=2, alpha=0.6, zorder=3)
    if before:
        Pp = np.array([p for p, _ in before])
        a.scatter(Pp[:, 0], Pp[:, 1], s=110, c=C_PED3D, edgecolors="white", zorder=5)
    n_before = sum(inside(p, lims) for p, _ in before)
    a.set_title(f"Before: camera detections + depth ({n_before} people)")
    a.text(0.02, 0.02, "positions only inside the camera's view,\ncomputed in a drifting odometry frame,"
           "\nno identities beyond short tracks", transform=a.transAxes,
           fontsize=11, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="0.7"), zorder=8)
    a = ax[1]
    a.scatter(static[:, 0], static[:, 1], s=0.35, c="0.75", zorder=2, rasterized=True)
    robot(a, t)
    x, y, _ = smooth_pose(t)
    labelled = []
    for pid, lab, p, tail in sorted(after, key=lambda z: np.hypot(z[2][0] - x, z[2][1] - y)):   # nearest first
        col = C_LABEL[lab]
        if len(tail) > 1:
            a.plot(tail[:, 0], tail[:, 1], "-", color=col, lw=2, alpha=0.6, zorder=3)
        a.scatter([p[0]], [p[1]], s=110, c=col, edgecolors="white", zorder=5)
        if inside(p, lims, 0.8) and all(np.hypot(*(p - q)) > 0.9 for q in labelled):
            a.text(p[0] + 0.3, p[1] + 0.3, f"P{pid}", fontsize=9, color=col, weight="bold", zorder=7)
            labelled.append(p)
    n_in = sum(inside(p, lims) for _, _, p, _ in after)
    a.set_title(f"After: fused 360° tracks with identities ({n_in} people)")
    for a in ax:
        a.set_xlim(*lims[0]); a.set_ylim(*lims[1])
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("0.6")
    handles = [Line2D([], [], marker="o", ls="", ms=10, mfc=C_PED3D, mec="white", label="camera detection (before)")]
    handles += [Line2D([], [], marker="o", ls="", ms=10, mfc=C_LABEL[l], mec="white", label=NICE[l]) for l in LABELS]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.0))
    rel = t - T0
    fig.suptitle(suptitle or f"t = {rel:4.1f} s  (2x speed)", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.1, wspace=0.03)   # fixed layout: no jumps
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return rel, before, after


def job(args):
    i, scan, out = args
    render(scan, Path(out) / f"f{i:04d}.png", dpi=80)
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()
    ps = int(np.argmin(np.abs(tl[:, 0] - (T0 + POSTER_T))))
    render(ps, FIG / f"{NAME}.png", dpi=120,
           suptitle=f"The same moment, {POSTER_T:.1f} s into {PLACE}")
    scans = list(range(len(tl)))
    base = os.environ.get("P3D_SCRATCH")
    tmp = Path(tempfile.mkdtemp(dir=(base + "/tmp") if base else None))
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(job, [(i, s, str(tmp)) for i, s in enumerate(scans)]), 1):
            if n % 100 == 0:
                print(f"rendered {n}/{len(scans)}", flush=True)
    mp4, gif = FIG / f"{NAME}.mp4", FIG / f"{NAME}.gif"
    subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart", str(mp4)], check=True)
    for fps, width, colors in ((20, 640, 48), (20, 560, 48), (10, 800, 48)):   # smoothness first
        subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
                        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
                               f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
                               "[b][p]paletteuse=dither=none:diff_mode=rectangle",
                        str(gif)], check=True)
        if gif.stat().st_size < 15.5e6:
            break
    print(f"frames {len(scans)} at {FPS} fps; mp4 {mp4.stat().st_size / 1e6:.1f} MB, gif {gif.stat().st_size / 1e6:.1f} MB "
          f"(gif fps {fps}, {width} px, {colors} colours); frames in {tmp}")


if __name__ == "__main__":
    main()
