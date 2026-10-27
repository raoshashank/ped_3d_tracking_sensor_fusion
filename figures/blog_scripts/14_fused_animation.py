#!/usr/bin/env python3
"""Figure 14: animated top-down view of the fused persons (demo bag), following the robot.
Static map (grey), lidar points that are not in the static map (the residual, black dots), robot +
camera view wedge (camera_model.npz, SLAM base_link pose), fused persons coloured by label with
P<id> and 1.5 s tails. Every 2nd lidar scan (0.2 s) at 10 fps = 2x real time."""
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from tf_common import C_LABEL, E, FIG, LABELS, OP, SO, rows, static_map_slice, style

plt = style()
STEP, FPS, TAIL, WEDGE_R = 2, 10, 1.5, 12.0
GROUND_Z = -0.25     # map z of the ground

# static map (background) and a KD-tree over the full map for the residual
import open3d as o3d
full = np.asarray(o3d.io.read_point_cloud(str(OP / "static_map_slam_frame.pcd")).points)
tree = cKDTree(full)
bg = static_map_slice(step=4)

b = np.loadtxt(SO / "poses_base_link_tum.txt")
t_scan, base_xy = b[:, 0], b[:, 1:3]
base_R = R.from_quat(b[:, 4:8]).as_matrix()
lid_poses = np.loadtxt(E / "data/Merged/poses.txt").reshape(-1, 3, 4)
cam = np.load(E / "camera_model.npz")
bTc = np.linalg.inv(cam["cam_T_base"])
axis_b = bTc[:3, 2]                                    # camera optical axis in base_link
half_fov = np.arctan(cam["W"] / 2 / cam["K"][0, 0])
t0 = t_scan[0]

F = rows(OP / "fused_tracks.csv")
fused = {}
for r in F:
    fused.setdefault(int(r["person"]), {"label": r["label"], "t": [], "xy": []})
    fused[int(r["person"])]["t"].append(float(r["t"]))
    fused[int(r["person"])]["xy"].append((float(r["x"]), float(r["y"])))
for p in fused.values():
    o = np.argsort(p["t"])
    p["t"], p["xy"] = np.array(p["t"])[o], np.array(p["xy"])[o]


def residual(i):
    s = np.fromfile(E / f"data/Merged/velodyne/{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    g = np.fromfile(E / f"data/Merged/patchwork/{i:06d}.label", np.uint32)
    s = s[(g == 0) & (np.linalg.norm(s, axis=1) < 20)]
    w = s @ lid_poses[i, :, :3].T + lid_poses[i, :, 3]
    w = w[(w[:, 2] - GROUND_Z > 0.2) & (w[:, 2] - GROUND_Z < 2.2)]
    return w[np.isinf(tree.query(w, distance_upper_bound=0.15, workers=4)[0])]


frames = list(range(1, len(t_scan), STEP))
fig, ax = plt.subplots(figsize=(8.4, 5.6))
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.93)


def draw(k):
    i = frames[k]
    t = t_scan[i]
    ax.clear()
    rx, ry = base_xy[i]
    ax.scatter(bg[:, 0], bg[:, 1], s=0.3, c="#c8c8c8", lw=0, rasterized=True)
    res = residual(i)
    ax.scatter(res[:, 0], res[:, 1], s=1.5, c="k", lw=0, alpha=0.7, zorder=3)
    a = base_R[i] @ axis_b
    yaw = np.arctan2(a[1], a[0])
    ang = np.linspace(yaw - half_fov, yaw + half_fov, 20)
    wedge = np.c_[np.r_[rx, rx + WEDGE_R * np.cos(ang)], np.r_[ry, ry + WEDGE_R * np.sin(ang)]]
    ax.add_patch(Polygon(wedge, closed=True, fc="#fff0a8", ec="#e0c040", alpha=0.45, zorder=1))
    ax.plot(base_xy[: i + 1, 0], base_xy[: i + 1, 1], "k-", lw=1.2, zorder=3)
    hy = np.arctan2(base_R[i][1, 0], base_R[i][0, 0])
    ax.plot(rx, ry, marker=(3, 0, np.degrees(hy) - 90), ms=13, c="k", zorder=6)
    for pid, p in fused.items():
        if not (p["t"][0] - 0.05 <= t <= p["t"][-1] + 0.05):
            continue
        j = int(np.argmin(np.abs(p["t"] - t)))
        c = C_LABEL[p["label"]]
        tail = (p["t"] >= t - TAIL) & (p["t"] <= t)
        ax.plot(p["xy"][tail, 0], p["xy"][tail, 1], "-", color=c, lw=2, alpha=0.8, zorder=4)
        ax.add_patch(plt.Circle(p["xy"][j], 0.3, color=c, zorder=5))
        ax.text(p["xy"][j, 0] + 0.35, p["xy"][j, 1] + 0.35, f"P{pid}", color=c, fontsize=9, weight="bold", zorder=7)
    ax.set_xlim(rx - 12, rx + 22)
    ax.set_ylim(ry - 11, ry + 11)
    ax.set_aspect("equal")
    ax.set_title(f"fused persons, utown_1   t = {t - t0:4.1f} s   (2x speed)", loc="left", fontsize=12)
    ax.tick_params(labelsize=9)
    for lab in LABELS:
        ax.plot([], [], "o", color=C_LABEL[lab], label=lab.replace("_", " "))
    ax.plot([], [], "s", color="#fff0a8", label="camera view")
    ax.plot([], [], ".", color="k", label="lidar points not in the map")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, ncol=2)
    return []


draw(0)
fig.savefig(FIG / "14_fused_animation_first.png", dpi=120)
import sys
if "--preview" in sys.argv:
    for k in (40, 90, 150):
        draw(k)
        fig.savefig(f"/tmp/claude_preview_14_{k}.png", dpi=90)
    sys.exit()
anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
anim.save(FIG / "14_fused_animation.gif", writer=PillowWriter(fps=FPS), dpi=80)
import os
print("frames", len(frames), "size MB", os.path.getsize(FIG / "14_fused_animation.gif") / 1e6)
