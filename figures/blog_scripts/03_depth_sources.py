#!/usr/bin/env python3
"""Figure 03: VGGT-Omega depth map + per-person positions from each depth source (top-down, base_link)."""
import matplotlib.patches as patches
import numpy as np

from _camera_common import C, FIG, FRAME, SRC_LABEL, draw_fov, fov_half_angle_and_yaw, lidar_in_base, ped3d, plt, traj

T = traj()
d = ped3d()
k = FRAME
depth = T.depth(k)
boxes, ids = T.boxes(k)
pb = lidar_in_base(T, k)
pb = pb[(np.linalg.norm(pb[:, :2], axis=1) < 20) & (pb[:, 2] > -0.25) & (pb[:, 2] < 1.8)]
half, cyaw, cxy = fov_half_angle_and_yaw(T)

fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = axs[0]
sy, sx = depth.shape[0] / T.H, depth.shape[1] / T.W
im = a.imshow(np.where(depth > 0, depth, np.nan), cmap="turbo", vmin=0, vmax=25)
for (x, y, w, h), tid in zip(boxes, ids):
    a.add_patch(patches.Rectangle((x * sx, y * sy), w * sx, h * sy, fill=False, ec="white", lw=1.2))
    a.text(x * sx, y * sy - 3, str(tid), color="black", fontsize=8, bbox=dict(fc="white", ec="none", pad=0.8))
a.set_xticks([]); a.set_yticks([])
a.set_title("(a) VGGT-Omega metric depth (camera only)")
fig.colorbar(im, ax=a, fraction=0.03, pad=0.01).set_label("depth [m]")

a = axs[1]
draw_fov(a, cxy, cyaw, half, 16, color="gold", alpha=0.25, zorder=0)
a.scatter(pb[:, 0], pb[:, 1], s=1.5, c="0.75", zorder=1, label="lidar scan (this frame)")
srcs = ["lidar", "lidar_g", "vggt", "vggt_c", "vggt_a", "da"]
style = dict(lidar=("o", C["lidar"], "none"), lidar_g=("o", C["lidar"], C["lidar"]), vggt=("^", C["vggt"], C["vggt"]),
             vggt_c=("s", C["vggt_c"], C["vggt_c"]), vggt_a=("D", C["vggt_a"], C["vggt_a"]), da=("v", C["da"], C["da"]))
shown = set()
for det in d["frames"][k]:
    tr = d["tracks"].get(det["id"])
    anchor = float(tr.get("anchor", 1.0)) if tr else 1.0
    P = {s: np.asarray(det[f"pos_{s}"], float)[:2] for s in ("lidar", "lidar_g", "vggt", "vggt_c", "da")}
    P["vggt_a"] = P["vggt_c"] * anchor
    ok = [s for s in srcs if np.isfinite(P[s]).all() and np.linalg.norm(P[s]) < 18]
    if not ok or all(np.linalg.norm(P[s]) > 15 for s in ok):
        continue
    xy = np.array([P[s] for s in ok])
    a.plot(xy[:, 0], xy[:, 1], "-", color="0.35", lw=0.8, zorder=2)
    for s in ok:
        mk, ec, fc = style[s]
        a.scatter(*P[s], marker=mk, s=45, edgecolors=ec, facecolors=fc, linewidths=1.5, zorder=3,
                  label=None if s in shown else SRC_LABEL[s])
        shown.add(s)
    ref = P["lidar_g"] if np.isfinite(P["lidar_g"]).all() else xy[0]
    a.text(ref[0] + 0.3, ref[1] + 0.3, str(det["id"]), fontsize=9, zorder=4)
a.plot(0, 0, "k>", ms=11, zorder=5)
a.set_aspect("equal")
a.set_xlim(-1, 16); a.set_ylim(-8, 8)
a.set_xlabel("x forward [m] (robot frame)"); a.set_ylabel("y left [m]")
a.set_title("(b) each person's position from each depth source")
a.legend(loc="lower left", fontsize=9, markerscale=1.1, framealpha=0.9)
fig.tight_layout()
fig.savefig(FIG / "03_depth_sources.png", bbox_inches="tight")
print("saved; dets", len(d["frames"][k]))
