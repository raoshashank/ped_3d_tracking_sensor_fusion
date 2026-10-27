"""10: one lidar scan against the static map: the residual (what is not in the map) and the
person-sized detections, inside (blue) and outside (red) the camera field of view."""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from blog_common import *  # noqa

sys.path.insert(0, str(E))
import lidar_ped_detect as LD  # noqa

SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 215
D = E / "data" / "Merged"
P3 = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
t_scan, T_base = LD.base_poses()
cam = np.load(E / "camera_model.npz"); cam = {k: cam[k] for k in cam.files}
static = read_pcd(E / "output_ped" / "static_map_slam_frame.pcd")
st_sl, st_h = height_slice(static, 0.35, 2.0)

s = np.fromfile(D / "velodyne" / f"{SCAN:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
g = np.fromfile(D / "patchwork" / f"{SCAN:06d}.label", np.uint32)
P = P3[SCAN]
w = s @ P[:, :3].T + P[:, 3]
rng = np.linalg.norm(s, axis=1)
tree = cKDTree(static)
res = (g == 0) & (rng < LD.MAX_RANGE) & np.isinf(tree.query(w, distance_upper_bound=LD.R_RES)[0])
dets = np.load(E / "output_ped" / "lidar_dets_raw.npz")["dets"]
dd = dets[dets[:, 0] == SCAN]
rob = T_base[SCAN][:2, 3]

# camera FOV wedge: in_fov test on a ring of points 12 m around the robot, 1 m above ground
ang = np.linspace(-np.pi, np.pi, 1441)
ring = rob + 12 * np.c_[np.cos(ang), np.sin(ang)]
inside = LD.in_fov(ring, np.full(len(ring), P[2, 3] - 0.48 + 1.0), T_base[SCAN], cam)
a_in = ang[inside]
a0, a1 = a_in.min(), a_in.max()
if a1 - a0 > np.pi:     # wraps around +-pi
    a_in = np.where(a_in < 0, a_in + 2 * np.pi, a_in); a0, a1 = a_in.min(), a_in.max()
wedge = np.vstack([rob, rob + 12 * np.c_[np.cos(np.linspace(a0, a1, 60)), np.sin(np.linspace(a0, a1, 60))], rob])

R_VIEW = 14
fig, axs = plt.subplots(1, 2, figsize=(15, 7.0), sharex=True, sharey=True)
for ax in axs:
    m = np.linalg.norm(st_sl[:, :2] - rob, axis=1) < R_VIEW + 3
    ax.scatter(st_sl[m, 0], st_sl[m, 1], s=0.08, c=MAP_GREY, rasterized=True)
    ax.plot(*T_base[:, :2, 3].T, "k-", lw=1, alpha=0.5)
    ax.fill(wedge[:, 0], wedge[:, 1], color="#ffd54f", alpha=0.25, lw=0)
    ax.plot(*rob, marker=(3, 0, np.degrees(np.arctan2(T_base[SCAN][1, 0], T_base[SCAN][0, 0])) - 90), ms=14, color="k")
    ax.set_aspect("equal"); ax.set_xlim(rob[0] - R_VIEW, rob[0] + R_VIEW); ax.set_ylim(rob[1] - R_VIEW * 0.75, rob[1] + R_VIEW * 0.75)
    ax.set_xlabel("x [m]")
axs[0].set_ylabel("y [m]")
ax = axs[0]
ng = (g == 0) & (rng < LD.MAX_RANGE)
ax.scatter(w[ng, 0], w[ng, 1], s=0.6, c="#444444", rasterized=True)
ax.plot([], [], "o", color="#444444", ms=4, label="lidar scan (non-ground)")
ax.set_title(f"(a) one scan (t = {t_scan[SCAN] - t_scan[0]:.1f} s) over the static map")
ax.legend(loc="lower left")
ax = axs[1]
ax.scatter(w[res, 0], w[res, 1], s=2, c="#ff7f0e", rasterized=True)
ax.plot([], [], "o", color="#ff7f0e", ms=4, label="residual: not in the static map")
for x, y, fv in dd[:, [2, 3, 8]]:
    ax.add_patch(plt.Circle((x, y), 0.45, fill=False, lw=2.2, color=C_IN if fv > 0 else C_OUT))
ax.plot([], [], "o", mfc="none", mec=C_IN, mew=2, ms=7, label="person-sized detection, camera sees it")
ax.plot([], [], "o", mfc="none", mec=C_OUT, mew=2, ms=7, label="person-sized detection, outside camera view")
ax.fill([], [], color="#ffd54f", alpha=0.4, label="camera field of view")
ax.set_title("(b) residual + person-sized clusters")
ax.legend(loc="lower left", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG / "10_residual.png")
print(f"scan {SCAN}: {ng.sum()} non-ground points, residual {res.sum()} ({res.sum() / max(ng.sum(), 1):.1%}); "
      f"detections {len(dd)} (in FOV {int((dd[:, 8] > 0).sum())}, outside {int((dd[:, 8] == 0).sum())}); FOV width {np.degrees(a1 - a0):.0f} deg")
