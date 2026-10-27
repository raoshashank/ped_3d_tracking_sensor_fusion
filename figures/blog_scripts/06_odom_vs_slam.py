"""06: Go2 onboard odometry vs offline SLAM, and why ped3d positions are re-projected per timestamp."""
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from blog_common import *  # noqa

sys.path.insert(0, str(E))
from ped_tracks import PED3D, load_tracks  # noqa

ts, ps, qs = slam_poses()
od = np.loadtxt(S / "robot_odom_tum.txt")          # odom re-expressed in the map frame, aligned at the start
raw = read_pcd(S / "map.pcd")
sl, h = height_slice(raw)

# drift: odom vs SLAM at SLAM times
ox = np.interp(ts, od[:, 0], od[:, 1]); oy = np.interp(ts, od[:, 0], od[:, 2])
drift = np.hypot(ox - ps[:, 0], oy - ps[:, 1])
path_len = np.sum(np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1))

# one ped3d track: naive (single rigid 2D fit of odom->SLAM trajectory) vs per-timestamp re-projection
d = pickle.load(open(PED3D, "rb"))
tp, rpos = d["timestamps"], d["robot_pos"]
sx = np.interp(tp, ts, ps[:, 0]); sy = np.interp(tp, ts, ps[:, 1])
P, Q = rpos, np.c_[sx, sy]
pc, qc = P.mean(0), Q.mean(0)
U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
Rm = Vt.T @ U.T
if np.linalg.det(Rm) < 0:
    Vt[1] *= -1; Rm = Vt.T @ U.T
rigid = lambda xy: (xy - pc) @ Rm.T + qc
fit_res = np.linalg.norm(rigid(P) - Q, axis=1)

per_ts = load_tracks("fused")   # per-timestamp re-projection (pipeline)
# evidence: distance from each ped3d detection (<= 12 m) to the nearest lidar detection of the same scan
dets = np.load(E / "output_ped" / "lidar_dets_raw.npz")["dets"]    # scan, t, x, y, ...
t_scan = ts
err_naive, err_ts = [], []
for pid, tr in d["tracks"].items():
    if pid not in per_ts:
        continue
    t_tr, xy_ts, _ = per_ts[pid]
    ok = np.isfinite(tr["fused"][:, :2]).all(1)
    det = tr["detected"][ok] & ~tr["interpolated"][ok]
    xy_n = rigid(tr["fused"][ok, :2])
    for t, a, b, dd in zip(t_tr, xy_n, xy_ts, det):
        if not dd:
            continue
        i = int(np.argmin(np.abs(t_scan - t)))
        if abs(t_scan[i] - t) > 0.06 or np.hypot(*(b - ps[i, :2])) > 12:
            continue
        m = dets[:, 0] == i
        if not m.any():
            continue
        err_naive.append(np.min(np.linalg.norm(dets[m, 2:4] - a, axis=1)))
        err_ts.append(np.min(np.linalg.norm(dets[m, 2:4] - b, axis=1)))
err_naive, err_ts = np.array(err_naive), np.array(err_ts)

fig = plt.figure(figsize=(14, 5.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 1.1])
ax = fig.add_subplot(gs[0])
ax.scatter(sl[::6, 0], sl[::6, 1], s=0.05, c=MAP_GREY, rasterized=True)
ax.plot(ps[:, 0], ps[:, 1], "k-", lw=2.2, label="offline SLAM (lidar)")
ax.plot(od[:, 1], od[:, 2], "--", color="#ff7f0e", lw=2.0, label="Go2 onboard odometry")
ax.plot(*ps[0, :2], "ko", ms=6)
ax.annotate(f"end: {drift[-1]:.2f} m apart", xy=(ps[-1, 0], ps[-1, 1]), xytext=(ps[-1, 0] - 14, ps[-1, 1] - 6),
            arrowprops=dict(arrowstyle="->"), fontsize=11)
ax.set_aspect("equal"); ax.set_xlim(-6, 32); ax.set_ylim(-14, 8)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="upper right", fontsize=10)
ax.set_title("(a) robot trajectory, 44 s")

ax = fig.add_subplot(gs[1])
ax.plot(ts - ts[0], drift, "-", color="#ff7f0e", lw=2)
ax.set_xlabel("time [s]"); ax.set_ylabel("odometry - SLAM position [m]")
ax.set_title(f"(b) odometry vs SLAM ({path_len:.0f} m walked)"); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2])
bins = np.linspace(0, 2.0, 41)
ax.hist(np.clip(err_naive, 0, 2), bins, color="#9467bd", alpha=0.6, label=f"one rigid fit (median {np.median(err_naive):.2f} m)")
ax.hist(np.clip(err_ts, 0, 2), bins, color=C_PED3D, alpha=0.6, label=f"per timestamp (median {np.median(err_ts):.2f} m)")
ax.set_xlabel("ped3d to nearest lidar det. [m]\n(last bin: 2 m or more)"); ax.set_ylabel("ped3d detections")
ax.legend(fontsize=10); ax.set_title("(c) placing ped3d in the SLAM map")
fig.tight_layout()
fig.savefig(FIG / "06_odom_vs_slam.png")
print(f"drift final {drift[-1]:.2f} m, max {drift.max():.2f} m, path {path_len:.1f} m; rigid fit residual median "
      f"{np.median(fit_res):.2f} max {fit_res.max():.2f}; ped3d-to-lidar n={len(err_ts)} naive median {np.median(err_naive):.2f} "
      f"per-ts median {np.median(err_ts):.2f}; frac<0.3 naive {np.mean(err_naive < 0.3):.2f} per-ts {np.mean(err_ts < 0.3):.2f}")
