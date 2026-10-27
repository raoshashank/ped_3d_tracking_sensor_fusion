"""08: raw map vs ERASOR2 alone vs ERASOR2 with pedestrian masking, and what each removed."""
import glob
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from blog_common import *  # noqa

LIFT = 0.38    # ERASOR2 saves its map lifted by the sensor height
ts, ps, _ = slam_poses()
# raw accumulation of the scans ERASOR2 saw (every 2nd scan, accum_interval 2), 5 cm voxels
import open3d as o3d
D = E / "data" / "Merged"
P3 = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
acc = []
for i in range(0, len(P3) - 1, 2):
    sc = np.fromfile(D / "velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    acc.append(sc @ P3[i, :, :3].T + P3[i, :, 3])
raw = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(acc))).voxel_down_sample(0.05).points)
e2 = read_pcd(E / "output" / "static_map_slam_frame.pcd")
fin = read_pcd(E / "output_ped" / "static_map_slam_frame.pcd")
bug = read_pcd(glob.glob(str(E / "output_minz-0.25" / "*_estimated.pcd"))[0]) - [0, 0, LIFT]

X0, X1, Y0, Y1 = -2, 30, -12, 8


def crop(p):
    p, h = height_slice(p, 0.35, 2.0)
    m = (p[:, 0] > X0) & (p[:, 0] < X1) & (p[:, 1] > Y0) & (p[:, 1] < Y1)
    return p[m], h[m]


R, hR = crop(raw)
t_e2, t_fin, t_bug = cKDTree(e2), cKDTree(fin), cKDTree(bug)
in_e2 = np.isfinite(t_e2.query(R, distance_upper_bound=0.1)[0])
in_fin = np.isfinite(t_fin.query(R, distance_upper_bound=0.1)[0])
in_bug = np.isfinite(t_bug.query(R, distance_upper_bound=0.1)[0])
rm_e2 = ~in_e2                    # removed by ERASOR2 alone
rm_ped = in_e2 & ~in_fin          # kept by ERASOR2 alone, removed once pedestrians are masked
stats = dict(raw=len(R), removed_bug=float((~in_bug).mean()), removed_erasor2=float(rm_e2.mean()),
             removed_extra_masking=float(rm_ped.mean()), removed_final=float((~in_fin).mean()))

fig, axs = plt.subplots(2, 2, figsize=(15, 8.3), sharex=True, sharey=True)
panels = [("(a) raw accumulation (the scans ERASOR2 uses)", R, hR, None),
          ("(b) ERASOR2 alone", *crop(e2), None),
          ("(c) ERASOR2 + pedestrian masking (final)", *crop(fin), None)]
for ax, (title, p, h, _) in zip(axs.ravel()[:3], panels):
    o = np.argsort(h)
    sc = ax.scatter(p[o, 0], p[o, 1], s=0.12, c=h[o], cmap="viridis", vmin=0.35, vmax=2.0, rasterized=True)
    ax.plot(ps[:, 0], ps[:, 1], "k-", lw=1.8)
    ax.set_title(title)
ax = axs[1, 1]
keep = ~rm_e2 & ~rm_ped
ax.scatter(R[keep, 0], R[keep, 1], s=0.08, c=MAP_GREY, rasterized=True)
ax.scatter(R[rm_e2, 0], R[rm_e2, 1], s=0.3, c="#d62728", rasterized=True, label=f"removed by ERASOR2 alone ({rm_e2.mean():.0%} of the points shown)")
ax.scatter(R[rm_ped, 0], R[rm_ped, 1], s=0.3, c="#1f77b4", rasterized=True,
           label=f"removed only with pedestrian masking ({rm_ped.mean():.0%})")
ax.plot(ps[:, 0], ps[:, 1], "k-", lw=1.8)
ax.legend(loc="lower left", markerscale=12, fontsize=10.5)
ax.set_title("(d) what was removed")
for ax in axs.ravel():
    ax.set_aspect("equal"); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
for ax in axs[1]:
    ax.set_xlabel("x [m]")
for ax in axs[:, 0]:
    ax.set_ylabel("y [m]")
cb = fig.colorbar(sc, ax=axs[:, :], shrink=0.6, pad=0.01); cb.set_label("height above ground [m] (a-c)")
fig.savefig(FIG / "08_static_maps.png", bbox_inches="tight")
print(stats)
