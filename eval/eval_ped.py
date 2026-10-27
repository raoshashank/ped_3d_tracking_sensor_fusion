#!/usr/bin/env python3
"""Compare ERASOR2 alone (output/) with ped3d masking + ERASOR2 (output_ped/).
Writes output_ped/{static_map_slam_frame,removed_points,removed_ped3d}.pcd and erasor2_ped_check.png."""
import numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

D = "data/Merged"; LIFT = 0.38   # ERASOR2 saves the map lifted by sensor_height
P = np.loadtxt(f"{D}/poses.txt").reshape(-1, 3, 4); traj = P[:, :, 3]

def vox(a, v=0.05):
    return np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a)).voxel_down_sample(v).points)

raw, ped = [], []
for i in range(0, 436, 2):   # the frames ERASOR2 uses (accum_interval 2)
    s = np.fromfile(f"{D}/velodyne/{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    m = np.fromfile(f"{D}/ped/{i:06d}.label", np.uint32).astype(bool)
    w = s @ P[i, :, :3].T + P[i, :, 3]
    raw.append(w); ped.append(w[m])
raw, ped = vox(np.vstack(raw)), vox(np.vstack(ped))
d_ped = cKDTree(traj[:, :2]).query(ped[:, :2])[0]
maps = {}
for name, f in [("ERASOR2 only", "output/Merged_0_frame_0_to_434_estimated.pcd"),
                ("ped3d + ERASOR2", "output_ped/MergedPed_0_frame_0_to_434_estimated.pcd")]:
    st = np.asarray(o3d.io.read_point_cloud(f).points) - [0, 0, LIFT]
    tree = cKDTree(st); maps[name] = (st, tree)
    surv = np.isfinite(tree.query(ped, distance_upper_bound=0.05)[0])
    rem = raw[np.isinf(tree.query(raw, distance_upper_bound=0.1)[0])]
    print(f"{name:16s}: static {len(st):8d} pts | raw voxels removed {len(rem) / len(raw):.2%} | "
          f"ped3d pedestrian voxels still in map: {surv.mean():.1%} "
          f"(<10 m from path {surv[d_ped < 10].mean():.1%}, 10-20 m {surv[d_ped >= 10].mean():.1%})")
    maps[name] += (rem,)
st, _, rem = maps["ped3d + ERASOR2"]
_, tree0, rem0 = maps["ERASOR2 only"]
new = rem[np.isfinite(tree0.query(rem, distance_upper_bound=0.05)[0])]   # removed now, kept before
print(f"removed now but kept by ERASOR2 alone: {len(new)} voxels; ped3d voxels total {len(ped)} "
      f"({np.mean(d_ped < 10):.0%} within 10 m of path)")
o3d.io.write_point_cloud("output_ped/static_map_slam_frame.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(st)))
o3d.io.write_point_cloud("output_ped/removed_points.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rem)))
o3d.io.write_point_cloud("output_ped/removed_ped3d.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ped)))

fig, ax = plt.subplots(1, 2, figsize=(22, 11), sharex=True, sharey=True)
sl = lambda a: a[(a[:, 2] > -0.2) & (a[:, 2] < 2.0)]
r = sl(raw); ax[0].scatter(r[::2, 0], r[::2, 1], s=0.03, c="0.5")
ax[0].scatter(rem0[:, 0], rem0[:, 1], s=0.4, c="r", label="removed by ERASOR2 alone")
ax[0].scatter(ped[:, 0], ped[:, 1], s=0.4, c="dodgerblue", alpha=0.6, label="ped3d pedestrians")
ax[0].legend(markerscale=15, loc="lower left"); ax[0].set_title("raw accumulation: ERASOR2 removals vs ped3d pedestrians")
k = sl(st); ax[1].scatter(k[::2, 0], k[::2, 1], s=0.03, c="k"); ax[1].plot(traj[:, 0], traj[:, 1], "g-", lw=1)
ax[1].set_title("static map, ped3d + ERASOR2 (0.2-2 m slice, green: trajectory)")
for a in ax:
    a.set_aspect("equal"); a.set_xlim(traj[:, 0].min() - 20, traj[:, 0].max() + 20); a.set_ylim(traj[:, 1].min() - 20, traj[:, 1].max() + 20)
plt.tight_layout(); plt.savefig("output_ped/erasor2_ped_check.png", dpi=80)
