#!/usr/bin/env python3
"""Static recall / pedestrian residue per map, against a *persistent* static reference.

Reference = non-ground 5 cm voxels within 12 m of the robot in window B (even scans 130-198, robot
standing near the bench) that are ALSO seen (within 0.1 m) in window A (scans 0-118) or C
(250-436), minus anything within 0.15 m of a ped3d-masked point (people standing for the whole
recording are persistent too). Anything moving through B (people walking past the robot) is not in
A or C at the same place and drops out. A map has good static recall if most reference voxels have
a map point within 7 cm. Pedestrian residue: ped3d-masked voxels (data/Merged/ped) with a map
point within 5 cm. Regions are height bands (lidar upper FOV is +15 deg, so h>1.8 is far stuff).
Usage: python recall_eval.py map1.pcd map2.pcd ...   (maps in the SLAM frame)"""
import sys, numpy as np, open3d as o3d
from scipy.spatial import cKDTree
P = np.loadtxt("data/Merged/poses.txt").reshape(-1, 3, 4)
def vox(a, v=0.05): return np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a)).voxel_down_sample(v).points)
A, B, C, ped = [], [], [], []
for i in range(0, 437):
    s = np.fromfile(f"data/Merged/velodyne/{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]; g = np.fromfile(f"data/Merged/patchwork/{i:06d}.label", np.uint32)
    m = np.fromfile(f"data/Merged/ped/{i:06d}.label", np.uint32).astype(bool); w = s @ P[i, :, :3].T + P[i, :, 3]; ped.append(w[m])
    if i % 2: continue
    if 130 <= i < 200: B.append(w[(g == 0) & (np.linalg.norm(s, axis=1) < 12)])
    elif i < 120: A.append(w)
    elif i >= 250: C.append(w)
B = vox(np.vstack(B)); A, C = vox(np.vstack(A)), vox(np.vstack(C)); ped = vox(np.vstack(ped))
persist = np.isfinite(cKDTree(A).query(B, distance_upper_bound=0.1)[0]) | np.isfinite(cKDTree(C).query(B, distance_upper_bound=0.1)[0])
notped = np.isinf(cKDTree(ped).query(B, distance_upper_bound=0.15)[0])
ref = B[persist & notped]; h = ref[:, 2] + 0.25
regions = {"all": np.ones(len(ref), bool), "h<0.5": h < 0.5, "0.5-1.8": (h >= 0.5) & (h < 1.8), "h>1.8": h >= 1.8}
print(f"reference: {len(B)} window-B voxels, {persist.sum()} persistent, {len(ref)} after dropping ped3d-masked; {len(ped)} ped3d voxels")
print(f"{'map':44s} {'pts':>8s} " + " ".join(f"{k:>8s}" for k in regions) + "  ped-resid")
for f in sys.argv[1:]:
    st = np.asarray(o3d.io.read_point_cloud(f).points); t = cKDTree(st)
    ok = np.isfinite(t.query(ref, distance_upper_bound=0.07)[0]); res = np.isfinite(t.query(ped, distance_upper_bound=0.05)[0]).mean()
    print(f"{f:44s} {len(st):8d} " + " ".join(f"{ok[m].mean():8.1%}" for m in regions.values()) + f"  {res:7.1%}")
