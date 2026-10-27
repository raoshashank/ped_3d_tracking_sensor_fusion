#!/usr/bin/env python3
"""Score ros2_leg_detector's tracks (output_ped/leg_tracks_map.csv, from
../leg_detector/record_leg_tracks.py) with the same metric as lidar_ped_detect.py: inside the
camera FOV and <= 12 m, recall of ped3d detections and precision vs ped3d tracks; outside the FOV,
counts. Also draws both detectors + ped3d on the map (output_ped/compare_leg.png).
Run with the erasor2 conda env:  PYTHONPATH=. python compare_leg.py"""
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

import lidar_ped_detect as L

rows = np.loadtxt(L.OUT / "leg_tracks_map.csv", delimiter=",", skiprows=1)
t_scan, T_base = L.base_poses()
poses = np.loadtxt(L.D / "poses.txt").reshape(-1, 3, 4)
cam = np.load(L.HERE / "camera_model.npz")
cam = {k: cam[k] for k in cam.files}
dets, track = [], []
for tid, t, x, y in rows:
    i = int(np.argmin(np.abs(t_scan - t)))
    if abs(t_scan[i] - t) > 0.06:
        continue
    gz = poses[i, 2, 3] - 0.48
    rng = np.linalg.norm(poses[i, :2, 3] - [x, y])
    fov = float(L.in_fov(np.array([[x, y]]), np.array([gz + 1.0]), T_base[i], cam)[0])
    dets.append([i, t_scan[i], x, y, 1.7, 1.7, 0, rng, fov])
    track.append(int(tid))
dets, track = np.array(dets), np.array(track)
print("=== ros2_leg_detector (people_tracked) ===")
L.OUT = Path(tempfile.mkdtemp())      # evaluate() writes lidar_tracks.csv / npz; keep ours intact
gt = L.evaluate(dets, track)
L.OUT = L.HERE / "output_ped"
lid = np.load(L.OUT / "lidar_dets.npz")
print("=== lidar-vs-map (lidar_ped_detect.py), same metric ===")
L.OUT = Path(tempfile.mkdtemp())
L.evaluate(lid["dets"], lid["track"])
L.OUT = L.HERE / "output_ped"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
st = np.asarray(o3d.io.read_point_cloud(str(L.MAP)).points)
sl = st[(st[:, 2] > -0.1) & (st[:, 2] < 2.0)]
fig, ax = plt.subplots(1, 2, figsize=(26, 11), sharex=True, sharey=True)
for a, (name, d) in zip(ax, [("ros2_leg_detector /people_tracked", dets), ("lidar-vs-map detections", lid["dets"])]):
    a.scatter(sl[::3, 0], sl[::3, 1], s=0.03, c="0.75")
    a.plot(poses[:, 0, 3], poses[:, 1, 3], "k-", lw=1)
    a.scatter(gt[:, 1], gt[:, 2], s=8, c="limegreen", label="ped3d (camera FOV)")
    m = d[:, 8] > 0
    a.scatter(d[m, 2], d[m, 3], s=4, c="dodgerblue", label="in FOV")
    a.scatter(d[~m, 2], d[~m, 3], s=4, c="red", label="outside FOV")
    a.set_aspect("equal"); a.legend(markerscale=4, loc="lower left"); a.set_title(name)
    a.set_xlim(poses[:, 0, 3].min() - 20, poses[:, 0, 3].max() + 20); a.set_ylim(poses[:, 1, 3].min() - 20, poses[:, 1, 3].max() + 20)
plt.tight_layout(); plt.savefig(L.OUT / "compare_leg.png", dpi=70)
