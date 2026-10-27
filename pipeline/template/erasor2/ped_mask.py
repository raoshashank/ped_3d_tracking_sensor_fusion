#!/usr/bin/env python3
"""Mask pedestrian points in every scan using the ped3d tracks, then write a copy of the ERASOR2
sequence with those points removed (data/MergedPed), so ERASOR2 only has to handle what the camera
missed.

Per scan, for each tracked pedestrian (ped_tracks.positions_at, <= MAX_RANGE from the lidar):
  - candidate points: non-ground (Patchwork++), 0.1 - 2.2 m above ground (lidar z in Z_MIN..Z_MAX)
  - core: candidates within R_CORE (xy) of the track position
  - region growing from the core through GROW_STEP neighbours, limited to R_MAX from the track:
    takes the whole body (arms, bags) but stops at walls / planters.
  - --guard: un-mask points whose STATIC_VOXEL voxel is occupied in >= STATIC_MIN_SCANS scans taken
    with no tracked pedestrian within CLEAR_RADIUS during +-CLEAR_TIME. Off by default: untracked
    people (behind the robot, missed by the camera) make their own voxels look static. The
    furniture problem is handled instead by vis_filter.py, which applies the mask to the final map
    and keeps masked points that no scan ever saw through.
Outputs:
  data/Merged/ped/NNNNNN.label   uint32 per raw .bin point, 1 = pedestrian
  data/MergedPed/{velodyne,patchwork,hdbscan}/  scans + labels without pedestrian points
  data/MergedPed/poses.txt       symlink to Merged/poses.txt
Run with the erasor2 conda env:  python ped_mask.py
"""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # slam_output: dataset_profile.py
import dataset_profile as DP  # noqa: E402
from ped_tracks import (load_tracks, load_lidar_tracks, positions_at, positions_within,
                        standing_group_completion)

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "data" / "Merged", HERE / "data" / "MergedPed"
_H = DP.load(HERE)["lidar_height"]
Z_MIN, Z_MAX = DP.PED_Z_MIN - _H, DP.CEIL_Z - _H   # lidar frame (go2nus -0.28, 1.8: ground at -0.38)
R_CORE = 0.4                       # m, fused tracks: median 0.04 m / p90 0.13 m off lidar within 10 m
GROW_STEP, R_MAX = 0.15, 0.8       # m
MAX_RANGE = 20.0                   # m, beyond this ped3d positions are unreliable (median 2 m off)
STATIC_VOXEL, STATIC_MIN_SCANS, CLEAR_RADIUS, CLEAR_TIME = 0.1, 5, 1.0, 3.0


def mask_scan(scan, ground, pose, peds):
    """scan: (N,4) lidar frame, pose: 3x4 map<-lidar, peds: (P,2) map xy. Returns bool mask (N,)."""
    ped = np.zeros(len(scan), bool)
    if not len(peds):
        return ped
    peds = peds[np.linalg.norm(peds - pose[:2, 3], axis=1) <= MAX_RANGE]
    if not len(peds):
        return ped
    cand = (ground == 0) & (scan[:, 2] > Z_MIN) & (scan[:, 2] < Z_MAX)
    idx = np.flatnonzero(cand)
    pts = scan[idx, :3] @ pose[:, :3].T + pose[:, 3]
    tree = cKDTree(pts)
    for p, seed in zip(peds, cKDTree(pts[:, :2]).query_ball_point(peds, R_CORE)):
        if not seed:
            continue
        allowed = np.hypot(*(pts[:, :2] - p).T) <= R_MAX
        sel = np.zeros(len(pts), bool)
        sel[seed] = True
        frontier = list(seed)
        while frontier:
            nb = np.unique(np.concatenate([np.asarray(n, int) for n in tree.query_ball_point(pts[frontier], GROW_STEP)]))
            nb = nb[allowed[nb] & ~sel[nb]]
            sel[nb] = True
            frontier = list(nb)
        ped[idx[sel]] = True
    return ped


def voxel_keys(w):
    v = np.floor(w / STATIC_VOXEL).astype(np.int64) + (1 << 20)
    return (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]


def static_voxels(tracks, t_scan, poses):
    """Voxels seen occupied (non-ground) in >= STATIC_MIN_SCANS scans with no pedestrian nearby
    (in space: CLEAR_RADIUS, in time: +-CLEAR_TIME)."""
    counts = {}
    for i, t in enumerate(t_scan):
        scan = np.fromfile(SRC / "velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)
        ground = np.fromfile(SRC / "patchwork" / f"{i:06d}.label", np.uint32)
        w = scan[ground == 0, :3] @ poses[i, :, :3].T + poses[i, :, 3]
        peds = positions_within(tracks, t - CLEAR_TIME, t + CLEAR_TIME)
        if len(peds):
            w = w[np.isinf(cKDTree(peds).query(w[:, :2], distance_upper_bound=CLEAR_RADIUS)[0])]
        for k in np.unique(voxel_keys(w)):
            counts[k] = counts.get(k, 0) + 1
    return np.array(sorted(k for k, c in counts.items() if c >= STATIC_MIN_SCANS), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", action="store_true", help="un-mask points with static evidence")
    ap.add_argument("--lidar-tracks", action="store_true",
                    help="also mask around lidar_ped_detect.py tracks (output_ped/lidar_tracks.csv)")
    ap.add_argument("--group-completion", metavar="MAP_PCD",
                    help="list person-sized static-map columns next to held standing tracks as candidates")
    ap.add_argument("--accept", default="", help="comma-separated candidate ids to mask (after review)")
    args = ap.parse_args()
    tracks = load_tracks("fused")
    if args.lidar_tracks:
        lt = load_lidar_tracks()
        print(f"+ {len(lt)} lidar tracks")
        tracks.update(lt)
    if args.group_completion:
        accept = tuple(int(a) for a in args.accept.split(",") if a.strip())
        tracks.update(standing_group_completion(tracks, args.group_completion, accept=accept))
    t_scan = np.loadtxt(HERE.parent / "poses_rslidar_tum.txt")[:, 0]
    poses = np.loadtxt(SRC / "poses.txt").reshape(-1, 3, 4)
    static = static_voxels(tracks, t_scan, poses) if args.guard else np.zeros(0, np.int64)
    n_unmasked = 0
    for sub in ("ped",):
        (SRC / sub).mkdir(exist_ok=True)
    for sub in ("velodyne", "patchwork", "hdbscan"):
        (DST / sub).mkdir(parents=True, exist_ok=True)
    if not (DST / "poses.txt").exists():
        os.symlink(SRC / "poses.txt", DST / "poses.txt")

    n_ped, n_tot, n_scans_with = 0, 0, 0
    for i, t in enumerate(t_scan):
        name = f"{i:06d}"
        scan = np.fromfile(SRC / "velodyne" / f"{name}.bin", np.float32).reshape(-1, 4)
        ground = np.fromfile(SRC / "patchwork" / f"{name}.label", np.uint32)
        inst_raw = np.fromfile(SRC / "hdbscan" / f"{name}.label", np.uint32)
        ped = mask_scan(scan, ground, poses[i], positions_at(tracks, t))
        if ped.any() and len(static):
            w = scan[ped, :3] @ poses[i, :, :3].T + poses[i, :, 3]
            keys = voxel_keys(w)
            pos = np.minimum(np.searchsorted(static, keys), len(static) - 1)
            is_static = static[pos] == keys
            ped[np.flatnonzero(ped)[is_static]] = False
            n_unmasked += is_static.sum()
        ped.astype(np.uint32).tofile(SRC / "ped" / f"{name}.label")
        keep = ~ped
        scan[keep].tofile(DST / "velodyne" / f"{name}.bin")
        ground[keep].tofile(DST / "patchwork" / f"{name}.label")
        inst_raw[keep].tofile(DST / "hdbscan" / f"{name}.label")
        n_ped += ped.sum(); n_tot += len(scan); n_scans_with += ped.any()
    print(f"{len(t_scan)} scans, {n_scans_with} with pedestrians; masked {n_ped} points "
          f"({n_ped / n_tot:.2%}), un-masked {n_unmasked} with static evidence; wrote {DST}")


if __name__ == "__main__":
    main()
