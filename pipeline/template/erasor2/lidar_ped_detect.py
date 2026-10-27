#!/usr/bin/env python3
"""Lidar pedestrian detection by background subtraction against the static map, 360 deg.

Per scan: points -> map frame (poses.txt); drop Patchwork ground; drop points with a static-map
point within R_RES; DBSCAN the residual; keep person-sized clusters; keep detections that persist
(neighbour within PERSIST_R in >= 2 of the 4 surrounding scans); link into tracks by nearest
neighbour. Clusters wider than a person (people side by side) are k-means split by width. Heights are relative to the local ground (median z of nearby Patchwork ground points),
so the Go2's stance does not matter. The camera FOV flag uses camera_model.npz (ped3d_lib K,
cam_T_base) and the SLAM base_link poses.

Evaluation against ped3d (camera+lidar fusion, 4 Hz), inside the FOV and <= EVAL_R:
  recall    = ped3d detections with a lidar detection within MATCH_R at the same time
  precision = lidar detections with a ped3d track position within MATCH_R (lower bound: ped3d misses too)
Outside the FOV ped3d has nothing, so we report counts and tracks.
Outputs: output_ped/lidar_dets.npz, lidar_tracks.csv, lidar_detect.png.  Run: erasor2 conda env.
"""
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN, KMeans

from ped_tracks import load_tracks, PED3D
import dataset_profile as DP  # noqa: E402  (on sys.path via ped_tracks)

HERE = Path(__file__).resolve().parent
D, OUT = HERE / "data" / "Merged", HERE / "output_ped"
MAP = OUT / "static_map_slam_frame.pcd"
R_RES, MAX_RANGE = 0.15, 20.0
EPS, MIN_PTS = 0.35, 6
FOOT_MIN, FOOT_MAX, TOP_MIN, TOP_MAX, BOTTOM_MAX, N_MIN = 0.15, 1.2, 0.45, 2.3, 1.0, 8
SPLIT_W = 0.7   # m: a cluster wider than FOOT_MAX is k-means split into ceil(width / SPLIT_W) people
PERSIST_R, LINK_R = 0.6, 0.7
EVAL_R, MATCH_R = 12.0, 0.8
TAN_FOV_UP = np.tan(np.radians(15))   # upper FOV (rslidar and VLP-16: +15 deg)
GROUND_BELOW = DP.load(HERE)["lidar_height"] + 0.10   # nominal ground below the lidar when no Patchwork ground is near (go2nus 0.48)


def base_poses():
    b = np.loadtxt(HERE.parent / "poses_base_link_tum.txt")
    T = np.tile(np.eye(4), (len(b), 1, 1))
    T[:, :3, :3] = R.from_quat(b[:, 4:8]).as_matrix()
    T[:, :3, 3] = b[:, 1:4]
    return b[:, 0], T


def in_fov(xy_map, z_map, T_base, cam):
    p = np.c_[xy_map, z_map, np.ones(len(xy_map))] @ np.linalg.inv(T_base).T @ cam["cam_T_base"].T
    z = p[:, 2]
    u = cam["K"][0, 0] * p[:, 0] / np.where(z > 0, z, 1) + cam["K"][0, 2]
    v = cam["K"][1, 1] * p[:, 1] / np.where(z > 0, z, 1) + cam["K"][1, 2]
    return (z > 0.3) & (u >= 0) & (u < cam["W"]) & (v >= 0) & (v < cam["H"])


def detect():
    """Detections after the persistence filter + greedy nearest-neighbour linking (v1 tracker;
    lidar_tracker.py replaces the tracking)."""
    dets = detect_raw()
    dets = dets[persistent(dets)]
    return dets, greedy_link(dets)


def detect_raw(cache=True):
    """Person-sized clusters of the residual against the static map, every scan, no temporal filter.
    Columns: scan, t, x, y, top_h, height, n_pts, range, in_fov. Cached in output_ped/lidar_dets_raw.npz."""
    cf = OUT / "lidar_dets_raw.npz"
    if cache and cf.exists() and cf.stat().st_mtime > MAP.stat().st_mtime:
        return np.load(cf)["dets"]
    poses = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
    t_scan, T_base = base_poses()[0], base_poses()[1]
    cam = np.load(HERE / "camera_model.npz")
    cam = {k: cam[k] for k in cam.files}
    tree = cKDTree(np.asarray(o3d.io.read_point_cloud(str(MAP)).points))
    dets = []   # scan, t, x, y, z_top, height, n, range, in_fov
    for i, P in enumerate(poses):
        s = np.fromfile(D / "velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
        g = np.fromfile(D / "patchwork" / f"{i:06d}.label", np.uint32)
        rng = np.linalg.norm(s, axis=1)
        w = s @ P[:, :3].T + P[:, 3]
        gtree = cKDTree(w[g == 1][:, :2]) if (g == 1).sum() > 10 else None
        cand = (g == 0) & (rng < MAX_RANGE)
        res = cand & np.isinf(tree.query(w, distance_upper_bound=R_RES)[0])
        idx = np.flatnonzero(res)
        if len(idx) < MIN_PTS:
            continue
        lab = DBSCAN(eps=EPS, min_samples=MIN_PTS).fit(w[idx]).labels_
        parts = []
        for k in range(lab.max() + 1):
            pts = w[idx[lab == k]]
            width = (pts[:, :2].max(0) - pts[:, :2].min(0)).max()
            if width > FOOT_MAX and width <= 4 * FOOT_MAX and len(pts) >= 2 * N_MIN:
                # people walking side by side merge at EPS: split by footprint width
                kk = int(np.ceil(width / SPLIT_W))
                sub = KMeans(kk, n_init=3, random_state=0).fit(pts[:, :2]).labels_
                parts += [pts[sub == q] for q in range(kk)]
            else:
                parts.append(pts)
        for pts in parts:
            if len(pts) < N_MIN:
                continue
            cen = pts[:, :2].mean(0)
            # local ground: median z of ground points within 1.5 m, else the lidar's nominal ground
            gz = np.median(w[g == 1][gtree.query_ball_point(cen, 1.5)][:, 2]) if gtree else P[2, 3] - GROUND_BELOW
            if gtree and not gtree.query_ball_point(cen, 1.5):
                gz = P[2, 3] - GROUND_BELOW
            top, bottom = pts[:, 2].max() - gz, pts[:, 2].min() - gz
            foot = (pts[:, :2].max(0) - pts[:, :2].min(0)).max()
            r = np.linalg.norm(cen - P[:2, 3])
            if not (FOOT_MIN <= foot <= FOOT_MAX and TOP_MIN <= top <= TOP_MAX and bottom <= BOTTOM_MAX):
                continue
            dets.append([i, t_scan[i], cen[0], cen[1], top, top - bottom, len(pts), r,
                         float(in_fov(cen[None], np.array([gz + 1.0]), T_base[i], cam)[0])])
    dets = np.array(dets).reshape(-1, 9)
    np.savez(cf, dets=dets)
    return dets


def persistent(dets):
    """Neighbour within PERSIST_R in >= 2 of scans i-2..i+2 (excluding i)."""
    by_scan = {}
    for j, d in enumerate(dets):
        by_scan.setdefault(int(d[0]), []).append(j)
    keep = np.zeros(len(dets), bool)
    for j, d in enumerate(dets):
        n = 0
        for di in (-2, -1, 1, 2):
            js = by_scan.get(int(d[0]) + di, [])
            if js and np.min(np.linalg.norm(dets[js, 2:4] - d[2:4], axis=1)) <= PERSIST_R:
                n += 1
        keep[j] = n >= 2
    return keep


def greedy_link(dets):
    """v1 tracker: greedy nearest-neighbour linking, no motion model."""
    track = -np.ones(len(dets), int)
    last = {}   # track id -> (scan, xy)
    nt = 0
    order = np.argsort(dets[:, 0], kind="stable")
    for j in order:
        i, xy = int(dets[j, 0]), dets[j, 2:4]
        best, bd = None, LINK_R
        for tid, (si, lxy) in last.items():
            if 0 < i - si <= 3:
                dd = np.linalg.norm(lxy - xy) / (i - si) ** 0.5
                if dd < bd:
                    best, bd = tid, dd
        if best is None:
            best, nt = nt, nt + 1
        track[j] = best
        last[best] = (i, xy)
    return track


def evaluate(dets, track):
    t_scan = np.loadtxt(HERE.parent / "poses_rslidar_tum.txt")[:, 0]
    import pickle
    d = pickle.load(open(PED3D, "rb"))
    tracks = load_tracks("fused")
    # ped3d positions in the SLAM frame at their own frame times, detected (not interpolated) rows only
    gt = []
    for tid, tr in d["tracks"].items():
        if tid not in tracks:
            continue
        t, xy, _ = tracks[tid]
        ok = np.isfinite(tr["fused"][:, :2]).all(1)
        det = tr["detected"][ok] & ~tr["interpolated"][ok]
        gt += [(tt, x, y) for tt, (x, y), dd in zip(t, xy, det) if dd]
    gt = np.array(gt)
    poses = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
    # recall: ped3d detections <= EVAL_R from the robot with a lidar detection within MATCH_R at the nearest scan
    hit = []
    for tt, x, y in gt:
        i = int(np.argmin(np.abs(t_scan - tt)))
        if abs(t_scan[i] - tt) > 0.08 or np.linalg.norm(poses[i, :2, 3] - [x, y]) > EVAL_R:
            continue
        m = dets[:, 0] == i
        hit.append(m.any() and np.min(np.linalg.norm(dets[m, 2:4] - [x, y], axis=1)) <= MATCH_R)
    hit = np.array(hit)
    # precision: in-FOV lidar detections <= EVAL_R with a ped3d track position (within its own span) within MATCH_R
    fov = (dets[:, 8] > 0) & (dets[:, 7] <= EVAL_R)
    prec = []
    for dd in dets[fov]:
        found = False
        for t, xy, _ in tracks.values():
            if t[0] - 0.15 <= dd[1] <= t[-1] + 0.15:
                p = [np.interp(dd[1], t, xy[:, 0]), np.interp(dd[1], t, xy[:, 1])]
                if np.linalg.norm(p - dd[2:4]) <= MATCH_R:
                    found = True
                    break
        prec.append(found)
    prec = np.array(prec)
    out_fov = (dets[:, 8] == 0)
    tids_out = np.unique(track[out_fov])
    long_out = [t for t in tids_out if (track == t).sum() >= 5 and out_fov[track == t].mean() > 0.5]
    print(f"lidar detections: {len(dets)} in {len(np.unique(track))} tracks; in camera FOV {int((dets[:, 8] > 0).sum())}, "
          f"outside {int(out_fov.sum())}")
    print(f"inside FOV, <= {EVAL_R:.0f} m: recall of ped3d detections {hit.mean():.1%} (n={len(hit)}), "
          f"precision vs ped3d tracks {prec.mean():.1%} (n={len(prec)})")
    print(f"outside FOV: {len(tids_out)} tracks, {len(long_out)} with >= 5 scans mostly outside the FOV "
          f"(median length {np.median([(track == t).sum() for t in long_out]) if long_out else 0:.0f} scans)")
    with open(OUT / "lidar_tracks.csv", "w") as f:
        f.write("track,scan,t,x,y,top_h,height,n_pts,range,in_fov\n")
        for j in np.argsort(track, kind="stable"):
            f.write(f"{track[j]}," + ",".join(f"{v:.3f}" for v in dets[j]) + "\n")
    np.savez(OUT / "lidar_dets.npz", dets=dets, track=track)
    return gt


def figure(dets, track, gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = np.asarray(o3d.io.read_point_cloud(str(MAP)).points)
    poses = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
    sl = st[(st[:, 2] > -0.1) & (st[:, 2] < 2.0)]
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(sl[::3, 0], sl[::3, 1], s=0.03, c="0.75")
    ax.plot(poses[:, 0, 3], poses[:, 1, 3], "k-", lw=1, label="robot")
    ax.scatter(gt[:, 1], gt[:, 2], s=6, c="limegreen", label="ped3d detections (camera FOV)")
    m = dets[:, 8] > 0
    ax.scatter(dets[m, 2], dets[m, 3], s=4, c="dodgerblue", label="lidar-vs-map detections in FOV")
    ax.scatter(dets[~m, 2], dets[~m, 3], s=4, c="red", label="lidar-vs-map detections outside FOV")
    ax.set_aspect("equal"); ax.legend(markerscale=4, loc="lower left")
    ax.set_xlim(poses[:, 0, 3].min() - 20, poses[:, 0, 3].max() + 20); ax.set_ylim(poses[:, 1, 3].min() - 20, poses[:, 1, 3].max() + 20)
    ax.set_title("pedestrian detection by background subtraction against the static map")
    plt.tight_layout(); plt.savefig(OUT / "lidar_detect.png", dpi=80)


if __name__ == "__main__":
    dets, track = detect()
    gt = evaluate(dets, track)
    figure(dets, track, gt)
