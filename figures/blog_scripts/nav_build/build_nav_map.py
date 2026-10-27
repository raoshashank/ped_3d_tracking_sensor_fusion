#!/usr/bin/env python3
"""2D occupancy grid for Nav2 (map_server .pgm + .yaml) from the offline SLAM + pedestrian-free map.

Occupancy-grid mapping with known poses (what a 2D SLAM mapper does, minus the scan matching: the
3D SLAM trajectory is used as is). Per lidar scan:
  - obstacle points: Patchwork++ non-ground points, BAND above the ground under the robot (median of
    that scan's ground points within GROUND_R), within MAX_RANGE, AND present in the final static map
    (within STATIC_TOL) -> pedestrians and other moving things never become obstacles
  - rays: per bearing bin, from the sensor to the nearest obstacle (occupied end), or, if the bin has
    no obstacle, to the farthest ground return (seeing the ground proves line of sight) -> free cells
  - log-odds update, each cell counted at most once per scan
The robot footprint along the trajectory is marked free. Cells: occupied if p > OCC_P, free if
p < FREE_P, unknown otherwise.

Inputs (from the bag's slam_output/): erasor2/data/Merged/{velodyne,patchwork,poses.txt},
erasor2/output_ped/static_map_slam_frame.pcd.
Outputs (nav_map/): map.pgm, map.yaml (Nav2 map_server, frame = the bag's SLAM `map` frame),
map_preview.png, logodds.npy.
Run with the erasor2 env:  python build_nav_map.py [slam_output_dir]
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
S = Path(ARGS[0]).resolve()          # slam_output dir (read only)
OUT = Path(ARGS[1]).resolve()        # output dir (blog scratch; the recording is not modified)
D = S / "erasor2" / "data" / "Merged"
STATIC = S / "erasor2" / "output_ped" / "static_map_slam_frame.pcd"
NO_STATIC_FILTER = "--no-static-filter" in sys.argv   # comparison only: all band points are obstacles

RES = 0.05                    # m / cell
BAND = (0.15, 0.60)           # m above ground: what the Go2 would hit (curb-height steps pass below)
MAX_RANGE, MIN_RANGE = 12.0, 0.6
STATIC_TOL = 0.10             # m, obstacle point must have a static-map point this close
GROUND_R = 3.0                # m, ground height under the robot from ground points within this radius
N_BINS = 720                  # bearing bins (0.5 deg)
L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -2.0, 3.5
OCC_P, FREE_P = 0.65, 0.25    # also written to map.yaml
FOOTPRINT_R = 0.25            # m, robot footprint cleared along the trajectory
PAD = 2.0                     # m around the data


def main():
    global OUT
    if NO_STATIC_FILTER:
        OUT = OUT / "no_static_filter"
    OUT.mkdir(parents=True, exist_ok=True)
    poses = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)          # map <- rslidar
    base = np.loadtxt(S / "poses_base_link_tum.txt")[:, 1:3]       # robot xy
    static = cKDTree(np.asarray(o3d.io.read_point_cloud(str(STATIC)).points))

    # grid extent: trajectory +- MAX_RANGE, clipped to the static map's extent
    st_pts = static.data
    lo = np.maximum(poses[:, :2, 3].min(0) - MAX_RANGE, st_pts[:, :2].min(0)) - PAD
    hi = np.minimum(poses[:, :2, 3].max(0) + MAX_RANGE, st_pts[:, :2].max(0)) + PAD
    lo = np.floor(lo / RES) * RES
    shape = np.ceil((hi - lo) / RES).astype(int)                   # (nx, ny)
    L = np.zeros(shape, np.float32)

    def cells(xy):
        ij = np.floor((xy - lo) / RES).astype(int)
        ok = (ij >= 0).all(1) & (ij < shape).all(1)
        return ij[ok]

    step = RES / 2
    for i, P in enumerate(poses):
        s = np.fromfile(D / "velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        g = np.fromfile(D / "patchwork" / f"{i:06d}.label", np.uint32)
        r_xy = np.hypot(s[:, 0], s[:, 1])
        near_g = (g == 1) & (r_xy < GROUND_R)
        gz = np.median(s[near_g, 2]) if near_g.sum() > 20 else -0.38
        h = s[:, 2] - gz
        w = s @ P[:, :3].T + P[:, 3]
        o = P[:2, 3]
        rng_ok = (r_xy > MIN_RANGE) & (r_xy < MAX_RANGE)
        obst = (g == 0) & rng_ok & (h > BAND[0]) & (h < BAND[1])
        idx = np.flatnonzero(obst)
        if len(idx) and not NO_STATIC_FILTER:
            idx = idx[np.isfinite(static.query(w[idx], distance_upper_bound=STATIC_TOL)[0])]
        gnd = np.flatnonzero((g == 1) & rng_ok)
        # bearing bins in the lidar frame
        bearing = ((np.arctan2(s[:, 1], s[:, 0]) + np.pi) / (2 * np.pi) * N_BINS).astype(int) % N_BINS
        end = np.full(N_BINS, -1)                                   # point index of the ray end
        end_r = np.full(N_BINS, np.inf)
        is_obst = np.zeros(N_BINS, bool)
        if len(idx):
            order = idx[np.argsort(r_xy[idx])[::-1]]                 # far -> near, nearest wins
            end[bearing[order]] = order
            end_r[bearing[order]] = r_xy[order]
            is_obst[bearing[order]] = True
        if len(gnd):
            order = gnd[np.argsort(r_xy[gnd])]                       # near -> far, farthest wins
            b = bearing[order]
            free_only = ~is_obst[b]
            end[b[free_only]] = order[free_only]
        ok = end >= 0
        ends = w[end[ok], :2]
        # free cells along each ray (excluding the end cell of obstacle rays)
        d = np.linalg.norm(ends - o, axis=1)
        n = np.maximum((d / step).astype(int), 1)
        k = np.concatenate([np.arange(m) for m in n])
        ray = np.repeat(np.arange(len(ends)), n)
        frac = k / np.repeat(n, n)
        pts = o + (ends[ray] - o) * frac[:, None]
        cut = np.repeat(is_obst[ok], n) & (np.repeat(d, n) * (1 - frac) < RES)   # stop short of obstacles
        free = cells(pts[~cut])
        occ = cells(w[idx, :2]) if len(idx) else np.zeros((0, 2), int)
        occ_u = np.unique(occ, axis=0)
        free_u = np.unique(free, axis=0)
        if len(occ_u) and len(free_u):   # a cell hit this scan is not also cleared by it
            keyf = free_u[:, 0] * shape[1] + free_u[:, 1]
            keyo = occ_u[:, 0] * shape[1] + occ_u[:, 1]
            free_u = free_u[~np.isin(keyf, keyo)]
        L[free_u[:, 0], free_u[:, 1]] += L_FREE
        L[occ_u[:, 0], occ_u[:, 1]] += L_OCC
        np.clip(L, L_MIN, L_MAX, out=L)
        if i % 100 == 0:
            print(f"scan {i}/{len(poses)}", flush=True)

    # robot footprint along the trajectory is free space
    fr = int(np.ceil(FOOTPRINT_R / RES))
    off = np.array([(a, b) for a in range(-fr, fr + 1) for b in range(-fr, fr + 1) if (a * a + b * b) * RES ** 2 <= FOOTPRINT_R ** 2])
    foot = cells(base)
    fc = (foot[:, None, :] + off[None]).reshape(-1, 2)
    fc = fc[(fc >= 0).all(1) & (fc < shape).all(1)]
    L[fc[:, 0], fc[:, 1]] = np.minimum(L[fc[:, 0], fc[:, 1]], L_MIN)

    p = 1 - 1 / (1 + np.exp(L))
    img = np.full(shape, 205, np.uint8)                              # unknown
    img[p < FREE_P] = 254
    img[p > OCC_P] = 0
    img = np.flipud(img.T)                                          # rows = y descending, cols = x
    with open(OUT / "map.pgm", "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    (OUT / "map.yaml").write_text(
        f"image: map.pgm\nmode: trinary\nresolution: {RES}\norigin: [{lo[0]:.3f}, {lo[1]:.3f}, 0.0]\n"
        f"negate: 0\noccupied_thresh: {OCC_P}\nfree_thresh: {FREE_P}\n")
    np.save(OUT / "logodds.npy", L)
    n_occ, n_free = int((img == 0).sum()), int((img == 254).sum())
    print(f"map {img.shape[1]}x{img.shape[0]} cells ({img.shape[1] * RES:.0f} x {img.shape[0] * RES:.0f} m), "
          f"occupied {n_occ}, free {n_free}, unknown {img.size - n_occ - n_free}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 14 * img.shape[0] / img.shape[1]))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=[lo[0], lo[0] + img.shape[1] * RES, lo[1], lo[1] + img.shape[0] * RES])
    ax.plot(base[:, 0], base[:, 1], "r-", lw=1)
    ax.set_title(f"Nav2 map ({RES} m, band {BAND[0]}-{BAND[1]} m above ground), red = robot trajectory")
    plt.tight_layout()
    plt.savefig(OUT / "map_preview.png", dpi=90)


if __name__ == "__main__":
    main()
