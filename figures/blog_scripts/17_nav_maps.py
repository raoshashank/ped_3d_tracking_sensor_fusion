#!/usr/bin/env python3
"""Figures 17-18: the maps left behind as a by-product, as 2D navigation occupancy grids
(black = occupied, white = free, grey = unknown), built from the pedestrian-free map with
build_nav_map.py (occupancy mapping with the SLAM poses; only obstacle points that survive in the
static map count, so pedestrians never become obstacles). Grids are cropped to the known area.
Input: NAV/<recording>/map.pgm + map.yaml, poses from the recording's slam_output.
Run with the erasor2 env:  python 17_nav_maps.py NAV_DIR"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

ROOT = Path(p3d_env.P("P3D_BAGS_GO2NUS"))
NAV = Path(sys.argv[1])
FIG = Path(__file__).resolve().parent.parent / "figures"
SHOWCASE = "utown_new_14_with_overhead_2025-05-06_13-11-27_3_merged"
GALLERY = [  # longest walk per location whose SLAM has no step > 0.3 m between scans (no SLAM jumps)
    ("UTown", "utown_3_2025-03-07_14-36-18_2_corrected_merged"),
    ("UTown (new route)", "utown_new_14_with_overhead_2025-05-06_13-11-27_3_merged"),
    ("Science", "science_new_4_no_overhead_2025-05-05_13-01-50_7_merged"),
    ("Science (2)", "science_2_with_overhead_2025-03-25_12-25-01_2_corrected_merged"),
    ("Deck", "deck_new_4_with_overhead_2025-04-10_12-21-26_1_corrected_merged"),
    ("Deck (2)", "deck_6_no_overhead_2025-03-26_13-53-23_2_corrected_merged"),
    ("Terrace", "terrace_new_3_with_overhead_2025-04-02_11-43-48_17_merged_corrected"),
    ("Techno Edge", "techno_new_1_no_overhead_2025-05-08_12-06-34_12_merged"),
    ("RoboSG event hall", "robosg_3_with_overhead_2025-03-14_13-52-58_2_corrected_merged"),
]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 120})


def load(bag):
    d = NAV / bag
    with open(d / "map.pgm", "rb") as f:
        f.readline(); w, h = map(int, f.readline().split()); f.readline()
        img = np.frombuffer(f.read(), np.uint8).reshape(h, w)
    y = dict(l.split(": ", 1) for l in (d / "map.yaml").read_text().splitlines())
    res = float(y["resolution"]); ox, oy = [float(v) for v in y["origin"].strip("[]").split(",")[:2]]
    known = img != 205
    rows, cols = np.nonzero(known)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    pad = int(2 / res)
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    r1, c1 = min(r1 + pad, h), min(c1 + pad, w)
    crop = img[r0:r1, c0:c1]
    # extent in map coordinates (image row 0 = max y)
    ext = [ox + c0 * res, ox + c1 * res, oy + (h - r1) * res, oy + (h - r0) * res]
    path = np.loadtxt(ROOT / bag / "slam_output/poses_base_link_tum.txt")[:, 1:3]
    return crop, ext, path


def draw(ax, bag, title=None, lw=1.2):
    img, ext, path = load(bag)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=ext, interpolation="nearest")
    ax.plot(path[:, 0], path[:, 1], "-", color="#d62728", lw=lw)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.7")
    L = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    if title:
        ax.set_title(f"{title}  ({L:.0f} m walk)")
    return L


fig, ax = plt.subplots(figsize=(15, 12))
L = draw(ax, SHOWCASE, lw=1.6)
ax.plot([], [], "-", color="#d62728", label="robot path")
ax.legend(loc="lower left")
ax.set_title(f"Navigation map of a {L:.0f} m walk across campus (black: obstacles, white: free space)")
fig.savefig(FIG / "17_static_map_showcase.png", dpi=110, bbox_inches="tight")
plt.close(fig)

fig, axs = plt.subplots(3, 3, figsize=(16, 14))
for a, (title, bag) in zip(axs.ravel(), GALLERY):
    draw(a, bag, title)
fig.suptitle("Navigation maps, one per location (black: obstacles, white: free space, red: robot path)", y=0.95)
fig.savefig(FIG / "18_static_map_gallery.png", dpi=110, bbox_inches="tight")
print("ok")
