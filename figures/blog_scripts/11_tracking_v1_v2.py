#!/usr/bin/env python3
"""Figure 11: lidar tracks v1 (persistence filter + greedy nearest-neighbour linking) vs v2
(constant-velocity Kalman filter + Hungarian assignment + RTS smoothing + re-linking), demo bag.
Each track has its own colour, so fragmentation shows as colour changes along one walking path."""
import numpy as np

from tf_common import DPI, FIG, OP, arr, draw_map, static_map_slice, style, track_colours

plt = style()
pts = static_map_slice()
v1 = arr(OP / "lidar_tracks_greedy.csv")      # track, scan, t, x, y, ...
v2 = arr(OP / "lidar_tracks_smooth.csv")      # track, scan, t, x, y, vx, vy, pos_std, detected, ...


def tracks(a):
    out = []
    for k in np.unique(a[:, 0]):
        r = a[a[:, 0] == k]
        out.append(r[np.argsort(r[:, 2])])
    return out


T1, T2 = tracks(v1), tracks(v2)
fig = plt.figure(figsize=(13.2, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], hspace=0.28, wspace=0.18, top=0.93)
xl, yl = (-3, 34), (-12.5, 5.5)
for row, (T, name) in enumerate([(T1, "v1: persistence filter + greedy nearest-neighbour linking"),
                                 (T2, "v2: Kalman filter + Hungarian matching + RTS smoothing + re-linking")]):
    ax = fig.add_subplot(gs[row, 0])
    draw_map(ax, pts)
    cols = track_colours(len(T), seed=row + 3)
    for c, r in zip(cols, T):
        ax.plot(r[:, 3], r[:, 4], "-", color=c, lw=1.6, alpha=0.95)
        ax.plot(r[0, 3], r[0, 4], "o", color=c, ms=3.5)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_title(f"{name}: {len(T)} tracks", loc="left")
    if row == 0:
        ax.legend(loc="lower left", framealpha=0.9)

dur1 = np.array([np.ptp(r[:, 2]) for r in T1])
dur2 = np.array([np.ptp(r[:, 2]) for r in T2])
ax = fig.add_subplot(gs[0, 1])
bins = np.arange(0, 32, 1.0)
ax.hist(dur1, bins=bins, color="#7f7f7f", alpha=0.75, label=f"v1 (median {np.median(dur1):.1f} s)")
ax.hist(dur2, bins=bins, color="#1f77b4", alpha=0.75, label=f"v2 (median {np.median(dur2):.1f} s)")
ax.set_xlabel("track duration [s]")
ax.set_ylabel("tracks")
ax.set_title("this bag: track durations", loc="left")
ax.legend(loc="upper right")

ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
txt = ("9 evaluation bags, scored\nagainst ped3d in camera view\n\n"
       "                    v1     v2\n"
       "median length    0.9 s  4.0 s\n"
       "id changes per\n"
       "  ped3d person    0.98   0.41\n"
       "swaps             289    305\n"
       "id errors       ~1950  ~1000")
ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=11.5)
fig.suptitle("Lidar pedestrian tracking: greedy linking vs Kalman tracking (utown_1, 44 s)", x=0.01, ha="left",
             fontsize=14, y=0.985)
fig.savefig(FIG / "11_tracking_v1_v2.png", dpi=DPI, bbox_inches="tight")
print("tracks v1", len(T1), "v2", len(T2), "median dur", np.median(dur1), np.median(dur2))
