#!/usr/bin/env python3
"""Figure 15: ros2_leg_detector (mowito, 2D leg detection on a shin-height slice of the lidar) vs
lidar-vs-static-map detections on the demo bag, with ped3d camera detections for reference; recall
of ped3d detections inside the camera view by range (numbers from compare_leg.py on this bag,
v1 lidar detections)."""
import numpy as np

from tf_common import C_IN, C_OUT, C_PED3D, DPI, FIG, OP, arr, draw_map, static_map_slice, style

plt = style()
pts = static_map_slice()
leg = arr(OP / "leg_tracks_map.csv")          # track, t, x, y (map frame)
lid = arr(OP / "lidar_tracks.csv")            # track, scan, t, x, y, top_h, height, n, range, in_fov
ped = arr(OP / "ped3d_tracks_map.csv")        # track, t, x, y

fig = plt.figure(figsize=(13.2, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], hspace=0.28, wspace=0.2, top=0.93)
xl, yl = (-3, 34), (-12.5, 5.5)
ax = fig.add_subplot(gs[0, 0])
draw_map(ax, pts)
ax.scatter(ped[:, 2], ped[:, 3], s=7, c=C_PED3D, lw=0, label="ped3d (camera)", zorder=3)
ax.scatter(leg[:, 2], leg[:, 3], s=5, c="#333333", lw=0, label="ros2_leg_detector people", zorder=4)
ax.set_xlim(*xl); ax.set_ylim(*yl)
ax.set_title(f"ros2_leg_detector /people_tracked: {len(np.unique(leg[:, 0]))} tracks", loc="left")
ax.legend(loc="lower left", markerscale=3, framealpha=0.9)
ax = fig.add_subplot(gs[1, 0])
draw_map(ax, pts)
ax.scatter(ped[:, 2], ped[:, 3], s=7, c=C_PED3D, lw=0, label="ped3d (camera)", zorder=3)
m = lid[:, 9] > 0
ax.scatter(lid[m, 3], lid[m, 4], s=4, c=C_IN, lw=0, label="lidar-vs-map, in camera view", zorder=4)
ax.scatter(lid[~m, 3], lid[~m, 4], s=4, c=C_OUT, lw=0, label="lidar-vs-map, outside", zorder=4)
ax.set_xlim(*xl); ax.set_ylim(*yl)
ax.set_title(f"lidar vs static map: {len(np.unique(lid[:, 0]))} tracks", loc="left")
ax.legend(loc="lower left", markerscale=3, framealpha=0.9)

bx = fig.add_subplot(gs[0, 1])
rng = ["<= 4 m", "<= 6 m", "<= 10 m"]
legr, lidr = [29, 12, 4], [97, 98, 95]
y = np.arange(3)[::-1]
bx.barh(y + 0.19, legr, 0.36, color="#333333", label="ros2_leg_detector")
bx.barh(y - 0.19, lidr, 0.36, color=C_IN, label="lidar vs map")
for yy, a, b in zip(y, legr, lidr):
    bx.text(a + 2, yy + 0.19, f"{a}%", va="center", fontsize=11)
    bx.text(b + 2, yy - 0.19, f"{b}%", va="center", fontsize=11)
bx.set_yticks(y)
bx.set_yticklabels(rng)
bx.set_xlim(0, 120)
bx.set_xticks([0, 25, 50, 75, 100])
bx.set_xlabel("recall of ped3d detections [%]")
bx.set_title("recall inside the camera view", loc="left", fontsize=12)
bx.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=10, frameon=False)
nx = fig.add_subplot(gs[1, 1])
nx.axis("off")
nx.text(0.0, 0.8, "ros2_leg_detector setup:\n- /rslidar_points sliced into a\n  2D scan 0.15-0.45 m above\n  the ground (shin height)\n"
        "- shipped random forest\n  (0.33 deg resolution)\n- joint leg tracker, 10 Hz\n\nA trotting quadruped and a\n16-beam lidar leave few\n"
        "points per leg in one slice;\nthe lidar-vs-map detector\nuses the whole body.",
        va="top", ha="left", fontsize=11)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
fig.suptitle("A 2D leg detector on a 16-beam quadruped lidar vs background subtraction against the static map",
             x=0.01, ha="left", fontsize=14, y=0.985)
fig.savefig(FIG / "15_leg_detector.png", dpi=DPI, bbox_inches="tight")
print("leg tracks", len(np.unique(leg[:, 0])), "lidar tracks", len(np.unique(lid[:, 0])))
