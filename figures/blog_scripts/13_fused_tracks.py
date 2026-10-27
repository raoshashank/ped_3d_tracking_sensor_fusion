#!/usr/bin/env python3
"""Figure 13: all fused persons of the demo bag, top-down, coloured by label, with the robot path;
persons per label in the legend."""
import numpy as np

from tf_common import C_LABEL, DPI, FIG, LABELS, OP, draw_map, rows, static_map_slice, style

plt = style()
pts = static_map_slice()
R = rows(OP / "fused_tracks.csv")
P = {}
for r in R:
    P.setdefault(int(r["person"]), {"label": r["label"], "xy": []})["xy"].append((float(r["t"]), float(r["x"]), float(r["y"])))

fig, ax = plt.subplots(figsize=(13.4, 7.0))
draw_map(ax, pts)
order = {"unconfirmed_in_fov": 1, "camera_only": 2, "lidar_only": 3, "camera_verified": 4}
for pid, p in sorted(P.items(), key=lambda kv: order[kv[1]["label"]]):
    a = np.array(sorted(p["xy"]))
    ax.plot(a[:, 1], a[:, 2], "-", color=C_LABEL[p["label"]], lw=1.8 if p["label"] == "camera_verified" else 1.2,
            alpha=0.95, zorder=order[p["label"]] + 5)
    ax.plot(a[-1, 1], a[-1, 2], "o", color=C_LABEL[p["label"]], ms=3, zorder=order[p["label"]] + 5)
counts = {lab: sum(p["label"] == lab for p in P.values()) for lab in LABELS}
names = {"camera_verified": "camera verified (lidar + ped3d)", "camera_only": "camera only (ped3d, no lidar)",
         "lidar_only": "lidar only (never near the camera view)", "unconfirmed_in_fov": "unconfirmed (in view, no ped3d)"}
for lab in LABELS:
    ax.plot([], [], "-", color=C_LABEL[lab], lw=2.5, label=f"{names[lab]}: {counts[lab]}")
ax.legend(loc="lower left", framealpha=0.92, ncol=2, title=f"{len(P)} persons")
ax.set_xlim(-3, 44)
ax.set_ylim(-19.5, 6.5)
ax.set_title("Fused pedestrian tracks (one track per person), utown_1, 44 s", loc="left")

fig.savefig(FIG / "13_fused_tracks.png", dpi=DPI, bbox_inches="tight")
print(counts)
