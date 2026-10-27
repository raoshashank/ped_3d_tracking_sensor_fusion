#!/usr/bin/env python3
"""Figure 12: identity timeline of the fused, camera-verified persons of the demo bag. One row per
person: bar colour = which sensors measured it at that moment (lidar at this scan and/or a ped3d camera
detection within +-0.13 s; otherwise filled by the smoother), yellow band = inside the camera view. Rows are annotated with how many
ped3d and lidar tracks were joined into the person."""
import numpy as np

from tf_common import C_SOURCE, DPI, FIG, OP, rows, style

plt = style()
R = rows(OP / "fused_tracks.csv")
M = rows(OP / "fused_track_members.csv")
t0 = min(float(r["t"]) for r in R)
persons = {}
for r in R:
    if r["label"] != "camera_verified":
        continue
    persons.setdefault(int(r["person"]), []).append((float(r["t"]) - t0, r["source"], int(r["in_fov"])))
members = {}
for m in M:
    members.setdefault(int(m["person"]), {"lidar": set(), "ped3d": set()})[m["kind"]].add(m["member_id"])

order = sorted(persons, key=lambda p: min(x[0] for x in persons[p]))


def runs(ts, vals):
    """[(start, end, value)] of consecutive equal values on the 10 Hz grid."""
    out = []
    for t, v in zip(ts, vals):
        if out and out[-1][2] == v and t - out[-1][1] < 0.15:
            out[-1][1] = t
        else:
            out.append([t, t, v])
    return out


fig, ax = plt.subplots(figsize=(13.2, 0.33 * len(order) + 2.2))
ylabels, n_cross, exits = [], 0, []
for row, pid in enumerate(order):
    rr = sorted(persons[pid])
    ts = np.array([x[0] for x in rr])
    src = np.array([x[1] for x in rr])
    # coverage per 10 Hz row: lidar measured at this scan; ped3d (4 Hz camera) within +-0.13 s
    lid = np.isin(src, ["lidar", "both"])
    pt = ts[np.isin(src, ["ped3d", "both"])]
    ped = np.array([len(pt) > 0 and np.min(np.abs(pt - t)) <= 0.13 for t in ts])
    cov = np.where(lid & ped, "both", np.where(lid, "lidar", np.where(ped, "ped3d", "filled")))
    for a, b, v in runs(ts, [x[2] for x in rr]):
        if v:
            ax.broken_barh([(a - 0.05, b - a + 0.1)], (row - 0.45, 0.9), color="#fff0a8", zorder=1)
    for a, b, v in runs(ts, list(cov)):
        ax.broken_barh([(a - 0.05, b - a + 0.1)], (row - 0.25, 0.5), color=C_SOURCE[v], zorder=2)
    fov = np.array([x[2] for x in rr])
    out_after = (fov[:-1] == 1) & (fov[1:] == 0)
    if out_after.any():
        k = int(np.argmax(out_after))
        if ts[-1] - ts[k] >= 1.0:
            n_cross += 1
            exits.append((ts[-1] - ts[k], row, ts[k], pid))
    mm = members.get(pid, {"lidar": set(), "ped3d": set()})
    ylabels.append(f"P{pid}  ({len(mm['ped3d'])} ped3d, {len(mm['lidar'])} lidar)")
for (dur, row, tx, pid), box in zip(sorted(sorted(exits, reverse=True)[:2], key=lambda e: e[1]),
                                    [(30.5, 3.6), (30.0, 12.4)]):
    ax.annotate(f"P{pid} leaves the camera view here;\nthe lidar keeps the same id for {dur:.0f} s", xy=(tx, row),
                xytext=box, fontsize=10.5, arrowprops=dict(arrowstyle="->", lw=1.2),
                bbox=dict(boxstyle="round", fc="white", ec="0.5"), zorder=5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(ylabels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("time in the recording [s]")
ax.set_xlim(0, 44)
ax.grid(axis="x", alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_SOURCE["both"], label="lidar + ped3d"),
                   Patch(color=C_SOURCE["lidar"], label="lidar only"),
                   Patch(color=C_SOURCE["ped3d"], label="ped3d only"),
                   Patch(color=C_SOURCE["filled"], label="filled by the smoother"),
                   Patch(color="#fff0a8", label="inside the camera view")],
          loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=5, frameon=False)
ax.set_title(f"Fused camera-verified persons ({len(order)}): {n_cross} keep their id >= 1 s after leaving the camera view",
             loc="left")
fig.savefig(FIG / "12_timeline.png", dpi=DPI, bbox_inches="tight")
print("persons", len(order), "cross", n_cross)
