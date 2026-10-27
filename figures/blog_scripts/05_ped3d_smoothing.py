#!/usr/bin/env python3
"""Figure 05: ped3d per-frame fused measurements vs the robust constant-velocity RTS-smoothed tracks
(odometry frame), gap-filled parts dashed, down-weighted outliers marked x."""
import numpy as np

from _camera_common import FIG, ped3d, plt, traj

T = traj()
d = ped3d()
t0 = d["timestamps"][0]
TRACKS = [79, 42, 64]
COLS = ["#2ca02c", "#1f77b4", "#9467bd"]


def segments(mask):
    """runs of True as (start, end) index pairs (inclusive)."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def draw(ax, tr, col, xs, ys, label):
    ok = np.isfinite(tr["fused"][:, 0])
    f, m = tr["fused"], tr["fused_meas"]
    X, Y = xs(tr, f), ys(tr, f)
    ax.plot(np.where(ok, X, np.nan), np.where(ok, Y, np.nan), "-", color=col, lw=2.2, label=label)
    interp = ok & tr["interpolated"]
    for a, b in segments(interp):
        a, b = max(a - 1, 0), min(b + 1, len(X) - 1)
        ax.plot(X[a:b + 1], Y[a:b + 1], "--", color="white", lw=2.2)
        ax.plot(X[a:b + 1], Y[a:b + 1], ":", color=col, lw=2.2)
    good = np.isfinite(m[:, 0]) & ~tr["outlier"]
    bad = np.isfinite(m[:, 0]) & tr["outlier"]
    ax.scatter(xs(tr, m)[good], ys(tr, m)[good], s=16, color=col, alpha=0.55, edgecolors="none")
    ax.scatter(xs(tr, m)[bad], ys(tr, m)[bad], s=40, marker="x", color="k", lw=1.5)


fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
a = fig.add_subplot(gs[:, 0])
pos = T.robot_pos[:T.n]
a.plot(pos[:, 0], pos[:, 1], "k-", lw=1.5, label="robot (odometry)")
for tid, col in zip(TRACKS, COLS):
    draw(a, d["tracks"][tid], col, lambda tr, v: v[:, 0], lambda tr, v: v[:, 1], f"ped3d track {tid}")
a.scatter([], [], s=16, color="0.4", label="per-frame fused measurement")
a.scatter([], [], s=40, marker="x", color="k", label="outlier (down-weighted)")
a.plot([], [], ":", color="0.3", lw=2.2, label="gap filled by the smoother")
a.set_aspect("equal"); a.set_xlabel("x [m] (odometry frame)"); a.set_ylabel("y [m]")
a.set_title("(a) raw per-frame positions vs smoothed tracks")
a.legend(loc="upper right", fontsize=10, framealpha=0.95)
a.set_xlim(26, 53)

tid = TRACKS[0]
tr = d["tracks"][tid]
tt = lambda tr, v: d["timestamps"][tr["frames"]] - t0  # noqa: E731
ax0 = None
for j, lab in enumerate(["x", "y"]):
    a = fig.add_subplot(gs[j, 1], sharex=ax0)
    ax0 = ax0 or a
    draw(a, tr, "#2ca02c", tt, lambda tr, v, j=j: v[:, j], None)
    a.set_ylabel(f"{lab} [m]")
    a.grid(alpha=0.3)
    if j == 0:
        a.set_title(f"(b) track {tid} over time (the person walks away; far frames are noisy)")
        a.tick_params(labelbottom=False)
    else:
        a.set_xlabel("time [s]")
fig.tight_layout()
fig.savefig(FIG / "05_ped3d_smoothing.png", bbox_inches="tight")
for t in TRACKS:
    tr = d["tracks"][t]
    ok = np.isfinite(tr["fused"][:, 0])
    print(t, "frames", ok.sum(), "interpolated", int((tr["interpolated"] & ok).sum()), "outliers", int(tr["outlier"].sum()))
