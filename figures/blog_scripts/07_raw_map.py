"""07: raw accumulated SLAM map (all scans stacked): pedestrians leave smears along the walkway."""
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from blog_common import *  # noqa

ts, ps, _ = slam_poses()
raw = read_pcd(S / "map.pcd")
sl, h = height_slice(raw, 0.35, 2.0)   # below ~0.35 m the ground's slope shows up as rings
o = np.argsort(h)
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axs[0]
sc = ax.scatter(sl[o][::2, 0], sl[o][::2, 1], s=0.08, c=h[o][::2], cmap="viridis", vmin=0.35, vmax=2.0, rasterized=True)
ax.plot(ps[:, 0], ps[:, 1], "k-", lw=2, label="robot path")
ax.add_patch(plt.Rectangle((2, -8), 20, 12, fill=False, ec="red", lw=1.8))
ax.set_aspect("equal"); ax.set_xlim(-15, 40); ax.set_ylim(-22, 16)
ax.set_title("(a) raw map: all 437 scans stacked (0.35-2 m above ground)"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.legend(loc="lower left")
ax = axs[1]
m = (sl[o][:, 0] > 2) & (sl[o][:, 0] < 22) & (sl[o][:, 1] > -8) & (sl[o][:, 1] < 4)
q, hq = sl[o][m], h[o][m]
ax.scatter(q[:, 0], q[:, 1], s=0.4, c=hq, cmap="viridis", vmin=0.35, vmax=2.0, rasterized=True)
ax.plot(ps[:, 0], ps[:, 1], "k-", lw=2)
ax.set_aspect("equal"); ax.set_xlim(2, 22); ax.set_ylim(-8, 4)
ax.set_title("(b) zoom on the walkway"); ax.set_xlabel("x [m]")
ax.annotate("group standing\nand talking", xy=(9.6, 0.9), xytext=(3.0, 2.2), arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=11, bbox=dict(fc="white", ec="0.5"))
ax.annotate("trails of people walking past", xy=(12, -3.3), xytext=(11.0, -7.4), arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=11, bbox=dict(fc="white", ec="0.5"))
cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.01); cb.set_label("height above ground [m]")
fig.savefig(FIG / "07_raw_map.png", bbox_inches="tight")
print(f"raw map {len(raw)} points, {len(sl)} in 0.35-2 m")
