"""09: people standing still look static to ERASOR2; holding ped3d standing tracks removes them."""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from blog_common import *  # noqa

D = E / "data" / "Merged"
P3 = np.loadtxt(D / "poses.txt").reshape(-1, 3, 4)
acc = []
for i in range(0, len(P3) - 1, 2):
    sc = np.fromfile(D / "velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    acc.append(sc @ P3[i, :, :3].T + P3[i, :, 3])
raw = np.vstack(acc)
maps = [("raw accumulation", raw), ("ERASOR2 alone", read_pcd(E / "output" / "static_map_slam_frame.pcd")),
        ("final (pedestrians masked)", read_pcd(E / "output_ped" / "static_map_slam_frame.pcd"))]
X0, X1, Y0, Y1 = 7.5, 12.5, -1.5, 2.5

from _blur import blur   # noqa: E402  (mask cached by the calib env: python _blur.py <img> proc_20)
img = Image.fromarray(blur(np.asarray(Image.open(PROC / "imgs" / "20.jpg").convert("RGB")), "proc_20"))
crop = img.crop((150, 95, 560, 305))

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 0.85], hspace=0.32, wspace=0.12)
ax = fig.add_subplot(gs[0, :])
ax.imshow(crop); ax.axis("off")
ax.set_title("(a) camera at t = 6 s: a group standing and talking on open pavement\n(the pillar with the sign box is at the top right of panels b-d)")
for k, (name, p) in enumerate(maps):
    h = p[:, 2] - GROUND_Z
    m = (p[:, 0] > X0) & (p[:, 0] < X1) & (p[:, 1] > Y0) & (p[:, 1] < Y1) & (h > 0.05) & (h < 2.2)
    q, hq = p[m], h[m]
    o = np.argsort(hq)
    ax = fig.add_subplot(gs[1, k])
    sc = ax.scatter(q[o, 0], q[o, 1], s=0.3, c=hq[o], cmap="viridis", vmin=0, vmax=2.0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    ax.set_title(f"({'bcd'[k]}) {name}")
    if k == 0:
        ax.set_ylabel("top view, y [m]")
    else:
        ax.set_yticklabels([])
    ax2 = fig.add_subplot(gs[2, k])
    ms = m & (np.abs(p[:, 1] - 0.5) < 1.2)
    ax2.scatter(p[ms, 0], h[ms], s=0.3, c=h[ms], cmap="viridis", vmin=0, vmax=2.0, rasterized=True)
    ax2.set_xlim(X0, X1); ax2.set_ylim(0, 2.2); ax2.set_xlabel("x [m]")
    ax2.grid(alpha=0.3)
    if k == 0:
        ax2.set_ylabel("side view, height [m]")
    else:
        ax2.set_yticklabels([])
cb = fig.colorbar(sc, ax=fig.axes[1:], shrink=0.5, pad=0.01); cb.set_label("height above ground [m]")
fig.savefig(FIG / "09_standing_group.png", bbox_inches="tight")
for name, p in maps:
    h = p[:, 2] - GROUND_Z
    near = (np.hypot(p[:, 0] - 10.0, p[:, 1] - 0.6) < 1.5) & (h > 1.0) & (h < 2.2)
    print(f"{name}: {near.sum()} points 1-2.2 m high within 1.5 m of (10, 0.6)")
