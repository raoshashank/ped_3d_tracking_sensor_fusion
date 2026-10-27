#!/usr/bin/env python3
"""Figure 01: what we start from in the camera: detector/tracker boxes, then masks + projected lidar."""
import matplotlib.patches as patches
import numpy as np

from _blur import blur
from _camera_common import FIG, FRAME, plt, project_lidar, traj

T = traj()
k = FRAME
img = blur(np.ascontiguousarray(T.image(k)[:, :, ::-1]), f"proc_{k}")   # privacy blur before overlays
mask = T.mask(k)
boxes, ids = T.boxes(k)
u, v, rng = project_lidar(T, k)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for a in axs:
    a.imshow(img)
    a.set_xticks([]); a.set_yticks([])
    for (x, y, w, h), tid in zip(boxes, ids):
        a.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec="yellow", lw=1.4))
        a.text(x, y - 3, str(tid), color="black", fontsize=8, bbox=dict(fc="yellow", ec="none", pad=0.8))
axs[0].set_title("(a) camera frame + tracker boxes (ids)")
if mask is not None:
    ov = np.zeros((*mask.shape, 4))
    ov[mask] = [0.0, 0.9, 1.0, 0.35]
    axs[1].imshow(ov)
sc = axs[1].scatter(u, v, c=rng, s=3, cmap="turbo", vmin=0, vmax=25)
axs[1].set_title("(b) + person masks (cyan) + projected lidar")
for a in axs:
    a.set_xlim(0, T.W); a.set_ylim(T.H, 0)
cb = fig.colorbar(sc, ax=axs[1], fraction=0.025, pad=0.01)
cb.set_label("lidar range [m]")
fig.tight_layout()
fig.savefig(FIG / "01_inputs_camera.png", bbox_inches="tight")
print("saved; boxes", len(boxes), "lidar pts in image", len(u))
