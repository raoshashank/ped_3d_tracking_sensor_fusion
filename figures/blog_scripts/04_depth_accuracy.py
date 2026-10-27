#!/usr/bin/env python3
"""Figure 04: camera depth sources vs lidar depth at pedestrians, all go2nus ped3d results.
Reference: ped3d's VGGT-guided lidar depth (depth_lidar_g) where it has >= 10 lidar points.
Note: vggt_a is anchored per track to that same lidar, so its error is optimistic (in-sample)."""
import glob
import pickle
from multiprocessing import Pool

import numpy as np

from _camera_common import C, FIG, SRC_LABEL, plt

SRCS = ["vggt", "vggt_c", "vggt_a", "da"]


def one(path):
    d = pickle.load(open(path, "rb"))
    rows = []
    for dets in d["frames"]:
        for det in dets:
            dl, n = det.get("depth_lidar_g", np.nan), det.get("n_lidar_g", 0)
            if not (np.isfinite(dl) and n >= 10):
                continue
            tr = d["tracks"].get(det["id"])
            anchor = float(tr.get("anchor", 1.0)) if tr else 1.0
            v = dict(vggt=det["depth_vggt"], vggt_c=det["depth_vggt_c"], vggt_a=det["depth_vggt_c"] * anchor, da=det["depth_da"])
            rows.append([dl] + [abs(v[s] - dl) if np.isfinite(v[s]) else np.nan for s in SRCS])
    return rows


if __name__ == "__main__":
    files = sorted(glob.glob("/media/shashank/ssd2/ped3d/go2nus/*_0.pkl"))
    with Pool(6) as p:
        rows = [r for rr in p.map(one, files, chunksize=8) for r in rr]
    a = np.array(rows)
    bins = np.arange(1, 16)
    fig, ax = plt.subplots(figsize=(12, 5))
    mid = bins[:-1] + 0.5
    print(f"{len(files)} trajectories, {len(a)} pedestrian detections with >= 10 lidar points")
    for j, s in enumerate(SRCS, start=1):
        med = [np.nanmedian(a[(a[:, 0] >= lo) & (a[:, 0] < lo + 1), j]) for lo in bins[:-1]]
        ax.plot(mid, med, "-o", color=C[s], lw=2, ms=5, label=SRC_LABEL[s])
        m10 = a[:, 0] <= 10
        print(f"  {s:7s} median |error| <= 10 m: {np.nanmedian(a[m10, j]):.3f} m   (1-5 m {np.nanmedian(a[(a[:, 0] < 5), j]):.2f}, 5-10 m {np.nanmedian(a[m10 & (a[:, 0] >= 5), j]):.2f}, 10-15 m {np.nanmedian(a[(a[:, 0] > 10) & (a[:, 0] <= 15), j]):.2f})")
    cnt = np.histogram(a[:, 0], bins)[0]
    ax2 = ax.twinx()
    ax2.bar(mid, cnt, width=0.8, color="0.85", zorder=0)
    ax2.set_ylabel("detections per bin", color="0.5")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xlabel("pedestrian range (lidar) [m]")
    ax.set_ylabel("median |depth - lidar depth| [m]")
    ax.set_title(f"Camera depth vs lidar at pedestrians, all go2nus ({len(files)} trajectories, {len(a):,} detections)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "04_depth_accuracy.png", bbox_inches="tight")
