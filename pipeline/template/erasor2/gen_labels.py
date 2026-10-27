#!/usr/bin/env python3
"""Per-frame ground (Patchwork++) and instance (HDBSCAN) labels for ERASOR2.

Same output format as ERASOR2/scripts/kitti_clustering.py, with Patchwork++ set up for
the Go2 (lidar ~0.38 m above ground instead of a car's 1.73 m) and frames processed in parallel.
Run with the erasor2 conda env:  python gen_labels.py
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # slam_output: dataset_profile.py
import dataset_profile as DP  # noqa: E402

SEQ = Path(__file__).resolve().parent / "data" / "Merged"
SENSOR_HEIGHT = DP.load(Path(__file__).resolve().parent)["lidar_height"]   # lidar above ground (go2nus 0.38)
GROUND_LABEL = 1       # kitti_clustering.py "V2" convention


_pw = None


def init_worker():
    global _pw
    import pypatchworkpp
    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.sensor_height = SENSOR_HEIGHT
    params.min_range = 0.6
    params.max_range = 80.0
    _pw = pypatchworkpp.patchworkpp(params)


def clusters_hdbscan(points):
    """Same HDBSCAN settings as ERASOR2/scripts/pcd_preprocess.py (whose version breaks on NumPy 2)."""
    import hdbscan
    return hdbscan.HDBSCAN(algorithm="best", alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=True,
                           leaf_size=100, metric="euclidean", min_cluster_size=15,
                           min_samples=None).fit(points).labels_


def label_frame(i):
    pw = _pw
    scan = np.fromfile(SEQ / "velodyne" / f"{i:06d}.bin", dtype=np.float32).reshape(-1, 4)
    pw.estimateGround(scan)
    ground = np.asarray(pw.getGroundIndices(), dtype=np.int64)
    ground_labels = np.zeros(len(scan), dtype=np.uint32)
    ground_labels[ground] = GROUND_LABEL
    ground_labels.tofile(SEQ / "patchwork" / f"{i:06d}.label")

    mask = ground_labels == 0
    pred = np.full(len(scan), -1, dtype=np.int64)
    if mask.sum() > 15:
        pred[mask] = clusters_hdbscan(scan[mask, :3].astype(np.float64))
    tmp = SEQ / "hdbscan" / f"{i:06d}.label.tmp"   # renamed when complete: frames with a label are done
    ((pred + 1) << 16).astype(np.uint32).tofile(tmp)
    os.replace(tmp, SEQ / "hdbscan" / f"{i:06d}.label")
    return len(ground) / len(scan), int(pred.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    (SEQ / "patchwork").mkdir(exist_ok=True)
    (SEQ / "hdbscan").mkdir(exist_ok=True)
    n = len(list((SEQ / "velodyne").glob("*.bin")))
    # A worker can die (this machine: sporadic general protection faults under load); multiprocessing.Pool
    # then hangs forever. ProcessPoolExecutor raises BrokenProcessPool instead: retry the missing frames.
    stats = []
    for attempt in range(4):
        todo = [i for i in range(n) if not (SEQ / "hdbscan" / f"{i:06d}.label").exists()]
        if not todo:
            break
        try:
            with ProcessPoolExecutor(args.workers, initializer=init_worker) as ex:
                stats += list(ex.map(label_frame, todo, chunksize=4))
        except BrokenProcessPool:
            print(f"worker died (attempt {attempt + 1}), retrying {len(todo)} frames", flush=True)
    missing = [i for i in range(n) if not (SEQ / "hdbscan" / f"{i:06d}.label").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} frames without labels after retries")
    if not stats:
        print(f"labelled {n} frames (all existed)")
        return
    g, k = np.array(stats).T
    print(f"labelled {n} frames: ground fraction median {np.median(g):.2f}, "
          f"instances per frame median {int(np.median(k))} (max {int(k.max())})")


if __name__ == "__main__":
    main()
