#!/usr/bin/env python3
"""Repair scan timestamps in slam_output/poses_*_tum.txt that break monotonicity (a partial scan with
a stray point timestamp made the old run_slam.py stamp, the min/max midpoint, jump by seconds).
An outlier stamp is replaced by the mean of its neighbours. Prints the repaired bags.
Usage: python3 fix_scan_stamps.py"""
from pathlib import Path
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)
ROOT = Path(p3d_env.P("P3D_BAGS_GO2NUS"))
FILES = ("poses_base_link_tum.txt", "poses_rslidar_tum.txt", "poses_base_link_kiss_only_tum.txt")
for d in sorted(ROOT.iterdir()):
    o = d / "slam_output"
    f0 = o / FILES[0]
    if not f0.exists():
        continue
    t = np.loadtxt(f0)[:, 0]
    if (np.diff(t) > 0).all():
        continue
    t2 = t.copy()
    for _ in range(10):
        if len(t2) > 2 and t2[0] >= t2[1] and t2[1] < t2[2]:      # first scan: extrapolate from the next two
            t2[0] = 2 * t2[1] - t2[2]
        if len(t2) > 2 and t2[-1] <= t2[-2] and t2[-3] < t2[-2]:  # last scan
            t2[-1] = 2 * t2[-2] - t2[-3]
        bad = [i for i in range(1, len(t2) - 1) if not (t2[i - 1] < t2[i] < t2[i + 1]) and t2[i - 1] < t2[i + 1]]
        if not bad:
            break
        for i in bad:
            t2[i] = 0.5 * (t2[i - 1] + t2[i + 1])

    if not (np.diff(t2) > 0).all():
        print(f"{d.name}: could not repair"); continue
    for name in FILES:
        f = o / name
        if not f.exists():
            continue
        hdr = [l for l in open(f) if l.startswith("#")]
        a = np.loadtxt(f)
        a[:, 0] = t2
        np.savetxt(f, a, fmt="%.9f", header="".join(hdr).lstrip("# ").rstrip("\n"))
    print(f"{d.name}: repaired {int((t2 != t).sum())} stamp(s)")
