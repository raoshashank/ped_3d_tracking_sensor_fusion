#!/usr/bin/env python3
"""Run fuse_tracks.py (ped3d + lidar -> one track set) on every finished bag; also refreshes the
replay script and RViz config. Writes output_ped/FUSED_V2 when done (skipped next time).
Usage: python3 fuse_all.py [--force]"""
import shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path("/media/dataset_access/go2nus_annotated")
T = Path(__file__).resolve().parent / "template"
PY = "/home/shashank/mambaforge/envs/erasor2/bin/python"
force = "--force" in sys.argv
bags = sorted(d for d in ROOT.iterdir() if (d / "slam_output/erasor2/output_ped/lidar_tracks_smooth.csv").exists())
for k, d in enumerate(bags, 1):
    e, o = d / "slam_output/erasor2", d / "slam_output"
    mark = e / "output_ped/FUSED_V2"
    if mark.exists() and not force:
        continue
    for f in ("fuse_tracks.py", "lidar_tracker.py"):   # fuse_tracks uses lidar_tracker's smoother
        shutil.copy(T / "erasor2" / f, e / f)
    for f in ("replay_detections.py", "detections_replay.rviz"):
        shutil.copy(T / f, o / f)
    t0 = time.time()
    r = subprocess.run(["taskset", "-c", "0-15", "nice", "-n", "10", PY, "fuse_tracks.py"], cwd=e,
                       env={"PYTHONPATH": str(e), "OMP_NUM_THREADS": "8", "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("[all]")]
    if r.returncode == 0:
        mark.write_text("\n".join(lines) + "\n")
        print(f"[{k}/{len(bags)}] {d.name}: ok {time.time() - t0:.0f}s | {lines[-1][6:] if lines else ''}", flush=True)
    else:
        print(f"[{k}/{len(bags)}] {d.name}: FAILED\n{r.stderr[-800:]}", flush=True)
