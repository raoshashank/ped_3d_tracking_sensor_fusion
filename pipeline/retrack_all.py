#!/usr/bin/env python3
"""Re-run lidar detection + the v2 tracker (lidar_tracker.py) on every finished bag's final map.
Maps are not rebuilt. Writes <bag>/slam_output/erasor2/output_ped/RETRACK_V2 when done (skipped next time).
Usage: python3 retrack_all.py [--force]"""
import shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path("/media/dataset_access/go2nus_annotated")
T = Path(__file__).resolve().parent / "template"
PY = "/home/shashank/mambaforge/envs/erasor2/bin/python"
force = "--force" in sys.argv
bags = sorted(d for d in ROOT.iterdir() if (d / "slam_output/erasor2/output_ped/static_map_slam_frame.pcd").exists())
for k, d in enumerate(bags, 1):
    e, o = d / "slam_output/erasor2", d / "slam_output"
    mark = e / "output_ped/RETRACK_V2"
    if mark.exists() and not force:
        continue
    for f in ("lidar_ped_detect.py", "lidar_tracker.py"):
        shutil.copy(T / "erasor2" / f, e / f)
    shutil.copy(T / "replay_detections.py", o / "replay_detections.py")
    t0 = time.time()
    r = subprocess.run(["taskset", "-c", "0-15", "nice", "-n", "10", PY, "lidar_tracker.py"], cwd=e,
                       env={"PYTHONPATH": str(e), "OMP_NUM_THREADS": "8", "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("v2 smoothed")]
    if r.returncode == 0:
        mark.write_text(line[0] + "\n" if line else "ok\n")
        print(f"[{k}/{len(bags)}] {d.name}: ok {time.time() - t0:.0f}s | {line[0][15:] if line else ''}", flush=True)
    else:
        print(f"[{k}/{len(bags)}] {d.name}: FAILED\n{r.stderr[-600:]}", flush=True)
