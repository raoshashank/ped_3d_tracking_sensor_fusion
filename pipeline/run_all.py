#!/usr/bin/env python3
"""Static maps with pedestrians removed for every bag of a dataset: the utown_1 pipeline, batched.

Datasets (--dataset, default go2nus): robot/sensor setup in template/dataset_profile.py, written per bag to
slam_output/dataset.json.
  go2nus    /media/dataset_access/go2nus_annotated/<bag>/
  miraikan  /media/dataset_access/miraikan_reconstruction/<bag>/, unzipped on demand from
            /media/dataset_access/ros2_bags_acme_reconstruction/miraikan/<bag>.zip (one map per zip)

Per bag  <root>/<bag>/  ->  <root>/<bag>/slam_output/ :
  1. run_slam.py            KISS-ICP + GTSAM offline SLAM           (kiss_icp env)
  2. export_scans.py        scans + poses in ERASOR2 layout         (kiss_icp env)
  3. gen_labels.py          Patchwork++ ground + HDBSCAN instances  (erasor2 env, 8 workers)
  4. make_camera_model.py   per-session camera extrinsics for the FOV flag (calib env)
  5. run_erasor2 erasor2.yaml            -> output/          ERASOR2-only map
  6. ped_mask.py + run_erasor2 (ped)     -> output_ped/      ped3d tracks masked, map v1
  7. lidar_tracker.py                   360-deg detections against map v1 + KF/Hungarian tracking
  8. ped_mask.py --lidar-tracks + run_erasor2 -> output_ped/ final map
  9. lidar_tracker.py                   final detections + tracks
 10. fuse_tracks.py                     ped3d + lidar -> one track set (fused_tracks.csv); finalize; cleanup; DONE
Bags without a ped3d pickle stop after step 5 (ERASOR2-only map). Group completion (review-only) is
not run. Every step is skipped when its output exists; a failure writes FAILED and moves on.

Usage:  python3 run_all.py [--dataset go2nus|miraikan] [--bags list.txt] [--limit N] [--force] [--dry-run]
Logs: <pipeline dir>/logs/<bag>.log, summary: <pipeline dir>/summary[_<dataset>].csv (rebuilt after each bag).
"""
import argparse
import json
import re
import shutil
import subprocess
import os
import sys
import time
from pathlib import Path

PIPE = Path(__file__).resolve().parent   # /media/dataset_access/ped_3d_detection_pipeline
TEMPLATE = PIPE / "template"
sys.path.insert(0, str(TEMPLATE))
import dataset_profile as DP  # noqa: E402

DATASETS = {
    "go2nus": dict(root=Path("/media/dataset_access/go2nus_annotated"), zips=None, summary="summary.csv"),
    "miraikan": dict(root=Path("/media/dataset_access/miraikan_reconstruction"),
                     zips=Path("/media/dataset_access/ros2_bags_acme_reconstruction/miraikan"),
                     summary="summary_miraikan.csv"),
}
DATASET = "go2nus"   # set by --dataset
ROOT = DATASETS[DATASET]["root"]
ENV = {"kiss": "/home/shashank/mambaforge/envs/kiss_icp/bin/python",
       "erasor2": "/home/shashank/mambaforge/envs/erasor2/bin/python",
       "calib": "/home/shashank/mambaforge/envs/calib/bin/python"}
_ERASOR2_CANDIDATES = [PIPE.parent / "third_party/ERASOR2/build/run_erasor2",   # this repo's submodule build
                       Path.home() / "ERASOR2/build/run_erasor2"]              # stand-alone checkout
ERASOR2_BIN = os.environ.get("ERASOR2_BIN") or str(next((p for p in _ERASOR2_CANDIDATES if p.exists()),
                                                        _ERASOR2_CANDIDATES[-1]))
# this machine (i9-13900KF, Raptor Lake) crashes under all-core load: 16 threads, low priority
WRAP = ["taskset", "-c", "0-15", "nice", "-n", "10"]
MIN_SCANS = 30
STEP_TIMEOUT_S = 3600   # a hung step (e.g. a dead worker) fails the bag instead of stalling the batch


class Bag:
    def __init__(self, d):
        self.dir, self.name = d, d.name
        self.out = d / "slam_output"
        self.e = self.out / "erasor2"
        self.log = PIPE / "logs" / f"{self.name}.log"
        prof = DP.PROFILES[DATASET]
        self.prof = prof
        self.pkl = Path(prof["ped3d_pkl"].format(bag=self.name))
        zips = DATASETS[DATASET]["zips"]
        self.zip = zips / f"{self.name}.zip" if zips else None

    def unzip(self):
        """Unpack <bag>.zip (one top-level dir <bag>/ with the .db3 + metadata.yaml) into ROOT."""
        if self.zip is None or list(self.dir.glob("*.db3")):
            return True
        if not self.zip.exists() or self.zip.stat().st_size == 0:   # download incomplete
            return False
        subprocess.run(["unzip", "-q", "-o", str(self.zip), "-d", str(self.dir.parent)], check=True)
        return bool(list(self.dir.glob("*.db3")))

    def run(self, cmd, cwd, env_extra=None):
        import os
        env = {**os.environ, "OMP_NUM_THREADS": "8", "PYTHONPATH": str(cwd), **(env_extra or {})}
        with open(self.log, "a") as f:
            f.write(f"\n$ {' '.join(map(str, cmd))}   (cwd {cwd})\n")
            f.flush()
            subprocess.run([str(c) for c in WRAP + cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True,
                           timeout=STEP_TIMEOUT_S)

    def n_scans(self):
        return len(list((self.e / "data/Merged/velodyne").glob("*.bin")))

    def write_yamls(self, n):
        t = (TEMPLATE / "erasor2/go2_utown.yaml").read_text()
        t = re.sub(r"end_frame: \d+.*", f"end_frame: {n - 3}", t)
        h = self.prof["lidar_height"]   # heights in the yaml are in the lidar frame (see dataset_profile.py)
        t = re.sub(r"min_z_voi: [-\d.]+", f"min_z_voi: {-(h + 0.62):.3f}", t)
        t = re.sub(r"max_z_voi: [-\d.]+", f"max_z_voi: {DP.CEIL_Z - h:.3f}", t)
        t = re.sub(r"sensor_height: [-\d.]+", f"sensor_height: {h}", t)
        t = t.replace("rerun:\n  enabled: true", "rerun:\n  enabled: false")
        for seq, out in (("Merged", "output"), ("MergedPed", "output_ped")):
            s = t.replace('sequence: "Merged"', f'sequence: "{seq}"')
            s = re.sub(r'abs_data_dir: ".*"', f'abs_data_dir: "{self.e}/data"', s)
            s = re.sub(r'abs_save_dir: ".*"', f'abs_save_dir: "{self.e}/{out}"', s)
            s = re.sub(r'save_path: ".*"', f'save_path: "{self.e}/{out}/erasor2_run.rrd"', s)
            (self.e / f"erasor2{'_ped' if seq == 'MergedPed' else ''}.yaml").write_text(s)

    def erasor2(self, yaml, out):
        (self.e / out).mkdir(exist_ok=True)
        self.run([ERASOR2_BIN, self.e / yaml], self.e)
        self.run([ENV["erasor2"], "finalize_map.py", out, "--rm-lifted"], self.e)

    def process(self, force):
        if (self.out / "DONE").exists() and not force:
            return "done"
        if not self.unzip() or not list(self.dir.glob("*.db3")):   # some dirs name the .db3 differently from the dir
            return "no bag"
        self.out.mkdir(exist_ok=True)
        if DATASET != "go2nus":   # go2nus: no dataset.json = go2nus defaults (bags processed before profiles)
            DP.write(self.out, DATASET, self.name)
        self.e.mkdir(exist_ok=True)
        for f in TEMPLATE.iterdir():
            if f.is_file():
                shutil.copy(f, self.out / f.name)
        for f in (TEMPLATE / "erasor2").iterdir():
            if f.is_file():
                shutil.copy(f, self.e / f.name)
        (self.out / "FAILED").unlink(missing_ok=True)
        t0 = time.time()
        step = "slam"
        try:
            if not (self.out / "poses_rslidar_tum.txt").exists():
                self.run([ENV["kiss"], "run_slam.py", self.dir], self.out)
            step = "export"
            if not (self.e / "data/Merged/poses.txt").exists():
                self.run([ENV["kiss"], "export_scans.py"], self.e)
            n = self.n_scans()
            if n < MIN_SCANS:
                (self.out / "DONE").write_text(f"skipped: only {n} scans\n")
                return f"skipped ({n} scans)"
            step = "labels"
            if len(list((self.e / "data/Merged/hdbscan").glob("*.label"))) != n:
                self.run([ENV["erasor2"], "gen_labels.py", "--workers", "8"], self.e, {"OMP_NUM_THREADS": "1"})
            step = "camera"
            has_ped3d = self.pkl.exists()
            if has_ped3d and not (self.e / "camera_model.npz").exists():
                self.run([ENV["calib"], "make_camera_model.py", self.out], self.out)
            self.write_yamls(n)
            step = "erasor2"
            if not (self.e / "output/static_map_slam_frame.pcd").exists():
                self.erasor2("erasor2.yaml", "output")
            if has_ped3d:
                step = "ped_mask"
                if not (self.e / "output_ped/lidar_tracks.csv").exists():
                    self.run([ENV["erasor2"], "ped_mask.py"], self.e)
                    self.erasor2("erasor2_ped.yaml", "output_ped")
                    step = "detect"
                    self.run([ENV["erasor2"], "lidar_tracker.py"], self.e)
                step = "ped_mask+lidar"
                if not (self.out / "DONE").exists():
                    self.run([ENV["erasor2"], "ped_mask.py", "--lidar-tracks"], self.e)
                    self.erasor2("erasor2_ped.yaml", "output_ped")
                    step = "detect_final"
                    self.run([ENV["erasor2"], "lidar_tracker.py"], self.e)
                    step = "fuse"
                    self.run([ENV["erasor2"], "fuse_tracks.py"], self.e)
                    self.run([ENV["erasor2"], "-c", "from ped_tracks import load_tracks\n"
                              "tr=load_tracks('fused')\n"
                              "f=open('output_ped/ped3d_tracks_map.csv','w'); f.write('track,t,x,y\\n')\n"
                              "[f.write(f'{k},{tt:.3f},{x:.3f},{y:.3f}\\n') for k,(t,xy,v) in tr.items() for tt,(x,y) in zip(t,xy)]"], self.e)
                    self.run([ENV["erasor2"], "finalize_map.py", "output"], self.e)   # re-score with ped labels
            step = "cleanup"
            shutil.rmtree(self.e / "data/MergedPed", ignore_errors=True)
            for out in ("output", "output_ped"):
                shutil.rmtree(self.e / out / "mos", ignore_errors=True)
                for f in (self.e / out).glob("*.rrd"):
                    f.unlink()
            shutil.rmtree(self.e / "__pycache__", ignore_errors=True)
            (self.out / "DONE").write_text(f"ped3d={'yes' if has_ped3d else 'no'} scans={n} seconds={time.time() - t0:.0f}\n")
            return f"ok ({n} scans, {time.time() - t0:.0f} s{'' if has_ped3d else ', no ped3d'})"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as ex:
            (self.out / "FAILED").write_text(f"step {step}: {ex}\n")
            return f"FAILED at {step} (see {self.log})"


def summary():
    rows = ["bag,status,scans,seconds,ped3d,points_erasor2,points_final,ped3d_residue_erasor2,ped3d_residue_final,raw_removed_final,slam_qc_trans_m,slam_qc_yaw_deg,slam_qc_flag"]
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or not (d / "slam_output").exists():
            continue
        b = Bag(d)
        st = "done" if (b.out / "DONE").exists() else "FAILED" if (b.out / "FAILED").exists() else "todo"
        info = dict(kv.split("=") for kv in (b.out / "DONE").read_text().split() if "=" in kv) if st == "done" else {}
        s = json.load(open(b.e / "summary.json")) if (b.e / "summary.json").exists() else {}
        o, p = s.get("output", {}), s.get("output_ped", {})
        q = json.load(open(b.out / "slam_qc.json")) if (b.out / "slam_qc.json").exists() else {}
        rows.append(",".join(str(v) for v in [d.name, st, info.get("scans", ""), info.get("seconds", ""), info.get("ped3d", ""),
                                               o.get("points", ""), p.get("points", ""), o.get("ped3d_residue", ""),
                                               p.get("ped3d_residue", ""), p.get("raw_voxels_removed", ""),
                                               round(q["max_window_trans_m"], 2) if q else "", round(q["max_window_yaw_deg"], 1) if q else "",
                                               int(q["flag"]) if q else ""]))
    (PIPE / DATASETS[DATASET]["summary"]).write_text("\n".join(rows) + "\n")


def main():
    global DATASET, ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="go2nus", choices=sorted(DATASETS))
    ap.add_argument("--bags", help="file with bag dir names, one per line (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    DATASET, ROOT = a.dataset, DATASETS[a.dataset]["root"]
    ROOT.mkdir(exist_ok=True)
    zips = DATASETS[DATASET]["zips"]
    if a.bags:
        names = [l.strip() for l in open(a.bags) if l.strip()]
    elif zips:
        names = sorted(z.stem for z in zips.glob("*.zip"))
    else:
        names = sorted(d.name for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if a.limit:
        names = names[:a.limit]
    (PIPE / "logs").mkdir(exist_ok=True)
    for k, name in enumerate(names, 1):
        b = Bag(ROOT / name)
        if a.dry_run:
            print(name, "done" if (b.out / "DONE").exists() else "todo", "ped3d" if b.pkl.exists() else "no-ped3d")
            continue
        print(f"[{k}/{len(names)}] {name}: ", end="", flush=True)
        print(b.process(a.force), flush=True)
        summary()


if __name__ == "__main__":
    main()
