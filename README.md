# ped_3d_tracking_sensor_fusion

Offline 3D pedestrian tracking and static-scene reconstruction from mobile-robot bags
(go2nus Unitree Go2 recordings, Miraikan CaBot recordings). Per bag the pipeline produces a
static point-cloud map with pedestrians removed and one fused set of pedestrian tracks
(camera + lidar, 360 degrees) in the SLAM map frame.

Method and data formats: `docs/PEDESTRIAN_3D.md`. Batch driver: `docs/batch_pipeline.md`.
Step-by-step notes from the development bag (utown_1): `docs/erasor2_utown1_dev_notes.md`.

```
bag (.db3)
 ├─ camera frames + 2D tracks (REGROUP) + lidar ──► ped3d (VGGT-Omega depth + lidar) ──► ped3d tracks (camera FOV)
 ├─ lidar + odom ──► offline SLAM (KISS-ICP + GTSAM) ──► poses, raw map
 │                        ▼
 │        Patchwork++/HDBSCAN labels ──► ERASOR2 + pedestrian masking ──► static map
 └─ lidar ──► residual vs static map ──► lidar detections ──► lidar tracks (360 deg)
                                                      ▼
                                             fusion ──► fused_tracks.csv per bag
```

## Layout

| path | what | runs in |
|---|---|---|
| `pipeline/run_all.py` | batch driver over all bags: SLAM, labels, ERASOR2, ped3d masking, lidar detection/tracking, fusion, final map. `--dataset miraikan` for CaBot bags. | system python3 (calls the envs below) |
| `pipeline/fuse_all.py`, `retrack_all.py`, `qc_slam_all.py`, `fix_scan_stamps.py`, `redo_stamp_fixed.sh` | re-run single stages over all bags; SLAM quality check; scan-stamp repair | |
| `pipeline/template/` | scripts copied into every `<bag>/slam_output/`: `run_slam.py` (SLAM), `dataset_profile.py` (sensor setup per dataset), `make_camera_model.py`, RViz replays (`replay_detections.py`, `replay_with_map.py`, `*.rviz`) | kiss_icp / calib / ROS 2 |
| `pipeline/template/erasor2/` | copied into `<bag>/slam_output/erasor2/`: `export_scans.py`, `gen_labels.py`, `ped_tracks.py`, `ped_mask.py`, `lidar_ped_detect.py`, `lidar_tracker.py`, `fuse_tracks.py`, `finalize_map.py`, `ped_groups.py`, `go2_utown.yaml` (ERASOR2 config template) | kiss_icp (export) / erasor2 (rest) |
| `pipeline/groups_eval/` | evaluation of `ped_groups.py` (3D social grouping) against REGROUP | erasor2 |
| `pipeline/*.txt`, `pipeline/summary*.csv` | bag lists used by the batch scripts; per-bag summary tables of the last batch run (2026-09/10) | |
| `ped3d/` | camera-based pedestrian 3D estimation: `ped3d_lib.py`, `run_ped3d.py`, `run_all_ped3d.sh`, `relink.py`, `export_tracks.py`, `ped_heatmap.py`, `app.py` (Streamlit viewer), extrinsics estimation for other datasets, `tracking/` (2D-track evaluation) | calib |
| `vggt_depth/` | VGGT-Omega metric depth for the camera frames (`run_vggt_omega.py`, `generate_vggt_omega_depth*.py`, `queue_vggt_omega.sh`), depth accuracy checks | calib (GPU) |
| `eval/` | development-bag evaluation scripts, run from inside `<bag>/slam_output/erasor2/` (`eval_ped.py`, `recall_eval.py`, `compare_leg.py`) or `<bag>/slam_output/` (`map_sharpness.py`) | erasor2 |
| `baselines/leg_detector/` | ros2_leg_detector comparison: launch file, RViz config, `record_leg_tracks.py`, and the patch applied to the upstream package | ROS 2 Humble |
| `baselines/lvdot_eval/` | LV-DOT comparison scripts | calib |
| `regroup/` | the 2D front-end (YOLO + BoT-SORT + REGROUP grouping) that produces `regroup_tracks*.pkl`: patch against upstream `regroup-hri` plus the added scripts and Docker files (`added/`) | Docker |
| `figures/` | paper figure (`paper_fig/fig_pipeline.py`) and the blog figure scripts it imports (`blog_scripts/`) | erasor2 / calib |
| `envs/` | conda environments: `<env>.yml` (`conda env export --from-history`) and `<env>_pip_freeze.txt` | |
| `p3d_env.py`, `config/paths.env.example` | machine-specific paths: every script reads `P3D_*` variables; copy the example to `config/paths.env` (git-ignored) and edit it | |
| `docs/` | pipeline documentation (see above) | |
| `third_party/` | submodules, see below | |

## Third-party code (submodules)

| submodule | upstream | pinned | role | local changes |
|---|---|---|---|---|
| `third_party/ERASOR2` | github.com/url-kaist/ERASOR2, mirrored at `raoshashank/ERASOR2_minimal` | `d43d94f` | dynamic-object removal; builds `run_erasor2` (ROS-free, catkin-free upstream version) | none; the mirror only protects against upstream history changes |
| `third_party/vggt-omega` | github.com/facebookresearch/vggt-omega | `282ec70` | metric multi-view depth (`vggt_omega` package, weights from HF `facebook/VGGT-Omega`) | none |
| `third_party/regroup-hri` | github.com/UCSD-RHC-Lab/regroup-hri | `f692f50` | 2D person tracking + grouping | `regroup/regroup-hri_local_changes.patch` + files in `regroup/added/` |
| `third_party/ros2_leg_detector` | github.com/mowito/ros2_leg_detector | `b4be35a` | baseline only | `baselines/leg_detector/ros2_leg_detector_humble.patch` |
| `third_party/lv-dot` | github.com/zhefan-xu/lv-dot | `449bf2c` | baseline only | none |
| `third_party/sbpd-dataset-pipeline` | github.com/social-navigation-group/sbpd-dataset-pipeline | `9ca27c8` (remote HEAD) | bag extraction that writes the per-trajectory frames, lidar scans and `traj_data.pkl` under `$P3D_PROCESSED/<dataset>/` (`robot_data_analysis/data_sampling/process_bags_ros2.py`) | the workstation's local checkout is at `5b00be2` with uncommitted edits; commit and push there, then re-pin |

Other dependencies are plain packages (see `envs/`): KISS-ICP 1.3.0, GTSAM, pypatchworkpp, hdbscan,
open3d, rosbags, ultralytics, huggingface_hub, torch. ROS 2 Humble is needed only for the RViz replays
and the leg-detector baseline.

## Setup

```bash
git clone --recurse-submodules <this repo>
cd ped_3d_tracking_sensor_fusion
```

Paths: copy `config/paths.env.example` to `config/paths.env` and fill in the data roots and the conda
env directory for this machine (variables below). Scripts inside the repo load that file through
`p3d_env.py`; `run_all.py` exports the variables to every step it runs, so the scripts it copies into
`<bag>/slam_output/` need nothing else. To run one of those copied scripts by hand, `source
config/paths.env` first.

ERASOR2 submodule: `.gitmodules` points at `git@github.com:raoshashank/ERASOR2_minimal.git`, a mirror
of upstream `d43d94f`. Build it once:

```bash
git submodule update --init third_party/ERASOR2
cd third_party/ERASOR2 && cmake -B build -S . && cmake --build build -j
```

`pipeline/run_all.py` uses `$P3D_ERASOR2_BIN` if set, else `third_party/ERASOR2/build/run_erasor2`.

Environments (`envs/`): `kiss_icp` (python 3.11: SLAM, scan export), `erasor2` (python 3.12: labels,
masking, lidar detection/tracking, fusion, grouping), `calib` (python 3.10: ped3d, VGGT-Omega,
`make_camera_model.py`; this env also picks up packages from `~/.local/lib/python3.10/site-packages`).
`vggt_omega` is installed in `calib` as an editable package: `pip install -e third_party/vggt-omega`.
Interpreters are resolved as `$P3D_CONDA_ENVS/<env>/bin/python` (override one with `P3D_PY_<ENV>`, e.g. `P3D_PY_CALIB`).

Apply the local patches when rebuilding a baseline or the 2D front-end:

```bash
git -C third_party/regroup-hri apply ../../regroup/regroup-hri_local_changes.patch && cp -r regroup/added/. third_party/regroup-hri/
git -C third_party/ros2_leg_detector apply ../../baselines/leg_detector/ros2_leg_detector_humble.patch
```

## Data roots (`config/paths.env`)

| variable | content | used by |
|---|---|---|
| `P3D_BAGS_GO2NUS` | go2nus ROS 2 bags, one directory per bag; outputs go to `<bag>/slam_output/` | `pipeline/*` |
| `P3D_ZIPS_MIRAIKAN`, `P3D_BAGS_MIRAIKAN` | Miraikan bags (zips) and their unzipped working dirs | `pipeline/run_all.py --dataset miraikan` |
| `P3D_PROCESSED` | per dataset and trajectory: extracted camera frames (~4 Hz), `timestamps.txt`, lidar scans, `traj_data.pkl`, `regroup_tracks_sampled.pkl`, masks; `intrinsics/<dataset>/intrinsics.txt`, `extrinsics/<dataset>/<traj>/extrinsics.txt` | `ped3d/`, `vggt_depth/`, `ped_groups.py`, `groups_eval/` |
| `P3D_DEPTH` (`P3D_DEPTH_ALT` for datasets stored on a second drive) | VGGT-Omega depth per dataset | `vggt_depth/`, `ped3d/` |
| `P3D_MASKS` | person masks per dataset | `ped3d/`, `vggt_depth/` |
| `P3D_REGROUP` | 15 Hz REGROUP tracks per dataset | `vggt_depth/`, `ped_groups.py`, `groups_eval/` |
| `P3D_PED3D_OUT` | ped3d output pickles `<dataset>/<traj>.pkl` (input to `ped_tracks.py` / `fuse_tracks.py`), `tracks_<dataset>.pkl`, `extrinsics_est/`, `tracking/` | `ped3d/`, `pipeline/template/dataset_profile.py` |
| `P3D_CONDA_ENVS` | directory holding the `kiss_icp`, `erasor2`, `calib` (and `pytorch3d` for LV-DOT) conda envs | everything |
| `P3D_ERASOR2_BIN` (optional) | `run_erasor2` binary; default `third_party/ERASOR2/build/run_erasor2` | `pipeline/run_all.py` |
| `P3D_FIG_OUT`, `P3D_LVDOT_OUT`, `P3D_UMICH_BAGS`, `P3D_YOLO_SEG`, `P3D_SCRATCH`, `P3D_LEG_DETECTOR_WS` | figure output dir, LV-DOT results, UMich split bags, YOLO-seg weights for the blur script, scratch dir, leg-detector ROS workspace | `figures/`, `baselines/`, `ped3d/extract_umich_extrinsics.py`, `groups_eval/run.py` |

## Running

```bash
# 1. 2D tracks (REGROUP, Docker) and bag extraction (sbpd-dataset-pipeline) -> $P3D_PROCESSED/<dataset>/
# 2. VGGT-Omega depth
$P3D_CONDA_ENVS/calib/bin/python vggt_depth/generate_vggt_omega_depth.py
# 3. camera-based 3D pedestrian estimation, all datasets (resumable)
ped3d/run_all_ped3d.sh
# 4. SLAM + static map + lidar tracking + fusion, all bags (resumable; DONE/FAILED markers per bag)
python3 pipeline/run_all.py [--dataset miraikan] [--bags list.txt] [--dry-run]
python3 pipeline/retrack_all.py      # lidar tracking + fusion only
python3 pipeline/fuse_all.py         # fusion only
```

`run_all.py` writes `logs/` and `summary.csv` next to itself, so running it from this repo keeps those
out of git (`logs/` is ignored; the committed `summary*.csv` are the tables of the batch run of
2026-09-26 to 2026-10-10).

Per-bag outputs (`<bag>/slam_output/erasor2/output_ped/`): `static_map_slam_frame.pcd` (final map),
`fused_tracks.csv` + `fused_track_members.csv` (fused persons), `lidar_tracks_smooth.csv`,
`ped3d_tracks_map.csv`, `groups.csv` (when `ped_groups.py` has been run). Details in
`docs/PEDESTRIAN_3D.md`.

## Provenance (consolidated 2026-10-27)

| repo path | copied from |
|---|---|
| `pipeline/` | the batch pipeline directory next to the go2nus bags (without `logs/`, `backup_failed_slam/`, `groups_eval/sheets/`) |
| `ped3d/`, `vggt_depth/`, `baselines/lvdot_eval/` | the socnav notebooks folder (untracked there) |
| `eval/`, `baselines/leg_detector/`, `docs/erasor2_utown1_dev_notes.md` | `slam_output/` of the development bag `utown_1_2025-03-07_12-41-18_10_corrected_merged` |
| `figures/blog_scripts/` | the ped3d blog scripts |
| `regroup/` | the local regroup-hri checkout (diff + untracked files) |

The originals were left in place. All workstation paths in the copies were replaced by `P3D_*`
variables (`p3d_env.py`, `config/paths.env`); the docs under `docs/` were written against the old
layout and name the same variables where paths occurred.

Not included: the Nav2 shadow-mode experiments and the RL primitive-selection work
(github.com/raoshashank/social_RL_scene_reconstruction), which consume the maps and tracks produced here.
