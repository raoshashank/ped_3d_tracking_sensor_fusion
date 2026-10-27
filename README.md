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
| `docs/` | pipeline documentation (see above) | |
| `third_party/` | submodules, see below | |

## Third-party code (submodules)

| submodule | upstream | pinned | role | local changes |
|---|---|---|---|---|
| `third_party/ERASOR2` | github.com/url-kaist/ERASOR2 (fork at `raoshashank/ERASOR2`, see setup) | `d43d94f` | dynamic-object removal; builds `run_erasor2` | heavily modified: ROS-free, catkin-free, yaml-cpp config, rerun.io viewer, `erasor1` fixes; history squashed to one commit |
| `third_party/vggt-omega` | github.com/facebookresearch/vggt-omega | `282ec70` | metric multi-view depth (`vggt_omega` package, weights from HF `facebook/VGGT-Omega`) | none |
| `third_party/regroup-hri` | github.com/UCSD-RHC-Lab/regroup-hri | `f692f50` | 2D person tracking + grouping | `regroup/regroup-hri_local_changes.patch` + files in `regroup/added/` |
| `third_party/ros2_leg_detector` | github.com/mowito/ros2_leg_detector | `b4be35a` | baseline only | `baselines/leg_detector/ros2_leg_detector_humble.patch` |
| `third_party/lv-dot` | github.com/zhefan-xu/lv-dot | `449bf2c` | baseline only | none |
| `third_party/sbpd-dataset-pipeline` | github.com/social-navigation-group/sbpd-dataset-pipeline | `9ca27c8` (remote HEAD) | bag extraction that writes the per-trajectory frames, lidar scans and `traj_data.pkl` under `/media/shashank/T1/processed/<dataset>/` (`robot_data_analysis/data_sampling/process_bags_ros2.py`) | the workstation checkout `~/code/my_projects/sbpd-dataset-pipeline` is at `5b00be2` with uncommitted edits; commit and push there, then re-pin |

Other dependencies are plain packages (see `envs/`): KISS-ICP 1.3.0, GTSAM, pypatchworkpp, hdbscan,
open3d, rosbags, ultralytics, huggingface_hub, torch. ROS 2 Humble is needed only for the RViz replays
and the leg-detector baseline.

## Setup

```bash
git clone --recurse-submodules <this repo>
cd ped_3d_tracking_sensor_fusion
```

ERASOR2 submodule: `.gitmodules` points at `git@github.com:raoshashank/ERASOR2.git`, which has to be
created once from the modified checkout (`~/ERASOR2`, whose `origin` still names the upstream repo):

```bash
# create an empty repo raoshashank/ERASOR2 on GitHub, then
cd ~/ERASOR2 && git remote add fork git@github.com:raoshashank/ERASOR2.git && git push fork main
cd <this repo> && git submodule update --init third_party/ERASOR2
cd third_party/ERASOR2 && cmake -B build -S . && cmake --build build -j
```

`pipeline/run_all.py` uses `$ERASOR2_BIN` if set, else `third_party/ERASOR2/build/run_erasor2`, else
`~/ERASOR2/build/run_erasor2`.

Environments (`envs/`): `kiss_icp` (python 3.11: SLAM, scan export), `erasor2` (python 3.12: labels,
masking, lidar detection/tracking, fusion, grouping), `calib` (python 3.10: ped3d, VGGT-Omega,
`make_camera_model.py`; this env also picks up packages from `~/.local/lib/python3.10/site-packages`).
`vggt_omega` is installed in `calib` as an editable package: `pip install -e third_party/vggt-omega`.
The env python paths are hard-coded in `pipeline/run_all.py` (`ENV`) and in the ped3d shell scripts
(`/home/shashank/mambaforge/envs/<env>/bin/python`).

Apply the local patches when rebuilding a baseline or the 2D front-end:

```bash
git -C third_party/regroup-hri apply ../../regroup/regroup-hri_local_changes.patch && cp -r regroup/added/. third_party/regroup-hri/
git -C third_party/ros2_leg_detector apply ../../baselines/leg_detector/ros2_leg_detector_humble.patch
```

## Data roots (hard-coded, workstation-specific)

| path | content | used by |
|---|---|---|
| `/media/dataset_access/go2nus_annotated/<bag>/` | go2nus ROS 2 bags; outputs go to `<bag>/slam_output/` | `pipeline/*` |
| `/media/dataset_access/ros2_bags_acme_reconstruction/miraikan/*.zip`, `/media/dataset_access/miraikan_reconstruction/<bag>/` | Miraikan bags (zips) and their unzipped working dirs | `pipeline/run_all.py --dataset miraikan` |
| `/media/shashank/T1/processed/<dataset>/<traj>/` | extracted camera frames (~4 Hz), `timestamps.txt`, lidar scans, `traj_data.pkl`, `regroup_tracks_sampled.pkl`, masks; intrinsics/extrinsics under `T1/processed/{intrinsics,extrinsics}/` | `ped3d/`, `ped_groups.py`, `groups_eval/` |
| `/media/shashank/ssd2/processed_depth/<dataset>/` | VGGT-Omega depth | `vggt_depth/`, `ped3d/` |
| `/media/shashank/ssd2/regroup_depth/regroup_detections/<dataset>/` | 15 Hz REGROUP tracks | `ped_groups.py`, `groups_eval/` |
| `/media/shashank/ssd2/ped3d/<dataset>/<traj>.pkl` | ped3d output pickles (input to `ped_tracks.py` / `fuse_tracks.py`) | `pipeline/template/dataset_profile.py` |

## Running

```bash
# 1. 2D tracks (REGROUP, Docker) and bag extraction (sbpd-dataset-pipeline) -> /media/shashank/T1/processed/<dataset>/
# 2. VGGT-Omega depth
~/mambaforge/envs/calib/bin/python vggt_depth/generate_vggt_omega_depth.py
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
| `pipeline/` | `/media/dataset_access/ped_3d_detection_pipeline/` (without `logs/`, `backup_failed_slam/`, `groups_eval/sheets/`) |
| `ped3d/`, `vggt_depth/`, `baselines/lvdot_eval/` | `~/code/my_projects/socnav/notebooks/` (untracked there) |
| `eval/`, `baselines/leg_detector/`, `docs/erasor2_utown1_dev_notes.md` | `/media/dataset_access/go2nus_annotated/utown_1_2025-03-07_12-41-18_10_corrected_merged/slam_output/` (development bag) |
| `figures/blog_scripts/` | `/media/dataset_access/ped3d_blog/scripts/` |
| `regroup/` | `~/code/packages/regroup-hri` (diff + untracked files) |

The originals were left in place. Three paths were changed in the copies: `pipeline/run_all.py`
(`ERASOR2_BIN` lookup), `pipeline/template/make_camera_model.py` (`PED3D_DIR`, default `ped3d/` of this
repo), `figures/paper_fig/fig_pipeline.py` (`BLOG` relative to the repo). Everything else still
carries the workstation paths listed above.

Not included: the Nav2 shadow-mode experiments (`/media/dataset_access/reconstruction_experiments/`)
and the RL primitive-selection work (github.com/raoshashank/social_RL_scene_reconstruction), which
consume the maps and tracks produced here.
