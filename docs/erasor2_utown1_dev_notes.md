# ERASOR2 dynamic-object removal on this bag

Input: the offline SLAM outputs in `..` (`poses_rslidar_tum.txt`) + `/rslidar_points` from the bag.
ERASOR2 source/binary: `third_party/ERASOR2` (`build/run_erasor2`, conda env `$P3D_CONDA_ENVS/erasor2`).

## Re-run

```bash
$P3D_CONDA_ENVS/kiss_icp/bin/python export_scans.py              # scans + KITTI poses -> data/Merged (only if SLAM changed)
OMP_NUM_THREADS=1 $P3D_CONDA_ENVS/erasor2/bin/python gen_labels.py --workers 8   # Patchwork++ ground + HDBSCAN instances
nice -n 10 $P3D_ERASOR2_BIN go2_utown.yaml             # ~25 s
```

Keep workers <= 8 on this machine (all-core load makes it crash).

## Outputs (`output/`)

| File | What |
|---|---|
| `Merged_0_frame_0_to_434_estimated.pcd` | ERASOR2 static map, **lifted +0.38 m in z** (ERASOR2 adds `sensor_height`) |
| `static_map_slam_frame.pcd` | same map shifted back into the SLAM `map` frame (matches `../map.pcd`) |
| `removed_points.pcd` | raw-accumulation points (5 cm voxels, SLAM frame) with no static-map point within 10 cm |
| `erasor2_check.png` | top-down: raw map + removed points (red) vs. static map |
| `mos/NNNNNN.label` | per-frame labels (251 = dynamic) for even frames 0..434; indexed by ERASOR2's internal filtered/voxelised cloud, NOT the raw `.bin` point order, and upstream `save_dyn_label` overwrites some dynamic points with potential-dynamic ones |
| `erasor2_run.rrd` | rerun recording (`rerun output/erasor2_run.rrd`) |
| `run_erasor2.log` | log |

`output_minz-0.25/` is the first run with `min_z_voi: -0.25`: that puts the ground outside the
volume of interest, so ERASOR2's ground-likelihood grid gets no ground points and it removes almost
nothing (0.2 % of the map vs 4.4 % with `-1.0`). Kept for comparison only.

## Pedestrian-aware variant (ped3d + ERASOR2) — recommended map

Uses the ped3d tracks (`$P3D_PED3D_OUT/go2nus/<bag>_0.pkl`, RGB-D + lidar fusion) plus
the lidar-vs-map detections below to remove pedestrians from every scan before ERASOR2, which then
also sees the emptied space and cleans up what neither found.

```bash
$P3D_CONDA_ENVS/erasor2/bin/python ped_mask.py                                  # ped3d only
nice -n 10 $P3D_ERASOR2_BIN go2_utown_ped.yaml                          # map v1
PYTHONPATH=. $P3D_CONDA_ENVS/erasor2/bin/python lidar_ped_detect.py             # 360-deg detections vs map v1
cp output_ped/static_map_slam_frame.pcd /tmp/map_v1.pcd
$P3D_CONDA_ENVS/erasor2/bin/python ped_mask.py --lidar-tracks --group-completion /tmp/map_v1.pcd
#   -> prints/writes output_ped/completion_candidates.csv; review with camera crops, then:
$P3D_CONDA_ENVS/erasor2/bin/python ped_mask.py --lidar-tracks --group-completion /tmp/map_v1.pcd --accept 0
nice -n 10 $P3D_ERASOR2_BIN go2_utown_ped.yaml                          # final map
$P3D_CONDA_ENVS/erasor2/bin/python eval_ped.py; $P3D_CONDA_ENVS/erasor2/bin/python recall_eval.py output/static_map_slam_frame.pcd output_ped/static_map_slam_frame.pcd
```

- `ped_tracks.py`: ped3d positions live in the onboard-odometry frame, which drifts ~0.85 m (median)
  against SLAM here, so each position goes odom -> robot (ped3d pose) -> map (SLAM pose, same stamp);
  raw ped3d lidar detections land 0.06 m (median) from a lidar point at zero time offset. Standing
  tracks (< 0.3 m/s) are held in place for the rest of the recording: the forward camera loses a
  standing person as soon as the robot has passed, the lidar keeps seeing them.
- `ped_mask.py`: 0.4 m cylinder around each track, region-grown through 0.15 m neighbours up to
  0.8 m from the track. No static-evidence guard (untracked standing people make their own voxels
  look static). `--group-completion` lists person-shaped columns in the map next to camera-confirmed
  standing people; lidar shape alone cannot tell a person from a 1.5 m sign box or a staked sapling
  (candidates 2 and 1 on this bag), so nothing is masked without `--accept`.
- `recall_eval.py`: static recall against a persistent reference (voxels seen both while the robot
  stood near the group and in another time window, minus everything masked).

| | static pts | persistent-static recall | ped3d pedestrian voxels left in map |
|---|---|---|---|
| ERASOR2 only (`output/`) | 3.72 M | 98.5 % | 20.3 % |
| ped3d + lidar tracks + accepted completion (`output_ped/`) | 3.63 M | 98.6 % | 6.3 % |

Standing group at ~(10, 0.6) (camera frames 0-40: seven people on open pavement, no furniture):
five tracked by ped3d, a sixth behind them never detected by either sensor (accepted completion
candidate 0), the seventh (man in the dark shirt by the pillar) tracked briefly by ped3d. Map points
0.1-2.2 m high within 0.8 m of them: 48k -> ~1.7k, all below 0.6 m (`output_ped/group_views.png`).
Side effect to know about: the pillar's sign box at (11.4, 2.8), 0.6 m from where the seventh
person stood, lost most of its points (mask growth reaches it, ERASOR2 removes the rest);
the sapling at (11.0, 4.1) is kept.

Lidar caveat: the rslidar's upper FOV is +15 deg, so from 2.5-4 m a standing person only returns
points up to ~1.0-1.4 m; do not judge "nobody there" from the absence of tall points.

`output_ped/static_map_slam_frame.pcd` is the map to use (SLAM `map` frame). Also:
`removed_points.pcd`, `removed_ped3d.pcd`, `erasor2_ped_check.png`, `group_views.png`.
`data/Merged/ped/NNNNNN.label` is a per-point pedestrian mask (1 = pedestrian) in raw `.bin` order.

## Lidar pedestrian detection against the static map (360 deg) + feedback into the map

`lidar_ped_detect.py`: per scan, drop Patchwork ground and every point with a static-map point
within 15 cm; DBSCAN the residual (eps 0.35); k-means-split clusters wider than a person (people
side by side); keep person-sized clusters (footprint 0.15-1.2 m, top 0.45-2.3 m above the local
ground, bottom <= 1 m); keep detections that persist in >= 2 of the 4 surrounding scans; link by
nearest neighbour. Camera FOV from `camera_model.npz` (ped3d_lib K, cam_T_base). ~10 s per bag.

Evaluated inside the camera FOV, <= 12 m, against ped3d (camera+lidar fusion, 4 Hz):
recall of ped3d detections 87 %, precision vs ped3d tracks 94.5 % (lower bound: ped3d misses too),
position offset 0.14-0.16 m median at all ranges. Outside the FOV (behind / beside the robot, where
ped3d has nothing): ~2.9k detections in 62 tracks of >= 5 scans (median 33 scans), see
`output_ped/lidar_detect.png` (red). Misses inside the FOV before the width split were mostly people
walking side by side merged into one cluster; map residue costs only ~3 %.

Feedback loop (`ped_mask.py --lidar-tracks` reads `output_ped/lidar_tracks.csv`, then ERASOR2 again):
masks people the camera never saw while they moved. Static recall stays 98.5 % (persistent
reference minus everything masked); detector recall on the re-made map 84.9 -> 87 %. Note that
background subtraction can never find a person who is IN the map (stood still the whole recording
and was never detected): that is what the reviewed `--group-completion` step is for.

(Commands: see the pipeline above.) Final detector numbers on the final map: recall 87.0 %,
precision 89.1 % (n=1703; the precision drop from 94.5 % comes from static bits the masks carved
out, which now show up as residual).
Outputs: `output_ped/lidar_tracks.csv` (track, scan, t, x, y, top_h, height, n_pts, range, in_fov),
`lidar_dets.npz`, `lidar_detect.png`. Caveats: a person standing for the whole recording and never
detected by either sensor stays in the map and is invisible to the residual; groups closer than
~0.35 m are split by width only; the +15 deg upper lidar FOV clips heads within ~4 m.

## Visualizing the 360-deg detections with the bag playing (RViz2)

`../replay_detections.py` extends `../replay_with_map.py`: cleaned map + SLAM path + map->odom TF,
plus `/residual` (scan points farther than 15 cm from the static map = moving things, yellow) and
`/detections` (MarkerArray: cylinders per lidar-vs-map detection, blue inside the camera FOV, red
outside, track id above; green spheres = ped3d tracks). Reads `output_ped/lidar_tracks.csv`,
`output_ped/ped3d_tracks_map.csv`, `output_ped/static_map_slam_frame.pcd`.

```bash
# three terminals, ROS 2 Humble sourced, from slam_output/
ros2 bag play .. --clock            # add -r 0.5 to slow down
python3 replay_detections.py        # ~20 s to build the KD-tree before it says "loaded"
rviz2 -d detections_replay.rviz
```

## Comparison with ros2_leg_detector (mowito), same bag

`../leg_detector/leg_detector_bag.launch.py` runs pointcloud_to_laserscan (shin-height slice of
/rslidar_points) + detect_leg_clusters + joint_leg_tracker (workspace `$P3D_LEG_DETECTOR_WS`).
Visual side-by-side in one RViz (fixed frame `map`; the leg detector works in `odom`, the replay
node's map->odom TF puts it on the map):

```bash
# ROS 2 Humble + $P3D_LEG_DETECTOR_WS sourced, from slam_output/
ros2 launch leg_detector/leg_detector_bag.launch.py
python3 replay_detections.py
rviz2 -d compare_replay.rviz          # adds /scan (cyan) + /visualization_marker (leg detector)
ros2 bag play .. --clock              # start last
```

Numbers: `leg_detector/record_leg_tracks.py` records `/people_tracked` into the map frame
(`output_ped/leg_tracks_map.csv`) while the bag plays; `compare_leg.py` scores it with the same
ped3d-in-FOV metric as `lidar_ped_detect.py` and draws `output_ped/compare_leg.png`.

## Lidar tracking v2 (`lidar_tracker.py`, replaces the greedy linker)

Raw per-scan detections (`lidar_ped_detect.detect_raw`, cached in `output_ped/lidar_dets_raw.npz`)
-> constant-velocity Kalman filter per track, Hungarian assignment with a Mahalanobis gate (chi^2 95 %),
confirmation after 3 hits, coasting up to 0.5 s -> offline RTS smoothing with robust re-weighting
(ped3d's smoother) -> duplicate merge + re-linking across gaps <= 1.5 s (ped3d `relink.py` logic)
-> re-smoothing. Measurement noise: ped3d's `sigma_lidar(range, n_points)`.

Outputs: `output_ped/lidar_tracks_smooth.csv` (track, scan, t, x, y, vx, vy, pos_std, detected,
in_fov, range, top_h; every scan of every track, `detected=0` = filled by the smoother),
`lidar_tracks.csv` / `lidar_dets.npz` (detected scans only, v1 format). The replay shows the
smoothed tracks (faint = filled) and labels ped3d spheres `p<id>`.

Tuning on 9 bags (utown_1 + 8 random crowded ones) against ped3d inside the camera FOV (ped3d is
not ground truth; swaps = the lidar id moves between two ped3d people that both exist, each
followed for >= 3 matches):

| | tracks | median length | recall / precision | lidar-id changes per ped3d track | swaps | id errors |
|---|---|---|---|---|---|---|
| v1 greedy | 5362 | ~0.9 s | 79.5 / 86.0 % | 0.98 | 289 | ~1950 |
| v2 (coast 1 s, gap 3 s, gate 99 %, q 1.2) | 1583 | 7.4 s | 87.8 / 80.6 % | 0.45 | 382 | ~1150 |
| v2 final (coast 0.5 s, gap 1.5 s, gate 95 %, q 0.8) | 2149 | 4.0 s | 86.2 / 83.0 % | 0.41 | 305 | ~1000 |

`PYTHONPATH=. python lidar_tracker.py --compare-v1 [--no-write] [--set MAX_COAST_S=0.5 ...]`.
All bags: `<repo>/pipeline/retrack_all.py` (maps not rebuilt).

## ped3d + lidar fusion (`fuse_tracks.py`)

One track set per bag: `output_ped/fused_tracks.csv` (10 Hz, person ids, body-centre positions,
source lidar / ped3d / both / filled, label camera_verified / camera_only / lidar_only /
unconfirmed_in_fov) + `fused_track_members.csv`. Method in
`<repo>/pipeline/PEDESTRIAN_3D.md` section 6.

Findings while building it (9 evaluation bags):
- ped3d positions are body centres, lidar detections visible-surface centroids: ped3d sits
  +0.08..+0.13 m farther along the viewing ray, no sideways offset -> lidar pushed by 0.12 m.
  Agreement with ped3d single frames within 10 m then 0.08 m median (0.12 m before).
- ped3d single-frame positions scatter by metres beyond ~15 m (unmatched ped3d tracks: median
  range 21 m, 3 frames) -> agreement is scored within 10 m only.
- Lidar-dropout hold-out (hide 1 s / 2.5 s of a camera-verified person's lidar detections): fused
  positions are as accurate as lidar alone bridged by the smoother, with full instead of 97-98 %
  coverage. ped3d's value is identity (verification, joining fragments, repairing swaps) and
  coverage, not extra accuracy where the lidar sees the person.
- About two thirds of camera-verified persons keep their id for >= 1 s outside the camera FOV.

`PYTHONPATH=. python fuse_tracks.py [--holdout 1 2.5] [--no-write]`; all bags: `fuse_all.py`.
