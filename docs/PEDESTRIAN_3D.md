# Pedestrian 3D estimation on go2nus: pipeline outline

Per trajectory, offline (non-causal). Two sources of pedestrian positions, expressed in a common SLAM
map frame:
- ped3d (camera FOV only): people detected and tracked in the camera images, located in 3D by fusing
  lidar with camera depth estimation (VGGT-Omega)
- lidar tracks (360 deg): background subtraction of each lidar scan against a static map
The two are fused into one track set per trajectory (section 6).

```
bag (.db3)
 ├─ camera frames + lidar ─────► ped3d ──────────────► ped3d tracks (odom frame, camera FOV)
 │                                                          │ re-expressed per timestamp
 ├─ lidar + odom ──► offline SLAM ──► poses, raw map        ▼
 │                        │                          ped3d tracks (SLAM map frame)
 │                        ▼                                 │
 │          ERASOR2 + pedestrian masking ◄──────────────────┤
 │                        │                                 │
 │                        ▼                                 │
 │               static map (pedestrians removed)           │
 │                        │                                 │
 └─ lidar ───► residual vs static map ──► lidar detections ─► lidar tracks (360 deg)
                                                            │
                                                   fusion ◄─┘──► one track set per trajectory
```

## 1. ped3d: camera-based pedestrian positions

Code: `<repo>/ped3d/` (`ped3d_lib.py`, `run_ped3d.py`, `run_all_ped3d.sh`,
`relink.py`, `export_tracks.py`, `app.py` viewer). Environment: `$P3D_CONDA_ENVS/calib`.

Inputs per trajectory (`$P3D_PROCESSED/go2nus/<traj>/`): camera frames (~4 Hz),
person tracks (boxes + ids, `regroup_tracks_sampled.pkl`), person masks, lidar scans, robot pose
(`traj_data.pkl`, onboard odometry), per-trajectory camera extrinsics (camera pitch differs between
recording sessions) and shared intrinsics; VGGT-Omega metric depth maps.

Per person per frame:
1. Region: person mask ∩ track box (torso box if the mask is missing).
2. Depth candidates:
   - `lidar`: nearest substantial lidar cluster projected into the region
   - `lidar_g`: the same, with the cluster chosen with VGGT guidance
   - `vggt`: robust foreground depth mode of VGGT-Omega in the region
   - `vggt_c`: VGGT with a per-dataset range correction (`vggt_correction.json`)
   - `vggt_a`: VGGT anchored per track to the lidar
   - `da`: DepthAnything (single-image depth), scaled per frame; computed for comparison, not fused
3. Back-projection to the robot `base_link` frame, then to the odometry world frame with the
   robot pose of that frame.

Per track:
4. Lidar + camera-depth fusion: camera depth guides the lidar cluster choice (`lidar_g`), the lidar
   fixes the camera depth's scale per track (`vggt_a`), and each frame's position is the
   inverse-variance combination of the two (lidar noise by range and point count, VGGT noise by
   depth, doubled when the depth is multimodal); frames with only one of them use it alone.
5. Constant-velocity Kalman filter + RTS smoother with robust re-weighting (outliers
   down-weighted); gaps inside a track filled and flagged `interpolated`.
6. Corpus export only: re-linking of broken tracks (duplicate merge, constant-velocity gap
   bridging, Hungarian assignment, ambiguity rejection).

Outputs:
- `$P3D_PED3D_OUT/go2nus/<traj>_0.pkl`: per-frame detections (`pos_<source>`, 3D in
  `base_link`), per-track `world[source]`, `fused` (x, y, vx, vy in the odometry frame), flags,
  variances; robot poses per frame
- `$P3D_PED3D_OUT/tracks_go2nus.pkl`: corpus export for training (per-track ragged
  layout, re-linked `merged_tracks`)

## 2. Offline SLAM

`<bag>/slam_output/run_slam.py` (env `kiss_icp`): KISS-ICP scan-to-map odometry on
`/rslidar_points` (scans already motion-compensated, no deskew) + GTSAM pose graph with IMU
gravity-direction factors from `/utlidar/robot_odom` + position smoothing. No loop closures.

Outputs: `poses_base_link_tum.txt`, `poses_rslidar_tum.txt`, `map.pcd` / `map_full_res.pcd`
(raw accumulated map, pedestrians included). Map frame: origin at `base_link` of the first scan,
z up, x = initial heading. Per bag, not shared across bags.

## 3. ped3d tracks in the SLAM frame

`erasor2/ped_tracks.py`: each ped3d fused position is taken back to the robot frame with ped3d's
own robot pose at that timestamp and forward with the SLAM `base_link` pose at the same timestamp,
so odometry drift cancels. Standing tracks are held in place for the rest of the recording (the
forward camera loses standing people once the robot has passed them; the lidar keeps seeing them).
Output: `erasor2/output_ped/ped3d_tracks_map.csv` (track, t, x, y).

## 4. Static map with pedestrians removed

In `<bag>/slam_output/erasor2/` (env `erasor2`, ERASOR2 at `third_party/ERASOR2`):
1. `export_scans.py`: scans + poses in ERASOR2's layout.
2. `gen_labels.py`: Patchwork++ ground and HDBSCAN instance labels per scan.
3. ERASOR2 on the raw scans → `output/` (reference map, no pedestrian information).
4. `ped_mask.py`: per scan, points of tracked pedestrians removed (cylinder around the track
   position, region-grown over the body), then ERASOR2 on the masked scans.
5. Lidar detection/tracking (section 5) against that map; its tracks are masked as well and
   ERASOR2 runs again → `output_ped/static_map_slam_frame.pcd` (final map, SLAM frame).
6. Optional, review only: `ped_mask.py --group-completion` proposes person-shaped columns next to
   standing groups (people never detected by either sensor); candidates are accepted by hand.

ERASOR2 notes: the volume-of-interest lower bound must lie below the ground; the saved map is lifted
by the sensor height (undone in `finalize_map.py`).

## 5. Lidar pedestrian detection and tracking (360 deg)

Detection (`erasor2/lidar_ped_detect.py`, `detect_raw`): per scan, points to the map frame; drop
ground; drop points close to the static map (the residual = things that moved); DBSCAN; split
clusters wider than one person; keep person-sized clusters (footprint, top and bottom height
relative to the local ground); flag whether the camera could see them.

Tracking (`erasor2/lidar_tracker.py`):
1. Online: constant-velocity Kalman filter per track, Hungarian assignment with a Mahalanobis
   gate, confirmation after several hits, short coasting.
2. Offline: RTS smoothing with robust re-weighting (ped3d's smoother), duplicate merge, re-linking
   across short gaps (ped3d's re-linking logic), re-smoothing.

Outputs (`erasor2/output_ped/`): `lidar_tracks_smooth.csv` (every scan of every track: x, y, vx, vy,
position std, detected / filled, in FOV, range, top height), `lidar_tracks.csv` + `lidar_dets.npz`
(detected scans only), `lidar_detect.png`.

## 6. Fusion into one track set

`erasor2/fuse_tracks.py`:
1. Measurements, without counting the same lidar cluster twice: every lidar detection; from ped3d's
   per-frame detections (3D in `base_link`, placed with the SLAM pose at the camera time), the
   lidar-anchored VGGT position always and ped3d's own lidar position only when the lidar detector
   has no detection of that person at that moment. Lidar detections (visible-surface centroids)
   are moved to the body centre, ped3d's convention.
   Plausibility checks on ped3d: its lidar position is dropped when it disagrees with VGGT along the
   viewing ray by more than 3x their combined uncertainty (it sometimes takes the cluster of someone
   in front); isolated outlier frames are dropped, and ped3d tracks are split wherever consecutive
   frames imply more than 3 m/s plus noise (identity switches in its tracker).
2. Association: per camera frame, Hungarian matching between ped3d detections and lidar tracks.
3. Swap repair: a lidar track is cut where its matched ped3d person switches to another ped3d
   person present at the same time.
4. Persons: lidar segments and ped3d tracks with enough position agreement are grouped, refusing
   groupings that would put two simultaneous, distant tracks into one person (ped3d fragments are
   joined through lidar, lidar gaps through ped3d).
5. Per person: all measurements through the robust RTS smoother, with a walking-speed prior (velocity
   std 1.4 m/s) so that 2-3-frame camera-only tracks don't take their speed from depth noise; persons
   re-linked across short gaps.
6. Labels: `camera_verified` (lidar + ped3d), `camera_only` (ped3d without lidar: far, occluded, or
   part of the static map), `lidar_only` (never seen near the camera), `unconfirmed_in_fov` (in the
   camera FOV nearby without any ped3d detection: likely a bike, cart or foliage; flagged, kept).
   Standing people are not held in place: only observed positions are output.

Outputs (`erasor2/output_ped/`): `fused_tracks.csv` (person, scan, t, x, y, vx, vy, pos_std, source
lidar / ped3d / both / filled, in_fov, range, top_h, label; 10 Hz), `fused_track_members.csv`
(which lidar and ped3d tracks make up each person).

## Frames and conventions

- Times: Unix seconds (bag clock) everywhere.
- ped3d `pos_<source>`: 3D in `base_link` (x forward, y left, z up). ped3d tracks: 2D, onboard
  odometry frame (drifts against SLAM).
- Everything under `slam_output/`: SLAM `map` frame of that bag, 2D positions; ground height
  from the SLAM pose (`base_link` z − 0.35).
- Track ids: ped3d and lidar ids are separate namespaces; fused person ids (`P<id>`) are their own,
  with the mapping in `fused_track_members.csv`.
- Positions: fused tracks and ped3d are body centres; `lidar_tracks*.csv` are visible-surface
  centroids (~0.1 m nearer the robot).

## Running

- ped3d: `run_all_ped3d.sh` in the ped3d directory.
- Everything else, all bags: `python3 <repo>/pipeline/run_all.py`
  (resumable; per-bag logs in `logs/`, table in `summary.csv` of that directory).
- Re-tracking only (maps unchanged): `retrack_all.py`; fusion only: `fuse_all.py` (same directory).
- `fix_scan_stamps.py`: repairs scan timestamps broken by a stray point timestamp (older SLAM runs).
- Machine: everything runs pinned to 16 cores at low priority with ≤ 8 workers (the desktop is
  unstable under all-core load).

## Visualization

- RViz replay with the bag, from `<bag>/slam_output/`: `ros2 bag play .. --clock`,
  `python3 replay_detections.py`, `rviz2 -d detections_replay.rviz` (all layers) or `rviz2 -d fused_replay.rviz` (fused persons only) (static map, residual, lidar
  tracks blue/red by FOV with ids, ped3d tracks as `p<id>`, fused persons as `P<id>` coloured by
  label).
- Comparison with ros2_leg_detector (utown_1): `compare_replay.rviz` +
  `slam_output/leg_detector/leg_detector_bag.launch.py` (workspace `$P3D_LEG_DETECTOR_WS`).
- ped3d viewer: `app.py` (Streamlit) in the ped3d directory.

## Known limitations

- People standing still for the whole recording and never detected are part of the static map and
  invisible to the lidar residual.
- Outside the camera FOV there is no reference: lidar swaps there cannot be corrected, and lidar-only
  tracks are person-shaped but not camera-verified.
- Lidar upper FOV (+15 deg) clips heads of people close to the robot.
- Maps and track positions are per bag; there is no cross-bag alignment.
