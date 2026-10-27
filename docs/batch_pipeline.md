# go2nus static maps with pedestrians removed (batch)

`run_all.py` runs the utown_1 pipeline on every bag under `/media/dataset_access/go2nus_annotated/`:
offline SLAM (KISS-ICP + GTSAM) -> Patchwork++/HDBSCAN labels -> ERASOR2 -> ped3d pedestrian masking
-> ERASOR2 -> lidar-vs-map 360-deg detections -> masking with those too -> ERASOR2 (final map).
Details of each step: `<bag>/slam_output/erasor2/README.md` of the utown_1 dev bag
(`utown_1_2025-03-07_12-41-18_10_corrected_merged`).

```bash
python3 run_all.py                    # all bags, skips finished ones (DONE marker), ~1 min per 15 s of bag
python3 run_all.py --bags list.txt    # subset
python3 run_all.py --dry-run          # status per bag
tail -f logs/run_all.out              # progress; per-bag logs in logs/<bag>.log; summary.csv rebuilt after every bag
```

Per bag, in `<bag>/slam_output/`:

| file | what |
|---|---|
| `erasor2/output_ped/static_map_slam_frame.pcd` | **final static map**, SLAM `map` frame |
| `erasor2/output/static_map_slam_frame.pcd` | ERASOR2-only map (no ped3d), for comparison |
| `map.pcd`, `map_full_res.pcd`, `poses_*_tum.txt` | SLAM outputs (raw accumulated map, trajectories) |
| `erasor2/data/Merged/{velodyne,patchwork,hdbscan,ped}/` | per-scan points, ground, instances, pedestrian mask |
| `erasor2/output_ped/lidar_tracks.csv`, `lidar_dets.npz`, `lidar_detect.png` | 360-deg pedestrian detections vs the map |
| `erasor2/output_ped/ped3d_tracks_map.csv` | ped3d tracks re-expressed in the SLAM frame |
| `erasor2/camera_model.npz` | per-session camera intrinsics/extrinsics (camera pitch differs per session) |
| `erasor2/summary.json` | points, ped3d residue (fraction of pedestrian voxels left), raw voxels removed |
| `replay_detections.py`, `detections_replay.rviz` | RViz replay with the bag (`ros2 bag play <bag> --clock`) |
| `DONE` / `FAILED` | markers; FAILED names the step, see the log |

## Other datasets (`--dataset`)

Robot/sensor setup per dataset lives in `template/dataset_profile.py`; `run_all.py` writes it to
`<bag>/slam_output/dataset.json`, which every script reads (no file = go2nus).

```bash
python3 run_all.py --dataset miraikan            # tail -f logs/run_all_miraikan.out; summary_miraikan.csv
```

- **miraikan** (CaBot: Velodyne VLP-16, RealSense D455, `/odom`, `/cabot/imu/data`): bags come as zips in
  `/media/dataset_access/ros2_bags_acme_reconstruction/miraikan/`, unzipped on demand to
  `/media/dataset_access/miraikan_reconstruction/<bag>/` (outputs in `<bag>/slam_output/` as above).
  One map per zip; consecutive splits of a session may start from different places, so joining them is
  left for later. 0-byte zips (incomplete download) are reported as `no bag` and picked up by a rerun.
  Differences from go2nus: no per-point lidar stamps (header stamp, no corrupt-scan span check);
  gravity from RANSAC floor planes in each scan (the IMU tilt is off by up to ~2 deg in some sessions)
  plus a soft prior that the floor stays at map z = 0 (VLP-16 ICP drifts in z: 11 cm over 10 m);
  lidar 0.553 m above the floor (measured from floor planes, 6 sessions); ped3d from
  `/media/shashank/ssd2/ped3d/miraikan/<bag>.pkl`.

Not run in batch: the review-only group completion (people standing for the whole recording that
neither sensor ever detected stay in the map; see the utown_1 README). Bags without a ped3d pickle
(2) get the ERASOR2-only map; bags with < 30 scans are skipped.

Machine note: everything runs under `taskset -c 0-15 nice -n 10` and <= 8 workers, because this
i9-13900KF crashes under all-core load.
