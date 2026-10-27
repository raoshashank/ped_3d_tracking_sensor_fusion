#!/bin/bash
# after fuse_all.py finishes: re-track + fuse the bags whose scan stamps were repaired (fix_scan_stamps.py)
cd /media/dataset_access/ped_3d_detection_pipeline
# (waited for fuse_all.py in the first run)
PY=/home/shashank/mambaforge/envs/erasor2/bin/python
for b in $(cat stamp_fixed_bags.txt); do
  e=/media/dataset_access/go2nus_annotated/$b/slam_output/erasor2
  [ -f $e/output_ped/static_map_slam_frame.pcd ] || { echo "$b: no final map, skipped"; continue; }
  cp template/erasor2/lidar_tracker.py template/erasor2/lidar_ped_detect.py template/erasor2/fuse_tracks.py $e/
  cp template/replay_detections.py template/detections_replay.rviz $e/../
  (cd $e && PYTHONPATH=. OMP_NUM_THREADS=8 taskset -c 0-15 nice -n 10 $PY lidar_tracker.py > /dev/null 2>&1 \
     && PYTHONPATH=. OMP_NUM_THREADS=8 taskset -c 0-15 nice -n 10 $PY fuse_tracks.py > output_ped/fuse.log 2>&1 \
     && grep "^\[all\]" output_ped/fuse.log > output_ped/FUSED_V1 && touch output_ped/RETRACK_V2 \
     && echo "$b: ok" || echo "$b: FAILED")
done
