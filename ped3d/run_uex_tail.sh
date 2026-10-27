#!/bin/bash
# after run_uex_chain.sh: depth + estimates for the 17 trajectories without a per-trajectory extrinsics file
# (static mount -> dataset-wide extrinsics), then refresh heatmap + export
PY=/home/shashank/mambaforge/envs/calib/bin/python
cd "$(dirname "$0")"
until grep -q "all done" run_uex_chain.log; do sleep 30; done
echo "$(date) == depth for remaining"; "$PY" ../generate_vggt_omega_depth_frames.py --dataset uex 2>&1 | grep -v Warning | grep -v "skip$" | tail -20
echo "$(date) == estimate remaining"; "$PY" run_ped3d.py --dataset uex --workers 6 2>&1 | tail -8
echo "$(date) == heatmap"; "$PY" ped_heatmap.py --dataset uex 2>&1 | head -6
echo "$(date) == export";  "$PY" export_tracks.py --dataset uex 2>&1 | tail -1
echo "$(date) tail done"
