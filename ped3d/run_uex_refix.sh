#!/bin/bash
[ -n "$P3D_CONDA_ENVS" ] || source "$(dirname "$0")/../config/paths.env"
PY=$P3D_CONDA_ENVS/calib/bin/python
cd "$(dirname "$0")"
echo "$(date) == re-estimate (lidar-primary)"; "$PY" run_ped3d.py --dataset uex --workers 6 --force 2>&1 | tail -8
echo "$(date) == heatmap"; "$PY" ped_heatmap.py --dataset uex 2>&1 | head -3
echo "$(date) == export";  "$PY" export_tracks.py --dataset uex 2>&1 | tail -1
echo "$(date) refix done"
