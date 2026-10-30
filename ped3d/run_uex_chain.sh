#!/bin/bash
# after the uex depth generation finishes: estimate -> fit uex VGGT correction -> re-estimate -> heatmap -> export
[ -n "$P3D_CONDA_ENVS" ] || source "$(dirname "$0")/../config/paths.env"
PY=$P3D_CONDA_ENVS/calib/bin/python
cd "$(dirname "$0")"
while pgrep -f "generate_vggt_omega_depth_frames.py --dataset uex" > /dev/null; do sleep 60; done
echo "$(date) depth done: $(tail -1 ../vggt_omega_uex.log)"
echo "$(date) == first pass";  "$PY" run_ped3d.py --dataset uex --workers 6 2>&1 | tail -8
echo "$(date) == fit";         "$PY" fit_vggt_correction.py --dataset uex 2>&1 | tail -6
echo "$(date) == second pass"; "$PY" run_ped3d.py --dataset uex --workers 6 --force 2>&1 | tail -8
echo "$(date) == heatmap";     "$PY" ped_heatmap.py --dataset uex 2>&1 | head -12
echo "$(date) == export";      "$PY" export_tracks.py --dataset uex 2>&1 | tail -2
echo "$(date) all done"
