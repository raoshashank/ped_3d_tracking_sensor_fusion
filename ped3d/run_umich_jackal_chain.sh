#!/bin/bash
# after umich depth generation: estimate -> fit per-dataset VGGT correction (umich only, jackal has no
# lidar) -> re-estimate -> heatmaps -> training pickles, for umich and scand_jackal
[ -n "$P3D_CONDA_ENVS" ] || source "$(dirname "$0")/../config/paths.env"
PY=$P3D_CONDA_ENVS/calib/bin/python
cd "$(dirname "$0")"
while pgrep -f "generate_vggt_omega_depth_frames.py --dataset umich" > /dev/null; do sleep 60; done
echo "$(date) depth done: $(tail -1 ../vggt_omega_umich.log)"
echo "$(date) == umich first pass";  "$PY" run_ped3d.py --dataset umich --workers 6 2>&1 | tail -7
echo "$(date) == umich fit";        "$PY" fit_vggt_correction.py --dataset umich 2>&1 | tail -5
echo "$(date) == umich second pass"; "$PY" run_ped3d.py --dataset umich --workers 6 --force 2>&1 | tail -7
echo "$(date) == jackal";           "$PY" run_ped3d.py --dataset scand_jackal --workers 6 2>&1 | tail -7
for ds in umich scand_jackal; do
    echo "$(date) == $ds heatmap"; "$PY" ped_heatmap.py --dataset $ds 2>&1 | head -4
    echo "$(date) == $ds export";  "$PY" export_tracks.py --dataset $ds 2>&1 | tail -1
done
echo "$(date) chain done"
