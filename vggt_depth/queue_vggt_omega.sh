#!/bin/bash
# Queue: wait for the running miraikan depth generation to finish, then run
# gmu, keio, scand_spot sequentially (generate_vggt_omega_depth_multi.py).
# Each dataset logs to vggt_omega_<ds>.log; everything is resumable, so if
# this queue dies just rerun it (finished trajectories are skipped).
[ -n "$P3D_CONDA_ENVS" ] || source "$(dirname "$0")/../config/paths.env"
PY=$P3D_CONDA_ENVS/calib/bin/python
cd "$(dirname "$0")"

echo "$(date) queue started; waiting for miraikan run to finish..."
while pgrep -f generate_vggt_omega_depth_miraikan.py > /dev/null; do
    sleep 60
done
echo "$(date) miraikan run no longer running"

for ds in gmu keio scand_spot; do
    echo "$(date) starting ${ds}"
    "$PY" generate_vggt_omega_depth_multi.py --dataset "${ds}" > "vggt_omega_${ds}.log" 2>&1
    echo "$(date) finished ${ds} (exit $?): $(tail -1 "vggt_omega_${ds}.log")"
done
echo "$(date) queue complete"
