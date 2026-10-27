#!/bin/bash
# process every dataset (resumable; FORCE=1 recomputes everything), then heatmaps + training pickles
PY=/home/shashank/mambaforge/envs/calib/bin/python
cd "$(dirname "$0")"
EXTRA=""; [ "${FORCE:-0}" = "1" ] && EXTRA="--force"
for ds in go2nus miraikan keio gmu scand_spot uex; do
    echo "$(date) == $ds"
    "$PY" run_ped3d.py --dataset "$ds" --workers 6 $EXTRA 2>&1 | tail -8
done
echo "$(date) == heatmaps"; "$PY" ped_heatmap.py --all 2>&1 | grep "groups ->"
echo "$(date) == export";   "$PY" export_tracks.py --all 2>&1
echo "$(date) all done"
