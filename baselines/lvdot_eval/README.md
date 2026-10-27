# LV-DOT on the socnav corpora

Can [LV-DOT](../lv-dot) (Xu et al. 2025, lidar + colour dynamic-obstacle detector for drones) replace or
improve the pedestrian estimates we already have from `notebooks/ped3d` (tracker boxes + VGGT-Omega depth
+ lidar)? LV-DOT is a ROS1 Noetic C++ node; only ROS2 Humble is installed, so `lvdot_port.py` re-implements
the detection front end (lidar preprocessing, DBSCAN, box projection, YOLO fusion, size constraint) and the
evaluation runs it offline on the 4 Hz processed frames. Tracking (30 Hz Kalman + point-velocity voting) is
not ported.

    PY="env PYTHONNOUSERSITE=1 /home/shashank/mambaforge/envs/pytorch3d/bin/python"
    $PY run_yolo.py            # LV-DOT's yolo11n person boxes (stock 352x352 BGR tensor + default 640 letterbox)
    $PY eval_lvdot.py          # port vs ped3d, 65 sampled trajectories, 12k frames -> /media/shashank/ssd2/lvdot_eval/results_*.pkl
    $PY summarize.py           # tables -> summary.md
    MPLBACKEND=Agg $PY viz.py --dataset go2nus --traj <name> --frames 83 --cfg tuned   # image + BEV panel

Configs: `stock` = cfg/detector_param.yaml (Gaussian range downsampling sigma 6 m, DBSCAN radius
sqrt(0.05) = 0.22 m / 10 pts); `tuned` = same algorithm, no downsampling, 15 m box, radius 0.30 m / 6 pts.

## Result (details in /media/shashank/ssd2/lvdot_eval/summary.md)

Recall of ped3d pedestrians within 10 m, 33.7k instances, 4 corpora:

| range  | stock: cluster / human | tuned: cluster / human |
|--------|------------------------|------------------------|
| 0-3 m  | 92% / 53%              | 75% / 55%              |
| 3-6 m  | 94% / 57%              | 82% / 64%              |
| 6-10 m |  9% /  4%              | 92% / 67%              |

* Stock parameters do not cluster anyone beyond ~6 m on any of our lidars (16/32-beam ring spacing >
  0.22 m); the Gaussian downsampling alone is not the cause (disabling it gives 6-12%).
* With the radius adapted, the 3D front end isolates 86% of pedestrians (95% outdoors on scand_spot),
  but crowds within 3 m merge (near recall drops) and the YOLO-to-cluster association is depth-agnostic
  (best 2D IOU > 0 with the projected AABB), so ~20% of "human" boxes land on background clusters and
  the human recall stays ~20 points below cluster recall.
* When it does hit, xy error vs ped3d is 0.13 m median, 0.3 m p90 -- same as ped3d's own lidar/VGGT
  agreement. Runtime ~10 ms/frame single core.
* ped3d's lidar coverage on the same frames is 98-99% (box-guided point gathering does not depend on
  clustering). Nothing in LV-DOT's front end improves on it; its value would be the tracker/dynamic
  classification, which needs 30 Hz data we do not store.
* miraikan `imgs` are anonymised (people replaced by flat silhouettes) so YOLO only fires on 12-32% of
  them there; cluster recall is unaffected.
