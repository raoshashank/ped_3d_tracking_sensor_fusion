"""Shared configuration for the LV-DOT evaluation on the socnav corpora."""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ped3d'))
import ped3d_lib as L  # noqa: E402  (Traj loader, DATASETS, PROCESSED_ROOT, OUT_ROOT)

OUT_ROOT = '/media/shashank/ssd2/lvdot_eval'
YOLO_WEIGHTS = os.path.join(HERE, '..', 'lv-dot', 'onboard_detector', 'scripts', 'yolo_detector', 'weights', 'yolo11n.pt')

# lidar height above the floor (bev_cache/lidar_mounts.yaml; scand_spot measured). The "ground" frame is the
# ped3d base frame shifted down by (LIDAR_H - base_T_lidar z) so that z = 0 is the floor.
LIDAR_H = {'go2nus': 0.483, 'gmu': 0.428, 'miraikan': 0.522, 'keio': 0.522, 'scand_spot': 0.86}

SAMPLE = {'go2nus': 25, 'gmu': 15, 'miraikan': 15, 'scand_spot': 10}
SEED = 0
EVAL_RANGE = 10.0
MATCH_R = 0.75        # m, LV-DOT box centre vs ped3d body centre


def sample_trajs(dataset):
    """Deterministic random subset of the trajectories that have ped3d results."""
    d = os.path.join(L.OUT_ROOT, dataset)
    trajs = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.pkl') and not f.startswith('_'))
    rng = random.Random(f'{SEED}-{dataset}')
    rng.shuffle(trajs)
    return sorted(trajs[:SAMPLE[dataset]])


def ground_frame(T):
    """(g_T_lidar, cam_T_g) for a ped3d Traj."""
    h = LIDAR_H[T.dataset] - T.base_T_lidar[2, 3]
    g_T_base = np.eye(4); g_T_base[2, 3] = h
    g_T_lidar = g_T_base @ T.base_T_lidar
    cam_T_g = T.cam_T_base @ np.linalg.inv(g_T_base)
    return g_T_lidar, cam_T_g


def yolo_path(dataset, traj):
    return os.path.join(OUT_ROOT, 'yolo', dataset, traj + '.pkl')
