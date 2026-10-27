"""Full-dataset pedestrian depth accuracy for go2nus: DepthAnything vs VGGT-Omega vs lidar.

Per pedestrian instance (connected component >=300 px of the eroded masked_imgs mask,
>=10 lidar points inside): gt = median lidar z-depth in the component, each model's
estimate = median of its metric depth (depth/ and vggt_omega_depth/ uint16 mm PNGs)
at those same lidar pixels.

Writes one row per instance to --out (npz):
    traj_idx, frame, gt, da, vggt, n_lidar, area   + trajs (list of names)

Run with the calib env:
    /home/shashank/mambaforge/envs/calib/bin/python ped_depth_accuracy.py
"""
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

cv2.setNumThreads(1)

DATA_ROOT = '/media/shashank/T1/processed/go2nus'
DEPTH_ROOT = '/media/shashank/ssd2/processed_depth/go2nus'
MASK_ROOT = '/media/shashank/ssd2/mask_dataset/go2nus'
EXTRINSICS_ROOT = '/media/shashank/T1/processed/extrinsics/go2nus'
INTRINSICS_FILE = '/media/shashank/T1/processed/intrinsics/go2nus/intrinsics.txt'
IMG_W, IMG_H = 848, 480
MIN_Z_CAM, MAX_RANGE = 0.3, 30.0
MIN_AREA, MIN_LIDAR = 300, 10

K = np.loadtxt(INTRINSICS_FILE, delimiter=',').reshape(3, 3)
BASE_T_LIDAR = np.eye(4)
BASE_T_LIDAR[:3, 3] = [0.14543, 0.0, 0.13312]


def metric_png(path):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / 1000.0
    d[d <= 0] = np.nan
    return d


def process_traj(arg):
    ti, traj = arg
    try:
        cam_T_base = np.loadtxt(os.path.join(EXTRINSICS_ROOT, traj, 'extrinsics.txt')).reshape(4, 4)
    except OSError:
        return traj, np.zeros((0, 7)), 'no extrinsics'
    cam_T_lidar = cam_T_base @ BASE_T_LIDAR
    img_dir = os.path.join(DATA_ROOT, traj, 'imgs')
    if not os.path.isdir(img_dir):
        return traj, np.zeros((0, 7)), 'no processed dir'
    n = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    rows = []
    missing = 0
    for i in range(n):
        m = cv2.imread(os.path.join(MASK_ROOT, traj, 'masked_imgs', f'{i}.jpg'), cv2.IMREAD_GRAYSCALE)
        da = metric_png(os.path.join(DEPTH_ROOT, traj, 'depth', f'{i}.png'))
        vg = metric_png(os.path.join(DEPTH_ROOT, traj, 'vggt_omega_depth', f'{i}.png'))
        pcd_path = os.path.join(DATA_ROOT, traj, 'pcd', f'{i}.npz')
        if m is None or da is None or vg is None or not os.path.exists(pcd_path):
            missing += 1
            continue
        m = cv2.erode((m > 127).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        if not m.any():
            continue
        pts = np.load(pcd_path)['arr_0'][:, :3]
        pc = (cam_T_lidar[:3, :3] @ pts.T).T + cam_T_lidar[:3, 3]
        z = pc[:, 2]
        keep = (z > MIN_Z_CAM) & (z < MAX_RANGE)
        pc, z = pc[keep], z[keep]
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(int)
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(int)
        inb = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        u, v, z = u[inb], v[inb], z[inb]
        n_comp, labels = cv2.connectedComponents(m)
        lab = labels[v, u]
        for c in range(1, n_comp):
            area = int((labels == c).sum())
            if area < MIN_AREA:
                continue
            sel = lab == c
            if sel.sum() < MIN_LIDAR:
                continue
            gt = np.median(z[sel])
            da_med = np.nanmedian(da[v[sel], u[sel]])
            vg_med = np.nanmedian(vg[v[sel], u[sel]])
            rows.append((ti, i, gt, da_med, vg_med, int(sel.sum()), area))
    note = f'{len(rows)} instances' + (f', {missing} frames missing data' if missing else '')
    return traj, np.array(rows) if rows else np.zeros((0, 7)), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/media/shashank/ssd2/processed_depth/ped_depth_accuracy_go2nus.npz')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    trajs = sorted(t for t in os.listdir(DEPTH_ROOT)
                   if os.path.isdir(os.path.join(DEPTH_ROOT, t, 'vggt_omega_depth')))
    print(f'{len(trajs)} trajectories with vggt_omega_depth')
    t0 = time.time()
    all_rows = []
    with Pool(args.workers) as pool:
        for k, (traj, rows, note) in enumerate(pool.imap_unordered(process_traj, enumerate(trajs))):
            all_rows.append(rows)
            if (k + 1) % 25 == 0 or 'missing' in note or 'no ' in note:
                print(f'[{k + 1}/{len(trajs)} {time.time() - t0:.0f}s] {traj}: {note}', flush=True)
    R = np.concatenate(all_rows)
    np.savez_compressed(args.out, records=R, trajs=np.array(trajs))
    print(f'{len(R)} pedestrian instances total -> {args.out}')


if __name__ == '__main__':
    main()
