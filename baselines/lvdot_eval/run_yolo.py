"""Run LV-DOT's YOLO11n person detector on the sampled trajectories.

    PYTHONNOUSERSITE=1 $P3D_CONDA_ENVS/pytorch3d/bin/python run_yolo.py [--datasets go2nus ...]

Two variants per frame, saved to $P3D_LVDOT_OUT/yolo/<ds>/<traj>.pkl as
{'stock': {frame: (M,5) [x1,y1,x2,y2,conf]}, 'tuned': {...}}:
  stock : exactly yolov11_detector.py -- BGR image squashed to 352x352, fed as a raw tensor
          (no letterbox, no channel swap), half precision, default conf 0.25, class "person"
  tuned : ultralytics default preprocessing (letterbox 640, RGB), same weights, class "person"
"""
import argparse
import os
import pickle
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import common as C

BATCH = 32


def run_traj(model, dataset, traj):
    img_dir = os.path.join(C.L.PROCESSED_ROOT, dataset, traj, 'imgs')
    n = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    out = {'stock': {}, 'tuned': {}}
    for s in range(0, n, BATCH):
        idx = list(range(s, min(n, s + BATCH)))
        imgs = [cv2.imread(os.path.join(img_dir, f'{i}.jpg')) for i in idx]
        ok = [k for k, im in enumerate(imgs) if im is not None]
        idx = [idx[k] for k in ok]; imgs = [imgs[k] for k in ok]
        if not imgs:
            continue
        H, W = imgs[0].shape[:2]
        # --- stock: yolov11_detector.inference()
        x = np.stack([cv2.resize(im, (352, 352), interpolation=cv2.INTER_LINEAR) for im in imgs])
        x = torch.from_numpy(x.transpose(0, 3, 1, 2)).cuda().float() / 255.0
        with torch.no_grad():
            res = model(x, device=0, half=True, verbose=False)
        for i, r in zip(idx, res):
            b = r.boxes
            keep = b.cls.cpu().numpy() == 0
            xyxy = b.xyxyn.cpu().numpy()[keep] * np.array([W, H, W, H])
            conf = b.conf.cpu().numpy()[keep]
            out['stock'][i] = np.concatenate([xyxy.astype(int), conf[:, None]], 1) if keep.any() else np.zeros((0, 5))
        # --- tuned: default ultralytics preprocessing
        with torch.no_grad():
            res = model.predict(imgs, imgsz=640, classes=[0], device=0, half=True, verbose=False)
        for i, r in zip(idx, res):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy(); conf = b.conf.cpu().numpy()
            out['tuned'][i] = np.concatenate([xyxy.astype(int), conf[:, None]], 1) if len(xyxy) else np.zeros((0, 5))
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(C.SAMPLE))
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    model = YOLO(C.YOLO_WEIGHTS)
    for ds in args.datasets:
        trajs = C.sample_trajs(ds)
        t0 = time.time(); nf = 0
        for k, traj in enumerate(trajs):
            p = C.yolo_path(ds, traj)
            if os.path.exists(p) and not args.force:
                continue
            out, n = run_traj(model, ds, traj)
            nf += n
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'wb') as f:
                pickle.dump(out, f)
            print(f'[{ds} {k + 1}/{len(trajs)} {time.time() - t0:.0f}s] {traj}: {n} frames, '
                  f'{sum(len(v) for v in out["stock"].values())} stock / {sum(len(v) for v in out["tuned"].values())} tuned person boxes', flush=True)
        print(f'{ds}: {nf} frames in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
