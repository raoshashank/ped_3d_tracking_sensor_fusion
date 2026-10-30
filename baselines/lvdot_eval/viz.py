"""Qualitative panels: camera image with YOLO / LV-DOT / ped3d boxes + bird's-eye view.

    PYTHONNOUSERSITE=1 MPLBACKEND=Agg $P3D_CONDA_ENVS/pytorch3d/bin/python viz.py --dataset go2nus --traj <name> --frames 0 40 80 [--cfg stock]

Image: blue = YOLO person boxes, green = LV-DOT is_human boxes projected as in filterLVBBoxes
(thick = passes size constraint), red = ped3d tracker boxes. BEV: grey = cloud DBSCAN sees,
thin boxes = all clusters, green = is_human, red x = ped3d reference positions.
"""
import argparse
import os
import pickle

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import common as C
import lvdot_port as P
from eval_lvdot import ref_position


def panel(dataset, traj, frame, cfg, out_png):
    T = C.L.Traj(dataset, traj)
    res = pickle.load(open(C.L.result_path(dataset, traj), 'rb'))
    yolo = pickle.load(open(C.yolo_path(dataset, traj), 'rb'))[cfg]
    gTl, cTg = C.ground_frame(T)
    pts = T.lidar(frame)
    yb = yolo.get(frame, np.zeros((0, 5)))
    d = P.detect_frame(pts, gTl, cTg, T.K, yb[:, :4].astype(int), P.CONFIGS[cfg], np.random.default_rng(0))
    img = cv2.cvtColor(T.image(frame), cv2.COLOR_BGR2RGB)
    for b in res['frames'][frame]:
        x, y, w, h = b['box'].astype(int)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    for y in yb:
        cv2.rectangle(img, (int(y[0]), int(y[1])), (int(y[2]), int(y[3])), (0, 90, 255), 2)
    for b in d['humans']:
        pb = P.project_box(b, cTg, T.K)
        if pb:
            cv2.rectangle(img, (pb[0], pb[1]), (pb[2], pb[3]), (0, 200, 0), 4 if P.size_ok(b, P.CONFIGS[cfg]) else 2)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax[0].imshow(img); ax[0].set_axis_off()
    ax[0].set_title(f'{dataset} / {traj[:40]} f{frame} [{cfg}]  blue=YOLO  green=LV-DOT human  red=ped3d')
    q = P.preprocess((gTl[:3, :3] @ pts.T).T + gTl[:3, 3], P.CONFIGS[cfg], np.random.default_rng(0))
    ax[1].scatter(-q[:, 1], q[:, 0], s=1, c='0.6')
    for b in d['clusters']:
        c, s = b['center'], b['size']
        ax[1].add_patch(Rectangle((-(c[1] + s[1] / 2), c[0] - s[0] / 2), s[1], s[0], fill=False,
                                  ec='g' if b['is_human'] else '0.3', lw=2.5 if b['is_human'] else 0.6))
    refs = [ref_position(b)[0] for b in res['frames'][frame]]
    refs = np.array([r for r in refs if r is not None]).reshape(-1, 2)
    if len(refs):
        ax[1].scatter(-refs[:, 1], refs[:, 0], marker='x', c='r', s=80, lw=2)
    ax[1].set_xlim(-8, 8); ax[1].set_ylim(-1, 12); ax[1].set_aspect('equal'); ax[1].grid(alpha=.3)
    ax[1].set_xlabel('left  <-  y  ->  right   [m]'); ax[1].set_ylabel('forward x [m]')
    ax[1].set_title(f'{len(d["clusters"])} clusters, {len(d["humans"])} human, {len(d["dynamic"])} pass size')
    fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--traj', required=True)
    ap.add_argument('--frames', nargs='+', type=int, required=True); ap.add_argument('--cfg', default='stock')
    ap.add_argument('--out', default=os.path.join(C.OUT_ROOT, 'viz'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in a.frames:
        p = os.path.join(a.out, f'{a.dataset}_{a.traj[:30]}_f{f}_{a.cfg}.png')
        panel(a.dataset, a.traj, f, a.cfg, p); print(p)


if __name__ == '__main__':
    main()
