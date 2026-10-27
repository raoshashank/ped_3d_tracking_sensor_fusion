"""Score the LV-DOT port against the ped3d pedestrian estimates on the sampled trajectories.

    PYTHONNOUSERSITE=1 /home/shashank/mambaforge/envs/pytorch3d/bin/python eval_lvdot.py [--datasets ...] [--workers 8]

Reference per frame = ped3d instances (tracker box + fused lidar/VGGT position, base frame) within
EVAL_RANGE. For each config (stock / tuned) and frame the port returns DBSCAN clusters, is_human
boxes and size-constrained "dynamic" boxes; matching is greedy on xy distance <= MATCH_R.

Writes /media/shashank/ssd2/lvdot_eval/results_<ds>.pkl with
  refs   : one row per reference pedestrian (matched?, distance, 2D YOLO IOU, nearest cluster ...)
  frames : one row per frame (counts, timing)
  extras : one row per is_human box with no reference within 1 m (for inspection)
"""
import argparse
import os
import pickle
import time
import traceback
from multiprocessing import Pool

import numpy as np

import common as C
import lvdot_port as P

CFGS = ['stock', 'tuned']


def greedy_match(a_xy, b_xy, r):
    """-> (idx_b for each a or -1, distance). Greedy nearest pairs first."""
    ma = np.full(len(a_xy), -1); da = np.full(len(a_xy), np.nan)
    if len(a_xy) == 0 or len(b_xy) == 0:
        return ma, da
    D = np.linalg.norm(a_xy[:, None, :] - b_xy[None, :, :], axis=2)
    used_b = set()
    for k in np.argsort(D, axis=None):
        i, j = divmod(k, D.shape[1])
        if D[i, j] > r:
            break
        if ma[i] >= 0 or j in used_b:
            continue
        ma[i] = j; da[i] = D[i, j]; used_b.add(j)
    return ma, da


def ref_position(inst):
    if inst['n_lidar_g'] >= 4 and np.all(np.isfinite(inst['pos_lidar_g'])):
        return inst['pos_lidar_g'][:2], True
    if np.all(np.isfinite(inst['pos_vggt_c'])):
        return inst['pos_vggt_c'][:2], False
    return None, False


def eval_traj(arg):
    dataset, traj = arg
    try:
        return _eval_traj(dataset, traj), None
    except Exception:
        return None, f'{traj}: ' + traceback.format_exc(limit=3)


def _eval_traj(dataset, traj):
    T = C.L.Traj(dataset, traj)
    with open(C.L.result_path(dataset, traj), 'rb') as f:
        res = pickle.load(f)
    with open(C.yolo_path(dataset, traj), 'rb') as f:
        yolo = pickle.load(f)
    g_T_lidar, cam_T_g = C.ground_frame(T)
    K = T.K
    rng = np.random.default_rng(0)
    refs, frames, extras = [], [], []
    for i in range(res['n']):
        pts = T.lidar(i)
        if pts is None:
            continue
        insts = res['frames'][i]
        ref_xy, ref_lidar, ref_box, ref_n = [], [], [], []
        for inst in insts:
            xy, has_l = ref_position(inst)
            if xy is None:
                continue
            ref_xy.append(xy); ref_lidar.append(has_l); ref_box.append(inst['box']); ref_n.append(inst['n_lidar_g'])
        ref_xy = np.asarray(ref_xy).reshape(-1, 2)
        rng_m = np.hypot(ref_xy[:, 0], ref_xy[:, 1]) if len(ref_xy) else np.zeros(0)
        inrange = rng_m <= C.EVAL_RANGE
        frow = dict(dataset=dataset, traj=traj, frame=i, n_ref=int(inrange.sum()), n_ref_all=len(ref_xy), n_pts=len(pts))
        per_cfg = {}
        for cfg in CFGS:
            yb = yolo[cfg].get(i, np.zeros((0, 5)))
            d = P.detect_frame(pts, g_T_lidar, cam_T_g, K, yb[:, :4].astype(int), P.CONFIGS[cfg], rng)
            hum_xy = np.asarray([b['center'][:2] for b in d['humans']]).reshape(-1, 2)
            dyn_xy = np.asarray([b['center'][:2] for b in d['dynamic']]).reshape(-1, 2)
            clu_xy = np.asarray([b['center'][:2] for b in d['clusters']]).reshape(-1, 2)
            m_h, d_h = greedy_match(ref_xy, hum_xy, C.MATCH_R)
            m_d, d_d = greedy_match(ref_xy, dyn_xy, C.MATCH_R)
            # nearest cluster of any kind (3D front end alone)
            if len(clu_xy) and len(ref_xy):
                d_c = np.linalg.norm(ref_xy[:, None] - clu_xy[None], axis=2).min(1)
            else:
                d_c = np.full(len(ref_xy), np.nan)
            # 2D detector vs tracker box
            iou_y = np.zeros(len(ref_xy))
            for k, bx in enumerate(ref_box):
                a = (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])
                iou_y[k] = max([P.iou2d(a, y[:4]) for y in yb], default=0.0)
            per_cfg[cfg] = dict(m_h=m_h, d_h=d_h, m_d=m_d, d_d=d_d, d_c=d_c, iou_y=iou_y)
            # extras: human boxes with no ped3d instance (any range) within 1 m
            m_extra, _ = greedy_match(hum_xy, ref_xy, 1.0) if len(ref_xy) else (np.full(len(hum_xy), -1), None)
            for k, b in enumerate(d['humans']):
                if m_extra[k] < 0:
                    extras.append(dict(dataset=dataset, traj=traj, frame=i, cfg=cfg,
                                       x=float(b['center'][0]), y=float(b['center'][1]), z=float(b['center'][2]),
                                       sx=float(b['size'][0]), sy=float(b['size'][1]), sz=float(b['size'][2]),
                                       n_pts=len(b['pts']), size_ok=P.size_ok(b, P.CONFIGS[cfg]),
                                       yolo_conf=float(yb[b['yolo'], 4]) if b['yolo'] >= 0 else np.nan))
            frow.update({f'{cfg}_n_in': d['n_in'], f'{cfg}_n_clusters': len(d['clusters']),
                         f'{cfg}_n_yolo': len(yb), f'{cfg}_n_human': len(d['humans']), f'{cfg}_n_dynamic': len(d['dynamic']),
                         f'{cfg}_n_extra': int((m_extra < 0).sum()),
                         f'{cfg}_t_pre': d['t_pre'], f'{cfg}_t_db': d['t_db'], f'{cfg}_t_fuse': d['t_fuse']})
        frames.append(frow)
        for k in range(len(ref_xy)):
            if not inrange[k]:
                continue
            row = dict(dataset=dataset, traj=traj, frame=i, range=float(rng_m[k]), ref_lidar=bool(ref_lidar[k]),
                       n_lidar=int(ref_n[k]), box_h=float(ref_box[k][3]))
            for cfg in CFGS:
                pc = per_cfg[cfg]
                row.update({f'{cfg}_hit': bool(pc['m_h'][k] >= 0), f'{cfg}_dist': float(pc['d_h'][k]),
                            f'{cfg}_dyn_hit': bool(pc['m_d'][k] >= 0),
                            f'{cfg}_cluster_dist': float(pc['d_c'][k]), f'{cfg}_yolo_iou': float(pc['iou_y'][k])})
            refs.append(row)
    return dict(refs=refs, frames=frames, extras=extras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(C.SAMPLE))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    for ds in args.datasets:
        trajs = C.sample_trajs(ds)[:args.limit]
        t0 = time.time()
        refs, frames, extras, errs = [], [], [], []
        with Pool(args.workers) as pool:
            for k, (out, err) in enumerate(pool.imap_unordered(eval_traj, [(ds, t) for t in trajs])):
                if err:
                    errs.append(err); print(err[-300:]); continue
                refs += out['refs']; frames += out['frames']; extras += out['extras']
                print(f'[{ds} {k + 1}/{len(trajs)} {time.time() - t0:.0f}s] {len(out["frames"])} frames, {len(out["refs"])} refs', flush=True)
        out = os.path.join(C.OUT_ROOT, f'results_{ds}.pkl')
        with open(out, 'wb') as f:
            pickle.dump(dict(refs=refs, frames=frames, extras=extras, errors=errs, trajs=trajs), f)
        print(f'{ds}: {len(frames)} frames, {len(refs)} reference pedestrians, {len(errs)} errors -> {out}')


if __name__ == '__main__':
    main()
