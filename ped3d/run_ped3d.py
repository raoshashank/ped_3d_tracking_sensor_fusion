"""Batch pedestrian 3D estimation (ped3d_lib) over a dataset.

    /home/shashank/mambaforge/envs/calib/bin/python run_ped3d.py --dataset go2nus [--limit 50] [--workers 8]

Writes /media/shashank/ssd2/ped3d/<dataset>/<traj>.pkl per trajectory (skipped if present unless
--force) and <dataset>/_metrics.csv with the per-trajectory self-consistency metrics.
"""
import argparse
import csv
import os
import pickle
import time
import traceback
from multiprocessing import Pool

import numpy as np

import ped3d_lib as L


def _one(arg):
    dataset, traj, force = arg
    try:
        res = L.load_or_process(dataset, traj, force=force)
        return traj, res['metrics'], None
    except Exception:
        return traj, None, traceback.format_exc(limit=2).replace('\n', ' | ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(L.DATASETS))
    ap.add_argument('--trajs', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    trajs = args.trajs or L.list_trajs(args.dataset)
    if args.limit:
        trajs = trajs[:args.limit]
    print(f'{len(trajs)} trajectories in {args.dataset}')
    t0 = time.time()
    rows, errs = [], []
    with Pool(args.workers) as pool:
        for k, (traj, m, err) in enumerate(pool.imap_unordered(_one, [(args.dataset, t, args.force) for t in trajs])):
            if err:
                errs.append((traj, err))
                print(f'[{k + 1}/{len(trajs)}] {traj}: ERROR {err[:200]}', flush=True)
                continue
            rows.append(dict(traj=traj, **m))
            if (k + 1) % 20 == 0 or k + 1 == len(trajs):
                print(f'[{k + 1}/{len(trajs)} {time.time() - t0:.0f}s] {traj}: '
                      f'{m["n_det"]} det, lidar-vggt_a {m["agree_lidar_vggt_a_med"]:.2f}m', flush=True)
    # the csv covers every processed trajectory of the dataset, not just this run
    done = {r['traj'] for r in rows}
    ds_dir = os.path.join(L.OUT_ROOT, args.dataset)
    for f in sorted(os.listdir(ds_dir)) if os.path.isdir(ds_dir) else []:
        if f.endswith('.pkl') and f[:-4] not in done:
            with open(os.path.join(ds_dir, f), 'rb') as fh:
                rows.append(dict(traj=f[:-4], **pickle.load(fh)['metrics']))
    if rows:
        out = os.path.join(L.OUT_ROOT, args.dataset, '_metrics.csv')
        keys = list(rows[0].keys())
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'metrics -> {out}')
        tot = sum(r['n_det'] for r in rows)
        print(f'{len(rows)} trajs, {tot} detections within {L.EVAL_RANGE:.0f} m; '
              f'lidar coverage {sum(r["cov_lidar"] for r in rows) / max(tot, 1):.1%}, '
              f'vggt coverage {sum(r["cov_vggt"] for r in rows) / max(tot, 1):.1%}')
        for s in ('vggt', 'vggt_c', 'vggt_a', 'da', 'lidar_g'):
            med = np.array([r[f'agree_lidar_{s}_med'] for r in rows])
            n = np.array([r[f'agree_lidar_{s}_n'] for r in rows])
            ok = np.isfinite(med) & (n > 0)
            print(f'  lidar vs {s:7s}: median-of-medians {np.median(med[ok]):.3f} m '
                  f'(weighted {np.sum(med[ok] * n[ok]) / n[ok].sum():.3f} m, {n[ok].sum()} pairs)')
    if errs:
        print(f'{len(errs)} errors')


if __name__ == '__main__':
    main()
