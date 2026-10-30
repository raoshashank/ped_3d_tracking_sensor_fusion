"""Tables from results_<ds>.pkl.

    PYTHONNOUSERSITE=1 $P3D_CONDA_ENVS/pytorch3d/bin/python summarize.py [--datasets ...] > summary.md
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

import common as C

CFGS = ['stock', 'tuned']
BINS = [0, 3, 6, 10]


def pct(x):
    return f'{100 * np.mean(x):.0f}%' if len(x) else '-'


def summarize(ds, R, F, X):
    print(f'\n## {ds}: {len(F)} frames / {R.traj.nunique()} trajectories, {len(R)} reference pedestrians within {C.EVAL_RANGE:.0f} m '
          f'({pct(R.ref_lidar)} with a ped3d lidar estimate)\n')
    R = R.copy(); R['bin'] = pd.cut(R.range, BINS, labels=[f'{a}-{b} m' for a, b in zip(BINS[:-1], BINS[1:])])
    rows = []
    for b, g in list(R.groupby('bin', observed=True)) + [('all', R)]:
        row = dict(range=b, n=len(g))
        for c in CFGS:
            row[f'{c} YOLO 2D'] = pct(g[f'{c}_yolo_iou'] >= 0.3)
            row[f'{c} cluster'] = pct(g[f'{c}_cluster_dist'] <= C.MATCH_R)
            row[f'{c} human'] = pct(g[f'{c}_hit'])
            row[f'{c} dynamic'] = pct(g[f'{c}_dyn_hit'])
            d = g.loc[g[f'{c}_hit'], f'{c}_dist']
            row[f'{c} err med/p90'] = f'{d.median():.2f}/{d.quantile(.9):.2f}' if len(d) else '-'
        rows.append(row)
    print('Recall of ped3d pedestrians (YOLO 2D = a YOLO person box with IOU>=0.3 to the tracker box; cluster = any DBSCAN box '
          f'within {C.MATCH_R} m; human = is_human box within {C.MATCH_R} m; dynamic = human + size constraint; err = xy distance to ped3d position [m])\n')
    print(pd.DataFrame(rows).to_string(index=False))
    # lidar-confirmed subset
    g = R[R.ref_lidar & (R.n_lidar >= 10)]
    print(f'\nSubset with >=10 ped3d lidar points ({len(g)}): human recall stock {pct(g.stock_hit)}, tuned {pct(g.tuned_hit)}; '
          f'cluster recall stock {pct(g.stock_cluster_dist <= C.MATCH_R)}, tuned {pct(g.tuned_cluster_dist <= C.MATCH_R)}')
    # extras / precision proxy
    print('\nPer-frame counts (mean): ' + '; '.join(
        f'{c}: {F[f"{c}_n_in"].mean():.0f} pts in, {F[f"{c}_n_clusters"].mean():.1f} clusters, {F[f"{c}_n_yolo"].mean():.2f} YOLO, '
        f'{F[f"{c}_n_human"].mean():.2f} human, {F[f"{c}_n_dynamic"].mean():.2f} dynamic, {F[f"{c}_n_extra"].mean():.2f} human boxes with no ped3d pedestrian within 1 m'
        for c in CFGS))
    print(f'ped3d reference pedestrians per frame: {F.n_ref.mean():.2f} (<= {C.EVAL_RANGE:.0f} m), {F.n_ref_all.mean():.2f} (all)')
    for c in CFGS:
        x = X[X.cfg == c]
        if len(x):
            print(f'{c} extras: {len(x)} boxes, {pct(x.size_ok)} pass size constraint, range med {np.hypot(x.x, x.y).median():.1f} m, '
                  f'pts med {x.n_pts.median():.0f}, height med {x.sz.median():.2f} m, YOLO conf med {x.yolo_conf.median():.2f}; '
                  f'human boxes matched to a ped3d ped: {F[f"{c}_n_human"].sum() - len(x)} of {F[f"{c}_n_human"].sum()} '
                  f'({pct(np.ones(F[f"{c}_n_human"].sum() - len(x)).tolist() + np.zeros(len(x)).tolist())})')
    print('\nTiming per frame (ms, single core): ' + '; '.join(
        f'{c}: pre {1e3 * F[f"{c}_t_pre"].mean():.1f}, DBSCAN {1e3 * F[f"{c}_t_db"].mean():.1f}, fuse {1e3 * F[f"{c}_t_fuse"].mean():.1f}' for c in CFGS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(C.SAMPLE))
    args = ap.parse_args()
    allR, allF, allX = [], [], []
    for ds in args.datasets:
        p = os.path.join(C.OUT_ROOT, f'results_{ds}.pkl')
        if not os.path.exists(p):
            continue
        d = pickle.load(open(p, 'rb'))
        R, F, X = pd.DataFrame(d['refs']), pd.DataFrame(d['frames']), pd.DataFrame(d['extras'])
        summarize(ds, R, F, X)
        allR.append(R); allF.append(F); allX.append(X)
    if len(allR) > 1:
        summarize('ALL', pd.concat(allR), pd.concat(allF), pd.concat(allX))


if __name__ == '__main__':
    main()
