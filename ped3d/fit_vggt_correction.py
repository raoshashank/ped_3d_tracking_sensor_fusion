"""Fit the per-dataset VGGT-Omega pedestrian depth correction from processed ped3d results.

    /home/shashank/mambaforge/envs/calib/bin/python fit_vggt_correction.py [--dataset go2nus ...]

For each dataset, pairs (raw VGGT depth, independent lidar depth) of pedestrian instances with
>= GOOD_LIDAR_PTS lidar points and lidar depth < 12 m are binned by VGGT depth (1-10 m); the
median lidar/VGGT ratio per bin is fitted with ratio = a + b d + c d^2 and written to
vggt_correction.json (loaded by ped3d_lib.vggt_correct). Prints the before/after bias and MAE
per range band. Reprocess with run_ped3d.py --force afterwards.
"""
import argparse
import json
import os
import pickle

import numpy as np

import ped3d_lib as L

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vggt_correction.json')


def pairs(dataset):
    d = os.path.join(L.OUT_ROOT, dataset)
    vg, li = [], []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.pkl'):
            continue
        with open(os.path.join(d, f), 'rb') as fh:
            res = pickle.load(fh)
        for insts in res['frames']:
            for inst in insts:
                if inst['n_lidar'] >= L.GOOD_LIDAR_PTS and np.isfinite(inst['depth_vggt']) and inst['depth_lidar'] < 12:
                    vg.append(inst['depth_vggt'])
                    li.append(inst['depth_lidar'])
    return np.array(vg), np.array(li)


def fit(vg, li, lo=1.0, hi=10.0, bin_m=0.5):
    c, r = [], []
    for b in np.arange(lo, hi, bin_m):
        m = (vg >= b) & (vg < b + bin_m)
        if m.sum() >= 200:
            c.append(b + bin_m / 2)
            r.append(np.median(li[m] / vg[m]))
    c, r = np.array(c), np.array(r)
    if len(c) < 4:
        return None
    A = np.vstack([np.ones_like(c), c, c ** 2]).T
    return tuple(float(x) for x in np.linalg.lstsq(A, r, rcond=None)[0])


def report(name, f, vg, li):
    e = f(vg) - li
    cells = []
    for lo in (2, 4, 6, 8):
        m = (li >= lo) & (li < lo + 2)
        cells.append(f'{lo}-{lo + 2}m bias {np.median(e[m]):+.2f} MAE {np.median(np.abs(e[m])):.2f}')
    print(f'   {name:10s} ' + ' | '.join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', action='append', choices=list(L.DATASETS))
    args = ap.parse_args()
    coeffs = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in (args.dataset or list(L.DATASETS)):
        vg, li = pairs(ds)
        print(f'== {ds}: {len(vg)} pedestrian lidar/VGGT pairs')
        if len(vg) < 2000:
            print('   too few pairs, keeping default')
            continue
        q = fit(vg, li)
        if q is None:
            print('   fit failed, keeping default')
            continue
        a, b, c = q
        print(f'   ratio = {a:.4f} + {b:.5f} d + {c:.6f} d^2')
        report('raw', lambda d: d, vg, li)
        report('go2nus fit', lambda d: L.vggt_correct(d, 'go2nus'), vg, li)
        dc = np.clip(vg, 1.0, L.VGGT_CORR_MAX_D)
        report('own fit', lambda d: d * (a + b * dc + c * dc * dc), vg, li)
        coeffs[ds] = [a, b, c]
    with open(OUT, 'w') as f:
        json.dump(coeffs, f, indent=1)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
