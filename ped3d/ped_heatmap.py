"""Heatmaps of pedestrian positions around the robot per sub-dataset (location group).

    $P3D_CONDA_ENVS/calib/bin/python ped_heatmap.py --dataset go2nus [--dataset miraikan ...]
    $P3D_CONDA_ENVS/calib/bin/python ped_heatmap.py --all

Uses the fused (VGGT + lidar, smoothed) estimates from $P3D_PED3D_OUT/<dataset>/*.pkl at
detected frames, in the robot frame (x forward, y left). Colour = expected number of pedestrians per
frame per m^2 (log scale), so groups with different numbers of trajectories are comparable.
Writes <dataset>/_heatmap_<dataset>.png and _heatmap_<dataset>.npz.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import ped3d_lib as L


def plot_dataset(dataset, bin_m=0.25, max_range=12.0, min_trajs=1):
    groups = L.collect_positions(dataset, max_range=max_range + 3)
    if not groups:
        print(f'{dataset}: no processed trajectories')
        return None
    names = ['ALL'] + sorted(g for g in groups if g != 'ALL' and groups[g]['n_trajs'] >= min_trajs)
    x_range, y_range = (-4.0, max_range), (-max_range * 0.75, max_range * 0.75)
    maps = {g: L.heatmap(groups[g]['xy'], groups[g]['n_frames'], bin_m, x_range, y_range) for g in names}
    vmax = max(m[0].max() for m in maps.values()) or 1.0
    vmin = vmax / 1000.0
    ncol = min(4, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow), squeeze=False)
    for ax, g in zip(axes.ravel(), names):
        H, xe, ye = maps[g]
        e = groups[g]
        im = ax.pcolormesh(ye, xe, np.maximum(H, vmin), cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax), shading='flat')
        ax.set_aspect('equal')
        ax.invert_xaxis()   # y-left positive to the left of the plot, so it reads like a top-down view
        th = np.linspace(0, 2 * np.pi, 100)
        for r in (2, 5, 10):
            ax.plot(r * np.sin(th), r * np.cos(th), color='w', lw=0.5, alpha=0.5)
        ax.arrow(0, 0, 0, 1.2, color='cyan', width=0.12, head_width=0.5, length_includes_head=True)
        peds_per_frame = len(e['xy']) / max(e['n_frames'], 1)
        ax.set_title(f"{g}\n{e['n_trajs']} traj, {e['n_frames']} frames, {len(e['xy'])} det "
                     f"({peds_per_frame:.2f}/frame)", fontsize=9)
        ax.set_xlabel('y left [m]'); ax.set_ylabel('x forward [m]')
        ax.set_xlim(y_range[1], y_range[0]); ax.set_ylim(x_range)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='pedestrians / frame / m²')
    fig.suptitle(f'{dataset}: pedestrian positions around the robot (fused VGGT+lidar, detected frames, <{max_range:.0f} m)')
    out = os.path.join(L.OUT_ROOT, dataset, f'_heatmap_{dataset}')
    fig.savefig(out + '.png', dpi=130, bbox_inches='tight')
    np.savez_compressed(out + '.npz', groups=np.array(names),
                        **{f'H_{g}': maps[g][0] for g in names},
                        xedges=maps[names[0]][1], yedges=maps[names[0]][2],
                        n_frames=np.array([groups[g]['n_frames'] for g in names]),
                        n_trajs=np.array([groups[g]['n_trajs'] for g in names]))
    plt.close(fig)
    print(f'{dataset}: {len(names) - 1} groups -> {out}.png')
    for g in names:
        e = groups[g]
        print(f'   {g:32s} {e["n_trajs"]:4d} traj {e["n_frames"]:7d} frames {len(e["xy"]):8d} det  '
              f'{len(e["xy"]) / max(e["n_frames"], 1):.2f} peds/frame')
    return out + '.png'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', action='append', choices=list(L.DATASETS))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--bin', type=float, default=0.25)
    ap.add_argument('--range', type=float, default=12.0)
    args = ap.parse_args()
    dss = list(L.DATASETS) if args.all or not args.dataset else args.dataset
    for ds in dss:
        plot_dataset(ds, bin_m=args.bin, max_range=args.range)


if __name__ == '__main__':
    main()
