"""VGGT-Omega metric depth from the 4 Hz sampled image folders of a processed dataset (no 30 fps
video available, e.g. uex). Same pipeline as generate_vggt_omega_depth_miraikan.py otherwise:
chunked joint inference over consecutive frames, per-frame lidar scale (median lidar_z / vggt_z at
projected lidar pixels), uint16 mm PNGs.

    /home/shashank/mambaforge/envs/calib/bin/python generate_vggt_omega_depth_frames.py --dataset uex [--trajs ...]

--scale-from lidar (default): per-frame median lidar_z / vggt_z (needs pcd + extrinsics)
--scale-from odom: per-chunk scale from the VGGT camera path vs the odometry path (traj_data.pkl);
    no lidar/extrinsics needed. Also writes vggt_omega_poses.npz (per-frame VGGT extrinsics
    world->camera in the chunk frame, chunk id, chunk scale) used by estimate_extrinsics.py.

Resumable: trajectories with a complete output folder are skipped.
"""
import argparse
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ped3d'))
import ped3d_lib as L
from generate_vggt_omega_depth_miraikan import preprocess_frames, lidar_uv_z, MIN_LIDAR_PTS
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import _balanced_target_shape
from vggt_omega.utils.pose_enc import encoding_to_camera
import pickle

import generate_vggt_omega_depth_miraikan as M

OUT_SUBDIR = 'vggt_omega_depth'


def process_traj(dataset, traj, model, K, chunk, scale_from='lidar'):
    cfg = L.DATASETS[dataset]
    W, H = cfg['img_wh']
    M.IMG_W, M.IMG_H = W, H          # lidar_uv_z uses the module globals
    proc = os.path.join(L.PROCESSED_ROOT, dataset, traj)
    img_dir = os.path.join(proc, cfg.get('imgs_dir', 'imgs'))
    out_dir = os.path.join(L.depth_root(dataset), dataset, traj, OUT_SUBDIR)
    n_out = len(np.loadtxt(os.path.join(proc, 'timestamps.txt')))
    files = [os.path.join(img_dir, f'{i}.jpg') for i in range(n_out)]
    if not all(os.path.exists(f) for f in files):
        return f'missing images ({sum(os.path.exists(f) for f in files)}/{n_out})'
    if os.path.isdir(out_dir) and len([f for f in os.listdir(out_dir) if f.endswith('.png')]) == n_out:
        return 'skip'
    cam_T_lidar = None
    if scale_from == 'lidar':
        ext = L.extrinsics_file(dataset, traj)
        if ext is None or cfg.get('base_T_lidar') is None:
            return 'no extrinsics / lidar transform (use --scale-from odom)'
        cam_T_lidar = np.loadtxt(ext).reshape(4, 4) @ cfg['base_T_lidar']
    with open(os.path.join(proc, 'traj_data.pkl'), 'rb') as f:
        odom_xy = np.asarray(pickle.load(f)['position'], float)[:n_out]
    target_hw = _balanced_target_shape(H / W, 512, 16)

    depths = {}
    poses = np.full((n_out, 3, 4), np.nan, np.float32)
    chunk_id = np.full(n_out, -1, int)
    chunk_scale = {}
    buf, buf_idx = [], []

    def flush():
        if not buf:
            return
        images = preprocess_frames(buf, target_hw).to('cuda')
        with torch.inference_mode():
            pred = model(images)
        d = pred['depth'].squeeze(0).squeeze(-1).float().cpu().numpy()
        ext, _ = encoding_to_camera(pred['pose_enc'].float(), target_hw, build_intrinsics=False)
        ext = ext.squeeze(0).cpu().numpy()          # (S, 3, 4) world->camera, chunk frame
        cid = len(chunk_scale)
        for j, fi in enumerate(buf_idx):
            depths[fi] = d[j]
            poses[fi] = ext[j]
            chunk_id[fi] = cid
        # camera centres vs odometry: robust ratio of pairwise distances
        C = np.stack([-ext[j, :3, :3].T @ ext[j, :3, 3] for j in range(len(buf_idx))])
        P = odom_xy[buf_idx]
        ratios = []
        for a in range(len(buf_idx)):
            for b in range(a + 1, len(buf_idx)):
                dp, dc = np.linalg.norm(P[b] - P[a]), np.linalg.norm(C[b] - C[a])
                if dp > 0.3 and dc > 1e-4:
                    ratios.append(dp / dc)
        chunk_scale[cid] = float(np.median(ratios)) if len(ratios) >= 5 else np.nan
        buf.clear()
        buf_idx.clear()

    for i, f in enumerate(files):
        frame = cv2.imread(f)
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))
        buf.append(frame)
        buf_idx.append(i)
        if len(buf) == chunk:
            flush()
    flush()

    scales = np.full(n_out, np.nan)
    if scale_from == 'odom':
        cs = np.array([chunk_scale.get(int(c), np.nan) for c in chunk_id])
        if np.isfinite(cs).sum() == 0:
            return 'no odometry scale (robot static?)'
        cs[~np.isfinite(cs)] = np.nanmedian(cs)
        scales = cs
    for i in range(n_out if scale_from == 'lidar' else 0):
        pcd_path = os.path.join(proc, 'pcd', f'{i}.npz')
        if not os.path.exists(pcd_path):
            continue
        uv, lz = lidar_uv_z(np.load(pcd_path)['arr_0'][:, :3], cam_T_lidar, K)
        if len(uv) < MIN_LIDAR_PTS:
            continue
        d = cv2.resize(depths[i], (W, H), interpolation=cv2.INTER_LINEAR)
        dv = d[uv[:, 1], uv[:, 0]]
        ok = dv > 0
        if ok.sum() < MIN_LIDAR_PTS:
            continue
        scales[i] = np.median(lz[ok] / dv[ok])
    if np.all(np.isnan(scales)):
        return 'no lidar scale could be computed'
    fallback = float(np.nanmedian(scales))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(L.depth_root(dataset), dataset, traj, OUT_SUBDIR.replace('_depth', '_poses') + '.npz'),
                        extrinsics=poses, chunk_id=chunk_id, chunk_scale=np.array([chunk_scale.get(c, np.nan) for c in range(len(chunk_scale))]),
                        scale_from=scale_from, target_hw=np.array(target_hw))
    with open(os.path.join(L.depth_root(dataset), dataset, traj, OUT_SUBDIR.replace('_depth', '_scales') + '.txt'), 'w') as f:
        for i in range(n_out):
            s = scales[i] if np.isfinite(scales[i]) else fallback
            d = cv2.resize(depths[i], (W, H), interpolation=cv2.INTER_LINEAR)
            mm = np.clip(d * s * 1000.0, 0, 65535).astype(np.uint16)
            mm[d <= 0] = 0
            cv2.imwrite(os.path.join(out_dir, f'{i}.png'), mm)
            f.write(f'{i} {scales[i] if np.isfinite(scales[i]) else -1:.6f}\n')
    sc = scales[np.isfinite(scales)]
    return f'ok ({n_out} frames, {scale_from} scale, fallback used {int(np.sum(~np.isfinite(scales)))}x, median scale {fallback:.2f}, ' \
           f'scale IQR/median {(np.percentile(sc, 75) - np.percentile(sc, 25)) / fallback:.2f})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(L.DATASETS))
    ap.add_argument('--chunk', type=int, default=24)
    ap.add_argument('--trajs', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out-subdir', default='vggt_omega_depth', help='output folder name (experiments)')
    ap.add_argument('--scale-from', default='lidar', choices=['lidar', 'odom'])
    args = ap.parse_args()
    global OUT_SUBDIR
    OUT_SUBDIR = args.out_subdir
    K = L.load_intrinsics(args.dataset)
    ckpt = hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt')
    model = VGGTOmega().to('cuda').eval()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    root = os.path.join(L.PROCESSED_ROOT, args.dataset)
    trajs = args.trajs or sorted(t for t in os.listdir(root) if os.path.isdir(os.path.join(root, t)))
    if args.limit:
        trajs = trajs[:args.limit]
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    for k, traj in enumerate(trajs):
        try:
            msg = process_traj(args.dataset, traj, model, K, args.chunk, args.scale_from)
        except Exception:
            msg = 'EXC ' + traceback.format_exc(limit=3).replace('\n', ' | ')
        if msg == 'skip':
            n_skip += 1
        elif msg.startswith('ok'):
            n_ok += 1
        else:
            n_err += 1
        print(f'[{k + 1}/{len(trajs)} {time.time() - t0:.0f}s ok={n_ok} skip={n_skip} err={n_err}] {traj}: {msg}', flush=True)
    print(f'done in {(time.time() - t0) / 3600:.2f} h')


if __name__ == '__main__':
    main()
