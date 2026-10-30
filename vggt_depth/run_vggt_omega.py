"""Run VGGT-Omega (facebook/VGGT-Omega, 1B @ 512) over a go2nus trajectory's frames
and save per-frame depth + confidence for comparison against lidar.

Run with the `calib` env (torch 2.6 + cu124):
    $P3D_CONDA_ENVS/calib/bin/python run_vggt_omega.py --traj <traj_name>

Frames are processed jointly in chunks (VGGT-Omega is multi-view): depth within a
chunk shares one scene scale, but the scale is arbitrary (NOT metric) and differs
between chunks — align to lidar per frame or per chunk downstream.

Output: <out_root>/<traj>/vggt_omega_depth.npz with
    depths  (T, 384, 688) float16   z-depth at model resolution (848x480 -> 688x384, plain resize)
    conf    (T, 384, 688) float16   depth confidence
    extrinsics (T, 3, 4), intrinsics (T, 3, 3) float32  VGGT-predicted cameras (model res, chunk-local scale)
    chunk_id (T,) int16             which joint-inference chunk each frame belonged to
"""
import argparse
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', default='deck_1_without_overhead_2025-03-26_12-05-52_1_corrected_merged_0')
    ap.add_argument('--data_root', default=f"{p3d_env.P('P3D_PROCESSED')}/go2nus")
    ap.add_argument('--out_root', default=f"{p3d_env.P('P3D_DEPTH')}/go2nus")
    ap.add_argument('--chunk', type=int, default=24, help='frames per joint forward pass')
    args = ap.parse_args()

    img_dir = os.path.join(args.data_root, args.traj, 'imgs')
    n = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    paths = [os.path.join(img_dir, f'{i}.jpg') for i in range(n)]

    ckpt = hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt')
    model = VGGTOmega().to('cuda').eval()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    depths, confs, extrs, intrs, chunk_ids = [], [], [], [], []
    for c, s in enumerate(range(0, n, args.chunk)):
        batch_paths = paths[s:s + args.chunk]
        images = load_and_preprocess_images(batch_paths, image_resolution=512).to('cuda')
        with torch.inference_mode():
            pred = model(images)
        extrinsics, intrinsics = encoding_to_camera(pred['pose_enc'], pred['images'].shape[-2:])
        depths.append(pred['depth'].squeeze(0).squeeze(-1).cpu().to(torch.float16).numpy())
        confs.append(pred['depth_conf'].squeeze(0).cpu().to(torch.float16).numpy())
        extrs.append(extrinsics.squeeze(0).cpu().float().numpy())
        intrs.append(intrinsics.squeeze(0).cpu().float().numpy())
        chunk_ids.append(np.full(len(batch_paths), c, dtype=np.int16))
        print(f'chunk {c}: frames {s}..{s + len(batch_paths) - 1}, '
              f'depth {depths[-1].shape}, vram {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')

    out_dir = os.path.join(args.out_root, args.traj)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'vggt_omega_depth.npz')
    np.savez_compressed(
        out,
        depths=np.concatenate(depths),
        conf=np.concatenate(confs),
        extrinsics=np.concatenate(extrs),
        intrinsics=np.concatenate(intrs),
        chunk_id=np.concatenate(chunk_ids),
    )
    print(f'saved {out}')


if __name__ == '__main__':
    main()
