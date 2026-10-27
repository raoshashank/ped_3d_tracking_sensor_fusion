"""Generate lidar-aligned metric VGGT-Omega depth for all go2nus trajectories.

Pipeline per trajectory:
  1. Run VGGT-Omega (1B @ 512) over the FULL 30fps fpv video
     (regroup_detections/<traj>/_camera_camera_color_image_raw_compressed.avi),
     jointly in chunks of --chunk consecutive frames.
  2. Keep only the frames matching the 4Hz sampled timesteps: nearest-neighbour
     match between the video timestamps (.txt, "<idx> <ns>" per line) and
     processed/<traj>/timestamps.txt — same convention as sample_depth_frames.py.
  3. Align each kept frame to metric with its lidar scan (pcd/<i>.npz):
     scale = median(lidar_z / vggt_depth) at projected lidar pixels, using the
     per-trajectory extrinsics (per-session calibration fixes baked in) and the
     static base_link->rslidar transform. Frames with too few lidar points fall
     back to the trajectory's median scale.
  4. Save 848x480 uint16 grayscale PNGs in millimeters (same convention as the
     video-depth-anything depth/ folders, 0 = invalid, 65535 = clipped):
        <out_root>/<traj>/vggt_omega_depth/<i>.png
     plus vggt_omega_scales.txt ("<i> <scale>" per line, -1 = fallback used).

Run with the calib env:
    /home/shashank/mambaforge/envs/calib/bin/python generate_vggt_omega_depth.py
Resumable: trajectories with a complete output folder are skipped.
"""
import argparse
import os
import time
import traceback

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import _balanced_target_shape

REGROUP_ROOT = '/media/shashank/ssd2/regroup_depth/regroup_detections/go2nus'
PROCESSED_ROOT = '/media/shashank/T1/processed/go2nus'
EXTRINSICS_ROOT = '/media/shashank/T1/processed/extrinsics/go2nus'
INTRINSICS_FILE = '/media/shashank/T1/processed/intrinsics/go2nus/intrinsics.txt'
OUT_ROOT = '/media/shashank/ssd2/processed_depth/go2nus'
VIDEO_NAME = '_camera_camera_color_image_raw_compressed.avi'
TS_NAME = '_camera_camera_color_image_raw_compressed.txt'

IMG_W, IMG_H = 848, 480
MIN_Z_CAM, MAX_RANGE = 0.3, 30.0
MIN_LIDAR_PTS = 50

# base_link <- rslidar (data/go2nus/static_transforms.yaml)
BASE_T_LIDAR = np.eye(4)
BASE_T_LIDAR[:3, 3] = [0.14543, 0.0, 0.13312]


def preprocess_frames(frames_bgr, target_hw):
    """cv2 BGR frames -> model input tensor (N,3,H,W), matching load_and_preprocess_images."""
    th, tw = target_hw
    out = np.empty((len(frames_bgr), th, tw, 3), dtype=np.float32)
    for i, f in enumerate(frames_bgr):
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        out[i] = np.asarray(img.resize((tw, th), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(0, 3, 1, 2)


def lidar_uv_z(pts_xyz, cam_T_lidar, K):
    """Project lidar into the image; return pixel coords (int) and camera z-depth."""
    pts_cam = (cam_T_lidar[:3, :3] @ pts_xyz.T).T + cam_T_lidar[:3, 3]
    z = pts_cam[:, 2]
    keep = (z > MIN_Z_CAM) & (z < MAX_RANGE)
    pts_cam, z = pts_cam[keep], z[keep]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    inb = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return np.stack([u[inb], v[inb]], 1).astype(int), z[inb]


def process_traj(traj, model, K, chunk):
    video_dir = os.path.join(REGROUP_ROOT, traj)
    proc_dir = os.path.join(PROCESSED_ROOT, traj)
    out_dir = os.path.join(OUT_ROOT, traj, 'vggt_omega_depth')

    split_ts = np.loadtxt(os.path.join(proc_dir, 'timestamps.txt'))
    n_out = len(split_ts)
    if os.path.isdir(out_dir) and len([f for f in os.listdir(out_dir) if f.endswith('.png')]) == n_out:
        return 'skip'

    video_ts = np.array([float(l.split()[1]) / 1e9 for l in open(os.path.join(video_dir, TS_NAME))])
    cap = cv2.VideoCapture(os.path.join(video_dir, VIDEO_NAME))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video != len(video_ts):
        cap.release()
        return f'length mismatch: video {n_video} vs ts {len(video_ts)}'

    # split step i -> video frame index
    frame_for_step = np.array([int(np.argmin(np.abs(video_ts - t))) for t in split_ts])
    needed = {}
    for i, fi in enumerate(frame_for_step):
        needed.setdefault(fi, []).append(i)

    cam_T_base = np.loadtxt(os.path.join(EXTRINSICS_ROOT, traj, 'extrinsics.txt')).reshape(4, 4)
    cam_T_lidar = cam_T_base @ BASE_T_LIDAR

    target_hw = _balanced_target_shape(IMG_H / IMG_W, 512, 16)

    # full-video inference, keeping model-res depth only for needed frames
    depths = {}  # video frame idx -> (h,w) float32, chunk-local scale
    buf, buf_idx = [], []

    def flush():
        if not buf:
            return
        images = preprocess_frames(buf, target_hw).to('cuda')
        with torch.inference_mode():
            pred = model(images)
        d = pred['depth'].squeeze(0).squeeze(-1).float().cpu().numpy()
        for j, fi in enumerate(buf_idx):
            if fi in needed:
                depths[fi] = d[j]
        buf.clear()
        buf_idx.clear()

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        buf_idx.append(fi)
        fi += 1
        if len(buf) == chunk:
            flush()
    flush()
    cap.release()
    missing = [f for f in needed if f not in depths]
    if missing:
        return f'{len(missing)} needed frames not decoded (video truncated?)'

    # per-frame lidar scale
    scales = np.full(n_out, np.nan)
    for i, fvi in enumerate(frame_for_step):
        pcd_path = os.path.join(proc_dir, 'pcd', f'{i}.npz')
        if not os.path.exists(pcd_path):
            continue
        uv, lz = lidar_uv_z(np.load(pcd_path)['arr_0'][:, :3], cam_T_lidar, K)
        if len(uv) < MIN_LIDAR_PTS:
            continue
        d = cv2.resize(depths[fvi], (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        dv = d[uv[:, 1], uv[:, 0]]
        ok_pix = dv > 0
        if ok_pix.sum() < MIN_LIDAR_PTS:
            continue
        scales[i] = np.median(lz[ok_pix] / dv[ok_pix])

    if np.all(np.isnan(scales)):
        return 'no lidar scale could be computed'
    fallback = float(np.nanmedian(scales))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(OUT_ROOT, traj, 'vggt_omega_scales.txt'), 'w') as f:
        for i, fvi in enumerate(frame_for_step):
            s = scales[i] if np.isfinite(scales[i]) else fallback
            d = cv2.resize(depths[fvi], (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
            mm = np.clip(d * s * 1000.0, 0, 65535).astype(np.uint16)
            mm[d <= 0] = 0
            cv2.imwrite(os.path.join(out_dir, f'{i}.png'), mm)
            f.write(f'{i} {scales[i] if np.isfinite(scales[i]) else -1:.6f}\n')
    return f'ok ({n_out} frames, {n_video} video frames, fallback used '\
           f'{int(np.sum(~np.isfinite(scales)))}x, median scale {fallback:.2f})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=24)
    ap.add_argument('--trajs', nargs='*', default=None, help='subset; default all')
    args = ap.parse_args()

    K = np.loadtxt(INTRINSICS_FILE, delimiter=',').reshape(3, 3)
    ckpt = hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt')
    model = VGGTOmega().to('cuda').eval()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    trajs = args.trajs or sorted(os.listdir(REGROUP_ROOT))
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    for k, traj in enumerate(trajs):
        try:
            msg = process_traj(traj, model, K, args.chunk)
        except Exception:
            msg = 'EXC ' + traceback.format_exc(limit=3).replace('\n', ' | ')
        if msg == 'skip':
            n_skip += 1
        elif msg.startswith('ok'):
            n_ok += 1
        else:
            n_err += 1
        print(f'[{k + 1}/{len(trajs)} {time.time() - t0:.0f}s ok={n_ok} skip={n_skip} err={n_err}] {traj}: {msg}',
              flush=True)
    print(f'done in {(time.time() - t0) / 3600:.2f} h')


if __name__ == '__main__':
    main()
