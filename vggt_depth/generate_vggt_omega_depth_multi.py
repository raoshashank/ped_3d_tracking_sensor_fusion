"""Multi-dataset variant of generate_vggt_omega_depth.py (gmu / keio / scand_spot).

Same pipeline as the go2nus original: VGGT-Omega (1B @ 512) over consecutive
30fps video frames in chunks, keep the 4Hz-sampled frames, scale each to
metric with its lidar scan, save uint16 mm PNGs + per-frame scales.

Per-dataset differences are held in DATASETS below:
  - gmu:        640x360 _zed2_zed_node_rgb_image_rect_color_compressed.avi, 658 trajs
  - keio:       640x360 anonymized.avi (d455, same robot/layout as miraikan), 206 trajs
  - scand_spot: 1280x720 _image_raw_compressed.avi. Videos are LONG unsplit
                recordings shared by several processed segments; a segment
                "<video>_0[_a_b]" maps to video "<video>", and only the
                segment's frame window (with a small margin) is decoded.

Unlike the original, decoding is windowed to [min needed frame - margin,
max needed frame + margin] — identical result for per-traj videos, required
for scand's shared videos.

extrinsics.txt is cam_T_base (world2cam with world = base_link frame; same
convention as socnav_behavior_learning's mask_dataset.py). The pcd npz files
are RAW lidar-frame points, and the per-robot base->lidar transforms come from
sbpd-dataset-pipeline/robot_data_analysis/move_base_experiments/lidar_odometry_topic_details.yaml.
`--probe` prints lidar/VGGT depth correlation for candidate transforms as a
sanity check.

Run with the calib env:
    /home/shashank/mambaforge/envs/calib/bin/python generate_vggt_omega_depth_multi.py --dataset keio
Resumable: trajectories with a complete output folder are skipped.
"""
import argparse
import os
import re
import time
import traceback

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import _balanced_target_shape

MIN_Z_CAM, MAX_RANGE = 0.3, 30.0
MIN_LIDAR_PTS = 50
WINDOW_MARGIN = 12  # extra consecutive frames decoded on each side of the needed window

def _base_T_lidar(translation, rpy):
    """4x4 base->lidar transform from translation + roll/pitch/yaw (Rz@Ry@Rx)."""
    from scipy.spatial.transform import Rotation
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    M[:3, 3] = translation
    return M


# Per-robot base->lidar transforms from
# sbpd-dataset-pipeline/robot_data_analysis/move_base_experiments/lidar_odometry_topic_details.yaml.
# The pcd npz files hold RAW lidar-frame xyz (process_bags saves them untransformed),
# so projection is always extrinsics(cam_T_base) @ base_T_lidar.
GO2NUS_BASE_T_LIDAR = _base_T_lidar([0.1453, 0.0, 0.13312], [0, 0, 0])
CABOT_BASE_T_LIDAR = _base_T_lidar([0.248, -0.003, 0.522], [0, 0, 0.002])   # miraikan + keio
GMU_BASE_T_LIDAR = _base_T_lidar([0.090, 0.0, 0.250], [0, 0, 0])
SCAND_SPOT_BASE_T_LIDAR = _base_T_lidar([0.0, 0.0, 0.86], [0, 0, 0])


def _scand_video_for_traj(traj):
    m = re.match(r'^(.*)_0(?:_\d+_\d+)?$', traj)
    return m.group(1) if m else traj


DATASETS = {
    'gmu': dict(
        regroup_root='/media/shashank/ssd2/regroup_depth/regroup_detections/gmu',
        processed_root='/media/shashank/T1/processed/gmu',
        extrinsics_root='/media/shashank/T1/processed/extrinsics/gmu',
        intrinsics_file='/media/shashank/T1/processed/intrinsics/gmu/intrinsics.txt',
        out_root='/media/shashank/ssd2/processed_depth/gmu',
        video_name='_zed2_zed_node_rgb_image_rect_color_compressed.avi',
        ts_name='_zed2_zed_node_rgb_image_rect_color_compressed.txt',
        img_w=640, img_h=360,
        base_T_lidar=GMU_BASE_T_LIDAR,
        video_for_traj=lambda t: t,
    ),
    'keio': dict(
        regroup_root='/media/shashank/ssd2/regroup_depth/regroup_detections/keio',
        processed_root='/media/shashank/T1/processed/keio',
        extrinsics_root='/media/shashank/T1/processed/extrinsics/keio',
        intrinsics_file='/media/shashank/T1/processed/intrinsics/keio/intrinsics.txt',
        out_root='/media/shashank/ssd2/processed_depth/keio',
        video_name='anonymized.avi',
        ts_name='_d455_rs1_color_image_raw_compressed_raw.txt',
        img_w=640, img_h=360,
        base_T_lidar=CABOT_BASE_T_LIDAR,
        video_for_traj=lambda t: t,
    ),
    'scand_spot': dict(
        regroup_root='/media/shashank/ssd2/regroup_depth/regroup_detections/scand_boxmotions/scand_spot',
        processed_root='/media/shashank/T1/processed/scand_spot',
        extrinsics_root='/media/shashank/T1/processed/extrinsics/scand_spot',
        intrinsics_file='/media/shashank/T1/processed/intrinsics/scand_spot/intrinsics.txt',
        out_root='/media/shashank/ssd2/processed_depth/scand_spot',
        video_name='_image_raw_compressed.avi',
        ts_name='_image_raw_compressed.txt',
        img_w=1280, img_h=720,
        base_T_lidar=SCAND_SPOT_BASE_T_LIDAR,
        video_for_traj=_scand_video_for_traj,
    ),
}


def preprocess_frames(frames_bgr, target_hw):
    th, tw = target_hw
    out = np.empty((len(frames_bgr), th, tw, 3), dtype=np.float32)
    for i, f in enumerate(frames_bgr):
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        out[i] = np.asarray(img.resize((tw, th), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(0, 3, 1, 2)


def lidar_uv_z(pts_xyz, cam_T_lidar, K, img_w, img_h):
    pts_cam = (cam_T_lidar[:3, :3] @ pts_xyz.T).T + cam_T_lidar[:3, 3]
    z = pts_cam[:, 2]
    keep = (z > MIN_Z_CAM) & (z < MAX_RANGE)
    pts_cam, z = pts_cam[keep], z[keep]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    inb = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return np.stack([u[inb], v[inb]], 1).astype(int), z[inb]


def cam_T_lidar_for_traj(cfg, traj):
    E = np.loadtxt(os.path.join(cfg['extrinsics_root'], traj, 'extrinsics.txt')).reshape(4, 4)
    return E if cfg['base_T_lidar'] is None else E @ cfg['base_T_lidar']


def infer_depths(cfg, model, traj, needed, chunk):
    """Run windowed chunked inference; return {video frame idx -> model-res depth} or error str."""
    video_dir = os.path.join(cfg['regroup_root'], cfg['video_for_traj'](traj))
    video_ts = np.array([float(l.split()[1]) / 1e9 for l in open(os.path.join(video_dir, cfg['ts_name']))])
    cap = cv2.VideoCapture(os.path.join(video_dir, cfg['video_name']))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video != len(video_ts):
        cap.release()
        return None, video_ts, f'length mismatch: video {n_video} vs ts {len(video_ts)}'

    lo = max(0, min(needed) - WINDOW_MARGIN)
    hi = min(n_video - 1, max(needed) + WINDOW_MARGIN)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)

    target_hw = _balanced_target_shape(cfg['img_h'] / cfg['img_w'], 512, 16)
    depths, buf, buf_idx = {}, [], []

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

    for fi in range(lo, hi + 1):
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        buf_idx.append(fi)
        if len(buf) == chunk:
            flush()
    flush()
    cap.release()
    missing = [f for f in needed if f not in depths]
    if missing:
        return None, video_ts, f'{len(missing)} needed frames not decoded (video truncated?)'
    return depths, video_ts, None


def frame_map(cfg, traj):
    video_dir = os.path.join(cfg['regroup_root'], cfg['video_for_traj'](traj))
    video_ts = np.array([float(l.split()[1]) / 1e9 for l in open(os.path.join(video_dir, cfg['ts_name']))])
    split_ts = np.loadtxt(os.path.join(cfg['processed_root'], traj, 'timestamps.txt'), ndmin=1)
    return np.array([int(np.argmin(np.abs(video_ts - t))) for t in split_ts]), split_ts


def process_traj(cfg, traj, model, K, chunk):
    proc_dir = os.path.join(cfg['processed_root'], traj)
    out_dir = os.path.join(cfg['out_root'], traj, 'vggt_omega_depth')
    W, H = cfg['img_w'], cfg['img_h']

    frame_for_step, split_ts = frame_map(cfg, traj)
    n_out = len(split_ts)
    if os.path.isdir(out_dir) and len([f for f in os.listdir(out_dir) if f.endswith('.png')]) == n_out:
        return 'skip'

    needed = {}
    for i, fi in enumerate(frame_for_step):
        needed.setdefault(fi, []).append(i)

    depths, _, err = infer_depths(cfg, model, traj, needed, chunk)
    if err:
        return err

    cam_T_lidar = cam_T_lidar_for_traj(cfg, traj)
    scales = np.full(n_out, np.nan)
    for i, fvi in enumerate(frame_for_step):
        pcd_path = os.path.join(proc_dir, 'pcd', f'{i}.npz')
        if not os.path.exists(pcd_path):
            continue
        uv, lz = lidar_uv_z(np.load(pcd_path)['arr_0'][:, :3], cam_T_lidar, K, W, H)
        if len(uv) < MIN_LIDAR_PTS:
            continue
        d = cv2.resize(depths[fvi], (W, H), interpolation=cv2.INTER_LINEAR)
        dv = d[uv[:, 1], uv[:, 0]]
        ok_pix = dv > 0
        if ok_pix.sum() < MIN_LIDAR_PTS:
            continue
        scales[i] = np.median(lz[ok_pix] / dv[ok_pix])

    if np.all(np.isnan(scales)):
        return 'no lidar scale could be computed'
    fallback = float(np.nanmedian(scales))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(cfg['out_root'], traj, 'vggt_omega_scales.txt'), 'w') as f:
        for i, fvi in enumerate(frame_for_step):
            s = scales[i] if np.isfinite(scales[i]) else fallback
            d = cv2.resize(depths[fvi], (W, H), interpolation=cv2.INTER_LINEAR)
            mm = np.clip(d * s * 1000.0, 0, 65535).astype(np.uint16)
            mm[d <= 0] = 0
            cv2.imwrite(os.path.join(out_dir, f'{i}.png'), mm)
            f.write(f'{i} {scales[i] if np.isfinite(scales[i]) else -1:.6f}\n')
    return f'ok ({n_out} frames, fallback used {int(np.sum(~np.isfinite(scales)))}x, median scale {fallback:.2f})'


def probe(cfg, model, K, chunk, traj=None):
    """Test candidate extrinsics conventions on one trajectory: correlation between
    VGGT depth and projected lidar depth at the first few steps with lidar."""
    trajs = sorted(os.listdir(cfg['processed_root'])) if traj is None else [traj]
    traj = next(t for t in trajs
                if os.path.exists(os.path.join(cfg['processed_root'], t, 'timestamps.txt'))
                and os.path.isdir(os.path.join(cfg['processed_root'], t, 'pcd')))
    print(f'probe trajectory: {traj}')
    frame_for_step, _ = frame_map(cfg, traj)
    steps = [i for i in range(len(frame_for_step))
             if os.path.exists(os.path.join(cfg['processed_root'], traj, 'pcd', f'{i}.npz'))][:6]
    needed = {frame_for_step[i]: [i] for i in steps}
    depths, _, err = infer_depths(cfg, model, traj, needed, chunk)
    if err:
        print('probe failed:', err)
        return
    E = np.loadtxt(os.path.join(cfg['extrinsics_root'], traj, 'extrinsics.txt')).reshape(4, 4)
    W, H = cfg['img_w'], cfg['img_h']
    candidates = [('extrinsics only', E),
                  ('extrinsics @ go2nus base_T_lidar', E @ GO2NUS_BASE_T_LIDAR),
                  ('inv(extrinsics)', np.linalg.inv(E))]
    if cfg['base_T_lidar'] is not None:
        candidates.insert(1, ('extrinsics @ dataset base_T_lidar', E @ cfg['base_T_lidar']))
    for name, M in candidates:
        cs, rs, np_ = [], [], 0
        for i in steps:
            pts = np.load(os.path.join(cfg['processed_root'], traj, 'pcd', f'{i}.npz'))['arr_0'][:, :3]
            uv, lz = lidar_uv_z(pts, M, K, W, H)
            if len(uv) < MIN_LIDAR_PTS:
                continue
            d = cv2.resize(depths[frame_for_step[i]], (W, H), interpolation=cv2.INTER_LINEAR)
            dv = d[uv[:, 1], uv[:, 0]]
            ok = dv > 0
            if ok.sum() < MIN_LIDAR_PTS:
                continue
            cs.append(np.corrcoef(lz[ok], dv[ok])[0, 1])
            rs.append(np.median(lz[ok] / dv[ok]))
            np_ += len(uv)
        if cs:
            print(f'  {name:36s}: frames={len(cs)} pts={np_} corr={np.mean(cs):.3f} scale={np.median(rs):.2f}')
        else:
            print(f'  {name:36s}: no usable lidar projections')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--chunk', type=int, default=24)
    ap.add_argument('--trajs', nargs='*', default=None, help='subset; default all')
    ap.add_argument('--probe', action='store_true', help='test extrinsics conventions on one trajectory and exit')
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]

    K = np.loadtxt(cfg['intrinsics_file'], delimiter=',').reshape(3, 3)
    ckpt = hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt')
    model = VGGTOmega().to('cuda').eval()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    if args.probe:
        probe(cfg, model, K, args.chunk, args.trajs[0] if args.trajs else None)
        return

    trajs = args.trajs or sorted(t for t in os.listdir(cfg['processed_root'])
                                 if os.path.exists(os.path.join(cfg['processed_root'], t, 'timestamps.txt')))
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    for k, traj in enumerate(trajs):
        try:
            msg = process_traj(cfg, traj, model, K, args.chunk)
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
