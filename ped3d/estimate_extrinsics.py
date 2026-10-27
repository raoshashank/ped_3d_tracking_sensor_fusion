"""Estimate the camera-to-base extrinsics (cam_T_base, 4x4, same convention as extrinsics.txt) of a
trajectory from VGGT-Omega camera poses + robot odometry, for datasets without a calibration file
(scand_jackal, umich).

    python estimate_extrinsics.py --dataset scand_jackal [--trajs ...] [--report]

Per chunk (the 24-frame VGGT window; poses live in the chunk's own frame):
  1. Umeyama similarity alignment of the VGGT camera centres C_k to the odometry base positions
     P_k = (x, y, 0): P ~ s R C + t  -> world(odom) <- vggt frame
  2. per frame: cam_T_base = cam_T_vggt . vggt_T_odom . odom_T_base(k), odom_T_base from
     traj_data (position, yaw), assuming the base moves on the z=0 plane
  3. robust average over all frames of all chunks with enough motion (>= 1 m path, not straight-line
     only for yaw: the camera yaw is well constrained by the heading, roll/pitch by the alignment of the
     planar path + the odom z=0 plane)
Writes /media/shashank/ssd2/ped3d/extrinsics_est/<dataset>/<traj>/extrinsics.txt and a summary line
(camera height, pitch, yaw, spread across chunks) so the consistency of the estimate can be judged.
"""
import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

import ped3d_lib as L


def umeyama(src, dst):
    """Similarity s, R, t with dst ~ s R src + t (Umeyama 1991). src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1
    R = U @ d @ Vt
    s = np.trace(np.diag(sig) @ d) / (S ** 2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return s, R, t


def estimate(dataset, traj, min_path=1.0, verbose=False):
    proc = os.path.join(L.PROCESSED_ROOT, dataset, traj)
    pz = os.path.join(L.depth_root(dataset), dataset, traj, 'vggt_omega_poses.npz')
    if not os.path.exists(pz):
        return None, 'no poses'
    z = np.load(pz)
    ext, cid = z['extrinsics'], z['chunk_id']
    with open(os.path.join(proc, 'traj_data.pkl'), 'rb') as f:
        td = pickle.load(f)
    pos, yaw = np.asarray(td['position'], float), np.asarray(td['yaw'], float)
    n = min(len(ext), len(pos))
    mats = []
    for c in sorted(set(cid[:n]) - {-1}):
        idx = np.nonzero(cid[:n] == c)[0]
        if len(idx) < 6:
            continue
        P = np.c_[pos[idx], np.zeros(len(idx))]
        path = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        if path < min_path:
            continue
        # yaw needs a heading change or the roll about the path is unconstrained -> require some turning
        C = np.stack([-ext[i, :3, :3].T @ ext[i, :3, 3] for i in idx])
        s, R, t = umeyama(C, P)                       # odom <- vggt
        # residual check
        res = np.linalg.norm((s * (R @ C.T).T + t) - P, axis=1)
        if np.median(res) > 0.25 * max(path, 1e-3) / len(idx) * 6 + 0.15:
            continue
        for i in idx:
            cam_T_vggt = np.eye(4)
            cam_T_vggt[:3, :3] = ext[i, :3, :3]
            cam_T_vggt[:3, 3] = ext[i, :3, 3] * s       # metric
            odom_T_vggt = np.eye(4)
            odom_T_vggt[:3, :3] = R
            odom_T_vggt[:3, 3] = t
            odom_T_base = np.eye(4)
            cy, sy = np.cos(yaw[i]), np.sin(yaw[i])
            odom_T_base[:3, :3] = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
            odom_T_base[:3, 3] = [pos[i, 0], pos[i, 1], 0.0]
            mats.append(cam_T_vggt @ np.linalg.inv(odom_T_vggt) @ odom_T_base)
    if len(mats) < 12:
        return None, f'too little motion ({len(mats)} usable frames)'
    M = np.array(mats)
    # robust average: median translation, chrodal-mean rotation after outlier rejection
    rot = Rotation.from_matrix(M[:, :3, :3])
    q = rot.as_quat()
    q[q[:, 3] < 0] *= -1
    q_mean = q.mean(0); q_mean /= np.linalg.norm(q_mean)
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(q @ q_mean), -1, 1)))
    keep = ang < np.percentile(ang, 70)
    q_mean = q[keep].mean(0); q_mean /= np.linalg.norm(q_mean)
    Rm = Rotation.from_quat(q_mean).as_matrix()
    tm = np.median(M[keep, :3, 3], axis=0)
    cam_T_base = np.eye(4)
    cam_T_base[:3, :3] = Rm
    cam_T_base[:3, 3] = tm
    base_T_cam = np.linalg.inv(cam_T_base)
    # camera pose in base: position + optical axis direction
    axis = base_T_cam[:3, :3] @ np.array([0, 0, 1.0])
    pitch = np.degrees(np.arcsin(-axis[2]))
    yaw_c = np.degrees(np.arctan2(axis[1], axis[0]))
    spread = np.degrees(2 * np.arccos(np.clip(np.abs(q[keep] @ q_mean), -1, 1)))
    info = dict(n_frames=int(keep.sum()), cam_xyz=base_T_cam[:3, 3].round(3).tolist(), pitch_deg=round(float(pitch), 1),
                yaw_deg=round(float(yaw_c), 1), rot_spread_deg=round(float(np.percentile(spread, 75)), 1),
                height_spread=round(float(np.percentile(np.abs(np.linalg.inv(M[keep])[:, 2, 3] - base_T_cam[2, 3]), 75)), 3))
    return cam_T_base, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--trajs', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    root = os.path.join(L.depth_root(args.dataset), args.dataset)
    trajs = args.trajs or sorted(t for t in os.listdir(root) if os.path.exists(os.path.join(root, t, 'vggt_omega_poses.npz')))
    if args.limit:
        trajs = trajs[:args.limit]
    n_ok = 0
    heights, pitches, yaws = [], [], []
    for traj in trajs:
        out = os.path.join(L.EXTRINSICS_EST_ROOT, args.dataset, traj, 'extrinsics.txt')
        if os.path.exists(out) and not args.force:
            n_ok += 1
            continue
        M, info = estimate(args.dataset, traj)
        if M is None:
            print(f'{traj}: {info}')
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savetxt(out, M.reshape(-1))
        heights.append(info['cam_xyz'][2]); pitches.append(info['pitch_deg']); yaws.append(info['yaw_deg'])
        n_ok += 1
        print(f'{traj}: cam xyz {info["cam_xyz"]} pitch {info["pitch_deg"]:+.1f} yaw {info["yaw_deg"]:+.1f} '
              f'(frames {info["n_frames"]}, rot spread p75 {info["rot_spread_deg"]} deg, height spread {info["height_spread"]} m)')
    if heights:
        print(f'{n_ok}/{len(trajs)} ok | camera height median {np.median(heights):.3f} IQR [{np.percentile(heights, 25):.3f}, {np.percentile(heights, 75):.3f}] '
              f'| pitch median {np.median(pitches):+.1f} IQR [{np.percentile(pitches, 25):+.1f}, {np.percentile(pitches, 75):+.1f}] '
              f'| yaw median {np.median(yaws):+.1f} IQR [{np.percentile(yaws, 25):+.1f}, {np.percentile(yaws, 75):+.1f}]')


if __name__ == '__main__':
    main()
