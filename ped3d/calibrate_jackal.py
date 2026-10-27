"""Calibrate the SCAND jackal camera extrinsics (cam_T_base) against the velodyne of the one ROS1 bag
that has both, using the metric (odometry-scaled) VGGT depth as the dense reference.

    python calibrate_jackal.py [--init-from-estimate] [--frames 120]

SCAND publishes no camera/lidar TF, and the VGGT-pose + odometry estimator only constrains the
rotation (planar motion leaves the lever arm unobservable). Here the 6 DOF are fitted so that lidar
points projected into the image agree in depth with the VGGT depth map at the same pixels:
    cost = robust median | z_lidar - depth_vggt(u, v) |  over sampled frames
Writes the result to every scand_jackal trajectory (fixed rig) under ped3d/extrinsics_est.
"""
import argparse
import os

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import ped3d_lib as L

TRAJ = 'A_Jackal_AHG_Library_Thu_Oct_28_7_merged_0'
PCD = f'/media/shashank/f6cd4c6b-0d44-43e7-9288-faac85e86149/ped3d_depth/lidar_check/scand_jackal/{TRAJ}/pcd'


def load_frames(n_frames):
    T = L.Traj('scand_jackal', TRAJ)
    idx = [i for i in range(T.n) if os.path.exists(f'{PCD}/{i}.npz')]
    idx = idx[::max(1, len(idx) // n_frames)][:n_frames]
    out = []
    for i in idx:
        d = T.depth(i)
        if d is None:
            continue
        pts = np.load(f'{PCD}/{i}.npz')['arr_0'][:, :3]
        pts = pts[np.isfinite(pts).all(1)]
        pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 20]
        base = (T.base_T_lidar[:3, :3] @ pts.T).T + T.base_T_lidar[:3, 3]
        out.append((base.astype(np.float32), d))
    return T, out


def cost(p, frames, K, W, H, ret_stats=False):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    M[:3, 3] = p[3:]
    errs = []
    n_used = 0
    for base, d in frames:
        pc = (M[:3, :3] @ base.T).T + M[:3, 3]
        z = pc[:, 2]
        keep = (z > 1.0) & (z < 15.0)
        pc, z = pc[keep], z[keep]
        if len(z) < 30:
            continue
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(int)
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(int)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inb.sum() < 30:
            continue
        dv = d[v[inb], u[inb]]
        ok = dv > 0
        if ok.sum() < 30:
            continue
        e = np.abs(z[inb][ok] - dv[ok])
        errs.append(np.median(np.minimum(e, 3.0)))     # robust: cap at 3 m
        n_used += int(ok.sum())
    if not errs:
        return (1e3, 0, 0) if ret_stats else 1e3
    c = float(np.mean(errs))
    return (c, len(errs), n_used) if ret_stats else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    T, frames = load_frames(args.frames)
    K, W, H = T.K, T.W, T.H
    print(f'{len(frames)} frames with lidar + depth')

    # start from the pose-based estimate's rotation (its translation is not observable)
    est = np.loadtxt(os.path.join(L.EXTRINSICS_EST_ROOT, 'scand_jackal', TRAJ, 'extrinsics.txt')).reshape(4, 4)
    starts = []
    r_est = Rotation.from_matrix(est[:3, :3]).as_rotvec()
    for h in (0.4, 0.6, 0.8, 1.0):
        # cam_T_base translation for a camera at height h, no lever arm: t = -R @ [0, 0, h]
        starts.append(np.r_[r_est, -Rotation.from_rotvec(r_est).as_matrix() @ np.array([0.2, 0.0, h])])
    best = None
    for s in starts:
        res = minimize(cost, s, args=(frames, K, W, H), method='Powell',
                       options=dict(maxiter=4000, xtol=1e-4, ftol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
        print(f'   start h={-(Rotation.from_rotvec(s[:3]).as_matrix().T @ s[3:])[2]:.2f} -> cost {res.fun:.3f}')
    p = best.x
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    M[:3, 3] = p[3:]
    B = np.linalg.inv(M)
    axis = B[:3, :3] @ np.array([0, 0, 1.0])
    c, nf, npts = cost(p, frames, K, W, H, ret_stats=True)
    print(f'final cost {c:.3f} m over {nf} frames / {npts} lidar pixels')
    print(f'camera in base: xyz {B[:3, 3].round(3).tolist()}  pitch {np.degrees(np.arcsin(-axis[2])):+.1f} deg  '
          f'yaw {np.degrees(np.arctan2(axis[1], axis[0])):+.1f} deg  roll {np.degrees(np.arctan2(B[1, 0], B[0, 0])):+.1f}')
    print(f'estimate-only rotation was pitch {np.degrees(np.arcsin(-(np.linalg.inv(est)[:3, :3] @ [0, 0, 1])[2])):+.1f} deg')
    if args.write:
        n = 0
        for t in sorted(os.listdir(os.path.join(L.PROCESSED_ROOT, 'scand_jackal'))):
            out = os.path.join(L.EXTRINSICS_EST_ROOT, 'scand_jackal', t, 'extrinsics.txt')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savetxt(out, M.reshape(-1))
            n += 1
        print(f'written to {n} trajectories')


if __name__ == '__main__':
    main()
