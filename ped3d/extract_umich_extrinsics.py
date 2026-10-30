"""Extract cam_T_base (extrinsics.txt convention) for umich from the raw ROS2 bags' TF tree.

    source /opt/ros/humble/setup.bash
    python3 extract_umich_extrinsics.py [--limit N] [--report]

The Stretch's camera sits on the pan/tilt head, so the base_link -> camera_color_optical_frame chain
runs through dynamic joints published on /tf (only the leaf links are on /tf_static). For each bag we
compose the chain with the median dynamic transform and report how much each dynamic link moved
during the recording.

Images are stored rotated 90 deg clockwise (bag_processing/rotate_images.py), so the saved matrix is
expressed in the rotated camera frame: X' = -Y, Y' = X, Z' = Z (matching ped3d_lib's umich K).

Writes $P3D_PED3D_OUT/extrinsics_est/umich/<traj>/extrinsics.txt (16 numbers).
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

BAG_ROOT = p3d_env.P('P3D_UMICH_BAGS')
OUT_ROOT = _os.path.join(p3d_env.P('P3D_PED3D_OUT'), 'extrinsics_est', 'umich')
PROCESSED_ROOT = _os.path.join(p3d_env.P('P3D_PROCESSED'), 'umich')
BASE, CAM = 'base_link', 'camera_color_optical_frame'
# image rotated 90 deg clockwise -> new camera axes X' = -Y, Y' = X, Z' = Z
R_ROT = np.eye(4)
R_ROT[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]


def tf_to_mat(t):
    M = np.eye(4)
    q = t.transform.rotation
    M[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    tr = t.transform.translation
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


def read_edges(bag_dir, max_dynamic=4000):
    """{(parent, child): [4x4, ...]} from /tf_static (one entry) and /tf (sampled)."""
    db3 = [f for f in os.listdir(bag_dir) if f.endswith('.db3')]
    if not db3:
        return None, None
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    reader.set_filter(StorageFilter(topics=['/tf', '/tf_static']))
    static, dynamic = {}, defaultdict(list)
    n_dyn = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, TFMessage)
        for t in msg.transforms:
            key = (t.header.frame_id.lstrip('/'), t.child_frame_id.lstrip('/'))
            if topic == '/tf_static':
                static.setdefault(key, tf_to_mat(t))
            elif n_dyn < max_dynamic:
                dynamic[key].append(tf_to_mat(t))
                n_dyn += 1
    return static, dynamic


def compose(static, dynamic, src, dst):
    """Path src -> dst through the (undirected) TF graph; returns (matrix, per-edge motion info)."""
    edges = {}
    for k, v in static.items():
        edges[k] = ('static', np.array([v]))
    for k, v in dynamic.items():
        if k not in edges:
            edges[k] = ('dynamic', np.array(v))
    adj = defaultdict(list)
    for (p, c) in edges:
        adj[p].append((c, (p, c), False))
        adj[c].append((p, (p, c), True))
    # BFS
    prev = {src: None}
    queue = [src]
    while queue:
        node = queue.pop(0)
        if node == dst:
            break
        for nxt, key, inv in adj[node]:
            if nxt not in prev:
                prev[nxt] = (node, key, inv)
                queue.append(nxt)
    if dst not in prev:
        return None, None
    chain = []
    node = dst
    while prev[node] is not None:
        parent, key, inv = prev[node]
        chain.append((key, inv))
        node = parent
    chain.reverse()
    M = np.eye(4)
    info = []
    for key, inv in chain:
        kind, mats = edges[key]
        if len(mats) > 1:
            t_spread = float(np.abs(mats[:, :3, 3] - np.median(mats[:, :3, 3], 0)).max())
            q = Rotation.from_matrix(mats[:, :3, :3]).as_quat()
            q[q[:, 3] < 0] *= -1
            qm = np.median(q, 0)
            qm /= np.linalg.norm(qm)
            r_spread = float(np.degrees(2 * np.arccos(np.clip(np.abs(q @ qm), -1, 1))).max())
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(qm).as_matrix()
            T[:3, 3] = np.median(mats[:, :3, 3], 0)
        else:
            T, t_spread, r_spread = mats[0], 0.0, 0.0
        info.append((key, kind, len(mats), t_spread, r_spread))
        M = M @ (np.linalg.inv(T) if inv else T)
    return M, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--report', action='store_true', help='print the chain for the first bag')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    bags = sorted(os.listdir(BAG_ROOT))
    if args.limit:
        bags = bags[:args.limit]
    heights, pitches, rows = [], [], []
    for k, bag in enumerate(bags):
        traj = bag + '_0'          # processed trajectories carry a _0 suffix
        if not os.path.isdir(os.path.join(PROCESSED_ROOT, traj)):
            cand = [d for d in os.listdir(PROCESSED_ROOT) if d.startswith(bag)]
            if not cand:
                continue
            traj = cand[0]
        out = os.path.join(OUT_ROOT, traj, 'extrinsics.txt')
        if os.path.exists(out) and not args.force:
            continue
        static, dynamic = read_edges(os.path.join(BAG_ROOT, bag))
        if static is None:
            print(f'{bag}: no db3')
            continue
        base_T_cam, info = compose(static, dynamic, BASE, CAM)
        if base_T_cam is None:
            print(f'{bag}: no TF path {BASE} -> {CAM} (frames: {len(set([f for e in list(static) + list(dynamic) for f in e]))})')
            continue
        if args.report and k == 0:
            for key, kind, n, ts, rs in info:
                print(f'   {key[0]:28s} -> {key[1]:28s} {kind:7s} n={n:5d} t_spread {ts:.3f} m  r_spread {rs:.2f} deg')
        cam_T_base = R_ROT @ np.linalg.inv(base_T_cam)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savetxt(out, cam_T_base.reshape(-1))
        pos = base_T_cam[:3, 3]
        axis = base_T_cam[:3, :3] @ np.array([0, 0, 1.0])
        pitch = np.degrees(np.arcsin(-axis[2]))
        heights.append(pos[2]); pitches.append(pitch)
        rows.append((traj, pos, pitch, max(i[3] for i in info), max(i[4] for i in info)))
        if k < 5 or args.report:
            print(f'{traj}: camera at {np.round(pos, 3).tolist()} m in base_link, optical-axis pitch {pitch:+.1f} deg '
                  f'(max dynamic-link motion {max(i[3] for i in info):.3f} m / {max(i[4] for i in info):.1f} deg)')
    if heights:
        print(f'{len(heights)} bags | camera height median {np.median(heights):.3f} IQR [{np.percentile(heights, 25):.3f}, {np.percentile(heights, 75):.3f}] '
              f'| pitch median {np.median(pitches):+.1f} IQR [{np.percentile(pitches, 25):+.1f}, {np.percentile(pitches, 75):+.1f}]')


if __name__ == '__main__':
    main()
