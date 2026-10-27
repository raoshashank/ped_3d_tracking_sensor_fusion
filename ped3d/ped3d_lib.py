"""Pedestrian 3D position estimation from dense VGGT-Omega depth + sparse lidar.

Per timestep, per tracked pedestrian (regroup_tracks_sampled.pkl box, [x, y, w, h] top-left):
  region   = person mask (masked_imgs, binary) inside the box, eroded; falls back to the
             inner torso box when the mask is missing
  vggt     = robust foreground depth mode of the VGGT-Omega metric depth inside the region
             (pixel weights favour the box centre so an occluder/occludee sharing the box
             does not win), back-projected through K -> camera -> base frame
  vggt_c   = same, with the global range-dependent correction fitted against lidar
             (fit_vggt_correction below; ratio = a + b d + c d^2 on 1-10 m)
  lidar    = lidar points projecting into the region, depth-mode clustered, base frame
  da       = DepthAnything depth aligned per frame with a single lidar scale (baseline)

Track level (non-causal):
  anchor   = per-track median lidar/vggt_c depth ratio where both exist -> rescales vggt_c
             in frames without usable lidar
  fused    = inverse-variance fusion of lidar and anchored vggt_c per frame, then a
             constant-velocity RTS smoother in the odom/world frame with robust
             re-weighting (2 passes); gaps inside a track are interpolated by the smoother.

Frames: extrinsics.txt is cam_T_base; pcd npz are raw lidar-frame points; base_T_lidar per
robot from sbpd-dataset-pipeline/.../lidar_odometry_topic_details.yaml; robot odom pose from
traj_data.pkl (position, yaw), world = odom frame of the trajectory.
"""
import os
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

cv2.setNumThreads(1)

# ----------------------------------------------------------------------------- datasets
PROCESSED_ROOT = '/media/shashank/T1/processed'
DEPTH_ROOT = '/media/shashank/ssd2/processed_depth'
# ssd2 is full; datasets processed later write their depth to the 3.4 TB drive (per-dataset override)
DEPTH_ROOT_ALT = '/media/shashank/f6cd4c6b-0d44-43e7-9288-faac85e86149/ped3d_depth'
MASK_ROOT = '/media/shashank/ssd2/mask_dataset'
OUT_ROOT = '/media/shashank/ssd2/ped3d'


def _base_T_lidar(translation, rpy):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    M[:3, 3] = translation
    return M


DATASETS = {
    'go2nus': dict(img_wh=(848, 480), base_T_lidar=_base_T_lidar([0.1453, 0.0, 0.13312], [0, 0, 0])),
    'miraikan': dict(img_wh=(640, 360), base_T_lidar=_base_T_lidar([0.248, -0.003, 0.522], [0, 0, 0.002])),
    'keio': dict(img_wh=(640, 360), base_T_lidar=_base_T_lidar([0.248, -0.003, 0.522], [0, 0, 0.002])),
    'gmu': dict(img_wh=(640, 360), base_T_lidar=_base_T_lidar([0.090, 0.0, 0.250], [0, 0, 0])),
    'scand_spot': dict(img_wh=(1280, 720), base_T_lidar=_base_T_lidar([0.0, 0.0, 0.86], [0, 0, 0])),
    # uex: no regroup tracks (stock robot tracks.pkl, string ids) and two cameras; the ZED one is calibrated
    'uex': dict(img_wh=(640, 360), base_T_lidar=_base_T_lidar([-0.168, 0.0, 1.077], [0, 0, 0.055]),
                imgs_dir='imgs_zed_zed_node_rgb_image_rect_color', masks_dir='masks_zed_zed_node_rgb_image_rect_color',
                static_extrinsics=True,
                # VGGT places the anonymized silhouettes at background depth in bright halls (+1..+3.7 m vs
                # lidar AND vs ground-contact depth, day dependent); the Helios lidar is dense, so lidar leads
                vggt_trust=0.2),
    # umich (Stretch): RealSense on the pan/tilt head, images stored rotated 90 deg clockwise
    # (imgs = rot90(original, 3), 720x1280); K from _camera_color_camera_info.txt rotated accordingly;
    # extrinsics estimated per trajectory (estimate_extrinsics.py) since the head pose is not static
    'umich': dict(depth_root=DEPTH_ROOT_ALT, img_wh=(720, 1280), base_T_lidar=_base_T_lidar([0.027, -0.023, 1.458], [0.0349065850398865, 0.2700024352835228, 0.0]),
                  intrinsics=np.array([[906.758911, 0.0, 719 - 375.463501], [0.0, 906.704163, 653.498169], [0.0, 0.0, 1.0]])),
    # scand_jackal: no lidar and no extrinsics -> VGGT depth scaled by odometry, extrinsics estimated
    # scand_jackal: SCAND publishes no camera TF -> cam_T_base calibrated against the velodyne of the
    # one ROS1 bag we have (calibrate_jackal.py); processed trajectories carry no pcd, so depth is
    # scaled from odometry and the estimator runs VGGT-only
    'scand_jackal': dict(depth_root=DEPTH_ROOT_ALT, img_wh=(1280, 720),
                         base_T_lidar=_base_T_lidar([0.0, 0.0, 0.35], [0, 0, 0]), static_extrinsics=True),
}
EXTRINSICS_EST_ROOT = os.path.join(OUT_ROOT, 'extrinsics_est')


def depth_root(dataset):
    return DATASETS[dataset].get('depth_root', DEPTH_ROOT)


def load_intrinsics(dataset):
    if 'intrinsics' in DATASETS[dataset]:
        return np.asarray(DATASETS[dataset]['intrinsics'], float)
    return np.loadtxt(os.path.join(PROCESSED_ROOT, 'intrinsics', dataset, 'intrinsics.txt'), delimiter=',').reshape(3, 3)


def tracks_file(proc_dir):
    """regroup tracks when available, else the stock robot tracks (same per-frame (boxes, ids) layout)."""
    for name in ('regroup_tracks_sampled.pkl', 'tracks.pkl'):
        p = os.path.join(proc_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_tracks_pkl(path):
    with open(path, 'rb') as f:
        o = pickle.load(f)
    seq = o['tracks'] if isinstance(o, dict) else o
    # stock tracks carry string ids
    return [(boxes, [int(i) for i in ids]) for boxes, ids in seq]


def extrinsics_file(dataset, traj):
    """Per-trajectory cam_T_base; for robots with a fixed static mount (uex: all 329 files identical and
    equal to the _tf_static chain) fall back to the dataset-wide file when a trajectory has none."""
    p = os.path.join(PROCESSED_ROOT, 'extrinsics', dataset, traj, 'extrinsics.txt')
    if os.path.exists(p):
        return p
    p = os.path.join(EXTRINSICS_EST_ROOT, dataset, traj, 'extrinsics.txt')   # estimated (estimate_extrinsics.py)
    if os.path.exists(p):
        return p
    if DATASETS[dataset].get('static_extrinsics'):
        d = os.path.join(PROCESSED_ROOT, 'extrinsics', dataset)
        for t in sorted(os.listdir(d)):
            q = os.path.join(d, t, 'extrinsics.txt')
            if os.path.exists(q):
                return q
    return None

MIN_Z_CAM, MAX_RANGE = 0.3, 30.0
EVAL_RANGE = 10.0
MIN_REGION_PX = 150          # below this the mask is considered missing -> torso box
MIN_LIDAR_PTS = 4            # minimum lidar points for a lidar estimate at all
GOOD_LIDAR_PTS = 10          # lidar estimates with >= this many points anchor the track
BODY_RADIUS = 0.15           # surface -> body centre push-back along the viewing ray

# VGGT-Omega depth correction: lidar/vggt median ratio at pedestrians vs vggt depth, quadratic on
# 1-10 m. Per-dataset coefficients come from vggt_correction.json (fit_vggt_correction.py); the
# default is the go2nus fit on ped_depth_accuracy_go2nus.npz (331k instances).
VGGT_CORR_DEFAULT = (0.8751, 0.04352, -0.002336)
VGGT_CORR_MAX_D = 10.0
_CORR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vggt_correction.json')
try:
    import json as _json
    with open(_CORR_FILE) as _f:
        VGGT_CORR = {k: tuple(v) for k, v in _json.load(_f).items()}
except (OSError, ValueError):
    VGGT_CORR = {}


def vggt_correct(d, dataset=None):
    dc = np.clip(d, 1.0, VGGT_CORR_MAX_D)
    a, b, c = VGGT_CORR.get(dataset, VGGT_CORR_DEFAULT)
    return d * (a + b * dc + c * dc * dc)


def sigma_vggt(d, dataset=None):
    """1-sigma depth error of corrected VGGT at a pedestrian (residual MAE from the fit); inflated by
    1/vggt_trust for datasets where VGGT is known to be unreliable on pedestrians."""
    return (0.05 + 0.065 * d) / DATASETS.get(dataset, {}).get('vggt_trust', 1.0)


def sigma_lidar(d, n):
    return (0.10 + 0.015 * d) * (1.0 + 4.0 / max(n, 1))


def sigma_da(d):
    return 0.3 + 0.15 * d


# ----------------------------------------------------------------------------- loading
class Traj:
    """Lazy per-trajectory data access."""

    def __init__(self, dataset, traj):
        cfg = DATASETS[dataset]
        self.dataset, self.traj = dataset, traj
        self.W, self.H = cfg['img_wh']
        self.base_T_lidar = cfg['base_T_lidar']
        self.proc = os.path.join(PROCESSED_ROOT, dataset, traj)
        self.depth_dir = os.path.join(depth_root(dataset), dataset, traj)
        self.K = load_intrinsics(dataset)
        self.cam_T_base = np.loadtxt(extrinsics_file(dataset, traj)).reshape(4, 4)
        self.base_T_cam = np.linalg.inv(self.cam_T_base)
        self.cam_T_lidar = self.cam_T_base @ self.base_T_lidar if self.base_T_lidar is not None else None
        self.timestamps = np.loadtxt(os.path.join(self.proc, 'timestamps.txt'))
        with open(os.path.join(self.proc, 'traj_data.pkl'), 'rb') as f:
            td = pickle.load(f)
        self.robot_pos = np.asarray(td['position'], dtype=float)
        self.robot_yaw = np.asarray(td['yaw'], dtype=float)
        self.tracks = load_tracks_pkl(tracks_file(self.proc))
        self.n = min(len(self.timestamps), len(self.tracks), len(self.robot_pos))
        self.img_dir = os.path.join(self.proc, cfg.get('imgs_dir', 'imgs'))
        # person masks live in different places per dataset
        self.mask_dir = None
        for cand in (os.path.join(self.proc, cfg.get('masks_dir', 'masked_imgs')), os.path.join(self.proc, 'masks'),
                     os.path.join(MASK_ROOT, dataset, traj, 'masked_imgs')):
            if os.path.isdir(cand):
                self.mask_dir = cand
                break

    def image(self, i):
        return cv2.imread(os.path.join(self.img_dir, f'{i}.jpg'))

    def mask(self, i):
        if self.mask_dir is None:
            return None
        m = cv2.imread(os.path.join(self.mask_dir, f'{i}.jpg'), cv2.IMREAD_GRAYSCALE)
        return None if m is None else (m > 127)

    def depth(self, i, kind='vggt_omega_depth'):
        d = cv2.imread(os.path.join(self.depth_dir, kind, f'{i}.png'), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        return d.astype(np.float32) / 1000.0

    def lidar(self, i):
        if self.cam_T_lidar is None:
            return None
        p = os.path.join(self.proc, 'pcd', f'{i}.npz')
        return np.load(p)['arr_0'][:, :3] if os.path.exists(p) else None

    def boxes(self, i):
        b, ids = self.tracks[i]
        return np.asarray(b, dtype=float).reshape(-1, 4), list(ids)

    def base_to_world(self, i, p_base_xy):
        c, s = np.cos(self.robot_yaw[i]), np.sin(self.robot_yaw[i])
        R = np.array([[c, -s], [s, c]])
        return (R @ np.asarray(p_base_xy).T).T + self.robot_pos[i]

    def world_to_base(self, i, p_world_xy):
        c, s = np.cos(self.robot_yaw[i]), np.sin(self.robot_yaw[i])
        R = np.array([[c, -s], [s, c]])
        return (R.T @ (np.asarray(p_world_xy) - self.robot_pos[i]).T).T


# ----------------------------------------------------------------------------- geometry
def project_cam(pts_cam, K, W, H):
    """camera-frame points -> (u, v int, z) for points in front of the camera and in-bounds."""
    z = pts_cam[:, 2]
    keep = (z > MIN_Z_CAM) & (z < MAX_RANGE)
    pc, z = pts_cam[keep], z[keep]
    u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(int)
    v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[inb], v[inb], z[inb], pc[inb]


def backproject(u, v, d, K):
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    return np.stack([x, y, d], 1)


def cluster_select(d, peak):
    return np.abs(d - peak) < max(0.35, 0.08 * peak)


def depth_clusters(d, w, rel_bin=0.06):
    """Depth clusters as (centre, weighted mass, count), nearest first.

    Peaks of a weighted histogram on log-spaced bins (~rel_bin wide); each peak's members are
    the values within the cluster_select window, peaks that fall inside another's window merge.
    """
    lo, hi = max(float(d.min()), MIN_Z_CAM), float(d.max()) + 1e-3
    edges = np.exp(np.arange(np.log(lo), np.log(hi) + rel_bin, rel_bin)) if hi > lo else None
    if edges is None or len(edges) < 4:
        return [(float(np.median(d)), float(w.sum()), int(len(d)))]
    h, _ = np.histogram(d, bins=edges, weights=w)
    hs = np.convolve(h, [0.25, 0.5, 0.25], mode='same')[:len(h)]
    centres = np.sqrt(edges[:-1] * edges[1:])
    peaks = [k for k in range(len(hs)) if hs[k] > 0 and hs[k] >= hs[max(k - 1, 0)] and hs[k] >= hs[min(k + 1, len(hs) - 1)]]
    out = []
    for k in sorted(peaks, key=lambda k: -hs[k]):
        c = float(centres[k])
        if any(abs(c - c0) < max(0.35, 0.08 * c0) for c0, _, _ in out):
            continue
        sel = cluster_select(d, c)
        if sel.any():
            out.append((float(np.median(d[sel])), float(w[sel].sum()), int(sel.sum())))
    return sorted(out, key=lambda t: t[0])


def pick_cluster(clusters, guide=None, min_frac=0.5, min_count=1):
    """Nearest cluster with >= min_frac of the heaviest cluster's mass (background inside a
    person mask is always farther than the person); with a guide depth, the cluster closest
    to it in log-depth among those with >= 0.3 of the max mass."""
    cl = [c for c in clusters if c[2] >= min_count]
    if not cl:
        if any(c[2] >= max(min_count // 2, 1) for c in clusters):
            cl = [c for c in clusters if c[2] >= max(min_count // 2, 1)]
        else:
            return None
    mmax = max(m for _, m, _ in cl)
    if guide is not None and np.isfinite(guide):
        ok = [c for c in cl if c[1] >= 0.3 * mmax]
        return min(ok, key=lambda c: abs(np.log(c[0] / guide)))
    ok = [c for c in cl if c[1] >= min_frac * mmax]
    return ok[0]


def region_weights(h, w, sub_v, sub_u):
    """Gaussian weights in normalised box coordinates, centred on the torso."""
    nu = (sub_u + 0.5) / max(w, 1)
    nv = (sub_v + 0.5) / max(h, 1)
    return np.exp(-0.5 * (((nu - 0.5) / 0.22) ** 2 + ((nv - 0.42) / 0.28) ** 2))


def body_centre_base(pts_cam, base_T_cam):
    """Median of surface points, pushed back along the ray by BODY_RADIUS, in base frame."""
    med = np.median(pts_cam, axis=0)
    r = np.linalg.norm(med)
    if r > 0:
        med = med * (r + BODY_RADIUS) / r
    return base_T_cam[:3, :3] @ med + base_T_cam[:3, 3]


# ----------------------------------------------------------------------------- per frame
def _nan3():
    return np.full(3, np.nan)


def _to_base(pts_cam, base_T_cam, cap=None):
    p = (base_T_cam[:3, :3] @ pts_cam.T).T + base_T_cam[:3, 3]
    if cap and len(p) > cap:
        p = p[np.linspace(0, len(p) - 1, cap).astype(int)]
    return p.astype(np.float32)


def estimate_frame(T, i, with_da=True, keep_points=False):
    """Returns a list of instance dicts for timestep i (one per tracked box).

    keep_points adds the base-frame 3D points behind each estimate (pts_vggt, pts_lidar,
    pts_lidar_g) for visualisation; they are not stored in the batch results.
    """
    boxes, ids = T.boxes(i)
    out = []
    if len(ids) == 0:
        return out
    mask = T.mask(i)
    dv_full = T.depth(i, 'vggt_omega_depth')
    da_full = T.depth(i, 'depth') if with_da else None
    lid = T.lidar(i)
    K, W, H = T.K, T.W, T.H

    lu = lv = lz = lpc = None
    da_scale = np.nan
    if lid is not None and len(lid):
        pc = (T.cam_T_lidar[:3, :3] @ lid.T).T + T.cam_T_lidar[:3, 3]
        lu, lv, lz, lpc = project_cam(pc, K, W, H)
        if da_full is not None and len(lz) > 50:
            dd = da_full[lv, lu]
            ok = dd > 0
            if ok.sum() > 50:
                da_scale = float(np.median(lz[ok] / dd[ok]))

    for box, tid in zip(boxes, ids):
        x, y, w, h = box
        x0, y0 = int(max(np.floor(x), 0)), int(max(np.floor(y), 0))
        x1, y1 = int(min(np.ceil(x + w), W)), int(min(np.ceil(y + h), H))
        inst = dict(id=int(tid), box=box.astype(float), flags=[], region_px=0,
                    n_lidar=0, n_lidar_g=0, n_vggt_px=0,
                    depth_vggt=np.nan, depth_vggt_c=np.nan, depth_lidar=np.nan, depth_lidar_g=np.nan,
                    depth_da=np.nan,
                    pos_vggt=_nan3(), pos_vggt_c=_nan3(), pos_lidar=_nan3(), pos_lidar_g=_nan3(),
                    pos_da=_nan3(), vggt_frac=np.nan)
        bw, bh = x1 - x0, y1 - y0
        if bw < 2 or bh < 2:
            inst['flags'].append('degenerate_box')
            out.append(inst)
            continue

        # ---- region: mask inside the box, else inner torso box
        region = None
        if mask is not None:
            sub = mask[y0:y1, x0:x1].astype(np.uint8)
            if bw >= 12 and bh >= 12:
                sub = cv2.erode(sub, np.ones((3, 3), np.uint8))
            if sub.sum() >= MIN_REGION_PX:
                region = sub.astype(bool)
        if region is None:
            inst['flags'].append('mask_missing')
            region = np.zeros((bh, bw), bool)
            region[int(0.25 * bh):int(0.70 * bh), int(0.30 * bw):int(0.70 * bw)] = True
        rv, ru = np.nonzero(region)
        inst['region_px'] = int(len(rv))
        wts = region_weights(bh, bw, rv, ru)

        # ---- dense VGGT-Omega depth
        if dv_full is not None:
            d = dv_full[y0 + rv, x0 + ru]
            ok = d > 0
            if ok.sum() >= 20:
                d, wk, ruk, rvk = d[ok], wts[ok], ru[ok], rv[ok]
                c = pick_cluster(depth_clusters(d, wk), min_count=10)
                peak = c[0] if c is not None else float(np.median(d))
                sel = cluster_select(d, peak)
                inst['n_vggt_px'] = int(sel.sum())
                inst['vggt_frac'] = float(sel.sum() / len(d))
                pts = backproject(x0 + ruk[sel] + 0.5, y0 + rvk[sel] + 0.5, d[sel], K)
                inst['depth_vggt'] = float(np.median(d[sel]))
                inst['pos_vggt'] = body_centre_base(pts, T.base_T_cam)
                dc = vggt_correct(d[sel], T.dataset)
                inst['depth_vggt_c'] = float(np.median(dc))
                pts_c = backproject(x0 + ruk[sel] + 0.5, y0 + rvk[sel] + 0.5, dc, K)
                inst['pos_vggt_c'] = body_centre_base(pts_c, T.base_T_cam)
                if keep_points:
                    inst['pts_vggt'] = _to_base(pts_c, T.base_T_cam, cap=600)
                if inst['vggt_frac'] < 0.4:
                    inst['flags'].append('vggt_multimodal')
            else:
                inst['flags'].append('vggt_missing')
        else:
            inst['flags'].append('vggt_missing')

        # ---- DepthAnything, per-frame lidar-scaled
        if da_full is not None and np.isfinite(da_scale):
            d = da_full[y0 + rv, x0 + ru] * da_scale
            ok = d > 0
            if ok.sum() >= 20:
                d, wk, ruk, rvk = d[ok], wts[ok], ru[ok], rv[ok]
                c = pick_cluster(depth_clusters(d, wk), min_count=10)
                peak = c[0] if c is not None else float(np.median(d))
                sel = cluster_select(d, peak)
                inst['depth_da'] = float(np.median(d[sel]))
                inst['pos_da'] = body_centre_base(
                    backproject(x0 + ruk[sel] + 0.5, y0 + rvk[sel] + 0.5, d[sel], K), T.base_T_cam)

        # ---- sparse lidar
        if lu is not None:
            inb = (lu >= x0) & (lu < x1) & (lv >= y0) & (lv < y1)
            if inb.any():
                inreg = region[lv[inb] - y0, lu[inb] - x0]
                idx = np.nonzero(inb)[0][inreg]
                if len(idx) >= MIN_LIDAR_PTS:
                    z = lz[idx]
                    # uniform weights: lidar rings are sparse rows, a ring on the legs is as
                    # valid as one on the torso (torso weighting starves small far people)
                    clusters = depth_clusters(z, np.ones_like(z), rel_bin=0.08)
                    # independent lidar: nearest substantial cluster
                    c = pick_cluster(clusters, min_count=MIN_LIDAR_PTS)
                    if c is not None:
                        sel = cluster_select(z, c[0])
                        inst['n_lidar'] = int(sel.sum())
                        inst['depth_lidar'] = float(np.median(z[sel]))
                        inst['pos_lidar'] = body_centre_base(lpc[idx][sel], T.base_T_cam)
                        if keep_points:
                            inst['pts_lidar'] = _to_base(lpc[idx][sel], T.base_T_cam)
                    # VGGT-guided lidar: the dense map disambiguates which sparse cluster is the person
                    guide = inst['depth_vggt_c'] if DATASETS[T.dataset].get('vggt_trust', 1.0) >= 1.0 else None
                    cg = pick_cluster(clusters, guide=guide, min_count=MIN_LIDAR_PTS)
                    if cg is not None:
                        sel = cluster_select(z, cg[0])
                        inst['n_lidar_g'] = int(sel.sum())
                        inst['depth_lidar_g'] = float(np.median(z[sel]))
                        inst['pos_lidar_g'] = body_centre_base(lpc[idx][sel], T.base_T_cam)
                        if keep_points:
                            inst['pts_lidar_g'] = _to_base(lpc[idx][sel], T.base_T_cam)
                        if len(clusters) > 1:
                            inst['flags'].append('lidar_ambiguous')
        if inst['n_lidar'] < MIN_LIDAR_PTS:
            inst['flags'].append('lidar_sparse')
        out.append(inst)
    return out


# ----------------------------------------------------------------------------- smoothing
def rts_smooth(t, meas, sig, q_acc=1.2):
    """Constant-velocity Kalman filter + RTS smoother on 2D positions.

    t (n,), meas (n,2) with nan rows for missing, sig (n,) 1-sigma measurement std.
    Returns xs (n,4) [x, y, vx, vy] and P_pos (n,) marginal position variance.
    """
    n = len(t)
    have = np.isfinite(meas).all(1)
    xs = np.full((n, 4), np.nan)
    Pp = np.full(n, np.nan)
    if not have.any():
        return xs, Pp
    first = int(np.argmax(have))
    xf = np.zeros((n, 4)); Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pq = np.zeros((n, 4, 4))
    Fs = np.zeros((n, 4, 4))
    Hm = np.zeros((2, 4)); Hm[0, 0] = Hm[1, 1] = 1
    xf[first] = [meas[first, 0], meas[first, 1], 0, 0]
    Pf[first] = np.diag([sig[first] ** 2, sig[first] ** 2, 4.0, 4.0])
    xp[first], Pq[first], Fs[first] = xf[first], Pf[first], np.eye(4)
    for k in range(first + 1, n):
        dt = max(float(t[k] - t[k - 1]), 1e-3)
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q = q_acc ** 2
        Qa = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Qa
        Q[np.ix_([1, 3], [1, 3])] = Qa
        xp[k] = F @ xf[k - 1]
        Pq[k] = F @ Pf[k - 1] @ F.T + Q
        Fs[k] = F
        if have[k]:
            R = np.eye(2) * sig[k] ** 2
            S = Hm @ Pq[k] @ Hm.T + R
            Kg = Pq[k] @ Hm.T @ np.linalg.inv(S)
            xf[k] = xp[k] + Kg @ (meas[k] - Hm @ xp[k])
            Pf[k] = (np.eye(4) - Kg @ Hm) @ Pq[k]
        else:
            xf[k], Pf[k] = xp[k], Pq[k]
    # backward pass
    xs[n - 1] = xf[n - 1]
    Ps = Pf[n - 1].copy()
    Pp[n - 1] = Ps[0, 0]
    for k in range(n - 2, first - 1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.pinv(Pq[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps = Pf[k] + C @ (Ps - Pq[k + 1]) @ C.T
        Pp[k] = Ps[0, 0]
    return xs, Pp


def robust_rts(t, meas, sig, passes=2, gate=3.0):
    sig = sig.copy()
    xs = Pp = None
    for _ in range(passes):
        xs, Pp = rts_smooth(t, meas, sig)
        r = np.linalg.norm(meas - xs[:, :2], axis=1)
        nr = r / np.sqrt(sig ** 2 + np.nan_to_num(Pp, nan=0.0))
        bad = np.isfinite(nr) & (nr > gate)
        if not bad.any():
            break
        sig[bad] *= 5.0
    return xs, Pp, sig


# ----------------------------------------------------------------------------- trajectory
SOURCES = ('lidar', 'lidar_g', 'vggt', 'vggt_c', 'da')


def build_tracks(T, frames):
    """Group per-frame instances by track id, express in world frame, fuse and smooth."""
    by_id = {}
    for i, insts in enumerate(frames):
        for inst in insts:
            by_id.setdefault(inst['id'], []).append((i, inst))
    tracks = {}
    for tid, items in by_id.items():
        fi = np.array([i for i, _ in items])
        f0, f1 = fi.min(), fi.max()
        span = np.arange(f0, f1 + 1)
        n = len(span)
        t = T.timestamps[span]
        det = np.zeros(n, bool)
        world = {s: np.full((n, 2), np.nan) for s in SOURCES}
        depth = {s: np.full(n, np.nan) for s in SOURCES}
        n_lidar = np.zeros(n, int)      # points behind the independent lidar estimate
        n_lidar_g = np.zeros(n, int)    # points behind the VGGT-guided lidar estimate
        flags = [[] for _ in range(n)]
        for i, inst in items:
            k = i - f0
            det[k] = True
            n_lidar[k] = inst['n_lidar']
            n_lidar_g[k] = inst['n_lidar_g']
            flags[k] = inst['flags']
            for s in SOURCES:
                p = inst['pos_' + s]
                if np.isfinite(p).all():
                    world[s][k] = T.base_to_world(i, p[:2])
                    depth[s][k] = inst['depth_' + s]
        # track anchor: guided-lidar / corrected-vggt depth ratio where both are good
        both = (n_lidar_g >= GOOD_LIDAR_PTS) & np.isfinite(depth['vggt_c']) & (depth['lidar_g'] < EVAL_RANGE)
        if both.sum() >= 3:
            anchor = float(np.clip(np.median(depth['lidar_g'][both] / depth['vggt_c'][both]), 0.8, 1.25))
        else:
            anchor = 1.0
        # anchored vggt in world: rescale the base-frame position radially before transforming
        world['vggt_a'] = np.full((n, 2), np.nan)
        depth['vggt_a'] = depth['vggt_c'] * anchor
        for i, inst in items:
            k = i - f0
            p = inst['pos_vggt_c']
            if np.isfinite(p).all():
                world['vggt_a'][k] = T.base_to_world(i, p[:2] * anchor)
        # per-frame fused measurement (inverse variance)
        fused = np.full((n, 2), np.nan)
        fsig = np.full(n, np.nan)
        src = np.array([''] * n, dtype=object)
        for k in range(n):
            cands = []
            if n_lidar_g[k] >= MIN_LIDAR_PTS and np.isfinite(world['lidar_g'][k]).all():
                cands.append(('lidar', world['lidar_g'][k], sigma_lidar(depth['lidar_g'][k], n_lidar_g[k])))
            if np.isfinite(world['vggt_a'][k]).all():
                s_v = sigma_vggt(depth['vggt_a'][k], T.dataset)
                if 'vggt_multimodal' in flags[k]:
                    s_v *= 2.0
                cands.append(('vggt', world['vggt_a'][k], s_v))
            if not cands:
                continue
            wsum = sum(1.0 / s ** 2 for _, _, s in cands)
            fused[k] = sum(p / s ** 2 for _, p, s in cands) / wsum
            fsig[k] = np.sqrt(1.0 / wsum)
            src[k] = '+'.join(c[0] for c in cands)
        xs, Pp, sig_used = robust_rts(t, fused, np.nan_to_num(fsig, nan=1.0)) if np.isfinite(fused).any(1).any() \
            else (np.full((n, 4), np.nan), np.full(n, np.nan), fsig)
        outlier = np.isfinite(fsig) & (sig_used > fsig * 1.01)
        # per-source smoothed tracks (for the experiment comparison)
        smooth = {}
        for s in ('lidar', 'lidar_g', 'vggt', 'vggt_c', 'vggt_a', 'da'):
            m = world[s]
            if np.isfinite(m).all(1).sum() >= 2:
                sg = {'lidar': lambda k: sigma_lidar(depth['lidar'][k], n_lidar[k]),
                      'lidar_g': lambda k: sigma_lidar(depth['lidar_g'][k], n_lidar_g[k]),
                      'da': lambda k: sigma_da(depth['da'][k])}.get(s, lambda k: sigma_vggt(depth[s][k], T.dataset))
                sig_s = np.array([sg(k) if np.isfinite(m[k]).all() else 1.0 for k in range(n)])
                smooth[s], _, _ = robust_rts(t, m, sig_s)
            else:
                smooth[s] = np.full((n, 4), np.nan)
        tracks[int(tid)] = dict(
            frames=span, t=t, detected=det, n_lidar=n_lidar, n_lidar_g=n_lidar_g, flags=flags,
            world=world, depth=depth, anchor=anchor,
            fused_meas=fused, fused_sig=fsig, fused_src=src, outlier=outlier,
            fused=xs, fused_var=Pp, interpolated=~det, smooth=smooth,
        )
    return tracks


def traj_metrics(T, tracks):
    """Self-consistency metrics within EVAL_RANGE (no ground truth available)."""
    rows = dict(n_tracks=len(tracks), n_det=0, cov_lidar=0, cov_vggt=0, cov_fused=0,
                agree_lidar_vggt=[], agree_lidar_vggt_c=[], agree_lidar_vggt_a=[], agree_lidar_da=[],
                agree_lidar_lidar_g=[],
                resid_lidar=[], resid_lidar_g=[], resid_vggt=[], resid_vggt_c=[], resid_vggt_a=[], resid_da=[],
                n_outlier=0, n_interp=0, n_lidar_ambiguous=0)
    for tr in tracks.values():
        det = tr['detected']
        rng = tr['depth']['vggt_c'] if np.isfinite(tr['depth']['vggt_c']).any() else tr['depth']['lidar']
        near = det & (np.nan_to_num(rng, nan=EVAL_RANGE + 1) < EVAL_RANGE)
        rows['n_det'] += int(near.sum())
        rows['cov_lidar'] += int((near & np.isfinite(tr['world']['lidar']).all(1)).sum())
        rows['cov_vggt'] += int((near & np.isfinite(tr['world']['vggt']).all(1)).sum())
        rows['cov_fused'] += int((near & np.isfinite(tr['fused']).all(1)).sum())
        rows['n_outlier'] += int(tr['outlier'].sum())
        rows['n_interp'] += int((~det & np.isfinite(tr['fused']).all(1)).sum())
        rows['n_lidar_ambiguous'] += sum('lidar_ambiguous' in f for f in tr['flags'])
        good_l = near & (tr['n_lidar'] >= GOOD_LIDAR_PTS)
        for s in ('vggt', 'vggt_c', 'vggt_a', 'da', 'lidar_g'):
            m = good_l & np.isfinite(tr['world'][s]).all(1)
            rows['agree_lidar_' + s] += list(np.linalg.norm(tr['world'][s][m] - tr['world']['lidar'][m], axis=1))
        for s in ('lidar', 'lidar_g', 'vggt', 'vggt_c', 'vggt_a', 'da'):
            m = near & np.isfinite(tr['world'][s]).all(1) & np.isfinite(tr['fused']).all(1)
            rows['resid_' + s] += list(np.linalg.norm(tr['world'][s][m] - tr['fused'][m, :2], axis=1))
    out = {k: v for k, v in rows.items() if not isinstance(v, list)}
    for k, v in rows.items():
        if isinstance(v, list):
            v = np.array(v)
            out[k + '_n'] = int(len(v))
            out[k + '_med'] = float(np.median(v)) if len(v) else np.nan
            out[k + '_p90'] = float(np.percentile(v, 90)) if len(v) else np.nan
    return out


def process_traj(dataset, traj, with_da=True):
    T = Traj(dataset, traj)
    frames = [estimate_frame(T, i, with_da=with_da) for i in range(T.n)]
    tracks = build_tracks(T, frames)
    return dict(dataset=dataset, traj=traj, n=T.n, timestamps=T.timestamps[:T.n],
                robot_pos=T.robot_pos[:T.n], robot_yaw=T.robot_yaw[:T.n],
                frames=frames, tracks=tracks, metrics=traj_metrics(T, tracks))


def result_path(dataset, traj):
    return os.path.join(OUT_ROOT, dataset, traj + '.pkl')


def load_or_process(dataset, traj, force=False):
    p = result_path(dataset, traj)
    if not force and os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    res = process_traj(dataset, traj)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        pickle.dump(res, f)
    return res


# ----------------------------------------------------------------------------- distributions
_WEEKDAYS = {'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'}
_STOP_TOKENS = {'merged', 'corrected', 'new', 'with', 'without', 'no', 'one', 'overhead', 'segment', 'split'}


def subdataset(dataset, traj):
    """Location group of a trajectory from its name, e.g. go2nus 'utown_new_6_with_overhead_...'
    -> 'utown', miraikan '03-08-2025-1-3F_Geo_Cosmos_0_0_1' -> '3F_Geo_Cosmos',
    gmu 'engr_elev_1_10_merged_0' -> 'engr_elev', scand_spot 'A_Spot_Ahg_EERC_Thu_Nov_13_...' -> 'Ahg_EERC'."""
    import re
    if dataset == 'uex':      # rosbag2_2025_03_06-17_58_18_1_merged_0 -> recording day
        m = re.match(r'^rosbag2_(\d{4}_\d{2}_\d{2})', traj)
        return m.group(1) if m else traj
    if dataset == 'umich':    # one recording session per day+start time
        m = re.match(r'^rosbag2_(\d{4}_\d{2}_\d{2}-\d{2}_\d{2})', traj)
        return m.group(1) if m else traj
    name = re.sub(r'^\d{2}-\d{2}-\d{4}-\d+-', '', traj)
    toks = name.split('_')
    if dataset == 'scand_spot' and len(toks) > 2 and toks[1] == 'Spot':
        toks = toks[2:]
    keep = []
    for t in toks:
        if t in _STOP_TOKENS or t in _WEEKDAYS or (re.search(r'\d', t) and not re.fullmatch(r'\d+F', t)):
            break
        keep.append(t)
    return '_'.join(keep) if keep else toks[0]


def collect_positions(dataset, max_range=15.0, detected_only=True):
    """Robot-frame xy of every fused pedestrian estimate in the processed results of a dataset.

    Returns {group: dict(xy (N,2) [x forward, y left], n_frames, n_trajs)} with an extra
    'ALL' group for the whole dataset.
    """
    d = os.path.join(OUT_ROOT, dataset)
    groups = {}
    if not os.path.isdir(d):
        return groups
    for f in sorted(os.listdir(d)):
        if not f.endswith('.pkl'):
            continue
        with open(os.path.join(d, f), 'rb') as fh:
            res = pickle.load(fh)
        g = subdataset(dataset, res['traj'])
        pts = []
        cos, sin = np.cos(res['robot_yaw']), np.sin(res['robot_yaw'])
        for tr in res['tracks'].values():
            ok = np.isfinite(tr['fused'][:, :2]).all(1)
            if detected_only:
                ok &= tr['detected']
            if not ok.any():
                continue
            fr = tr['frames'][ok]
            dxy = tr['fused'][ok, :2] - res['robot_pos'][fr]
            c, s = cos[fr], sin[fr]
            base = np.stack([c * dxy[:, 0] + s * dxy[:, 1], -s * dxy[:, 0] + c * dxy[:, 1]], 1)
            pts.append(base[np.linalg.norm(base, axis=1) < max_range])
        xy = np.concatenate(pts) if pts else np.zeros((0, 2))
        for key in (g, 'ALL'):
            e = groups.setdefault(key, dict(xy=[], n_frames=0, n_trajs=0))
            e['xy'].append(xy)
            e['n_frames'] += int(res['n'])
            e['n_trajs'] += 1
    for e in groups.values():
        e['xy'] = np.concatenate(e['xy']) if e['xy'] else np.zeros((0, 2))
    return groups


def heatmap(xy, n_frames, bin_m=0.25, x_range=(-4.0, 12.0), y_range=(-8.0, 8.0)):
    """Pedestrian density around the robot: expected pedestrians per frame per m^2."""
    xe = np.arange(x_range[0], x_range[1] + 1e-6, bin_m)
    ye = np.arange(y_range[0], y_range[1] + 1e-6, bin_m)
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    H = H / max(n_frames, 1) / (bin_m * bin_m)
    return H, xe, ye


def load_tracks(dataset):
    """Load the consolidated training pickle written by export_tracks.py (see its docstring)."""
    with open(os.path.join(OUT_ROOT, f'tracks_{dataset}.pkl'), 'rb') as f:
        return pickle.load(f)


def dense_tracks(entry, key='tracks'):
    """Per-track trajectory entry (export_tracks) -> dense (T, P, ...) arrays, NaN = absent.
    key: 'tracks' (raw tracker ids) or 'merged_tracks' (after re-linking)."""
    T = entry['n_frames']
    tracks = entry[key]
    ids = np.array(sorted(tracks, key=lambda k: tracks[k]['frames'][0]), dtype=np.int32)
    P = len(ids)
    out = dict(ids=ids,
               pos_world=np.full((T, P, 2), np.nan, np.float32), vel_world=np.full((T, P, 2), np.nan, np.float32),
               pos_robot=np.full((T, P, 2), np.nan, np.float32), pos_std=np.full((T, P), np.nan, np.float32),
               detected=np.zeros((T, P), bool), source=np.zeros((T, P), np.int8),
               box=np.full((T, P, 4), np.nan, np.float32))
    for j, tid in enumerate(ids):
        tr = tracks[int(tid)]
        fr = tr['frames']
        for k in ('pos_world', 'vel_world', 'pos_robot', 'pos_std', 'detected', 'source', 'box'):
            out[k][fr, j] = tr[k]
    out['range'] = np.linalg.norm(out['pos_robot'], axis=-1).astype(np.float32)
    return out


_FRAME_KEYS = ('pos_world', 'vel_world', 'pos_robot', 'pos_std', 'detected', 'source', 'box')
_FRAME_SHAPES = {'pos_world': (2,), 'vel_world': (2,), 'pos_robot': (2,), 'pos_std': (), 'detected': (),
                 'source': (), 'box': (4,)}


def frame_pedestrians(entry, t, detected_only=False, key='tracks'):
    """Pedestrians present at frame t of a trajectory entry -> dict of stacked arrays (ids, pos_world,
    vel_world, pos_robot, pos_std, detected, source, box). key: 'tracks' or 'merged_tracks'."""
    rows = []
    for tid, tr in entry[key].items():
        fr = tr['frames']
        if fr[0] <= t <= fr[-1]:
            k = t - fr[0]
            if np.isfinite(tr['pos_world'][k]).all() and (tr['detected'][k] or not detected_only):
                rows.append((tid, k, tr))
    out = {'ids': np.array([r[0] for r in rows], dtype=np.int32)}
    for name in _FRAME_KEYS:
        out[name] = np.array([r[2][name][r[1]] for r in rows]) if rows else np.zeros((0,) + _FRAME_SHAPES[name])
    return out


def list_trajs(dataset):
    root = os.path.join(depth_root(dataset), dataset)
    proc = os.path.join(PROCESSED_ROOT, dataset)
    ext = os.path.join(PROCESSED_ROOT, 'extrinsics', dataset)
    out = []
    for t in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, t, 'vggt_omega_depth')) and \
                tracks_file(os.path.join(proc, t)) is not None and \
                extrinsics_file(dataset, t) is not None:
            out.append(t)
    return out
