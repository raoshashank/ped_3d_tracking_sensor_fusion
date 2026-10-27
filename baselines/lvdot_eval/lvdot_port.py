"""Python port of the LV-DOT detection front end (Xu et al. 2025, lv-dot/onboard_detector).

LV-DOT is a ROS1 Noetic C++ node; only ROS2 Humble is installed here, so the parts that matter for
pedestrian detection are re-implemented from dynamicDetector.cpp / lidarDetector.cpp / dbscan.cpp:

  lidarPoseCB      : pass-through box |x|,|y| <= local_lidar_range, Gaussian range downsampling
                     (keep prob exp(-d^2 / 2 sigma^2), sigma = gaussian_downsample_rate), transform to
                     the world frame, keep ground_height <= z <= roof_height, voxel grid 0.1 m grown by
                     1.1x until <= downsample_threshold points
  lidarDetect      : DBSCAN (min_pts, eps) -> per-cluster AABB, drop boxes larger than max_object_size.
                     NOTE dbscan.cpp compares the SQUARED distance against epsilon, so the yaml
                     lidar_DBSCAN_epsilon = 0.05 is a radius of sqrt(0.05) = 0.224 m.
  filterLVBBoxes   : STEP 5 only (no depth camera in our data): project each 3D box into the colour
                     image (8 corners -> camera-frame AABB -> top-left / bottom-right at the centre
                     depth), give every YOLO "person" box the 3D box with the best 2D IOU (> 0),
                     a 3D box with one YOLO match becomes is_human, one with several is split by
                     assigning its points to the YOLO box containing their pixel.
  classificationCB : is_human boxes are dynamic obstacles; with target_constrain_size they must be
                     within (0.8, 0.8, 1.0) m of target_object_size (0.5, 0.5, 1.5).

The world frame here is the robot "ground" frame: base_link axes, origin moved down to the floor,
so the z thresholds mean the same as in LV-DOT (which assumes a flat floor at world z = 0).
Tracking (Kalman + point-velocity voting at 30 Hz) is not ported: our frames are sampled at 4 Hz.
"""
import time

import numpy as np
from sklearn.cluster import DBSCAN

# cfg/detector_param.yaml + dynamicDetector.h defaults
STOCK = dict(
    name='stock',
    local_lidar_range=(10.0, 10.0),     # localLidarRange_ {10, 10, 5}
    gaussian_sigma=6.0,                 # gaussian_downsample_rate
    ground_height=0.2, roof_height=2.0,
    voxel=0.1, downsample_threshold=3500,
    db_eps=0.05, eps_is_squared=True,   # lidar_DBSCAN_epsilon (squared-distance semantics)
    db_min_pts=10,                      # lidar_DBSCAN_min_points
    max_object_size=(3.0, 3.0, 2.0),
    target_object_size=(0.5, 0.5, 1.5), size_tol=(0.8, 0.8, 1.0),
    constrain_size=True,
)
# same algorithm, parameters adapted to a 16/32-beam spinning lidar on a ground robot and 10 m range
TUNED = dict(STOCK, name='tuned',
             local_lidar_range=(15.0, 15.0), gaussian_sigma=None,
             db_eps=0.30, eps_is_squared=False, db_min_pts=6,
             roof_height=2.2)

CONFIGS = {'stock': STOCK, 'tuned': TUNED}


def eps_radius(P):
    return float(np.sqrt(P['db_eps'])) if P['eps_is_squared'] else float(P['db_eps'])


# ----------------------------------------------------------------------------- lidarPoseCB
def voxel_grid(pts, leaf):
    """pcl::VoxelGrid: one point per occupied voxel, at the centroid of its points."""
    if len(pts) == 0:
        return pts
    idx = np.floor(pts / leaf).astype(np.int64)
    idx -= idx.min(0)
    key = (idx[:, 0] * (idx[:, 1].max() + 1) + idx[:, 1]) * (idx[:, 2].max() + 1) + idx[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, pts)
    return out / cnt[:, None]


def preprocess(pts_g, P, rng):
    """pts_g: (N,3) points in the ground frame (z = 0 on the floor). Returns the cloud DBSCAN sees."""
    rx, ry = P['local_lidar_range']
    keep = (np.abs(pts_g[:, 0]) <= rx) & (np.abs(pts_g[:, 1]) <= ry)
    p = pts_g[keep]
    if P['gaussian_sigma']:
        d2 = p[:, 0] ** 2 + p[:, 1] ** 2
        prob = np.exp(-d2 / (2.0 * P['gaussian_sigma'] ** 2))
        p = p[rng.random(len(p)) < prob]
    p = p[(p[:, 2] >= P['ground_height']) & (p[:, 2] <= P['roof_height'])]
    leaf = P['voxel']
    q = voxel_grid(p, leaf)          # the C++ always applies the 0.1 m grid once via the while loop
    while len(q) > P['downsample_threshold']:
        leaf *= 1.1
        q = voxel_grid(p, leaf)
    return q


# ----------------------------------------------------------------------------- lidarDetect
def cluster_boxes(pts, P):
    """DBSCAN clusters -> list of dict(center, size, pts) after the max_object_size filter."""
    if len(pts) < P['db_min_pts']:
        return []
    labels = DBSCAN(eps=eps_radius(P), min_samples=P['db_min_pts'], algorithm='kd_tree').fit_predict(pts)
    out = []
    mx = np.asarray(P['max_object_size'])
    for lab in range(labels.max() + 1):
        c = pts[labels == lab]
        lo, hi = c.min(0), c.max(0)
        size = hi - lo
        if np.any(size > mx):
            continue
        out.append(dict(center=(lo + hi) / 2, size=size, pts=c, is_human=False, yolo=-1))
    return out


# ----------------------------------------------------------------------------- filterLVBBoxes STEP 5
def project_box(box, cam_T_g, K):
    """transformBBox + the two-corner projection of filterLVBBoxes -> (x1, y1, x2, y2) int or None."""
    c, s = box['center'], box['size']
    corners = c + np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)]) * s
    pc = (cam_T_g[:3, :3] @ corners.T).T + cam_T_g[:3, 3]
    lo, hi = pc.min(0), pc.max(0)
    cz = (lo[2] + hi[2]) / 2
    if cz <= 0.05:
        return None
    tl = np.array([lo[0], lo[1], cz]); br = np.array([hi[0], hi[1], cz])
    x1 = int((K[0, 0] * tl[0] + K[0, 2] * cz) / cz); y1 = int((K[1, 1] * tl[1] + K[1, 2] * cz) / cz)
    x2 = int((K[0, 0] * br[0] + K[0, 2] * cz) / cz); y2 = int((K[1, 1] * br[1] + K[1, 2] * cz) / cz)
    return x1, y1, x2, y2


def iou2d(a, b):
    xo = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    yo = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = xo * yo
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _box_from_pts(c):
    lo, hi = c.min(0), c.max(0)
    return dict(center=(lo + hi) / 2, size=hi - lo, pts=c)


def yolo_fuse(boxes, yolo_xyxy, cam_T_g, K):
    """STEP 5 of filterLVBBoxes. yolo_xyxy: (M,4) int pixel boxes of class person."""
    if len(boxes) == 0 or len(yolo_xyxy) == 0:
        return boxes
    proj = [project_box(b, cam_T_g, K) for b in boxes]
    best3d = np.full(len(yolo_xyxy), -1)
    for i, yb in enumerate(yolo_xyxy):
        best, bi = 0.0, -1
        for j, pb in enumerate(proj):
            if pb is None:
                continue
            v = iou2d(pb, yb)
            if v > best:
                best, bi = v, j
        if best > 0.0:
            best3d[i] = bi
    box2yolo = {}
    for i, j in enumerate(best3d):
        if j >= 0:
            box2yolo.setdefault(j, []).append(i)
    out = []
    for j, b in enumerate(boxes):
        ys = box2yolo.get(j)
        if ys is None:
            out.append(b)
        elif len(ys) == 1:
            out.append(dict(b, is_human=True, yolo=ys[0]))
        else:  # split the cluster between the YOLO boxes that contain its pixels
            pc = (cam_T_g[:3, :3] @ b['pts'].T).T + cam_T_g[:3, 3]
            z = np.maximum(pc[:, 2], 1e-6)
            u = ((K[0, 0] * pc[:, 0] + K[0, 2] * z) / z).astype(int)
            v = ((K[1, 1] * pc[:, 1] + K[1, 2] * z) / z).astype(int)
            assign = np.full(len(u), -1)
            closest = np.full(len(u), np.iinfo(np.int64).max)
            for yi in ys:
                x1, y1, x2, y2 = yolo_xyxy[yi]
                inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                hd = np.maximum(x1 - u, u - x2)          # "horizontalDistance" (negative inside)
                better = inside & (hd < closest)
                assign[better] = yi
                closest[better] = hd[better]
            for yi in ys:
                sub = b['pts'][assign == yi]
                if len(sub) == 0:
                    continue
                nb = _box_from_pts(sub)
                if nb['size'][0] <= 0 or nb['size'][1] <= 0:
                    continue
                out.append(dict(nb, is_human=True, yolo=yi))
    return out


# ----------------------------------------------------------------------------- classificationCB
def size_ok(box, P):
    if not P['constrain_size']:
        return True
    d = np.abs(box['size'] - np.asarray(P['target_object_size']))
    return bool(np.all(d < np.asarray(P['size_tol'])))


# ----------------------------------------------------------------------------- one frame
def detect_frame(pts_lidar, g_T_lidar, cam_T_g, K, yolo_xyxy, P, rng):
    """Full LV-DOT lidar + colour pipeline on one frame.

    Returns dict(clusters=all DBSCAN boxes after fusion, humans=is_human boxes,
                 dynamic=is_human boxes passing the size constraint, n_in=points after preprocessing,
                 t_pre, t_db, t_fuse seconds).
    """
    t0 = time.perf_counter()
    pts_g = (g_T_lidar[:3, :3] @ pts_lidar[:, :3].T).T + g_T_lidar[:3, 3]
    q = preprocess(pts_g, P, rng)
    t1 = time.perf_counter()
    boxes = cluster_boxes(q, P)
    t2 = time.perf_counter()
    boxes = yolo_fuse(boxes, yolo_xyxy, cam_T_g, K)
    t3 = time.perf_counter()
    humans = [b for b in boxes if b['is_human']]
    dynamic = [b for b in humans if size_ok(b, P)]
    return dict(clusters=boxes, humans=humans, dynamic=dynamic, n_in=len(q),
                t_pre=t1 - t0, t_db=t2 - t1, t_fuse=t3 - t2)
