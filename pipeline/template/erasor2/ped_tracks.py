"""ped3d pedestrian tracks (go2nus, /media/shashank/ssd2/ped3d) re-expressed in the SLAM `map` frame.

ped3d stores fused positions in the onboard-odometry world frame (traj_data), which drifts by ~1 m
against the SLAM trajectory over this bag. So each fused position is taken back to the robot frame
with ped3d's own robot pose at that timestamp, then forward with the SLAM base_link pose at the same
time: drift cancels, only the (lidar/camera) robot->pedestrian offset is kept.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # slam_output: dataset_profile.py
import dataset_profile as DP  # noqa: E402

HERE = Path(__file__).resolve().parent
BAG_NAME = HERE.parent.parent.name
PROF = DP.load(HERE)
PED3D = Path(PROF["ped3d_pkl"].format(bag=BAG_NAME))   # go2nus: /media/shashank/ssd2/ped3d/go2nus/<bag>_0.pkl
EXTRAP = 0.3       # s: extend each track past its ends at constant velocity
HOLD_STILL = np.inf  # s: a track slower than STILL_SPEED is held in place this long past its ends
STILL_SPEED = 0.3    # m/s. The forward camera loses standing people once the robot has passed them,
                     # while the lidar keeps seeing them; ped3d tracks also break and restart.


def slam_base_poses():
    b = np.loadtxt(HERE.parent / "poses_base_link_tum.txt")
    yaw = np.unwrap(R.from_quat(b[:, 4:8]).as_euler("zyx")[:, 0])
    return b[:, 0], b[:, 1:3], yaw


def load_tracks(key="fused"):
    """Returns {track_id: (t [T], xy_map [T,2], v_map [T,2])} with only finite rows."""
    d = pickle.load(open(PED3D, "rb"))
    ts, pxy, pyaw = slam_base_poses()
    rpos, ryaw = d["robot_pos"], d["robot_yaw"]
    out = {}
    for tid, tr in d["tracks"].items():
        st = tr[key]
        ok = np.isfinite(st[:, :2]).all(1)
        fr = tr["frames"][ok]
        if not ok.any():
            continue
        t = d["timestamps"][fr]
        # world(odom) -> robot at that frame
        c, s = np.cos(ryaw[fr]), np.sin(ryaw[fr])
        dx, dy = st[ok, 0] - rpos[fr, 0], st[ok, 1] - rpos[fr, 1]
        bx, by = c * dx + s * dy, -s * dx + c * dy
        vx, vy = c * st[ok, 2] + s * st[ok, 3], -s * st[ok, 2] + c * st[ok, 3]
        # robot -> SLAM map at the same time
        yaw = np.interp(t, ts, pyaw)
        c2, s2 = np.cos(yaw), np.sin(yaw)
        xy = np.stack([np.interp(t, ts, pxy[:, 0]) + c2 * bx - s2 * by,
                       np.interp(t, ts, pxy[:, 1]) + s2 * bx + c2 * by], 1)
        v = np.stack([c2 * vx - s2 * vy, s2 * vx + c2 * vy], 1)
        out[tid] = (t, xy, v)
    return out


def _end_reach(v):
    return HOLD_STILL if np.linalg.norm(v) < STILL_SPEED else EXTRAP


def load_lidar_tracks(csv=HERE / "output_ped" / "lidar_tracks.csv", min_len=5):
    """Tracks from lidar_ped_detect.py (background subtraction against the static map, 360 deg),
    same layout as load_tracks(). Velocity by finite differences."""
    rows = np.loadtxt(csv, delimiter=",", skiprows=1)
    out = {}
    for tid in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == tid]
        r = r[np.argsort(r[:, 2])]
        if len(r) < min_len:
            continue
        t, xy = r[:, 2], r[:, 3:5]
        v = np.gradient(xy, t, axis=0) if len(t) > 1 else np.zeros_like(xy)
        out[f"lidar{tid}"] = (t, xy, v)
    return out


def standing_group_completion(tracks, map_pcd, radius=2.0, cell=0.2, min_cells=4, accept=()):
    """People standing in a group with a tracked standing person but never detected by either sensor
    (occluded from the camera, and in the map so invisible to background subtraction). A standing
    person is a continuous vertical column in the static map: 0.2 m cells near a held standing track
    whose points cover every height band 0.15-0.5, 0.5-1.0 and 1.0-1.5 m and reach no higher than
    2.1 m are person-cells (foliage starts higher, furniture stops lower); 4-connected blobs of them
    with a person-sized footprint are CANDIDATES. Anchors are camera-confirmed (ped3d) standing
    tracks only, and a blob needs >= min_cells: a person accumulated over a recording is >= 0.6 m
    wide, a sign post or bollard is one or two cells. Lidar shape alone cannot tell a person from a
    1.5 m sign box or a staked sapling (both happened on utown_1), so candidates are written to
    output_ped/completion_candidates.csv and only those listed in `accept` become held tracks;
    review them with the camera crops from `completion_crops` before accepting."""
    import open3d as o3d
    from scipy import ndimage
    from scipy.spatial import cKDTree
    st = np.asarray(o3d.io.read_point_cloud(str(map_pcd)).points)
    h = st[:, 2] + 0.25
    anchors = [(t, xy) for k, (t, xy, v) in tracks.items() if not str(k).startswith("lidar")
               and (_end_reach(v[0]) == HOLD_STILL or _end_reach(v[-1]) == HOLD_STILL)]
    if not anchors:
        return {}
    axy = np.vstack([xy for _, xy in anchors])
    near = (h > 0.15) & (h < 2.5) & (cKDTree(axy).query(st[:, :2])[0] < radius)
    p, hh = st[near, :2], h[near]
    lo = p.min(0) - cell
    ij = np.floor((p - lo) / cell).astype(int)
    shape = ij.max(0) + 2
    bands = [(0.15, 0.5), (0.5, 1.0), (1.0, 1.5)]
    counts = np.zeros((len(bands), *shape), int)
    for b, (a, c) in enumerate(bands):
        m = (hh >= a) & (hh < c)
        np.add.at(counts[b], (ij[m, 0], ij[m, 1]), 1)
    hmax = np.zeros(shape)
    np.maximum.at(hmax, (ij[:, 0], ij[:, 1]), hh)
    person = (counts >= 3).all(0) & (hmax <= 2.1)
    lab, n = ndimage.label(person)
    t_all = np.concatenate([t for t, _ in anchors])
    out, rows = {}, []
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        foot = (cells.max(0) - cells.min(0) + 1).max() * cell
        cen = (cells.mean(0) + 0.5) * cell + lo
        if not (0.2 <= foot <= 1.4 and len(cells) >= min_cells):
            continue
        q = len(rows)
        rows.append((q, cen[0], cen[1], len(cells), foot, hmax[lab == k].max()))
        status = "ACCEPTED" if q in accept else "candidate"
        print(f"  group completion {q}: column at {cen.round(2)}, {len(cells)} cells, footprint {foot:.1f} m, "
              f"top {hmax[lab == k].max():.2f} m -> {status}")
        if q in accept:
            out[f"group{q}"] = (np.array([t_all.min(), t_all.max()]), np.vstack([cen, cen]), np.zeros((2, 2)))
    with open(HERE / "output_ped" / "completion_candidates.csv", "w") as f:
        f.write("id,x,y,cells,footprint,top\n")
        f.writelines(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]},{r[4]:.1f},{r[5]:.2f}\n" for r in rows)
    return out


def positions_at(tracks, t_query):
    """Pedestrian map-frame xy at time t_query: interpolated inside a track; past its ends either
    constant-velocity extrapolated (EXTRAP s) or, for a standing person, held (HOLD_STILL s)."""
    pts = []
    for t, xy, v in tracks.values():
        if t_query < t[0] - _end_reach(v[0]) or t_query > t[-1] + _end_reach(v[-1]):
            continue
        if t_query < t[0]:
            pts.append(xy[0] + (v[0] * (t_query - t[0]) if _end_reach(v[0]) == EXTRAP else 0))
        elif t_query > t[-1]:
            pts.append(xy[-1] + (v[-1] * (t_query - t[-1]) if _end_reach(v[-1]) == EXTRAP else 0))
        else:
            pts.append([np.interp(t_query, t, xy[:, 0]), np.interp(t_query, t, xy[:, 1])])
    return np.array(pts).reshape(-1, 2)


def positions_within(tracks, t0, t1):
    """All track samples (incl. held/extrapolated ends) with a time in [t0, t1]."""
    pts = []
    for t, xy, v in tracks.values():
        lo, hi = t[0] - _end_reach(v[0]), t[-1] + _end_reach(v[-1])
        if hi < t0 or lo > t1:
            continue
        m = (t >= t0) & (t <= t1)
        pts.append(xy[m])
        for te in (max(t0, lo), min(t1, hi)):
            pts.append(positions_at({0: (t, xy, v)}, te))
    return np.vstack(pts) if pts else np.zeros((0, 2))
