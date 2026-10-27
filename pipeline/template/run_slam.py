#!/usr/bin/env python3
"""Offline LiDAR SLAM on one rosbag2 recording (go2nus Go2 by default; topics, lidar layout, extrinsics
and the attitude source come from dataset_profile.py / slam_output/dataset.json).

Frontend : KISS-ICP scan-to-map odometry on /rslidar_points (deskew OFF: scans in this
           "corrected" bag are already motion-compensated; deskewing again makes ICP oscillate).
Backend  : GTSAM pose graph = KISS-ICP relative motions
           + gravity-direction (attitude) factors from the Go2 IMU orientation in
           /utlidar/robot_odom (removes roll/pitch drift), followed by least-squares position
           smoothing (removes gait-frequency registration jitter; 1 s steps keep the lidar's scale).
           (Trajectory has no revisits, so no loop closures are added.)

Outputs (next to this script):
  map.pcd                    - global 3D map (x,y,z,intensity), 5 cm voxels, `map` frame
  map_full_res.pcd           - denser map, 2 cm voxels
  poses_base_link_tum.txt    - base_link pose per scan, TUM format (t x y z qx qy qz qw)
  poses_rslidar_tum.txt      - rslidar pose per scan, TUM format
  poses_base_link_kitti.txt  - base_link poses, KITTI 3x4 row-major
  poses_base_link_kiss_only_tum.txt - frontend-only (no gravity fusion) poses, for reference
  robot_odom_tum.txt         - Go2 onboard /utlidar/robot_odom re-expressed in `map`, for comparison

`map` frame: origin at base_link of the first scan, z up (gravity-aligned), x = initial heading.

Usage:  python run_slam.py [path/to/bag_dir]
"""
import json
import sys
from pathlib import Path

import gtsam
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as R

from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_profile as DP  # noqa: E402

BAG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
PROF = DP.load(OUT)
LIDAR_TOPIC, ODOM_TOPIC, IMU_TOPIC = PROF["lidar_topic"], PROF["odom_topic"], PROF["imu_topic"]
MIN_RANGE, MAX_RANGE, ICP_VOXEL = 0.6, 80.0, 0.5   # min range removes returns from the robot body
BAD_SPAN_S, BAD_FRACTION = 0.2, 0.5   # corrupt scan: point stamps spread > 0.2 s, or < half the usual points
QC_WINDOW_S, QC_TRANS_M, QC_YAW_DEG = 1.0, 1.0, 30.0   # SLAM vs odometry check over 1 s windows
MAP_VOXEL, DENSE_VOXEL = 0.05, 0.02
# Pose graph noise per scan (rot x3 [rad], trans x3 [m])
ODOM_SIGMAS = np.array([0.002, 0.002, 0.002, 0.01, 0.01, 0.01])
# Position smoothing [m]: KISS-ICP positions jitter by several cm per scan at the gait frequency, so
# single-scan steps are weak, 1 s steps (LONG_LAG scans) keep the lidar's scale, and an acceleration
# penalty per scan (CV_SIGMA; 0.5 m/s^2 * 0.1 s^2 = 5 mm) does the smoothing.
STEP_SIGMA, LONG_LAG, LONG_SIGMA, CV_SIGMA = 0.05, 10, 0.03, 0.005
GRAVITY_SIGMA = np.radians(PROF["gravity_sigma_deg"])   # go2nus 2 deg: IMU tilt is noisy during trotting
FLOOR_R, FLOOR_BAND, FLOOR_DIST, FLOOR_MIN_INLIER = (1.0, 6.0), 0.15, 0.03, 0.25   # floor planes (profile floor_attitude)

# base -> lidar from /tf_static (go2nus: base_link -> rslidar, pure translation)
T_BASE_LIDAR = DP.base_T_lidar(PROF)


def pc2_to_numpy(msg):
    """xyz, intensity, per-point stamps; lidars without absolute point stamps get the header stamp."""
    names, formats, offsets = ["x", "y", "z", "intensity"], ["<f4"] * 4, list(PROF["lidar_offsets"])
    if PROF["lidar_time_offset"] is not None:
        names, formats, offsets = names + ["timestamp"], formats + ["<f8"], offsets + [PROF["lidar_time_offset"]]
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    ok = np.isfinite(xyz).all(axis=1)
    if PROF["lidar_time_offset"] is not None:
        ts = arr["timestamp"][ok].astype(np.float64)
    else:
        ts = np.full(ok.sum(), msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    return xyz[ok], arr["intensity"][ok].astype(np.float64), ts


def floor_up(xyz):
    """(up direction in the lidar frame, floor distance below the lidar) from a RANSAC floor plane (points
    1-6 m away, within FLOOR_BAND of the nominal floor), or None if the plane holds < FLOOR_MIN_INLIER."""
    h = PROF["lidar_height"]
    rng = np.linalg.norm(xyz[:, :2], axis=1)
    q = xyz[(rng > FLOOR_R[0]) & (rng < FLOOR_R[1]) & (np.abs(xyz[:, 2] + h) < FLOOR_BAND)]
    if len(q) < 200:
        return None
    (a, b, c, d), inl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q)).segment_plane(FLOOR_DIST, 3, 200)
    if len(inl) < FLOOR_MIN_INLIER * len(q):
        return None
    n = np.array([a, b, c])
    return n / np.linalg.norm(n) * np.sign(c), abs(d) / np.linalg.norm(n)


def to_tum(t, T):
    return [t, *T[:3, 3], *R.from_matrix(T[:3, :3]).as_quat()]  # qx qy qz qw


def save_tum(path, stamps, poses, header):
    np.savetxt(path, [to_tum(t, T) for t, T in zip(stamps, poses)], fmt="%.9f", header=header)


def smooth_positions(pos, z_target=None):
    """Sparse linear least squares over positions (N x 3, world frame, p0 fixed): keep per-scan steps
    (STEP_SIGMA) and 1 s steps (LONG_SIGMA) from `pos`, and penalize acceleration (CV_SIGMA).
    z_target (N, NaN = none): soft prior on each base z (floor_z_sigma), solved for z only."""
    n = len(pos)
    rows, cols, vals, rhs, r = [], [], [], [], 0
    def add(terms, target, sigma):
        nonlocal r
        for j, c in terms:
            rows.append(r); cols.append(j); vals.append(c / sigma)
        rhs.append(target / sigma); r += 1
    for i in range(1, n):
        add([(i, 1.0), (i - 1, -1.0)], pos[i] - pos[i - 1], STEP_SIGMA)
        if i >= LONG_LAG:
            add([(i, 1.0), (i - LONG_LAG, -1.0)], pos[i] - pos[i - LONG_LAG], LONG_SIGMA)
        if i >= 2:
            add([(i, 1.0), (i - 1, -2.0), (i - 2, 1.0)], np.zeros(3), CV_SIGMA)
    add([(0, 1.0)], pos[0], 1e-6)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    out = spsolve((A.T @ A).tocsc(), A.T @ np.array(rhs))
    if z_target is not None and np.isfinite(z_target).any():
        k = np.flatnonzero(np.isfinite(z_target))
        Az = sparse.vstack([A[:-1], sparse.csr_matrix((np.full(len(k), 1 / PROF["floor_z_sigma"]), (np.arange(len(k)), k)),
                                                      shape=(len(k), n))]).tocsr()
        bz = np.r_[np.array(rhs)[:-1, 2], z_target[k] / PROF["floor_z_sigma"]]
        out[:, 2] = spsolve((Az.T @ Az).tocsc(), Az.T @ bz)
    return out


def optimize_poses(init, ups, floor_h=None):
    """`init`: map <- base_link poses from the frontend; `ups`: gravity-up direction in each base frame.
    Returns smoothed, gravity-consistent poses.
    1) GTSAM pose graph: KISS-ICP relative poses + gravity-direction factors (fixes roll/pitch drift).
    2) Position smoothing (removes gait-frequency registration jitter while keeping the lidar's scale)."""
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    X = lambda i: gtsam.symbol("x", i)
    Diag = gtsam.noiseModel.Diagonal.Sigmas
    graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(init[0]), Diag(np.full(6, 1e-6))))
    grav_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                                gtsam.noiseModel.Isotropic.Sigma(2, GRAVITY_SIGMA))
    up = gtsam.Unit3(np.array([0.0, 0.0, 1.0]))
    for i, T in enumerate(init):
        values.insert(X(i), gtsam.Pose3(T))
        if i > 0:
            rel = np.linalg.inv(init[i - 1]) @ T
            graph.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), gtsam.Pose3(rel), Diag(ODOM_SIGMAS)))
        graph.add(gtsam.AttitudeFactorPose3(X(i), up, grav_noise, gtsam.Unit3(ups[i])))
    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    poses = [result.atPose3(X(i)).matrix() for i in range(len(init))]
    z_target = None
    if floor_h is not None and PROF["floor_z_sigma"]:
        # floor at map z = 0: base z = lidar z - floor distance below the lidar - lidar z in the base frame
        z_target = np.array([np.nan if h is None else T_BASE_LIDAR[2, 3] - h for h in floor_h])
    smoothed = smooth_positions(np.array([T[:3, 3] for T in poses]), z_target)
    for T, p in zip(poses, smoothed):
        T[:3, 3] = p
    return poses


def slam_qc(stamps, poses, odom_t, odom_T):
    """Compare SLAM with the onboard odometry over short windows (odometry drifts over a whole walk,
    but over 1 s both must agree): motion of the robot within each window, in the robot frame at the
    window start. A SLAM jump or fold shows as metres of disagreement in one window."""
    odom_t = np.asarray(odom_t)
    keep = np.r_[True, np.diff(odom_t) > 0]
    ot, oT = odom_t[keep], [T for T, k in zip(odom_T, keep) if k]
    oxy = np.array([T[:2, 3] for T in oT])
    oyaw = np.unwrap([np.arctan2(T[1, 0], T[0, 0]) for T in oT])
    sxy = np.array([T[:2, 3] for T in poses])
    syaw = np.unwrap([np.arctan2(T[1, 0], T[0, 0]) for T in poses])
    t = np.arange(stamps[0], stamps[-1] - QC_WINDOW_S, 0.1)
    if len(t) == 0:
        return dict(max_window_trans_m=0.0, max_window_yaw_deg=0.0, worst_t_s=0.0, flag=False)

    def at(tt, ts, xy, yaw):
        return np.stack([np.interp(tt, ts, xy[:, 0]), np.interp(tt, ts, xy[:, 1])], 1), np.interp(tt, ts, yaw)

    def local(p0, p1, y0):
        d = p1 - p0
        return np.stack([np.cos(y0) * d[:, 0] + np.sin(y0) * d[:, 1], -np.sin(y0) * d[:, 0] + np.cos(y0) * d[:, 1]], 1)

    s0, sy0 = at(t, stamps, sxy, syaw); s1, sy1 = at(t + QC_WINDOW_S, stamps, sxy, syaw)
    o0, oy0 = at(t, ot, oxy, oyaw); o1, oy1 = at(t + QC_WINDOW_S, ot, oxy, oyaw)
    dtr = np.linalg.norm(local(s0, s1, sy0) - local(o0, o1, oy0), axis=1)
    dyaw = np.degrees(np.abs((sy1 - sy0) - (oy1 - oy0)))
    i = int(np.argmax(dtr))
    return dict(max_window_trans_m=float(dtr.max()), max_window_yaw_deg=float(dyaw.max()),
                worst_t_s=float(t[i] - stamps[0]), flag=bool(dtr.max() > QC_TRANS_M or dyaw.max() > QC_YAW_DEG))


def main():
    o3d.utility.random.seed(0)   # deterministic RANSAC floor planes
    cfg = KISSConfig()
    cfg.data.deskew, cfg.data.min_range, cfg.data.max_range = False, MIN_RANGE, MAX_RANGE
    cfg.mapping.voxel_size = ICP_VOXEL
    kiss = KissICP(cfg)

    stamps, kiss_poses, scans, odom_t, odom_T, imu_t, imu_R = [], [], [], [], [], [], []
    skipped, n_pts_good, prev_odom, floor_ups = [], [], None, []
    T_LB = np.linalg.inv(T_BASE_LIDAR)
    with AnyReader([BAG], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:   # some bags lack type definitions
        conns = [c for c in reader.connections if c.topic in (LIDAR_TOPIC, ODOM_TOPIC, IMU_TOPIC)]
        n_scans = sum(c.msgcount for c in conns if c.topic == LIDAR_TOPIC)
        k_msg = -1
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == IMU_TOPIC:
                q = msg.orientation
                imu_t.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                imu_R.append(R.from_quat([q.x, q.y, q.z, q.w]).as_matrix())
                continue
            if conn.topic == ODOM_TOPIC:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                T = np.eye(4)
                T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T[:3, 3] = [p.x, p.y, p.z]
                odom_t.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                odom_T.append(T)
                continue
            k_msg += 1
            xyz, inten, ts = pc2_to_numpy(msg)
            # Corrupt partial scans (seen in 22 bags: a fragment arriving ~20 ms after a full scan, with a
            # few points stamped seconds off) throw ICP off for good: skip them here AND in export_scans.py
            # (skipped_scans.txt lists their message indices).
            ref = np.median(n_pts_good[-50:]) if n_pts_good else len(xyz)
            if len(xyz) == 0 or np.ptp(ts) > BAD_SPAN_S or len(xyz) < BAD_FRACTION * ref:
                skipped.append(k_msg)
                continue
            n_pts_good.append(len(xyz))
            # Motion prior from the onboard odometry instead of constant velocity: a trotting robot
            # turning sharply breaks the constant-velocity guess (science_new_4 bags folded that way).
            t_scan = float(np.median(ts))
            if odom_T:
                cur_odom = odom_T[-1]
                if prev_odom is not None:
                    d_base = np.linalg.inv(prev_odom) @ cur_odom                 # base_{k-1} <- base_k
                    kiss.last_delta = T_LB @ d_base @ T_BASE_LIDAR               # same motion, lidar frame
                prev_odom = cur_odom
            kiss.register_frame(xyz, ts)
            if PROF["floor_attitude"]:
                floor_ups.append(floor_up(xyz))
            stamps.append(t_scan)   # median: robust to stray point timestamps (seen: one point +7 s off)
            kiss_poses.append(T_BASE_LIDAR @ kiss.last_pose @ np.linalg.inv(T_BASE_LIDAR))  # base0 <- base_k

            rng = np.linalg.norm(xyz, axis=1)
            keep = (rng > MIN_RANGE) & (rng < MAX_RANGE)
            pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(xyz[keep]))
            pcd.point.intensity = o3d.core.Tensor(inten[keep][:, None])
            scans.append(pcd.voxel_down_sample(DENSE_VOXEL))
            if len(stamps) % 100 == 0:
                print(f"  frontend {len(stamps)}/{n_scans}", flush=True)

    stamps, odom_t = np.array(stamps), np.array(odom_t)
    imu_idx = np.abs(odom_t[None, :] - stamps[:, None]).argmin(axis=1)
    if IMU_TOPIC:   # planar odometry (no roll/pitch): gravity direction from the IMU orientation
        imu_t = np.array(imu_t)
        R_imu = [imu_R[j] for j in np.abs(imu_t[None, :] - stamps[:, None]).argmin(axis=1)]
    else:           # Go2: the onboard odometry carries the IMU attitude
        R_imu = [odom_T[i][:3, :3] for i in imu_idx]  # gravity-aligned world <- base_k

    ups = [Ri.T @ [0.0, 0.0, 1.0] for Ri in R_imu]   # gravity-up in each base frame
    src, floor_h = "IMU", None
    if PROF["floor_attitude"]:   # level floors: the floor plane seen by the lidar (IMU where no plane fits)
        n_floor = sum(u is not None for u in floor_ups)
        floor_h = [None if u is None else u[1] for u in floor_ups]
        floor_ups = [None if u is None else u[0] for u in floor_ups]
        ups = [T_BASE_LIDAR[:3, :3] @ u if u is not None else v for u, v in zip(floor_ups, ups)]
        imu_dev = [np.degrees(np.arccos(np.clip((T_BASE_LIDAR[:3, :3] @ u) @ (Ri.T @ [0, 0, 1]), -1, 1)))
                   for u, Ri in zip(floor_ups, R_imu) if u is not None]
        src = f"floor plane ({n_floor}/{len(ups)} scans; IMU off by {np.median(imu_dev) if imu_dev else 0:.2f} deg median)"

    # Initial map frame: first base_link with the measured roll/pitch (yaw = 0)
    T_map_base0 = np.eye(4)
    T_map_base0[:3, :3] = R.align_vectors([[0.0, 0.0, 1.0]], [ups[0]])[0].as_matrix()
    init = [T_map_base0 @ T for T in kiss_poses]

    base_poses = optimize_poses(init, ups, floor_h)

    def tilt_err(poses):
        return np.mean([np.degrees(np.arccos(np.clip((P[:3, :3].T @ [0, 0, 1]) @ u, -1, 1)))
                        for P, u in zip(poses, ups)])
    print(f"mean tilt error vs gravity from {src}: frontend {tilt_err(init):.2f} deg -> optimized {tilt_err(base_poses):.2f} deg")

    # ---- Save trajectories ----
    hdr = "timestamp x y z qx qy qz qw  (map <- {})"
    save_tum(OUT / "poses_base_link_tum.txt", stamps, base_poses, hdr.format("base_link"))
    save_tum(OUT / "poses_rslidar_tum.txt", stamps, [T @ T_BASE_LIDAR for T in base_poses], hdr.format("rslidar"))
    save_tum(OUT / "poses_base_link_kiss_only_tum.txt", stamps, init, hdr.format("base_link, frontend only"))
    np.savetxt(OUT / "poses_base_link_kitti.txt", [T[:3].ravel() for T in base_poses], fmt="%.9f")
    T_align = base_poses[0] @ np.linalg.inv(odom_T[imu_idx[0]])
    save_tum(OUT / "robot_odom_tum.txt", odom_t, [T_align @ T for T in odom_T],
             f"timestamp x y z qx qy qz qw  (onboard {ODOM_TOPIC} re-expressed in map frame)")

    # ---- Build map from optimized poses ----
    pts, inten = [], []
    for T, scan in zip(base_poses, scans):
        T_ml = T @ T_BASE_LIDAR
        p = scan.point.positions.numpy()
        pts.append(p @ T_ml[:3, :3].T + T_ml[:3, 3])
        inten.append(scan.point.intensity.numpy())
    pts, inten = np.concatenate(pts).astype(np.float32), np.concatenate(inten).astype(np.float32)
    for name, vox in (("map_full_res.pcd", DENSE_VOXEL), ("map.pcd", MAP_VOXEL)):
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(pts))
        pcd.point.intensity = o3d.core.Tensor(inten)
        pcd = pcd.voxel_down_sample(vox)
        o3d.t.io.write_point_cloud(str(OUT / name), pcd)
        print(f"wrote {name}: {len(pcd.point.positions)} points")

    (OUT / "skipped_scans.txt").write_text("".join(f"{k}\n" for k in skipped))
    qc = slam_qc(stamps, base_poses, odom_t, odom_T)
    (OUT / "slam_qc.json").write_text(json.dumps(qc, indent=1))
    print(f"skipped {len(skipped)} corrupt scans; QC vs odometry (worst {QC_WINDOW_S:.0f} s window): "
          f"{qc['max_window_trans_m']:.2f} m, {qc['max_window_yaw_deg']:.1f} deg -> {'FLAG' if qc['flag'] else 'ok'}")

    P = np.array([T[:3, 3] for T in base_poses])
    print(f"{len(P)} poses over {stamps[-1] - stamps[0]:.1f} s, path length "
          f"{np.linalg.norm(np.diff(P, axis=0), axis=1).sum():.1f} m, end position {P[-1].round(2)}")


if __name__ == "__main__":
    main()
