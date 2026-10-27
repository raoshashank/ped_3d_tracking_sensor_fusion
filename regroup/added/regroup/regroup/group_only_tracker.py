from __future__ import absolute_import
import numpy as np
import statistics
from collections import deque, namedtuple
from sklearn.preprocessing import normalize

from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .grouptrack import GroupTrack
from . import detection as detection_module
from . import utils


PedTrack = namedtuple('PedTrack', ['ped_id', 'tlwh', 'feature'])
"""
A single pedestrian observation from an external tracker.
  ped_id  : int        — stable ID assigned by the external tracker
  tlwh    : ndarray[4] — bounding box (top-left x, top-left y, width, height)
  feature : ndarray[D] — L2-normalised re-ID appearance feature
"""


class GroupOnlyTracker:
    """
    Tracks groups of people given pre-tracked pedestrian observations.

    Unlike GroupTracker, this class does NOT perform pedestrian tracking.
    The caller supplies pedestrian track IDs from an external tracker
    (e.g. BoxMOT, ByteTrack) and this class runs only the REGROUP
    group-detection and group-level Kalman tracking on top of them.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        Distance metric for group track association (appearance gallery).
    cost_metric : int
        0 = appearance only, 1 = motion only, 2 = both (default).
    gp_dist : float
        Ground-plane distance threshold (pixels).  Pedestrians whose feet
        differ vertically by more than this value are not grouped.
    h_ratio : float
        Height-ratio threshold in [0, 1].  Pairs whose height ratio falls
        below this are not grouped (filters out very different-sized boxes).
    cif : int
        Crowd-indication-feature threshold.  When the number of mutually
        connected pedestrians exceeds this value the scene is treated as
        crowded and the previous frame's group assignments are reused.
    max_age : int
        Frames a group track survives without a match before deletion.
    n_init : int
        Consecutive matches needed to confirm a new group track.
    """

    def __init__(self, metric, cost_metric=2, gp_dist=20, h_ratio=0.8,
                 cif=4, max_age=30, n_init=3, count_alpha=0.3, n_stable=2,
                 motion_threshold=0.0, min_speed=1.0, vel_window=5,
                 max_gap_factor=1.0):
        self.metric = metric
        self.cost_metric = cost_metric
        self.gp_dist = gp_dist
        self.h_ratio = h_ratio
        self.cif = cif
        self.max_age = max_age
        self.n_init = n_init
        # Weight of count-difference penalty added to the matching cost matrix.
        self.count_alpha = count_alpha
        # Frames a recently-active ped-pair stays connected after proximity is lost.
        self.n_stable = n_stable
        # Minimum cosine similarity between velocity vectors to allow grouping.
        # 0.0 rejects only opposite-direction pairs; raise toward 1.0 for stricter
        # direction agreement.  Set to -1.0 to disable motion gating entirely.
        self.motion_threshold = motion_threshold
        # Peds whose speed (px/frame) is below this are skipped by motion gating
        # because their direction estimate is too noisy.
        self.min_speed = min_speed
        # Number of frames used to estimate each ped's velocity (longer = smoother).
        self._vel_window = vel_window
        # Multiplier on max_w in the gap check: gap > max_gap_factor * max_w splits the pair.
        # Increase above 1.0 to allow pedestrians standing further apart to be grouped.
        self.max_gap_factor = max_gap_factor

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []          # list[GroupTrack]
        self._next_id = 1
        # ped_id -> last assigned group_track_id; used for crowd stability
        self._last_ped_group: dict = {}
        # frozenset({pid_a, pid_b}) -> remaining hysteresis frames
        self._pair_stable_frames: dict = {}
        # ped_id -> deque of recent (cx, cy) positions for velocity estimation
        self._ped_pos_history: dict = {}

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self):
        """Kalman predict step — call once per frame before update()."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, ped_tracks):
        """
        Update group tracks given externally-tracked pedestrians.

        Parameters
        ----------
        ped_tracks : list of PedTrack
            Active pedestrian observations for this frame, each as
            PedTrack(ped_id, tlwh, feature).

        Returns
        -------
        ped_group_ids : dict[int, int]
            Maps each ped_id to a group_track_id.  0 means the pedestrian
            is not currently part of a confirmed group.
        group_boxes : list[ndarray]
            Merged bounding boxes (tlwh) for each active group detection.
        group_track_ids : list[int]
            Group track IDs parallel to group_boxes.
        """
        if len(ped_tracks) < 2:
            self._advance_tracks()
            return {pt.ped_id: 0 for pt in ped_tracks}, [], []

        boxes = np.array([pt.tlwh for pt in ped_tracks])

        # Estimate per-ped velocities from position history.
        velocities = self._update_velocities(ped_tracks)

        # Cluster pedestrians by proximity, gated by motion direction.
        group_labels, adj = self._split_bad_boxes(boxes, velocities)
        # Stabilise labels: retain recently-active pair edges for n_stable frames.
        group_labels = self._apply_membership_hysteresis(ped_tracks, group_labels)

        # Crowd detection: reuse previous group assignments when crowded.
        if self.cif > 0 and adj.size > 0:
            v = utils.GraphDriver(adj)
            if len(v) > self.cif:
                group_labels = np.array(
                    [self._last_ped_group.get(pt.ped_id, -1) for pt in ped_tracks],
                    dtype=float)

        # Build one merged Detection per group cluster.
        group_dets, memberships = self._build_group_detections(ped_tracks, boxes, group_labels)

        # Match group detections against existing group tracks.
        matches, unmatched_tracks, unmatched_dets = self._match(group_dets)

        det_track_ids = np.zeros(len(group_dets), dtype=int)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(self.kf, group_dets[det_idx])
            det_track_ids[det_idx] = self.tracks[track_idx].track_id

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_dets:
            det_track_ids[det_idx] = self._initiate_group_track(group_dets[det_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update appearance gallery.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        feat_list, tgt_list = [], []
        for t in self.tracks:
            if not t.is_confirmed():
                continue
            feat_list += t.features
            tgt_list += [t.track_id] * len(t.features)
            t.features = []
        if feat_list:
            self.metric.group_partial_fit(
                np.asarray(feat_list), np.asarray(tgt_list), active_targets)

        # Map ped_id -> group_track_id.
        ped_group_ids = {pt.ped_id: 0 for pt in ped_tracks}
        for det_idx, member_indices in enumerate(memberships):
            gid = det_track_ids[det_idx]
            for i in member_indices:
                ped_group_ids[ped_tracks[i].ped_id] = gid
        self._last_ped_group = dict(ped_group_ids)

        # Return only confirmed group tracks.
        confirmed = {t.track_id for t in self.tracks if t.is_confirmed()}
        group_boxes, group_track_ids = [], []
        for det_idx, _ in enumerate(memberships):
            gid = det_track_ids[det_idx]
            if gid in confirmed:
                group_boxes.append(group_dets[det_idx].tlwh)
                group_track_ids.append(gid)

        return ped_group_ids, group_boxes, group_track_ids

    # ── internal helpers ──────────────────────────────────────────────────────

    def _advance_tracks(self):
        """Mark all tracks missed and prune deleted ones (used when < 2 peds)."""
        for t in self.tracks:
            t.mark_missed()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _initiate_group_track(self, det):
        gid = self._next_id
        mean, cov = self.kf.initiate(det.to_xyah())
        self.tracks.append(
            GroupTrack(mean, cov, gid, gid, self.n_init, self.max_age, det, det.feature))
        self._next_id += 1
        return gid

    def _build_group_detections(self, ped_tracks, boxes, group_labels):
        """
        Merge pedestrian boxes per cluster label into one Detection each.

        Returns
        -------
        group_dets : list[Detection]
        memberships : list[list[int]]
            Index into ped_tracks for each group detection.
        """
        if len(group_labels) == 0:
            return [], []

        group_dets, memberships = [], []
        for lbl in np.unique(group_labels):
            if lbl == -1:
                continue
            idx = np.where(group_labels == lbl)[0]
            if len(idx) < 2:
                continue
            feat_sum = np.sum([ped_tracks[i].feature for i in idx], axis=0)
            x, y, w, h = self._merge_box(boxes[idx])
            det = detection_module.Detection((x, y, w, h), -1, feat_sum)
            det.member_count = len(idx)
            group_dets.append(det)
            memberships.append(idx.tolist())

        return group_dets, memberships

    def _merge_box(self, boxes):
        """Return tlwh bounding box enclosing all input boxes (tlwh format)."""
        x = max(1.0, float(np.min(boxes[:, 0])))
        y = max(1.0, float(np.min(boxes[:, 1])))
        w = float(np.max(boxes[:, 0] + boxes[:, 2])) - x
        h = float(np.max(boxes[:, 1] + boxes[:, 3])) - y
        return x, y, w, h

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, det_indices):
            feats = np.array([dets[i].feature for i in det_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost = self.metric.group_distance(feats, targets)
            cost = linear_assignment.gate_cost_matrix(
                self.kf, cost, tracks, dets, track_indices, det_indices)
            # Count-aware penalty: penalise implausible membership-count jumps.
            if self.count_alpha > 0:
                for ri, ti in enumerate(track_indices):
                    track = tracks[ti]
                    if not track.member_count_history:
                        continue
                    for ci, di in enumerate(det_indices):
                        if cost[ri, ci] >= linear_assignment.INFTY_COST:
                            continue  # already gated out
                        det_count = getattr(dets[di], 'member_count', None)
                        if det_count is None:
                            continue
                        z = abs(det_count - track.expected_count) / track.count_std
                        cost[ri, ci] += self.count_alpha * z
            return cost

        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        if self.cost_metric == 1:
            matches_a, unmatched_a, unmatched_d = [], [], list(range(len(detections)))
        else:
            matches_a, unmatched_a, unmatched_d = linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed)

        iou_cands = unconfirmed + [
            k for k in unmatched_a if self.tracks[k].time_since_update == 1]
        unmatched_a = [k for k in unmatched_a if self.tracks[k].time_since_update != 1]

        matches_b, unmatched_b, unmatched_d = linear_assignment.min_cost_matching(
            iou_matching.iou_cost, 0.9, self.tracks, detections, iou_cands, unmatched_d)

        if self.cost_metric == 1:
            return matches_b, list(set(unmatched_b)), unmatched_d
        return matches_a + matches_b, list(set(unmatched_a + unmatched_b)), unmatched_d

    def _update_velocities(self, ped_tracks):
        """
        Update per-pedestrian position history and return a velocity list.

        Returns a list of length len(ped_tracks) where each element is either
        a 2-D numpy velocity vector (dx, dy) in pixels/frame or None when there
        is not yet enough history to estimate direction reliably.
        """
        velocities = []
        active_ids = {pt.ped_id for pt in ped_tracks}
        # Drop history for peds that have left the scene.
        stale = [pid for pid in self._ped_pos_history if pid not in active_ids]
        for pid in stale:
            del self._ped_pos_history[pid]

        for pt in ped_tracks:
            cx = pt.tlwh[0] + pt.tlwh[2] / 2
            cy = pt.tlwh[1] + pt.tlwh[3] / 2
            pos = np.array([cx, cy], dtype=float)

            if pt.ped_id not in self._ped_pos_history:
                self._ped_pos_history[pt.ped_id] = deque(maxlen=self._vel_window)
            self._ped_pos_history[pt.ped_id].append(pos)

            hist = self._ped_pos_history[pt.ped_id]
            if len(hist) >= 2:
                # Displacement from oldest to newest sample in the window.
                velocities.append(hist[-1] - hist[0])
            else:
                velocities.append(None)

        return velocities

    def _apply_membership_hysteresis(self, ped_tracks, labels):
        """
        Stabilise cluster labels using pair-wise temporal continuity.

        For every pair of pedestrians that were in the same group last frame,
        their connection is kept alive for up to `n_stable` additional frames
        after the geometry-based clustering separates them.  A pair's counter
        resets to `n_stable` each frame they remain connected, and decrements
        by one each frame they are apart.
        """
        if self.n_stable <= 0:
            return labels

        labels = labels.copy()
        n = len(ped_tracks)

        for i in range(n):
            for j in range(i + 1, n):
                pi = ped_tracks[i].ped_id
                pj = ped_tracks[j].ped_id
                key = frozenset({pi, pj})

                currently_together = (labels[i] != -1 and labels[i] == labels[j])

                if currently_together:
                    self._pair_stable_frames[key] = self.n_stable
                    continue

                # Pair is separated this frame — check hysteresis window.
                remaining = self._pair_stable_frames.get(key, 0)
                if remaining <= 0:
                    self._pair_stable_frames.pop(key, None)
                    continue

                self._pair_stable_frames[key] = remaining - 1

                # Restore connection: merge the two labels.
                li, lj = labels[i], labels[j]
                if li != -1 and lj == -1:
                    labels[j] = li
                elif lj != -1 and li == -1:
                    labels[i] = lj
                elif li != -1 and lj != -1:
                    # Both in different clusters: fold the larger label into the smaller.
                    lo, hi = min(li, lj), max(li, lj)
                    labels[labels == hi] = lo
                else:
                    # Both ungrouped: assign a fresh label.
                    new_lbl = int(np.max(labels[labels != -1]) + 1) if np.any(labels != -1) else 1
                    labels[i] = new_lbl
                    labels[j] = new_lbl

        # Drop singletons that hysteresis may have created.
        for lbl in np.unique(labels[labels != -1]):
            if np.sum(labels == lbl) == 1:
                labels[labels == lbl] = -1

        return labels

    def _split_bad_boxes(self, boxes, velocities=None):
        """
        Cluster pedestrian boxes by spatial proximity and motion direction.

        Parameters
        ----------
        boxes : ndarray, shape (n, 4)
            Bounding boxes in tlwh format.
        velocities : list[ndarray | None], optional
            Per-pedestrian velocity vectors (dx, dy) indexed to match boxes.
            When provided, pairs whose velocity cosine similarity is below
            self.motion_threshold are removed from the adjacency graph.

        Returns per-pedestrian integer group labels (numpy float array) and the
        binary adjacency matrix used for crowd detection.  Label -1 means the
        pedestrian is not part of any group.
        """
        n = len(boxes)
        if n < 2:
            return np.full(n, -1.0), np.zeros((n, n))

        dist_w = np.zeros((n, n))
        dist_h = np.zeros((n, n))
        dist_bot = np.zeros((n, n))

        for j in range(n):
            for k in range(n):
                bj, bk = boxes[j], boxes[k]
                cj = bj[0] + bj[2] / 2
                ck = bk[0] + bk[2] / 2
                gap = abs(cj - ck) - (bj[2] / 2 + bk[2] / 2)
                bot_diff = abs((bj[1] + bj[3]) - (bk[1] + bk[3]))
                max_w = statistics.mean([bj[2], bk[2]])
                min_h = min(bj[3], bk[3])
                max_h = max(bj[3], bk[3])
                h_ratio = round(min_h / max_h, 2) if max_h > 0 else 0.0

                if (gap > self.max_gap_factor * max_w or bot_diff >= self.gp_dist
                        or h_ratio <= self.h_ratio or j == k):
                    dist_w[j, k] = 0.0
                else:
                    dist_w[j, k] = gap

                dist_h[j, k] = h_ratio
                dist_bot[j, k] = bot_diff

        adj = (dist_w != 0).astype(float)

        # Motion gating: remove edges between peds moving in incompatible directions.
        if velocities is not None and self.motion_threshold > -1.0:
            for j in range(n):
                for k in range(j + 1, n):
                    if adj[j, k] == 0:
                        continue
                    vj = velocities[j]
                    vk = velocities[k]
                    if vj is None or vk is None:
                        continue  # no history yet — keep edge
                    sj = np.linalg.norm(vj)
                    sk = np.linalg.norm(vk)
                    if sj < self.min_speed or sk < self.min_speed:
                        continue  # nearly stationary — direction unreliable, keep edge
                    cos_sim = np.dot(vj, vk) / (sj * sk)
                    if cos_sim < self.motion_threshold:
                        adj[j, k] = adj[k, j] = 0.0
                        dist_w[j, k] = dist_w[k, j] = 0.0

        # Normalise for aggregate proximity score.
        dist_w_n = normalize(dist_w, norm='l1')
        dist_h_n = normalize(dist_h, norm='l1')
        dist_b_n = normalize(dist_bot, norm='l1')
        aggregate = np.where(adj > 0, dist_h_n * dist_b_n * dist_w_n, 100.0)

        # Greedy nearest-pair assignment.
        pairs = np.full(n, -1, dtype=int)
        for j in range(n):
            if not np.all(aggregate[j] >= 100.0):
                pairs[j] = int(np.argmin(aggregate[j]))

        labels = np.zeros(n)
        next_id = 1
        for j in range(n):
            p = pairs[j]
            if p == -1:
                continue
            if labels[j] != 0 or labels[p] != 0:
                lbl = max(labels[j], labels[p])
                labels[j] = lbl
                labels[p] = lbl
            else:
                labels[j] = next_id
                labels[p] = next_id
                next_id += 1

        # Singletons and unassigned get label -1.
        for i in range(n):
            if labels[i] == 0:
                labels[i] = -1
        for lbl in np.unique(labels[labels != -1]):
            idx = np.where(labels == lbl)[0]
            if len(idx) == 1:
                labels[idx] = -1

        return labels.astype(float), adj


class OdometryGroupTracker(GroupOnlyTracker):
    """
    GroupOnlyTracker with per-track ego-motion compensation via robot odometry.

    The robot's odometry (position + orientation quaternion in world frame) is
    fed once per frame through ``update_pose()``.  During ``predict()`` each
    group track's Kalman mean is shifted by the pixel displacement that a
    stationary ground-plane (z = 0 in world frame) point at the track's foot
    position would undergo due to the camera moving between the previous and
    current frame.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
    K : array-like, shape (3, 3)
        Camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    T_cam_robot : array-like, shape (4, 4)
        Fixed rigid transform from robot body frame to camera frame:
        ``p_cam = T_cam_robot @ p_robot``.
    **kwargs
        Forwarded to :class:`GroupOnlyTracker` (e.g. cost_metric, gp_dist,
        h_ratio, cif, max_age, n_init, count_alpha, n_stable,
        motion_threshold, min_speed, vel_window).

    Usage
    -----
    ::

        tracker = OdometryGroupTracker(metric, K=K, T_cam_robot=T_cam_robot,
                                       gp_dist=20, h_ratio=0.8, cif=4)
        # per-frame loop:
        tracker.update_pose(position, orientation)   # odometry for this frame
        tracker.predict()                            # ego-compensated KF predict
        ped_gids, boxes, gids = tracker.update(ped_tracks)
    """

    def __init__(self, metric, K, T_cam_robot, **kwargs):
        super().__init__(metric, **kwargs)
        self._K = np.asarray(K, dtype=float)
        self._K_inv = np.linalg.inv(self._K)
        self._T_cam_robot = np.asarray(T_cam_robot, dtype=float)
        # T_world_cam for the two most recent frames (None until first pose update).
        self._T_world_cam_curr = None
        self._T_world_cam_prev = None

    # ── public API ────────────────────────────────────────────────────────────

    def update_pose(self, position, orientation):
        """
        Record the latest robot pose from odometry.

        Must be called once per frame **before** :meth:`predict`.

        Parameters
        ----------
        position : array-like, shape (3,)
            Robot origin in world frame [x, y, z].
        orientation : array-like, shape (4,)
            Robot orientation as quaternion [qx, qy, qz, qw].
        """
        self._T_world_cam_prev = self._T_world_cam_curr
        T_world_robot = self._pose_to_matrix(position, orientation)
        T_robot_cam = np.linalg.inv(self._T_cam_robot)
        self._T_world_cam_curr = T_world_robot @ T_robot_cam

    def predict(self):
        """
        Kalman predict with per-track ego-motion pixel compensation.

        Each group track's foot point is backprojected to the z = 0 world
        plane using the previous camera pose, then reprojected using the
        current camera pose.  The resulting pixel shift is injected as
        ``ego_delta_xyah`` into :meth:`GroupTrack.predict`.

        Falls back to uncompensated prediction whenever poses are unavailable
        or the backprojection is degenerate (ray parallel to ground, etc.).
        """
        for track in self.tracks:
            ego_delta = self._compute_ego_delta_xyah(track)
            track.predict(self.kf, ego_delta)

    # ── geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pose_to_matrix(position, orientation):
        """Build a 4x4 T_world_robot from position and quaternion [qx,qy,qz,qw]."""
        qx, qy, qz, qw = orientation
        R = np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    def _backproject_to_ground(self, u, v, T_world_cam):
        """
        Backproject pixel (u, v) to the z = 0 ground plane in world frame.

        Parameters
        ----------
        u, v : float
            Pixel coordinates.
        T_world_cam : ndarray, shape (4, 4)
            Transform from camera frame to world frame.

        Returns
        -------
        ndarray of shape (3,), or None if the ray is degenerate.
        """
        d_cam = self._K_inv @ np.array([u, v, 1.0])
        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]       # camera origin in world frame
        d_world = R_wc @ d_cam
        if abs(d_world[2]) < 1e-6:      # ray nearly parallel to ground
            return None
        t = -t_wc[2] / d_world[2]
        if t < 0:                        # intersection behind the camera
            return None
        return t_wc + t * d_world

    def _project_to_image(self, P_world, T_world_cam):
        """
        Project a 3-D world point to pixel coordinates.

        Returns
        -------
        ndarray of shape (2,), or None if the point is behind the camera.
        """
        T_cam_world = np.linalg.inv(T_world_cam)
        P_cam = T_cam_world @ np.array([P_world[0], P_world[1], P_world[2], 1.0])
        if P_cam[2] <= 0:
            return None
        uv = self._K @ P_cam[:3] / P_cam[2]
        return uv[:2]

    def _compute_ego_delta_xyah(self, track):
        """
        Compute the xyah ego-motion correction [du, dv, 0, 0] for *track*.

        The correction is the pixel shift a stationary ground-plane point at
        the track's foot would experience due solely to camera ego-motion.

        Returns None when poses are unavailable or geometry is degenerate.
        """
        if self._T_world_cam_prev is None or self._T_world_cam_curr is None:
            return None

        # Foot point of the group box: bottom-centre of the xyah bounding box.
        cx, cy, h = track.mean[0], track.mean[1], track.mean[3]
        u_foot = cx
        v_foot = cy + h / 2.0

        # Backproject using the *previous* camera pose (where the box was).
        P_world = self._backproject_to_ground(u_foot, v_foot, self._T_world_cam_prev)
        if P_world is None:
            return None

        # Reproject using the *current* camera pose.
        uv_curr = self._project_to_image(P_world, self._T_world_cam_curr)
        if uv_curr is None:
            return None

        # Pixel displacement = ego-motion translation for the whole box.
        du = uv_curr[0] - u_foot
        dv = uv_curr[1] - v_foot
        return np.array([du, dv, 0.0, 0.0])
