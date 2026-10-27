#!/usr/bin/env python3
"""Multi-target tracking of the lidar-vs-map pedestrian detections (360 deg, 10 Hz, SLAM map frame).

Replaces v1 (persistence filter + greedy nearest-neighbour linking, no motion model):
  1. online: constant-velocity Kalman filter per track; each scan, Hungarian assignment of the raw
     detections (lidar_ped_detect.detect_raw) with a Mahalanobis gate; a track is confirmed after
     N_CONFIRM hits (tentative tracks die after MAX_MISS_TENT misses), a confirmed track coasts up
     to MAX_COAST_S without detections
  2. offline, per confirmed track: constant-velocity RTS smoother with robust re-weighting
     (ped3d_lib.rts_smooth / robust_rts, copied below) -> positions, velocities, gaps filled
  3. duplicate merge + re-linking of broken tracks across gaps <= MAX_GAP_S (ped3d relink.py logic,
     chi^2 gate on forward/backward constant-velocity prediction, Hungarian, ambiguity rejection),
     linked chains re-smoothed
Measurement noise per detection: ped3d's sigma_lidar(range, n_points).

Outputs (output_ped/):
  lidar_tracks_smooth.csv  track, scan, t, x, y, vx, vy, pos_std, detected, in_fov, range
                           every scan of every track (detected = 0: filled by the smoother)
  lidar_tracks.csv         detected scans only, raw detection columns (same format as v1, used by
                           ped_mask.py --lidar-tracks and replay_detections.py)
  lidar_dets.npz           dets, track (detected rows, raw detection columns)

Metrics against ped3d (camera tracks, inside the FOV, <= EVAL_R) -- ped3d is not ground truth:
  recall / precision   as lidar_ped_detect.evaluate
  id_changes           changes of the matched lidar id along each ped3d track (fragmentation)
  swaps                changes of the matched ped3d id along a lidar track between two ped3d
                       tracks that both exist at that moment (two different people)
Run with the erasor2 env from erasor2/:  PYTHONPATH=. python lidar_tracker.py [--compare-v1]
"""
import argparse
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import lidar_ped_detect as LD
from ped_tracks import PED3D, load_tracks

# Tuned on 9 go2nus bags (utown_1 + 8 random crowded ones) against ped3d, total id errors
# (lidar-id changes along ped3d tracks + swaps between co-existing people): v1 greedy ~1950,
# coast 1 s / gap 3 s / gate 9.21 / q 1.2: ~1150, these values: ~1000 (fewer swaps, same fragmentation)
Q_ACC = 0.8                 # m/s^2, process noise (ped3d rts_smooth uses 1.2 at 4 Hz)
GATE_ASSIGN = 5.99          # chi^2 2 dof 95 %
MAX_ASSIGN_DIST = 1.5       # m
N_CONFIRM, MAX_MISS_TENT = 3, 2
MAX_COAST_S = 0.5
P0_VEL = 1.0                # m/s initial velocity std
MIN_HITS = 5                # confirmed tracks with fewer detections are dropped
# re-linking (relink.py logic, frames at 10 Hz)
MAX_GAP_S, DUP_DIST = 1.5, 0.4
GATE, LAMBDA_T, AMBIG_RATIO, V_MAX, DV_MAX = 5.99, 0.5, 0.85, 2.5, 2.0
SIG_V, MIN_VEL_FRAMES = 0.25, 10
EVAL_R, MATCH_R = LD.EVAL_R, LD.MATCH_R


def sigma_lidar(d, n):
    return (0.10 + 0.015 * d) * (1.0 + 4.0 / max(n, 1))


# ------------------------------------------------------------------ smoother (ped3d_lib, verbatim)
def rts_smooth(t, meas, sig, q_acc=None, vel_std=2.0):
    q_acc = Q_ACC if q_acc is None else q_acc
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
    Pf[first] = np.diag([sig[first] ** 2, sig[first] ** 2, vel_std ** 2, vel_std ** 2])   # vel_std: prior on speed
    xp[first], Pq[first], Fs[first] = xf[first], Pf[first], np.eye(4)
    for k in range(first + 1, n):
        dt = max(float(t[k] - t[k - 1]), 1e-3)
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        Qa = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * q_acc ** 2
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Qa
        Q[np.ix_([1, 3], [1, 3])] = Qa
        xp[k] = F @ xf[k - 1]
        Pq[k] = F @ Pf[k - 1] @ F.T + Q
        Fs[k] = F
        if have[k]:
            S = Hm @ Pq[k] @ Hm.T + np.eye(2) * sig[k] ** 2
            Kg = Pq[k] @ Hm.T @ np.linalg.inv(S)
            xf[k] = xp[k] + Kg @ (meas[k] - Hm @ xp[k])
            Pf[k] = (np.eye(4) - Kg @ Hm) @ Pq[k]
        else:
            xf[k], Pf[k] = xp[k], Pq[k]
    xs[n - 1] = xf[n - 1]
    Ps = Pf[n - 1].copy()
    Pp[n - 1] = Ps[0, 0]
    for k in range(n - 2, first - 1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.pinv(Pq[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps = Pf[k] + C @ (Ps - Pq[k + 1]) @ C.T
        Pp[k] = Ps[0, 0]
    return xs, Pp


def robust_rts(t, meas, sig, passes=2, gate=3.0, vel_std=2.0):
    sig = sig.copy()
    xs = Pp = None
    for _ in range(passes):
        xs, Pp = rts_smooth(t, meas, sig, vel_std=vel_std)
        r = np.linalg.norm(meas - xs[:, :2], axis=1)
        nr = r / np.sqrt(sig ** 2 + np.nan_to_num(Pp, nan=0.0))
        bad = np.isfinite(nr) & (nr > gate)
        if not bad.any():
            break
        sig[bad] *= 5.0
    return xs, Pp, sig


# ------------------------------------------------------------------ online tracker
class KF:
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def __init__(self, tid, z, sig, t, scan, j):
        self.tid, self.t = tid, t
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([sig ** 2, sig ** 2, P0_VEL ** 2, P0_VEL ** 2])
        self.hits, self.misses, self.confirmed = 1, 0, False
        self.t_hit = t
        self.assoc = [(scan, j)]          # (scan, detection row)

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        Qa = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * Q_ACC ** 2
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Qa
        Q[np.ix_([1, 3], [1, 3])] = Qa
        self.x, self.P, self.t = F @ self.x, F @ self.P @ F.T + Q, t

    def d2(self, z, sig):
        S = self.H @ self.P @ self.H.T + np.eye(2) * sig ** 2
        y = z - self.x[:2]
        return float(y @ np.linalg.solve(S, y))

    def update(self, z, sig, scan, j):
        S = self.H @ self.P @ self.H.T + np.eye(2) * sig ** 2
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x[:2])
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hits += 1
        self.misses = 0
        self.t_hit = self.t
        self.assoc.append((scan, j))
        if self.hits >= N_CONFIRM:
            self.confirmed = True


def track_online(dets, t_scan):
    """dets: raw detections (scan, t, x, y, top_h, height, n, range, in_fov). Returns list of KF."""
    by_scan = {}
    for j, d in enumerate(dets):
        by_scan.setdefault(int(d[0]), []).append(j)
    live, done, nid = [], [], 0
    for i, t in enumerate(t_scan):
        js = by_scan.get(i, [])
        for tr in live:
            tr.predict(t)
        z = dets[js, 2:4] if js else np.zeros((0, 2))
        sig = np.array([sigma_lidar(dets[j, 7], dets[j, 6]) for j in js])
        matched_t, matched_d = set(), set()
        if live and js:
            C = np.full((len(live), len(js)), 1e6)
            for a, tr in enumerate(live):
                for b in range(len(js)):
                    if np.linalg.norm(z[b] - tr.x[:2]) > MAX_ASSIGN_DIST:
                        continue
                    d2 = tr.d2(z[b], sig[b])
                    if d2 < GATE_ASSIGN:
                        C[a, b] = d2 + (0.0 if tr.confirmed else 1.0)   # prefer confirmed tracks
            ra, cb = linear_sum_assignment(C)
            for a, b in zip(ra, cb):
                if C[a, b] < 1e6:
                    live[a].update(z[b], sig[b], i, js[b])
                    matched_t.add(a)
                    matched_d.add(b)
        keep = []
        for a, tr in enumerate(live):
            if a not in matched_t:
                tr.misses += 1
            dead = (not tr.confirmed and tr.misses > MAX_MISS_TENT) or (tr.confirmed and t - tr.t_hit > MAX_COAST_S)
            (done if dead else keep).append(tr)
        live = keep
        for b in range(len(js)):
            if b not in matched_d:
                live.append(KF(nid, z[b], sig[b], t, i, js[b]))
                nid += 1
    return [tr for tr in done + live if tr.confirmed and tr.hits >= MIN_HITS]


# ------------------------------------------------------------------ offline smoothing + re-linking
class Trk:
    """Smoothed track on the scan grid. frames: consecutive scan indices; det_row: -1 when filled."""
    def __init__(self, tid, frames, pos, vel, std, det_row, t, members=None):
        self.tid, self.frames, self.pos, self.vel, self.std, self.det_row, self.t = tid, frames, pos, vel, std, det_row, t
        self.members = list(members) if members is not None else [tid]

    @property
    def det(self):
        return self.det_row >= 0

    @property
    def n_det(self):
        return int(self.det.sum())


def smooth_assoc(tid, assoc, dets, t_scan, members=None):
    """assoc: [(scan, det_row)] (several rows per scan allowed: the lowest sigma wins)."""
    f0 = min(s for s, _ in assoc)
    f1 = max(s for s, _ in assoc)
    frames = np.arange(f0, f1 + 1)
    meas = np.full((len(frames), 2), np.nan)
    sig = np.full(len(frames), 1.0)
    row = -np.ones(len(frames), int)
    for s, j in assoc:
        k = s - f0
        sj = sigma_lidar(dets[j, 7], dets[j, 6])
        if row[k] < 0 or sj < sig[k]:
            meas[k], sig[k], row[k] = dets[j, 2:4], sj, j
    t = t_scan[frames]
    xs, Pp, _ = robust_rts(t, meas, sig)
    return Trk(tid, frames, xs[:, :2], xs[:, 2:4], np.sqrt(np.maximum(np.nan_to_num(Pp, nan=1.0), 0)), row, t, members)


def assoc_of(tr):
    return [(int(f), int(r)) for f, r in zip(tr.frames, tr.det_row) if r >= 0]


def merge_duplicates(tracks, dets, t_scan):
    """Union tracks within DUP_DIST for >= 70 % of an overlap of >= 5 scans (one person, two clusters)."""
    tracks = sorted(tracks, key=lambda tr: tr.frames[0])
    parent = list(range(len(tracks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            b = tracks[j]
            if b.frames[0] > a.frames[-1]:
                break
            lo, hi = b.frames[0], min(a.frames[-1], b.frames[-1])
            if hi - lo + 1 < 5:
                continue
            d = np.linalg.norm(a.pos[lo - a.frames[0]:hi - a.frames[0] + 1] - b.pos[lo - b.frames[0]:hi - b.frames[0] + 1], axis=1)
            if (d < DUP_DIST).mean() >= 0.7:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(tracks)):
        groups.setdefault(find(i), []).append(tracks[i])
    out, n_dup = [], 0
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        n_dup += len(g) - 1
        out.append(smooth_assoc(min(tr.tid for tr in g), [a for tr in g for a in assoc_of(tr)], dets, t_scan,
                                [m for tr in g for m in tr.members]))
    return out, n_dup


def end_state(tr, which):
    k = -1 if which == "end" else 0
    v = tr.vel[k].copy()
    sv = SIG_V
    if tr.n_det < MIN_VEL_FRAMES or not np.isfinite(v).all():
        v[:] = 0.0
        sv = 0.8
    return tr.t[k], tr.pos[k], v, max(float(np.nan_to_num(tr.std[k], nan=0.3)), 0.05), sv


def link_cost(a, b):
    ta, pa, va, sa, sva = end_state(a, "end")
    tb, pb, vb, sb, svb = end_state(b, "start")
    dt = tb - ta
    if dt <= 0 or dt > MAX_GAP_S:
        return None
    if np.linalg.norm(pb - pa) > V_MAX * dt + 0.5:
        return None
    if np.linalg.norm(va - vb) > DV_MAX and a.n_det >= MIN_VEL_FRAMES and b.n_det >= MIN_VEL_FRAMES:
        return None
    fwd = pa + va * dt - pb
    bwd = pb - vb * dt - pa
    var = sa ** 2 + sb ** 2 + ((sva + svb) / 2 * dt) ** 2
    d2 = 0.5 * (fwd @ fwd + bwd @ bwd) / var
    return d2 if d2 < GATE else None


def relink(tracks, dets, t_scan):
    tracks = sorted(tracks, key=lambda tr: tr.frames[0])
    n = len(tracks)
    ends = np.array([tr.t[-1] for tr in tracks])
    starts = np.array([tr.t[0] for tr in tracks])
    C = {}
    for i in range(n):
        for j in np.nonzero((starts > ends[i]) & (starts <= ends[i] + MAX_GAP_S))[0]:
            c = link_cost(tracks[i], tracks[j])
            if c is not None:
                C[(i, j)] = c + LAMBDA_T * (starts[j] - ends[i])
    if not C:
        return tracks, 0, 0
    rows, cols = sorted({i for i, _ in C}), sorted({j for _, j in C})
    ri, ci = {r: k for k, r in enumerate(rows)}, {c: k for k, c in enumerate(cols)}
    M = np.full((len(rows) + len(cols), len(cols) + len(rows)), 1e6)
    for (i, j), c in C.items():
        M[ri[i], ci[j]] = c
    for k in range(len(rows)):
        M[k, len(cols) + k] = GATE
    for k in range(len(cols)):
        M[len(rows) + k, k] = GATE
    M[len(rows):, len(cols):] = 0.0
    links, n_amb = [], 0
    for r, c in zip(*linear_sum_assignment(M)):
        if r >= len(rows) or c >= len(cols) or M[r, c] >= GATE:
            continue
        i, j = rows[r], cols[c]
        runner = min(min([v for (a, b), v in C.items() if a == i and b != j], default=np.inf),
                     min([v for (a, b), v in C.items() if b == j and a != i], default=np.inf))
        if runner < GATE and M[r, c] > AMBIG_RATIO * runner:
            n_amb += 1
            continue
        links.append((i, j))
    nxt, has_prev = dict(links), {j for _, j in links}
    out = []
    for i in range(n):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) == 1:
            out.append(tracks[i])
        else:
            parts = [tracks[k] for k in chain]
            out.append(smooth_assoc(parts[0].tid, [a for tr in parts for a in assoc_of(tr)], dets, t_scan,
                                    [m for tr in parts for m in tr.members]))
    return out, len(links), n_amb


def run_tracker(dets, t_scan):
    online = track_online(dets, t_scan)
    tracks = [smooth_assoc(tr.tid, tr.assoc, dets, t_scan) for tr in online]
    tracks, n_dup = merge_duplicates(tracks, dets, t_scan)
    tracks, n_link, n_amb = relink(tracks, dets, t_scan)
    tracks = sorted(tracks, key=lambda tr: tr.frames[0])
    for k, tr in enumerate(tracks):
        tr.tid = k
    return tracks, dict(online=len(online), duplicates=n_dup, links=n_link, ambiguous=n_amb, final=len(tracks))


# ------------------------------------------------------------------ outputs + metrics
def fov_flags(tracks, t_scan):
    """in_fov for every smoothed row (camera_model.npz, SLAM base_link poses)."""
    _, T_base = LD.base_poses()
    poses = np.loadtxt(LD.D / "poses.txt").reshape(-1, 3, 4)
    cam = np.load(LD.HERE / "camera_model.npz")
    cam = {k: cam[k] for k in cam.files}
    out = []
    for tr in tracks:
        f = np.array([LD.in_fov(p[None], np.array([poses[s, 2, 3] - LD.GROUND_BELOW + 1.0]), T_base[s], cam)[0]
                      for s, p in zip(tr.frames, tr.pos)])
        out.append(f)
    return out


def write_outputs(tracks, dets, t_scan):
    poses = np.loadtxt(LD.D / "poses.txt").reshape(-1, 3, 4)
    fov = fov_flags(tracks, t_scan)
    with open(LD.OUT / "lidar_tracks_smooth.csv", "w") as f:
        f.write("track,scan,t,x,y,vx,vy,pos_std,detected,in_fov,range,top_h\n")
        for tr, fv in zip(tracks, fov):
            for k, s in enumerate(tr.frames):
                rng = np.linalg.norm(tr.pos[k] - poses[s, :2, 3])
                r = tr.det_row[k]
                top = f"{dets[r, 4]:.2f}" if r >= 0 else "nan"
                f.write(f"{tr.tid},{s},{tr.t[k]:.3f},{tr.pos[k, 0]:.3f},{tr.pos[k, 1]:.3f},{tr.vel[k, 0]:.3f},"
                        f"{tr.vel[k, 1]:.3f},{tr.std[k]:.3f},{int(r >= 0)},{int(fv[k])},{rng:.2f},{top}\n")
    rows = [(tr.tid, r) for tr in tracks for r in tr.det_row if r >= 0]
    track = np.array([t for t, _ in rows], int)
    d = dets[[r for _, r in rows]]
    with open(LD.OUT / "lidar_tracks.csv", "w") as f:
        f.write("track,scan,t,x,y,top_h,height,n_pts,range,in_fov\n")
        for tid, row in zip(track, d):
            f.write(f"{tid}," + ",".join(f"{v:.3f}" for v in row) + "\n")
    np.savez(LD.OUT / "lidar_dets.npz", dets=d, track=track)
    return d, track


def ped3d_reference():
    """ped3d detected (not interpolated) positions in the SLAM frame: (pid, t, x, y) + track spans."""
    d = pickle.load(open(PED3D, "rb"))
    tracks = load_tracks("fused")
    gt, spans = [], {}
    for tid, tr in d["tracks"].items():
        if tid not in tracks:
            continue
        t, xy, _ = tracks[tid]
        ok = np.isfinite(tr["fused"][:, :2]).all(1)
        det = tr["detected"][ok] & ~tr["interpolated"][ok]
        gt += [(tid, tt, x, y) for tt, (x, y), dd in zip(t, xy, det) if dd]
        spans[tid] = (t[0], t[-1], t, xy)
    return np.array(gt).reshape(-1, 4), spans


def tracking_metrics(rows, name, t_scan):
    """rows: (track, scan, t, x, y, in_fov) of every output row of a tracker."""
    gt, spans = ped3d_reference()
    poses = np.loadtxt(LD.D / "poses.txt").reshape(-1, 3, 4)
    by_scan = {}
    for r in rows:
        by_scan.setdefault(int(r[1]), []).append(r)
    by_scan = {k: np.array(v) for k, v in by_scan.items()}
    match = []   # (pid, t, lidar id or -1)
    for pid, tt, x, y in gt:
        i = int(np.argmin(np.abs(t_scan - tt)))
        if abs(t_scan[i] - tt) > 0.08 or np.linalg.norm(poses[i, :2, 3] - [x, y]) > EVAL_R:
            continue
        lid = -1
        if i in by_scan:
            dd = np.linalg.norm(by_scan[i][:, 3:5] - [x, y], axis=1)
            if dd.min() <= MATCH_R:
                lid = int(by_scan[i][dd.argmin(), 0])
        match.append((int(pid), tt, lid))
    match = np.array(match)
    recall = (match[:, 2] >= 0).mean()
    id_changes = 0
    for pid in np.unique(match[:, 0]):
        seq = match[(match[:, 0] == pid) & (match[:, 2] >= 0)]
        seq = seq[np.argsort(seq[:, 1]), 2]
        id_changes += int((np.diff(seq) != 0).sum())
    # precision + swaps: every in-FOV lidar row <= EVAL_R vs ped3d positions at that time
    fov = rows[(rows[:, 5] > 0)]
    fov = fov[np.linalg.norm(fov[:, 3:5] - poses[fov[:, 1].astype(int), :2, 3], axis=1) <= EVAL_R]
    prec, lmatch = [], []
    for r in fov:
        best, bd = None, MATCH_R
        for pid, (t0, t1, t, xy) in spans.items():
            if t0 - 0.15 <= r[2] <= t1 + 0.15:
                p = np.array([np.interp(r[2], t, xy[:, 0]), np.interp(r[2], t, xy[:, 1])])
                dd = np.linalg.norm(p - r[3:5])
                if dd <= bd:
                    best, bd = pid, dd
        prec.append(best is not None)
        if best is not None:
            lmatch.append((int(r[0]), r[2], best))
    # a swap: the lidar track follows ped3d person A for >= MIN_RUN matches, then B for >= MIN_RUN,
    # and A and B both exist at the switch (two people, not ped3d id churn); shorter runs are
    # nearest-neighbour flicker between people standing close together and are ignored
    MIN_RUN = 3
    swaps = 0
    lmatch = np.array(lmatch).reshape(-1, 3)
    for lid in np.unique(lmatch[:, 0]):
        seq = lmatch[lmatch[:, 0] == lid]
        seq = seq[np.argsort(seq[:, 1])]
        runs = []   # (pid, t_first, t_last, n)
        for _, tt, pid in seq:
            if runs and runs[-1][0] == pid:
                runs[-1][2], runs[-1][3] = tt, runs[-1][3] + 1
            else:
                runs.append([pid, tt, tt, 1])
        runs = [r for r in runs if r[3] >= MIN_RUN]
        for (pa, _, ta, _), (pb, tb, _, _) in zip(runs[:-1], runs[1:]):
            if pa != pb:
                a0, a1 = spans[int(pa)][:2]
                b0, b1 = spans[int(pb)][:2]
                if a0 <= tb <= a1 and b0 <= ta <= b1:   # both people exist at the switch: a real swap
                    swaps += 1
    tids = np.unique(rows[:, 0])
    dur = np.array([np.ptp(rows[rows[:, 0] == k, 2]) for k in tids])
    out_fov = [k for k in tids if (rows[rows[:, 0] == k, 5] == 0).mean() > 0.5]
    dur_out = np.array([np.ptp(rows[rows[:, 0] == k, 2]) for k in out_fov]) if out_fov else np.zeros(1)
    n_ped = len(np.unique(match[:, 0]))
    print(f"{name:14s} tracks {len(tids):4d} (median {np.median(dur):4.1f} s, >=3 s {np.mean(dur >= 3):4.0%}; "
          f"mostly outside FOV {len(out_fov):3d}, median {np.median(dur_out):4.1f} s) | vs ped3d in FOV: recall {recall:5.1%} "
          f"precision {np.mean(prec):5.1%} | lidar-id changes per ped3d track {id_changes / max(n_ped, 1):.2f} "
          f"({id_changes} over {n_ped}) | swaps {swaps}")
    return dict(tracks=len(tids), recall=recall, precision=float(np.mean(prec)), id_changes=id_changes,
                ped3d_tracks=n_ped, swaps=swaps, median_dur=float(np.median(dur)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-v1", action="store_true", help="also score the v1 greedy tracker on the same detections")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override module constants, e.g. MAX_COAST_S=0.5")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        assert k in globals(), k
        globals()[k] = type(globals()[k])(float(v))
    t_scan = LD.base_poses()[0]
    dets = LD.detect_raw()
    tracks, st = run_tracker(dets, t_scan)
    print(f"raw detections {len(dets)}; tracker: {st}")
    if a.compare_v1:
        keep = LD.persistent(dets)
        dv1 = dets[keep]
        tv1 = LD.greedy_link(dv1)
        tracking_metrics(np.c_[tv1, dv1[:, [0, 1, 2, 3, 8]]], "v1 greedy", t_scan)
    fov = fov_flags(tracks, t_scan)
    det_rows = np.array([(tr.tid, s, t, *dets[r, 2:4], dets[r, 8]) for tr in tracks
                         for s, t, r in zip(tr.frames, tr.t, tr.det_row) if r >= 0])
    all_rows = np.array([(tr.tid, s, t, x, y, f) for tr, fv in zip(tracks, fov)
                         for s, t, (x, y), f in zip(tr.frames, tr.t, tr.pos, fv)])
    tracking_metrics(det_rows, "v2 detected", t_scan)
    tracking_metrics(all_rows, "v2 smoothed", t_scan)
    if not a.no_write:
        d, track = write_outputs(tracks, dets, t_scan)
        gt, _ = ped3d_reference()
        LD.figure(d, track, gt[:, 1:])


if __name__ == "__main__":
    main()
