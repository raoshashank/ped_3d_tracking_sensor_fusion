#!/usr/bin/env python3
"""Fuse ped3d (camera, FOV only) and lidar (360 deg) pedestrian tracks into one track set per bag.

Inputs (SLAM map frame):
  ped3d   $P3D_PED3D_OUT/go2nus/<bag>_0.pkl, per-frame detections (frames[k]): 3D positions
          in base_link at camera time timestamps[k], placed in the map with the SLAM base_link pose
  lidar   output_ped/lidar_tracks_smooth.csv (smoothed, every scan) + lidar_tracks.csv (raw detections),
          moved from the visible-surface centroid to the body centre (BODY_PUSH along the ray), the
          convention ped3d uses; fused positions are body centres

1. Measurements, no double counting. ped3d's per-frame position mixes the same lidar cluster the
   lidar detector sees, so its two components are used separately:
     vggt   pos_vggt_c (xy scaled by the track's lidar anchor), sigma_vggt(depth), x2 if multimodal;
            independent of the lidar detector -> always used
     lidar  pos_lidar_g (>= MIN_LIDAR_PTS points), sigma_lidar(range, n); used only when the lidar
            detector has no detection of that person within REDUNDANT_DT
   Lidar detections: raw positions, sigma_lidar(range, n_points).
2. Association: per camera frame, Hungarian between ped3d detections (inverse-variance of their
   components) and lidar tracks (smoothed, interpolated to the frame time), gate MATCH_R.
3. Swap repair: along a lidar track, runs of >= MIN_RUN frames matched to one ped3d track; when the
   run changes to another ped3d track that co-exists at that moment, the lidar track is cut there.
4. Persons: (lidar segment, ped3d track) edges with enough agreement (>= MIN_MATCH frames, >= MIN_FRAC
   of the frames where both exist, median distance <= MED_MAX) are merged greedily by support,
   refusing merges that would put two members co-existing > CONFLICT_D apart into one person.
   ped3d fragments get joined through lidar, lidar gaps through ped3d.
5. Per person: all measurements -> robust constant-velocity RTS smoother (lidar_tracker) on the
   union of scan times and measurement times; then re-linking of persons across gaps
   (lidar_tracker.link_cost) and re-fusion of linked chains.
6. Labels: camera_verified (lidar + ped3d), camera_only (ped3d, no lidar: far, occluded, or part of
   the static map), lidar_only (no ped3d; in the camera FOV within FOV_R for < UNCONF_S),
   unconfirmed_in_fov (no ped3d although in the FOV within FOV_R for >= UNCONF_S: likely a bike,
   cart or foliage; kept, flagged). Standing people are not held: only evidence is output.

Outputs (output_ped/):
  fused_tracks.csv          person, scan, t, x, y, vx, vy, pos_std, source (lidar|ped3d|both|filled),
                            in_fov, range, top_h, label       (10 Hz scan grid)
  fused_track_members.csv   person, kind (lidar|ped3d), member_id, t_start, t_end, n_matched, label
Evaluation (printed): association / person stats, agreement with ped3d single frames within FOV_R, and a
lidar-dropout hold-out: hide 1 s of lidar detections of camera-verified persons, compare lidar alone
(smoother bridges the gap) with fused-with-ped3d at the hidden (lidar-accurate) detections.
Run with the erasor2 env from erasor2/:  PYTHONPATH=. python fuse_tracks.py [--holdout] [--no-write] [--out-dir DIR]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R, Slerp

import lidar_ped_detect as LD
import lidar_tracker as LT
from ped_tracks import PED3D

MIN_LIDAR_PTS = 4
COMP_K = 3.0          # ped3d lidar vs VGGT: disagreement (along the ray) above COMP_K x combined sigma -> drop lidar
VEL_PRIOR = 1.4       # m/s, prior std of a person's velocity in the smoother (people walk at ~1-1.4 m/s):
                      # keeps 2-3-frame camera-only tracks from getting their speed from depth noise
V_MAX = 3.0           # m/s, fastest plausible pedestrian (walking is ~1-1.4 m/s; leaves room for jogging)
SPLIT_ID = 100000     # ped3d tracks split at implausible jumps: segment j gets id pid + j * SPLIT_ID
REDUNDANT_DT = 0.06
MATCH_R = 0.8
MIN_RUN = 3
MIN_MATCH, MIN_FRAC, MED_MAX = 3, 0.6, 0.5
CONFLICT_D, CONFLICT_OVERLAP = 1.0, 0.3
FOV_R, UNCONF_S = 10.0, 1.0
BASE_TO_GROUND = 0.35
# measured: ped3d (vggt and lidar components) sit +0.08..+0.13 m farther along the viewing ray than the
# lidar detections (median over 3 bags, <= 10 m), no tangential offset
BODY_PUSH = 0.12


def sigma_vggt(d):
    return 0.05 + 0.065 * d          # ped3d_lib.sigma_vggt, go2nus vggt_trust = 1


def sigma_lidar(d, n):
    return (0.10 + 0.015 * d) * (1.0 + 4.0 / max(n, 1))


# ------------------------------------------------------------------ inputs
class Pose:
    def __init__(self):
        b = np.loadtxt(LD.HERE.parent / "poses_base_link_tum.txt")
        self.t, self.p = b[:, 0], b[:, 1:4]
        self.slerp = Slerp(self.t, R.from_quat(b[:, 4:8]))

    def to_map(self, t, q):
        tc = float(np.clip(t, self.t[0], self.t[-1]))
        Rm = self.slerp([tc]).as_matrix()[0]
        p = np.array([np.interp(tc, self.t, self.p[:, k]) for k in range(3)])
        return (Rm @ q + p)[:2]


class PedTrack:
    """ped3d track: per detected frame (k, t, comps[(kind, xy, sigma)], xy combined, sigma combined)."""
    def __init__(self, pid):
        self.pid, self.obs = pid, []

    def finish(self):
        self.obs.sort(key=lambda o: o[1])
        self.t = np.array([o[1] for o in self.obs])
        self.xy = np.array([o[3] for o in self.obs]).reshape(-1, 2)
        self.sig = np.array([o[4] for o in self.obs])
        self.t0, self.t1 = (self.t[0], self.t[-1]) if len(self.t) else (np.inf, -np.inf)
        return self


def load_ped3d(pose, frame_filter=None):
    d = pickle.load(open(PED3D, "rb"))
    tracks = {}
    for k, dets in enumerate(d["frames"]):
        if frame_filter is not None and not frame_filter(k):
            continue
        t = float(d["timestamps"][k])
        for det in dets:
            tr = d["tracks"].get(det["id"])
            if tr is None:
                continue
            comps = []
            q = np.asarray(det.get("pos_vggt_c", [np.nan] * 3), float)
            if np.isfinite(q).all():
                a = float(tr.get("anchor", 1.0))
                q = q.copy(); q[:2] *= a
                s = sigma_vggt(float(det["depth_vggt_c"]) * a) * (2.0 if "vggt_multimodal" in det["flags"] else 1.0)
                comps.append(("vggt", pose.to_map(t, q), s))
            q = np.asarray(det.get("pos_lidar_g", [np.nan] * 3), float)
            if det.get("n_lidar_g", 0) >= MIN_LIDAR_PTS and np.isfinite(q).all():
                cl = ("lidar", pose.to_map(t, q), sigma_lidar(float(det["depth_lidar_g"]), det["n_lidar_g"]))
                # ped3d sometimes takes the lidar cluster of someone/something in front of the person
                # (1.5 m vs VGGT's 9 m): drop the lidar part when it disagrees with VGGT along the ray
                if comps:
                    rob = pose.to_map(t, np.zeros(3))
                    dr = abs(np.linalg.norm(cl[1] - rob) - np.linalg.norm(comps[0][1] - rob))
                    if dr > COMP_K * np.hypot(cl[2], comps[0][2]):
                        cl = None
                if cl is not None:
                    comps.append(cl)
            if not comps:
                continue
            w = np.array([1 / c[2] ** 2 for c in comps])
            xy = sum(c[1] * wi for c, wi in zip(comps, w)) / w.sum()
            tracks.setdefault(int(det["id"]), PedTrack(int(det["id"]))).obs.append((k, t, comps, xy, float(np.sqrt(1 / w.sum()))))
    out = {}
    for pid, tr in tracks.items():
        for j, seg in enumerate(plausible_segments(sorted(tr.obs, key=lambda o: o[1]))):
            p = PedTrack(pid if j == 0 else pid + SPLIT_ID * j)   # split ped3d tracks get new ids
            p.obs = seg
            out[p.pid] = p.finish()
    return out


def implausible(a, b):
    """Consecutive ped3d observations (k, t, comps, xy, sigma) faster than V_MAX, allowing for noise."""
    return np.linalg.norm(a[3] - b[3]) > V_MAX * abs(b[1] - a[1]) + 3 * np.hypot(a[4], b[4]) + 0.3


def plausible_segments(obs):
    """Pedestrians walk at ~1-1.4 m/s. Drop isolated outliers (inconsistent with both neighbours while the
    neighbours agree), then split the track wherever consecutive observations still imply > V_MAX
    (an identity switch in ped3d's tracker, or a depth jump): returns lists of observations."""
    obs = list(obs)
    changed = True
    while changed and len(obs) >= 3:
        changed = False
        for i in range(1, len(obs) - 1):
            if implausible(obs[i - 1], obs[i]) and implausible(obs[i], obs[i + 1]) and not implausible(obs[i - 1], obs[i + 1]):
                del obs[i]
                changed = True
                break
    segs = [[obs[0]]] if obs else []
    for o in obs[1:]:
        if implausible(segs[-1][-1], o):
            segs.append([o])
        else:
            segs[-1].append(o)
    return segs


class LidarTrack:
    def __init__(self, lid, sm, raw):
        self.lid = lid
        self.ts, self.xys = sm[:, 2], sm[:, 3:5]             # smoothed, every scan
        self.t0, self.t1 = self.ts[0], self.ts[-1]
        self.raw = raw                                       # scan, t, x, y, top_h, height, n, range, in_fov

    def at(self, t):
        return np.array([np.interp(t, self.ts, self.xys[:, 0]), np.interp(t, self.ts, self.xys[:, 1])])


def body_centre(xy, scans, poses):
    """Lidar detections are centroids of the visible (robot-facing) surface; ped3d positions are body
    centres (surface median pushed back 0.15 m along the ray). Push lidar positions BODY_PUSH along
    the ray from the lidar so both use the body-centre convention."""
    rob = poses[scans.astype(int), :2, 3]
    u = xy - rob
    return xy + BODY_PUSH * u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)


def load_lidar(poses):
    sm = np.loadtxt(LD.OUT / "lidar_tracks_smooth.csv", delimiter=",", skiprows=1).reshape(-1, 12)
    raw = np.loadtxt(LD.OUT / "lidar_tracks.csv", delimiter=",", skiprows=1).reshape(-1, 10)
    sm[:, 3:5] = body_centre(sm[:, 3:5], sm[:, 1], poses)
    raw[:, 3:5] = body_centre(raw[:, 3:5], raw[:, 1], poses)
    out = {}
    for lid in np.unique(sm[:, 0]).astype(int):
        s = sm[sm[:, 0] == lid]
        s = s[np.argsort(s[:, 2])]
        r = raw[raw[:, 0] == lid][:, 1:]
        out[lid] = LidarTrack(lid, s, r[np.argsort(r[:, 1])])
    return out


# ------------------------------------------------------------------ association
class Segment:
    """Time slice [t0, t1] of a lidar track (tracks are cut where they swap between people)."""
    def __init__(self, L, t0, t1, sid):
        self.L, self.t0, self.t1, self.sid = L, t0, t1, sid
        m = (L.ts >= t0) & (L.ts <= t1)
        self.ts, self.xys = L.ts[m], L.xys[m]
        mr = (L.raw[:, 1] >= t0) & (L.raw[:, 1] <= t1)
        self.raw = L.raw[mr]

    def at(self, t):
        return self.L.at(t)


def frame_links(P, L):
    """Per camera frame Hungarian between ped3d detections and lidar tracks -> [(t, pid, lid, dist)]."""
    by_t = {}
    for pid, p in P.items():
        for t, xy in zip(p.t, p.xy):
            by_t.setdefault(t, []).append((pid, xy))
    links = []
    Ls = list(L.values())
    for t, dets in by_t.items():
        live = [l for l in Ls if l.t0 - 0.05 <= t <= l.t1 + 0.05]
        if not live:
            continue
        C = np.array([[np.linalg.norm(xy - l.at(t)) for l in live] for _, xy in dets])
        for a, b in zip(*linear_sum_assignment(C)):
            if C[a, b] <= MATCH_R:
                links.append((t, dets[a][0], live[b].lid, C[a, b]))
    return links


def cut_segments(L, P, links):
    """Split lidar tracks where their matched ped3d id switches between co-existing ped3d tracks."""
    segs, n_cuts = [], 0
    by_l = {}
    for t, pid, lid, dd in links:
        by_l.setdefault(lid, []).append((t, pid))
    for lid, l in L.items():
        seq = sorted(by_l.get(lid, []))
        runs = []
        for t, pid in seq:
            if runs and runs[-1][0] == pid:
                runs[-1][2], runs[-1][3] = t, runs[-1][3] + 1
            else:
                runs.append([pid, t, t, 1])
        runs = [r for r in runs if r[3] >= MIN_RUN]
        cuts = []
        for (pa, _, ta, _), (pb, tb, _, _) in zip(runs[:-1], runs[1:]):
            if pa != pb and P[pa].t0 <= tb <= P[pa].t1 and P[pb].t0 <= ta <= P[pb].t1:
                cuts.append(0.5 * (ta + tb))
        edges = [l.t0 - 1e-6] + cuts + [l.t1 + 1e-6]
        for k in range(len(edges) - 1):
            segs.append(Segment(l, edges[k], edges[k + 1], (lid, k)))
        n_cuts += len(cuts)
    return segs, n_cuts


def edges_between(segs, P, links):
    by = {}
    for t, pid, lid, dd in links:
        by.setdefault((lid, pid), []).append((t, dd))
    out = []
    for s in segs:
        for pid, p in P.items():
            v = [(t, dd) for t, dd in by.get((s.L.lid, pid), []) if s.t0 <= t <= s.t1]
            if len(v) < MIN_MATCH:
                continue
            opp = ((p.t >= max(s.t0, s.L.t0)) & (p.t <= min(s.t1, s.L.t1))).sum()
            dd = np.array([x[1] for x in v])
            if len(v) >= MIN_FRAC * max(opp, 1) and np.median(dd) <= MED_MAX:
                out.append((len(v), s, p))
    return out


def member_track(m):
    return (m.ts, m.xys) if isinstance(m, Segment) else (m.t, m.xy)


def conflict(a, b):
    ta, xa = member_track(a)
    tb, xb = member_track(b)
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if hi - lo < CONFLICT_OVERLAP:
        return False
    m = (ta >= lo) & (ta <= hi)
    if m.sum() < 3:
        return False
    d = np.hypot(xa[m, 0] - np.interp(ta[m], tb, xb[:, 0]), xa[m, 1] - np.interp(ta[m], tb, xb[:, 1]))
    return np.median(d) > CONFLICT_D


def build_persons(segs, P, edges):
    comp = {}          # member key -> component id
    members = {}       # component id -> list of members
    nxt = 0

    def key(m):
        return ("L",) + m.sid if isinstance(m, Segment) else ("P", m.pid)

    for m in list(segs) + list(P.values()):
        comp[key(m)] = nxt
        members[nxt] = [m]
        nxt += 1
    rejected = 0
    for n, s, p in sorted(edges, key=lambda e: -e[0]):
        ca, cb = comp[key(s)], comp[key(p)]
        if ca == cb:
            continue
        if any(conflict(a, b) for a in members[ca] for b in members[cb]):
            rejected += 1
            continue
        for m in members[cb]:
            comp[key(m)] = ca
        members[ca] += members.pop(cb)
    return [ms for ms in members.values()], rejected


# ------------------------------------------------------------------ fusion per person
class Person:
    def __init__(self, members, t_scan):
        self.members = members
        self.segs = [m for m in members if isinstance(m, Segment)]
        self.peds = [m for m in members if not isinstance(m, Segment)]
        rows = []   # t, x, y, sigma, kind
        lid_t = []
        for s in self.segs:
            for r in s.raw:
                rows.append((r[1], r[2], r[3], sigma_lidar(r[7], r[6]), "lidar", r[4]))
                lid_t.append(r[1])
        lid_t = np.sort(np.array(lid_t)) if lid_t else np.zeros(0)
        for p in self.peds:
            for k, t, comps, _, _ in p.obs:
                has_lidar = len(lid_t) and np.min(np.abs(lid_t - t)) <= REDUNDANT_DT
                for kind, xy, sg in comps:
                    if kind == "lidar" and has_lidar:
                        continue
                    rows.append((t, xy[0], xy[1], sg, "ped3d", np.nan))
        rows.sort(key=lambda r: r[0])
        # combine same-instant measurements (inverse variance)
        comb = []
        for r in rows:
            if comb and abs(comb[-1][0] - r[0]) < 1e-4:
                c = comb[-1]
                w1, w2 = 1 / c[3] ** 2, 1 / r[3] ** 2
                comb[-1] = (c[0], (c[1] * w1 + r[1] * w2) / (w1 + w2), (c[2] * w1 + r[2] * w2) / (w1 + w2),
                            np.sqrt(1 / (w1 + w2)), c[4] if c[4] == r[4] else "both",
                            c[5] if np.isfinite(c[5]) else r[5])
            else:
                comb.append(r)
        self.meas = comb
        self.t0, self.t1 = comb[0][0], comb[-1][0]
        # grid = scan times inside the span + measurement times
        si = np.nonzero((t_scan >= self.t0 - 1e-6) & (t_scan <= self.t1 + 1e-6))[0]
        if len(si) == 0:
            si = np.array([int(np.argmin(np.abs(t_scan - 0.5 * (self.t0 + self.t1))))])
        gt = np.concatenate([t_scan[si], [m[0] for m in comb]])
        order = np.argsort(gt, kind="stable")
        gt = gt[order]
        n_s = len(si)
        is_scan = order < n_s
        meas = np.full((len(gt), 2), np.nan)
        sig = np.ones(len(gt))
        pos_in_grid = np.argsort(order)
        for j, m in enumerate(comb):
            g = pos_in_grid[n_s + j]
            meas[g], sig[g] = (m[1], m[2]), m[3]
        xs, Pp, _ = LT.robust_rts(gt, meas, sig, vel_std=VEL_PRIOR)
        keep = is_scan & np.isfinite(xs[:, 0])
        if not keep.any():   # too short to span a scan time (e.g. one ped3d frame): output the measurements
            keep = ~is_scan & np.isfinite(xs[:, 0])
            near = np.array([int(np.argmin(np.abs(t_scan - t))) for t in gt[keep]])
            order = order.copy()
            order[keep] = n_s + np.arange(keep.sum())
            si = np.concatenate([si, near])
        self.scan = si[order[keep]]
        self.t = gt[keep]
        self.pos, self.vel = xs[keep, :2], xs[keep, 2:4]
        self.std = np.sqrt(np.maximum(np.nan_to_num(Pp[keep], nan=1.0), 0))
        mt = np.array([m[0] for m in comb])
        mk = np.array([m[4] for m in comb])
        mtop = np.array([m[5] for m in comb])
        self.source, self.top = [], []
        for t in self.t:
            near = np.abs(mt - t) <= 0.05
            kinds = set(mk[near])
            lid = "lidar" in kinds or "both" in kinds
            ped = "ped3d" in kinds or "both" in kinds
            self.source.append("both" if lid and ped else "lidar" if lid else "ped3d" if ped else "filled")
            tops = mtop[near & np.isfinite(mtop)]
            self.top.append(tops[0] if len(tops) else np.nan)
        self.n_det = int(sum(s != "filled" for s in self.source))

    @property
    def label(self):
        return self._label


def relink_persons(persons, t_scan):
    """lidar_tracker re-linking on persons; linked chains are re-fused from their members."""
    ps = [p for p in persons if len(p.t) >= 2]
    rest = [p for p in persons if len(p.t) < 2]
    ps.sort(key=lambda p: p.t[0])
    n = len(ps)
    ends = np.array([p.t[-1] for p in ps])
    starts = np.array([p.t[0] for p in ps])
    C = {}
    for i in range(n):
        for j in np.nonzero((starts > ends[i]) & (starts <= ends[i] + LT.MAX_GAP_S))[0]:
            c = LT.link_cost(ps[i], ps[j])
            if c is not None:
                C[(i, j)] = c + LT.LAMBDA_T * (starts[j] - ends[i])
    if not C:
        return persons, 0
    rows, cols = sorted({i for i, _ in C}), sorted({j for _, j in C})
    ri, ci = {r: k for k, r in enumerate(rows)}, {c: k for k, c in enumerate(cols)}
    M = np.full((len(rows) + len(cols), len(cols) + len(rows)), 1e6)
    for (i, j), c in C.items():
        M[ri[i], ci[j]] = c
    for k in range(len(rows)):
        M[k, len(cols) + k] = LT.GATE
    for k in range(len(cols)):
        M[len(rows) + k, k] = LT.GATE
    M[len(rows):, len(cols):] = 0.0
    links = []
    for r, c in zip(*linear_sum_assignment(M)):
        if r >= len(rows) or c >= len(cols) or M[r, c] >= LT.GATE:
            continue
        i, j = rows[r], cols[c]
        runner = min(min([v for (a, b), v in C.items() if a == i and b != j], default=np.inf),
                     min([v for (a, b), v in C.items() if b == j and a != i], default=np.inf))
        if runner < LT.GATE and M[r, c] > LT.AMBIG_RATIO * runner:
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
        out.append(ps[i] if len(chain) == 1 else Person([m for k in chain for m in ps[k].members], t_scan))
    return out + rest, len(links)


def label_persons(persons, poses, T_base, cam):
    for p in persons:
        p.in_fov = np.array([LD.in_fov(xy[None], np.array([poses[s, 2, 3] - LD.GROUND_BELOW + 1.0]), T_base[s], cam)[0]
                             for s, xy in zip(p.scan, p.pos)], bool)
        p.range = np.linalg.norm(p.pos - poses[p.scan, :2, 3], axis=1)
        near_fov_s = 0.1 * np.sum(p.in_fov & (p.range <= FOV_R))
        if p.peds and p.segs:
            p._label = "camera_verified"
        elif p.peds:
            p._label = "camera_only"
        elif near_fov_s >= UNCONF_S:
            p._label = "unconfirmed_in_fov"
        else:
            p._label = "lidar_only"


def fuse(P, L, t_scan):
    links = frame_links(P, L)
    segs, n_cuts = cut_segments(L, P, links)
    edges = edges_between(segs, P, links)
    groups, rejected = build_persons(segs, P, edges)
    persons = [Person(g, t_scan) for g in groups]
    persons, n_rel = relink_persons(persons, t_scan)
    persons.sort(key=lambda p: p.t[0] if len(p.t) else np.inf)
    for k, p in enumerate(persons):
        p.pid = k
    stats = dict(lidar_tracks=len(L), ped3d_tracks=len(P), frame_links=len(links), swap_cuts=n_cuts,
                 edges=len(edges), rejected_conflicts=rejected, relinks=n_rel, persons=len(persons))
    return persons, stats


# ------------------------------------------------------------------ outputs + evaluation
def write(persons, out=None):
    out = Path(out) if out else LD.OUT
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fused_tracks.csv", "w") as f:
        f.write("person,scan,t,x,y,vx,vy,pos_std,source,in_fov,range,top_h,label\n")
        for p in persons:
            for k in range(len(p.t)):
                f.write(f"{p.pid},{p.scan[k]},{p.t[k]:.3f},{p.pos[k, 0]:.3f},{p.pos[k, 1]:.3f},{p.vel[k, 0]:.3f},"
                        f"{p.vel[k, 1]:.3f},{p.std[k]:.3f},{p.source[k]},{int(p.in_fov[k])},{p.range[k]:.2f},"
                        f"{p.top[k]:.2f},{p.label}\n")
    with open(out / "fused_track_members.csv", "w") as f:
        f.write("person,kind,member_id,t_start,t_end,n_obs,label\n")
        for p in persons:
            for s in p.segs:
                f.write(f"{p.pid},lidar,{s.L.lid},{max(s.t0, s.L.t0):.3f},{min(s.t1, s.L.t1):.3f},{len(s.raw)},{p.label}\n")
            for q in p.peds:
                f.write(f"{p.pid},ped3d,{q.pid},{q.t0:.3f},{q.t1:.3f},{len(q.t)},{p.label}\n")


def person_at(persons, t):
    out = []
    for p in persons:
        if len(p.t) and p.t[0] - 0.05 <= t <= p.t[-1] + 0.05:
            out.append((p, np.array([np.interp(t, p.t, p.pos[:, 0]), np.interp(t, p.t, p.pos[:, 1])])))
    return out


def report(persons, P, stats, name):
    lab = {}
    for p in persons:
        lab[p.label] = lab.get(p.label, 0) + 1
    ver = [p for p in persons if p.peds]
    frag = np.mean([len(p.peds) for p in ver]) if ver else 0
    cross = [p for p in persons if p.peds and p.segs and 0.1 * np.sum(~p.in_fov) >= 1.0]
    # agreement: each ped3d detection vs its own person's fused position at that time
    owner = {q.pid: p for p in persons for q in p.peds}
    d = []   # within FOV_R of the robot (ped3d single frames scatter by metres beyond ~15 m)
    for pid, q in P.items():
        p = owner.get(pid)
        if p is None or len(p.t) < 2:
            continue
        for t, xy in zip(q.t, q.xy):
            if p.t[0] <= t <= p.t[-1]:
                k = min(np.searchsorted(p.t, t), len(p.t) - 1)
                if p.range[k] <= FOV_R:
                    d.append(np.hypot(*(xy - [np.interp(t, p.t, p.pos[:, 0]), np.interp(t, p.t, p.pos[:, 1])])))
    d = np.array(d) if d else np.full(1, np.nan)
    dur = np.array([p.t[-1] - p.t[0] for p in persons if len(p.t) > 1])
    print(f"[{name}] {stats}")
    print(f"[{name}] labels {lab}; ped3d tracks per camera-verified person {frag:.2f}; "
          f"camera-verified persons followed >= 1 s outside the FOV: {len(cross)}/{len([p for p in persons if p.peds and p.segs])}; "
          f"median person duration {np.median(dur):.1f} s; fused vs ped3d single-frame positions (<= {FOV_R:.0f} m) "
          f"median {np.median(d):.3f} m p90 {np.percentile(d, 90):.3f} m")


def holdout(persons, t_scan, window_s=1.0, every_s=4.0, seed=0):
    """Lidar-dropout test on camera-verified persons: hide lidar detections in 1 s windows, then
    compare, at the hidden detections (lidar-accurate reference), the person re-fused with ped3d
    against the same person from lidar alone (the smoother bridges the gap)."""
    import copy
    rng = np.random.default_rng(seed)
    err_f, err_l, gaps = [], [], 0
    for p in persons:
        if not (p.peds and p.segs) or p.t[-1] - p.t[0] < 3 * window_s:
            continue
        n_win = max(1, int((p.t[-1] - p.t[0]) / every_s))
        for _ in range(n_win):
            w0 = rng.uniform(p.t[0] + window_s, p.t[-1] - 2 * window_s)
            w1 = w0 + window_s
            segs, hidden = [], []
            for s in p.segs:
                s2 = copy.copy(s)
                m = (s.raw[:, 1] >= w0) & (s.raw[:, 1] < w1)
                hidden += [r for r in s.raw[m] if r[7] <= FOV_R]
                s2.raw = s.raw[~m]
                segs.append(s2)
            if len(hidden) < 3 or not any(len(s.raw) for s in segs):
                continue
            gaps += 1
            fused = Person(segs + p.peds, t_scan)
            lidar = Person(segs, t_scan)
            for r in hidden:
                for q, e in ((fused, err_f), (lidar, err_l)):
                    if q.t[0] <= r[1] <= q.t[-1]:
                        e.append(np.hypot(r[2] - np.interp(r[1], q.t, q.pos[:, 0]), r[3] - np.interp(r[1], q.t, q.pos[:, 1])))
                    else:
                        e.append(np.nan)
    err_f, err_l = np.array(err_f), np.array(err_l)
    ok = np.isfinite(err_f) & np.isfinite(err_l)
    print(f"[holdout] {gaps} hidden {window_s:.1f} s lidar windows, {ok.sum()} hidden detections <= {FOV_R:.0f} m: "
          f"lidar alone (smoother bridges the gap) median {np.median(err_l[ok]):.3f} p90 {np.percentile(err_l[ok], 90):.3f} m | "
          f"fused with ped3d median {np.median(err_f[ok]):.3f} p90 {np.percentile(err_f[ok], 90):.3f} m | "
          f"coverage: lidar alone {np.isfinite(err_l).mean():.0%}, fused {np.isfinite(err_f).mean():.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=float, nargs="*", default=None, metavar="WINDOW_S",
                    help="lidar-dropout test with these window lengths (default 1.0)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out-dir", help="write the outputs here instead of output_ped/ (for comparisons)")
    a = ap.parse_args()
    t_scan, T_base = LD.base_poses()
    poses = np.loadtxt(LD.D / "poses.txt").reshape(-1, 3, 4)
    cam = np.load(LD.HERE / "camera_model.npz")
    cam = {k: cam[k] for k in cam.files}
    pose = Pose()
    P = {k: v for k, v in load_ped3d(pose).items() if len(v.t)}
    L = load_lidar(poses)
    persons, stats = fuse(P, L, t_scan)
    label_persons(persons, poses, T_base, cam)
    report(persons, P, stats, "all")
    if a.holdout is not None:
        for w in (a.holdout or [1.0]):
            holdout(persons, t_scan, window_s=w)
    if not a.no_write:
        write(persons, a.out_dir)


if __name__ == "__main__":
    main()
