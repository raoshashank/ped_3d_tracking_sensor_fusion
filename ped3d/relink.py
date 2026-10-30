"""Track re-linking in metric space (fix tracker ID churn) + validation. Used by export_tracks.py
(merged_tracks / merged_id fields); run directly for the analysis report.

    $P3D_CONDA_ENVS/calib/bin/python relink.py [--dataset go2nus ...] [--n-val 150]

Input: the per-corpus training pickles (tracks_<dataset>.pkl, export_tracks format 2).

Algorithm (per trajectory, world/odom frame, non-causal):
  1. duplicate merge: two tracks overlapping in time whose positions stay within DUP_DIST for
     most of the overlap are one person -> keep the union
  2. candidates: track A ending at t1, track B starting at t2 in (t1, t1 + MAX_GAP]; A is
     extrapolated forward and B backward with their smoother velocities; the normalised
     prediction error (isotropic sigma from both tracks' posterior std, velocity uncertainty and
     process noise growing with the gap) must pass a chi^2 gate, plus a speed-plausibility bound
     and a velocity-consistency bound
  3. one-to-one assignment (Hungarian) with cost = d^2 + LAMBDA_T * gap, "leave unlinked" cost
     = gate; links whose cost is not clearly better than the runner-up for either end are
     rejected (ambiguity rejection, crowds)
  4. chains are followed to merged tracks and re-smoothed with the RTS smoother (gap frames
     flagged interpolated)

Validation: long tracks are artificially cut (gap 1-4 s) and we count whether the two halves are
re-linked to each other (correct), to something else (wrong) or not at all (miss).
"""
import argparse
import os
import pickle
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import ped3d_lib as L

MAX_GAP_S = 4.6       # 16 frames at ~4 Hz
DUP_DIST = 0.4
# gate / ambiguity chosen on synthetic go2nus cuts (120 trajectories): 5.99 / 0.85 gives 69 % correct,
# 5 % wrong, 26 % missed re-links; ambiguity off (1.0) buys +2 % correct for +3 % wrong links
GATE = 5.99           # chi^2 2 dof, 95 %
LAMBDA_T = 0.5        # cost per second of gap
AMBIG_RATIO = 0.85    # best cost must be < ratio * runner-up (when the runner-up is within the gate)
V_MAX = 2.5           # m/s, plausibility bound
DV_MAX = 2.0          # m/s, velocity consistency
# prediction error of a constant-velocity extrapolation across a gap, measured on artificially cut
# go2nus tracks: median 0.14 m at 1 s, 0.28 m at 2 s, 0.6 m at 3-4 s -> sigma ~ SIG_V * gap
SIG_V = 0.25          # m/s (larger for short tracks whose velocity is unreliable)
MIN_VEL_FRAMES = 4    # velocity considered reliable from this many detected frames


class Trk:
    __slots__ = ('tid', 'frames', 'pos', 'vel', 'std', 'det', 't', 'members')

    def __init__(self, tid, frames, pos, vel, std, det, t, members=None):
        self.tid, self.frames, self.pos, self.vel, self.std, self.det, self.t = tid, frames, pos, vel, std, det, t
        self.members = list(members) if members is not None else [tid]   # original track ids

    @property
    def n_det(self):
        return int(self.det.sum())


def tracks_from_entry(entry):
    ts = entry['timestamps']
    out = []
    for tid, tr in entry['tracks'].items():
        ok = np.isfinite(tr['pos_world']).all(1)
        if ok.sum() < 2:
            continue
        out.append(Trk(int(tid), tr['frames'], tr['pos_world'].astype(float), tr['vel_world'].astype(float),
                       tr['pos_std'].astype(float), tr['detected'].astype(bool), ts[tr['frames']]))
    return out


def end_state(tr, which):
    """(t, pos, vel, pos_std, vel_std) at the start or end of a track."""
    k = -1 if which == 'end' else 0
    v = tr.vel[k].copy()
    sv = SIG_V
    if tr.n_det < MIN_VEL_FRAMES or not np.isfinite(v).all():
        v[:] = 0.0
        sv = 0.8
    return tr.t[k], tr.pos[k], v, max(float(np.nan_to_num(tr.std[k], nan=0.3)), 0.05), sv


def link_cost(a, b):
    """Normalised squared prediction error between track a's end and track b's start (or None)."""
    ta, pa, va, sa, sva = end_state(a, 'end')
    tb, pb, vb, sb, svb = end_state(b, 'start')
    dt = tb - ta
    if dt <= 0 or dt > MAX_GAP_S:
        return None
    disp = np.linalg.norm(pb - pa)
    if disp > V_MAX * dt + 0.5:
        return None
    if np.linalg.norm(va - vb) > DV_MAX and a.n_det >= MIN_VEL_FRAMES and b.n_det >= MIN_VEL_FRAMES:
        return None
    fwd = pa + va * dt - pb
    bwd = pb - vb * dt - pa
    var = sa ** 2 + sb ** 2 + ((sva + svb) / 2 * dt) ** 2
    d2 = 0.5 * (fwd @ fwd + bwd @ bwd) / var
    return d2 if d2 < GATE else None


def merge_duplicates(tracks):
    """Union tracks that co-exist within DUP_DIST for >= 70 % of their temporal overlap (>= 3 frames)."""
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
            if hi - lo + 1 < 3:
                continue
            pa = a.pos[lo - a.frames[0]:hi - a.frames[0] + 1]
            pb = b.pos[lo - b.frames[0]:hi - b.frames[0] + 1]
            ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
            if ok.sum() < 3:
                continue
            d = np.linalg.norm(pa[ok] - pb[ok], axis=1)
            if (d < DUP_DIST).mean() >= 0.7:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(tracks)):
        groups.setdefault(find(i), []).append(tracks[i])
    merged, n_dup = [], 0
    for g in groups.values():
        if len(g) == 1:
            merged.append(g[0])
            continue
        n_dup += len(g) - 1
        merged.append(fuse_tracks(g, keep_id=min(tr.tid for tr in g)))
    return merged, n_dup


def fuse_tracks(parts, keep_id, ts_all=None):
    """Combine tracks (disjoint or overlapping in time) into one, re-smoothed with the RTS smoother."""
    f0 = min(tr.frames[0] for tr in parts)
    f1 = max(tr.frames[-1] for tr in parts)
    n = f1 - f0 + 1
    meas = np.full((n, 2), np.nan)
    sig = np.full(n, 1.0)
    det = np.zeros(n, bool)
    t = np.full(n, np.nan)
    for tr in parts:
        for k, fr in enumerate(tr.frames):
            i = fr - f0
            t[i] = tr.t[k]
            if tr.det[k] and np.isfinite(tr.pos[k]).all():
                s = max(float(np.nan_to_num(tr.std[k], nan=0.3)), 0.08)
                if not det[i] or s < sig[i]:
                    meas[i], sig[i], det[i] = tr.pos[k], s, True
    # timestamps inside gaps: linear interpolation of the sampling times
    bad = np.isnan(t)
    if bad.any():
        t[bad] = np.interp(np.nonzero(bad)[0], np.nonzero(~bad)[0], t[~bad])
    xs, Pp, _ = L.robust_rts(t, meas, sig)
    members = [m for tr in sorted(parts, key=lambda tr: tr.frames[0]) for m in tr.members]
    return Trk(keep_id, np.arange(f0, f1 + 1), xs[:, :2], xs[:, 2:4], np.sqrt(np.maximum(np.nan_to_num(Pp, nan=1.0), 0)),
               det, t, members=members)


def relink_entry(entry):
    """Full pipeline on an export trajectory entry: duplicate merge + re-linking.
    Returns (merged Trk list, links, n_duplicates, n_ambiguous)."""
    tracks = tracks_from_entry(entry)
    if not tracks:
        return [], [], 0, 0
    tracks, n_dup = merge_duplicates(tracks)
    merged, links, n_amb = relink(tracks)
    return merged, links, n_dup, n_amb


def relink(tracks):
    """Returns (merged tracks, links [(id_a, id_b, gap_s, cost)], n_rejected_ambiguous)."""
    tracks = sorted(tracks, key=lambda tr: tr.frames[0])
    n = len(tracks)
    ends = np.array([tr.t[-1] for tr in tracks])
    starts = np.array([tr.t[0] for tr in tracks])
    C = {}
    for i in range(n):
        cand = np.nonzero((starts > ends[i]) & (starts <= ends[i] + MAX_GAP_S))[0]
        for j in cand:
            c = link_cost(tracks[i], tracks[j])
            if c is not None:
                C[(i, j)] = c + LAMBDA_T * (starts[j] - ends[i])
    if not C:
        return tracks, [], 0
    rows = sorted({i for i, _ in C})
    cols = sorted({j for _, j in C})
    ri = {r: k for k, r in enumerate(rows)}
    ci = {c: k for k, c in enumerate(cols)}
    BIG = 1e6
    M = np.full((len(rows) + len(cols), len(cols) + len(rows)), BIG)
    M[:len(rows), :len(cols)] = BIG
    for (i, j), c in C.items():
        M[ri[i], ci[j]] = c
    # "leave unlinked" options: row i -> dummy column, dummy row -> column j, at the gate cost
    for k in range(len(rows)):
        M[k, len(cols) + k] = GATE
    for k in range(len(cols)):
        M[len(rows) + k, k] = GATE
    M[len(rows):, len(cols):] = 0.0
    r_idx, c_idx = linear_sum_assignment(M)
    links, n_amb = [], 0
    for r, c in zip(r_idx, c_idx):
        if r >= len(rows) or c >= len(cols) or M[r, c] >= GATE:
            continue
        i, j = rows[r], cols[c]
        best = M[r, c]
        alt_i = min([v for (a, b), v in C.items() if a == i and b != j], default=np.inf)
        alt_j = min([v for (a, b), v in C.items() if b == j and a != i], default=np.inf)
        runner = min(alt_i, alt_j)
        if runner < GATE and best > AMBIG_RATIO * runner:
            n_amb += 1
            continue
        links.append((i, j, float(starts[j] - ends[i]), float(best)))
    nxt = {i: j for i, j, _, _ in links}
    has_prev = {j for _, j, _, _ in links}
    merged = []
    for i in range(n):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) == 1:
            merged.append(tracks[i])
        else:
            merged.append(fuse_tracks([tracks[k] for k in chain], keep_id=tracks[chain[0]].tid))
    return merged, [(tracks[i].tid, tracks[j].tid, g, c) for i, j, g, c in links], n_amb


# ----------------------------------------------------------------------------- validation
def synthetic_validation(tracks, rng, gaps_s=(1, 2, 3, 4), fps=4, per_traj=4):
    """Cut long tracks into two fragments and check the re-linking. Returns rows (gap_s, outcome)."""
    long_tracks = [tr for tr in tracks if tr.n_det >= 40]
    if not long_tracks:
        return [], tracks
    rng.shuffle(long_tracks)
    rows, new_tracks = [], list(tracks)
    next_id = max(tr.tid for tr in tracks) + 1
    frag_pairs = {}
    for tr in long_tracks[:per_traj]:
        g = int(rng.choice(gaps_s)) * fps
        n = len(tr.frames)
        if n < 2 * 8 + g + 2:
            continue
        cut = int(rng.integers(8, n - g - 8))
        a = Trk(tr.tid, tr.frames[:cut], tr.pos[:cut], tr.vel[:cut], tr.std[:cut], tr.det[:cut], tr.t[:cut])
        b = Trk(next_id, tr.frames[cut + g:], tr.pos[cut + g:], tr.vel[cut + g:], tr.std[cut + g:],
                tr.det[cut + g:], tr.t[cut + g:])
        if a.n_det < 4 or b.n_det < 4:
            continue
        new_tracks.remove(tr)
        new_tracks += [a, b]
        frag_pairs[(a.tid, b.tid)] = g / fps
        next_id += 1
    merged, links, _ = relink(new_tracks)
    linkset = {(i, j) for i, j, _, _ in links}
    linked_from = {i: j for i, j, _, _ in links}
    linked_to = {j: i for i, j, _, _ in links}
    for (ia, ib), g in frag_pairs.items():
        if (ia, ib) in linkset:
            rows.append((g, 'correct'))
        elif ia in linked_from or ib in linked_to:
            rows.append((g, 'wrong'))
        else:
            rows.append((g, 'miss'))
    return rows, merged


def analyse_dataset(dataset, n_val, rng):
    data = L.load_tracks(dataset)
    trajs = data['trajectories']
    stats = dict(n_traj=0, tracks_before=0, tracks_after=0, dup=0, links=0, amb=0,
                 len_before=[], len_after=[], gaps=[], per_sub={})
    val_rows = []
    t0 = time.time()
    names = list(trajs)
    for k, name in enumerate(names):
        entry = trajs[name]
        tracks = tracks_from_entry(entry)
        if not tracks:
            continue
        tracks, n_dup = merge_duplicates(tracks)
        before = [tr.n_det for tr in tracks]
        merged, links, n_amb = relink(tracks)
        after = [tr.n_det for tr in merged]
        stats['n_traj'] += 1
        stats['tracks_before'] += len(entry['tracks'])
        stats['tracks_after'] += len(merged)
        stats['dup'] += n_dup
        stats['links'] += len(links)
        stats['amb'] += n_amb
        stats['len_before'] += before
        stats['len_after'] += after
        stats['gaps'] += [g for _, _, g, _ in links]
        sub = stats['per_sub'].setdefault(entry['subdataset'], dict(before=0, after=0, links=0))
        sub['before'] += len(tracks); sub['after'] += len(merged); sub['links'] += len(links)
        if k < n_val:
            rows, _ = synthetic_validation(tracks, rng)
            val_rows += rows
    lb, la = np.array(stats['len_before']), np.array(stats['len_after'])
    print(f'== {dataset}: {stats["n_traj"]} trajectories ({time.time() - t0:.0f}s)')
    print(f'   tracks {stats["tracks_before"]} -> {stats["tracks_after"]}  '
          f'({stats["dup"]} duplicate merges, {stats["links"]} links, {stats["amb"]} rejected as ambiguous)')
    print(f'   detected frames per track: median {np.median(lb):.0f} -> {np.median(la):.0f}, '
          f'mean {lb.mean():.1f} -> {la.mean():.1f}, p90 {np.percentile(lb, 90):.0f} -> {np.percentile(la, 90):.0f}, '
          f'tracks >= 10 s: {(lb >= 40).mean():.1%} -> {(la >= 40).mean():.1%}')
    g = np.array(stats['gaps'])
    if len(g):
        print(f'   link gaps: median {np.median(g):.2f} s, ' +
              ', '.join(f'<={s}s {np.mean(g <= s):.0%}' for s in (1, 2, 3, 4)))
    for sub, s in sorted(stats['per_sub'].items(), key=lambda kv: -kv[1]['before'])[:8]:
        print(f'   {sub:28s} tracks {s["before"]:6d} -> {s["after"]:6d}  ({s["links"]} links)')
    if val_rows:
        print(f'   synthetic validation ({len(val_rows)} cut tracks):')
        for gs in sorted({r[0] for r in val_rows}):
            rr = [r[1] for r in val_rows if r[0] == gs]
            print(f'      gap {gs:.0f}s n={len(rr):4d}  correct {rr.count("correct") / len(rr):.0%}  '
                  f'wrong {rr.count("wrong") / len(rr):.0%}  miss {rr.count("miss") / len(rr):.0%}')
        rr = [r[1] for r in val_rows]
        print(f'      overall     n={len(rr):4d}  correct {rr.count("correct") / len(rr):.0%}  '
              f'wrong {rr.count("wrong") / len(rr):.0%}  miss {rr.count("miss") / len(rr):.0%}')
    return stats, val_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', action='append', choices=list(L.DATASETS))
    ap.add_argument('--n-val', type=int, default=150, help='trajectories used for synthetic validation')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for ds in (args.dataset or list(L.DATASETS)):
        analyse_dataset(ds, args.n_val, rng)


if __name__ == '__main__':
    main()
