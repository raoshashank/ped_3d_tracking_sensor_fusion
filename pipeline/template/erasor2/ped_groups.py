#!/usr/bin/env python3
"""Pedestrian groups from the fused 3D tracks, with REGROUP's image-space groups as weak evidence.

REGROUP groups people from their image boxes (horizontal gap < box width, feet within 20 px, similar
box heights), so people at different depths get grouped and memberships flicker frame to frame.
Here grouping is done on the fused tracks (output_ped/fused_tracks.csv: 10 Hz, SLAM frame, 360 deg):

Cues (pedestrians in a group: walk in the same direction at similar speeds, or stand together facing
each other; stay closer to each other, for longer, than to anyone else; groups of 2..4):

1. Pair evidence per scan, both persons within MAX_D:
     both walking (speed > V_MOVE): s = f(d) * exp(-dheading^2 / 2 SH^2) * exp(-dspeed_rel^2 / 2 SS^2)
     both standing:                 s = S_STATIC * f(d)    (facing is not used, so standing pairs need
                                                             more time to link)
     one walking, one standing:     s = S_MIXED            (walking past / up to someone)
   with f(d) = exp(-max(d - D0, 0)^2 / 2 SD^2).
   Relative closeness (soft, like the size cap): with r = max(rank of b among a's neighbours, rank of a
   among b's), odds x NN_PEN[min(r - K_NN, len(NN_PEN) - 1)] for r > K_NN (4th nearest x0.6, 5th+ x0.3).
   REGROUP (15 Hz camera frames, both persons visible): votes +1 grouped / -1 not grouped, averaged over
   the camera frames nearest each scan to v in [-1, 1]; odds x RG_YES^max(v,0) x RG_NO^max(-v,0).
2. Per pair a two-state HMM (apart / together), sticky transitions (P_STAY per scan), Viterbi; runs
   shorter than MIN_LINK_S dropped. Pair affinity w = together scans / scans the pair was within MAX_D.
3. Per scan, greedy agglomeration of the 'together' graph: repeatedly merge the two clusters with the
   highest mean cross affinity (w if together else 0), only if >= LINK_FRAC of the cross pairs are
   together. Soft size cap: a result with n > MAX_GROUP persons needs a linked fraction of
   LINK_FRAC + SIZE_FRAC * (n - MAX_GROUP) (5 -> 0.75, 6 -> 1.0, 7+ impossible) and its merge priority
   is scaled by SIZE_DECAY^(n - MAX_GROUP).
4. Group ids carried over scans by membership Jaccard >= J_MIN; groups shorter than MIN_GROUP_S dropped.

Outputs (output_ped/):
  groups.csv                 person, scan, t, group           (only persons in a group)
  group_summary.csv          group, t_start, t_end, n_scans, size_max, members, regroup_agree
  groups_sampled.pkl         REGROUP format at the camera frames of regroup_tracks_sampled.pkl:
                             {'tracks': (unchanged), 'groups': [[{'group_id', 'members' (botsort ids),
                             'box' (tlwh union of the visible members)}]]} -- drop-in for consumers of
                             regroup_tracks_sampled.pkl
"""
import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BAG = HERE.parent.parent.name
PROC = Path("/media/shashank/T1/processed/go2nus")          # ~4 Hz sampled frames (groups_sampled.pkl output)
RGD = Path("/media/shashank/ssd2/regroup_depth/regroup_detections/go2nus")   # 15 Hz REGROUP (evidence)

MAX_D = 3.0          # m, pairs farther apart give no evidence for 'together'
D0, SD = 1.2, 0.4    # m, distance up to D0 is fully compatible with a group, then gaussian falloff
V_MOVE = 0.4         # m/s, walking vs standing
SH = np.radians(20)  # heading difference falloff (walking pairs)
SS = 0.2             # relative speed difference |va - vb| / max(va, vb) falloff
S_STATIC = 0.6       # standing pairs: no facing cue, so weaker per-scan evidence
S_MIXED = 0.05
K_NN = 3             # max group size 4 -> up to 3 partners
NN_PEN = (1.0, 0.6, 0.3)
MAX_GROUP = 4        # soft: larger groups need (near) all pairs linked
SIZE_FRAC = 0.25
SIZE_DECAY = 0.8
S_CLIP = (0.03, 0.97)
P_STAY = 0.98        # per 0.1 s scan: expected run ~5 s before a switch
P_PRIOR = 0.1        # prior of 'together' at the start of a pair
RG_YES, RG_NO = 2.0, 0.7   # REGROUP odds multipliers (both visible in the camera frame)
J_MIN = 0.5
LINK_FRAC = 0.5
MIN_LINK_S = 1.0    # 'together' runs shorter than this are dropped (passers-by chaining onto a group)
MIN_GROUP_S = 1.0
SKIP_LABELS = {"unconfirmed_in_fov"}   # likely bikes / non-pedestrians


def load(out):
    F = pd.read_csv(out / "fused_tracks.csv")
    F = F[~F.label.isin(SKIP_LABELS)]
    M = pd.read_csv(out / "fused_track_members.csv")
    return F, M[M.kind == "ped3d"]


def regroup_links(M, scan_t):
    """REGROUP evidence per (person pair, scan): mean vote in [-1, 1] over the 15 Hz camera frames
    nearest that scan (+1 grouped, -1 both visible but not grouped).
    Also returns the ~4 Hz sampled file, its frame times and the botsort->person lookup."""
    spans = {}
    for r in M.itertuples():
        spans.setdefault(int(r.member_id) % 100000, []).append((r.t_start - 0.15, r.t_end + 0.15, int(r.person)))

    def person(bid, t):
        for a, b, p in spans.get(bid, []):
            if a <= t <= b:
                return p
        return None

    votes = {}
    d = RGD / f"{BAG}_0"
    f = d / "regroup_tracks.pkl"
    if f.exists():
        rg15 = pickle.load(open(f, "rb"))
        ts15 = np.array([int(l.split()[1]) for l in open(d / "_camera_camera_color_image_raw_compressed.txt")]) / 1e9
        for k, t in enumerate(ts15[:len(rg15["groups"])]):
            s = int(np.argmin(abs(scan_t - t)))
            if abs(scan_t[s] - t) > 0.1:
                continue
            vis = {}
            for bid in rg15["tracks"][k][1]:
                p = person(int(bid), t)
                if p is not None:
                    vis[int(bid)] = p
            grouped = set()
            for g in rg15["groups"][k]:
                ps = {vis[m] for m in g["members"] if m in vis}
                grouped |= {tuple(sorted(x)) for x in itertools.combinations(ps, 2)}
            for pr in itertools.combinations(sorted(set(vis.values())), 2):
                votes.setdefault((pr, s), []).append(1 if pr in grouped else -1)
    ev = {key: float(np.mean(v)) for key, v in votes.items()}

    ds = PROC / f"{BAG}_0"
    if not (ds / "regroup_tracks_sampled.pkl").exists():
        return ev, None, None, person
    rg = pickle.load(open(ds / "regroup_tracks_sampled.pkl", "rb"))
    return ev, rg, np.loadtxt(ds / "timestamps.txt", ndmin=1), person


def viterbi(lik1):
    """Two-state Viterbi; lik1 = p(obs | together) per step with p(obs | apart) = 1 - lik1."""
    n = len(lik1)
    lt = np.log([[P_STAY, 1 - P_STAY], [1 - P_STAY, P_STAY]])
    e = np.log(np.stack([1 - lik1, lik1], 1))
    dp = np.log([1 - P_PRIOR, P_PRIOR]) + e[0]
    bp = np.zeros((n, 2), int)
    for i in range(1, n):
        c = dp[:, None] + lt
        bp[i] = c.argmax(0)
        dp = c.max(0) + e[i]
    st = np.zeros(n, int)
    st[-1] = dp.argmax()
    for i in range(n - 1, 0, -1):
        st[i - 1] = bp[i, st[i]]
    return st


def pair_score(P, V):
    """Per-scan evidence matrix for all persons of one scan (see module doc, step 1)."""
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    sp = np.linalg.norm(V, axis=1)
    hd = np.arctan2(V[:, 1], V[:, 0])
    dh = np.abs((hd[:, None] - hd[None] + np.pi) % (2 * np.pi) - np.pi)
    ds = np.abs(sp[:, None] - sp[None]) / np.maximum(np.maximum(sp[:, None], sp[None]), 1e-6)
    fd = np.exp(-np.maximum(D - D0, 0) ** 2 / (2 * SD ** 2))
    mv = sp > V_MOVE
    S = np.where(mv[:, None] & mv[None], fd * np.exp(-dh ** 2 / (2 * SH ** 2)) * np.exp(-ds ** 2 / (2 * SS ** 2)),
                 np.where(~mv[:, None] & ~mv[None], S_STATIC * fd, S_MIXED))
    rank = D.argsort(1).argsort(1)            # 0 = self
    r = np.maximum(rank, rank.T)
    pen = np.asarray(NN_PEN)[np.clip(r - K_NN, 0, len(NN_PEN) - 1)]
    return D, np.clip(S, *S_CLIP), pen


def pair_states(F, ev):
    """{(a, b): array of scans where the pair is 'together'}, {(a, b): affinity w}"""
    obs = {}   # (a, b) -> list of (scan, odds)
    for s, g in F.groupby("scan"):
        if len(g) < 2:
            continue
        D, S, pen = pair_score(g[["x", "y"]].values, g[["vx", "vy"]].values)
        ids = g.person.values
        for i, j in zip(*np.nonzero(np.triu(D < MAX_D, 1))):
            o = S[i, j] / (1 - S[i, j]) * pen[i, j]
            a, b = sorted((int(ids[i]), int(ids[j])))
            obs.setdefault((a, b), []).append((s, o))
    # pairs farther apart than MAX_D give no evidence: the HMM runs only over scans where they are close,
    # and a pair that drifts apart restarts (gap -> new segment)
    out, aff = {}, {}
    for pr, o in obs.items():
        o.sort()
        scans = np.array([x[0] for x in o])
        odds = np.array([x[1] for x in o])
        for i, s in enumerate(scans):
            v = ev.get((pr, s))
            if v is not None:
                odds[i] *= RG_YES ** max(v, 0) * RG_NO ** max(-v, 0)
        lik = np.clip(odds / (1 + odds), *S_CLIP)
        cuts = np.nonzero(np.diff(scans) > 3)[0] + 1   # > 0.3 s gap: apart or not both tracked
        tog = []
        for seg_s, seg_l in zip(np.split(scans, cuts), np.split(lik, cuts)):
            st = viterbi(seg_l)
            for run in np.split(np.arange(len(st)), np.nonzero(np.diff(st))[0] + 1):
                if st[run[0]] == 1 and len(run) * 0.1 >= MIN_LINK_S:
                    tog.append(seg_s[run])
        if tog:
            tog = np.concatenate(tog)
            out[pr] = tog
            aff[pr] = len(tog) / len(scans)
    return out, aff


def components(F, together, aff):
    """Per scan capped greedy agglomeration (module doc, step 3) -> {scan: [frozenset(persons)]}"""
    links = {}
    for pr, scans in together.items():
        for s in scans:
            links.setdefault(int(s), []).append(pr)
    comps = {}
    for s, ls in links.items():
        nodes = sorted({x for l in ls for x in l})
        ix = {p: i for i, p in enumerate(nodes)}
        W = np.zeros((len(nodes), len(nodes)))
        for a, b in ls:
            W[ix[a], ix[b]] = W[ix[b], ix[a]] = aff[(a, b)]
        cl = [[i] for i in range(len(nodes))]
        while True:
            best = None
            for u in range(len(cl)):
                for v in range(u + 1, len(cl)):
                    extra = max(len(cl[u]) + len(cl[v]) - MAX_GROUP, 0)
                    c = W[np.ix_(cl[u], cl[v])]
                    if (c > 0).mean() < LINK_FRAC + SIZE_FRAC * extra:
                        continue
                    sc = c.mean() * SIZE_DECAY ** extra
                    if best is None or sc > best[0]:
                        best = (sc, u, v)
            if best is None:
                break
            _, u, v = best
            cl[u] += cl.pop(v)
        comps[s] = [frozenset(nodes[i] for i in c) for c in cl if len(c) >= 2]
    return comps


def track_groups(comps, scan_t):
    """Carry group ids by membership Jaccard. Returns [(scan, gid, members)]."""
    rows, prev, nxt = [], [], 0
    for s in sorted(comps):
        cur = []
        used = set()
        cands = sorted(((len(m & pm) / len(m | pm), gi, ci)
                        for ci, m in enumerate(comps[s]) for gi, (gid, pm, ps) in enumerate(prev)
                        if s - ps <= 5), reverse=True)
        assign = {}
        for j, gi, ci in cands:
            if j < J_MIN or gi in used or ci in assign:
                continue
            used.add(gi)
            assign[ci] = prev[gi][0]
        for ci, m in enumerate(comps[s]):
            gid = assign.get(ci)
            if gid is None:
                gid, nxt = nxt, nxt + 1
            cur.append((gid, m, s))
            rows.append((s, gid, m))
        # keep unmatched previous groups alive for short gaps
        alive = {g[0] for g in cur}
        prev = cur + [g for g in prev if g[0] not in alive and s - g[2] <= 5]
    # drop short groups
    span = {}
    for s, gid, m in rows:
        a, b = span.get(gid, (scan_t[s], scan_t[s]))
        span[gid] = (min(a, scan_t[s]), max(b, scan_t[s]))
    keep = {g for g, (a, b) in span.items() if b - a >= MIN_GROUP_S}
    ren = {g: i for i, g in enumerate(sorted(keep))}
    return [(s, ren[g], m) for s, g, m in rows if g in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None, help="default: output_ped")
    ap.add_argument("--no-regroup", action="store_true")
    args = ap.parse_args()
    src = HERE / "output_ped"
    out = args.out_dir or src
    out.mkdir(parents=True, exist_ok=True)
    F, M = load(src)
    scan_t = F.groupby("scan").t.first().reindex(range(F.scan.max() + 1)).interpolate().bfill().ffill().values
    ev, rg, rts, person = ({}, None, None, None) if args.no_regroup else regroup_links(M, scan_t)
    together, aff = pair_states(F, ev)
    rows = track_groups(components(F, together, aff), scan_t)

    pd.DataFrame([(p, s, round(scan_t[s], 3), g) for s, g, m in rows for p in sorted(m)],
                 columns=["person", "scan", "t", "group"]).to_csv(out / "groups.csv", index=False)
    summ = {}
    for s, g, m in rows:
        e = summ.setdefault(g, dict(group=g, t_start=scan_t[s], t_end=scan_t[s], n_scans=0, size_max=0,
                                    members=set(), rg_yes=0, rg_n=0))
        e["t_end"] = scan_t[s]
        e["n_scans"] += 1
        e["size_max"] = max(e["size_max"], len(m))
        e["members"] |= m
        for pr in itertools.combinations(sorted(m), 2):
            r = ev.get((pr, s))
            if r is not None:
                e["rg_n"] += 1
                e["rg_yes"] += (r + 1) / 2
    S = pd.DataFrame(summ.values())
    if len(S):
        S["members"] = S.members.map(lambda x: " ".join(map(str, sorted(x))))
        S["regroup_agree"] = (S.rg_yes / S.rg_n.replace(0, np.nan)).round(2)
        S = S.drop(columns=["rg_yes", "rg_n"])
        S[["t_start", "t_end"]] = S[["t_start", "t_end"]].round(3)
    S.to_csv(out / "group_summary.csv", index=False)

    if rg is not None:     # REGROUP-format groups at the camera frames
        by_scan = {}
        for s, g, m in rows:
            by_scan.setdefault(s, []).append((g, m))
        groups = []
        for k, t in enumerate(rts[:len(rg["tracks"])]):
            boxes, bids = rg["tracks"][k]
            s = int(np.argmin(abs(scan_t - t)))
            fr = []
            for g, m in (by_scan.get(s, []) if abs(scan_t[s] - t) <= 0.1 else []):
                mem = [(int(b), bx) for bx, b in zip(boxes, bids) if person(int(b), t) in m]
                if len(mem) < 2:     # fewer than two members visible: not a group in this image
                    continue
                bb = np.array([bx for _, bx in mem], float)
                x0, y0 = bb[:, :2].min(0)
                x1, y1 = (bb[:, :2] + bb[:, 2:]).max(0)
                fr.append({"group_id": g, "members": [b for b, _ in mem],
                           "box": np.array([x0, y0, x1 - x0, y1 - y0])})
            groups.append(fr)
        pickle.dump({"tracks": rg["tracks"], "groups": groups}, open(out / "groups_sampled.pkl", "wb"))

    n_in = F.groupby("scan").person.nunique()
    gs = pd.read_csv(out / "groups.csv")
    print(f"{BAG}: {len(S)} groups, sizes {S.size_max.value_counts().sort_index().to_dict() if len(S) else {}}, "
          f"median duration {np.median(S.t_end - S.t_start) if len(S) else 0:.1f} s, "
          f"{len(gs) / max(n_in.sum(), 1):.0%} of person-scans in a group, "
          f"REGROUP agreement (pair-frames) {S.regroup_agree.mean() if len(S) else float('nan'):.2f}")


if __name__ == "__main__":
    main()
