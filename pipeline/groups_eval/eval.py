"""Per-bag comparison REGROUP (15 Hz, image-space) vs 3D groups (ped_groups.py output dir)."""
import pickle, sys, itertools, numpy as np, pandas as pd, json
B, O = sys.argv[1], sys.argv[2]
R = f"/media/shashank/ssd2/regroup_depth/regroup_detections/go2nus/{B}_0"
E = f"/media/dataset_access/go2nus_annotated/{B}/slam_output/erasor2/output_ped"
rg = pickle.load(open(f"{R}/regroup_tracks.pkl", "rb"))
ts = np.array([int(l.split()[1]) for l in open(f"{R}/_camera_camera_color_image_raw_compressed.txt")]) / 1e9
F = pd.read_csv(f"{E}/fused_tracks.csv"); M = pd.read_csv(f"{E}/fused_track_members.csv"); M = M[M.kind == "ped3d"]
G = pd.read_csv(f"{O}/groups.csv"); S = pd.read_csv(f"{O}/group_summary.csv")
spans = {}
for r in M.itertuples(): spans.setdefault(int(r.member_id) % 100000, []).append((r.t_start - .15, r.t_end + .15, int(r.person)))
def person(b, t):
    for a, c, p in spans.get(b, []):
        if a <= t <= c: return p
scan_t = F.groupby("scan").t.first()
st, sv = scan_t.index.values, scan_t.values
FS = {s: g.set_index("person")[["x", "y", "vx", "vy"]] for s, g in F.groupby("scan")}
GS = {s: g.set_index("person").group for s, g in G.groupby("scan")}
def state(p, s):
    f = FS.get(s); return None if f is None or p not in f.index else f.loc[p].values
rows = []; rg_life = {}; rg_sets = {}
for k, t in enumerate(ts[:len(rg["groups"])]):
    i = np.argmin(abs(sv - t))
    if abs(sv[i] - t) > .1: continue
    s = st[i]; vis = {}
    for b in rg["tracks"][k][1]:
        p = person(int(b), t)
        if p is not None: vis[int(b)] = p
    yes = set()
    for g in rg["groups"][k]:
        rg_life.setdefault(g["group_id"], []).append(k)
        rg_sets.setdefault(g["group_id"], []).append(tuple(sorted(g["members"])))
        ps = {vis[m] for m in g["members"] if m in vis}
        yes |= {tuple(sorted(x)) for x in itertools.combinations(ps, 2)}
    gs = GS.get(s, pd.Series(dtype=int))
    for a, b in itertools.combinations(sorted(set(vis.values())), 2):
        A, Bb = state(a, s), state(b, s)
        if A is None or Bb is None: continue
        ours = a in gs.index and b in gs.index and gs[a] == gs[b]
        rows.append((int((a, b) in yes), int(ours), np.hypot(*(A[:2] - Bb[:2])), np.hypot(*(A[2:] - Bb[2:]))))
P = pd.DataFrame(rows, columns=["rg", "ours", "d", "dv"])
def changes(sets):  # membership changes per group-minute
    return sum(a != b for a, b in zip(sets, sets[1:]))
rg_ch = sum(changes(v) for v in rg_sets.values()); rg_min = sum(len(v) for v in rg_life.values()) / 15 / 60
og = {g: [frozenset(x.person) for _, x in d.groupby("scan")] for g, d in G.groupby("group")}
o_ch = sum(changes(v) for v in og.values()); o_min = G.groupby("group").scan.nunique().sum() / 10 / 60
rgp, op = P[P.rg == 1], P[P.ours == 1]
sizes = S.size_max if len(S) else pd.Series(dtype=int)
out = dict(bag=B, people=F.person.nunique(), dur_s=round(F.t.max() - F.t.min(), 1),
    rg_groups=len(rg_life), rg_life_med_s=round(np.median([len(v) for v in rg_life.values()]) / 15, 2) if rg_life else np.nan,
    rg_ch_per_min=round(rg_ch / max(rg_min, 1e-9), 1) if rg_life else np.nan,
    rg_pairs=len(rgp), rg_far2m=round((rgp.d > 2).mean(), 3) if len(rgp) else np.nan, rg_dv05=round((rgp.dv > .5).mean(), 3) if len(rgp) else np.nan,
    groups=len(S), size2=int((sizes == 2).sum()), size3=int((sizes == 3).sum()), size4=int((sizes == 4).sum()), size5p=int((sizes >= 5).sum()),
    dur_med_s=round((S.t_end - S.t_start).median(), 1) if len(S) else np.nan,
    grouped_frac=round(len(G) / len(F), 3), ch_per_min=round(o_ch / max(o_min, 1e-9), 1) if len(S) else np.nan,
    ours_pairs=len(op), ours_far2m=round((op.d > 2).mean(), 3) if len(op) else np.nan, ours_dv05=round((op.dv > .5).mean(), 3) if len(op) else np.nan,
    ours_in_rg=round(op.rg.mean(), 3) if len(op) else np.nan,                     # of our camera pairs, REGROUP also grouped
    rg_in_ours=round(rgp.ours.mean(), 3) if len(rgp) else np.nan,                 # of REGROUP pairs, we also grouped
    rg_plaus_in_ours=round(rgp[(rgp.d < 1.5) & (rgp.dv < .4)].ours.mean(), 3) if len(rgp[(rgp.d < 1.5) & (rgp.dv < .4)]) else np.nan)
print(json.dumps(out))
