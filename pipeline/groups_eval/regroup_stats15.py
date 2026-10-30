import pickle, numpy as np, pandas as pd, sys, itertools, collections
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)
B=sys.argv[1]
P=f"{p3d_env.P('P3D_REGROUP')}/go2nus/{B}_0"
E=f"{p3d_env.P('P3D_BAGS_GO2NUS')}/{B}/slam_output/erasor2/output_ped"
d=pickle.load(open(f"{P}/regroup_tracks.pkl","rb")); ts=np.array([int(l.split()[1]) for l in open(f"{P}/_camera_camera_color_image_raw_compressed.txt")])/1e9
F=pd.read_csv(f"{E}/fused_tracks.csv"); M=pd.read_csv(f"{E}/fused_track_members.csv")
pm=M[M.kind=="ped3d"]; p2f={}
for r in pm.itertuples(): p2f.setdefault(int(r.member_id)%100000,[]).append((r.t_start,r.t_end,r.person))
G={p:g.sort_values("t") for p,g in F.groupby("person")}
def pos(bid,t):
    for a,b,p in p2f.get(bid,[]):
        if a-0.15<=t<=b+0.15:
            g=G[p]; i=np.argmin(abs(g.t.values-t))
            if abs(g.t.values[i]-t)<0.15: return p, g[["x","y","vx","vy"]].values[i]
    return None,None
D=[];V=[];miss=0;tot=0
for k,gs in enumerate(d['groups']):
    for g in gs:
        for a,b in itertools.combinations(g['members'],2):
            tot+=1; pa,A=pos(a,ts[k]); pb,Bv=pos(b,ts[k])
            if A is None or Bv is None or pa==pb: miss+=1; continue
            D.append(np.linalg.norm(A[:2]-Bv[:2])); V.append(np.linalg.norm(A[2:]-Bv[2:]))
D=np.array(D);V=np.array(V)
print(B, "pairs",tot,"unmapped",miss, "dist pct50/75/90", np.percentile(D,[50,75,90]).round(2), ">2m %.0f%%"%(100*(D>2).mean()), "dv>0.5 %.0f%%"%(100*(V>0.5).mean()))
# temporal flicker: per regroup group id, number of distinct member sets and number of frame-to-frame changes
seq=collections.defaultdict(list)
for k,gs in enumerate(d['groups']):
    for g in gs: seq[g['group_id']].append((k,tuple(sorted(g['members']))))
ch=sum(sum(1 for (k1,m1),(k2,m2) in zip(s,s[1:]) if m1!=m2 or k2-k1>1) for s in seq.values())
L=[len(s) for s in seq.values()]
print("  group ids",len(seq),"median life (frames)",np.median(L),"membership changes/gaps",ch)
