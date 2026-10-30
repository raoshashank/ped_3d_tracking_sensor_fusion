import pickle, sys, numpy as np, pandas as pd, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)
B, O = sys.argv[1], sys.argv[2]; frames = [int(x) for x in sys.argv[3].split(",")]
P=f"{p3d_env.P('P3D_PROCESSED')}/go2nus/{B}_0"; E=f"{p3d_env.P('P3D_BAGS_GO2NUS')}/{B}/slam_output/erasor2/output_ped"
rg=pickle.load(open(f"{P}/regroup_tracks_sampled.pkl","rb")); ng=pickle.load(open(f"{O}/groups_sampled.pkl","rb"))
ts=np.loadtxt(f"{P}/timestamps.txt"); F=pd.read_csv(f"{E}/fused_tracks.csv"); G=pd.read_csv(f"{O}/groups.csv")
cols=plt.cm.tab10.colors
def draw(img, k, groups):
    im=img.copy()
    for bx,b in zip(*rg["tracks"][k]):
        x,y,w,h=map(int,bx); cv2.rectangle(im,(x,y),(x+w,y+h),(200,200,200),1)
    for g in groups[k]:
        c=tuple(int(255*v) for v in cols[g["group_id"]%10][::-1]); m=set(g["members"])
        for bx,b in zip(*rg["tracks"][k]):
            if b in m: x,y,w,h=map(int,bx); cv2.rectangle(im,(x,y),(x+w,y+h),c,2)
        x,y,w,h=map(int,g["box"]); cv2.rectangle(im,(x-3,y-3),(x+w+3,y+h+3),c,1); cv2.putText(im,f"G{g['group_id']}",(x,max(y-6,10)),0,0.5,c,2)
    return im[...,::-1]
fig,ax=plt.subplots(len(frames),3,figsize=(18,4.2*len(frames)))
for r,k in enumerate(frames):
    img=cv2.imread(f"{P}/imgs/{k}.jpg")
    ax[r,0].imshow(draw(img,k,rg["groups"])); ax[r,0].set_title(f"REGROUP  frame {k}  t={ts[k]-ts[0]:.1f}s")
    ax[r,1].imshow(draw(img,k,ng["groups"])); ax[r,1].set_title("3D groups (camera frame)")
    s=F.iloc[(F.t-ts[k]).abs().argsort()[:1]].scan.values[0]; f=F[F.scan==s]; g=G[G.scan==s].set_index("person").group
    for p in f.itertuples():
        gi=g.get(p.person); c=cols[gi%10] if gi is not None else "0.6"
        ax[r,2].scatter(p.x,p.y,c=[c],s=40 if gi is not None else 12); ax[r,2].arrow(p.x,p.y,p.vx*0.8,p.vy*0.8,color=c,width=0.02)
        if gi is not None: ax[r,2].text(p.x,p.y,f" G{gi}",fontsize=8)
    ax[r,2].set_aspect("equal"); ax[r,2].set_title("BEV (SLAM frame), 360 deg, arrows = 0.8 s of velocity")
    for a in ax[r,:2]: a.axis("off")
plt.tight_layout(); plt.savefig(f"{O}/compare.png",dpi=60)
