"""Map crispness: median point-to-local-plane RMS (cm) over sampled planar neighborhoods (lower = sharper)."""
import sys, numpy as np, open3d as o3d
p = o3d.t.io.read_point_cloud(sys.argv[1]).point.positions.numpy().astype(np.float64)
rng = np.random.default_rng(0)
tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)))
res = []
for i in rng.choice(len(p), 20000, replace=False):
    _, idx, _ = tree.search_radius_vector_3d(p[i], 0.25)
    if len(idx) < 20: continue
    ev = np.linalg.eigvalsh(np.cov(p[np.asarray(idx)].T))
    if ev[1] > 5 * ev[0]:  # planar neighborhoods (ground, walls)
        res.append(np.sqrt(ev[0]))
print(f"{sys.argv[1]}: plane thickness median {100*np.median(res):.2f} cm, mean {100*np.mean(res):.2f} cm over {len(res)} planar patches")
