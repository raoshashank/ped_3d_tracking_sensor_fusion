#!/usr/bin/env python3
"""Figure 01, animated: the full-rate camera stream of the demo recording (~15 Hz, decoded from the
bag's /camera/camera/color/image_raw/compressed with its header stamps) with
  - tracker boxes + ids, linearly interpolated in time between the ~4 Hz detection frames (a box is
    drawn only when the same id is present in the detection frames on both sides of the image time),
  - the lidar scan nearest in time (10 Hz; slam_output/erasor2/data/Merged/velodyne, rslidar frame,
    scan times from poses_rslidar_tum.txt) projected into the image, coloured by range (turbo, 0-25 m).
Every person is blurred first (_blur.py, YOLO11x-seg) for privacy.
Masks are left out (4 Hz only; they would lag). Light stabilisation of the gait shake is applied
after drawing (ffmpeg vidstab), so overlays move with the image.
Writes 01_inputs_camera.mp4 (1.5x real time) and 01_inputs_camera.gif.
Run with $P3D_CONDA_ENVS/calib/bin/python (needs rosbags + ped3d_lib)."""
import io
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from _blur import blur, precompute
from _camera_common import FIG, traj
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))  # repo root: p3d_env
import p3d_env  # noqa: E402  (P3D_* data roots: config/paths.env, see config/paths.env.example)

BAG = Path(p3d_env.P("P3D_BAGS_GO2NUS")) / "utown_1_2025-03-07_12-41-18_10_corrected_merged"
SO = BAG / "slam_output"
SCALE = 1.5                       # 848x480 -> 1272x720
SPEED = 1.5
STABILISE = True
CMAP = matplotlib.colormaps["turbo"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TMP = Path(os.environ.get("P3D_SCRATCH", tempfile.gettempdir())) / "anim01"
MAX_GAP = 0.6                     # s: don't interpolate a box across a longer detection gap

G = {}


def load_images():
    out = []
    with AnyReader([BAG], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        c = [c for c in r.connections if c.topic == "/camera/camera/color/image_raw/compressed"]
        for conn, _, raw in r.messages(connections=c):
            m = r.deserialize(raw, conn.msgtype)
            out.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, bytes(m.data)))
    out.sort(key=lambda x: x[0])
    return out


def boxes_at(t, det_t, det):
    """Interpolated boxes {id: (x, y, w, h)} at time t from detection frames det_t / det[k] = {id: box}."""
    j = int(np.searchsorted(det_t, t))
    for jj in (j - 1, j):
        if 0 <= jj < len(det_t) and abs(det_t[jj] - t) < 1e-3:
            return det[jj]
    if j == 0 or j >= len(det_t) or det_t[j] - det_t[j - 1] > MAX_GAP:
        return {}
    a, b = det[j - 1], det[j]
    w = (t - det_t[j - 1]) / (det_t[j] - det_t[j - 1])
    return {i: (1 - w) * a[i] + w * b[i] for i in a if i in b}


def init():
    T = traj()
    G["cam_T_lidar"] = T.cam_T_lidar
    G["K"], G["W"], G["H"] = T.K, T.W, T.H
    G["det_t"] = np.asarray(T.timestamps[:T.n], float)
    det = []
    for k in range(T.n):
        bx, ids = T.boxes(k)
        det.append({int(i): np.asarray(b, float) for b, i in zip(bx, ids)})
    G["det"] = det
    G["scan_t"] = np.loadtxt(SO / "poses_rslidar_tum.txt")[:, 0]
    G["t0"] = G["det_t"][0]


def lidar_uv(t):
    i = int(np.argmin(np.abs(G["scan_t"] - t)))
    p = np.fromfile(SO / "erasor2/data/Merged/velodyne" / f"{i:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    M = G["cam_T_lidar"]
    pc = p @ M[:3, :3].T + M[:3, 3]
    z = pc[:, 2]
    m = z > 0.3
    K = G["K"]
    u = K[0, 0] * pc[m, 0] / z[m] + K[0, 2]
    v = K[1, 1] * pc[m, 1] / z[m] + K[1, 2]
    rng = np.linalg.norm(p[m], axis=1)
    ok = (u >= 0) & (u < G["W"]) & (v >= 0) & (v < G["H"])
    return u[ok], v[ok], rng[ok]


def frame(args):
    n, t, jpg = args
    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
    im = Image.fromarray(blur(rgb, f"bag_{t:.3f}")).convert("RGBA")   # privacy blur before overlays
    W, H = int(im.width * SCALE), int(im.height * SCALE)
    im = im.resize((W, H), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    u, v, rng = lidar_uv(t)
    cols = (CMAP(np.clip(rng / 25.0, 0, 1))[:, :3] * 255).astype(int)
    r = 2
    for x, y, c in zip(u * SCALE, v * SCALE, cols):
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(c))
    font = ImageFont.truetype(FONT, 15)
    for tid, (x, y, w, h) in sorted(boxes_at(t, G["det_t"], G["det"]).items()):
        x0, y0, x1, y1 = x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE
        d.rectangle([x0, y0, x1, y1], outline=(255, 230, 0), width=2)
        tb = d.textbbox((x0, y0), str(tid), font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([x0, y0 - th - 6, x0 + tw + 6, y0], fill=(255, 230, 0))
        d.text((x0 + 3, y0 - th - 5), str(tid), fill=(0, 0, 0), font=font)
    out = TMP / f"f{n:05d}.png"
    im.convert("RGB").save(out)
    return out


def legend_png(path):
    bw = 220
    im = Image.new("RGBA", (bw + 24, 56), (0, 0, 0, 170))
    d = ImageDraw.Draw(im)
    for i in range(bw):
        c = (np.array(CMAP(i / (bw - 1))[:3]) * 255).astype(int)
        d.line([12 + i, 26, 12 + i, 38], fill=tuple(c))
    small = ImageFont.truetype(FONT, 13)
    d.text((12, 4), "lidar range", fill=(255, 255, 255), font=small)
    d.text((12, 40), "0 m", fill=(255, 255, 255), font=small)
    d.text((12 + bw - 34, 40), "25 m", fill=(255, 255, 255), font=small)
    im.save(path)


def ff(*args):
    subprocess.run(["/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    init()
    imgs = [(t, j) for t, j in load_images() if G["det_t"][0] - 0.05 <= t <= G["det_t"][-1] + 0.05]
    fps_src = 1.0 / np.median(np.diff([t for t, _ in imgs]))
    n_new = precompute([(f"bag_{t:.3f}", np.asarray(Image.open(io.BytesIO(j)).convert("RGB"))) for t, j in imgs])
    print(f"person masks: {n_new} computed, {len(imgs) - n_new} cached")
    with Pool(8, initializer=init) as pool:
        pool.map(frame, [(n, t, j) for n, (t, j) in enumerate(imgs)], chunksize=8)
    fps = round(fps_src * SPEED)
    raw = TMP / "raw.mp4"
    ff("-framerate", str(fps), "-i", str(TMP / "f%05d.png"), "-c:v", "libx264", "-crf", "14",
       "-pix_fmt", "yuv420p", str(raw))
    vf = "null"
    if STABILISE:
        trf = TMP / "transforms.trf"
        ff("-i", str(raw), "-vf", f"vidstabdetect=shakiness=6:accuracy=15:result={trf}", "-f", "null", "-")
        vf = f"vidstabtransform=input={trf}:smoothing=12:optzoom=1,unsharp=5:5:0.4:3:3:0.0"
    mp4 = FIG / "01_inputs_camera.mp4"
    leg = TMP / "legend.png"
    legend_png(leg)
    t_first = imgs[0][0] - G["t0"]
    T = f"({t_first:.3f}+n/{fps_src:.4f})"
    # timestamp (recording time = frame index / source rate) + legend, drawn after stabilisation
    txt = (f"drawtext=fontfile={FONT}:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8:"
           f"x=16:y=h-th-20:text='t = %{{eif\\:floor({T})\\:d}}.%{{eif\\:floor({T}*10)-floor({T})*10\\:d}} s'")
    lavfi = (f"[0:v]{vf},scale=1024:-2:flags=lanczos,{txt}[v];"
             f"[v][1:v]overlay=W-w-14:H-h-14")
    ff("-i", str(raw), "-i", str(leg), "-filter_complex", lavfi, "-c:v", "libx264", "-crf", "30",
       "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4))
    gif = FIG / "01_inputs_camera.gif"
    pal = TMP / "pal.png"
    gvf = "fps=10,scale=448:-1:flags=lanczos"
    ff("-i", str(mp4), "-vf", f"{gvf},palettegen=max_colors=48:stats_mode=diff", str(pal))
    ff("-i", str(mp4), "-i", str(pal), "-lavfi", f"{gvf}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle",
       "-loop", "0", str(gif))
    print(f"{len(imgs)} frames at {fps_src:.1f} Hz -> mp4 {fps} fps ({SPEED}x), "
          f"mp4 {mp4.stat().st_size / 1e6:.1f} MB, gif {gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
