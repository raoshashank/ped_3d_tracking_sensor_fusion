#!/usr/bin/env python3
"""Privacy blur for the blog's RGB images: every person is blurred before any overlay is drawn.

Person instance segmentation (YOLO11x-seg, class person, conf CONF, imgsz IMGSZ so that small far people
are found), union of the masks dilated by ~DIL_FRAC of the image width, replaced by a size-scaled Gaussian
blur; each detected person's box additionally gets a lighter blur as a fallback for thin mask gaps.

Masks are cached per image key in _blur_cache/<key>.npz (packed bits + boxes), so reruns are cheap and
scripts running in an env without ultralytics (e.g. the erasor2 env for figure 09) can use them as
long as the cache was filled once with the calib env:
    ~/mambaforge/envs/calib/bin/python _blur.py <image.jpg> <key>
Usage in scripts: blur(rgb_uint8_array, key) -> blurred array (same shape).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

CACHE = Path(__file__).resolve().parent / "_blur_cache"
MODEL = "/home/shashank/code/packages/regroup-hri/regroup/yolo11x-seg.pt"
CONF, IMGSZ = 0.1, 1920       # high res + low conf: catches people down to ~13 px tall
DIL_FRAC = 0.003          # mask dilation, fraction of the image width (~2-3 px at 848)
REL_BLUR = 0.035          # Gaussian radius = this x the person's box height (unidentifiable, not a blob)
MIN_R, MAX_R = 2.0, 9.0   # px (at 848 px width)
_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL)
    return _model


def _dilate(mask, px):
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    size = 2 * px + 1
    return np.asarray(im.filter(ImageFilter.MaxFilter(size))) > 127


def detect(rgb_list):
    """Batch person segmentation -> list of (mask HxW bool, boxes Nx4 xyxy)."""
    res = _get_model().predict([im[:, :, ::-1] for im in rgb_list], imgsz=IMGSZ, conf=CONF, classes=[0],
                               retina_masks=True, verbose=False, half=True)
    out = []
    for im, r in zip(rgb_list, res):
        H, W = im.shape[:2]
        m = np.zeros((H, W), bool)
        if r.masks is not None:
            mm = r.masks.data.cpu().numpy() > 0.5
            if mm.shape[1:] != (H, W):     # safety: resize to the image
                mm = np.stack([np.asarray(Image.fromarray(x.astype(np.uint8) * 255).resize((W, H))) > 127 for x in mm])
            m = mm.any(0)
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        out.append((m, boxes))
    return out


def _save(key, mask, boxes):
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE / f"{key}.npz", bits=np.packbits(mask), shape=np.array(mask.shape), boxes=boxes)


def _load(key):
    f = CACHE / f"{key}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    shape = tuple(z["shape"])
    return np.unpackbits(z["bits"])[:shape[0] * shape[1]].reshape(shape).astype(bool), z["boxes"]


def precompute(items, batch=8):
    """items: list of (key, rgb array). Fills the cache for keys not cached yet."""
    todo = [(k, im) for k, im in items if not (CACHE / f"{k}.npz").exists()]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for (k, _), (m, b) in zip(chunk, detect([im for _, im in chunk])):
            _save(k, m, b)
    return len(todo)


def blur(rgb, key):
    """Blur every person in rgb (HxWx3 uint8). Uses the cache, runs the model if needed."""
    got = _load(key)
    if got is None:
        precompute([(key, rgb)])
        got = _load(key)
    mask, boxes = got
    H, W = rgb.shape[:2]
    pil = Image.fromarray(rgb)
    m = _dilate(mask, max(1, int(round(DIL_FRAC * W))))
    out = rgb.copy()
    # just enough to make people unidentifiable: blur radius scales with the person's size
    # (REL_BLUR of the box height), so near faces are unrecognisable and far people stay person-shaped
    for x0, y0, x1, y1 in sorted(boxes, key=lambda b: b[3] - b[1]):
        h = y1 - y0
        r = float(np.clip(REL_BLUR * h, MIN_R * W / 848, MAX_R * W / 848))
        pad = int(3 * r) + 2
        x0, y0 = max(int(x0) - pad, 0), max(int(y0) - pad, 0)
        x1, y1 = min(int(np.ceil(x1)) + pad, W), min(int(np.ceil(y1)) + pad, H)
        crop = np.asarray(pil.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(r)))
        mm = m[y0:y1, x0:x1]
        out[y0:y1, x0:x1][mm] = crop[mm]
    rest = m & (out == rgb).all(-1)             # mask pixels not covered by any box (rare)
    if rest.any():
        out[rest] = np.asarray(pil.filter(ImageFilter.GaussianBlur(MIN_R * 2)))[rest]
    return out


if __name__ == "__main__":
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
    precompute([(sys.argv[2], img)])
    print("cached", sys.argv[2])
