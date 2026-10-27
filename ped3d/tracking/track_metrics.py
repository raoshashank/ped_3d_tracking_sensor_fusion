"""Ground-truth-free tracker continuity metrics for per-frame (boxes_tlwh, ids) sequences.

  ids_per_100det : unique ids per 100 detections (lower = fewer fragments for the same people)
  frag_rate      : fraction of track starts that look like a re-acquisition: within `gap` frames of a
                   track end whose constant-velocity image prediction lands within 0.6 x box height
                   of the new box and with a similar box height (ratio in [0.6, 1.6])
  med_len / p90  : detected frames per track
Only boxes with height >= min_h are counted (excludes far/tiny detections, which differ between detectors).
"""
import numpy as np


def _tracks(seq, min_h):
    tr = {}
    for f, (boxes, ids) in enumerate(seq):
        for b, i in zip(boxes, ids):
            b = np.asarray(b, float)
            if b[3] < min_h:
                continue
            tr.setdefault(i, []).append((f, b))
    return tr


def metrics(seq, gap, min_h=40, fps=4.0):
    tr = _tracks(seq, min_h)
    if not tr:
        return None
    n_det = sum(len(v) for v in tr.values())
    starts, ends = [], []
    for i, v in tr.items():
        (f0, b0), (f1, b1) = v[0], v[-1]
        vel = np.zeros(2)
        if len(v) >= 3:
            fa, ba = v[-3]
            if f1 > fa:
                vel = ((b1[:2] + b1[2:] / 2) - (ba[:2] + ba[2:] / 2)) / (f1 - fa)
        starts.append((f0, b0[:2] + b0[2:] / 2, b0[3]))
        ends.append((f1, b1[:2] + b1[2:] / 2, b1[3], vel))
    n_re = 0
    for f0, c0, h0 in starts:
        for f1, c1, h1, vel in ends:
            d = f0 - f1
            if 0 < d <= gap and 0.6 < h0 / h1 < 1.6 and np.linalg.norm(c0 - (c1 + vel * d)) < 0.6 * (h0 + h1) / 2:
                n_re += 1
                break
    lens = np.array([len(v) for v in tr.values()])
    return dict(n_tracks=len(tr), n_det=n_det, ids_per_100det=100 * len(tr) / n_det,
                frag_rate=n_re / len(tr), n_frag=n_re, med_len_s=np.median(lens) / fps,
                p90_len_s=np.percentile(lens, 90) / fps, det_per_frame=n_det / len(seq))


def summarize(rows):
    rows = [r for r in rows if r]
    if not rows:
        return 'no data'
    n_tr = sum(r['n_tracks'] for r in rows)
    n_det = sum(r['n_det'] for r in rows)
    n_fr = sum(r['n_frag'] for r in rows)
    return (f'trajs {len(rows):3d} tracks {n_tr:6d} dets {n_det:7d} | ids/100det {100 * n_tr / n_det:5.2f} | '
            f'reacq-frag {n_fr / n_tr:5.1%} | med len {np.median([r["med_len_s"] for r in rows]):4.1f}s '
            f'p90 {np.median([r["p90_len_s"] for r in rows]):4.1f}s | det/frame {np.mean([r["det_per_frame"] for r in rows]):4.2f}')
