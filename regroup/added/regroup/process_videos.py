import argparse
import pickle as pkl
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from ultralytics import YOLO

from regroup import nn_matching
from regroup.group_only_tracker import GroupOnlyTracker, PedTrack


# ── Group tracker defaults ────────────────────────────────────────────────────
MAX_COSINE_DIST = 0.4
NN_BUDGET       = 100
CIF             = 4
GP_DIST         = 20
H_RATIO         = 0.8
# Motion-only association: BoT-SORT does not expose ReID crops, so appearance
# features are unavailable here. cost_metric=1 uses IoU matching only.
COST_METRIC     = 1
FEATURE_DIM     = 128   # dummy zero-vector dimension

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]
# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_ped_tracks(frame, bbox_tlwh, track_ids):
    """Draw per-person boxes, coloured by track ID."""
    for box, tid in zip(bbox_tlwh, track_ids):
        x, y, w, h = map(int, box)
        color = _PALETTE[tid % len(_PALETTE)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        label = f"{tid}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, y - th - bl - 3), (x + tw + 2, y), color, -1)
        cv2.putText(frame, label, (x + 1, y - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _draw_groups(frame, group_boxes, group_ids):
    """Draw thick group bounding boxes with 'G<id>' labels."""
    for box, gid in zip(group_boxes, group_ids):
        x, y, w, h = map(int, box)
        color = _PALETTE[gid % len(_PALETTE)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        label = f"G{gid}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - bl - 4), (x + tw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ── Group tracker helpers ─────────────────────────────────────────────────────

def _make_ped_tracks(bbox_tlwh, track_ids):
    return [
        PedTrack(ped_id=tid,
                 tlwh=np.array(box, dtype=float),
                 feature=np.zeros(FEATURE_DIM))
        for box, tid in zip(bbox_tlwh, track_ids)
    ]


def _run_group_tracker(ped_tracks_sequence):
    metric  = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = GroupOnlyTracker(metric, cost_metric=COST_METRIC,
                               cif=CIF, gp_dist=GP_DIST, h_ratio=H_RATIO,
                               max_gap_factor=1.2)
    results = []
    for ped_tracks in ped_tracks_sequence:
        tracker.predict()
        results.append(tracker.update(ped_tracks))
    return results


# ── Forward / backward merge ──────────────────────────────────────────────────

def merge_fwd_bwd(fwd_results, bwd_results, ped_tracks_sequence,
                  min_jaccard=0.5, min_duration=3):
    next_id     = [1]
    prev_groups = {}
    hit_counter = Counter()

    def _member_sets(ped_group_ids):
        buckets = defaultdict(set)
        for pid, gid in ped_group_ids.items():
            if gid != 0:
                buckets[gid].add(pid)
        return [s for s in buckets.values() if len(s) >= 2]

    def _jaccard(a, b):
        u = len(a | b)
        return len(a & b) / u if u > 0 else 0.0

    def _merge_box(ped_tlwh, members):
        boxes = np.array([ped_tlwh[p] for p in members if p in ped_tlwh])
        if len(boxes) < 2:
            return None
        x = max(1.0, float(np.min(boxes[:, 0])))
        y = max(1.0, float(np.min(boxes[:, 1])))
        w = float(np.max(boxes[:, 0] + boxes[:, 2])) - x
        h = float(np.max(boxes[:, 1] + boxes[:, 3])) - y
        return np.array([x, y, w, h])

    merged = []
    for (f_ped, _, _), (b_ped, _, _), ped_tracks in \
            zip(fwd_results, bwd_results, ped_tracks_sequence):

        fwd_sets = _member_sets(f_ped)
        bwd_sets = _member_sets(b_ped)

        confirmed, used_bwd = [], set()
        for f_set in fwd_sets:
            best_bi, best_score = -1, 0.0
            for bi, b_set in enumerate(bwd_sets):
                if bi in used_bwd:
                    continue
                score = _jaccard(f_set, b_set)
                if score > best_score:
                    best_score, best_bi = score, bi
            if best_score >= min_jaccard:
                members = frozenset(f_set & bwd_sets[best_bi])
                if len(members) >= 2:
                    confirmed.append(members)
                used_bwd.add(best_bi)

        cur_groups, used_prev = {}, set()
        for members in confirmed:
            best_id, best_score = None, 0.0
            for prev_members, prev_id in prev_groups.items():
                if prev_id in used_prev:
                    continue
                score = _jaccard(members, prev_members)
                if score > best_score:
                    best_score, best_id = score, prev_id
            if best_id is not None:
                cur_groups[members] = best_id
                used_prev.add(best_id)
            else:
                cur_groups[members] = next_id[0]
                next_id[0] += 1
        prev_groups = cur_groups

        ped_tlwh = {pt.ped_id: pt.tlwh for pt in ped_tracks}
        merged_ped_group_ids = {pt.ped_id: 0 for pt in ped_tracks}
        merged_boxes, merged_ids = [], []
        for members, gid in cur_groups.items():
            for pid in members:
                merged_ped_group_ids[pid] = gid
            box = _merge_box(ped_tlwh, members)
            if box is not None:
                merged_boxes.append(box)
                merged_ids.append(gid)
        merged.append((merged_ped_group_ids, merged_boxes, merged_ids))

    for _, _, gids in merged:
        hit_counter.update(gids)
    valid = {gid for gid, cnt in hit_counter.items() if cnt >= min_duration}

    filtered = []
    for ped_group_ids, group_boxes, group_track_ids in merged:
        filtered.append((
            {pid: (gid if gid in valid else 0) for pid, gid in ped_group_ids.items()},
            [b for b, gid in zip(group_boxes, group_track_ids) if gid in valid],
            [gid for gid in group_track_ids if gid in valid],
        ))
    return filtered


# ── Main ──────────────────────────────────────────────────────────────────────

def process_video(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    suffix    = Path(args.input).suffix.lower()
    out_video = args.output or str(Path(args.input).with_suffix("")) + "_tracked" + suffix
    out_pkl   = str(Path(out_video).with_suffix("")) + "_tracks.pkl"

    # ── Pass 1: BoT-SORT pedestrian tracking ─────────────────────────────────
    print("Pass 1/4: pedestrian tracking (BoT-SORT)...")
    model     = YOLO(args.yolo_model)
    track_log = []   # list[(bbox_tlwh, track_ids)]

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, tracker=args.tracker_config, persist=True,
                               classes=[0], device=args.device,
                               conf=args.conf, iou=args.iou, verbose=False)
        boxes = results[0].boxes
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            tlwh = boxes.xywh.cpu().numpy().copy()
            tlwh[:, 0] -= tlwh[:, 2] / 2
            tlwh[:, 1] -= tlwh[:, 3] / 2
            bbox_tlwh = tlwh.tolist()
        else:
            track_ids, bbox_tlwh = [], []
        track_log.append((bbox_tlwh, track_ids))
        print(f"  frame {frame_idx + 1}/{total}  peds={len(track_ids)}", end="\r", flush=True)

    cap.release()
    print()

    ped_seq = [_make_ped_tracks(b, t) for b, t in track_log]

    # ── Pass 2: group tracking forward ───────────────────────────────────────
    print("Pass 2/4: group tracking (forward)...")
    fwd = _run_group_tracker(ped_seq)

    # ── Pass 3: group tracking backward ──────────────────────────────────────
    print("Pass 3/4: group tracking (backward)...")
    bwd = list(reversed(_run_group_tracker(list(reversed(ped_seq)))))

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("Merging forward + backward group results...")
    merged = merge_fwd_bwd(fwd, bwd, ped_seq,
                           min_jaccard=args.min_jaccard,
                           min_duration=args.min_duration)

    # ── Pass 4: write annotated video ────────────────────────────────────────
    print("Pass 4/4: writing annotated video...")
    cap2  = cv2.VideoCapture(args.input)
    codec = "XVID" if suffix == ".avi" else "avc1"
    out   = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

    for frame_idx in range(len(track_log)):
        ret, frame = cap2.read()
        if not ret:
            break
        bbox_tlwh, track_ids          = track_log[frame_idx]
        _, g_boxes, g_ids = merged[frame_idx]
        _draw_ped_tracks(frame, bbox_tlwh, track_ids)
        _draw_groups(frame, g_boxes, g_ids)
        out.write(frame)
        print(f"  frame {frame_idx + 1}/{len(track_log)}", end="\r", flush=True)

    print()
    cap2.release()
    out.release()

    with open(out_pkl, "wb") as f:
        pkl.dump(track_log, f)

    print(f"video  → {out_video}")
    print(f"tracks → {out_pkl}")


def get_args():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="BoT-SORT pedestrian tracking + fwd/bwd group detection. "
                    "Writes an annotated video with per-person and group boxes."
    )
    parser.add_argument("input",  help="Input video (.mp4 or .avi).")
    parser.add_argument("--output", default=None,
                        help="Output video path (default: <input>_tracked<ext>).")
    parser.add_argument("--yolo_model",      default=str(here / "yolo11x.pt"))
    parser.add_argument("--tracker_config",  default=str(here / "botsort.yaml"))
    parser.add_argument("--device",          default=0)
    parser.add_argument("--conf",            type=float, default=0.25)
    parser.add_argument("--iou",             type=float, default=0.7)
    parser.add_argument("--min_jaccard",     type=float, default=0.5,
                        help="Min fwd/bwd member-set Jaccard to confirm a group.")
    parser.add_argument("--min_duration",    type=int,   default=3,
                        help="Min frames a group must appear to be kept.")
    return parser.parse_args()


if __name__ == "__main__":
    process_video(get_args())
