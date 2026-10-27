import argparse
import pickle as pkl
from pathlib import Path

import cv2

from process_videos import (
    _make_ped_tracks, _run_group_tracker, merge_fwd_bwd,
    _draw_ped_tracks, _draw_groups,
)


def _load_track_log(pkl_path):
    """
    Load a track log pkl and return a list of (bbox_tlwh, track_ids) per frame.

    Applies the coordinate fix (x *= 2, y *= 2) for pkl files saved with the
    off-by-2x bug.
    """
    with open(pkl_path, "rb") as f:
        raw = pkl.load(f)

    track_log = []
    for entry in raw:
        _, (bbox_tlwh, track_ids) = entry
        fixed = []
        for box in bbox_tlwh:
            x, y, w, h = map(int, box)
            fixed.append([x * 2, y * 2, w, h])
        track_log.append((fixed, [int(t) for t in track_ids]))
    return track_log


def plot(args):
    track_log = _load_track_log(args.tracks)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Group tracking (forward + backward) ───────────────────────────────────
    ped_seq = [_make_ped_tracks(b, t) for b, t in track_log]

    print("Running group tracker (forward)...")
    fwd = _run_group_tracker(ped_seq)

    print("Running group tracker (backward)...")
    bwd = list(reversed(_run_group_tracker(list(reversed(ped_seq)))))

    print("Merging...")
    merged = merge_fwd_bwd(fwd, bwd, ped_seq,
                           min_jaccard=args.min_jaccard,
                           min_duration=args.min_duration)

    # ── Write annotated video ─────────────────────────────────────────────────
    suffix   = Path(args.input).suffix.lower()
    out_path = args.output or str(Path(args.input).with_suffix("")) + "_plotted" + suffix
    codec    = "XVID" if suffix == ".avi" else "avc1"
    out      = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

    print("Writing video...")
    for frame_idx in range(min(total, len(track_log))):
        ret, frame = cap.read()
        if not ret:
            break
        bbox_tlwh, track_ids          = track_log[frame_idx]
        _, g_boxes, g_ids = merged[frame_idx]
        _draw_ped_tracks(frame, bbox_tlwh, track_ids)
        _draw_groups(frame, g_boxes, g_ids)
        out.write(frame)
        print(f"  frame {frame_idx + 1}/{total}", end="\r", flush=True)

    print()
    cap.release()
    out.release()
    print(f"saved → {out_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Replay a saved track log, run fwd/bwd group detection, "
                    "and write an annotated video."
    )
    parser.add_argument("input",  help="Input video (.mp4 or .avi).")
    parser.add_argument("tracks", help="Track log .pkl file.")
    parser.add_argument("--output",       default=None)
    parser.add_argument("--min_jaccard",  type=float, default=0.5)
    parser.add_argument("--min_duration", type=int,   default=3)
    return parser.parse_args()


if __name__ == "__main__":
    plot(get_args())
