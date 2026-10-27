#!/usr/bin/env python3
"""
regroup_anonymize.py

End-to-end anonymising group-tracking pipeline on a single ROS2 bag:

  Step 1 – Extract raw RGB video + odometry from the bag.
  Step 2 – YOLO-seg + BoT-SORT:
             • raw video (from Step 1) – clean frames, used in Step 3.
             • masks.avi – each pedestrian pixel = palette colour of its track ID;
               background = (0, 0, 0).
             • botsort_tracked_tracks.pkl
  Step 3 – Forward/backward group tracking → regroup_tracks.pkl + annotated video.
"""
import argparse
import os
import pickle as pkl
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from ament_index_python import get_package_prefix
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_types_from_idl, get_typestore
from ultralytics import YOLO

from regroup import nn_matching
from regroup.group_only_tracker import GroupOnlyTracker, PedTrack

# ── Colour palette ─────────────────────────────────────────────────────────────
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

# ── Group tracker defaults ─────────────────────────────────────────────────────
MAX_COSINE_DIST = 0.4
NN_BUDGET       = 100
CIF             = 4
GP_DIST         = 20
H_RATIO         = 0.8
COST_METRIC     = 1
FEATURE_DIM     = 128


# ════════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════════

def _quat_to_yaw(x, y, z, w):
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)


def _nav_to_xy_yaw_vel(odom_msg, ang_offset=0.0):
    pos = odom_msg.pose.pose.position
    ori = odom_msg.pose.pose.orientation
    yaw = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w) + ang_offset
    v   = odom_msg.twist.twist.linear
    w   = odom_msg.twist.twist.angular
    return [pos.x, pos.y], yaw, [v.x, v.y], w.z


def _draw_ped_tracks(frame, bbox_tlwh, track_ids):
    for box, tid in zip(bbox_tlwh, track_ids):
        x, y, w, h = map(int, box)
        color = get_color(tid)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        label = f"{tid}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, y - th - bl - 3), (x + tw + 2, y), color, -1)
        cv2.putText(frame, label, (x + 1, y - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _draw_groups(frame, group_boxes, group_ids):
    for box, gid in zip(group_boxes, group_ids):
        x, y, w, h = map(int, box)
        color = get_color(gid)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        label = f"G{gid}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - bl - 4), (x + tw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ════════════════════════════════════════════════════════════════════════════════
# Step 1 – Extract raw video + odometry from bag
# ════════════════════════════════════════════════════════════════════════════════

def extract_bag(bag_path, output_path, img_topic, odom_topic):
    """
    Demultiplex a ROS2 bag into:
      <safe_topic>_raw.avi   – raw BGR video
      <safe_topic>_raw.txt   – per-frame timestamps
      <safe_topic>_raw.yaml  – topic / message-type metadata
      odometry.pkl           – list of (timestamp, (xy, yaw, vel_xy, ang_vel))

    Returns the path to the raw video, or None if no images were found.
    Skips entirely if both outputs already exist.
    """
    safe      = img_topic.replace('/', '_')
    raw_video = os.path.join(output_path, safe + "_raw.avi")
    odom_pkl  = os.path.join(output_path, "odometry.pkl")

    if os.path.exists(raw_video) and os.path.exists(odom_pkl):
        print("[Step 1] Outputs already exist – skipping.")
        return raw_video

    os.makedirs(output_path, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    add_types = {}
    for idl in glob(os.path.join(get_package_prefix('rclpy'), "share", "vision_msgs/msg/*.idl")):
        add_types.update(get_types_from_idl(Path(idl).read_text()))
    typestore.register(add_types)

    odometry     = []
    video_writer = None
    txt_file     = None
    meta_file    = None
    frame_count  = 0

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == img_topic and 'Image' in conn.msgtype and 'depth' not in conn.topic.lower():
                try:
                    img_msg  = reader.deserialize(rawdata, conn.msgtype)
                    if hasattr(img_msg, 'format') and img_msg.format != 'rgb8; jpeg compressed bgr8':
                        continue
                    cv_image = message_to_cvimage(img_msg, 'bgr8')
                except Exception as e:
                    print(f"\n  Error at frame {frame_count}: {e}")
                    continue

                if video_writer is None:
                    h, w         = cv_image.shape[:2]
                    video_writer = cv2.VideoWriter(
                        raw_video, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
                    txt_file  = open(os.path.join(output_path, safe + "_raw.txt"), 'w')
                    meta_file = open(os.path.join(output_path, safe + "_raw.yaml"), 'w')
                    yaml.dump({'topic': conn.topic, 'message': conn.msgtype,
                               'frame_id': img_msg.header.frame_id}, meta_file)
                    print(f"[Step 1] Video writer: {w}x{h} for '{conn.topic}'")

                video_writer.write(cv_image)
                txt_file.write(f"{frame_count} {timestamp}\n")
                frame_count += 1
                print(f"  frame {frame_count}", end='\r', flush=True)

            elif conn.topic == odom_topic and 'Odometry' in conn.msgtype:
                pose = _nav_to_xy_yaw_vel(reader.deserialize(rawdata, conn.msgtype))
                odometry.append((timestamp, pose))

    if video_writer is not None:
        video_writer.release()
    if txt_file is not None:
        txt_file.close()
    if meta_file is not None:
        meta_file.close()

    with open(odom_pkl, 'wb') as f:
        pkl.dump(odometry, f)

    if video_writer is None:
        print(f"\n[Step 1] Warning: no image frames found for topic '{img_topic}'.")
        return None

    print(f"\n[Step 1] raw video  → {raw_video}  ({frame_count} frames)")
    print(f"[Step 1] odometry   → {odom_pkl}  ({len(odometry)} messages)")
    return raw_video


# ════════════════════════════════════════════════════════════════════════════════
# Step 2 – YOLO-seg + BoT-SORT → ID-mask video + tracks.pkl
# ════════════════════════════════════════════════════════════════════════════════

def run_seg_tracking(raw_video, output_path, yolo_model, tracker_config, device, conf, iou,
                     save_mask=True, save_anon=True):
    """
    Run YOLO-seg + BoT-SORT on raw_video.

    Outputs:
      masks.avi                  – 3-channel video; pedestrian pixels carry the
                                   palette colour of their track ID, background = (0,0,0).
      anonymized.avi             – original frame with all pedestrian pixels blacked out,
                                   background pixels unchanged.
      botsort_tracked_tracks.pkl – list[(bbox_tlwh, track_ids)] one entry per frame.

    Returns (mask_video_path, tracks_pkl_path).
    Skips if all outputs already exist.
    """
    mask_video  = os.path.join(output_path, "masks.avi")
    anon_video  = os.path.join(output_path, "anonymized.avi")
    tracks_pkl  = os.path.join(output_path, "botsort_tracked_tracks.pkl")

    required = [tracks_pkl]
    if save_mask:
        required.append(mask_video)
    if save_anon:
        required.append(anon_video)
    if all(os.path.exists(p) for p in required):
        print("[Step 2] Outputs already exist – skipping.")
        return mask_video, tracks_pkl

    cap = cv2.VideoCapture(raw_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {raw_video}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model       = YOLO(yolo_model)
    track_log   = []
    fourcc      = cv2.VideoWriter_fourcc(*'XVID')
    mask_writer = cv2.VideoWriter(mask_video, fourcc, fps, (width, height)) if save_mask else None
    anon_writer = cv2.VideoWriter(anon_video, fourcc, fps, (width, height)) if save_anon else None

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, tracker=tracker_config, persist=True,
                              classes=[0], device=device,
                              conf=conf, iou=iou, verbose=False)
        boxes      = results[0].boxes
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
        anon_frame = frame.copy() if save_anon else None

        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            tlwh      = boxes.xywh.cpu().numpy().copy()
            tlwh[:, 0] -= tlwh[:, 2] / 2
            tlwh[:, 1] -= tlwh[:, 3] / 2
            bbox_tlwh = tlwh.tolist()

            if results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()  # (N, mh, mw)
                for mask_arr, tid in zip(masks_data, track_ids):
                    # resize to original frame size in case of resolution mismatch
                    bin_mask = cv2.resize(
                        mask_arr.astype(np.float32), (width, height)) > 0.5
                    if save_mask:
                        mask_frame[bin_mask] = get_color(tid)
                    if save_anon:
                        anon_frame[bin_mask] = 0
        else:
            track_ids, bbox_tlwh = [], []

        track_log.append((bbox_tlwh, track_ids))
        if mask_writer is not None:
            mask_writer.write(mask_frame)
        if anon_writer is not None:
            anon_writer.write(anon_frame)
        print(f"  frame {frame_idx + 1}/{total}  peds={len(track_ids)}", end="\r", flush=True)

    cap.release()
    if mask_writer is not None:
        mask_writer.release()
    if anon_writer is not None:
        anon_writer.release()
    print()

    with open(tracks_pkl, 'wb') as f:
        pkl.dump(track_log, f)

    if save_mask:
        print(f"[Step 2] mask video  → {mask_video}")
    if save_anon:
        print(f"[Step 2] anon video  → {anon_video}")
    print(f"[Step 2] tracks      → {tracks_pkl}")
    return mask_video, tracks_pkl


# ════════════════════════════════════════════════════════════════════════════════
# Step 3 – Forward/backward group tracking
# ════════════════════════════════════════════════════════════════════════════════

def _make_ped_tracks(bbox_tlwh, track_ids):
    return [
        PedTrack(ped_id=tid, tlwh=np.array(box, dtype=float), feature=np.zeros(FEATURE_DIM))
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


def _merge_fwd_bwd(fwd_results, bwd_results, ped_tracks_sequence,
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

        ped_tlwh             = {pt.ped_id: pt.tlwh for pt in ped_tracks}
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


def run_group_tracking(raw_video, output_path, tracks_pkl_path,
                       min_jaccard=0.5, min_duration=3, save_video=True,
                       anon_video=None, fwd_only=False):
    """
    Forward/backward group tracking on raw_video using pre-computed BoT-SORT tracks.

    Outputs:
      regroup_tracked.avi   – anonymized video (or raw if anon_video is None) annotated
                              with group boxes.
      regroup_tracks.pkl    – {'tracks': track_log, 'groups': group_log}

    Skips if all outputs already exist.
    """
    regroup_pkl   = os.path.join(output_path, "regroup_tracks.pkl")
    suffix        = Path(raw_video).suffix.lower()
    regroup_video = os.path.join(output_path, "regroup_tracked" + suffix)

    required = [regroup_pkl] + ([regroup_video] if save_video else [])
    if all(os.path.exists(p) for p in required):
        print("[Step 3] Outputs already exist – skipping.")
        return

    with open(tracks_pkl_path, 'rb') as f:
        track_log = pkl.load(f)

    cap = cv2.VideoCapture(raw_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {raw_video}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert len(track_log) == total, \
        f"Track log length {len(track_log)} != video frame count {total}"

    ped_seq = [_make_ped_tracks(b, t) for b, t in track_log]

    print("[Step 3] Group tracking (forward)...")
    fwd = _run_group_tracker(ped_seq)

    if not fwd_only:
        print("[Step 3] Group tracking (backward)...")
        bwd = list(reversed(_run_group_tracker(list(reversed(ped_seq)))))

        print("[Step 3] Merging fwd/bwd results...")
        merged = _merge_fwd_bwd(fwd, bwd, ped_seq,
                                min_jaccard=min_jaccard, min_duration=min_duration)
    else:
        print("[Step 3] Skipping backward group tracking.")
        merged = fwd

    video_source = anon_video if (anon_video and os.path.exists(anon_video)) else raw_video
    if save_video:
        print(f"[Step 3] Writing annotated video (base: {Path(video_source).name})...")
    cap2  = cv2.VideoCapture(video_source) if save_video else None
    codec = "XVID" if suffix == ".avi" else "avc1"
    out   = cv2.VideoWriter(regroup_video, cv2.VideoWriter_fourcc(*codec), fps, (width, height)) if save_video else None

    group_log = []
    for frame_idx in range(len(track_log)):
        bbox_tlwh, track_ids         = track_log[frame_idx]
        ped_group_ids, g_boxes, g_ids = merged[frame_idx]

        members_by_gid = defaultdict(list)
        for pid, gid in ped_group_ids.items():
            if gid != 0:
                members_by_gid[gid].append(pid)
        group_log.append([
            {"group_id": gid, "members": members_by_gid[gid], "box": box}
            for gid, box in zip(g_ids, g_boxes)
        ])

        if cap2 is not None:
            ret, frame = cap2.read()
            if not ret:
                break
            if video_source == raw_video:
                _draw_ped_tracks(frame, bbox_tlwh, track_ids)
            _draw_groups(frame, g_boxes, g_ids)
            out.write(frame)
        print(f"  frame {frame_idx + 1}/{len(track_log)}", end="\r", flush=True)

    print()
    if cap2 is not None:
        cap2.release()
    if out is not None:
        out.release()

    with open(regroup_pkl, 'wb') as f:
        pkl.dump({"tracks": track_log, "groups": group_log}, f)

    if save_video:
        print(f"[Step 3] annotated video → {regroup_video}")
    print(f"[Step 3] regroup pickle  → {regroup_pkl}")


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════

def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Anonymising group-tracking pipeline: bag → raw video + ID masks + group tracks."
    )
    parser.add_argument("bag_path",    help="Path to the input ROS2 bag directory.")
    parser.add_argument("output_path", help="Directory for all outputs.")
    parser.add_argument("--img-topic",  default="/image_raw/compressed",
                        help="Image topic in the bag.")
    parser.add_argument("--odom-topic", default="/odom",
                        help="Odometry topic in the bag.")
    parser.add_argument("--yolo_model",     default=str(here / "yolo11x-seg.pt"),
                        help="YOLO-seg model weights (.pt).")
    parser.add_argument("--tracker_config", default=str(here / "botsort.yaml"))
    parser.add_argument("--device",         default=0)
    parser.add_argument("--conf",  type=float, default=0.25)
    parser.add_argument("--iou",   type=float, default=0.7)
    parser.add_argument("--min_jaccard",  type=float, default=0.5,
                        help="Min fwd/bwd Jaccard to confirm a group.")
    parser.add_argument("--min_duration", type=int,   default=3,
                        help="Min frames a group must appear to be kept.")
    parser.add_argument("--no-mask-video",   action="store_false", dest="save_mask_video",
                        help="Skip saving masks.avi.")
    parser.add_argument("--no-anon-video",   action="store_false", dest="save_anon_video",
                        help="Skip saving anonymized.avi.")
    parser.add_argument("--no-regroup-video", action="store_false", dest="save_regroup_video",
                        help="Skip saving regroup_tracked.avi.")
    parser.add_argument("--fwd-only", action="store_true",dest="fwd_only",
                        help="Run only forward group tracking (no backward).")
    parser.set_defaults(save_mask_video=True, save_anon_video=True, save_regroup_video=True)
    args = parser.parse_args()

    print("=== Step 1: Extract bag ===")
    raw_video = extract_bag(
        args.bag_path, args.output_path, args.img_topic, args.odom_topic)
    if raw_video is None:
        print("No image frames extracted. Aborting.")
        return

    print("\n=== Step 2: YOLO-seg + BoT-SORT tracking ===")
    _, tracks_pkl = run_seg_tracking(
        raw_video, args.output_path,
        args.yolo_model, args.tracker_config,
        args.device, args.conf, args.iou,
        save_mask=args.save_mask_video,
        save_anon=args.save_anon_video,
    )

    anon_video = os.path.join(args.output_path, "anonymized.avi")

    print("\n=== Step 3: Group tracking ===")
    run_group_tracking(
        raw_video, args.output_path, tracks_pkl,
        min_jaccard=args.min_jaccard,
        min_duration=args.min_duration,
        save_video=args.save_regroup_video,
        anon_video=anon_video,
        fwd_only=args.fwd_only
    )
    # delete the raw video
    os.remove(raw_video)
    print("\nDone.")
    print(f"Outputs in: {args.output_path}")


if __name__ == "__main__":
    main()
