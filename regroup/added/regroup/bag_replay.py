#!/usr/bin/env python3
"""
bag_replay.py — offline REGROUP group detection from a ROS2 bag.

Reads Image or CompressedImage messages from a bag topic, runs the full
REGROUP pipeline, and writes an annotated MP4.

Usage (inside Docker via entrypoint):
    BAG_PATH=/data/mybag TOPIC=/camera/image_raw OUTPUT_VIDEO=/output/result.mp4 \\
        docker compose run --rm bag_replay

Or directly:
    python3 bag_replay.py --bag /data/mybag \\
                          --topic /camera/image_raw \\
                          --output /output/result.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Bag decoding helpers ───────────────────────────────────────────────────────

def _detect_storage_id(bag_path: Path) -> str:
    """Read storage format from the bag's metadata.yaml (sqlite3 or mcap)."""
    meta = bag_path / "metadata.yaml"
    if meta.exists():
        with open(meta) as f:
            data = yaml.safe_load(f)
        return (data.get("rosbag2_bagfile_information", {})
                    .get("storage_identifier", "sqlite3"))
    return "sqlite3"


def _decode_image_msg(msg) -> np.ndarray:
    """Decode sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    enc = msg.encoding.lower()
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("bgr8", "rgb8"):
        frame = raw.reshape((msg.height, msg.width, 3))
        if enc == "rgb8":
            frame = frame[..., ::-1].copy()
    elif enc in ("bgra8", "rgba8"):
        frame = raw.reshape((msg.height, msg.width, 4))[..., :3]
        if enc == "rgba8":
            frame = frame[..., ::-1].copy()
    elif enc == "mono8":
        frame = cv2.cvtColor(raw.reshape((msg.height, msg.width)),
                             cv2.COLOR_GRAY2BGR)
    elif enc == "mono16":
        raw16 = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        frame = cv2.cvtColor((raw16.reshape((msg.height, msg.width)) >> 8)
                             .astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    return frame


def _decode_compressed_msg(msg) -> np.ndarray | None:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ── REGROUP pipeline helpers (mirrors regroup_detector_node.py) ────────────────

def _detect(yolo, frame, conf_thresh, device):
    results = yolo(frame, classes=[0], conf=conf_thresh,
                   verbose=False, device=device)[0]
    if len(results.boxes) == 0:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
    xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    tlwh = np.column_stack([xyxy[:, 0], xyxy[:, 1],
                            xyxy[:, 2] - xyxy[:, 0],
                            xyxy[:, 3] - xyxy[:, 1]]).astype(np.float32)
    return tlwh, confs


def _encode(reid, frame, tlwh_boxes):
    if len(tlwh_boxes) == 0:
        return np.empty((0, 512), np.float32)
    h, w = frame.shape[:2]
    crops = []
    for x, y, bw, bh in tlwh_boxes:
        x1, y1 = max(0, int(x)),  max(0, int(y))
        x2, y2 = min(w, int(x+bw)), min(h, int(y+bh))
        crops.append(frame[y1:y2, x1:x2])
    return reid(crops)


def _draw(frame, group_tracker, group_boxes, group_ids, colors):
    out = frame.copy()
    for t in group_tracker.pedTracker.tracks:
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        x, y, w, h = t.to_tlwh().astype(int)
        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 100, 0), 1)
    for box, gid in zip(group_boxes, group_ids):
        if gid == 0:
            continue
        x, y, w, h = box.astype(int)
        color = colors[int(gid) % len(colors)].tolist()
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 3)
        cv2.putText(out, f"G{gid}", (x, max(y-8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="REGROUP offline bag processor")
    ap.add_argument("--bag",    required=True, help="ROS2 bag directory path")
    ap.add_argument("--topic",  required=True, help="Image or CompressedImage topic")
    ap.add_argument("--output", required=True, help="Output video path (.mp4)")
    ap.add_argument("--fps",    type=float, default=0.0,
                    help="Output FPS. 0 = estimate from bag timestamps (default)")
    ap.add_argument("--device",          default="cuda:0")
    ap.add_argument("--yolo-model",      default="yolo11x.pt")
    ap.add_argument("--reid-weights",    default="osnet_x0_25_msmt17.pt")
    ap.add_argument("--min-confidence",  type=float, default=0.5)
    ap.add_argument("--nms-max-overlap", type=float, default=0.5)
    ap.add_argument("--min-det-height",  type=int,   default=0)
    ap.add_argument("--max-cosine-dist", type=float, default=0.4)
    ap.add_argument("--nn-budget",       type=int,   default=100)
    ap.add_argument("--cif",             type=int,   default=4)
    ap.add_argument("--gp-dist",         type=float, default=20.0)
    ap.add_argument("--h-ratio",         type=float, default=0.8)
    ap.add_argument("--cost-metric",     type=int,   default=2)
    args = ap.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"ERROR: bag not found: {bag_path}")
        sys.exit(1)

    # ── Open bag ──────────────────────────────────────────────────────────────
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_id = _detect_storage_id(bag_path)
    print(f"Bag storage format: {storage_id}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in type_map:
        print(f"ERROR: topic '{args.topic}' not in bag.")
        print(f"Available topics: {sorted(type_map)}")
        sys.exit(1)

    msg_type_str = type_map[args.topic]
    msg_type     = get_message(msg_type_str)
    is_compressed = "CompressedImage" in msg_type_str
    print(f"Topic type: {msg_type_str}")

    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    # ── Load models ───────────────────────────────────────────────────────────
    from ultralytics import YOLO
    from boxmot.reid.core.reid import ReID

    print(f"Loading YOLO: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    print(f"Loading ReID: {args.reid_weights}")
    reid = ReID(path=Path(args.reid_weights), device=args.device, half=False)

    # ── Init REGROUP ──────────────────────────────────────────────────────────
    from regroup import nn_matching
    from regroup.detection import Detection
    from regroup.groupTracker import GroupTracker
    from application_util import preprocessing

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", args.max_cosine_dist, args.nn_budget
    )
    group_tracker = GroupTracker(metric, args.cost_metric)
    group_tracker.setparams(args.cif, args.gp_dist, args.h_ratio)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    # ── Process frames ────────────────────────────────────────────────────────
    writer      = None
    frame_count = 0
    timestamps  = []   # nanoseconds, used to estimate FPS if args.fps == 0

    print("Processing bag...")
    while reader.has_next():
        _, data, ts_ns = reader.read_next()
        msg = deserialize_message(data, msg_type)

        frame = (_decode_compressed_msg(msg) if is_compressed
                 else _decode_image_msg(msg))
        if frame is None:
            continue

        timestamps.append(ts_ns)

        # Lazy-init video writer on first frame
        if writer is None:
            H, W = frame.shape[:2]
            fps = args.fps if args.fps > 0 else 30.0   # refined after loop
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
            print(f"Output: {args.output}  ({W}x{H})")

        # 1. Detect
        tlwh, confs = _detect(yolo, frame, args.min_confidence, args.device)

        # 2. Re-ID features
        features = _encode(reid, frame, tlwh)

        # 3. Build Detection objects
        dets = [
            Detection(box, conf, feat)
            for box, conf, feat in zip(tlwh, confs, features)
            if box[3] >= args.min_det_height
        ]

        # 4. NMS
        if dets:
            keep = preprocessing.non_max_suppression(
                np.array([d.tlwh for d in dets]),
                args.nms_max_overlap,
                np.array([d.confidence for d in dets]),
            )
            dets = [dets[i] for i in keep]

        # 5. Track + group
        group_tracker.predict_groups()
        _, group_boxes, group_ids, _ = group_tracker.update_groups(dets)

        # 6. Draw + write
        writer.write(_draw(frame, group_tracker, group_boxes, group_ids, colors))
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  {frame_count} frames")

    if writer:
        # If FPS was not specified, re-encode the video with the correct FPS
        # derived from actual bag timestamps.
        if args.fps == 0 and len(timestamps) > 1:
            duration_s = (timestamps[-1] - timestamps[0]) / 1e9
            actual_fps = (len(timestamps) - 1) / duration_s
            print(f"Estimated FPS from bag timestamps: {actual_fps:.2f}")
            # cv2.VideoWriter can't change FPS after creation; re-mux with ffmpeg if available
            _remux(args.output, actual_fps)
        writer.release()

    print(f"Done. {frame_count} frames → {args.output}")


def _remux(path: str, fps: float):
    """Re-mux the MP4 with the correct FPS using ffmpeg (no re-encode)."""
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        print(f"  (ffmpeg not found; output FPS may not be exact)")
        return
    tmp = path + ".tmp.mp4"
    ret = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vf", f"fps={fps:.4f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", tmp],
        capture_output=True,
    )
    if ret.returncode == 0:
        import os
        os.replace(tmp, path)
    else:
        print("  ffmpeg re-mux failed; keeping original FPS")


if __name__ == "__main__":
    main()
