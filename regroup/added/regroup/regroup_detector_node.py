#!/usr/bin/env python3
"""
ROS2 node: regroup_detector

Subscribes to an image topic (sensor_msgs/Image or sensor_msgs/CompressedImage),
runs the REGROUP group-detection pipeline, and publishes the annotated frame
as sensor_msgs/Image on ~/image_annotated.

Parameters
----------
image_topic      : str   -- base topic (e.g. /camera/image_raw).
                           If it ends with /compressed, CompressedImage is used directly.
                           Otherwise the node subscribes to both <topic> and
                           <topic>/compressed and uses whichever arrives first.
yolo_model       : str   -- YOLO weights file (default: yolo11x.pt)
reid_weights     : str   -- OSNet weights file (default: osnet_x0_25_msmt17.pt)
device           : str   -- torch device string, e.g. "cuda:0" or "cpu" (default: cuda:0)
min_confidence   : float -- YOLO detection threshold (default: 0.5)
nms_max_overlap  : float -- NMS IoU threshold (default: 0.5)
min_det_height   : int   -- discard detections shorter than this (default: 0)
max_cosine_dist  : float -- re-ID gallery distance threshold (default: 0.4)
nn_budget        : int   -- max gallery size per track (default: 100)
cif              : int   -- crowd-indication-feature threshold (default: 4)
gp_dist          : float -- ground-plane distance threshold in px (default: 20.0)
h_ratio          : float -- height-ratio threshold [0,1] (default: 0.8)
cost_metric      : int   -- 0=appearance, 1=motion, 2=both (default: 2)
"""

import os
import sys
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

# Make the regroup package importable when running the node from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class RegroupDetectorNode(Node):

    def __init__(self):
        super().__init__("regroup_detector")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("image_topic", "")
        self.declare_parameter("yolo_model", "yolo11x.pt")
        self.declare_parameter("reid_weights", "osnet_x0_25_msmt17.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("nms_max_overlap", 0.5)
        self.declare_parameter("min_det_height", 0)
        self.declare_parameter("max_cosine_dist", 0.4)
        self.declare_parameter("nn_budget", 100)
        self.declare_parameter("cif", 4)
        self.declare_parameter("gp_dist", 20.0)
        self.declare_parameter("h_ratio", 0.8)
        self.declare_parameter("cost_metric", 2)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        if not image_topic:
            self.get_logger().fatal("image_topic parameter is required")
            raise RuntimeError("image_topic parameter is required")

        self._min_confidence = self.get_parameter("min_confidence").get_parameter_value().double_value
        self._nms_max_overlap = self.get_parameter("nms_max_overlap").get_parameter_value().double_value
        self._min_det_height = self.get_parameter("min_det_height").get_parameter_value().integer_value
        self._device = self.get_parameter("device").get_parameter_value().string_value

        # ── Load models (done once, before subscriptions start) ───────────────
        self._load_models()
        self._init_regroup()

        # ── Pub / sub setup ───────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, "~/image_annotated", 10)

        # Frame queue: worker thread drains it; maxsize=2 keeps latency low
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)

        # Which message type is active ('image' | 'compressed' | None)
        self._active_type: str | None = None
        self._type_lock = threading.Lock()

        if image_topic.endswith("/compressed"):
            self._sub_compressed = self.create_subscription(
                CompressedImage, image_topic, self._compressed_cb, 10
            )
            self.get_logger().info(f"Subscribed (CompressedImage): {image_topic}")
        else:
            self._sub_image = self.create_subscription(
                Image, image_topic, self._image_cb, 10
            )
            self._sub_compressed = self.create_subscription(
                CompressedImage, image_topic + "/compressed", self._compressed_cb, 10
            )
            self.get_logger().info(
                f"Subscribed (Image): {image_topic}  |  "
                f"(CompressedImage): {image_topic}/compressed  —  "
                "will use whichever arrives first"
            )

        # ── Worker thread ──────────────────────────────────────────────────────
        np.random.seed(42)
        self._colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

        self.get_logger().info("regroup_detector ready.")

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self):
        from ultralytics import YOLO
        from boxmot.reid.core.reid import ReID

        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        reid_weights = self.get_parameter("reid_weights").get_parameter_value().string_value

        self.get_logger().info(f"Loading YOLO: {yolo_model}")
        self._yolo = YOLO(yolo_model)

        self.get_logger().info(f"Loading ReID: {reid_weights}")
        self._reid = ReID(path=Path(reid_weights), device=self._device, half=False)

    def _init_regroup(self):
        from regroup import nn_matching
        from regroup.groupTracker import GroupTracker

        max_cosine_dist = self.get_parameter("max_cosine_dist").get_parameter_value().double_value
        nn_budget = self.get_parameter("nn_budget").get_parameter_value().integer_value
        cost_metric = self.get_parameter("cost_metric").get_parameter_value().integer_value
        cif = self.get_parameter("cif").get_parameter_value().integer_value
        gp_dist = self.get_parameter("gp_dist").get_parameter_value().double_value
        h_ratio = self.get_parameter("h_ratio").get_parameter_value().double_value

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_dist, nn_budget)
        self._group_tracker = GroupTracker(metric, cost_metric)
        self._group_tracker.setparams(cif, gp_dist, h_ratio)

    # ── Subscription callbacks ─────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        with self._type_lock:
            if self._active_type is None:
                self._active_type = "image"
                self.get_logger().info("Using sensor_msgs/Image")
            elif self._active_type != "image":
                return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"imgmsg_to_cv2 failed: {e}")
            return
        self._enqueue(frame, msg.header)

    def _compressed_cb(self, msg: CompressedImage):
        with self._type_lock:
            if self._active_type is None:
                self._active_type = "compressed"
                self.get_logger().info("Using sensor_msgs/CompressedImage")
            elif self._active_type != "compressed":
                return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("cv2.imdecode returned None — skipping frame")
            return
        self._enqueue(frame, msg.header)

    def _enqueue(self, frame: np.ndarray, header):
        """Non-blocking enqueue; drops the oldest frame if queue is full."""
        try:
            self._frame_q.put_nowait((frame, header))
        except queue.Full:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_q.put_nowait((frame, header))
            except queue.Full:
                pass

    # ── Worker thread ──────────────────────────────────────────────────────────

    def _process_loop(self):
        from regroup.detection import Detection
        from application_util import preprocessing

        while rclpy.ok():
            try:
                frame, header = self._frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # 1. Detect persons
                tlwh_boxes, confs = self._detect(frame)

                # 2. Re-ID features
                features = self._encode(frame, tlwh_boxes)

                # 3. Build Detection objects
                dets = [
                    Detection(box, conf, feat)
                    for box, conf, feat in zip(tlwh_boxes, confs, features)
                    if box[3] >= self._min_det_height
                ]

                # 4. NMS
                if dets:
                    boxes_arr = np.array([d.tlwh for d in dets])
                    scores_arr = np.array([d.confidence for d in dets])
                    keep = preprocessing.non_max_suppression(
                        boxes_arr, self._nms_max_overlap, scores_arr
                    )
                    dets = [dets[i] for i in keep]

                # 5. REGROUP: track + group
                self._group_tracker.predict_groups()
                _, group_boxes, group_ids, _ = self._group_tracker.update_groups(dets)

                # 6. Annotate
                annotated = self._draw(frame, group_boxes, group_ids)

                # 7. Publish
                out_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out_msg.header = header
                self._pub.publish(out_msg)

            except Exception as exc:
                self.get_logger().error(f"Processing error: {exc}", throttle_duration_sec=2.0)

    # ── Pipeline helpers ───────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray):
        results = self._yolo(
            frame, classes=[0], conf=self._min_confidence,
            verbose=False, device=self._device
        )[0]
        if len(results.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        tlwh = np.column_stack([
            xyxy[:, 0], xyxy[:, 1],
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1],
        ]).astype(np.float32)
        return tlwh, confs

    def _encode(self, frame: np.ndarray, tlwh_boxes: np.ndarray):
        if len(tlwh_boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)
        h_img, w_img = frame.shape[:2]
        crops = []
        for x, y, w, h in tlwh_boxes:
            x1 = max(0, int(x));      y1 = max(0, int(y))
            x2 = min(w_img, int(x+w)); y2 = min(h_img, int(y+h))
            crops.append(frame[y1:y2, x1:x2])
        return self._reid(crops)   # (N, feat_dim) float32, L2-normalised

    def _draw(self, frame: np.ndarray, group_boxes, group_ids) -> np.ndarray:
        out = frame.copy()
        for t in self._group_tracker.pedTracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            x, y, w, h = t.to_tlwh().astype(int)
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 100, 0), 1)
        for box, gid in zip(group_boxes, group_ids):
            if gid == 0:
                continue
            x, y, w, h = box.astype(int)
            color = self._colors[int(gid) % len(self._colors)].tolist()
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                out, f"G{gid}", (x, max(y - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
        return out


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RegroupDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[regroup_detector] Fatal: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
