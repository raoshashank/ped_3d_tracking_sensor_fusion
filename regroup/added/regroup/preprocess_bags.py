#!/usr/bin/env python3
import argparse
from glob import glob
import os
import sys
import shutil
import cv2
import yaml
from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_idl, get_typestore, get_types_from_msg
from rosbags.image import message_to_cvimage
import numpy as np
from typing import List, Tuple
from ament_index_python import get_package_prefix  # utility for CompressedImage conversion
import pickle as pkl    
from ultralytics import YOLO

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]

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

def process_video(input_video,
                  output_video,
                  yolo_model,
                  tracker_config,
                  device,
                  conf,
                  iou):
    
    suffix    = Path(input_video).suffix.lower()
    out_video = output_video or str(Path(input_video).with_suffix("")) + "_tracked" + suffix
    out_pkl   = str(Path(out_video).with_suffix("")) + "_tracks.pkl"
    if os.path.exists(out_video) and os.path.exists(out_pkl):
        print(f"Processed video and track log already exist. Skipping processing step.")
        return
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_video}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    model     = YOLO(yolo_model)
    track_log = []   # list[(bbox_tlwh, track_ids)]

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, tracker=tracker_config, persist=True,
                               classes=[0], device=device,
                               conf=conf, iou=iou, verbose=False)
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
    
    cap2  = cv2.VideoCapture(input_video)
    codec = "XVID" if suffix == ".avi" else "avc1"
    out   = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*codec), fps, (width, height))

    for frame_idx in range(len(track_log)):
        ret, frame = cap2.read()
        if not ret:
            break
        bbox_tlwh, track_ids          = track_log[frame_idx]
        _draw_ped_tracks(frame, bbox_tlwh, track_ids)
        out.write(frame)
        print(f"  frame {frame_idx + 1}/{len(track_log)}", end="\r", flush=True)

    print()
    cap2.release()
    out.release()

    with open(out_pkl, "wb") as f:
        pkl.dump(track_log, f)

    print(f"video  → {out_video}")
    print(f"tracks → {out_pkl}")

def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw

def process_detection2d(detections):
    tlwhs = []
    obj_ids = []
    if detections is None:
        return (None,None)
    for box in detections.detections:
        x = box.bbox.center.position.x - box.bbox.size_x / 2
        y = box.bbox.center.position.y - box.bbox.size_y / 2
        w = box.bbox.size_x
        h = box.bbox.size_y
        tlwhs.append([x, y, w, h])
        obj_ids.append(box.id)
    return tlwhs,obj_ids

def nav_to_xy_yaw_vel(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """
    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    #roll, pitch, yaw = R.from_quat(()
    v = odom_msg.twist.twist.linear
    w = odom_msg.twist.twist.angular
    return [position.x, position.y],yaw, [v.x, v.y], w.z

def filter_bag(input_bag_path, output_path, topics_to_save):
    """
    Read an existing bag file (using rosbags), filter messages for the specified topics,
    and write them into a new bag file. For topics that contain images (e.g. CompressedImage),
    also extract the image to compile a video and log a timestamp file.
    
    The output video file and timestamp text file will be saved in the same directory
    as the filtered bag.
    """
    # Prepare video and log handlers for image topics.
    image_handlers = {}  # key: topic name -> dict with video_writer and txt_file
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    add_types = {}
    for idl in glob(os.path.join(get_package_prefix('rclpy'),"share","vision_msgs/msg/*.idl")):
        add_types.update(get_types_from_idl(Path(idl).read_text()))
    typestore.register(add_types)
    Detection2DArray = typestore.types['vision_msgs/msg/Detection2DArray']
    Detection2D = typestore.types['vision_msgs/msg/Detection2D']
    BoundingBox2D = typestore.types['vision_msgs/msg/BoundingBox2D']
    odometry = []
    tracks = []
    # Open the bag files with rosbags.
    # skip if exists        
    if os.path.exists(os.path.join(output_path, "odometry.pkl")) and os.path.exists(os.path.join(output_path, "tracks.pkl")):                   
        return
    
    with AnyReader([Path(input_bag_path)],default_typestore=typestore) as reader:
        # Ensure output bag directory exists; rosbags' Writer writes to a directory
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Loop through all messages in the bag.
        for connection, timestamp, rawdata in reader.messages():
            # Check if the current topic is one of the topics to be saved.
            if (connection.topic in topics_to_save):
                # Process if the message is a CompressedImage message.
                # (Adjust the check if your topic uses uncompressed sensor_msgs/Image).
                if ('Image' in connection.msgtype) and (not 'depth' in connection.topic.lower()):
                    try:
                        # Deserialize the message to a CompressedImage.
                        image_msg = reader.deserialize(rawdata, connection.msgtype)
                        # Convert to OpenCV image (BGR format) using rosbags-image.
                        cv_image = message_to_cvimage(image_msg, 'bgr8')
                    except Exception as e:
                        print(f"Error converting image message from topic '{connection.topic}' at time {timestamp}: {e}")
                        continue

                    # Initialize video writer and log file for this topic if not done already.
                    if connection.topic not in image_handlers:
                        safe_topic_name = connection.topic.replace('/', '_')
                        video_filename = os.path.join(output_path, safe_topic_name + ".avi")
                        txt_filename = os.path.join(output_path, safe_topic_name + ".txt")
                        meta_filename = os.path.join(output_path, safe_topic_name + ".yaml")
                        
                        # Initialize the video writer once we know the frame size.
                        height, width = cv_image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (width, height))
                        txt_file = open(txt_filename, 'w')
                        meta_file = open(meta_filename, 'w')
                        image_handlers[connection.topic] = {
                            'video_writer': video_writer,
                            'txt_file': txt_file,
                            'frame_count': 0,
                            'meta_file': meta_file,
                        }
                        meta_dict = {
                            'topic': connection.topic,
                            'message': connection.msgtype,
                            'frame_id': image_msg.header.frame_id,
                        }
                        yaml.dump(meta_dict, meta_file)
                        print(f"Initialized video writer for topic '{connection.topic}' at resolution {width}x{height}")
                    
                    # Write the current frame and record the timestamp.
                    handler = image_handlers[connection.topic]
                    handler['video_writer'].write(cv_image)
                    handler['txt_file'].write(f"{handler['frame_count']} {timestamp}\n")
                    handler['frame_count'] += 1
                    print(f"Saved frame {handler['frame_count']} from topic '{connection.topic}' at time {timestamp}", end='\r', flush=True)
                elif ('Odometry' in connection.msgtype):
                    # save the pose information to a .pkl file
                    pose = nav_to_xy_yaw_vel(reader.deserialize(rawdata, connection.msgtype), ang_offset=0.0)
                    odometry.append((timestamp, pose))
                elif ('Detection2DArray' in connection.msgtype):
                    # save the pose information to a .pkl file
                    detections = process_detection2d(reader.deserialize(rawdata, connection.msgtype))
                    tracks.append((timestamp, detections))
                
    # Close video writers and log files.
    for topic, handler in image_handlers.items():
        handler['video_writer'].release()
        handler['txt_file'].close()
        handler['meta_file'].close()
        print(f"Released resources for topic '{topic}'")
    
    # Save odometry and tracks to .pkl files
    with open(os.path.join(output_path, "odometry.pkl"), 'wb') as f:
        pkl.dump(odometry, f)
    with open(os.path.join(output_path, "tracks.pkl"), 'wb') as f:
        pkl.dump(tracks, f)
    print(f"Saved odometry and tracks data to .pkl files in '{output_path}'")
    print("Filtering complete.")    
    if len(image_handlers) == 0:
        print("Warning: No image topics were processed. Please check the topic names and message types.")
        return False
    
    return True
    
if __name__ == '__main__':
    here = Path(__file__).parent    
    parser = argparse.ArgumentParser(description="Filter ROS2 bag files by topic and extract images to video.")
    parser.add_argument('bag_path', type=str, help="Path to the input ROS2 bag file.")
    parser.add_argument('output_path', type=str, help="Path to the output directory for the extracted videos.")
    parser.add_argument('--img-topic', default='/image_raw/compressed', help="List of topics to save. If empty, all topics will be saved.")
    parser.add_argument('--odom-topic', default='/odom', help="List of topics to save. If empty, all topics will be saved.")
    parser.add_argument('--tracks-topic', default='/tracks_image_raw_compressed', help="tracking topic if available already (optional)")
    parser.add_argument("--yolo_model",      default=str(here / "yolo11x.pt"))
    parser.add_argument("--tracker_config",  default=str(here / "botsort.yaml"))
    parser.add_argument("--device",          default=0)
    parser.add_argument("--conf",            type=float, default=0.25)
    parser.add_argument("--iou",             type=float, default=0.7)
    args = parser.parse_args()

    print(f"Starting search from base path: {args.bag_path}")
    print(f"Output path: {args.output_path}")
    print(f"Image topic: {args.img_topic}")
    print(f"Odometry topic: {args.odom_topic}")
    print(f"Tracks topic: {args.tracks_topic}")
    
    result = filter_bag(args.bag_path, args.output_path, [args.img_topic, args.odom_topic, args.tracks_topic])
    if not result:
        print("No image topics were processed. Exiting without video generation.")
    else:    
        safe_topic_name = args.img_topic.replace('/', '_')
        process_video(
        os.path.join(args.output_path, safe_topic_name + ".avi"),
        os.path.join(args.output_path, "botsort_tracked.mp4"),
        args.yolo_model,
        args.tracker_config,
        args.device,
        args.conf,
        args.iou   
        )