"""ros2_leg_detector on the Go2 bag replay.

/rslidar_points (16-ring RoboSense, ~0.48 m above ground) is sliced into a planar /scan at
0.15-0.45 m above ground (base_link z = -0.20..0.10; base_footprint is 0.35 m below base_link),
i.e. shin/knee height, with the 0.33 deg angular resolution the shipped forest was trained on.

Run (workspace $P3D_LEG_DETECTOR_WS sourced), then:  ros2 bag play <bag_dir> --clock
"""
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

FOREST = get_package_share_directory("leg_detector") + "/config/trained_leg_detector_res=0.33.yaml"
FIXED_FRAME = "odom"   # provided by the bag's /tf (odom -> base_link)
SIM = {"use_sim_time": True}


def generate_launch_description():
    return LaunchDescription([
        # The bag's /tf_static is recorded without transient-local QoS, so republish what we need.
        Node(package="tf2_ros", executable="static_transform_publisher", name="base_to_rslidar",
             arguments=["--x", "0.14543", "--z", "0.13312", "--frame-id", "base_link", "--child-frame-id", "rslidar"],
             parameters=[SIM]),
        Node(package="tf2_ros", executable="static_transform_publisher", name="base_to_footprint",
             arguments=["--z", "-0.35", "--frame-id", "base_link", "--child-frame-id", "base_footprint"],
             parameters=[SIM]),

        Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
             name="pointcloud_to_laserscan",
             remappings=[("cloud_in", "/rslidar_points"), ("scan", "/scan")],
             parameters=[SIM, {
                 "target_frame": "base_link",
                 "transform_tolerance": 0.05,
                 "min_height": -0.20, "max_height": 0.10,
                 "angle_min": -math.pi, "angle_max": math.pi,
                 "angle_increment": math.radians(0.33),
                 "scan_time": 0.1,
                 "range_min": 0.6,          # drop returns from the robot's own body/legs
                 "range_max": 12.0,
                 "use_inf": True,
             }]),

        Node(package="leg_detector", executable="detect_leg_clusters", name="detect_leg_clusters",
             parameters=[SIM, {"forest_file": FOREST, "scan_topic": "/scan", "fixed_frame": FIXED_FRAME,
                               "max_detect_distance": 10.0,
                               "use_scan_header_stamp_for_tfs": True}]),
        Node(package="leg_detector", executable="joint_leg_tracker.py", name="joint_leg_tracker",
             parameters=[SIM, {"scan_topic": "/scan", "fixed_frame": FIXED_FRAME, "scan_frequency": 10.0}]),
        Node(package="leg_detector", executable="local_occupancy_grid_mapping", name="local_occupancy_grid_mapping",
             parameters=[SIM, {"scan_topic": "/scan", "fixed_frame": FIXED_FRAME, "base_frame": "base_link",
                               "use_scan_header_stamp_for_tfs": True}]),
    ])
