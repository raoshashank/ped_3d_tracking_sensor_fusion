#!/usr/bin/env python3
"""Record ros2_leg_detector's /people_tracked (odom frame) into the SLAM map frame as a CSV, so it
can be scored with the same ped3d-in-FOV metric as lidar_ped_detect.py.

map <- odom at time t is T_map_base_SLAM(t) * inv(T_odom_base_ODOM(t)), with both poses
interpolated at the message stamp (same as replay_with_map.py).

Run with the leg detector launch + `ros2 bag play <bag> --clock`; Ctrl-C (or the bag ending) writes
../erasor2/output_ped/leg_tracks_map.csv (track, t, x, y).
"""
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from leg_detector_msgs.msg import PersonArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation as R, Slerp

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "erasor2" / "output_ped" / "leg_tracks_map.csv"


class Recorder(Node):
    def __init__(self):
        super().__init__("record_leg_tracks", parameter_overrides=[Parameter("use_sim_time", value=True)])
        tum = np.loadtxt(HERE.parent / "poses_base_link_tum.txt")
        self.t, self.pos, self.slerp = tum[:, 0], tum[:, 1:4], Slerp(tum[:, 0], R.from_quat(tum[:, 4:8]))
        self.odom = []   # (t, x, y, z, qx, qy, qz, qw)
        self.rows = []
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.on_odom, 200)
        self.create_subscription(PersonArray, "/people_tracked", self.on_people, 50)
        self.last_msg = None
        self.done = False

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def T_map_odom(self, t):
        od = np.array(self.odom[-400:])
        if len(od) < 2 or t < od[0, 0] - 0.1:
            return None
        j = int(np.argmin(np.abs(od[:, 0] - t)))   # odom is ~200 Hz (with duplicate stamps): nearest is fine
        T_ob = np.eye(4)
        T_ob[:3, 3] = od[j, 1:4]
        T_ob[:3, :3] = R.from_quat(od[j, 4:8]).as_matrix()
        ts = float(np.clip(t, self.t[0], self.t[-1]))
        T_mb = np.eye(4)
        T_mb[:3, 3] = [np.interp(ts, self.t, self.pos[:, k]) for k in range(3)]
        T_mb[:3, :3] = self.slerp([ts]).as_matrix()[0]
        return T_mb @ np.linalg.inv(T_ob)

    def on_people(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.last_msg = time.time()   # wall clock: the sim clock freezes when the bag ends
        T = self.T_map_odom(t)
        if T is None:
            return
        for p in m.people:
            q = T @ np.array([p.pose.position.x, p.pose.position.y, 0.0, 1.0])
            self.rows.append((p.id, t, q[0], q[1]))

    def finished(self):
        """Wall-clock check (a sim-time ROS timer freezes when the bag ends)."""
        return self.last_msg is not None and time.time() - self.last_msg > 3.0

    def save(self):
        with open(OUT, "w") as f:
            f.write("track,t,x,y\n")
            f.writelines(f"{int(r[0])},{r[1]:.3f},{r[2]:.3f},{r[3]:.3f}\n" for r in self.rows)
        self.get_logger().info(f"wrote {len(self.rows)} rows to {OUT}")


def main():
    rclpy.init(args=sys.argv)
    node = Recorder()
    try:
        while rclpy.ok() and not node.finished():
            rclpy.spin_once(node, timeout_sec=0.5)
        node.get_logger().info("no /people_tracked for 3 s: bag finished")
    except BaseException as e:   # KeyboardInterrupt, ExternalShutdownException, ...
        node.get_logger().info(f"stopping: {type(e).__name__}")
    finally:
        node.save()


if __name__ == "__main__":
    main()
