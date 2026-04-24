#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan


def write_params(args):
    params = f"""
slam_toolbox:
  ros__parameters:
    use_sim_time: {str(args.use_sim_time).lower()}
    mode: mapping

    odom_frame: {args.odom_frame}
    map_frame: {args.map_frame}
    base_frame: {args.base_frame}
    scan_topic: {args.scan_topic}

    resolution: {args.resolution}
    max_laser_range: {args.max_laser_range}

    map_update_interval: 0.5
    use_scan_matching: true
    use_scan_barycenter: true
    minimum_travel_distance: 0.05
    minimum_travel_heading: 0.05

    do_loop_closing: true
    transform_timeout: 0.5
    tf_buffer_duration: 30.0
    stack_size_to_use: 40000000
"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
    tmp.write(params)
    tmp.close()
    return tmp.name


def run_cmd(cmd, name):
    print(f"\n[starting {name}]")
    print(" ".join(cmd))
    return subprocess.Popen(cmd, preexec_fn=os.setsid)


class MatplotlibMapViewer(Node):
    def __init__(self, args):
        super().__init__("matplotlib_slam_viewer")

        self.args = args

        self.map_msg = None
        self.odom_msg = None
        self.scan_msg = None

        self.lock = threading.Lock()

        self.create_subscription(
            OccupancyGrid,
            args.map_topic,
            self.map_callback,
            10
        )

        self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_callback,
            10
        )

        self.create_subscription(
            LaserScan,
            args.scan_topic,
            self.scan_callback,
            10
        )

        self.get_logger().info(f"Viewing map topic: {args.map_topic}")
        self.get_logger().info(f"Viewing odom topic: {args.odom_topic}")
        self.get_logger().info(f"Viewing scan topic: {args.scan_topic}")

    def map_callback(self, msg):
        with self.lock:
            self.map_msg = msg

    def odom_callback(self, msg):
        with self.lock:
            self.odom_msg = msg

    def scan_callback(self, msg):
        with self.lock:
            self.scan_msg = msg

    def get_data(self):
        with self.lock:
            return self.map_msg, self.odom_msg, self.scan_msg


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


def draw_loop(viewer):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    while rclpy.ok():
        map_msg, odom_msg, scan_msg = viewer.get_data()

        ax.clear()
        ax.set_title("TurtleBot4 SLAM Map - Matplotlib")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")

        if map_msg is not None:
            width = map_msg.info.width
            height = map_msg.info.height
            res = map_msg.info.resolution
            ox = map_msg.info.origin.position.x
            oy = map_msg.info.origin.position.y

            data = np.array(map_msg.data, dtype=np.int16).reshape((height, width))

            display = np.zeros_like(data, dtype=np.float32)
            display[data == -1] = 0.5      # unknown gray
            display[data == 0] = 1.0       # free white
            display[data > 50] = 0.0       # occupied black

            extent = [
                ox,
                ox + width * res,
                oy,
                oy + height * res
            ]

            ax.imshow(
                display,
                cmap="gray",
                origin="lower",
                extent=extent,
                vmin=0.0,
                vmax=1.0
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Waiting for /map...",
                transform=ax.transAxes,
                ha="center",
                va="center"
            )

        if odom_msg is not None:
            px = odom_msg.pose.pose.position.x
            py = odom_msg.pose.pose.position.y
            yaw = quat_to_yaw(odom_msg.pose.pose.orientation)

            ax.plot(px, py, "ro", markersize=6)
            ax.arrow(
                px,
                py,
                0.35 * np.cos(yaw),
                0.35 * np.sin(yaw),
                head_width=0.12,
                head_length=0.15,
                color="red"
            )

        if scan_msg is not None and odom_msg is not None and viewer.args.show_scan:
            px = odom_msg.pose.pose.position.x
            py = odom_msg.pose.pose.position.y
            yaw = quat_to_yaw(odom_msg.pose.pose.orientation)

            xs = []
            ys = []

            angle = scan_msg.angle_min
            for r in scan_msg.ranges:
                if np.isfinite(r) and scan_msg.range_min < r < scan_msg.range_max:
                    lx = r * np.cos(angle)
                    ly = r * np.sin(angle)

                    wx = px + np.cos(yaw) * lx - np.sin(yaw) * ly
                    wy = py + np.sin(yaw) * lx + np.cos(yaw) * ly

                    xs.append(wx)
                    ys.append(wy)

                angle += scan_msg.angle_increment

            ax.scatter(xs, ys, s=1)

        plt.pause(viewer.args.plot_period)

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--odom-topic", default="/odom")

    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--map-frame", default="map")

    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--max-laser-range", type=float, default=12.0)

    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--show-scan", action="store_true")
    parser.add_argument("--plot-period", type=float, default=0.25)

    parser.add_argument("--save-map", default="")

    args = parser.parse_args()

    params_file = write_params(args)
    procs = []

    try:
        slam_cmd = [
            "ros2", "run", "slam_toolbox", "async_slam_toolbox_node",
            "--ros-args",
            "--params-file", params_file,
        ]

        procs.append(run_cmd(slam_cmd, "slam_toolbox"))

        rclpy.init()
        viewer = MatplotlibMapViewer(args)

        spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(viewer,),
            daemon=True
        )
        spin_thread.start()

        draw_loop(viewer)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if args.save_map:
            save_path = Path(args.save_map).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\nSaving map to: {save_path}")
            subprocess.run([
                "ros2", "run", "nav2_map_server", "map_saver_cli",
                "-f", str(save_path)
            ])

        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception:
                pass

        try:
            viewer.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

        try:
            os.remove(params_file)
        except Exception:
            pass

        print("Done.")


if __name__ == "__main__":
    main()