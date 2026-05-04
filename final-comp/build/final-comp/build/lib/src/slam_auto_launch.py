#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

import tf2_ros
from tf_transformations import euler_from_quaternion


class SlamAutoViewer(Node):
    def __init__(self):
        super().__init__("slam_auto_viewer")

        self.declare_parameter("namespace", "/tb4_3")
        self.namespace = self.get_parameter("namespace").value.rstrip("/")

        self.map_topic = f"{self.namespace}/map"

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_map = None

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.timer = self.create_timer(0.25, self.update_plot)

        self.get_logger().info(f"Listening for map on {self.map_topic}")

    def map_callback(self, msg):
        self.latest_map = msg

    def update_plot(self):
        if self.latest_map is None:
            return

        msg = self.latest_map
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = np.array(msg.data).reshape((height, width))

        # Convert occupancy values:
        # -1 unknown, 0 free, 100 occupied
        display = np.zeros_like(data, dtype=float)
        display[data == -1] = 0.5
        display[data == 0] = 1.0
        display[data > 50] = 0.0

        extent = [
            origin_x,
            origin_x + width * resolution,
            origin_y,
            origin_y + height * resolution
        ]

        self.ax.clear()
        self.ax.imshow(
            display,
            cmap="gray",
            origin="lower",
            extent=extent
        )

        self.ax.set_title(f"SLAM Map: {self.map_topic}")
        self.ax.set_xlabel("x position, meters")
        self.ax.set_ylabel("y position, meters")
        self.ax.set_aspect("equal")

        self.draw_robot_pose()

        plt.pause(0.001)

    def draw_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                f"{self.namespace}/base_link",
                rclpy.time.Time()
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            arrow_len = 0.35
            dx = arrow_len * math.cos(yaw)
            dy = arrow_len * math.sin(yaw)

            self.ax.plot(x, y, "ro")
            self.ax.arrow(
                x, y, dx, dy,
                head_width=0.15,
                head_length=0.15,
                fc="red",
                ec="red"
            )

        except Exception:
            # TF may not be available immediately
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SlamAutoViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()