#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import json
import math
import os

class PremappedSolver(Node):
    def __init__(self):
        super().__init__("premapped_solver")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.path = self.load_path()
        self.index = 0

        self.current_pose = None

        # Control loop
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Premapped solver started.")


    def load_path(self):
        filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "path.json")

        if not os.path.exists(filepath):
            self.get_logger().error(f"Path file not found: {filepath}")
            return []

        with open(filepath, "r") as f:
            path = json.load(f)

        self.get_logger().info(f"Loaded path with {len(path)} points")
        return path


    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

        self.current_pose = (x, y, yaw)

    def control_loop(self):
        if self.current_pose is None or self.index >= len(self.path):
            self.stop_robot()
            return

        x, y, yaw = self.current_pose
        goal_x, goal_y = self.path[self.index]

        dx = goal_x - x
        dy = goal_y - y
        dist = math.hypot(dx, dy)

        angle_to_goal = math.atan2(dy, dx)
        angle_error = (angle_to_goal - yaw + math.pi) % (2 * math.pi) - math.pi

        cmd = Twist()

        # move to next point if close enough
        if dist < 0.1:
            self.index += 1
            return

        cmd.linear.x = 0.15
        cmd.angular.z = 1.5 * angle_error

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PremappedSolver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()