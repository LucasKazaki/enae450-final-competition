#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import json
import math
import os

class PathRecorder(Node):
    def __init__(self):
        super().__init__("path_recorder")

        self.sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.path = []
        self.last_saved = None
        self.min_dist = 0.05  # only save every 5 cm

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.last_saved is None:
            self.path.append([x, y])
            self.last_saved = (x, y)
            return

        dx = x - self.last_saved[0]
        dy = y - self.last_saved[1]

        if math.hypot(dx, dy) > self.min_dist:
            self.path.append([x, y])
            self.last_saved = (x, y)

    def save_path(self):
        filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "path.json"
        )
        
        with open("path.json", "w") as f:
            json.dump(self.path, f)
        self.get_logger().info("Path saved.")
    
def main(args=None):
    rclpy.init(args=args)
    node = PathRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_path()

    node.destroy_node()
    rclpy.shutdown()


        
if __name__ == "__main__":
    main()