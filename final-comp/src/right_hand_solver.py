#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import time


class RightHandWallFollower(Node):
    def __init__(self):
        super().__init__("right_hand_wall_follower")

        # Change these if your simulator uses namespaced topics like /tb4_3/scan
        self.scan_topic = self.declare_parameter("scan_topic", "/scan").value
        self.cmd_topic = self.declare_parameter("cmd_topic", "/cmd_vel").value

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            self.cmd_topic,
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.latest_scan = None

        # Tunable parameters
        self.target_right_dist = 0.45   # meters from right wall
        self.too_close_right = 0.30
        self.front_blocked_dist = 0.55
        self.open_space_dist = 0.75

        self.forward_speed = 0.18
        self.turn_speed = 0.55

        self.get_logger().info("Right hand wall follower started.")
        self.get_logger().info(f"Subscribing to {self.scan_topic}, publishing to {self.cmd_topic}")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def get_range_at_angle(self, scan, angle_deg, window_deg=10):
        """
        Returns the minimum valid lidar distance near a desired angle.

        Assumes standard ROS LaserScan angle convention:
        0 degrees = front
        +90 degrees = left
        -90 degrees = right
        """
        angle_rad = math.radians(angle_deg - 90) #-90 degrees is front, offset
        window_rad = math.radians(window_deg)

        values = []

        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment

            if abs(angle - angle_rad) <= window_rad:
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    values.append(r)

        if len(values) == 0:
            return float("inf")

        return min(values)

    def control_loop(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan

        front = self.get_range_at_angle(scan, 0, 15)
        front_right = self.get_range_at_angle(scan, -45, 15)
        right = self.get_range_at_angle(scan, -90, 15)

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_footprint"

        # Simpler right-hand rule:
        # 1. If front is clear and there is a wall on the right, move forward.
        # 2. If there is no wall on the right, turn right.
        # 3. If there is a wall on the right and a wall in front, turn left.

        front_clear = front > self.front_blocked_dist
        right_wall = right < self.open_space_dist
        
        #check if out of maze first
        if self.get_range_at_angle(scan, -90, 90) > self.front_blocked_dist:
            #do a 180 and stop
            cmd.twist.angular.z = self.turn_speed
            time.sleep(1 / self.turn_speed * math.pi / 2)
            cmd.twist.angular.z = 0
            exit


        if front_clear and right_wall:
            # Normal case: follow the right wall forward.
            cmd.twist.linear.x = self.forward_speed

            # Small correction to keep a reasonable distance from the wall.
            if right < self.too_close_right:
                cmd.twist.angular.z = self.turn_speed * 0.35
                state = "front clear, right wall too close, steering left"
            elif right > self.target_right_dist:
                cmd.twist.angular.z = -self.turn_speed * 0.20
                state = "front clear, right wall too far, steering right"
            else:
                cmd.twist.angular.z = 0.0
                state = "front clear, following right wall"

        elif not right_wall:
            # No wall on the right, so turn right until we find one.
            cmd.twist.linear.x = 0.05
            cmd.twist.angular.z = -self.turn_speed * 0.75
            state = "no right wall, turning right"

        else:
            # Right wall exists, but front is blocked, so turn left.
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = self.turn_speed
            state = "right wall and front blocked, turning left"

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"{state} | front={front:.2f}, right={right:.2f}, front_right={front_right:.2f}",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = RightHandWallFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    stop = TwistStamped()
    stop.header.stamp = node.get_clock().now().to_msg()
    stop.header.frame_id = "base_footprint"
    node.cmd_pub.publish(stop)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

'''python3 right_hand_solver.py --ros-args \
  -p scan_topic:=/tb4_3/scan \
  -p cmd_topic:=/tb4_3/cmd_vel'''