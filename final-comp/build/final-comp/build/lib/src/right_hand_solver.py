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

        self.scan_topic = self.declare_parameter("scan_topic", "scan").value
        self.cmd_topic = self.declare_parameter("cmd_topic", "cmd_vel").value

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

        self.timer = self.create_timer(0.01, self.control_loop)

        self.latest_scan = None

        # Tunable parameters
        self.target_right_dist = float(0.5)   # meters from right wall
        self.too_close_right = float(0.4)
        self.front_blocked_dist = float(0.4)
        self.open_space_dist = float(1.0)

        self.forward_speed = float(0.3)
        self.turn_speed = float(0.5)


        self.get_logger().info("Right hand wall follower started.")
        self.get_logger().info(f"Subscribing to {self.scan_topic}, publishing to {self.cmd_topic}")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def get_max_range_at_angle(self, scan, angle_deg, window_deg=10):
        """
        Returns the maximum valid lidar distance near a desired angle.

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
            angle_diff = (angle - angle_rad + math.pi) % (2 * math.pi) - math.pi
   
            # if abs(angle_diff) <= window_rad:
            if abs(angle - angle_rad) <= window_rad:
                    if math.isfinite(r) and scan.range_min < r < scan.range_max:
                        values.append(r)


        if len(values) == 0:
            return float("inf")

        return max(values)

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
            angle_diff = (angle - angle_rad + math.pi) % (2 * math.pi) - math.pi
   
            # print("angle_diff:" + str(angle-angle_rad) + "; window_rad:" + str(window_rad))
            # if abs(angle_diff) <= window_rad:
            if abs(angle - angle_rad) <= window_rad:
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    values.append(r)

        if len(values) == 0:
            print("failed values: " + str(values))
            print("target angle (rad):", angle_rad)
            print("scan min/max:", scan.angle_min, scan.angle_max)
            return float("inf")

        # print(min(values))
        return min(values)

    def control_loop(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan

        front = self.get_range_at_angle(scan, -10, 10)
        print("front: " + str(front))
        right = self.get_range_at_angle(scan, -90, 60)
        print("right: " + str(right))

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_footprint"

        # Simpler right-hand rule:
        # 1. If front is clear and there is a wall on the right, move forward.
        # 2. If there is no wall on the right, turn right.
        # 3. If there is a wall on the right and a wall in front, turn left.

        front_clear = front > self.front_blocked_dist
        right_wall = right < self.open_space_dist
       
        """
        # spin-in-place testing
        # left turn is positive angular.z
        # right turn is negative angular.z
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = self.turn_speed * 0.5
        self.cmd_pub.publish(cmd)
        return
        """
       
        #check if out of maze first
        if self.get_range_at_angle(scan, 0, 70) > self.open_space_dist:
            print(self.get_range_at_angle(scan, 0, 70))
            print("\n|||***open space on all sides, likely out of maze***|||")
            print("- doing 180 and stopping")
            # do a 180 and stop
            # state = "open space on all sides, likely out of maze, doing 180 and stopping"
            cmd.twist.linear.x = 0.75
            cmd.twist.angular.z = 0.0
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = "base_footprint"
            self.cmd_pub.publish(cmd)
            time.sleep(0.5)
            print("moved forward a bit")
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = self.turn_speed
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = "base_footprint"
            self.cmd_pub.publish(cmd)
            print("spinning")
            #send spin command for self.turn_speed * math.py constantly
            t0 = self.get_clock().now().to_msg()
            while (self.get_clock().now().to_msg().sec - t0.sec) < (math.pi / self.turn_speed):
                self.cmd_pub.publish(cmd)
                time.sleep(0.1)
            # time.sleep(self.turn_speed * math.pi * 10)
            cmd.twist.angular.z = 0.0
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = "base_footprint"
            cmd.twist.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            print("180 done, stopping")
            quit()
       
        elif front_clear and right_wall:
            # Normal case: follow the right wall forward.
            print("\n***front clear and right wall exists***")
            cmd.twist.linear.x = self.forward_speed * 0.5

            # Small correction to keep a reasonable distance from the wall.
            if right < self.too_close_right:
                """
                if right < 0.22:
                    print("- right wall WAY too close, steering left")
                    cmd.twist.linear.x = self.forward_speed * 0.01
                    cmd.twist.angular.z = self.turn_speed * 0.1
                else:
                    print("- right wall too close, steering left")
                    cmd.twist.angular.z = self.turn_speed * 0.5
                """
                print("- right wall too close, steering left")
                cmd.twist.angular.z = self.turn_speed * 0.2
                   
            elif right > self.target_right_dist:
                cmd.twist.angular.z = -self.turn_speed
                print("- right wall too far, steering right")
            else:
                cmd.twist.angular.z = self.turn_speed * 0.0
                print("- following right wall")

        elif not right_wall:
            print("\n***no right wall***")
            print("- turning right")
            # print("right_wall:" + str(right_wall) + "; right:" + str(right) + "; right_proper:" + str(self.get_range_at_angle(scan, -90, 0)) + "; open_space_dist:" + str(self.open_space_dist))
            # No wall on the right, so turn right until we find one.
            cmd.twist.linear.x = 0.05
            cmd.twist.angular.z = -self.turn_speed * 0.75

        else:
            print("\n***right wall exists but front blocked***")
            # Right wall exists, but front is blocked, so turn left.
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = self.turn_speed
            print("- turning left")

        state = "ignore, look to print statements instead"
       
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"{state} | front={front:.2f}, right={right:.2f}",
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

'''ros2 run final-comp right_hand_solver --ros-args \
  -p scan_topic:=/tb4_3/scan \
  -p cmd_topic:=/tb4_3/cmd_vel'''