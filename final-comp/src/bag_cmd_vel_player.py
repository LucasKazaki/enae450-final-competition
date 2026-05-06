#!/usr/bin/env python3

import argparse
import os
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan

import rosbag2_py


TOPIC_TYPE = "geometry_msgs/msg/TwistStamped"


class CmdVelBagTool(Node):
    def __init__(self, mode, bag_path, namespace, rate_scale, center_before_run):
        super().__init__("cmd_vel_bag_tool")

        self.mode = mode
        self.bag_path = bag_path
        namespace = namespace.rstrip("/")
        if namespace == "":
            self.cmd_topic = "/cmd_vel"
            self.scan_topic = "/scan"
        else:
            self.cmd_topic = namespace + "/cmd_vel"
            self.scan_topic = namespace + "/scan"
        self.rate_scale = rate_scale
        self.center_before_run = center_before_run

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data
        )
        self.latest_scan = None
        self.pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)

        if self.mode == "record":
            self.start_recording()
        elif self.mode == "follow":
            self.start_following()
        else:
            raise ValueError("mode must be record or follow")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def start_recording(self):
        if self.center_before_run:
            if not self.center_robot():
                raise RuntimeError("Centering failed before recording.")

        if os.path.exists(self.bag_path):
            raise RuntimeError(f"Bag path already exists: {self.bag_path}")

        self.writer = rosbag2_py.SequentialWriter()

        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_path,
            storage_id="sqlite3",
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        self.writer.open(storage_options, converter_options)

        topic_metadata = rosbag2_py.TopicMetadata(
            id=0,
            name=self.cmd_topic,
            type=TOPIC_TYPE,
            serialization_format="cdr",
            offered_qos_profiles=[],
        )

        self.writer.create_topic(topic_metadata)

        self.sub = self.create_subscription(
            TwistStamped,
            self.cmd_topic,
            self.cmd_callback,
            10,
        )

        self.get_logger().info(f"Recording {self.cmd_topic} to bag: {self.bag_path}")
        self.get_logger().info("Drive with teleop. Press Ctrl+C to stop recording.")

    def cmd_callback(self, msg):
        timestamp_ns = self.get_clock().now().nanoseconds
        self.writer.write(
            self.cmd_topic,
            serialize_message(msg),
            timestamp_ns,
        )

    def start_following(self):
        if not os.path.exists(self.bag_path):
            raise RuntimeError(f"Bag path does not exist: {self.bag_path}")

        if self.center_before_run:
            if not self.center_robot():
                raise RuntimeError("Centering failed before following.")

        self.get_logger().info(f"Following bag: {self.bag_path}")
        self.get_logger().info(f"Publishing commands to: {self.cmd_topic}")

        self.reader = rosbag2_py.SequentialReader()

        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_path,
            storage_id="sqlite3",
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        self.reader.open(storage_options, converter_options)

        self.timer = self.create_timer(0.1, self.play_bag_once)

    def play_bag_once(self):
        self.timer.cancel()

        last_time = None

        while self.reader.has_next() and rclpy.ok():
            topic, data, timestamp = self.reader.read_next()

            if not topic.endswith("/cmd_vel"):
                continue

            if last_time is not None:
                dt = (timestamp - last_time) / 1e9
                dt = max(0.0, dt / self.rate_scale)
                time.sleep(dt)

            msg = deserialize_message(data, TwistStamped)

            # Restamp command so the robot receives a fresh message.
            msg.header.stamp = self.get_clock().now().to_msg()

            self.pub.publish(msg)
            last_time = timestamp

        self.publish_stop()
        self.get_logger().info("Finished following bag. Published stop command.")

    def publish_stop(self):
        stop = TwistStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.header.frame_id = "base_footprint"
        stop.twist.linear.x = 0.0
        stop.twist.linear.y = 0.0
        stop.twist.linear.z = 0.0
        stop.twist.angular.x = 0.0
        stop.twist.angular.y = 0.0
        stop.twist.angular.z = 0.0
        self.pub.publish(stop)

    
    def make_cmd(self, linear_x=0.0, angular_z=0.0):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_footprint"
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)
        return cmd

    def wait_for_scan(self, timeout_sec=5.0):
        start = time.time()

        while rclpy.ok() and self.latest_scan is None:
            rclpy.spin_once(self, timeout_sec=0.05)

            if time.time() - start > timeout_sec:
                self.get_logger().warn("Timed out waiting for LaserScan.")
                return False

        return True

    def stop_robot(self, repeats=5):
        for _ in range(repeats):
            self.pub.publish(self.make_cmd(0.0, 0.0))
            time.sleep(0.05)

    def angle_diff(self, a, b):
        """
        Smallest signed angular difference a - b, in radians.
        """
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    def get_sector_ranges(self, scan, center_deg, window_deg=8.0):
        """
        Returns valid lidar ranges in a body-frame sector.

        Convention used here:
        0 deg = front
        +90 deg = left
        -90 deg = right
        +/-180 deg = back

        This assumes the LaserScan angles are already in the robot/base scan frame,
        which is normal for ROS LaserScan. If your robot's scan appears rotated,
        change lidar_angle_offset_deg below.
        """
        lidar_angle_offset_deg = 0.0

        center_rad = math.radians(center_deg + lidar_angle_offset_deg)
        window_rad = math.radians(window_deg)

        values = []

        for i, r in enumerate(scan.ranges):
            raw_angle = scan.angle_min + i * scan.angle_increment

            if abs(self.angle_diff(raw_angle, center_rad)) <= window_rad:
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    values.append(float(r))
                elif math.isinf(r):
                    # Treat inf as "very open" but cap it at range_max.
                    values.append(float(scan.range_max))

        return values

    def percentile(self, values, p):
        """
        Simple percentile without numpy.
        p is 0.0 to 1.0.
        """
        if not values:
            return None

        vals = sorted(values)
        idx = int(round((len(vals) - 1) * p))
        idx = max(0, min(idx, len(vals) - 1))
        return vals[idx]

    def get_wall_distance(self, scan, angle_deg, window_deg=12.0):
        """
        Estimate nearby wall distance in a direction.

        Uses a low percentile instead of raw min to avoid one bad lidar return
        dominating the result.
        """
        values = self.get_sector_ranges(scan, angle_deg, window_deg)

        if len(values) < 3:
            return None

        return self.percentile(values, 0.20)

    def get_open_distance(self, scan, angle_deg, window_deg=10.0):
        """
        Estimate how open a direction is.

        Uses a high percentile so an open corridor scores high even if there are
        a few noisy closer readings.
        """
        values = self.get_sector_ranges(scan, angle_deg, window_deg)

        if len(values) < 3:
            return None

        return self.percentile(values, 0.80)

    def find_farthest_open_angle(self, scan):
        """
        Find the direction with the largest open-space score.
        Returns angle in degrees, using:
        0 = front, +90 = left, -90 = right.
        """
        best_angle = 0.0
        best_score = -1.0

        # Search every 5 degrees around the robot.
        for angle_deg in range(-180, 181, 5):
            score = self.get_open_distance(scan, angle_deg, window_deg=10.0)

            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_angle = float(angle_deg)

        return best_angle, best_score

    def rotate_by_angle(self, angle_deg, angular_speed=0.35):
        """
        Open-loop rotation using time.

        Positive angle = turn left.
        Negative angle = turn right.
        """
        if abs(angle_deg) < 2.0:
            self.stop_robot()
            return

        sign = 1.0 if angle_deg > 0.0 else -1.0
        duration = abs(math.radians(angle_deg)) / angular_speed

        self.get_logger().info(f"Rotating {angle_deg:.1f} deg for {duration:.2f} sec")

        start = time.time()
        while rclpy.ok() and time.time() - start < duration:
            self.pub.publish(self.make_cmd(0.0, sign * angular_speed))
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)

        self.stop_robot()
        time.sleep(0.2)

    def drive_distance(self, distance_m, linear_speed=0.08, safety_dist=0.28):
        """
        Open-loop forward/backward drive using time.

        Positive distance = forward.
        Negative distance = backward.

        While moving, it watches the lidar sector in the direction of travel and
        stops early if something is too close.
        """
        if abs(distance_m) < 0.015:
            self.stop_robot()
            return

        sign = 1.0 if distance_m > 0.0 else -1.0
        speed = sign * abs(linear_speed)
        duration = abs(distance_m) / abs(linear_speed)

        travel_angle = 0.0 if sign > 0.0 else 180.0

        self.get_logger().info(
            f"Driving {distance_m:.3f} m for {duration:.2f} sec"
        )

        start = time.time()
        while rclpy.ok() and time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.latest_scan is not None:
                obstacle_dist = self.get_wall_distance(
                    self.latest_scan,
                    travel_angle,
                    window_deg=18.0
                )

                if obstacle_dist is not None and obstacle_dist < safety_dist:
                    self.get_logger().warn(
                        f"Stopping drive early. Obstacle at {obstacle_dist:.2f} m."
                    )
                    break

            self.pub.publish(self.make_cmd(speed, 0.0))
            time.sleep(0.02)

        self.stop_robot()
        time.sleep(0.2)

    def move_sideways_by_rotation(self, lateral_offset_m):
        """
        Move sideways even though TurtleBot cannot strafe.

        Positive offset = move left.
        Negative offset = move right.

        Implementation:
        - turn 90 degrees toward the desired side
        - drive forward
        - turn back to original heading
        """
        if abs(lateral_offset_m) < 0.015:
            return

        if lateral_offset_m > 0.0:
            self.rotate_by_angle(90.0)
            self.drive_distance(abs(lateral_offset_m))
            self.rotate_by_angle(-90.0)
        else:
            self.rotate_by_angle(-90.0)
            self.drive_distance(abs(lateral_offset_m))
            self.rotate_by_angle(90.0)

    def center_robot(self):
        """
        Try to place the robot in a repeatable starting pose using only lidar.

        Strategy:
        1. Wait for scan.
        2. Face the farthest open direction.
        3. Center left/right between side walls if both side walls are visible.
        4. Center front/back if both front and back walls are visible.
        5. Repeat centering a few times because each move changes the scan.
        6. Face the farthest open direction again.

        This works best when the robot starts inside a 2-wall or 3-wall region,
        such as a maze starting box or corridor entrance.
        """
        if not self.wait_for_scan(timeout_sec=5.0):
            return False

        self.get_logger().info("Starting lidar-based centering routine.")

        # Tunables
        max_single_correction_m = 0.35
        min_correction_m = 0.025
        side_wall_window_deg = 14.0
        front_back_window_deg = 14.0
        max_wall_for_centering_m = 2.50

        # Step 1: face the farthest open direction first.
        scan = self.latest_scan
        far_angle, far_score = self.find_farthest_open_angle(scan)
        self.get_logger().info(
            f"Initial farthest open direction: {far_angle:.1f} deg, score={far_score:.2f} m"
        )
        self.rotate_by_angle(far_angle)

        # Step 2: iteratively center.
        for iteration in range(3):
            if not self.wait_for_scan(timeout_sec=2.0):
                return False

            scan = self.latest_scan

            left = self.get_wall_distance(scan, 90.0, side_wall_window_deg)
            right = self.get_wall_distance(scan, -90.0, side_wall_window_deg)
            front = self.get_wall_distance(scan, 0.0, front_back_window_deg)
            back = self.get_wall_distance(scan, 180.0, front_back_window_deg)

            self.get_logger().info(
                "Center iteration "
                f"{iteration + 1}: "
                f"front={front}, back={back}, left={left}, right={right}"
            )

            did_move = False

            # Center between left and right walls if both are visible.
            if (
                left is not None
                and right is not None
                and left < max_wall_for_centering_m
                and right < max_wall_for_centering_m
            ):
                # If left > right, robot is closer to right wall, so move left.
                lateral_offset = 0.5 * (left - right)

                lateral_offset = max(
                    -max_single_correction_m,
                    min(max_single_correction_m, lateral_offset)
                )

                if abs(lateral_offset) > min_correction_m:
                    self.get_logger().info(
                        f"Applying lateral correction: {lateral_offset:.3f} m "
                        "(positive means left)"
                    )
                    self.move_sideways_by_rotation(lateral_offset)
                    did_move = True

            # Re-scan after lateral motion before doing front/back correction.
            rclpy.spin_once(self, timeout_sec=0.1)
            scan = self.latest_scan

            front = self.get_wall_distance(scan, 0.0, front_back_window_deg)
            back = self.get_wall_distance(scan, 180.0, front_back_window_deg)

            # Center between front and back walls if both are visible.
            if (
                front is not None
                and back is not None
                and front < max_wall_for_centering_m
                and back < max_wall_for_centering_m
            ):
                # If front > back, robot is closer to back wall, so move forward.
                forward_offset = 0.5 * (front - back)

                forward_offset = max(
                    -max_single_correction_m,
                    min(max_single_correction_m, forward_offset)
                )

                if abs(forward_offset) > min_correction_m:
                    self.get_logger().info(
                        f"Applying forward/back correction: {forward_offset:.3f} m "
                        "(positive means forward)"
                    )
                    self.drive_distance(forward_offset)
                    did_move = True

            if not did_move:
                self.get_logger().info("Centering corrections are below threshold.")
                break

        # Step 3: face the farthest open direction again after centering.
        if not self.wait_for_scan(timeout_sec=2.0):
            return False

        scan = self.latest_scan
        far_angle, far_score = self.find_farthest_open_angle(scan)
        self.get_logger().info(
            f"Final farthest open direction: {far_angle:.1f} deg, score={far_score:.2f} m"
        )
        self.rotate_by_angle(far_angle)

        self.stop_robot()
        self.get_logger().info("Centering routine complete.")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["record", "follow"],
        help="record teleop cmd_vel into a bag, or follow a recorded bag",
    )
    parser.add_argument(
        "--bag",
        default="cmd_vel_teleop_bag",
        help="Path/name of the ros2 bag folder",
    )
    parser.add_argument(
        "--namespace",
        default="/tb4_4",
        help="namespace topic to record or publish",
    )
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 2.0 is twice as fast, 0.5 is half speed.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Run lidar-based centering before record/follow.",
    )

    args = parser.parse_args()

    if args.rate_scale <= 0.0:
        raise ValueError("--rate-scale must be greater than 0")

    rclpy.init()
    node = CmdVelBagTool(
        mode=args.mode,
        bag_path=args.bag,
        namespace=args.namespace,
        rate_scale=args.rate_scale,
        center_before_run=args.center,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if args.mode == "follow":
            node.publish_stop()
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

'''record mode: ros2 run final-comp bag_cmd_vel_player record \
  --bag my_teleop_run \
  --namespace /tb4_4

  follow mode: ros2 run final-comp bag_cmd_vel_player follow \
  --bag my_teleop_run \
  --namespace /tb4_4

  faster playback: ros2 run final-comp bag_cmd_vel_player follow \
  --bag my_teleop_run \
  --namespace /tb4_4 \
  --rate-scale 1.5'''