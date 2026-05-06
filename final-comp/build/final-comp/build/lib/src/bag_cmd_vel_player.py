#!/usr/bin/env python3

import argparse
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message, deserialize_message

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan

import rosbag2_py


TOPIC_TYPE = "geometry_msgs/msg/TwistStamped"


class CmdVelBagTool(Node):
    def __init__(
        self,
        mode,
        bag_path,
        namespace,
        rate_scale,
        center_before_run,
        lidar_angle_offset_deg=0.0,
        open_threshold_m=1.0,
    ):
        super().__init__("cmd_vel_bag_tool")

        namespace = namespace.rstrip("/")
        if namespace == "":
            self.cmd_topic = "/cmd_vel"
            self.scan_topic = "/scan"
        else:
            self.cmd_topic = namespace + "/cmd_vel"
            self.scan_topic = namespace + "/scan"

        self.mode = mode
        self.bag_path = bag_path
        self.rate_scale = rate_scale
        self.center_before_run = center_before_run
        self.lidar_angle_offset_deg = float(lidar_angle_offset_deg)
        self.open_threshold_m = float(open_threshold_m)

        self.latest_scan = None

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)

        self.get_logger().info(f"Scan topic: {self.scan_topic}")
        self.get_logger().info(f"Cmd topic:  {self.cmd_topic}")

        if self.mode == "center":
            if not self.center_robot():
                raise RuntimeError("Centering failed.")
            self.get_logger().info("Center-only mode complete.")
        elif self.mode == "record":
            self.start_recording()
        elif self.mode == "follow":
            self.start_following()
        else:
            raise ValueError("mode must be center, record, or follow")

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

            # Allows replaying a bag recorded under /tb4_4 onto /tb4_5, etc.
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
        self.stop_robot()

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
            rclpy.spin_once(self, timeout_sec=5.0)
            if time.time() - start > timeout_sec:
                self.get_logger().warn("Timed out waiting for LaserScan.")
                return False

        return self.latest_scan is not None

    def refresh_scan(self, samples=3, timeout_sec=0.25):
        """
        Spin a few times so latest_scan is not stale after a move.
        """
        if not self.wait_for_scan(timeout_sec=timeout_sec):
            return False

        for _ in range(samples):
            rclpy.spin_once(self, timeout_sec=timeout_sec)
            time.sleep(0.02)

        return self.latest_scan is not None

    def stop_robot(self, repeats=5):
        for _ in range(repeats):
            self.pub.publish(self.make_cmd(0.0, 0.0))
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

    def angle_diff(self, a, b):
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    def scan_index_to_body_angle_deg(self, scan, index):
        """
        Convert scan index to robot-body angle, using:
        0 deg = front, +90 deg = left, -90 deg = right.
        """
        raw_angle_rad = scan.angle_min + index * scan.angle_increment
        body_angle_deg = math.degrees(raw_angle_rad) - self.lidar_angle_offset_deg
        return (body_angle_deg + 180.0) % 360.0 - 180.0

    def body_angle_deg_to_scan_angle_rad(self, body_angle_deg):
        return math.radians(body_angle_deg + self.lidar_angle_offset_deg)

    def get_sector_ranges(self, scan, center_deg, window_deg=8.0):
        center_rad = self.body_angle_deg_to_scan_angle_rad(center_deg)
        window_rad = math.radians(window_deg)

        values = []

        for i, r in enumerate(scan.ranges):
            raw_angle = scan.angle_min + i * scan.angle_increment

            if abs(self.angle_diff(raw_angle, center_rad)) <= window_rad:
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    values.append(float(r))
                elif math.isinf(r):
                    values.append(float(scan.range_max))

        return values

    def percentile(self, values, p):
        if not values:
            return None

        vals = sorted(values)
        idx = int(round((len(vals) - 1) * p))
        idx = max(0, min(idx, len(vals) - 1))
        return vals[idx]

    def get_wall_distance(self, scan, angle_deg, window_deg=12.0):
        values = self.get_sector_ranges(scan, angle_deg, window_deg)
        if len(values) < 3:
            return None
        return self.percentile(values, 0.20)

    def get_open_distance(self, scan, angle_deg, window_deg=10.0):
        values = self.get_sector_ranges(scan, angle_deg, window_deg)
        if len(values) < 3:
            return None
        return self.percentile(values, 0.80)

    def is_open_reading(self, scan, r):
        if math.isinf(r):
            return True
        if not math.isfinite(r):
            return False
        if r <= scan.range_min:
            return False
        return r >= min(self.open_threshold_m, scan.range_max * 0.95)

    def circular_mean_deg(self, angles_deg):
        """
        Mean direction for angles that may wrap around +/-180.
        """
        if not angles_deg:
            return 0.0

        sx = 0.0
        sy = 0.0
        for a in angles_deg:
            ar = math.radians(a)
            sx += math.cos(ar)
            sy += math.sin(ar)

        return (math.degrees(math.atan2(sy, sx)) + 180.0) % 360.0 - 180.0

    def find_largest_opening_center_angle(self, scan):
        """
        Find the center of the largest contiguous open lidar arc.

        This is still cheap: it is O(N) over one LaserScan. With typical lidar
        sizes, it should take milliseconds, not seconds.
        """
        n = len(scan.ranges)
        if n == 0:
            return None, 0.0, 0

        open_flags = [self.is_open_reading(scan, r) for r in scan.ranges]

        if not any(open_flags):
            return None, 0.0, 0

        if all(open_flags):
            # Completely open around the robot, so preserve current heading.
            return 0.0, 360.0, n

        doubled = open_flags + open_flags
        best_start = 0
        best_len = 0
        cur_start = None
        cur_len = 0

        for i, flag in enumerate(doubled):
            if flag:
                if cur_start is None:
                    cur_start = i
                    cur_len = 1
                else:
                    cur_len += 1

                # Do not allow a segment longer than one full scan.
                if cur_len > n:
                    cur_start += 1
                    cur_len = n

                if cur_len > best_len and cur_start < n:
                    best_start = cur_start
                    best_len = cur_len
            else:
                cur_start = None
                cur_len = 0

        best_indices = [(best_start + k) % n for k in range(best_len)]
        angles = [self.scan_index_to_body_angle_deg(scan, i) for i in best_indices]
        center_angle = self.circular_mean_deg(angles)
        arc_width_deg = best_len * abs(math.degrees(scan.angle_increment))

        return center_angle, arc_width_deg, best_len

    def find_fallback_open_angle(self, scan):
        """
        Fallback if there is no clear contiguous opening above open_threshold_m.
        Uses a smoothed openness score every 5 degrees.
        """
        best_angle = 0.0
        best_score = -1.0

        for angle_deg in range(-180, 181, 5):
            score = self.get_open_distance(scan, angle_deg, window_deg=10.0)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_angle = float(angle_deg)

        return best_angle, best_score

    def find_repeatable_heading(self, scan):
        """
        Prefer the center of the largest open arc. Fall back to the farthest
        smoothed direction if no arc crosses the threshold.
        """
        arc_angle, arc_width, arc_count = self.find_largest_opening_center_angle(scan)

        if arc_angle is not None:
            self.get_logger().info(
                f"Largest open arc center={arc_angle:.1f} deg, "
                f"width={arc_width:.1f} deg, points={arc_count}"
            )
            return arc_angle

        far_angle, far_score = self.find_fallback_open_angle(scan)
        self.get_logger().info(
            f"Fallback farthest heading={far_angle:.1f} deg, score={far_score:.2f} m"
        )
        return far_angle

    def rotate_by_angle(self, angle_deg, angular_speed=0.35):
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
        self.refresh_scan(samples=2)

    def drive_distance(self, distance_m, linear_speed=0.08, safety_dist=0.28):
        if abs(distance_m) < 0.015:
            self.stop_robot()
            return

        sign = 1.0 if distance_m > 0.0 else -1.0
        speed = sign * abs(linear_speed)
        duration = abs(distance_m) / abs(linear_speed)
        travel_angle = 0.0 if sign > 0.0 else 180.0

        self.get_logger().info(f"Driving {distance_m:.3f} m for {duration:.2f} sec")

        start = time.time()
        while rclpy.ok() and time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.latest_scan is not None:
                obstacle_dist = self.get_wall_distance(
                    self.latest_scan,
                    travel_angle,
                    window_deg=18.0,
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
        self.refresh_scan(samples=2)

    def move_sideways_by_rotation(self, lateral_offset_m):
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

        Steps:
        1. Face the center of the largest contiguous open arc.
        2. Center left/right if both side walls are visible.
        3. Center front/back if both front and back walls are visible.
        4. Repeat centering a few times.
        5. Face the center of the largest contiguous open arc again.
        """
        if not self.refresh_scan(samples=5, timeout_sec=5.0):
            self.get_logger().warn("No scan available for centering.")
            return False

        self.get_logger().info("Starting lidar-based centering routine.")

        max_single_correction_m = 0.35
        min_correction_m = 0.05
        side_wall_window_deg = 30.0
        front_back_window_deg = 30.0
        max_wall_for_centering_m = 1.0

        scan = self.latest_scan
        initial_heading = self.find_repeatable_heading(scan)
        self.rotate_by_angle(initial_heading)

        for iteration in range(3):
            if not self.refresh_scan(samples=3, timeout_sec=0.3):
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

            if (
                left is not None
                and right is not None
                and left < max_wall_for_centering_m
                and right < max_wall_for_centering_m
            ):
                lateral_offset = 0.5 * (left - right)
                lateral_offset = max(
                    -max_single_correction_m,
                    min(max_single_correction_m, lateral_offset),
                )

                if abs(lateral_offset) > min_correction_m:
                    self.get_logger().info(
                        f"Applying lateral correction: {lateral_offset:.3f} m "
                        "(positive means left)"
                    )
                    self.move_sideways_by_rotation(lateral_offset)
                    did_move = True

            if not self.refresh_scan(samples=3, timeout_sec=0.3):
                return False

            scan = self.latest_scan
            front = self.get_wall_distance(scan, 0.0, front_back_window_deg)
            back = self.get_wall_distance(scan, 180.0, front_back_window_deg)

            if (
                front is not None
                and back is not None
                and front < max_wall_for_centering_m
                and back < max_wall_for_centering_m
            ):
                forward_offset = 0.5 * (front - back)
                forward_offset = max(
                    -max_single_correction_m,
                    min(max_single_correction_m, forward_offset),
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

        if not self.refresh_scan(samples=5, timeout_sec=0.5):
            return False

        scan = self.latest_scan
        final_heading = self.find_repeatable_heading(scan)
        self.rotate_by_angle(final_heading)

        self.stop_robot()
        self.get_logger().info("Centering routine complete.")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["center", "record", "follow"],
        help=(
            "center only, record teleop cmd_vel into a bag, "
            "or follow a recorded bag"
        ),
    )
    parser.add_argument(
        "--bag",
        default="cmd_vel_teleop_bag",
        help="Path/name of the ros2 bag folder",
    )
    parser.add_argument(
        "--namespace",
        default="/tb4_6",
        help="Robot namespace. Example: /tb4_4. Use empty string for no namespace.",
    )
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 2.0 is twice as fast, 0.5 is half speed.",
    )
    parser.add_argument(
        "--center",
        action="store_false",
        default=True,
        help="Run lidar-based centering before record/follow.",
    )
    parser.add_argument(
        "--lidar-offset-deg",
        type=float,
        default=-90.0,
        help=(
            "Angular offset between LaserScan frame and robot body frame. "
            "Use -90 or 90 if front/left/right appear rotated."
        ),
    )
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=1.0,
        help="Range in meters used to classify scan points as part of an opening.",
    )

    args = parser.parse_args()

    if args.rate_scale <= 0.0:
        raise ValueError("--rate-scale must be greater than 0")

    rclpy.init()
    node = None

    try:
        node = CmdVelBagTool(
            mode=args.mode,
            bag_path=args.bag,
            namespace=args.namespace,
            rate_scale=args.rate_scale,
            center_before_run=args.center,
            lidar_angle_offset_deg=args.lidar_offset_deg,
            open_threshold_m=args.open_threshold,
        )

        if args.mode in ["record", "follow", "center"]:
            rclpy.spin(node)

    except KeyboardInterrupt:
        if node is not None:
            node.publish_stop()
    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
Examples:

Center only:
ros2 run final-comp bag_cmd_vel_player center --namespace /tb4_4

Center only if the scan frame appears rotated:
ros2 run final-comp bag_cmd_vel_player center --namespace /tb4_4 --lidar-offset-deg -90

Record with centering first:
ros2 run final-comp bag_cmd_vel_player record --bag my_teleop_run --namespace /tb4_4 --center

Follow with centering first:
ros2 run final-comp bag_cmd_vel_player follow --bag my_teleop_run --namespace /tb4_4 --center

Faster playback:
ros2 run final-comp bag_cmd_vel_player follow --bag my_teleop_run --namespace /tb4_4 --rate-scale 1.5
"""
