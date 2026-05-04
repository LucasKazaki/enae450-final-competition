#!/usr/bin/env python3

import argparse
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message

from geometry_msgs.msg import TwistStamped

import rosbag2_py


TOPIC_TYPE = "geometry_msgs/msg/TwistStamped"


class CmdVelBagTool(Node):
    def __init__(self, mode, bag_path, cmd_topic, rate_scale):
        super().__init__("cmd_vel_bag_tool")

        self.mode = mode
        self.bag_path = bag_path
        self.cmd_topic = cmd_topic
        self.rate_scale = rate_scale

        if self.mode == "record":
            self.start_recording()
        elif self.mode == "follow":
            self.start_following()
        else:
            raise ValueError("mode must be record or follow")

    def start_recording(self):
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

        self.pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)

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

            if topic != self.cmd_topic:
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
        "--cmd-topic",
        default="/tb4_5/cmd_vel",
        help="cmd_vel topic to record or publish",
    )
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 2.0 is twice as fast, 0.5 is half speed.",
    )

    args = parser.parse_args()

    rclpy.init()
    node = CmdVelBagTool(
        mode=args.mode,
        bag_path=args.bag,
        cmd_topic=args.cmd_topic,
        rate_scale=args.rate_scale,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if args.mode == "follow":
            node.publish_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

'''record mode: python3 bag_cmd_vel_player.py record \
  --bag my_teleop_run \
  --cmd-topic /tb4_5/cmd_vel
  
  follow mode: python3 bag_cmd_vel_player.py follow \
  --bag my_teleop_run \
  --cmd-topic /tb4_5/cmd_vel
  
  faster playback: python3 bag_cmd_vel_player.py follow \
  --bag my_teleop_run \
  --cmd-topic /tb4_5/cmd_vel \
  --rate-scale 1.5'''