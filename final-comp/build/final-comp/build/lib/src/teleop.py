#!/usr/bin/env python3

import math
import sys
import termios
import time
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


MSG = """
Keyboard Teleop Controls:

    w
a   s   d

w: forward
s: backward
a: turn left
d: turn right

q: increase speed
e: decrease speed

space: stop
CTRL+C: quit
"""
cmd_topic = "/tb4_6/cmd_vel"

def get_key(timeout=0.1):
    """Non-blocking keyboard input"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return key


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")

        self.cmd_topic = self.declare_parameter(
            "cmd_topic", cmd_topic
        ).value

        self.pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)

        # Speeds
        self.linear_speed = 0.2
        self.angular_speed = 0.6

        self.current_linear = 0.0
        self.current_angular = 0.0

        self.timer = self.create_timer(0.05, self.control_loop)

        print(MSG)
        self.get_logger().info(f"Publishing to {self.cmd_topic}")

    def control_loop(self):
        key = get_key()

        if key == "w":
            self.current_linear = self.linear_speed
            self.current_angular = 0.0

        elif key == "s":
            self.current_linear = -self.linear_speed
            self.current_angular = 0.0

        elif key == "a":
            self.current_linear = 0.0
            self.current_angular = self.angular_speed

        elif key == "d":
            self.current_linear = 0.0
            self.current_angular = -self.angular_speed

        elif key == " ":
            self.current_linear = 0.0
            self.current_angular = 0.0

        elif key == "q":
            self.linear_speed *= 1.1
            self.angular_speed *= 1.1
            print(f"Speed increased: linear={self.linear_speed:.2f}, angular={self.angular_speed:.2f}")

        elif key == "e":
            self.linear_speed *= 0.9
            self.angular_speed *= 0.9
            print(f"Speed decreased: linear={self.linear_speed:.2f}, angular={self.angular_speed:.2f}")

        elif key == "\x03":  # Ctrl+C
            raise KeyboardInterrupt

        elif key is not None:
            # Unknown key → stop
            self.current_linear = 0.0
            self.current_angular = 0.0

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"

        msg.twist.linear.x = self.current_linear
        msg.twist.angular.z = self.current_angular

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = KeyboardTeleop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C detected. Stopping teleop.")
    finally:
        #do 180 at the end and stop
        node.get_logger().info("Doing 180-degree turn.")
        turn = TwistStamped()
        turn.header.stamp = node.get_clock().now().to_msg()
        turn.twist.linear.x = 0.0
        turn.twist.angular.z = 1.0 
        node.pub.publish(turn)
        time.sleep(math.pi) 
        # Send stop command on exit
        stop = TwistStamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.twist.linear.x = 0.0
        stop.twist.angular.z = 0.0
        node.pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

'''ros2 run final-comp teleop --ros-args -p cmd_topic:=/tb4_4/cmd_vel'''