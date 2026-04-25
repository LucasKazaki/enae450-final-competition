#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import time

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')
        self.publisher_ = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        
        # Parameters (Adjust based on robot speed)
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.5 # rad/s
        self.dist = 1.0          # m
        self.angle = 3.14159     # rad (180 deg)

        self.run_sequence()

    def send_vel(self, linear=0.0, angular=0.0, duration=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear
        msg.twist.angular.z = angular
        
        # Publish for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            self.publisher_.publish(msg)
            time.sleep(0.1)
        
        # Stop
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.publisher_.publish(msg)

    def run_sequence(self):
        self.get_logger().info('Moving forward 1m...')
        self.send_vel(linear=self.linear_speed, duration=self.dist / self.linear_speed)
        
        time.sleep(1.0) # Short pause
        
        self.get_logger().info('Turning 180 degrees...')
        self.send_vel(angular=self.angular_speed, duration=self.angle / self.angular_speed)
        
        time.sleep(1.0) # Short pause
        
        self.get_logger().info('Moving forward 1m again...')
        self.send_vel(linear=self.linear_speed, duration=self.dist / self.linear_speed)
        
        self.get_logger().info('Sequence complete.')

def main(args=None):
    rclpy.init(args=args)
    node = RobotMover()
    rclpy.shutdown()

if __name__ == '__main__':
    main()