#!/usr/bin/env python3

"""
Standalone TurtleBot4 maze solver using the same basic bones as:
  - slam_map_viewer.py: starts slam_toolbox, subscribes to /map, /odom, /scan, optional matplotlib view
  - move_robot.py: publishes TwistStamped commands to /cmd_vel

Default topics match the working setup from slam_map_viewer.py:
  /tb4_3/scan, /tb4_3/map, /tb4_3/odom, /tb4_3/cmd_vel

Core idea:
  1. Launch slam_toolbox in mapping mode.
  2. Continuously read lidar, odom, and occupancy grid map.
  3. Use a simple DFS-like wall-safe frontier behavior:
       - If a wide 180-degree front sector is open for exit_distance meters, declare exit and drive forward.
       - Otherwise prefer unexplored/free directions in a DFS order: forward, left, right, back.
       - Before moving one cell, check lidar clearance and map occupancy.
       - Mark blocked/dead-end cells and backtrack when stuck.
  4. Optionally save the map on shutdown.

This is intentionally conservative. Tune --cell-size, --linear-speed, --angular-speed,
--safe-distance, and --exit-distance for your robot/maze.
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# slam_toolbox launcher helpers
# -----------------------------

def write_slam_params(args):
    params = f"""
slam_toolbox:
  ros__parameters:
    use_sim_time: {str(args.use_sim_time).lower()}
    mode: mapping

    odom_frame: {args.odom_frame}
    map_frame: {args.map_frame}
    base_frame: {args.base_frame}
    scan_topic: {args.scan_topic}

    resolution: {args.resolution}
    max_laser_range: {args.max_laser_range}

    map_update_interval: 0.5
    use_scan_matching: true
    use_scan_barycenter: true
    minimum_travel_distance: 0.05
    minimum_travel_heading: 0.05

    do_loop_closing: true
    transform_timeout: 0.5
    tf_buffer_duration: 30.0
    stack_size_to_use: 40000000
"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
    tmp.write(params)
    tmp.close()
    return tmp.name


def run_cmd(cmd, name):
    print(f"\n[starting {name}]")
    print(" ".join(cmd))
    return subprocess.Popen(cmd, preexec_fn=os.setsid)


# -----------------------------
# Geometry helpers
# -----------------------------

def clamp_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_cardinal(yaw):
    """Return nearest grid heading: 0 east, 1 north, 2 west, 3 south."""
    yaw = clamp_angle(yaw)
    dirs = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
    errors = [abs(clamp_angle(yaw - d)) for d in dirs]
    return int(np.argmin(errors))


def cardinal_to_yaw(d):
    return [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0][d % 4]


def grid_step(cell, d):
    x, y = cell
    d = d % 4
    if d == 0:
        return (x + 1, y)
    if d == 1:
        return (x, y + 1)
    if d == 2:
        return (x - 1, y)
    return (x, y - 1)


def world_to_solver_cell(x, y, cell_size):
    return (int(round(x / cell_size)), int(round(y / cell_size)))


def solver_cell_to_world(cell, cell_size):
    return (cell[0] * cell_size, cell[1] * cell_size)


# -----------------------------
# Main ROS node
# -----------------------------

class SlamDfsMazeSolver(Node):
    def __init__(self, args):
        super().__init__("slam_dfs_maze_solver")
        self.args = args

        self.map_msg = None
        self.odom_msg = None
        self.scan_msg = None
        self.lock = threading.Lock()

        self.cmd_pub = self.create_publisher(TwistStamped, args.cmd_topic, 10)

        self.create_subscription(OccupancyGrid, args.map_topic, self.map_callback, 10)
        self.create_subscription(Odometry, args.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, args.scan_topic, self.scan_callback, 10)

        self.visited = set()
        self.blocked = set()
        self.parent = {}
        self.stack = []
        self.exit_found = False
        self.start_cell = None

        self.get_logger().info(f"scan: {args.scan_topic}")
        self.get_logger().info(f"map:  {args.map_topic}")
        self.get_logger().info(f"odom: {args.odom_topic}")
        self.get_logger().info(f"cmd:  {args.cmd_topic}")

    def map_callback(self, msg):
        with self.lock:
            self.map_msg = msg

    def odom_callback(self, msg):
        with self.lock:
            self.odom_msg = msg

    def scan_callback(self, msg):
        with self.lock:
            self.scan_msg = msg

    def get_data(self):
        with self.lock:
            return self.map_msg, self.odom_msg, self.scan_msg

    def pose(self):
        _, odom, _ = self.get_data()
        if odom is None:
            return None
        p = odom.pose.pose.position
        yaw = quat_to_yaw(odom.pose.pose.orientation)
        return p.x, p.y, yaw

    # -----------------------------
    # Velocity helpers, based on move_robot.py style
    # -----------------------------

    def publish_cmd(self, linear=0.0, angular=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.base_frame
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        for _ in range(3):
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.05)

    def turn_to_yaw(self, target_yaw, timeout=8.0):
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            pose = self.pose()
            if pose is None:
                self.stop()
                time.sleep(0.05)
                continue

            _, _, yaw = pose
            err = clamp_angle(target_yaw - yaw)
            if abs(err) < self.args.turn_tolerance:
                self.stop()
                return True

            angular = np.clip(self.args.turn_kp * err,
                              -self.args.angular_speed,
                              self.args.angular_speed)
            self.publish_cmd(0.0, angular)
            time.sleep(0.05)

        self.stop()
        return False

    def drive_distance(self, distance):
        pose0 = self.pose()
        if pose0 is None:
            return False
        x0, y0, _ = pose0
        sign = 1.0 if distance >= 0.0 else -1.0
        target = abs(distance)
        start = time.time()

        while rclpy.ok() and time.time() - start < self.args.drive_timeout:
            pose = self.pose()
            if pose is None:
                self.stop()
                time.sleep(0.05)
                continue

            x, y, _ = pose
            traveled = math.hypot(x - x0, y - y0)
            if traveled >= target:
                self.stop()
                return True

            # Safety stop if lidar sees something too close in front while moving forward.
            if sign > 0 and self.front_min_distance(width_deg=50.0) < self.args.safe_distance:
                self.get_logger().warn("Obstacle detected while driving; stopping early.")
                self.stop()
                return False

            self.publish_cmd(sign * self.args.linear_speed, 0.0)
            time.sleep(0.05)

        self.stop()
        return False

    # -----------------------------
    # Lidar and map checks
    # -----------------------------

    def ranges_in_sector(self, center_angle, half_width):
        _, _, scan = self.get_data()
        if scan is None:
            return []

        values = []
        angle = scan.angle_min
        for r in scan.ranges:
            rel = clamp_angle(angle - center_angle)
            if abs(rel) <= half_width:
                if np.isfinite(r) and scan.range_min < r < scan.range_max:
                    values.append(float(r))
            angle += scan.angle_increment
        return values

    def sector_min_distance(self, center_angle, width_deg):
        vals = self.ranges_in_sector(center_angle, math.radians(width_deg) / 2.0)
        if not vals:
            return float("inf")
        return min(vals)

    def front_min_distance(self, width_deg=50.0):
        # In standard LaserScan coordinates, angle 0 is forward.
        return self.sector_min_distance(0.0, width_deg)

    def exit_is_open(self):
        """Check the full 180-degree forward field, not just the direct front ray."""
        _, _, scan = self.get_data()
        if scan is None:
            return False

        vals = self.ranges_in_sector(0.0, math.pi / 2.0)
        if len(vals) < self.args.min_exit_rays:
            return False

        # Use a low percentile instead of min so one noisy ray does not prevent exit detection.
        p = np.percentile(vals, self.args.exit_percentile)
        return p > self.args.exit_distance

    def direction_clear_by_lidar(self, relative_angle):
        d = self.sector_min_distance(relative_angle, self.args.direction_width_deg)
        return d > self.args.safe_distance

    def map_cell_state(self, wx, wy):
        map_msg, _, _ = self.get_data()
        if map_msg is None:
            return "unknown"

        res = map_msg.info.resolution
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        mx = int((wx - ox) / res)
        my = int((wy - oy) / res)

        if mx < 0 or my < 0 or mx >= map_msg.info.width or my >= map_msg.info.height:
            return "unknown"

        idx = my * map_msg.info.width + mx
        val = map_msg.data[idx]
        if val < 0:
            return "unknown"
        if val > self.args.occupied_threshold:
            return "occupied"
        return "free"

    def direction_clear_by_map(self, from_cell, heading):
        # Sample a few points between this cell and the next cell.
        x0, y0 = solver_cell_to_world(from_cell, self.args.cell_size)
        x1, y1 = solver_cell_to_world(grid_step(from_cell, heading), self.args.cell_size)
        for t in np.linspace(0.25, 1.0, 4):
            wx = x0 + t * (x1 - x0)
            wy = y0 + t * (y1 - y0)
            state = self.map_cell_state(wx, wy)
            if state == "occupied":
                return False
        return True

    # -----------------------------
    # DFS behavior
    # -----------------------------

    def choose_next_heading(self, cell, current_heading):
        """
        DFS preference order relative to current heading:
          forward, left, right, back
        This keeps movement natural while still exploring unvisited cells.
        """
        relative_order = [0, 1, -1, 2]

        # First choose unvisited neighbors.
        for rel in relative_order:
            heading = (current_heading + rel) % 4
            nxt = grid_step(cell, heading)
            if nxt in self.visited or nxt in self.blocked:
                continue

            relative_angle = clamp_angle(cardinal_to_yaw(heading) - cardinal_to_yaw(current_heading))
            if not self.direction_clear_by_lidar(relative_angle):
                self.blocked.add(nxt)
                continue
            if not self.direction_clear_by_map(cell, heading):
                self.blocked.add(nxt)
                continue
            return heading, nxt

        # If all neighbors visited/blocked, backtrack to parent if available.
        if cell in self.parent:
            parent = self.parent[cell]
            for heading in range(4):
                if grid_step(cell, heading) == parent:
                    return heading, parent

        return None, None

    def solve_step(self):
        pose = self.pose()
        if pose is None:
            self.get_logger().info("Waiting for odom...")
            time.sleep(0.5)
            return

        x, y, yaw = pose
        cell = world_to_solver_cell(x, y, self.args.cell_size)
        heading = yaw_to_cardinal(yaw)

        if self.start_cell is None:
            self.start_cell = cell
            self.stack.append(cell)

        self.visited.add(cell)

        if self.exit_is_open():
            self.get_logger().info(
                f"Exit appears open: 180-degree front sector clear for {self.args.exit_distance:.2f} m. Driving out."
            )
            self.exit_found = True
            self.turn_to_yaw(cardinal_to_yaw(heading))
            self.drive_distance(self.args.exit_drive_distance)
            self.stop()
            return

        next_heading, next_cell = self.choose_next_heading(cell, heading)
        if next_heading is None:
            self.get_logger().warn("No DFS move available. Robot may be boxed in or map/lidar clearance is too conservative.")
            self.stop()
            time.sleep(0.5)
            return

        if next_cell not in self.visited and next_cell not in self.parent:
            self.parent[next_cell] = cell

        target_yaw = cardinal_to_yaw(next_heading)
        self.get_logger().info(f"DFS move: cell {cell} -> {next_cell}, heading {next_heading}")
        turned = self.turn_to_yaw(target_yaw)
        if not turned:
            self.get_logger().warn("Turn did not converge; skipping this step.")
            return

        moved = self.drive_distance(self.args.cell_size)
        if not moved:
            self.blocked.add(next_cell)
            return

        self.visited.add(next_cell)


# -----------------------------
# Optional matplotlib viewer
# -----------------------------

def draw_loop(node):
    if plt is None:
        node.get_logger().warn("matplotlib not available; running without plot.")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    while rclpy.ok() and not node.exit_found:
        map_msg, odom_msg, scan_msg = node.get_data()
        ax.clear()
        ax.set_title("SLAM DFS Maze Solver")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")

        if map_msg is not None:
            width = map_msg.info.width
            height = map_msg.info.height
            res = map_msg.info.resolution
            ox = map_msg.info.origin.position.x
            oy = map_msg.info.origin.position.y
            data = np.array(map_msg.data, dtype=np.int16).reshape((height, width))
            display = np.zeros_like(data, dtype=np.float32)
            display[data == -1] = 0.5
            display[data == 0] = 1.0
            display[data > node.args.occupied_threshold] = 0.0
            extent = [ox, ox + width * res, oy, oy + height * res]
            ax.imshow(display, cmap="gray", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
        else:
            ax.text(0.5, 0.5, "Waiting for map...", transform=ax.transAxes, ha="center", va="center")

        # Draw visited solver cells.
        if node.visited:
            xs, ys = zip(*[solver_cell_to_world(c, node.args.cell_size) for c in node.visited])
            ax.scatter(xs, ys, s=12, label="visited")

        if node.blocked:
            xs, ys = zip(*[solver_cell_to_world(c, node.args.cell_size) for c in node.blocked])
            ax.scatter(xs, ys, s=12, marker="x", label="blocked")

        if odom_msg is not None:
            px = odom_msg.pose.pose.position.x
            py = odom_msg.pose.pose.position.y
            yaw = quat_to_yaw(odom_msg.pose.pose.orientation)
            ax.plot(px, py, "ro", markersize=6)
            ax.arrow(px, py, 0.35 * math.cos(yaw), 0.35 * math.sin(yaw), head_width=0.12, head_length=0.15)

        ax.legend(loc="upper right")
        plt.pause(node.args.plot_period)

    plt.ioff()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    # Working defaults from the uploaded slam_map_viewer.py, plus cmd_vel.
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--map-topic", default="map")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-topic", default="cmd_vel")

    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--map-frame", default="map")

    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--max-laser-range", type=float, default=12.0)
    parser.add_argument("--use-sim-time", action="store_true")

    parser.add_argument("--cell-size", type=float, default=0.45,
                        help="DFS grid step size in meters. Use 0.35-0.60 depending on maze width.")
    parser.add_argument("--safe-distance", type=float, default=0.45,
                        help="Minimum lidar clearance before allowing motion.")
    parser.add_argument("--exit-distance", type=float, default=1.524,
                        help="5 ft in meters. Used for 180-degree forward exit detection.")
    parser.add_argument("--exit-drive-distance", type=float, default=1.25,
                        help="How far to drive after exit is detected.")
    parser.add_argument("--min-exit-rays", type=int, default=20)
    parser.add_argument("--exit-percentile", type=float, default=15.0,
                        help="Percentile of front 180-degree rays that must exceed exit distance.")
    parser.add_argument("--direction-width-deg", type=float, default=45.0)
    parser.add_argument("--occupied-threshold", type=int, default=50)

    parser.add_argument("--linear-speed", type=float, default=0.16)
    parser.add_argument("--angular-speed", type=float, default=0.45)
    parser.add_argument("--turn-kp", type=float, default=1.6)
    parser.add_argument("--turn-tolerance", type=float, default=0.08)
    parser.add_argument("--drive-timeout", type=float, default=8.0)

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-period", type=float, default=0.25)
    parser.add_argument("--save-map", default="")
    parser.add_argument("--no-slam", action="store_true",
                        help="Do not launch slam_toolbox. Use this if it is already running.")

    args = parser.parse_args()

    params_file = None
    procs = []
    node = None

    try:
        if not args.no_slam:
            params_file = write_slam_params(args)
            slam_cmd = [
                "ros2", "run", "slam_toolbox", "async_slam_toolbox_node",
                "--ros-args", "--params-file", params_file,
            ]
            procs.append(run_cmd(slam_cmd, "slam_toolbox"))
            time.sleep(1.0)

        rclpy.init()
        node = SlamDfsMazeSolver(args)

        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

        if args.plot:
            plot_thread = threading.Thread(target=draw_loop, args=(node,), daemon=True)
            plot_thread.start()

        node.get_logger().info("Waiting for scan, odom, and map...")
        while rclpy.ok():
            map_msg, odom_msg, scan_msg = node.get_data()
            if odom_msg is not None and scan_msg is not None and (map_msg is not None or args.no_slam):
                break
            time.sleep(0.2)

        node.get_logger().info("Starting DFS maze solver. Press Ctrl+C to stop.")
        while rclpy.ok() and not node.exit_found:
            node.solve_step()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if node is not None:
            node.stop()

        if args.save_map:
            save_path = Path(args.save_map).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving map to: {save_path}")
            subprocess.run([
                "ros2", "run", "nav2_map_server", "map_saver_cli",
                "-f", str(save_path)
            ])

        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception:
                pass

        try:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

        if params_file:
            try:
                os.remove(params_file)
            except Exception:
                pass

        print("Done.")


if __name__ == "__main__":
    main()
