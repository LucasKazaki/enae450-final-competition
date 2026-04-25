#!/usr/bin/env python3
"""
Standalone TurtleBot4 SLAM + junction-based DFS maze solver.

This is a safer, more continuous rewrite of the original maze_solver.py.

Major behavior:
  1. Optionally launches slam_toolbox in mapping mode.
  2. Subscribes to LaserScan, Odometry, and OccupancyGrid.
  3. Publishes TwistStamped velocity commands.
  4. Uses a junction/corridor graph instead of stopping every fixed grid cell.
  5. Runs a DFS-style planner at decision points, with branch ordering biased toward
     the branch that appears to continue the farthest before the next wall/junction.
  6. Drives continuously down corridors until it sees a junction, turn, dead end,
     exit condition, or emergency obstacle.
  7. Includes a continuous safety monitor that can override motion commands.
  8. Optionally shows a matplotlib view and saves the final map.

Recommended simulator example:
  python3 maze_solver_junction_dfs.py \
    --scan-topic /tb4_3/scan \
    --map-topic /tb4_3/map \
    --odom-topic /tb4_3/odom \
    --cmd-topic /tb4_3/cmd_vel \
    --use-sim-time \
    --plot

If slam_toolbox is already running:
  python3 maze_solver_junction_dfs.py --no-slam --plot
"""

import argparse
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


# -----------------------------------------------------------------------------
# SLAM Toolbox launcher
# -----------------------------------------------------------------------------

def write_slam_params(args) -> str:
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


def run_cmd(cmd: List[str], name: str):
    print(f"\n[starting {name}]")
    print(" ".join(cmd))
    return subprocess.Popen(cmd, preexec_fn=os.setsid)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def clamp_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_cardinal(yaw: float) -> int:
    """Nearest cardinal heading: 0 east, 1 north, 2 west, 3 south."""
    yaw = clamp_angle(yaw)
    dirs = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
    errors = [abs(clamp_angle(yaw - d)) for d in dirs]
    return int(np.argmin(errors))


def cardinal_to_yaw(d: int) -> float:
    return [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0][d % 4]


def relative_to_global_heading(current_heading: int, rel: int) -> int:
    """rel: 0 forward, +1 left, -1 right, 2 back."""
    return (current_heading + rel) % 4


def heading_to_vector(h: int) -> Tuple[int, int]:
    h %= 4
    if h == 0:
        return (1, 0)
    if h == 1:
        return (0, 1)
    if h == 2:
        return (-1, 0)
    return (0, -1)


def world_to_key(x: float, y: float, spacing: float) -> Tuple[int, int]:
    return (int(round(x / spacing)), int(round(y / spacing)))


def key_to_world(key: Tuple[int, int], spacing: float) -> Tuple[float, float]:
    return (key[0] * spacing, key[1] * spacing)


def distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class JunctionNode:
    key: Tuple[int, int]
    x: float
    y: float
    open_headings: Set[int] = field(default_factory=set)
    visits: int = 0


# -----------------------------------------------------------------------------
# Main ROS node
# -----------------------------------------------------------------------------

class JunctionDfsMazeSolver(Node):
    def __init__(self, args):
        super().__init__("junction_dfs_maze_solver")
        self.args = args

        self.map_msg: Optional[OccupancyGrid] = None
        self.odom_msg: Optional[Odometry] = None
        self.scan_msg: Optional[LaserScan] = None
        self.lock = threading.RLock()

        self.cmd_pub = self.create_publisher(TwistStamped, args.cmd_topic, 10)
        self.create_subscription(OccupancyGrid, args.map_topic, self.map_callback, 10)
        self.create_subscription(Odometry, args.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, args.scan_topic, self.scan_callback, 10)

        # Graph state. Nodes are junctions, turns, dead ends, and the start.
        self.nodes: Dict[Tuple[int, int], JunctionNode] = {}
        self.edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        self.visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self.blocked_edges: Set[Tuple[Tuple[int, int], int]] = set()
        self.explored_headings: Set[Tuple[Tuple[int, int], int]] = set()
        self.parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.dfs_stack: List[Tuple[int, int]] = []

        self.start_node: Optional[Tuple[int, int]] = None
        self.current_node: Optional[Tuple[int, int]] = None
        self.last_node: Optional[Tuple[int, int]] = None
        self.active_heading: Optional[int] = None
        self.exit_found = False
        self.estop_active = False
        self.last_motion_time = time.time()
        self.exit_hits = 0

        # These are for plotting/debugging.
        self.current_target_text = "initializing"
        self.path_trace: List[Tuple[float, float]] = []

        self.get_logger().info(f"scan: {args.scan_topic}")
        self.get_logger().info(f"map:  {args.map_topic}")
        self.get_logger().info(f"odom: {args.odom_topic}")
        self.get_logger().info(f"cmd:  {args.cmd_topic}")

    # ------------------------------------------------------------------
    # ROS callbacks and data access
    # ------------------------------------------------------------------

    def map_callback(self, msg):
        with self.lock:
            self.map_msg = msg

    def odom_callback(self, msg):
        with self.lock:
            self.odom_msg = msg
            p = msg.pose.pose.position
            if not self.path_trace or distance_xy(self.path_trace[-1], (p.x, p.y)) > 0.05:
                self.path_trace.append((p.x, p.y))
                if len(self.path_trace) > 4000:
                    self.path_trace = self.path_trace[-4000:]

    def scan_callback(self, msg):
        with self.lock:
            self.scan_msg = msg

    def get_data(self):
        with self.lock:
            return self.map_msg, self.odom_msg, self.scan_msg

    def pose(self) -> Optional[Tuple[float, float, float]]:
        _, odom, _ = self.get_data()
        if odom is None:
            return None
        p = odom.pose.pose.position
        yaw = quat_to_yaw(odom.pose.pose.orientation)
        return p.x, p.y, yaw

    # ------------------------------------------------------------------
    # Lidar processing
    # ------------------------------------------------------------------

    def ranges_in_sector(self, center_angle: float, half_width_rad: float) -> List[float]:
        """Return finite lidar ranges inside a sector in the robot frame."""
        _, _, scan = self.get_data()
        if scan is None:
            return []

        values = []
        angle = scan.angle_min
        for r in scan.ranges:
            rel = clamp_angle(angle - center_angle)
            if abs(rel) <= half_width_rad:
                if np.isfinite(r) and scan.range_min < r < scan.range_max:
                    values.append(float(r))
            angle += scan.angle_increment
        return values

    def sector_percentile(self, center_angle: float, width_deg: float, percentile: float) -> float:
        vals = self.ranges_in_sector(center_angle, math.radians(width_deg) / 2.0)
        if not vals:
            return float("inf")
        return float(np.percentile(vals, percentile))

    def sector_min_distance(self, center_angle: float, width_deg: float) -> float:
        vals = self.ranges_in_sector(center_angle, math.radians(width_deg) / 2.0)
        if not vals:
            return float("inf")
        return min(vals)

    def front_min_distance(self, width_deg: float = 50.0) -> float:
        return self.sector_min_distance(0.0, width_deg)

    def side_distance(self, left: bool) -> float:
        return self.sector_percentile(math.pi / 2.0 if left else -math.pi / 2.0,
                                      self.args.side_width_deg,
                                      self.args.side_percentile)

    def emergency_stop_triggered(self) -> bool:
        """Hard stop if anything is dangerously close around the robot."""
        _, _, scan = self.get_data()
        if scan is None:
            return False

        count = 0
        for r in scan.ranges:
            if np.isfinite(r) and scan.range_min < r < self.args.emergency_distance:
                count += 1
                if count >= self.args.emergency_min_rays:
                    return True
        return False

    def exit_is_open_instant(self) -> bool:
        """
        Exit detector: front 180 degrees clear for 5 ft, with left and right walls
        also disappearing. Uses percentiles to avoid one bad lidar ray.
        """
        _, _, scan = self.get_data()
        if scan is None:
            return False

        front_vals = self.ranges_in_sector(0.0, math.pi / 2.0)
        if len(front_vals) < self.args.min_exit_rays:
            return False

        front_clear = float(np.percentile(front_vals, self.args.exit_percentile)) > self.args.exit_distance
        left_open = self.side_distance(left=True) > self.args.exit_side_distance
        right_open = self.side_distance(left=False) > self.args.exit_side_distance
        return front_clear and left_open and right_open

    def update_exit_detector(self) -> bool:
        if self.exit_is_open_instant():
            self.exit_hits += 1
        else:
            self.exit_hits = 0
        return self.exit_hits >= self.args.exit_confirm_scans

    def direction_open(self, rel: int, clearance: Optional[float] = None) -> bool:
        """Check if a relative direction is open by lidar."""
        if clearance is None:
            clearance = self.args.passage_open_distance
        rel_angle = {0: 0.0, 1: math.pi / 2.0, -1: -math.pi / 2.0, 2: math.pi}[rel]
        # Use percentile rather than min so a single noisy ray does not close a branch.
        d = self.sector_percentile(rel_angle, self.args.direction_width_deg, self.args.open_percentile)
        return d > clearance

    def get_open_relative_dirs(self) -> List[int]:
        """Return open relative directions among forward, left, right, back."""
        dirs = []
        for rel in [0, 1, -1, 2]:
            if self.direction_open(rel):
                dirs.append(rel)
        return dirs

    def estimate_branch_distance(self, rel: int) -> float:
        """
        Heuristic branch cost: estimate how long a branch continues from lidar.
        This is not a true next-junction distance in unknown space, but it is a useful
        ordering signal for DFS: larger means try this branch first.
        """
        rel_angle = {0: 0.0, 1: math.pi / 2.0, -1: -math.pi / 2.0, 2: math.pi}[rel]
        d = self.sector_percentile(rel_angle, self.args.branch_score_width_deg, self.args.branch_score_percentile)
        if not np.isfinite(d):
            return self.args.max_laser_range
        return min(d, self.args.max_laser_range)

    def detect_lidar_corners(self) -> int:
        """
        Counts abrupt lidar range jumps. Doorways and maze openings often appear as
        sharp discontinuities where a wall ends and open space begins.
        """
        _, _, scan = self.get_data()
        if scan is None or len(scan.ranges) < 3:
            return 0

        valid = []
        for r in scan.ranges:
            if np.isfinite(r) and scan.range_min < r < scan.range_max:
                valid.append(float(r))
            else:
                valid.append(np.nan)

        jumps = 0
        for a, b in zip(valid[:-1], valid[1:]):
            if np.isfinite(a) and np.isfinite(b) and abs(b - a) > self.args.corner_jump_distance:
                jumps += 1
        return jumps

    # ------------------------------------------------------------------
    # Occupancy grid helpers
    # ------------------------------------------------------------------

    def map_cell_state(self, wx: float, wy: float) -> str:
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

    def map_clear_along_heading(self, x: float, y: float, heading: int, distance: float) -> bool:
        dx, dy = heading_to_vector(heading)
        for s in np.linspace(0.15, distance, 5):
            state = self.map_cell_state(x + dx * s, y + dy * s)
            if state == "occupied":
                return False
        return True

    # ------------------------------------------------------------------
    # Command publishing and safety
    # ------------------------------------------------------------------

    def publish_cmd(self, linear: float = 0.0, angular: float = 0.0):
        # Safety layer has final authority over forward motion.
        if linear > 0.0 and self.emergency_stop_triggered():
            self.estop_active = True
            linear = 0.0
            angular = 0.0
        else:
            self.estop_active = False

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.base_frame
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        for _ in range(3):
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.03)

    def turn_to_yaw(self, target_yaw: float, timeout: float = 8.0) -> bool:
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

            angular = float(np.clip(self.args.turn_kp * err,
                                    -self.args.angular_speed,
                                    self.args.angular_speed))
            self.publish_cmd(0.0, angular)
            time.sleep(0.04)

        self.stop()
        return False

    def drive_exit_distance(self):
        pose0 = self.pose()
        if pose0 is None:
            return False
        x0, y0, _ = pose0
        start = time.time()
        while rclpy.ok() and time.time() - start < self.args.drive_timeout:
            pose = self.pose()
            if pose is None:
                self.stop()
                time.sleep(0.05)
                continue
            x, y, _ = pose
            if math.hypot(x - x0, y - y0) >= self.args.exit_drive_distance:
                self.stop()
                return True
            if self.emergency_stop_triggered():
                self.get_logger().warn("Emergency obstacle while exiting; stopped.")
                self.stop()
                return False
            self.publish_cmd(self.args.linear_speed, 0.0)
            time.sleep(0.05)
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Junction graph and DFS
    # ------------------------------------------------------------------

    def current_node_key(self) -> Optional[Tuple[int, int]]:
        pose = self.pose()
        if pose is None:
            return None
        x, y, _ = pose
        return world_to_key(x, y, self.args.node_spacing)

    def add_or_update_node(self, open_headings: Set[int]) -> Optional[Tuple[int, int]]:
        pose = self.pose()
        if pose is None:
            return None
        x, y, _ = pose
        key = world_to_key(x, y, self.args.node_spacing)
        with self.lock:
            if key not in self.nodes:
                self.nodes[key] = JunctionNode(key=key, x=x, y=y, open_headings=set(open_headings), visits=0)
            else:
                self.nodes[key].open_headings.update(open_headings)
                # Smooth the displayed node location a little.
                self.nodes[key].x = 0.8 * self.nodes[key].x + 0.2 * x
                self.nodes[key].y = 0.8 * self.nodes[key].y + 0.2 * y
            self.nodes[key].visits += 1
        return key

    @staticmethod
    def edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    def connect_nodes(self, a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]):
        if a is None or b is None or a == b:
            return
        ax, ay = key_to_world(a, self.args.node_spacing)
        bx, by = key_to_world(b, self.args.node_spacing)
        cost = math.hypot(ax - bx, ay - by)
        ek = self.edge_key(a, b)
        self.edges[ek] = max(cost, self.edges.get(ek, 0.0))
        self.visited_edges.add(ek)
        if b not in self.parent:
            self.parent[b] = a

    def is_decision_point(self, open_rels: List[int], traveled_since_node: float) -> bool:
        """
        Decide when to stop and run DFS.
        We stop at: dead ends, side openings, turns, strong corner signatures, or
        after traveling too long without formalizing a graph node.
        """
        if traveled_since_node < self.args.min_node_separation:
            return False

        front = 0 in open_rels
        left = 1 in open_rels
        right = -1 in open_rels
        back = 2 in open_rels
        side_count = int(left) + int(right)
        total_nonback = int(front) + side_count

        dead_end = not front and not left and not right
        branch = side_count > 0 and total_nonback >= 2
        corner = (not front) and (left or right)
        lidar_corner_signature = self.detect_lidar_corners() >= self.args.corner_jump_count
        too_long = traveled_since_node > self.args.max_corridor_without_node

        # We do not stop for a perfectly straight corridor with only front/back open.
        return dead_end or branch or corner or lidar_corner_signature or too_long

    def choose_next_heading_at_node(self, node_key: Tuple[int, int], current_heading: int, open_rels: List[int]) -> Optional[int]:
        """
        Junction DFS.
        Among unvisited outgoing branches, sort by estimated branch distance descending.
        If all branches are visited/blocked, backtrack to the parent node when possible.
        """
        candidates = []
        pose = self.pose()
        if pose is None:
            return None
        x, y, _ = pose

        for rel in open_rels:
            heading = relative_to_global_heading(current_heading, rel)
            if (node_key, heading) in self.blocked_edges:
                continue
            if not self.map_clear_along_heading(x, y, heading, self.args.map_probe_distance):
                self.blocked_edges.add((node_key, heading))
                continue

            # Prefer not to immediately go backward unless there are no unvisited choices.
            score = self.estimate_branch_distance(rel)
            is_back = (rel == 2)
            candidates.append((is_back, -score, rel, heading))

        # First pass: unvisited-looking branches.
        unvisited = []
        for is_back, neg_score, rel, heading in candidates:
            if (node_key, heading) not in self.explored_headings and not is_back:
                unvisited.append((neg_score, rel, heading))

        if unvisited:
            unvisited.sort()  # neg_score ascending = longest branch first.
            return unvisited[0][2]

        # Second pass: any non-back open direction, longest first.
        non_back = [(neg_score, rel, heading) for is_back, neg_score, rel, heading in candidates if not is_back]
        if non_back:
            non_back.sort()
            return non_back[0][2]

        # Backtrack toward parent if all else fails.
        if node_key in self.parent:
            parent = self.parent[node_key]
            px, py = parent
            dx = px - node_key[0]
            dy = py - node_key[1]
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 2
            return 1 if dy > 0 else 3

        # Last resort: take any open direction.
        if candidates:
            candidates.sort()
            return candidates[0][3]
        return None

    # ------------------------------------------------------------------
    # Main behavior modes
    # ------------------------------------------------------------------

    def initialize_start(self):
        pose = self.pose()
        if pose is None:
            return False
        _, _, yaw = pose
        h = yaw_to_cardinal(yaw)
        open_rels = self.get_open_relative_dirs()
        open_headings = {relative_to_global_heading(h, rel) for rel in open_rels}
        key = self.add_or_update_node(open_headings)
        self.start_node = key
        self.current_node = key
        self.last_node = key
        if key is not None and key not in self.dfs_stack:
            self.dfs_stack.append(key)
        self.current_target_text = "start"
        return True

    def run_junction_dfs(self):
        """Continuous corridor driving with DFS decisions at junctions."""
        if self.start_node is None:
            if not self.initialize_start():
                self.get_logger().info("Waiting for odom before initializing DFS graph...")
                time.sleep(0.3)
                return

        pose = self.pose()
        if pose is None:
            self.get_logger().info("Waiting for odom...")
            time.sleep(0.3)
            return

        x0, y0, yaw0 = pose
        current_heading = yaw_to_cardinal(yaw0)
        open_rels = self.get_open_relative_dirs()
        open_headings = {relative_to_global_heading(current_heading, rel) for rel in open_rels}

        # Treat the current position as a node whenever we enter this method.
        node_key = self.add_or_update_node(open_headings)
        self.connect_nodes(self.last_node, node_key)
        self.current_node = node_key

        if self.update_exit_detector():
            self.get_logger().info("Exit detected: front is clear and both side walls opened up. Driving out.")
            self.exit_found = True
            self.stop()
            self.drive_exit_distance()
            return

        next_heading = self.choose_next_heading_at_node(node_key, current_heading, open_rels)
        if next_heading is None:
            self.get_logger().warn("No valid DFS branch found. Stopping.")
            self.stop()
            time.sleep(0.5)
            return

        self.current_target_text = f"heading {next_heading}"
        target_yaw = cardinal_to_yaw(next_heading)
        if not self.turn_to_yaw(target_yaw):
            self.get_logger().warn("Turn failed to converge. Stopping before retry.")
            self.stop()
            return

        self.active_heading = next_heading
        if node_key is not None:
            self.explored_headings.add((node_key, next_heading))
        self.last_node = node_key
        self.drive_until_next_decision(next_heading, node_key)

    def drive_until_next_decision(self, heading: int, from_node: Optional[Tuple[int, int]]):
        """Drive continuously until a junction/dead end/turn/exit/emergency condition."""
        pose0 = self.pose()
        if pose0 is None:
            return
        x_start, y_start, _ = pose0
        target_yaw = cardinal_to_yaw(heading)
        start_time = time.time()

        while rclpy.ok() and not self.exit_found:
            pose = self.pose()
            if pose is None:
                self.stop()
                time.sleep(0.05)
                continue
            x, y, yaw = pose
            traveled = math.hypot(x - x_start, y - y_start)

            if self.emergency_stop_triggered():
                self.get_logger().warn("Emergency stop: obstacle too close.")
                self.stop()
                # Mark this outgoing branch as blocked so DFS tries something else.
                if from_node is not None:
                    self.blocked_edges.add((from_node, heading))
                return

            if self.update_exit_detector():
                self.get_logger().info("Exit detected while moving. Driving out.")
                self.exit_found = True
                self.drive_exit_distance()
                return

            # Normal forward collision guard. This catches an obstacle before the hard estop.
            if self.front_min_distance(width_deg=self.args.front_guard_width_deg) < self.args.safe_distance:
                self.get_logger().info("Front obstacle/maze wall reached. Making a decision point.")
                self.stop()
                new_open = self.get_open_relative_dirs()
                h_now = yaw_to_cardinal(yaw)
                open_headings = {relative_to_global_heading(h_now, rel) for rel in new_open}
                new_node = self.add_or_update_node(open_headings)
                self.connect_nodes(from_node, new_node)
                self.current_node = new_node
                return

            open_rels = self.get_open_relative_dirs()
            if self.is_decision_point(open_rels, traveled):
                self.get_logger().info("Decision point detected: stopping to run DFS.")
                self.stop()
                h_now = yaw_to_cardinal(yaw)
                open_headings = {relative_to_global_heading(h_now, rel) for rel in open_rels}
                new_node = self.add_or_update_node(open_headings)
                self.connect_nodes(from_node, new_node)
                self.current_node = new_node
                return

            if time.time() - start_time > self.args.drive_timeout:
                self.get_logger().info("Drive segment timeout. Creating a conservative graph node.")
                self.stop()
                h_now = yaw_to_cardinal(yaw)
                open_headings = {relative_to_global_heading(h_now, rel) for rel in open_rels}
                new_node = self.add_or_update_node(open_headings)
                self.connect_nodes(from_node, new_node)
                self.current_node = new_node
                return

            yaw_err = clamp_angle(target_yaw - yaw)
            angular = float(np.clip(self.args.drive_heading_kp * yaw_err,
                                    -self.args.drive_max_angular_correction,
                                    self.args.drive_max_angular_correction))
            self.publish_cmd(self.args.linear_speed, angular)
            time.sleep(0.04)

    def run_grid_fallback_step(self):
        """
        Conservative old-style fallback. Kept for testing if junction mode is unstable.
        It makes one short move at a time using the same lidar/map safety layer.
        """
        pose = self.pose()
        if pose is None:
            self.get_logger().info("Waiting for odom...")
            time.sleep(0.3)
            return

        x, y, yaw = pose
        h = yaw_to_cardinal(yaw)
        key = world_to_key(x, y, self.args.cell_size)
        open_rels = self.get_open_relative_dirs()
        open_headings = {relative_to_global_heading(h, rel) for rel in open_rels}
        node_key = self.add_or_update_node(open_headings)

        if self.update_exit_detector():
            self.get_logger().info("Exit detected in grid fallback. Driving out.")
            self.exit_found = True
            self.drive_exit_distance()
            return

        next_heading = self.choose_next_heading_at_node(node_key, h, open_rels)
        if next_heading is None:
            self.stop()
            return
        if node_key is not None:
            self.explored_headings.add((node_key, next_heading))
        if not self.turn_to_yaw(cardinal_to_yaw(next_heading)):
            return

        # Drive one cell size, still under emergency and front guards.
        pose0 = self.pose()
        if pose0 is None:
            return
        x0, y0, _ = pose0
        start = time.time()
        while rclpy.ok() and time.time() - start < self.args.drive_timeout:
            pose = self.pose()
            if pose is None:
                break
            x1, y1, _ = pose
            if math.hypot(x1 - x0, y1 - y0) >= self.args.cell_size:
                break
            if self.emergency_stop_triggered() or self.front_min_distance() < self.args.safe_distance:
                break
            self.publish_cmd(self.args.linear_speed, 0.0)
            time.sleep(0.04)
        self.stop()


# -----------------------------------------------------------------------------
# Optional matplotlib viewer
# -----------------------------------------------------------------------------

def draw_loop(node: JunctionDfsMazeSolver):
    if plt is None:
        node.get_logger().warn("matplotlib not available; running without plot.")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    while rclpy.ok() and not node.exit_found:
        map_msg, odom_msg, _ = node.get_data()
        ax.clear()
        ax.set_title(f"Junction DFS Maze Solver | {node.current_target_text}")
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

        # Path trace.
        if node.path_trace:
            xs, ys = zip(*node.path_trace)
            ax.plot(xs, ys, linewidth=1, label="path")

        # Graph edges.
        for a, b in list(node.edges.keys()):
            if a in node.nodes and b in node.nodes:
                ax.plot([node.nodes[a].x, node.nodes[b].x], [node.nodes[a].y, node.nodes[b].y], linestyle="--", linewidth=1)

        # Graph nodes.
        if node.nodes:
            xs = [n.x for n in node.nodes.values()]
            ys = [n.y for n in node.nodes.values()]
            ax.scatter(xs, ys, s=25, label="junction nodes")

        if odom_msg is not None:
            px = odom_msg.pose.pose.position.x
            py = odom_msg.pose.pose.position.y
            yaw = quat_to_yaw(odom_msg.pose.pose.orientation)
            ax.plot(px, py, "ro", markersize=6, label="robot")
            ax.arrow(px, py, 0.35 * math.cos(yaw), 0.35 * math.sin(yaw), head_width=0.12, head_length=0.15)

        ax.legend(loc="upper right")
        plt.pause(node.args.plot_period)

    plt.ioff()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Plain defaults are easiest for non-namespaced sim. Pass /tb4_3/... if needed.
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

    parser.add_argument("--mode", choices=["junction", "grid"], default="junction",
                        help="junction = continuous junction DFS; grid = conservative old-style fallback")

    # Motion and safety.
    parser.add_argument("--linear-speed", type=float, default=0.14)
    parser.add_argument("--angular-speed", type=float, default=0.45)
    parser.add_argument("--turn-kp", type=float, default=1.6)
    parser.add_argument("--turn-tolerance", type=float, default=0.08)
    parser.add_argument("--drive-heading-kp", type=float, default=1.1)
    parser.add_argument("--drive-max-angular-correction", type=float, default=0.18)
    parser.add_argument("--drive-timeout", type=float, default=12.0)
    parser.add_argument("--safe-distance", type=float, default=0.45)
    parser.add_argument("--emergency-distance", type=float, default=0.28)
    parser.add_argument("--emergency-min-rays", type=int, default=3)
    parser.add_argument("--front-guard-width-deg", type=float, default=55.0)

    # Junction graph and branch detection.
    parser.add_argument("--cell-size", type=float, default=0.35,
                        help="Used only in grid fallback mode.")
    parser.add_argument("--node-spacing", type=float, default=0.45,
                        help="Quantization spacing for junction graph nodes.")
    parser.add_argument("--min-node-separation", type=float, default=0.35,
                        help="Do not create another node until this much travel has occurred.")
    parser.add_argument("--max-corridor-without-node", type=float, default=1.6,
                        help="Force a conservative node after this much corridor travel.")
    parser.add_argument("--passage-open-distance", type=float, default=0.75,
                        help="A direction is considered a passage if this much space is visible.")
    parser.add_argument("--direction-width-deg", type=float, default=45.0)
    parser.add_argument("--open-percentile", type=float, default=25.0)
    parser.add_argument("--branch-score-width-deg", type=float, default=35.0)
    parser.add_argument("--branch-score-percentile", type=float, default=60.0)
    parser.add_argument("--map-probe-distance", type=float, default=0.55)
    parser.add_argument("--occupied-threshold", type=int, default=50)
    parser.add_argument("--corner-jump-distance", type=float, default=0.8)
    parser.add_argument("--corner-jump-count", type=int, default=2)

    # Exit detection.
    parser.add_argument("--exit-distance", type=float, default=1.524,
                        help="5 ft in meters for front 180-degree exit detection.")
    parser.add_argument("--exit-side-distance", type=float, default=1.0,
                        help="Side wall must be farther than this to count as open exit space.")
    parser.add_argument("--exit-drive-distance", type=float, default=1.25)
    parser.add_argument("--min-exit-rays", type=int, default=20)
    parser.add_argument("--exit-percentile", type=float, default=15.0)
    parser.add_argument("--exit-confirm-scans", type=int, default=5)
    parser.add_argument("--side-width-deg", type=float, default=35.0)
    parser.add_argument("--side-percentile", type=float, default=30.0)

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
        node = JunctionDfsMazeSolver(args)

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

        node.get_logger().info(f"Starting {args.mode} DFS maze solver. Press Ctrl+C to stop.")
        while rclpy.ok() and not node.exit_found:
            if args.mode == "junction":
                node.run_junction_dfs()
            else:
                node.run_grid_fallback_step()
            time.sleep(0.05)

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

'''python3 maze_solver_junction_dfs.py \
  --scan-topic /tb4_3/scan \
  --map-topic /tb4_3/map \
  --odom-topic /tb4_3/odom \
  --cmd-topic /tb4_3/cmd_vel \
  --use-sim-time \
  --plot
  
  safe: 
  python3 maze_solver_junction_dfs.py \
  --scan-topic /tb4_3/scan \
  --map-topic /tb4_3/map \
  --odom-topic /tb4_3/odom \
  --cmd-topic /tb4_3/cmd_vel \
  --use-sim-time \
  --plot \
  --linear-speed 0.10 \
  --safe-distance 0.55 \
  --emergency-distance 0.32'''