#!/usr/bin/env python3
import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan


@dataclass
class JunctionState:
    branches: List[str]
    tried: Set[str]
    parent_branch: Optional[str]


class MazeMapperSolver(Node):
    def __init__(self, args):
        super().__init__('maze_mapper_solver')
        self.args = args

        self.scan_topic = args.scan_topic
        self.odom_topic = args.odom_topic
        self.map_topic = args.map_topic
        self.cmd_vel_topic = args.cmd_vel_topic

        self.mode = args.mode
        self.map_save_path = args.map_save
        self.map_load_path = args.map_load
        self.use_premap_in_solver = args.use_premap and os.path.exists(self.map_load_path)

        self.max_lin = args.max_linear_speed
        self.max_ang = args.max_angular_speed
        self.corridor_speed = args.corridor_speed
        self.fast_speed = args.fast_speed
        self.wall_clearance = args.wall_clearance
        self.front_stop = args.front_stop_distance
        self.dead_end_front = args.dead_end_front_distance
        self.junction_front = args.junction_front_distance
        self.side_open = args.side_open_distance
        self.pose_quant = args.pose_quantization
        self.goal_tol = args.goal_tolerance

        self.scan: Optional[LaserScan] = None
        self.odom: Optional[Odometry] = None
        self.map_msg: Optional[OccupancyGrid] = None
        self.map_np: Optional[np.ndarray] = None
        self.map_resolution: Optional[float] = None
        self.map_origin_xy: Optional[Tuple[float, float]] = None
        self.pose_xy = None
        self.yaw = 0.0
        self.start_pose = None

        self.path_xy: List[Tuple[float, float]] = []
        self.travel_distance = 0.0
        self.last_pose_for_distance = None
        self.blind_state = 'EXPLORE'
        self.turn_target_yaw = None
        self.current_cell = None
        self.junctions: Dict[Tuple[int, int, int], JunctionState] = {}
        self.edge_dead: Set[Tuple[Tuple[int, int, int], str]] = set()
        self.edge_visited: Set[Tuple[Tuple[int, int, int], str]] = set()
        self.backtrack_stack: List[Tuple[Tuple[int, int, int], str]] = []
        self.finish_detected = False
        self.finish_turning = False
        self.final_turn_start = None
        self.best_cell = None
        self.best_travel_distance = 0.0

        self.global_waypoints: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.nav_done = False
        self.goal_cell = None

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, map_qos)

        self.control_timer = self.create_timer(0.10, self.control_loop)
        self.display_timer = self.create_timer(0.25, self.update_display)
        self.save_timer = self.create_timer(2.0, self.periodic_save)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.get_logger().info(f"mode={self.mode}, use_premap_in_solver={self.use_premap_in_solver}")

    # --------------------------- ROS callbacks ---------------------------
    def scan_cb(self, msg: LaserScan):
        self.scan = msg

    def odom_cb(self, msg: Odometry):
        self.odom = msg
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose_xy = (p.x, p.y)
        self.yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
        if self.start_pose is None:
            self.start_pose = (p.x, p.y, self.yaw)
            self.get_logger().info(f"Start pose recorded at x={p.x:.2f}, y={p.y:.2f}, yaw={self.yaw:.2f}")
        if self.last_pose_for_distance is not None:
            dx = p.x - self.last_pose_for_distance[0]
            dy = p.y - self.last_pose_for_distance[1]
            self.travel_distance += math.hypot(dx, dy)
        self.last_pose_for_distance = (p.x, p.y)
        self.path_xy.append((p.x, p.y))
        if len(self.path_xy) > 6000:
            self.path_xy = self.path_xy[-6000:]

    def map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg
        w = msg.info.width
        h = msg.info.height
        self.map_np = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        self.map_resolution = float(msg.info.resolution)
        self.map_origin_xy = (msg.info.origin.position.x, msg.info.origin.position.y)

    # --------------------------- control loop ---------------------------
    def control_loop(self):
        if self.pose_xy is None or self.scan is None:
            return

        if self.mode == 'premap':
            return

        if self.mode == 'solve':
            if self.use_premap_in_solver and not self.global_waypoints:
                self.try_plan_from_saved_map()
            if self.global_waypoints:
                self.follow_global_path()
            else:
                self.blind_dfs_explore()

    # ----------------------- premap / persistence -----------------------
    def periodic_save(self):
        if self.map_np is None:
            return
        if self.mode == 'premap' or (self.mode == 'solve' and self.args.save_during_solve):
            self.save_current_map(self.map_save_path)

    def save_current_map(self, path: str):
        if self.map_np is None or self.map_resolution is None or self.map_origin_xy is None:
            return
        np.savez_compressed(
            path,
            map=self.map_np,
            resolution=self.map_resolution,
            origin=np.array(self.map_origin_xy, dtype=np.float64),
        )

    def load_saved_map(self, path: str):
        pack = np.load(path, allow_pickle=False)
        grid = pack['map']
        res = float(pack['resolution'])
        origin = tuple(pack['origin'].tolist())
        return grid, res, origin

    # ------------------------ mapped-run planning -----------------------
    def try_plan_from_saved_map(self):
        if self.start_pose is None:
            return
        try:
            grid, res, origin = self.load_saved_map(self.map_load_path)
        except Exception as exc:
            self.get_logger().warn(f"Failed to load saved map: {exc}")
            return

        start_cell = self.world_to_grid((self.pose_xy[0], self.pose_xy[1]), origin, res, grid.shape)
        if start_cell is None:
            self.get_logger().warn('Current pose is outside loaded map, falling back to blind mode.')
            self.use_premap_in_solver = False
            return

        free_grid = np.array(grid)
        # Inflate obstacles conservatively for TB4 footprint.
        inflated = self.inflate_obstacles(free_grid, radius_cells=max(1, int(0.18 / res)))
        goal_cell, path_cells = self.find_farthest_reachable_cell(inflated, start_cell)
        if goal_cell is None or not path_cells:
            self.get_logger().warn('Could not find a reachable goal in saved map, falling back to blind mode.')
            self.use_premap_in_solver = False
            return

        self.goal_cell = goal_cell
        raw_waypoints = [self.grid_to_world(c, origin, res) for c in path_cells]
        self.global_waypoints = self.compress_waypoints(raw_waypoints, keep_every=max(1, int(0.18 / res)))
        self.current_waypoint_idx = 0
        self.get_logger().info(f"Loaded map plan with {len(self.global_waypoints)} waypoints to farthest cell {goal_cell}.")

    def inflate_obstacles(self, grid: np.ndarray, radius_cells: int) -> np.ndarray:
        inflated = np.array(grid, copy=True)
        occ = np.argwhere(grid > 50)
        if len(occ) == 0:
            return inflated
        h, w = grid.shape
        for r, c in occ:
            r0 = max(0, r - radius_cells)
            r1 = min(h, r + radius_cells + 1)
            c0 = max(0, c - radius_cells)
            c1 = min(w, c + radius_cells + 1)
            inflated[r0:r1, c0:c1] = np.maximum(inflated[r0:r1, c0:c1], 100)
        return inflated

    def find_farthest_reachable_cell(self, grid: np.ndarray, start: Tuple[int, int]):
        h, w = grid.shape
        if not self.is_free_cell(grid, start):
            nearby = self.find_nearest_free_cell(grid, start, max_radius=8)
            if nearby is None:
                return None, []
            start = nearby

        from collections import deque
        q = deque([start])
        parent = {start: None}
        dist = {start: 0}
        best = start
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            cur = q.popleft()
            if dist[cur] > dist[best]:
                best = cur
            for dr, dc in nbrs:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in parent:
                    continue
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and self.is_free_cell(grid, nxt):
                    parent[nxt] = cur
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        path = []
        cur = best
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return best, path

    def is_free_cell(self, grid: np.ndarray, cell: Tuple[int, int]) -> bool:
        r, c = cell
        val = int(grid[r, c])
        return 0 <= val < 50

    def find_nearest_free_cell(self, grid: np.ndarray, start: Tuple[int, int], max_radius: int = 10):
        h, w = grid.shape
        sr, sc = start
        for rad in range(1, max_radius + 1):
            for r in range(max(0, sr - rad), min(h, sr + rad + 1)):
                for c in range(max(0, sc - rad), min(w, sc + rad + 1)):
                    if self.is_free_cell(grid, (r, c)):
                        return (r, c)
        return None

    def compress_waypoints(self, pts: List[Tuple[float, float]], keep_every: int = 4):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        for i in range(1, len(pts) - 1):
            prev = np.array(pts[i - 1])
            cur = np.array(pts[i])
            nxt = np.array(pts[i + 1])
            v1 = cur - prev
            v2 = nxt - cur
            turn = np.linalg.norm(v1 - v2)
            if i % keep_every == 0 or turn > 1e-6:
                out.append(pts[i])
        out.append(pts[-1])
        return out

    def follow_global_path(self):
        if self.pose_xy is None or not self.global_waypoints:
            return
        if self.current_waypoint_idx >= len(self.global_waypoints):
            self.finish_behavior()
            return

        fx, leftx, rightx = self.scan_openings()
        if fx < self.front_stop * 0.7:
            self.publish_cmd(0.0, 0.6)
            return

        wx, wy = self.global_waypoints[self.current_waypoint_idx]
        dx = wx - self.pose_xy[0]
        dy = wy - self.pose_xy[1]
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        err = self.wrap_to_pi(desired - self.yaw)

        if dist < self.goal_tol:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.global_waypoints):
                self.get_logger().info('Reached farthest map goal, executing finish turn.')
            return

        lin = min(self.fast_speed if abs(err) < 0.25 else 0.0, self.max_lin)
        if fx < self.front_stop:
            lin = 0.0
        ang = np.clip(2.2 * err, -self.max_ang, self.max_ang)
        self.publish_cmd(lin, ang)

    # -------------------------- blind DFS mode --------------------------
    def blind_dfs_explore(self):
        pose_key = self.quantized_pose_key(self.pose_xy[0], self.pose_xy[1], self.yaw)
        self.current_cell = pose_key
        if self.best_cell is None or self.travel_distance > self.best_travel_distance:
            self.best_cell = pose_key
            self.best_travel_distance = self.travel_distance

        front, left, right = self.scan_openings()
        branches = self.available_branches(front, left, right)
        at_dead_end = (front < self.dead_end_front) and ('left' not in branches) and ('right' not in branches)
        at_junction = len(branches) >= 2 or (front > self.junction_front and ('left' in branches or 'right' in branches))

        if at_dead_end:
            self.edge_dead.add((pose_key, 'front'))
            self.turn_target_yaw = self.wrap_to_pi(self.yaw + math.pi)
            self.blind_state = 'TURNING'
            self.publish_cmd(0.0, 0.0)
            return

        if at_junction:
            if pose_key not in self.junctions:
                parent_branch = self.reverse_branch(self.backtrack_stack[-1][1]) if self.backtrack_stack else None
                self.junctions[pose_key] = JunctionState(branches=branches, tried=set(), parent_branch=parent_branch)
            choice = self.choose_dfs_branch(pose_key, branches)
            if choice is not None:
                self.junctions[pose_key].tried.add(choice)
                self.edge_visited.add((pose_key, choice))
                self.backtrack_stack.append((pose_key, choice))
                self.turn_target_yaw = self.branch_target_yaw(choice)
                self.blind_state = 'TURNING'
                self.publish_cmd(0.0, 0.0)
                return
            else:
                self.turn_target_yaw = self.wrap_to_pi(self.yaw + math.pi)
                self.blind_state = 'TURNING'
                self.publish_cmd(0.0, 0.0)
                return

        self.follow_corridor(front, left, right)

    def choose_dfs_branch(self, pose_key, branches: List[str]) -> Optional[str]:
        state = self.junctions[pose_key]
        ordered = ['front', 'left', 'right']
        filtered = [b for b in ordered if b in branches]
        if state.parent_branch in filtered and len(filtered) > 1:
            filtered = [b for b in filtered if b != state.parent_branch]
        for b in filtered:
            if b not in state.tried and (pose_key, b) not in self.edge_dead:
                return b
        return None

    def follow_corridor(self, front, left, right):
        if self.blind_state == 'TURNING' and self.turn_target_yaw is not None:
            err = self.wrap_to_pi(self.turn_target_yaw - self.yaw)
            if abs(err) < 0.08:
                self.blind_state = 'EXPLORE'
                self.turn_target_yaw = None
                self.publish_cmd(0.0, 0.0)
            else:
                self.publish_cmd(0.0, np.clip(2.5 * err, -self.max_ang, self.max_ang))
            return

        if front < self.front_stop:
            if left > right:
                self.publish_cmd(0.0, self.max_ang * 0.7)
            else:
                self.publish_cmd(0.0, -self.max_ang * 0.7)
            return

        wall_error = (self.wall_clearance - left) - (self.wall_clearance - right)
        ang = np.clip(-1.3 * wall_error, -self.max_ang, self.max_ang)
        speed_scale = np.clip((front - self.front_stop) / max(0.01, (1.4 - self.front_stop)), 0.2, 1.0)
        lin = min(self.corridor_speed * speed_scale, self.max_lin)
        self.publish_cmd(lin, ang)

        if self.travel_distance > 1.5 and self.is_exit_condition(front, left, right):
            self.get_logger().info('Likely exit detected from blind solver. Executing finish behavior.')
            self.finish_detected = True
            self.finish_behavior()

    def is_exit_condition(self, front, left, right):
        return front > 2.5 and (left > 1.5 or right > 1.5)

    def finish_behavior(self):
        if not self.finish_turning:
            self.finish_turning = True
            self.final_turn_start = self.yaw
            self.turn_target_yaw = self.wrap_to_pi(self.yaw + math.pi)
        err = self.wrap_to_pi(self.turn_target_yaw - self.yaw)
        if abs(err) < 0.08:
            self.publish_cmd(0.0, 0.0)
            self.nav_done = True
            return
        self.publish_cmd(0.0, np.clip(2.5 * err, -self.max_ang, self.max_ang))

    # --------------------------- visualization --------------------------
    def update_display(self):
        if self.ax is None:
            return
        self.ax.clear()

        if self.map_np is not None and self.map_resolution is not None and self.map_origin_xy is not None:
            origin_x, origin_y = self.map_origin_xy
            h, w = self.map_np.shape
            extent = [origin_x, origin_x + w * self.map_resolution,
                      origin_y, origin_y + h * self.map_resolution]
            # display unknown gray, free white, occ black
            disp = np.array(self.map_np, dtype=np.float32)
            disp[disp < 0] = 50
            self.ax.imshow(disp, origin='lower', extent=extent, cmap='gray_r', alpha=0.85)

        if self.path_xy:
            xs, ys = zip(*self.path_xy)
            self.ax.plot(xs, ys, 'b-', linewidth=1.5, label='robot path')

        if self.global_waypoints:
            wx, wy = zip(*self.global_waypoints)
            self.ax.plot(wx, wy, 'g--', linewidth=1.0, label='planned path')
            if self.current_waypoint_idx < len(self.global_waypoints):
                gx, gy = self.global_waypoints[self.current_waypoint_idx]
                self.ax.plot([gx], [gy], 'go', markersize=8, label='current waypoint')

        if self.pose_xy is not None:
            x, y = self.pose_xy
            self.ax.plot([x], [y], 'ro', markersize=8, label='robot')
            self.ax.arrow(x, y, 0.18 * math.cos(self.yaw), 0.18 * math.sin(self.yaw),
                          head_width=0.07, head_length=0.08, color='r')

        title = f"mode={self.mode} | travel={self.travel_distance:.2f} m"
        if self.best_travel_distance > 0.0:
            title += f" | best DFS depth={self.best_travel_distance:.2f} m"
        self.ax.set_title(title)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.legend(loc='upper right')
        self.ax.axis('equal')
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    # ---------------------------- utilities ----------------------------
    def publish_cmd(self, linear: float, angular: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(np.clip(linear, -self.max_lin, self.max_lin))
        msg.twist.angular.z = float(np.clip(angular, -self.max_ang, self.max_ang))
        self.cmd_pub.publish(msg)

    def scan_openings(self):
        if self.scan is None:
            return 0.0, 0.0, 0.0
        front = self.min_range_in_sector(-15.0, 15.0)
        left = self.min_range_in_sector(55.0, 100.0)
        right = self.min_range_in_sector(-100.0, -55.0)
        return front, left, right

    def available_branches(self, front, left, right):
        branches = []
        if front > self.junction_front:
            branches.append('front')
        if left > self.side_open:
            branches.append('left')
        if right > self.side_open:
            branches.append('right')
        return branches

    def min_range_in_sector(self, start_deg: float, end_deg: float):
        if self.scan is None:
            return float('inf')
        ranges = np.asarray(self.scan.ranges, dtype=np.float32)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment
        s = math.radians(start_deg)
        e = math.radians(end_deg)
        mask = (angles >= s) & (angles <= e) & np.isfinite(ranges)
        valid = ranges[mask]
        valid = valid[(valid >= self.scan.range_min) & (valid <= self.scan.range_max)]
        if valid.size == 0:
            return float('inf')
        return float(np.min(valid))

    def branch_target_yaw(self, branch: str):
        if branch == 'left':
            return self.wrap_to_pi(self.yaw + math.pi / 2.0)
        if branch == 'right':
            return self.wrap_to_pi(self.yaw - math.pi / 2.0)
        return self.yaw

    def reverse_branch(self, branch: str):
        return {'left': 'right', 'right': 'left', 'front': 'front'}.get(branch, 'front')

    def quantized_pose_key(self, x, y, yaw):
        heading_bin = int(round(self.wrap_to_pi(yaw) / (math.pi / 2.0))) % 4
        return (int(round(x / self.pose_quant)), int(round(y / self.pose_quant)), heading_bin)

    def world_to_grid(self, xy, origin, res, shape):
        x, y = xy
        col = int(round((x - origin[0]) / res))
        row = int(round((y - origin[1]) / res))
        h, w = shape
        if 0 <= row < h and 0 <= col < w:
            return (row, col)
        return None

    def grid_to_world(self, rc, origin, res):
        r, c = rc
        x = origin[0] + c * res
        y = origin[1] + r * res
        return (x, y)

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_to_pi(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi



def build_arg_parser():
    p = argparse.ArgumentParser(description='TurtleBot maze premapper and solver using LiDAR, SLAM map, and DFS exploration.')
    p.add_argument('--mode', choices=['premap', 'solve'], default='solve',
                   help='premap: display/save SLAM map while another terminal teleops; solve: autonomous maze run.')
    p.add_argument('--use-premap', action='store_true',
                   help='In solve mode, load a previously saved map and plan to the farthest reachable free cell before falling back to blind DFS.')
    p.add_argument('--map-save', default='premap_maze_map.npz', help='Path to save the internal copy of the SLAM map.')
    p.add_argument('--map-load', default='premap_maze_map.npz', help='Path to previously saved map for solve mode.')
    p.add_argument('--save-during-solve', action='store_true', help='Also save / update map snapshots while solving.')

    p.add_argument('--scan-topic', default='/tb4_4/scan')
    p.add_argument('--odom-topic', default='/tb4_4/odom')
    p.add_argument('--map-topic', default='/map')
    p.add_argument('--cmd-vel-topic', default='/tb4_4/cmd_vel')

    p.add_argument('--max-linear-speed', type=float, default=0.30)
    p.add_argument('--max-angular-speed', type=float, default=1.20)
    p.add_argument('--corridor-speed', type=float, default=0.22)
    p.add_argument('--fast-speed', type=float, default=0.28)
    p.add_argument('--wall-clearance', type=float, default=0.45)
    p.add_argument('--front-stop-distance', type=float, default=0.38)
    p.add_argument('--dead-end-front-distance', type=float, default=0.32)
    p.add_argument('--junction-front-distance', type=float, default=0.70)
    p.add_argument('--side-open-distance', type=float, default=0.72)
    p.add_argument('--pose-quantization', type=float, default=0.35)
    p.add_argument('--goal-tolerance', type=float, default=0.16)
    return p


def strip_ros_args(argv):
    out = []
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == '--ros-args':
            continue
        if a in ('-r', '--remap', '-p', '--params-file'):
            skip = True
            continue
        out.append(a)
    return out


def main(argv=None):
    import sys
    raw = sys.argv[1:] if argv is None else argv
    cli_args = strip_ros_args(raw)
    parser = build_arg_parser()
    args = parser.parse_args(cli_args)

    rclpy.init(args=None)
    node = MazeMapperSolver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        if node.map_np is not None:
            node.save_current_map(node.map_save_path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#python3 mazesolver.py --mode premap --map-save premap_maze_map.npz
#python3 mazesolver.py --mode solve --use-premap --map-load premap_maze_map.npz
#python3 mazesolver.py --mode solve