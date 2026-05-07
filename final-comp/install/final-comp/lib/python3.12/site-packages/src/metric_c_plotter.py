#!/usr/bin/env python3

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def clean_namespace(ns: str) -> str:
    ns = ns.strip()
    if ns == "" or ns == "/":
        return ""
    if not ns.startswith("/"):
        ns = "/" + ns
    return ns.rstrip("/")


def yaw_from_quaternion(q) -> float:
    """
    Convert geometry_msgs/Quaternion to yaw angle.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_nanoseconds(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def open_bag_reader(bag_path: str, storage_id: str):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def get_topic_type_map(reader):
    topic_types = reader.get_all_topics_and_types()
    return {topic.name: topic.type for topic in topic_types}


def interpolate_pose(odom_times, odom_poses, t):
    """
    Interpolate robot pose at timestamp t.

    odom_times: np array of timestamps in ns
    odom_poses: np array of [x, y, yaw]
    t: scan timestamp in ns
    """
    if len(odom_times) == 0:
        return None

    if t <= odom_times[0]:
        return odom_poses[0]

    if t >= odom_times[-1]:
        return odom_poses[-1]

    idx = np.searchsorted(odom_times, t)

    t0 = odom_times[idx - 1]
    t1 = odom_times[idx]
    p0 = odom_poses[idx - 1]
    p1 = odom_poses[idx]

    if t1 == t0:
        return p0

    alpha = (t - t0) / float(t1 - t0)

    x = (1.0 - alpha) * p0[0] + alpha * p1[0]
    y = (1.0 - alpha) * p0[1] + alpha * p1[1]

    # Interpolate yaw through shortest angular distance.
    yaw0 = p0[2]
    yaw1 = p1[2]
    dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
    yaw = yaw0 + alpha * dyaw

    return np.array([x, y, yaw])


def read_bag_data(bag_path, storage_id, odom_topic, scan_topic):
    reader = open_bag_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    missing = []
    for topic in [odom_topic, scan_topic]:
        if topic not in topic_type_map:
            missing.append(topic)

    if missing:
        print("[metric_c_plotter] ERROR: Missing required topics:")
        for topic in missing:
            print(f"  {topic}")

        print("\n[metric_c_plotter] Available topics in bag:")
        for topic in sorted(topic_type_map.keys()):
            print(f"  {topic}")
        sys.exit(1)

    msg_type_cache = {
        topic: get_message(type_name)
        for topic, type_name in topic_type_map.items()
    }

    odom_records = []
    scan_records = []

    print("[metric_c_plotter] Reading bag...")

    while reader.has_next():
        topic, data, bag_timestamp = reader.read_next()

        if topic not in [odom_topic, scan_topic]:
            continue

        msg_type = msg_type_cache[topic]
        msg = deserialize_message(data, msg_type)

        if topic == odom_topic:
            t = stamp_to_nanoseconds(msg)
            if t == 0:
                t = bag_timestamp

            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            yaw = yaw_from_quaternion(ori)

            odom_records.append((t, float(pos.x), float(pos.y), float(yaw)))

        elif topic == scan_topic:
            t = stamp_to_nanoseconds(msg)
            if t == 0:
                t = bag_timestamp

            scan_records.append((t, msg))

    print(f"[metric_c_plotter] Odom messages: {len(odom_records)}")
    print(f"[metric_c_plotter] Scan messages: {len(scan_records)}")

    if len(odom_records) == 0:
        raise RuntimeError(f"No odom messages found on {odom_topic}")

    if len(scan_records) == 0:
        raise RuntimeError(f"No scan messages found on {scan_topic}")

    odom_records.sort(key=lambda r: r[0])
    scan_records.sort(key=lambda r: r[0])

    odom_times = np.array([r[0] for r in odom_records], dtype=np.int64)
    odom_poses = np.array([[r[1], r[2], r[3]] for r in odom_records], dtype=float)

    return odom_times, odom_poses, scan_records


def bresenham_cells(x0, y0, x1, y1):
    """
    Integer-grid Bresenham line from cell 0 to cell 1.
    """
    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    x, y = x0, y0

    while True:
        cells.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return cells


def build_occupancy_grid(
    odom_times,
    odom_poses,
    scan_records,
    resolution,
    max_range,
    scan_stride,
    ray_stride,
    laser_x_offset,
    laser_y_offset,
):
    """
    Build a simple log-odds occupancy grid using odom + LaserScan.

    Occupied cells:
      Laser endpoints

    Free cells:
      Ray cells before the endpoint

    This assumes the LaserScan frame is approximately aligned with base_link.
    For TurtleBot 4, this is good enough for a report plot, but not a replacement
    for full SLAM.
    """
    occupied_points = []
    free_segments = []

    print("[metric_c_plotter] Projecting scans into odom frame...")

    for scan_idx, (scan_t, scan_msg) in enumerate(scan_records):
        if scan_idx % scan_stride != 0:
            continue

        pose = interpolate_pose(odom_times, odom_poses, scan_t)
        if pose is None:
            continue

        robot_x, robot_y, robot_yaw = pose

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        # Approximate laser origin in odom frame.
        laser_origin_x = robot_x + cos_yaw * laser_x_offset - sin_yaw * laser_y_offset
        laser_origin_y = robot_y + sin_yaw * laser_x_offset + cos_yaw * laser_y_offset

        angle = scan_msg.angle_min

        for i, r in enumerate(scan_msg.ranges):
            if i % ray_stride != 0:
                angle += scan_msg.angle_increment
                continue

            valid = math.isfinite(r)
            if not valid:
                angle += scan_msg.angle_increment
                continue

            if r < scan_msg.range_min:
                angle += scan_msg.angle_increment
                continue

            clipped = False
            if r > scan_msg.range_max:
                angle += scan_msg.angle_increment
                continue

            if max_range > 0.0 and r > max_range:
                r = max_range
                clipped = True

            global_angle = robot_yaw + angle

            end_x = laser_origin_x + r * math.cos(global_angle)
            end_y = laser_origin_y + r * math.sin(global_angle)

            free_segments.append((laser_origin_x, laser_origin_y, end_x, end_y))

            # Only mark occupied if this was a real return, not an artificial max-range endpoint.
            if not clipped:
                occupied_points.append((end_x, end_y))

            angle += scan_msg.angle_increment

    path_points = odom_poses[:, 0:2]

    all_points = []

    if occupied_points:
        all_points.append(np.array(occupied_points))

    all_points.append(path_points)

    all_xy = np.vstack(all_points)

    min_x = float(np.min(all_xy[:, 0])) - 0.5
    max_x = float(np.max(all_xy[:, 0])) + 0.5
    min_y = float(np.min(all_xy[:, 1])) - 0.5
    max_y = float(np.max(all_xy[:, 1])) + 0.5

    width = max(1, int(math.ceil((max_x - min_x) / resolution)))
    height = max(1, int(math.ceil((max_y - min_y) / resolution)))

    log_odds = np.zeros((height, width), dtype=np.float32)

    def world_to_cell(x, y):
        cx = int((x - min_x) / resolution)
        cy = int((y - min_y) / resolution)
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))
        return cx, cy

    free_update = -0.35
    occupied_update = 0.85

    print("[metric_c_plotter] Building occupancy grid...")

    for x0, y0, x1, y1 in free_segments:
        c0x, c0y = world_to_cell(x0, y0)
        c1x, c1y = world_to_cell(x1, y1)

        cells = bresenham_cells(c0x, c0y, c1x, c1y)

        # Mark free cells before the endpoint.
        for cx, cy in cells[:-1]:
            log_odds[cy, cx] += free_update

    for x, y in occupied_points:
        cx, cy = world_to_cell(x, y)
        log_odds[cy, cx] += occupied_update

    log_odds = np.clip(log_odds, -4.0, 4.0)

    probability = 1.0 - 1.0 / (1.0 + np.exp(log_odds))

    extent = [min_x, max_x, min_y, max_y]

    return probability, extent, path_points, np.array(occupied_points)


def save_outputs(
    probability,
    extent,
    path_points,
    occupied_points,
    output_prefix,
    title,
    show_points,
):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    png_path = str(output_prefix) + ".png"
    csv_path = str(output_prefix) + "_path.csv"

    print("[metric_c_plotter] Saving path CSV...")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("x_m,y_m\n")
        for x, y in path_points:
            f.write(f"{x:.6f},{y:.6f}\n")

    print("[metric_c_plotter] Saving plot...")

    plt.figure(figsize=(9, 9))

    plt.imshow(
        probability,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    if show_points and len(occupied_points) > 0:
        plt.scatter(
            occupied_points[:, 0],
            occupied_points[:, 1],
            s=1,
            alpha=0.15,
            label="LiDAR occupied endpoints",
        )

    plt.plot(
        path_points[:, 0],
        path_points[:, 1],
        linewidth=2.0,
        label="Robot path from odom",
    )

    plt.scatter(
        path_points[0, 0],
        path_points[0, 1],
        s=80,
        marker="o",
        label="Start",
    )

    plt.scatter(
        path_points[-1, 0],
        path_points[-1, 1],
        s=80,
        marker="x",
        label="End",
    )

    plt.title(title)
    plt.xlabel("x position in odom frame [m]")
    plt.ylabel("y position in odom frame [m]")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"[metric_c_plotter] Saved plot: {png_path}")
    print(f"[metric_c_plotter] Saved path CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Metric C top-down map and robot path plot from a ROS 2 bag."
    )

    parser.add_argument(
        "bag",
        help="Path to the ROS 2 bag directory, for example metric_c_bags/metric_c_run_tb4_4_20260507_150000",
    )

    parser.add_argument(
        "--namespace",
        "-n",
        default="",
        help="Robot namespace, for example tb4_4 or /tb4_4. Leave blank for non-namespaced topics.",
    )

    parser.add_argument(
        "--storage-id",
        default="mcap",
        choices=["mcap", "sqlite3"],
        help="Bag storage backend. Metric C should use mcap.",
    )

    parser.add_argument(
        "--odom-topic",
        default=None,
        help="Override odom topic. Default is /<namespace>/odom or /odom.",
    )

    parser.add_argument(
        "--scan-topic",
        default=None,
        help="Override scan topic. Default is /<namespace>/scan or /scan.",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="metric_c_outputs/top_down_map",
        help="Output prefix. Script writes <output>.png and <output>_path.csv.",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.03,
        help="Occupancy grid resolution in meters per cell. Default: 0.03.",
    )

    parser.add_argument(
        "--max-range",
        type=float,
        default=3.5,
        help="Maximum LiDAR range to use in meters. Use 0 for scan max range.",
    )

    parser.add_argument(
        "--scan-stride",
        type=int,
        default=2,
        help="Use every Nth scan to reduce processing time. Default: 2.",
    )

    parser.add_argument(
        "--ray-stride",
        type=int,
        default=3,
        help="Use every Nth LiDAR ray to reduce processing time. Default: 3.",
    )

    parser.add_argument(
        "--laser-x-offset",
        type=float,
        default=0.0,
        help="Approximate laser x offset from base_link in meters.",
    )

    parser.add_argument(
        "--laser-y-offset",
        type=float,
        default=0.0,
        help="Approximate laser y offset from base_link in meters.",
    )

    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Overlay raw occupied LiDAR endpoints on the plot.",
    )

    args = parser.parse_args()

    ns = clean_namespace(args.namespace)

    odom_topic = args.odom_topic
    scan_topic = args.scan_topic

    if odom_topic is None:
        odom_topic = f"{ns}/odom" if ns else "/odom"

    if scan_topic is None:
        scan_topic = f"{ns}/scan" if ns else "/scan"

    bag_path = str(Path(args.bag).expanduser().resolve())

    if not os.path.exists(bag_path):
        print(f"[metric_c_plotter] ERROR: Bag path does not exist: {bag_path}")
        return 1

    print("[metric_c_plotter] Bag:", bag_path)
    print("[metric_c_plotter] Odom topic:", odom_topic)
    print("[metric_c_plotter] Scan topic:", scan_topic)

    odom_times, odom_poses, scan_records = read_bag_data(
        bag_path=bag_path,
        storage_id=args.storage_id,
        odom_topic=odom_topic,
        scan_topic=scan_topic,
    )

    probability, extent, path_points, occupied_points = build_occupancy_grid(
        odom_times=odom_times,
        odom_poses=odom_poses,
        scan_records=scan_records,
        resolution=args.resolution,
        max_range=args.max_range,
        scan_stride=max(1, args.scan_stride),
        ray_stride=max(1, args.ray_stride),
        laser_x_offset=args.laser_x_offset,
        laser_y_offset=args.laser_y_offset,
    )

    title = "Metric C Top-Down LiDAR Map and Robot Path"

    save_outputs(
        probability=probability,
        extent=extent,
        path_points=path_points,
        occupied_points=occupied_points,
        output_prefix=args.output,
        title=title,
        show_points=args.show_points,
    )

    print("[metric_c_plotter] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

'''ros2 run final-comp metric_c_plotter \
  metric_c_bags/metric_c_run_tb4_4_YYYYMMDD_HHMMSS \
  --namespace /tb4_4 \
  --output metric_c_outputs/tb4_4_top_down_map'''