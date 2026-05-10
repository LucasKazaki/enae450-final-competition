#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions, SequentialReader

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def yaw_from_quaternion(q):
    """
    Convert geometry_msgs/Quaternion to yaw.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def get_storage_id(bag_dir: Path):
    metadata = bag_dir / "metadata.yaml"

    if not metadata.exists():
        # Most official bags for the competition should be MCAP, but sqlite3 is possible.
        if list(bag_dir.glob("*.mcap")):
            return "mcap"
        if list(bag_dir.glob("*.db3")):
            return "sqlite3"
        raise RuntimeError("Could not determine bag storage type. No metadata.yaml, .mcap, or .db3 found.")

    text = metadata.read_text(errors="ignore")

    if "storage_identifier: mcap" in text:
        return "mcap"
    if "storage_identifier: sqlite3" in text:
        return "sqlite3"

    if list(bag_dir.glob("*.mcap")):
        return "mcap"
    if list(bag_dir.glob("*.db3")):
        return "sqlite3"

    raise RuntimeError("Could not determine storage type from metadata.yaml.")


def open_reader(bag_dir: Path):
    storage_id = get_storage_id(bag_dir)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    return reader


def topic_type_map(reader):
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def auto_pick_topic(all_topics, preferred_suffixes):
    """
    Picks namespaced or non-namespaced topics.
    Example: preferred_suffixes=["/odom", "odom"] can pick /tb4_6/odom or /odom.
    """
    for suffix in preferred_suffixes:
        matches = [t for t in all_topics if t == suffix or t.endswith(suffix)]
        if matches:
            # Prefer the longest namespaced match if there are several.
            return sorted(matches, key=len, reverse=True)[0]
    return None


def nearest_index(times, target):
    """
    Binary-search-ish nearest index without numpy dependency.
    """
    if not times:
        return None

    lo, hi = 0, len(times) - 1

    if target <= times[lo]:
        return lo
    if target >= times[hi]:
        return hi

    while lo <= hi:
        mid = (lo + hi) // 2
        if times[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    before = lo - 1
    after = lo

    if abs(times[before] - target) <= abs(times[after] - target):
        return before
    return after


def load_bag(bag_dir, odom_topic=None, scan_topic=None, cmd_topic=None, max_scans=1200):
    reader = open_reader(bag_dir)
    types = topic_type_map(reader)
    all_topics = list(types.keys())

    if odom_topic is None:
        odom_topic = auto_pick_topic(all_topics, ["/odom", "odom"])
    if scan_topic is None:
        scan_topic = auto_pick_topic(all_topics, ["/scan", "scan"])
    if cmd_topic is None:
        cmd_topic = auto_pick_topic(all_topics, ["/cmd_vel", "cmd_vel"])

    print("\nDetected topics:")
    print(f"  odom:    {odom_topic}")
    print(f"  scan:    {scan_topic}")
    print(f"  cmd_vel: {cmd_topic}")

    if odom_topic is None:
        raise RuntimeError("Could not find an odom topic. Pass --odom-topic manually.")
    if scan_topic is None:
        print("[WARN] Could not find scan topic. The scrubber will show path only.")

    msg_classes = {}
    for topic, msg_type in types.items():
        try:
            msg_classes[topic] = get_message(msg_type)
        except Exception:
            pass

    odom = []
    scans = []
    cmds = []

    first_ns = None
    last_ns = None

    scan_keep_every = 1
    scan_seen = 0

    # First pass count scans so we can downsample if needed.
    scan_count_reader = open_reader(bag_dir)
    total_scan_count = 0
    while scan_count_reader.has_next():
        topic, _, t = scan_count_reader.read_next()
        if first_ns is None:
            first_ns = t
        last_ns = t
        if topic == scan_topic:
            total_scan_count += 1

    if total_scan_count > max_scans:
        scan_keep_every = math.ceil(total_scan_count / max_scans)

    reader = open_reader(bag_dir)

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if first_ns is None:
            first_ns = t_ns
        last_ns = t_ns

        rel_t = (t_ns - first_ns) / 1e9

        if topic not in msg_classes:
            continue

        try:
            msg = deserialize_message(data, msg_classes[topic])
        except Exception:
            continue

        if topic == odom_topic:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = yaw_from_quaternion(q)
            odom.append((rel_t, p.x, p.y, yaw))

        elif topic == scan_topic:
            scan_seen += 1
            if scan_seen % scan_keep_every != 0:
                continue

            # Store compact scan data.
            scans.append((
                rel_t,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                list(msg.ranges),
            ))

        elif topic == cmd_topic:
            # Support Twist and TwistStamped.
            if hasattr(msg, "twist"):
                tw = msg.twist
            else:
                tw = msg

            lin = getattr(tw.linear, "x", 0.0)
            ang = getattr(tw.angular, "z", 0.0)
            cmds.append((rel_t, lin, ang))

    if not odom:
        raise RuntimeError("No odom messages were loaded.")

    duration = (last_ns - first_ns) / 1e9 if first_ns is not None and last_ns is not None else 0.0

    print("\nLoaded bag data:")
    print(f"  duration: {duration:.3f} sec")
    print(f"  odom messages: {len(odom)}")
    print(f"  scan messages shown: {len(scans)} of {total_scan_count}")
    print(f"  cmd_vel messages: {len(cmds)}")

    return {
        "duration": duration,
        "odom": odom,
        "scans": scans,
        "cmds": cmds,
        "topics": {
            "odom": odom_topic,
            "scan": scan_topic,
            "cmd_vel": cmd_topic,
        },
    }


def scan_points_in_odom(scan, pose, max_points=720):
    """
    Transform a LaserScan into odom coordinates using nearest odom pose.
    This assumes scan frame is approximately aligned with base_link.
    For this visual crop-selection tool, that approximation is usually good enough.
    """
    if scan is None or pose is None:
        return [], []

    _, angle_min, angle_inc, range_min, range_max, ranges = scan
    _, x, y, yaw = pose

    xs = []
    ys = []

    step = max(1, len(ranges) // max_points)

    for i in range(0, len(ranges), step):
        r = ranges[i]

        if r is None:
            continue
        if math.isnan(r) or math.isinf(r):
            continue
        if r < range_min or r > range_max:
            continue

        a = angle_min + i * angle_inc
        global_a = yaw + a

        xs.append(x + r * math.cos(global_a))
        ys.append(y + r * math.sin(global_a))

    return xs, ys


def plot_scrubber(data):
    odom = data["odom"]
    scans = data["scans"]
    cmds = data["cmds"]
    duration = data["duration"]

    odom_t = [o[0] for o in odom]
    odom_x = [o[1] for o in odom]
    odom_y = [o[2] for o in odom]

    scan_t = [s[0] for s in scans]

    cmd_t = [c[0] for c in cmds]
    cmd_lin = [c[1] for c in cmds]
    cmd_ang = [c[2] for c in cmds]

    crop_start = {"value": None}
    crop_end = {"value": None}

    fig = plt.figure(figsize=(12, 8))

    ax_path = fig.add_axes([0.07, 0.28, 0.60, 0.67])
    ax_cmd = fig.add_axes([0.72, 0.45, 0.24, 0.50])
    ax_slider = fig.add_axes([0.12, 0.16, 0.76, 0.04])

    ax_start = fig.add_axes([0.12, 0.06, 0.15, 0.06])
    ax_end = fig.add_axes([0.30, 0.06, 0.15, 0.06])
    ax_print = fig.add_axes([0.48, 0.06, 0.18, 0.06])
    ax_reset = fig.add_axes([0.69, 0.06, 0.15, 0.06])

    ax_path.set_title("Bag visual scrubber: path, current pose, and current LiDAR scan")
    ax_path.set_xlabel("x position from /odom [m]")
    ax_path.set_ylabel("y position from /odom [m]")
    ax_path.axis("equal")
    ax_path.grid(True)

    full_path_line, = ax_path.plot(odom_x, odom_y, linewidth=1, alpha=0.35, label="Full path")
    current_path_line, = ax_path.plot([], [], linewidth=2, label="Path up to current time")
    robot_point, = ax_path.plot([], [], marker="o", markersize=8, linestyle="None", label="Current robot pose")
    heading_line, = ax_path.plot([], [], linewidth=2, label="Robot heading")
    scan_scatter = ax_path.scatter([], [], s=3, alpha=0.6, label="Current scan points")

    start_vline = ax_path.axvline(0, alpha=0.0)
    end_vline = ax_path.axvline(0, alpha=0.0)

    ax_path.legend(loc="best")

    ax_cmd.set_title("/cmd_vel over time")
    ax_cmd.set_xlabel("time [s]")
    ax_cmd.set_ylabel("velocity")
    ax_cmd.grid(True)

    if cmds:
        ax_cmd.plot(cmd_t, cmd_lin, label="linear.x")
        ax_cmd.plot(cmd_t, cmd_ang, label="angular.z")
        cmd_cursor = ax_cmd.axvline(0.0, linewidth=2)
        ax_cmd.legend(loc="best")
    else:
        ax_cmd.text(0.5, 0.5, "No /cmd_vel loaded", ha="center", va="center")
        cmd_cursor = ax_cmd.axvline(0.0, linewidth=2)

    slider = Slider(
        ax=ax_slider,
        label="time [s]",
        valmin=0.0,
        valmax=max(duration, 0.001),
        valinit=0.0,
        valstep=0.05,
    )

    btn_start = Button(ax_start, "Mark start")
    btn_end = Button(ax_end, "Mark end")
    btn_print = Button(ax_print, "Print crop cmd")
    btn_reset = Button(ax_reset, "Reset marks")

    status = fig.text(0.07, 0.22, "", fontsize=10)

    def update(t):
        oi = nearest_index(odom_t, t)
        if oi is None:
            return

        pose = odom[oi]
        _, x, y, yaw = pose

        current_path_line.set_data(odom_x[:oi + 1], odom_y[:oi + 1])
        robot_point.set_data([x], [y])

        arrow_len = 0.25
        heading_line.set_data(
            [x, x + arrow_len * math.cos(yaw)],
            [y, y + arrow_len * math.sin(yaw)],
        )

        if scans:
            si = nearest_index(scan_t, t)
            scan = scans[si]
            sx, sy = scan_points_in_odom(scan, pose)
            scan_scatter.set_offsets(list(zip(sx, sy)) if sx else [])
            scan_time_text = f"{scan[0]:.2f}s"
        else:
            scan_scatter.set_offsets([])
            scan_time_text = "none"

        cmd_cursor.set_xdata([t, t])

        status.set_text(
            f"Current time: {t:.2f}s | "
            f"Pose: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f} deg | "
            f"nearest scan: {scan_time_text} | "
            f"crop start: {crop_start['value']} | crop end: {crop_end['value']}"
        )

        fig.canvas.draw_idle()

    def mark_start(event):
        crop_start["value"] = round(float(slider.val), 2)
        update(slider.val)

    def mark_end(event):
        crop_end["value"] = round(float(slider.val), 2)
        update(slider.val)

    def print_cmd(event):
        s = crop_start["value"]
        e = crop_end["value"]

        if s is None or e is None:
            print("\n[WARN] Mark both start and end first.")
            return

        if e <= s:
            print("\n[WARN] End time must be after start time.")
            return

        print("\nSuggested crop command:")
        print(f"  ./crop_rosbag.py <YOUR_BAG_FOLDER> -w {s}:{e} -o cropped_run")
        print("\nIf this scrubber was opened on the same folder, replace <YOUR_BAG_FOLDER> with that path.")

    def reset_marks(event):
        crop_start["value"] = None
        crop_end["value"] = None
        update(slider.val)

    slider.on_changed(update)
    btn_start.on_clicked(mark_start)
    btn_end.on_clicked(mark_end)
    btn_print.on_clicked(print_cmd)
    btn_reset.on_clicked(reset_marks)

    update(0.0)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visually scrub through a ROS 2 bag using odom, scan, and cmd_vel."
    )

    parser.add_argument(
        "bag_folder",
        help="Path to ROS 2 bag folder, not the .mcap or .db3 file directly.",
    )

    parser.add_argument("--odom-topic", default=None, help="Odom topic, e.g. /tb4_6/odom")
    parser.add_argument("--scan-topic", default=None, help="Scan topic, e.g. /tb4_6/scan")
    parser.add_argument("--cmd-topic", default=None, help="Command velocity topic, e.g. /tb4_6/cmd_vel")
    parser.add_argument(
        "--max-scans",
        type=int,
        default=1200,
        help="Maximum scan messages to load for visualization. Lower this if plotting is slow.",
    )

    args = parser.parse_args()

    rclpy.init(args=None)

    try:
        bag_dir = Path(args.bag_folder).expanduser().resolve()
        if not bag_dir.exists() or not bag_dir.is_dir():
            raise RuntimeError(f"Bag folder does not exist or is not a directory: {bag_dir}")

        data = load_bag(
            bag_dir=bag_dir,
            odom_topic=args.odom_topic,
            scan_topic=args.scan_topic,
            cmd_topic=args.cmd_topic,
            max_scans=args.max_scans,
        )

        plot_scrubber(data)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()