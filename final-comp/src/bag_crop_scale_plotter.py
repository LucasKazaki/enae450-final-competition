#!/usr/bin/env python3

import argparse
import math
import sys
import yaml
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

import rosbag2_py
from rosbag2_py import (
    SequentialReader,
    SequentialWriter,
    StorageOptions,
    ConverterOptions,
    TopicMetadata,
)

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# -----------------------------
# Basic math helpers
# -----------------------------

def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def nearest_index(times, target):
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


# -----------------------------
# Bag helpers
# -----------------------------

def get_storage_id(bag_dir: Path):
    metadata_path = bag_dir / "metadata.yaml"

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        try:
            return metadata["rosbag2_bagfile_information"]["storage_identifier"]
        except Exception:
            pass

    if list(bag_dir.glob("*.mcap")):
        return "mcap"
    if list(bag_dir.glob("*.db3")):
        return "sqlite3"

    raise RuntimeError("Could not determine bag storage type. Expected metadata.yaml, .mcap, or .db3.")


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


def auto_pick_topic(all_topics, suffixes):
    for suffix in suffixes:
        matches = [t for t in all_topics if t == suffix or t.endswith(suffix)]
        if matches:
            return sorted(matches, key=len, reverse=True)[0]
    return None


def make_topic_metadata(topic_info):
    try:
        return TopicMetadata(
            id=0,
            name=topic_info.name,
            type=topic_info.type,
            serialization_format=topic_info.serialization_format,
            offered_qos_profiles=topic_info.offered_qos_profiles,
        )
    except TypeError:
        meta = TopicMetadata()
        meta.name = topic_info.name
        meta.type = topic_info.type
        meta.serialization_format = topic_info.serialization_format

        if hasattr(meta, "offered_qos_profiles") and hasattr(topic_info, "offered_qos_profiles"):
            meta.offered_qos_profiles = topic_info.offered_qos_profiles

        return meta


def crop_bag(input_bag_dir: Path, output_bag_dir: Path, start_s: float, end_s: float, output_storage="mcap"):
    """
    Crops the raw bag messages unchanged using the same idea as crop_rosbag.py:
    time window is relative to the first message timestamp in the bag.
    """
    if output_bag_dir.exists():
        raise FileExistsError(f"Output bag folder already exists: {output_bag_dir}")

    input_storage = get_storage_id(input_bag_dir)

    # First pass: get absolute start timestamp.
    reader = open_reader(input_bag_dir)
    first_ns = None

    while reader.has_next():
        _, _, t = reader.read_next()
        first_ns = t
        break

    if first_ns is None:
        raise RuntimeError("Input bag is empty.")

    crop_start_ns = first_ns + int(start_s * 1e9)
    crop_end_ns = first_ns + int(end_s * 1e9)

    if crop_end_ns <= crop_start_ns:
        raise ValueError("Crop end must be after crop start.")

    # Second pass: write cropped bag.
    reader = open_reader(input_bag_dir)

    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(output_bag_dir), storage_id=output_storage),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    for topic_info in reader.get_all_topics_and_types():
        writer.create_topic(make_topic_metadata(topic_info))

    kept = 0
    read = 0
    first_written_ns = None

    while reader.has_next():
        topic, data, t = reader.read_next()
        read += 1

        if not (crop_start_ns <= t <= crop_end_ns):
            continue

        if first_written_ns is None:
            first_written_ns = t

        # Shift output bag to start at t=0, like the earlier crop script default.
        out_t = t - first_written_ns

        writer.write(topic, data, out_t)
        kept += 1

    print("\nCropped bag written:")
    print(f"  input:  {input_bag_dir}")
    print(f"  output: {output_bag_dir}")
    print(f"  window: {start_s:.3f}s to {end_s:.3f}s")
    print(f"  kept messages: {kept}")
    print(f"  dropped messages: {read - kept}")

    if kept == 0:
        print("[WARN] Cropped bag is empty. Check start/end times.")


# -----------------------------
# Load bag data for visualization
# -----------------------------

def load_bag_for_visualization(
    bag_dir: Path,
    odom_topic=None,
    scan_topic=None,
    cmd_topic=None,
    max_scans=1500,
):
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
        raise RuntimeError("Could not find odom topic. Pass --odom-topic manually.")
    if scan_topic is None:
        raise RuntimeError("Could not find scan topic. Pass --scan-topic manually.")

    msg_classes = {}
    for topic, msg_type in types.items():
        try:
            msg_classes[topic] = get_message(msg_type)
        except Exception:
            pass

    # Count scans for downsampling.
    count_reader = open_reader(bag_dir)
    first_ns = None
    last_ns = None
    total_scan_count = 0

    while count_reader.has_next():
        topic, _, t = count_reader.read_next()

        if first_ns is None:
            first_ns = t
        last_ns = t

        if topic == scan_topic:
            total_scan_count += 1

    if first_ns is None:
        raise RuntimeError("Bag has no messages.")

    scan_keep_every = 1
    if total_scan_count > max_scans:
        scan_keep_every = math.ceil(total_scan_count / max_scans)

    odom = []
    scans = []
    cmds = []

    reader = open_reader(bag_dir)
    scan_seen = 0

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
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

            scans.append((
                rel_t,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                list(msg.ranges),
            ))

        elif topic == cmd_topic:
            # Supports both Twist and TwistStamped.
            if hasattr(msg, "twist"):
                tw = msg.twist
            else:
                tw = msg

            lin_x = getattr(tw.linear, "x", 0.0)
            ang_z = getattr(tw.angular, "z", 0.0)
            cmds.append((rel_t, lin_x, ang_z))

    duration = (last_ns - first_ns) / 1e9

    if not odom:
        raise RuntimeError("No odom messages loaded.")
    if not scans:
        raise RuntimeError("No scan messages loaded.")

    print("\nLoaded:")
    print(f"  duration: {duration:.3f}s")
    print(f"  odom messages: {len(odom)}")
    print(f"  scan messages shown: {len(scans)} of {total_scan_count}")
    print(f"  cmd_vel messages: {len(cmds)}")

    return {
        "bag_dir": bag_dir,
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


# -----------------------------
# Visualization transformations
# -----------------------------

def scaled_pose(pose, scale):
    """
    Scale odom x/y while leaving yaw unchanged.
    If odom says the robot moved 10x too far, use scale around 0.1.
    """
    t, x, y, yaw = pose
    return t, x * scale, y * scale, yaw


def scan_points_in_scaled_odom(scan, pose, scale, scan_yaw_offset_rad=0.0, max_points=720):
    """
    Converts a scan into odom/world-like coordinates using scaled odom position,
    odom yaw, and a fixed scan yaw offset.

    scan_yaw_offset_rad corrects cases where the LiDAR scan frame is rotated
    relative to the odom/base frame. If the scan map appears rotated -90 degrees,
    use +90 degrees as the correction.
    """
    if scan is None or pose is None:
        return [], []

    _, angle_min, angle_inc, range_min, range_max, ranges = scan
    _, x, y, yaw = scaled_pose(pose, scale)

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

        # Main fix:
        # Add scan_yaw_offset_rad to correct the scan frame orientation.
        global_a = yaw + a + scan_yaw_offset_rad

        xs.append(x + r * math.cos(global_a))
        ys.append(y + r * math.sin(global_a))

    return xs, ys

def build_obstacle_cloud(
    data,
    start_s,
    end_s,
    scale,
    scan_yaw_offset_rad=0.0,
    scan_stride=1,
    max_points_per_scan=360,
):
    odom = data["odom"]
    scans = data["scans"]

    odom_t = [o[0] for o in odom]

    cloud_x = []
    cloud_y = []

    used_scans = 0

    for i, scan in enumerate(scans):
        if i % scan_stride != 0:
            continue

        scan_t = scan[0]

        if not (start_s <= scan_t <= end_s):
            continue

        oi = nearest_index(odom_t, scan_t)
        if oi is None:
            continue

        sx, sy = scan_points_in_scaled_odom(
            scan=scan,
            pose=odom[oi],
            scale=scale,
            scan_yaw_offset_rad=scan_yaw_offset_rad,
            max_points=max_points_per_scan,
        )

        cloud_x.extend(sx)
        cloud_y.extend(sy)
        used_scans += 1

    return cloud_x, cloud_y, used_scans


# -----------------------------
# Save final report image
# -----------------------------

def save_cropped_plot(data, start_s, end_s, scale, scan_yaw_offset_rad, output_png):
    odom = data["odom"]

    cropped_odom = [o for o in odom if start_s <= o[0] <= end_s]

    if not cropped_odom:
        raise RuntimeError("No odom data inside selected crop window.")

    path_x = [o[1] * scale for o in cropped_odom]
    path_y = [o[2] * scale for o in cropped_odom]

    # Use a stride so the final image does not get overloaded.
    cloud_x, cloud_y, used_scans = build_obstacle_cloud(
        data=data,
        start_s=start_s,
        end_s=end_s,
        scale=scale,
        scan_yaw_offset_rad=scan_yaw_offset_rad,
        scan_stride=1,
        max_points_per_scan=240,
    )   
    fig, ax = plt.subplots(figsize=(9, 8))

    if cloud_x:
        ax.scatter(cloud_x, cloud_y, s=1, alpha=0.25, label="LiDAR obstacle points")

    ax.plot(path_x, path_y, linewidth=2, label="Robot path from /odom")
    ax.scatter([path_x[0]], [path_y[0]], s=70, marker="o", label="Start")
    ax.scatter([path_x[-1]], [path_y[-1]], s=70, marker="x", label="End")

    ax.set_title(
        f"Cropped run path and LiDAR obstacles\n"
        f"window={start_s:.2f}s to {end_s:.2f}s, "
        f"odom scale={scale:.4f}, "
        f"scan yaw offset={math.degrees(scan_yaw_offset_rad):.1f} deg"
    )
    ax.set_xlabel("x position [m], scaled odom frame")
    ax.set_ylabel("y position [m], scaled odom frame")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_png, dpi=250)
    plt.close(fig)

    print("\nSaved final plot:")
    print(f"  {output_png}")
    print(f"  scans used for obstacle cloud: {used_scans}")


# -----------------------------
# Interactive GUI
# -----------------------------

def run_gui(data, initial_scale, initial_scan_yaw_offset_deg, output_prefix, output_storage):
    odom = data["odom"]
    scans = data["scans"]
    cmds = data["cmds"]
    duration = data["duration"]
    bag_dir = data["bag_dir"]

    odom_t = [o[0] for o in odom]
    scan_t = [s[0] for s in scans]

    cmd_t = [c[0] for c in cmds]
    cmd_lin = [c[1] for c in cmds]
    cmd_ang = [c[2] for c in cmds]

    crop_start = {"value": None}
    crop_end = {"value": None}

    fig = plt.figure(figsize=(14, 9))

    ax_path = fig.add_axes([0.06, 0.31, 0.62, 0.64])
    ax_cmd = fig.add_axes([0.73, 0.50, 0.23, 0.43])

    ax_time_slider = fig.add_axes([0.12, 0.235, 0.76, 0.03])
    ax_scale_slider = fig.add_axes([0.12, 0.185, 0.76, 0.03])
    ax_scan_yaw_slider = fig.add_axes([0.12, 0.135, 0.76, 0.03])

    ax_start = fig.add_axes([0.07, 0.06, 0.12, 0.055])
    ax_end = fig.add_axes([0.205, 0.06, 0.12, 0.055])
    ax_save_plot = fig.add_axes([0.34, 0.06, 0.14, 0.055])
    ax_crop = fig.add_axes([0.495, 0.06, 0.14, 0.055])
    ax_print = fig.add_axes([0.65, 0.06, 0.14, 0.055])
    ax_reset = fig.add_axes([0.805, 0.06, 0.12, 0.055])

    ax_path.set_title("Visual bag cropper with odom scale correction")
    ax_path.set_xlabel("x [m], scaled odom frame")
    ax_path.set_ylabel("y [m], scaled odom frame")
    ax_path.axis("equal")
    ax_path.grid(True)

    full_path_line, = ax_path.plot([], [], linewidth=1, alpha=0.35, label="Full path")
    current_path_line, = ax_path.plot([], [], linewidth=2, label="Path up to current time")
    robot_point, = ax_path.plot([], [], marker="o", markersize=8, linestyle="None", label="Current robot pose")
    heading_line, = ax_path.plot([], [], linewidth=2, label="Robot heading")
    scan_scatter = ax_path.scatter([], [], s=3, alpha=0.55, label="Current scan points")

    start_marker, = ax_path.plot([], [], marker="o", markersize=10, linestyle="None", label="Marked crop start")
    end_marker, = ax_path.plot([], [], marker="x", markersize=10, linestyle="None", label="Marked crop end")

    ax_path.legend(loc="best")

    ax_cmd.set_title("/cmd_vel over time")
    ax_cmd.set_xlabel("time [s]")
    ax_cmd.set_ylabel("velocity")
    ax_cmd.grid(True)

    if cmds:
        ax_cmd.plot(cmd_t, cmd_lin, label="linear.x")
        ax_cmd.plot(cmd_t, cmd_ang, label="angular.z")
        ax_cmd.legend(loc="best")
    else:
        ax_cmd.text(0.5, 0.5, "No /cmd_vel loaded", ha="center", va="center")

    cmd_cursor = ax_cmd.axvline(0.0, linewidth=2)

    time_slider = Slider(
        ax=ax_time_slider,
        label="time [s]",
        valmin=0.0,
        valmax=max(duration, 0.001),
        valinit=0.0,
        valstep=0.05,
    )

    scale_slider = Slider(
        ax=ax_scale_slider,
        label="odom x/y scale",
        valmin=0.02,
        valmax=2.0,
        valinit=initial_scale,
        valstep=0.05,
    )

    scan_yaw_slider = Slider(
        ax=ax_scan_yaw_slider,
        label="scan yaw offset [deg]",
        valmin=-180.0,
        valmax=180.0,
        valinit=initial_scan_yaw_offset_deg,
        valstep=1.0,
    )

    btn_start = Button(ax_start, "Mark start")
    btn_end = Button(ax_end, "Mark end")
    btn_save_plot = Button(ax_save_plot, "Save image")
    btn_crop = Button(ax_crop, "Crop bag")
    btn_print = Button(ax_print, "Print cmd")
    btn_reset = Button(ax_reset, "Reset")

    status = fig.text(0.06, 0.275, "", fontsize=10)

    def current_scaled_path(scale):
        return [o[1] * scale for o in odom], [o[2] * scale for o in odom]

    def pose_at_time(t):
        oi = nearest_index(odom_t, t)
        if oi is None:
            return None, None
        return oi, odom[oi]

    def update(t=None):
        if t is None:
            t = float(time_slider.val)

        scale = float(scale_slider.val)
        scan_yaw_offset_rad = math.radians(float(scan_yaw_slider.val))

        full_x, full_y = current_scaled_path(scale)
        full_path_line.set_data(full_x, full_y)

        oi, pose = pose_at_time(t)
        if pose is None:
            return

        _, sx, sy, yaw = scaled_pose(pose, scale)

        current_path_line.set_data(full_x[:oi + 1], full_y[:oi + 1])
        robot_point.set_data([sx], [sy])

        arrow_len = 0.25
        heading_line.set_data(
            [sx, sx + arrow_len * math.cos(yaw)],
            [sy, sy + arrow_len * math.sin(yaw)],
        )

        si = nearest_index(scan_t, t)
        if si is not None:
            scan = scans[si]
            px, py = scan_points_in_scaled_odom(
                scan=scan,
                pose=pose,
                scale=scale,
                scan_yaw_offset_rad=scan_yaw_offset_rad,
            )
            scan_scatter.set_offsets(list(zip(px, py)) if px else [])
            scan_time = scan[0]
        else:
            scan_scatter.set_offsets([])
            scan_time = None

        if crop_start["value"] is not None:
            _, spose = pose_at_time(crop_start["value"])
            _, x0, y0, _ = scaled_pose(spose, scale)
            start_marker.set_data([x0], [y0])
        else:
            start_marker.set_data([], [])

        if crop_end["value"] is not None:
            _, epose = pose_at_time(crop_end["value"])
            _, x1, y1, _ = scaled_pose(epose, scale)
            end_marker.set_data([x1], [y1])
        else:
            end_marker.set_data([], [])

        cmd_cursor.set_xdata([t, t])

        ax_path.relim()
        ax_path.autoscale_view()

        status.set_text(
            f"time={t:.2f}s | scale={scale:.4f} | scan yaw offset={float(scan_yaw_slider.val):.1f} deg | "
            f"pose scaled: x={sx:.2f}, y={sy:.2f}, yaw={math.degrees(yaw):.1f} deg | "
            f"nearest scan={scan_time:.2f}s" if scan_time is not None else
            f"time={t:.2f}s | scale={scale:.4f} | pose scaled: x={sx:.2f}, y={sy:.2f}"
        )

        fig.canvas.draw_idle()

    def mark_start(event):
        crop_start["value"] = round(float(time_slider.val), 2)
        print(f"Marked crop start: {crop_start['value']:.2f}s")
        update()

    def mark_end(event):
        crop_end["value"] = round(float(time_slider.val), 2)
        print(f"Marked crop end: {crop_end['value']:.2f}s")
        update()

    def get_valid_window():
        s = crop_start["value"]
        e = crop_end["value"]

        if s is None or e is None:
            print("[WARN] Mark start and end first.")
            return None

        if e <= s:
            print("[WARN] End must be after start.")
            return None

        return s, e

    def output_names():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scale = float(scale_slider.val)

        prefix = output_prefix
        if prefix is None:
            prefix = f"{bag_dir.name}_cropped_{stamp}"

        out_bag = bag_dir.parent / prefix
        out_png = bag_dir.parent / f"{prefix}_scale_{scale:.4f}.png"

        return out_bag, out_png

    def save_image(event):
        window = get_valid_window()
        if window is None:
            return

        s, e = window
        scale = float(scale_slider.val)
        scan_yaw_offset_rad = math.radians(float(scan_yaw_slider.val))
        _, out_png = output_names()

        try:
            save_cropped_plot(data, s, e, scale, scan_yaw_offset_rad, out_png)
        except Exception as ex:
            print(f"[ERROR] Could not save image: {ex}")

    def crop_action(event):
        window = get_valid_window()
        if window is None:
            return

        s, e = window
        out_bag, _ = output_names()

        try:
            crop_bag(
                input_bag_dir=bag_dir,
                output_bag_dir=out_bag,
                start_s=s,
                end_s=e,
                output_storage=output_storage,
            )
        except Exception as ex:
            print(f"[ERROR] Could not crop bag: {ex}")

    def print_cmd(event):
        window = get_valid_window()
        if window is None:
            return

        s, e = window
        scale = float(scale_slider.val)
        scan_yaw_offset_deg = float(scan_yaw_slider.val)
        out_bag, out_png = output_names()

        print("\nEquivalent crop command:")
        print(f"  ./crop_rosbag.py {bag_dir} -w {s}:{e} -o {out_bag}")
        print("\nEquivalent image settings:")
        print(f"  crop_start={s}")
        print(f"  crop_end={e}")
        print(f"  odom_scale={scale:.4f}")
        print(f"  output_image={out_png}")
        print(f"  scan_yaw_offset_deg={scan_yaw_offset_deg:.1f}")

    def reset(event):
        crop_start["value"] = None
        crop_end["value"] = None
        update()

    time_slider.on_changed(update)
    scale_slider.on_changed(update)
    scan_yaw_slider.on_changed(update)

    btn_start.on_clicked(mark_start)
    btn_end.on_clicked(mark_end)
    btn_save_plot.on_clicked(save_image)
    btn_crop.on_clicked(crop_action)
    btn_print.on_clicked(print_cmd)
    btn_reset.on_clicked(reset)

    update()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visual ROS 2 bag scrubber/cropper with odom x/y scale correction "
            "and final path/obstacle image output."
        )
    )

    parser.add_argument(
        "bag_folder",
        help="Path to ROS 2 bag folder, not the .mcap/.db3 file directly.",
    )

    parser.add_argument("--odom-topic", default=None, help="Odom topic, e.g. /tb4_6/odom")
    parser.add_argument("--scan-topic", default=None, help="Scan topic, e.g. /tb4_6/scan")
    parser.add_argument("--cmd-topic", default=None, help="Cmd vel topic, e.g. /tb4_6/cmd_vel")

    parser.add_argument(
        "--initial-scale",
        type=float,
        default=0.1,
        help="Initial odom x/y scale. If odom looks 10x too large, use 0.1. Default: 0.1",
    )
    parser.add_argument(
        "--initial-scan-yaw-offset-deg",
        type=float,
        default=90.0,
        help=(
            "Initial scan yaw correction in degrees. "
            "If the scan map appears rotated -90 degrees, use +90. Default: 90."
        ),
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=1500,
        help="Maximum scan messages loaded into the interactive viewer. Default: 1500",
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix/name for cropped bag folder and saved image. Default uses bag name plus timestamp.",
    )

    parser.add_argument(
        "--output-storage",
        default="mcap",
        choices=["mcap", "sqlite3"],
        help="Storage format for cropped output bag. Default: mcap",
    )

    args = parser.parse_args()

    rclpy.init(args=None)

    try:
        bag_dir = Path(args.bag_folder).expanduser().resolve()

        if not bag_dir.exists() or not bag_dir.is_dir():
            raise RuntimeError(f"Bag folder does not exist or is not a directory: {bag_dir}")

        data = load_bag_for_visualization(
            bag_dir=bag_dir,
            odom_topic=args.odom_topic,
            scan_topic=args.scan_topic,
            cmd_topic=args.cmd_topic,
            max_scans=args.max_scans,
        )

        run_gui(
            data=data,
            initial_scale=args.initial_scale,
            initial_scan_yaw_offset_deg=args.initial_scan_yaw_offset_deg,
            output_prefix=args.output_prefix,
            output_storage=args.output_storage,
        )

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()