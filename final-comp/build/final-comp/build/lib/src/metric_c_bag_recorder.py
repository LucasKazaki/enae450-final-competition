#!/usr/bin/env python3

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def clean_namespace(ns: str) -> str:
    """
    Converts namespace inputs like:
      tb4_4
      /tb4_4
      /tb4_4/
    into:
      /tb4_4
    """
    ns = ns.strip()

    if ns == "" or ns == "/":
        return ""

    if not ns.startswith("/"):
        ns = "/" + ns

    return ns.rstrip("/")


def build_topics(namespace: str, include_map: bool, include_extra: bool):
    """
    Metric C minimum:
      /tf
      /odom
      /scan
      /cmd_vel

    For namespaced TurtleBot 4 topics:
      /tb4_X/odom
      /tb4_X/scan
      /tb4_X/cmd_vel

    TF usually remains global as /tf and /tf_static.
    """
    ns = clean_namespace(namespace)

    topics = [
        "/tf",
        "/tf_static",
    ]

    if ns:
        topics.extend([
            f"{ns}/odom",
            f"{ns}/scan",
            f"{ns}/cmd_vel",
        ])
    else:
        topics.extend([
            "/odom",
            "/scan",
            "/cmd_vel",
        ])

    if include_map:
        if ns:
            # Some slam setups publish /map globally, some publish namespaced map.
            topics.extend([
                "/map",
                f"{ns}/map",
            ])
        else:
            topics.append("/map")

    if include_extra:
        if ns:
            topics.extend([
                f"{ns}/battery_state",
                f"{ns}/joint_states",
                f"{ns}/imu",
            ])
        else:
            topics.extend([
                "/battery_state",
                "/joint_states",
                "/imu",
            ])

    # Remove duplicates while preserving order.
    seen = set()
    unique_topics = []
    for topic in topics:
        if topic not in seen:
            unique_topics.append(topic)
            seen.add(topic)

    return unique_topics


def topic_exists(topic: str) -> bool:
    try:
        result = subprocess.run(
            ["ros2", "topic", "list", "--no-daemon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            return False

        available = set(line.strip() for line in result.stdout.splitlines())
        return topic in available

    except Exception:
        return False


def print_topic_check(topics):
    print("\n[metric_c_bag_recorder] Checking requested topics:")
    for topic in topics:
        status = "FOUND" if topic_exists(topic) else "NOT FOUND YET"
        print(f"  {status:13s} {topic}")
    print()


def make_output_path(base_dir: str, prefix: str, namespace: str) -> str:
    ns_clean = clean_namespace(namespace).strip("/").replace("/", "_")
    ns_part = ns_clean if ns_clean else "nonamespace"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(base_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir / f"{prefix}_{ns_part}_{timestamp}")


def main():
    parser = argparse.ArgumentParser(
        description="Record Metric C ROS 2 topics into an MCAP bag."
    )

    parser.add_argument(
        "--namespace",
        "-n",
        default="",
        help="Robot namespace, for example tb4_4 or /tb4_4. Leave blank for no namespace.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="metric_c_bags",
        help="Directory where the bag folder will be saved.",
    )

    parser.add_argument(
        "--prefix",
        default="metric_c_run",
        help="Prefix for the output bag folder name.",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=0.0,
        help="Optional recording duration in seconds. Use 0 to record until Ctrl+C.",
    )

    parser.add_argument(
        "--include-map",
        action="store_true",
        help="Also record /map and the namespaced map topic if applicable.",
    )

    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Also record battery_state, joint_states, and imu if available.",
    )

    parser.add_argument(
        "--no-topic-check",
        action="store_true",
        help="Skip pre-recording topic existence check.",
    )

    args = parser.parse_args()

    namespace = clean_namespace(args.namespace)
    topics = build_topics(
        namespace=namespace,
        include_map=args.include_map,
        include_extra=args.include_extra,
    )

    bag_path = make_output_path(
        base_dir=args.output_dir,
        prefix=args.prefix,
        namespace=namespace,
    )

    print("[metric_c_bag_recorder] Namespace:", namespace if namespace else "(none)")
    print("[metric_c_bag_recorder] Output bag:", bag_path)

    if not args.no_topic_check:
        print_topic_check(topics)

    cmd = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "-o",
        bag_path,
    ] + topics

    print("[metric_c_bag_recorder] Starting recording command:")
    print("  " + " ".join(cmd))
    print()
    print("[metric_c_bag_recorder] Press Ctrl+C to stop recording.")
    if args.duration > 0:
        print(f"[metric_c_bag_recorder] Auto-stop after {args.duration:.1f} seconds.")
    print()

    process = subprocess.Popen(cmd)

    try:
        if args.duration > 0:
            start = time.time()
            while process.poll() is None:
                elapsed = time.time() - start
                if elapsed >= args.duration:
                    print("\n[metric_c_bag_recorder] Duration reached. Stopping bag recorder...")
                    process.send_signal(signal.SIGINT)
                    break
                time.sleep(0.2)

            process.wait()
        else:
            process.wait()

    except KeyboardInterrupt:
        print("\n[metric_c_bag_recorder] Ctrl+C received. Stopping bag recorder...")
        process.send_signal(signal.SIGINT)

        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            print("[metric_c_bag_recorder] Recorder did not stop cleanly. Terminating...")
            process.terminate()
            process.wait(timeout=5.0)

    print()
    print("[metric_c_bag_recorder] Done.")
    print(f"[metric_c_bag_recorder] Bag saved at: {bag_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

'''ros2 run final-comp metric_c_bag_recorder --namespace /tb4_4'''