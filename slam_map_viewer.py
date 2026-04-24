#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def write_params(args):
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
    minimum_time_interval: 0.1
    transform_timeout: 0.2
    tf_buffer_duration: 30.0
    stack_size_to_use: 40000000

    map_update_interval: 0.5

    use_scan_matching: true
    use_scan_barycenter: true
    minimum_travel_distance: 0.05
    minimum_travel_heading: 0.05

    scan_buffer_size: 10
    scan_buffer_maximum_scan_distance: 10.0

    link_match_minimum_response_fine: 0.1
    link_scan_maximum_distance: 1.5
    loop_search_maximum_distance: 3.0
    do_loop_closing: true
    loop_match_minimum_chain_size: 10
    loop_match_maximum_variance_coarse: 3.0
    loop_match_minimum_response_coarse: 0.35
    loop_match_minimum_response_fine: 0.45

    correlation_search_space_dimension: 0.5
    correlation_search_space_resolution: 0.01
    correlation_search_space_smear_deviation: 0.1

    loop_search_space_dimension: 8.0
    loop_search_space_resolution: 0.05
    loop_search_space_smear_deviation: 0.03

    distance_variance_penalty: 0.5
    angle_variance_penalty: 1.0

    fine_search_angle_offset: 0.00349
    coarse_search_angle_offset: 0.349
    coarse_angle_resolution: 0.0349

    minimum_angle_penalty: 0.9
    minimum_distance_penalty: 0.5
    use_response_expansion: true
"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
    tmp.write(params)
    tmp.close()
    return tmp.name


def run_cmd(cmd, name):
    print(f"\n[starting {name}]")
    print(" ".join(cmd))
    return subprocess.Popen(cmd, preexec_fn=os.setsid)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone TurtleBot4 SLAM script using slam_toolbox."
    )

    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--map-frame", default="map")

    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--max-laser-range", type=float, default=12.0)

    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--save-map", default="")
    parser.add_argument("--map-saver", action="store_true")

    args = parser.parse_args()

    params_file = write_params(args)
    procs = []

    try:
        slam_cmd = [
            "ros2", "run", "slam_toolbox", "async_slam_toolbox_node",
            "--ros-args",
            "--params-file", params_file,
        ]

        procs.append(run_cmd(slam_cmd, "slam_toolbox"))

        time.sleep(2.0)

        if args.rviz:
            rviz_cmd = ["rviz2"]
            procs.append(run_cmd(rviz_cmd, "rviz2"))

        print("\nSLAM is running.")
        print("Map topic: /map")
        print("Pose graph handled internally by slam_toolbox.")
        print("In RViz, set Fixed Frame to: map")
        print("Press Ctrl+C to stop.")

        while True:
            time.sleep(1.0)
            print("sleeping")

    except KeyboardInterrupt:
        print("\nStopping SLAM...")

    finally:
        if args.save_map:
            save_path = Path(args.save_map).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\nSaving map to: {save_path}")
            save_cmd = [
                "ros2", "run", "nav2_map_server", "map_saver_cli",
                "-f", str(save_path)
            ]

            try:
                subprocess.run(save_cmd, check=False)
            except Exception as e:
                print(f"Map save failed: {e}")

        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception:
                pass

        time.sleep(1.0)

        for p in procs:
            try:
                if p.poll() is None:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass

        try:
            os.remove(params_file)
        except Exception:
            pass

        print("Done.")


if __name__ == "__main__":
    main()