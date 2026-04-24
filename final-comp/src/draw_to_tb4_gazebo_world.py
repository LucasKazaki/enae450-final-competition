#!/usr/bin/env python3
"""
draw_to_tb4_gazebo_world.py

Draw a maze/shape on a square canvas and export it as a Gazebo SDF world with
1 meter tall wall segments. Designed for TurtleBot4 / ROS 2 Gazebo simulation.

Controls:
  Left mouse drag      Draw walls
  Right mouse drag     Erase walls
  C                    Clear canvas
  S                    Save drawing as PNG
  L                    Load drawing PNG
  E                    Export SDF world

Run:
  python3 draw_to_tb4_gazebo_world.py

Then load the exported world with your TurtleBot4 Gazebo launch file, for example:
  ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=/absolute/path/to/drawn_tb4_world.sdf

If your launch file uses a different argument name, check:
  ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py --show-args
"""

from __future__ import annotations

import math
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, simpledialog
from typing import Iterable, List, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit(
        "This script requires Pillow. Install it with:\n"
        "  python3 -m pip install pillow"
    ) from exc


@dataclass
class Wall:
    x: float
    y: float
    length: float
    thickness: float
    height: float
    yaw: float


class DrawToGazeboWorld:
    def __init__(self) -> None:
        self.canvas_px = 700
        self.grid_cells = 140
        self.world_size_m = 14.0
        self.wall_height_m = 1.0
        self.wall_thickness_m = self.world_size_m / self.grid_cells
        self.robot_clear_radius_m = 0.45

        self.brush_cells = 2
        self.erase_cells = 4

        self.cell_px = self.canvas_px / self.grid_cells
        self.grid = [[0 for _ in range(self.grid_cells)] for _ in range(self.grid_cells)]

        self.root = tk.Tk()
        self.root.title("Draw TurtleBot4 Gazebo World")

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_px,
            height=self.canvas_px,
            bg="white",
            highlightthickness=1,
            highlightbackground="black",
        )
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        tk.Button(self.root, text="Export SDF", command=self.export_sdf_dialog).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        tk.Button(self.root, text="Save PNG", command=self.save_png_dialog).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        tk.Button(self.root, text="Load PNG", command=self.load_png_dialog).grid(row=1, column=2, sticky="ew", padx=5, pady=5)
        tk.Button(self.root, text="Clear", command=self.clear).grid(row=1, column=3, sticky="ew", padx=5, pady=5)

        self.status = tk.StringVar()
        self.status.set(self.help_text())
        tk.Label(self.root, textvariable=self.status, justify="left").grid(row=2, column=0, columnspan=4, sticky="w", padx=10)

        self.canvas.bind("<B1-Motion>", self.draw_event)
        self.canvas.bind("<Button-1>", self.draw_event)
        self.canvas.bind("<B3-Motion>", self.erase_event)
        self.canvas.bind("<Button-3>", self.erase_event)
        self.root.bind("c", lambda _event: self.clear())
        self.root.bind("C", lambda _event: self.clear())
        self.root.bind("s", lambda _event: self.save_png_dialog())
        self.root.bind("S", lambda _event: self.save_png_dialog())
        self.root.bind("l", lambda _event: self.load_png_dialog())
        self.root.bind("L", lambda _event: self.load_png_dialog())
        self.root.bind("e", lambda _event: self.export_sdf_dialog())
        self.root.bind("E", lambda _event: self.export_sdf_dialog())

        self.redraw()

    def help_text(self) -> str:
        return (
            f"Left drag: draw | Right drag: erase | C: clear | S: save PNG | L: load PNG | E: export SDF\n"
            f"Canvas maps to {self.world_size_m:.1f} m × {self.world_size_m:.1f} m. "
            f"Walls are {self.wall_height_m:.1f} m tall. Grid resolution: {self.grid_cells} × {self.grid_cells}."
        )

    def run(self) -> None:
        self.root.mainloop()

    def pixel_to_cell(self, px: float, py: float) -> Tuple[int, int]:
        c = max(0, min(self.grid_cells - 1, int(px / self.cell_px)))
        r = max(0, min(self.grid_cells - 1, int(py / self.cell_px)))
        return r, c

    def cell_to_world_center(self, r: int, c: int) -> Tuple[float, float]:
        # Canvas origin is top-left. Gazebo uses x/y with origin at world center.
        x = (c + 0.5) * self.wall_thickness_m - self.world_size_m / 2.0
        y = self.world_size_m / 2.0 - (r + 0.5) * self.wall_thickness_m
        return x, y

    def paint_cells(self, event: tk.Event, value: int, radius_cells: int) -> None:
        r0, c0 = self.pixel_to_cell(event.x, event.y)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= radius_cells * radius_cells:
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < self.grid_cells and 0 <= c < self.grid_cells:
                        self.grid[r][c] = value
        self.redraw_region(r0, c0, radius_cells + 1)

    def draw_event(self, event: tk.Event) -> None:
        self.paint_cells(event, 1, self.brush_cells)

    def erase_event(self, event: tk.Event) -> None:
        self.paint_cells(event, 0, self.erase_cells)

    def redraw_region(self, r0: int, c0: int, radius: int) -> None:
        # Simpler and still fast enough for this grid size.
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                if self.grid[r][c]:
                    x0 = c * self.cell_px
                    y0 = r * self.cell_px
                    x1 = x0 + self.cell_px
                    y1 = y0 + self.cell_px
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="black", outline="black")

        # Draw a small suggested start area at the center.
        center_px = self.canvas_px / 2
        rad_px = self.robot_clear_radius_m / self.world_size_m * self.canvas_px
        self.canvas.create_oval(
            center_px - rad_px,
            center_px - rad_px,
            center_px + rad_px,
            center_px + rad_px,
            outline="green",
            width=2,
        )
        self.canvas.create_text(center_px, center_px, text="start", fill="green")

    def clear(self) -> None:
        self.grid = [[0 for _ in range(self.grid_cells)] for _ in range(self.grid_cells)]
        self.redraw()

    def save_png_dialog(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save drawing PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if path:
            self.save_png(path)
            messagebox.showinfo("Saved", f"Saved drawing to:\n{path}")

    def save_png(self, path: str) -> None:
        img = Image.new("L", (self.grid_cells, self.grid_cells), 255)
        pixels = img.load()
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                pixels[c, r] = 0 if self.grid[r][c] else 255
        img = img.resize((self.canvas_px, self.canvas_px), Image.Resampling.NEAREST)
        img.save(path)

    def load_png_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Load drawing PNG",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")],
        )
        if path:
            self.load_png(path)
            self.redraw()

    def load_png(self, path: str) -> None:
        img = Image.open(path).convert("L").resize((self.grid_cells, self.grid_cells), Image.Resampling.NEAREST)
        pixels = img.load()
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                # Anything dark becomes wall.
                self.grid[r][c] = 1 if pixels[c, r] < 128 else 0

    def export_sdf_dialog(self) -> None:
        world_size = simpledialog.askfloat(
            "World size",
            "Square world size in meters:",
            initialvalue=self.world_size_m,
            minvalue=1.0,
        )
        if world_size is None:
            return
        self.world_size_m = float(world_size)
        self.wall_thickness_m = self.world_size_m / self.grid_cells

        path = filedialog.asksaveasfilename(
            title="Export Gazebo SDF world",
            defaultextension=".sdf",
            filetypes=[("SDF world", "*.sdf"), ("World file", "*.world"), ("All files", "*.*")],
            initialfile="drawn_tb4_world.sdf",
        )
        if path:
            walls = self.grid_to_wall_segments()
            write_sdf_world(
                path=path,
                walls=walls,
                world_size_m=self.world_size_m,
                wall_height_m=self.wall_height_m,
            )
            messagebox.showinfo(
                "Exported",
                f"Exported {len(walls)} wall segments to:\n{path}\n\n"
                "Use the absolute path when launching TurtleBot4 Gazebo.",
            )

    def grid_to_wall_segments(self) -> List[Wall]:
        """
        Converts filled grid cells into longer horizontal wall boxes.
        This keeps the SDF much smaller than creating one box per pixel.
        """
        walls: List[Wall] = []
        thickness = self.wall_thickness_m

        for r in range(self.grid_cells):
            c = 0
            while c < self.grid_cells:
                if self.grid[r][c] == 0:
                    c += 1
                    continue
                start_c = c
                while c < self.grid_cells and self.grid[r][c] == 1:
                    c += 1
                end_c = c - 1

                x0, y = self.cell_to_world_center(r, start_c)
                x1, _ = self.cell_to_world_center(r, end_c)
                length = (end_c - start_c + 1) * thickness
                x = (x0 + x1) / 2.0
                walls.append(
                    Wall(
                        x=x,
                        y=y,
                        length=length,
                        thickness=thickness,
                        height=self.wall_height_m,
                        yaw=0.0,
                    )
                )
        return walls


def sdf_box_link(name: str, pose: Tuple[float, float, float, float, float, float], size: Tuple[float, float, float], color: str) -> str:
    x, y, z, roll, pitch, yaw = pose
    sx, sy, sz = size
    return f"""
    <model name=\"{name}\">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} {roll:.4f} {pitch:.4f} {yaw:.4f}</pose>
      <link name=\"link\">
        <collision name=\"collision\">
          <geometry>
            <box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>
          </geometry>
        </collision>
        <visual name=\"visual\">
          <geometry>
            <box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>
          </geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def write_sdf_world(path: str, walls: Iterable[Wall], world_size_m: float, wall_height_m: float) -> None:
    wall_models = []
    for i, wall in enumerate(walls):
        wall_models.append(
            sdf_box_link(
                name=f"drawn_wall_{i:04d}",
                pose=(wall.x, wall.y, wall.height / 2.0, 0.0, 0.0, wall.yaw),
                size=(wall.length, wall.thickness, wall.height),
                color="0.1 0.1 0.1 1",
            )
        )

    # Thin boundary lip around the square, useful so the TurtleBot does not fall into nothingness.
    floor_thickness = 0.02
    floor = sdf_box_link(
        name="flat_floor",
        pose=(0.0, 0.0, -floor_thickness / 2.0, 0.0, 0.0, 0.0),
        size=(world_size_m + 2.0, world_size_m + 2.0, floor_thickness),
        color="0.8 0.8 0.8 1",
    )

    sdf = f"""<?xml version=\"1.0\"?>
<sdf version=\"1.9\">
  <world name=\"drawn_tb4_world\">
    <physics name=\"1ms\" type=\"ignored\">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename=\"gz-sim-physics-system\" name=\"gz::sim::systems::Physics\"/>
    <plugin filename=\"gz-sim-user-commands-system\" name=\"gz::sim::systems::UserCommands\"/>
    <plugin filename=\"gz-sim-scene-broadcaster-system\" name=\"gz::sim::systems::SceneBroadcaster\"/>
    <plugin filename=\"gz-sim-sensors-system\" name=\"gz::sim::systems::Sensors\">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <light type=\"directional\" name=\"sun\">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

{floor}
{''.join(wall_models)}

  </world>
</sdf>
"""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(sdf)


def main() -> None:
    app = DrawToGazeboWorld()
    app.run()


if __name__ == "__main__":
    main()

'''python3 -m pip install pillow
python3 draw_to_tb4_gazebo_world.py

ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=/absolute/path/to/drawn_tb4_world.sdf '''