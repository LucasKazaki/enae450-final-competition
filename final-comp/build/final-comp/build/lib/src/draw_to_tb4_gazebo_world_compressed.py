#!/usr/bin/env python3
"""
draw_to_tb4_gazebo_world_compressed.py

Draw a square maze/shape and export it as a TurtleBot4/Gazebo Sim SDF world.
This version is optimized for Gazebo stability:
  - Greedy rectangle compression instead of one wall per drawn run/cell
  - One static maze_walls model containing many wall links
  - Simple box geometry only
  - Plain ambient/diffuse colors only, no Gazebo/* material scripts
  - Automatically clears the robot/dock spawn zone around the origin

Controls:
  Left mouse drag   draw walls
  Right mouse drag  erase walls
  C                 clear
  S                 save PNG
  L                 load PNG
  E                 export SDF

Recommended TurtleBot4 launch:
  mkdir -p ~/tb4_worlds
  cp drawn_tb4_world.sdf ~/tb4_worlds/
  cd ~/tb4_worlds
  ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=drawn_tb4_world model:=lite visualize_rays:=false

If using standard model:
  ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=drawn_tb4_world model:=standard visualize_rays:=false
"""

from __future__ import annotations

import os
import re
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, simpledialog
from typing import List, Tuple

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Install Pillow first:\n  python3 -m pip install pillow") from exc


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    rows: int
    cols: int


class DrawToGazeboWorld:
    def __init__(self) -> None:
        self.canvas_px = 720
        self.grid_cells = 120
        self.world_size_m = 14.0
        self.wall_height_m = 1.0
        self.brush_cells = 2
        self.erase_cells = 4
        self.spawn_clear_radius_m = 0.85
        self.dock_clear_rect_m = (-1.2, -0.8, 1.2, 0.8)  # xmin, ymin, xmax, ymax around origin

        self.cell_px = self.canvas_px / self.grid_cells
        self.grid: List[List[int]] = [[0 for _ in range(self.grid_cells)] for _ in range(self.grid_cells)]

        self.root = tk.Tk()
        self.root.title("Draw compressed TurtleBot4 Gazebo world")

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_px,
            height=self.canvas_px,
            bg="white",
            highlightthickness=1,
            highlightbackground="black",
        )
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        tk.Button(self.root, text="Export SDF", command=self.export_sdf_dialog).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        tk.Button(self.root, text="Save PNG", command=self.save_png_dialog).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        tk.Button(self.root, text="Load PNG", command=self.load_png_dialog).grid(row=1, column=2, sticky="ew", padx=4, pady=4)
        tk.Button(self.root, text="Compress Preview", command=self.show_compression).grid(row=1, column=3, sticky="ew", padx=4, pady=4)
        tk.Button(self.root, text="Clear", command=self.clear).grid(row=1, column=4, sticky="ew", padx=4, pady=4)

        self.status = tk.StringVar(value=self.help_text())
        tk.Label(self.root, textvariable=self.status, justify="left").grid(row=2, column=0, columnspan=5, sticky="w", padx=10)

        self.canvas.bind("<Button-1>", self.draw_event)
        self.canvas.bind("<B1-Motion>", self.draw_event)
        self.canvas.bind("<Button-3>", self.erase_event)
        self.canvas.bind("<B3-Motion>", self.erase_event)
        for key, func in [("c", self.clear), ("s", self.save_png_dialog), ("l", self.load_png_dialog), ("e", self.export_sdf_dialog)]:
            self.root.bind(key, lambda _event, f=func: f())
            self.root.bind(key.upper(), lambda _event, f=func: f())

        self.redraw()

    def help_text(self) -> str:
        return (
            "Left drag: draw | Right drag: erase | C: clear | S: save PNG | L: load PNG | E: export SDF\n"
            f"World: {self.world_size_m:.1f} m × {self.world_size_m:.1f} m | "
            f"Grid: {self.grid_cells} × {self.grid_cells} | Wall height: {self.wall_height_m:.1f} m | "
            "Export auto-clears the robot/dock spawn area."
        )

    def run(self) -> None:
        self.root.mainloop()

    def cell_size_m(self) -> float:
        return self.world_size_m / self.grid_cells

    def pixel_to_cell(self, px: float, py: float) -> Tuple[int, int]:
        c = max(0, min(self.grid_cells - 1, int(px / self.cell_px)))
        r = max(0, min(self.grid_cells - 1, int(py / self.cell_px)))
        return r, c

    def cell_center_world(self, r: int, c: int) -> Tuple[float, float]:
        cell = self.cell_size_m()
        x = (c + 0.5) * cell - self.world_size_m / 2.0
        y = self.world_size_m / 2.0 - (r + 0.5) * cell
        return x, y

    def paint(self, event: tk.Event, value: int, radius_cells: int) -> None:
        r0, c0 = self.pixel_to_cell(event.x, event.y)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= radius_cells * radius_cells:
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < self.grid_cells and 0 <= c < self.grid_cells:
                        self.grid[r][c] = value
        self.redraw()

    def draw_event(self, event: tk.Event) -> None:
        self.paint(event, 1, self.brush_cells)

    def erase_event(self, event: tk.Event) -> None:
        self.paint(event, 0, self.erase_cells)

    def redraw(self) -> None:
        self.canvas.delete("all")
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                if self.grid[r][c]:
                    x0 = c * self.cell_px
                    y0 = r * self.cell_px
                    self.canvas.create_rectangle(x0, y0, x0 + self.cell_px, y0 + self.cell_px, fill="black", outline="black")

        # Draw clear zone indicator.
        center = self.canvas_px / 2
        rad = self.spawn_clear_radius_m / self.world_size_m * self.canvas_px
        self.canvas.create_oval(center - rad, center - rad, center + rad, center + rad, outline="green", width=2)
        self.canvas.create_text(center, center, text="spawn clear", fill="green")

    def clear(self) -> None:
        self.grid = [[0 for _ in range(self.grid_cells)] for _ in range(self.grid_cells)]
        self.redraw()

    def save_png_dialog(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.save_png(path)
            messagebox.showinfo("Saved", path)

    def save_png(self, path: str) -> None:
        img = Image.new("L", (self.grid_cells, self.grid_cells), 255)
        pix = img.load()
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                pix[c, r] = 0 if self.grid[r][c] else 255
        img.resize((self.canvas_px, self.canvas_px), Image.Resampling.NEAREST).save(path)

    def load_png_dialog(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.load_png(path)
            self.redraw()

    def load_png(self, path: str) -> None:
        img = Image.open(path).convert("L").resize((self.grid_cells, self.grid_cells), Image.Resampling.NEAREST)
        pix = img.load()
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                self.grid[r][c] = 1 if pix[c, r] < 128 else 0

    def export_grid(self) -> List[List[int]]:
        # Copy grid and clear a safe region around the robot/dock spawn.
        g = [row[:] for row in self.grid]
        xmin, ymin, xmax, ymax = self.dock_clear_rect_m
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                x, y = self.cell_center_world(r, c)
                if x * x + y * y <= self.spawn_clear_radius_m * self.spawn_clear_radius_m:
                    g[r][c] = 0
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    g[r][c] = 0
        return g

    def show_compression(self) -> None:
        rects = greedy_rectangles(self.export_grid())
        filled = sum(sum(row) for row in self.export_grid())
        messagebox.showinfo("Compression preview", f"Filled cells: {filled}\nExported wall rectangles: {len(rects)}")

    def export_sdf_dialog(self) -> None:
        world_size = simpledialog.askfloat("World size", "Square world size in meters:", initialvalue=self.world_size_m, minvalue=2.0)
        if world_size is None:
            return
        self.world_size_m = float(world_size)

        wall_height = simpledialog.askfloat("Wall height", "Wall height in meters:", initialvalue=self.wall_height_m, minvalue=0.1)
        if wall_height is None:
            return
        self.wall_height_m = float(wall_height)

        path = filedialog.asksaveasfilename(
            title="Export SDF",
            defaultextension=".sdf",
            filetypes=[("SDF world", "*.sdf")],
            initialfile="drawn_tb4_world.sdf",
        )
        if not path:
            return

        rects = greedy_rectangles(self.export_grid())
        write_world(path, rects, self.grid_cells, self.world_size_m, self.wall_height_m)
        messagebox.showinfo(
            "Exported",
            f"Exported {len(rects)} compressed wall rectangles to:\n{path}\n\n"
            "Recommended launch:\n"
            "ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=drawn_tb4_world model:=lite visualize_rays:=false",
        )


def greedy_rectangles(grid: List[List[int]]) -> List[Rect]:
    """Greedily cover filled cells with non-overlapping rectangles."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    used = [[False for _ in range(cols)] for _ in range(rows)]
    rects: List[Rect] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 or used[r][c]:
                continue

            # Max width on this row.
            width = 0
            while c + width < cols and grid[r][c + width] and not used[r][c + width]:
                width += 1

            # Extend downward while full width remains filled/unused.
            height = 1
            while r + height < rows:
                ok = True
                for cc in range(c, c + width):
                    if grid[r + height][cc] == 0 or used[r + height][cc]:
                        ok = False
                        break
                if not ok:
                    break
                height += 1

            for rr in range(r, r + height):
                for cc in range(c, c + width):
                    used[rr][cc] = True
            rects.append(Rect(r, c, height, width))

    return rects


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def rect_to_link(rect: Rect, i: int, grid_cells: int, world_size_m: float, wall_height_m: float) -> str:
    cell = world_size_m / grid_cells
    sx = rect.cols * cell
    sy = rect.rows * cell
    sz = wall_height_m

    # Center of rectangle in world coordinates.
    cx_cell = rect.c0 + rect.cols / 2.0
    cy_cell = rect.r0 + rect.rows / 2.0
    x = cx_cell * cell - world_size_m / 2.0
    y = world_size_m / 2.0 - cy_cell * cell
    z = wall_height_m / 2.0

    return f"""
      <link name="wall_{i:04d}">
        <pose>{x:.5f} {y:.5f} {z:.5f} 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>{sx:.5f} {sy:.5f} {sz:.5f}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{sx:.5f} {sy:.5f} {sz:.5f}</size>
            </box>
          </geometry>
          <material>
            <ambient>0.12 0.12 0.12 1</ambient>
            <diffuse>0.12 0.12 0.12 1</diffuse>
            <specular>0 0 0 1</specular>
          </material>
        </visual>
      </link>"""


def write_world(path: str, rects: List[Rect], grid_cells: int, world_size_m: float, wall_height_m: float) -> None:
    links = "".join(rect_to_link(rect, i, grid_cells, world_size_m, wall_height_m) for i, rect in enumerate(rects))
    ground_size = max(world_size_m + 2.0, 16.0)

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="drawn_tb4_world">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.70 0.72 0.75 1</background>
      <shadows>false</shadows>
    </scene>

    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.5 0.2 -1</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{ground_size:.3f} {ground_size:.3f}</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>{ground_size:.3f} {ground_size:.3f}</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.78 0.78 0.78 1</ambient>
            <diffuse>0.78 0.78 0.78 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="maze_walls">
      <static>true</static>
{links}
    </model>
  </world>
</sdf>
"""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(sdf)


def main() -> None:
    DrawToGazeboWorld().run()


if __name__ == "__main__":
    main()
