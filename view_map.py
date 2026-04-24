import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from PIL import Image
import os

def load_npz_map(path):
    data = np.load(path)
    grid = data['map']
    resolution = data['resolution']
    origin = data['origin']
    return grid, resolution, origin

def load_ros_map(yaml_path):
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    image_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
    img = Image.open(image_path).convert('L')
    grid = np.array(img)

    # Convert to occupancy-style (0 free, 100 occupied, -1 unknown)
    grid = 100 - (grid / 255.0 * 100)
    
    resolution = meta['resolution']
    origin = meta['origin']
    return grid, resolution, origin

def plot_map(grid, resolution, origin):
    plt.figure(figsize=(8, 8))

    # Convert pixel indices to world coordinates
    height, width = grid.shape
    x = np.linspace(origin[0], origin[0] + width * resolution, width)
    y = np.linspace(origin[1], origin[1] + height * resolution, height)

    plt.imshow(
        grid,
        cmap='gray',
        origin='lower',
        extent=[x[0], x[-1], y[0], y[-1]]
    )

    plt.colorbar(label="Occupancy")
    plt.title("2D Map Visualization")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(False)

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', required=True, help='Path to map file (.npz or .yaml)')
    args = parser.parse_args()

    if args.map.endswith('.npz'):
        grid, resolution, origin = load_npz_map(args.map)
    elif args.map.endswith('.yaml'):
        grid, resolution, origin = load_ros_map(args.map)
    else:
        raise ValueError("Unsupported file format. Use .npz or .yaml")

    plot_map(grid, resolution, origin)

if __name__ == '__main__':
    main()

#python3 view_map.py --map premap_maze_map.npz