from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'final-comp'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lucas Tao',
    maintainer_email='LucasKazaki@users.noreply.github.com',
    description='ROS 2 tools for TurtleBot 4 maze navigation and run analysis',
    license='Proprietary',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'move_robot = src.move_robot:main',
            'view_map = src.view_map:main',
            'slam_map_viewer = src.slam_map_viewer:main',
            'maze_solver = src.maze_solver:main',
            'right_hand_solver = src.right_hand_solver:main',
            'slam_launch = src.slam_launch:generate_launch_description',
            'slam_auto_viewer = src.slam_auto_viewer:main',
            'bag_cmd_vel_player = src.bag_cmd_vel_player:main',
            'teleop = src.teleop:main',
            "metric_c_bag_recorder = src.metric_c_bag_recorder:main",
            "metric_c_plotter = src.metric_c_plotter:main",
            "crop_bag = src.crop_bag:main",
            "bag_crop_scale_plotter = src.bag_crop_scale_plotter:main",
        ],
    },
)
