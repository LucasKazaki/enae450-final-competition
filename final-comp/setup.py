from setuptools import find_packages, setup

package_name = 'final-comp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kazaki',
    maintainer_email='kazaki@umd.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'move_robot = final_comp.move_robot:main',
            'view_map = final_comp.view_map:main',
            'slam_map_viewer = final_comp.slam_map_viewer:main',
            'maze_solver = final_comp.maze_solver:main',
        ],
    },
)
