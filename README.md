# enae450-final-competition
enae450 final competition using ros2 and turtlebot4 to solve a maze

SLAM_TOOLBOX tutorial: https://roboticsbackend.com/ros2-nav2-generate-a-map-with-slam_toolbox/

Build the package
From the package folder:
  cd ~/final-comp
  colcon build --symlink-install
Source the package
  source install/setup.bash

Check that the package built correctly
  ros2 pkg list | grep final_comp

If nothing appears, rebuild and source again:

  cd ~/final-comp
  colcon build --symlink-install
  source install/setup.bash

Run the Python nodes
Use these commands after sourcing:

  ros2 run final_comp maze_solver --ros-args -r __ns:=/tbX
  ros2 run final_comp move_robot --ros-args -r __ns:=/tbX
  ros2 run final_comp slam_map_viewer --ros-args -r __ns:=/tbX
  ros2 run final_comp view_map --ros-args -r __ns:=/tbX

Common commands

Rebuild after editing files:

  cd ~/final-comp
  colcon build --symlink-install
  source install/setup.bash

Run a file directly for debugging:

  python3 src/maze_solver.py

Check executable names:

  ros2 pkg executables final_comp
